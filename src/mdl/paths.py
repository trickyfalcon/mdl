"""Pure path/repo helpers.

Everything here is side-effect free and unit tested. Two Windows specifics drive the
design:

* Configured paths can contain ``%VAR%`` (e.g. ``%USERPROFILE%``) *and* a leading ``~``.
  :func:`expand_path` expands both, in that order.
* The GGUF master layout LM Studio requires is ``<gguf_dir>\\<publisher>\\<model>`` and that
  same directory is what we hand to ``hf download --local-dir``. :func:`lmstudio_target_dir`
  builds it from a repo id.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

__all__ = [
    "expand_path",
    "split_repo_id",
    "lmstudio_target_dir",
    "human_size",
    "path_size",
    "drive_letter",
    "same_path",
    "quant_glob",
    "detect_quant",
]


def expand_path(value: str | os.PathLike[str]) -> Path:
    """Expand ``%VAR%``/``$VAR`` and a leading ``~`` and return a :class:`Path`.

    Order matters: vars are expanded first so a value like ``%USERPROFILE%\\.lmstudio``
    resolves, then ``~`` is expanded for paths written in the unix style. The result is
    *not* resolved against the filesystem (no ``.resolve()``) so it works for paths that
    don't exist yet and for offline/unmounted drives.
    """
    raw = os.fspath(value)
    expanded = os.path.expandvars(raw)
    expanded = os.path.expanduser(expanded)
    return Path(expanded)


def split_repo_id(repo_id: str) -> tuple[str, str]:
    """Split ``"owner/name"`` into ``(owner, name)``.

    A Hugging Face repo id is always ``owner/name`` (the name never contains ``/``). A bare
    string with no slash returns ``("", value)`` so callers can decide on a fallback
    publisher rather than crashing.
    """
    cleaned = repo_id.strip().strip("/")
    if "/" in cleaned:
        owner, name = cleaned.split("/", 1)
        return owner, name
    return "", cleaned


def lmstudio_target_dir(gguf_dir: str | os.PathLike[str], repo_id: str) -> Path:
    """Directory that holds a repo's GGUF files: ``<gguf_dir>\\<publisher>\\<model>``.

    This is both the LM Studio-required layout and the ``--local-dir`` for downloads. If the
    repo id has no publisher we fall back to ``_local`` so files are never written loose at
    the gguf root (LM Studio only lists ``publisher\\model`` entries).
    """
    owner, name = split_repo_id(repo_id)
    owner = owner or "_local"
    return Path(gguf_dir) / owner / name


def human_size(num_bytes: int) -> str:
    """Format a byte count as a short human string (e.g. ``18.4 GB``)."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def path_size(path: str | os.PathLike[str]) -> int:
    """Total bytes of a file, or the recursive size of a directory. Missing -> 0.

    Symlinks are skipped so we never double-count the HF cache's internal links.
    """
    p = Path(path)
    try:
        if not p.exists():
            return 0
        if p.is_file():
            return p.stat().st_size
    except OSError:
        return 0
    total = 0
    for child in p.rglob("*"):
        try:
            if child.is_file() and not child.is_symlink():
                total += child.stat().st_size
        except OSError:
            continue
    return total


def drive_letter(path: str | os.PathLike[str]) -> str:
    """Return the uppercase drive (``"D:"``) of a path, or the UNC root, or ``""``."""
    return Path(path).drive.upper()


def same_path(a: str | os.PathLike[str], b: str | os.PathLike[str]) -> bool:
    """Windows-aware path equality: case-insensitive, separator/normalization tolerant.

    Compares the normalized forms without touching the filesystem, so it works for paths
    that don't exist yet (e.g. comparing config values during ``doctor``).
    """
    na = os.path.normcase(os.path.normpath(str(a)))
    nb = os.path.normcase(os.path.normpath(str(b)))
    return na == nb


def quant_glob(quant: str) -> str:
    """Glob/include pattern that selects a single quant's files (``*Q4_K_M*``)."""
    return f"*{quant}*"


_QUANT_RE = re.compile(
    r"(IQ\d+[A-Z_]*[0-9A-Z]|Q\d+(?:_[0-9A-Z]+)*|BF16|F16|F32)",
    re.IGNORECASE,
)


def detect_quant(filename: str) -> str | None:
    """Best-effort quant label from a GGUF filename (``...Q4_K_M.gguf`` -> ``Q4_K_M``)."""
    match = _QUANT_RE.search(Path(filename).name)
    return match.group(1).upper() if match else None
