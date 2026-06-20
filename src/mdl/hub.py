"""Hugging Face access: downloads via the ``hf`` CLI, metadata via ``huggingface_hub``.

Downloads shell out to ``hf download`` (the current CLI; ``huggingface-cli`` is deprecated)
so the exact command is visible under ``--dry-run`` and streamable under ``--verbose``. We
inject ``HF_HOME`` and ``HF_XET_HIGH_PERFORMANCE=1`` into the child env so the user doesn't
have to set them globally -- and, if a token exists at the default location, bridge it via
``HF_TOKEN`` so gated downloads keep working even when ``HF_HOME`` is relocated to H:.
"""

from __future__ import annotations

import os
import sys
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from .console import info, warn
from .errors import DownloadError, GatedRepoError, MdlError, ToolNotFoundError
from .paths import drive_letter, quant_glob, split_repo_id
from .proc import run


# -- environment --------------------------------------------------------------------------
def _existing_token() -> str | None:
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def download_env(cfg) -> dict:
    """Child-process env with HF_HOME + xet acceleration (and a bridged token)."""
    env = dict(os.environ)
    env["HF_HOME"] = str(cfg.hf_home)
    env["HF_XET_HIGH_PERFORMANCE"] = "1"
    # Abort a stalled *classic* request/metadata fetch instead of blocking forever. xet has
    # its own network stack and ignores these -- the proc.run stall watchdog covers that path.
    env.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(cfg.download_timeout))
    env.setdefault("HF_HUB_ETAG_TIMEOUT", str(cfg.download_timeout))
    if not env.get("HF_TOKEN"):
        token = _existing_token()
        if token:
            env["HF_TOKEN"] = token
    return env


def find_hf_cli() -> Path:
    """Locate ``hf.exe`` -- prefer the one next to the running interpreter (our venv)."""
    sibling = Path(sys.executable).with_name("hf.exe")
    if sibling.exists():
        return sibling
    found = shutil.which("hf")
    if found:
        return Path(found)
    raise ToolNotFoundError(
        "The `hf` CLI was not found.",
        hint="It ships with huggingface_hub. Run `uv sync` in the project, or `uv tool install huggingface_hub`.",
    )


# -- cache layout -------------------------------------------------------------------------
def cache_dir(cfg, repo_id: str) -> Path:
    """Where ``hf download <repo>`` (no --local-dir) stores a repo: the HF hub cache dir."""
    owner, name = split_repo_id(repo_id)
    return cfg.hf_home / "hub" / f"models--{owner}--{name}"


def snapshot_path(cfg, repo_id: str) -> Path | None:
    """Newest revision snapshot dir for a cached repo (has config.json + weights), or None."""
    base = cache_dir(cfg, repo_id) / "snapshots"
    if not base.exists():
        return None
    revs = [d for d in base.iterdir() if d.is_dir()]
    return max(revs, key=lambda d: d.stat().st_mtime) if revs else None


# -- error classification -----------------------------------------------------------------
def _raise_download_error(repo_id: str, out: str) -> None:
    low = out.lower()
    if "401" in out or "403" in out or "gated" in low or "awaiting a review" in low or (
        "access to model" in low and "restricted" in low
    ):
        raise GatedRepoError(
            f"Access to '{repo_id}' was denied (gated or private).",
            hint="Run `hf auth login` and accept the model's license on its Hub page.",
        )
    if "repositorynotfound" in low.replace(" ", "") or "404" in out or "not found" in low:
        raise DownloadError(
            f"Repository '{repo_id}' was not found on the Hub.",
            hint="Check the repo id, or pass --gguf-repo explicitly.",
        )
    tail = "\n".join([ln for ln in out.strip().splitlines() if ln.strip()][-6:])
    raise DownloadError(f"Download of '{repo_id}' failed.\n{tail}".rstrip())


def _permanent_access_error(repo_id: str) -> MdlError | None:
    """If a download failed for a *permanent* reason (gated/private/not-found), return the
    friendly error to raise. Return ``None`` when the repo is reachable -- i.e. the failure
    was transient (dropped connection, full disk, Ctrl-C) and a retry/resume is worthwhile.

    Streamed downloads inherit the terminal so we never captured the child's text; this
    re-derives the cause from a cheap metadata probe instead.
    """
    try:
        _api().repo_info(repo_id, token=_existing_token())
        return None  # reachable -> not a permanent access problem
    except Exception as exc:
        text = f"{type(exc).__name__}: {exc}"
        low = text.lower()
        signatures = ("401", "403", "404", "gated", "restricted", "awaiting a review", "repositorynotfound", "not found")
        if not any(sig in low for sig in signatures):
            return None  # probe itself failed (offline, rate limit) -> treat as transient
        try:
            _raise_download_error(repo_id, text)
        except MdlError as err:
            return err
    return None


def _run_download(
    repo_id: str,
    argv: list,
    env: dict,
    label: str,
    retries: int,
    *,
    watch_dir: Path | None = None,
    stall_timeout: float | None = None,
) -> None:
    """Run an ``hf download`` (streamed), retrying transient failures with backoff.

    ``hf`` resumes from disk, so each retry continues where the last left off rather than
    starting over. Permanent failures (gated/not-found) raise immediately -- retrying them
    is pointless. A watchdog (``watch_dir``/``stall_timeout``) turns a silent hang into a
    retryable non-zero exit.
    """
    attempt = 0
    while True:
        proc = run(argv, env=env, label=label, stream=True, watch_dir=watch_dir, stall_timeout=stall_timeout)
        if proc is None or proc.returncode == 0:  # None == dry-run (nothing ran)
            return
        permanent = _permanent_access_error(repo_id)
        if permanent is not None:
            raise permanent
        if attempt >= retries:
            raise DownloadError(
                f"Download of '{repo_id}' failed (see the output above).",
                hint="Re-run the same command to resume -- completed files are skipped.",
            )
        attempt += 1
        delay = min(60, 5 * (2 ** (attempt - 1)))  # 5, 10, 20, 40, 60, 60 ...
        warn(f"'{repo_id}' download failed; retry {attempt}/{retries} in {delay}s (resuming) ...")
        time.sleep(delay)


# -- downloads ----------------------------------------------------------------------------
def download_raw(cfg, repo_id: str, *, retries: int = 0) -> Path:
    """``hf download <repo>`` into the HF cache on HF_HOME. Idempotent (hf skips present)."""
    hf = find_hf_cli()
    env = download_env(cfg)
    _run_download(
        repo_id,
        [hf, "download", repo_id],
        env,
        f"hf download {repo_id}   (raw safetensors -> HF cache on {drive_letter(cfg.hf_home)})",
        retries,
        watch_dir=cache_dir(cfg, repo_id),
        stall_timeout=cfg.download_stall_timeout,
    )
    return cache_dir(cfg, repo_id)


def download_gguf(cfg, gguf_repo: str, quant: str, target_dir: Path, *, retries: int = 0) -> Path:
    """``hf download <repo> --include "*<quant>*" --local-dir <target>`` onto the gguf disk."""
    hf = find_hf_cli()
    env = download_env(cfg)
    _run_download(
        gguf_repo,
        [hf, "download", gguf_repo, "--include", quant_glob(quant), "--local-dir", target_dir],
        env,
        (
            f"hf download {gguf_repo} --include {quant_glob(quant)} "
            f"--local-dir {target_dir}   (-> {drive_letter(target_dir)})"
        ),
        retries,
        watch_dir=target_dir,
        stall_timeout=cfg.download_stall_timeout,
    )
    return target_dir


# -- local download status ----------------------------------------------------------------
@dataclass
class DownloadStatus:
    """How much of a repo (optionally filtered to a quant) is already on disk.

    ``state`` is the headline:
      * ``complete``  -- every expected file is present (verified against the Hub) -> skip.
      * ``partial``   -- some bytes present (a resume) *or* present-but-unverified (offline).
      * ``missing``   -- nothing on disk yet.

    ``expected_*`` come from a best-effort Hub metadata call; they are ``None`` when the Hub
    is unreachable, in which case ``verified`` is ``False`` and we never claim ``complete``
    (so we always hand off to ``hf``, which reconciles for real).
    """

    state: str
    present_bytes: int
    present_files: int
    expected_bytes: int | None
    expected_files: int | None
    incomplete: int  # count of ``*.incomplete`` blobs still transferring
    verified: bool

    @property
    def remaining_bytes(self) -> int | None:
        if self.expected_bytes is None:
            return None
        return max(0, self.expected_bytes - self.present_bytes)


def classify_download(
    present_bytes: int,
    present_files: int,
    incomplete: int,
    expected_bytes: int | None,
    expected_files: int | None,
) -> DownloadStatus:
    """Pure classifier (no I/O) so the state machine is unit-testable.

    Without Hub metadata we are deliberately conservative: anything on disk is ``partial``
    (never ``complete``), so a re-run still lets ``hf`` verify rather than skipping blindly.
    """
    verified = expected_files is not None
    if not verified:
        state = "partial" if (present_files or incomplete) else "missing"
        return DownloadStatus(state, present_bytes, present_files, None, None, incomplete, False)
    if present_files == 0 and incomplete == 0:
        state = "missing"
    elif incomplete == 0 and present_files >= expected_files and present_bytes >= int((expected_bytes or 0) * 0.999):
        state = "complete"
    else:
        state = "partial"
    return DownloadStatus(state, present_bytes, present_files, expected_bytes, expected_files, incomplete, True)


def _dir_logical_size(root: Path | None, predicate=None) -> tuple[int, int]:
    """``(bytes, files)`` under ``root``, following symlinks so HF-cache snapshots that link
    into ``blobs`` still report their true size. ``predicate`` filters on the posix relpath."""
    if root is None or not root.exists():
        return 0, 0
    total = files = 0
    try:
        walk = root.rglob("*")
    except OSError:
        return 0, 0
    for f in walk:
        try:
            if not f.is_file():
                continue
            if predicate is not None and not predicate(f.relative_to(root).as_posix()):
                continue
            total += f.stat().st_size  # follows symlinks -> logical (real) size
            files += 1
        except OSError:
            continue
    return total, files


def _incomplete_count(blobs_dir: Path) -> int:
    """Number of ``*.incomplete`` blobs (hf/xet's in-flight transfer markers)."""
    if not blobs_dir.exists():
        return 0
    try:
        return sum(1 for f in blobs_dir.iterdir() if f.name.endswith(".incomplete"))
    except OSError:
        return 0


def _repo_sizes(repo_id: str, predicate=None) -> tuple[int | None, int | None]:
    """``(total_bytes, file_count)`` of a repo's files from the Hub. ``(None, None)`` on any
    failure (offline, gated, not found) so callers degrade to a plain ``hf`` hand-off."""
    try:
        meta = _api().model_info(repo_id, files_metadata=True, token=_existing_token())
    except Exception:
        return None, None
    sibs = getattr(meta, "siblings", None) or []
    sel = [s for s in sibs if predicate is None or predicate(getattr(s, "rfilename", ""))]
    total = sum(int(getattr(s, "size", 0) or 0) for s in sel)
    return total, len(sel)


def incomplete_count(cfg, repo_id: str) -> int:
    """How many ``*.incomplete`` blobs the raw cache holds for ``repo_id`` (local, cheap)."""
    return _incomplete_count(cache_dir(cfg, repo_id) / "blobs")


def raw_status(cfg, repo_id: str) -> DownloadStatus:
    """Local status of the raw safetensors download for ``repo_id`` in the HF cache."""
    present_bytes, present_files = _dir_logical_size(snapshot_path(cfg, repo_id))
    incomplete = _incomplete_count(cache_dir(cfg, repo_id) / "blobs")
    exp_bytes, exp_files = _repo_sizes(repo_id)
    return classify_download(present_bytes, present_files, incomplete, exp_bytes, exp_files)


def gguf_status(cfg, gguf_repo: str, quant: str, target_dir: Path) -> DownloadStatus:
    """Local status of the ``quant`` GGUF(s) from ``gguf_repo`` under ``target_dir``."""
    q = quant.lower()
    pred = lambda name: q in name.lower() and name.lower().endswith(".gguf")  # noqa: E731
    present_bytes, present_files = _dir_logical_size(target_dir, pred)
    exp_bytes, exp_files = _repo_sizes(gguf_repo, pred)
    return classify_download(present_bytes, present_files, 0, exp_bytes, exp_files)


# -- metadata / discovery -----------------------------------------------------------------
def _api():
    from huggingface_hub import HfApi

    return HfApi()


def whoami() -> str | None:
    """Return the logged-in HF username, or ``None`` if not authenticated."""
    try:
        from huggingface_hub import whoami as _who

        token = _existing_token()
        data = _who(token=token) if token else _who()
        return data.get("name") if isinstance(data, dict) else None
    except Exception:
        return None


def list_gguf_files(repo_id: str, quant: str | None = None) -> list[str]:
    """GGUF filenames in a repo (optionally filtered to a quant). ``[]`` on any failure."""
    try:
        files = _api().list_repo_files(repo_id)
    except Exception:
        return []
    out = [f for f in files if f.lower().endswith(".gguf")]
    if quant:
        q = quant.lower()
        out = [f for f in out if q in f.lower()]
    return out


def find_gguf_repo(raw_repo: str, quant: str) -> str | None:
    """Best-effort search for a prebuilt ``*-GGUF`` repo that carries ``quant``."""
    owner, name = split_repo_id(raw_repo)
    base = name[:-5] if name.lower().endswith("-gguf") else name
    candidates = [
        f"{owner}/{base}-GGUF",
        f"bartowski/{base}-GGUF",
        f"lmstudio-community/{base}-GGUF",
        f"unsloth/{base}-GGUF",
    ]
    seen: set[str] = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        if list_gguf_files(cand, quant):
            return cand
    # fall back to a Hub search
    try:
        for model in _api().list_models(search=f"{base} GGUF", limit=25):
            mid = getattr(model, "id", "")
            if mid in seen:
                continue
            seen.add(mid)
            if base.lower() in mid.lower() and list_gguf_files(mid, quant):
                return mid
    except Exception:
        pass
    return None


def print_hf_home_hint(cfg) -> None:
    """One-time nudge to make transformers/vLLM share the same cache as mdl."""
    if not os.environ.get("HF_HOME"):
        info(
            f'[dim]hint:[/] run [cyan]setx HF_HOME "{cfg.hf_home}"[/] so transformers/vLLM '
            "use the same cache (new shells only)."
        )
