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
from pathlib import Path

from .console import info
from .errors import DownloadError, GatedRepoError, ToolNotFoundError
from .paths import drive_letter, quant_glob, split_repo_id
from .proc import output_of, run


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


# -- downloads ----------------------------------------------------------------------------
def download_raw(cfg, repo_id: str) -> Path:
    """``hf download <repo>`` into the HF cache on HF_HOME. Idempotent (hf skips present)."""
    hf = find_hf_cli()
    env = download_env(cfg)
    proc = run(
        [hf, "download", repo_id],
        env=env,
        label=f"hf download {repo_id}   (raw safetensors -> HF cache on {drive_letter(cfg.hf_home)})",
    )
    if proc is not None and proc.returncode != 0:
        _raise_download_error(repo_id, output_of(proc))
    return cache_dir(cfg, repo_id)


def download_gguf(cfg, gguf_repo: str, quant: str, target_dir: Path) -> Path:
    """``hf download <repo> --include "*<quant>*" --local-dir <target>`` onto the gguf disk."""
    hf = find_hf_cli()
    env = download_env(cfg)
    proc = run(
        [hf, "download", gguf_repo, "--include", quant_glob(quant), "--local-dir", target_dir],
        env=env,
        label=(
            f"hf download {gguf_repo} --include {quant_glob(quant)} "
            f"--local-dir {target_dir}   (-> {drive_letter(target_dir)})"
        ),
    )
    if proc is not None and proc.returncode != 0:
        _raise_download_error(gguf_repo, output_of(proc))
    return target_dir


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
