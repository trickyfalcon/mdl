"""``mdl add`` -- download a model once, place each chosen format, wire up the runtimes.

Order of operations:
  1. raw safetensors -> HF cache on H: (``--raw``)
  2. GGUF -> ``<gguf_dir>\\<publisher>\\<model>`` on D: (``--gguf``): a prebuilt ``*-GGUF`` repo
     if one is given/found, else a local conversion (``--convert``)
  3. register runtimes (``--register``): lmstudio = verify + advise, ollama = Modelfile import

Every step is idempotent (downloads skip present files; ollama imports skip existing models)
and honours ``--dry-run``.
"""

from __future__ import annotations

from pathlib import Path

from . import convert as convert_mod
from . import hub
from . import volume
from .console import console, info, is_dry, step, success, warn
from .errors import MdlError
from .library import Library
from .locks import repo_lock
from .paths import drive_letter, free_space, human_size, lmstudio_target_dir, split_repo_id
from .registry import lmstudio, ollama

_VALID_RUNTIMES = {"ollama", "lmstudio"}


def _report_and_preflight(status: "hub.DownloadStatus", dest, label: str) -> None:
    """Tell the user what's already on disk (so a resume doesn't look like a fresh start)
    and warn if the destination drive is too full for what's left to download."""
    if status.present_bytes:
        msg = f"{label}: resuming -- {human_size(status.present_bytes)} already on disk"
        if status.incomplete:
            msg += f", {status.incomplete} partial file(s)"
        if status.remaining_bytes is not None:
            msg += f", ~{human_size(status.remaining_bytes)} to go"
        info(f"  {msg}")
    remaining = status.remaining_bytes
    free = free_space(dest)
    if remaining is not None and free is not None and free < remaining:
        warn(
            f"low disk space on {drive_letter(dest)}: {human_size(free)} free but "
            f"~{human_size(remaining)} still to download -- it may fail partway."
        )


def parse_register(value: str) -> set[str]:
    out: set[str] = set()
    for token in (value or "").split(","):
        token = token.strip().lower()
        if not token or token == "none":
            continue
        if token in _VALID_RUNTIMES:
            out.add(token)
        else:
            warn(f"ignoring unknown runtime '{token}' (valid: {', '.join(sorted(_VALID_RUNTIMES))})")
    return out


def pick_primary_gguf(target_dir: Path | None, quant: str) -> Path | None:
    """Choose the single GGUF to hand to Ollama (prefer a non-split file for the quant)."""
    if not target_dir or not target_dir.exists():
        return None
    try:
        files = sorted(target_dir.rglob("*.gguf"))
    except OSError:
        return None
    if not files:
        return None
    q = quant.lower()
    pool = [f for f in files if q in f.name.lower()] or files
    singles = [f for f in pool if "-of-" not in f.name.lower()]
    if singles:
        return min(singles, key=lambda p: len(p.name))
    firsts = [f for f in pool if "00001-of-" in f.name.lower()]
    return firsts[0] if firsts else pool[0]


def add_model(
    cfg,
    library: Library,
    raw_repo: str,
    *,
    gguf_repo: str | None = None,
    quant: str | None = None,
    raw: bool = True,
    gguf: bool = True,
    convert: bool = False,
    register: str = "ollama,lmstudio",
    remote: str | None = None,
    force: bool = False,
    retries: int = 0,
) -> None:
    if "/" not in raw_repo:
        raise MdlError(
            f"'{raw_repo}' is not a valid Hugging Face repo id (expected owner/name).",
            hint="e.g. Qwen/Qwen3-32B",
        )
    # Serialize concurrent adds of the *same* repo (a no-op under --dry-run, which writes nothing).
    with repo_lock(raw_repo, enabled=not is_dry()) as acquired:
        if not acquired:
            raise MdlError(
                f"another `mdl add` for '{raw_repo}' is already running.",
                hint="Wait for it to finish (its download resumes), or add a different model.",
            )
        _add_model_locked(
            cfg, library, raw_repo,
            gguf_repo=gguf_repo, quant=quant, raw=raw, gguf=gguf,
            convert=convert, register=register, remote=remote, force=force, retries=retries,
        )


def _add_model_locked(
    cfg,
    library: Library,
    raw_repo: str,
    *,
    gguf_repo: str | None = None,
    quant: str | None = None,
    raw: bool = True,
    gguf: bool = True,
    convert: bool = False,
    register: str = "ollama,lmstudio",
    remote: str | None = None,
    force: bool = False,
    retries: int = 0,
) -> None:
    quant = quant or cfg.default_quant
    runtimes = parse_register(register)
    _owner, name = split_repo_id(raw_repo)

    console.rule(f"[bold]add[/] {raw_repo}  [dim](quant {quant})[/]")

    registered_ollama: list[str] = []
    resolved_gguf_repo: str | None = None
    target_dir: Path | None = None
    raw_done = False

    # --- 1. raw safetensors --------------------------------------------------------------
    if raw:
        status = hub.raw_status(cfg, raw_repo)
        if status.state == "complete" and not force:
            success(
                f"raw already downloaded ({human_size(status.present_bytes)}, "
                f"{status.present_files} files) -- skipping. [dim]--force to re-verify[/]"
            )
            raw_done = True
        else:
            step(f"raw safetensors -> HF cache ({cfg.hf_home})")
            if not is_dry():
                for w in volume.ensure_ready(cfg.hf_home, "raw HF cache"):
                    warn(w)
            _report_and_preflight(status, cfg.hf_home, "raw")
            hub.download_raw(cfg, raw_repo, retries=retries)
            raw_done = True

    # --- 2. GGUF -------------------------------------------------------------------------
    if gguf:
        if not gguf_repo and not convert:
            step("looking for a prebuilt *-GGUF repo on the Hub ...")
            gguf_repo = hub.find_gguf_repo(raw_repo, quant)
            if gguf_repo:
                info(f"  found [bold]{gguf_repo}[/]")
            else:
                warn("no prebuilt GGUF repo found.")

        if gguf_repo:
            resolved_gguf_repo = gguf_repo
            target_dir = lmstudio_target_dir(cfg.gguf_dir, gguf_repo)
            gstatus = hub.gguf_status(cfg, gguf_repo, quant, target_dir)
            if gstatus.state == "complete" and not force:
                success(
                    f"GGUF '{quant}' already present ({human_size(gstatus.present_bytes)}) "
                    f"-- skipping. [dim]--force to re-verify[/]"
                )
            else:
                step(f"GGUF '{quant}' from {gguf_repo} -> {target_dir}")
                if not is_dry():
                    for w in volume.ensure_ready(cfg.gguf_dir, "GGUF dir"):
                        warn(w)
                _report_and_preflight(gstatus, cfg.gguf_dir, "gguf")
                hub.download_gguf(cfg, gguf_repo, quant, target_dir, retries=retries)
        elif convert:
            target_dir = lmstudio_target_dir(cfg.gguf_dir, raw_repo)
            if remote:
                step(f"converting from the Hub (--remote {remote}) -> {target_dir}")
                source: str | Path = remote
                use_remote = True
            else:
                snap = hub.snapshot_path(cfg, raw_repo)
                if snap is None:
                    step("no local snapshot yet; downloading raw weights to convert ...")
                    hub.download_raw(cfg, raw_repo, retries=retries)
                    raw_done = True  # download_raw ran unconditionally; weights are now cached
                    snap = hub.snapshot_path(cfg, raw_repo)
                if snap is None and not is_dry():
                    raise MdlError(
                        f"could not locate a downloaded snapshot for '{raw_repo}' to convert.",
                        hint="The raw download may have failed, or the HF cache layout is unexpected.",
                    )
                source = snap if snap is not None else hub.cache_dir(cfg, raw_repo) / "snapshots"
                use_remote = False
                step(f"converting local snapshot -> {target_dir}")
            convert_mod.convert_model(
                cfg, source=source, quant=quant, target_dir=target_dir, model_name=name, remote=use_remote
            )
        else:
            warn("skipping GGUF: pass --gguf-repo <repo> or --convert to build one.")

    # --- 3. register runtimes ------------------------------------------------------------
    # have_gguf requires a GGUF to have actually landed (under dry-run nothing is downloaded,
    # so trust the plan instead of scanning an empty dir).
    have_gguf = target_dir is not None and (is_dry() or any(target_dir.rglob("*.gguf")))
    if runtimes and not have_gguf:
        warn(f"nothing to register with {', '.join(sorted(runtimes))}: no GGUF was placed.")

    if have_gguf and "lmstudio" in runtimes:
        lmstudio.register(cfg, target_dir)

    if have_gguf and "ollama" in runtimes:
        primary = pick_primary_gguf(target_dir, quant)
        if primary is None and is_dry():
            primary = target_dir / f"{name}-{quant}.gguf"  # representative path for the plan
        if primary is None:
            warn("ollama: no .gguf found to import.")
        else:
            if "-of-" in primary.name.lower():
                warn(f"ollama: importing split GGUF shard {primary.name}; verify it loads.")
            oname = ollama.model_name_for(resolved_gguf_repo or raw_repo, quant)
            ollama.import_gguf(cfg, primary, oname)
            registered_ollama.append(oname)

    # --- 4. persist + summarise ----------------------------------------------------------
    if not is_dry():
        library.upsert(
            raw_repo,
            raw_repo=raw_repo if raw_done else None,
            gguf_repo=resolved_gguf_repo,
            quants=[quant] if have_gguf else None,
            ollama=registered_ollama or None,
            # None preserves a previously-set flag (upsert merges); True only when (re)registered
            lmstudio=(True if (have_gguf and "lmstudio" in runtimes) else None),
        )
        library.save()

    _summary(cfg, raw_repo, raw_done, target_dir, quant, registered_ollama, "lmstudio" in runtimes and have_gguf)


def _summary(cfg, model, raw_done, target_dir, quant, ollama_names, lmstudio_done) -> None:
    console.print("\n[bold]Summary[/]")
    console.print(f"  model     : {model}")
    console.print(f"  raw       : {(drive_letter(cfg.hf_home) or 'local') + ' HF cache' if raw_done else '[dim]skipped[/]'}")
    if target_dir:
        console.print(f"  gguf      : {quant} -> {target_dir}")
    else:
        console.print("  gguf      : [dim]none[/]")
    console.print(f"  ollama    : {', '.join(ollama_names) if ollama_names else '[dim]none[/]'}")
    console.print(f"  lmstudio  : {'verified + advised' if lmstudio_done else '[dim]none[/]'}")
    if is_dry():
        console.print("[yellow]\\[dry-run] nothing was downloaded, created, or recorded.[/]")
