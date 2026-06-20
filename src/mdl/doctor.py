"""``mdl doctor`` -- a Windows-aware status table for the whole setup."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from shutil import which

from rich.table import Table

from .console import console, is_dry
from .hub import find_hf_cli, whoami
from .osenv import IS_WINDOWS, env_set_hint, exe
from .paths import drive_letter, expand_path, same_path
from .registry import lmstudio, ollama
from .volume import kind as volume_kind

OK, WARN, FAIL = "OK", "WARN", "FAIL"
_STYLE = {OK: "green", WARN: "yellow", FAIL: "red"}


@dataclass
class Check:
    name: str
    status: str
    detail: str
    fix: str | None = None


def _writable(path: Path) -> tuple[bool | None, str]:
    """``(True|False|None, detail)`` -- None means the drive isn't mounted."""
    drive = drive_letter(path)
    if drive and not Path(drive + "\\").exists():
        return None, f"{drive} is not mounted"
    if is_dry():
        # plan-only: never create dirs or write a probe under --dry-run
        if path.exists():
            return True, "exists (dry-run: write not probed)"
        return True, "drive mounted; dir created on first write (dry-run: not probed)"
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".mdl-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True, "exists & writable"
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc.strerror or exc}"


def check_drive(cfg, key: str, label: str) -> Check:
    path = cfg.expanded(key)
    drive = drive_letter(path) or "local"
    vkind = volume_kind(path)
    if vkind == "missing":
        return Check(label, FAIL, f"{drive} ({path}) is not mounted", fix=f"Mount {drive} or change `{key}` in config.")
    where = vkind if vkind != "unknown" else "local"
    res, detail = _writable(path)
    if res is None:
        return Check(label, FAIL, detail, fix=f"Mount {drive} or change `{key}` in config.")
    if res:
        return Check(label, OK, f"{drive}  {path}  ({where}, {detail})")
    return Check(
        label, FAIL, f"{path}: {detail}",
        fix=f"Check permissions / Controlled Folder Access for {path}.",
    )


def check_hf() -> Check:
    try:
        import huggingface_hub

        ver = huggingface_hub.__version__
    except Exception as exc:
        return Check("hf library", FAIL, f"huggingface_hub not importable: {exc}", fix="uv sync")
    try:
        cli = find_hf_cli()
        return Check("hf CLI", OK, f"huggingface_hub {ver}; hf at {cli}")
    except Exception as exc:
        return Check("hf CLI", WARN, f"huggingface_hub {ver}, but `hf` not found ({exc})", fix="uv sync")


def check_auth() -> Check:
    name = whoami()
    if name:
        return Check("HF auth", OK, f"logged in as {name}")
    return Check("HF auth", WARN, "not logged in", fix="Run `hf auth login` for gated/private models.")


def check_ollama(cfg) -> Check:
    exe = which(cfg.ollama_bin)
    if not exe:
        return Check("ollama", FAIL, f"'{cfg.ollama_bin}' not found on PATH", fix="Install Ollama or set `ollama_bin`.")
    running, detail = ollama.ollama_running(cfg)
    if running:
        return Check("ollama", OK, f"{exe} ({detail})")
    return Check("ollama service", WARN, f"{exe} found but {detail}", fix="Start Ollama (`ollama serve` or the app).")


def _ollama_has_data(path) -> bool:
    """True if an Ollama models dir actually holds blobs -- i.e. the daemon writes there."""
    try:
        return (Path(path) / "blobs").is_dir()
    except OSError:
        return False


def _ollama_models_check(env: str | None, want: Path, default_loc: Path) -> Check:
    """Pure decision for the OLLAMA_MODELS check (separated so it's unit-testable)."""
    default_home = "%USERPROFILE%\\.ollama" if IS_WINDOWS else "~/.ollama"
    if not env:
        extra = f"; existing blobs are in {default_loc}" if _ollama_has_data(default_loc) else ""
        return Check(
            "OLLAMA_MODELS", WARN, f"not set (Ollama stores blobs under {default_home} by default{extra})",
            fix=f"{env_set_hint('OLLAMA_MODELS', str(want))}   then restart Ollama",
        )
    env_path = expand_path(env)  # the env value may itself contain %VARS%/$VARS
    # Daemon/env mismatch: the env points somewhere with no blobs, but the default location HAS
    # them -> the running daemon was started before the env var and ignores it (needs a restart).
    if _ollama_has_data(default_loc) and not _ollama_has_data(env_path) and not same_path(env_path, default_loc):
        return Check(
            "OLLAMA_MODELS", WARN,
            f"set to {env_path}, but Ollama's blobs are in {default_loc} -- the running daemon predates the env var",
            fix="Fully quit & reopen Ollama so it reads OLLAMA_MODELS, then re-import (`mdl sync`).",
        )
    # On Windows, warn when imports land on the system drive (C:), where space is precious.
    if IS_WINDOWS and drive_letter(env_path) == "C:":
        return Check(
            "OLLAMA_MODELS", WARN, f"{env_path} (on C: -- import copies land on the system drive)",
            fix=f"Point at the fast disk: {env_set_hint('OLLAMA_MODELS', str(want))}   then restart Ollama",
        )
    suffix = " (matches config)" if same_path(env_path, want) else f" (config suggests {want})"
    return Check("OLLAMA_MODELS", OK, str(env_path) + suffix)


def check_ollama_models(cfg) -> Check:
    return _ollama_models_check(
        os.environ.get("OLLAMA_MODELS"), cfg.ollama_models, Path.home() / ".ollama" / "models"
    )


def check_llamacpp(cfg) -> Check:
    conv, quant = cfg.convert_script, cfg.llama_quantize
    detail = (
        f"convert_hf_to_gguf.py: {'found' if conv.exists() else 'MISSING'}; "
        f"{exe('llama-quantize')}: {'found' if quant.exists() else 'MISSING'}"
    )
    if conv.exists() and quant.exists():
        return Check("llama.cpp", OK, detail)
    return Check(
        "llama.cpp", WARN, detail,
        fix="Set `llamacpp_dir`/`llama_quantize`, or build llama.cpp (convert/quantize stay disabled until then).",
    )


def check_lmstudio(cfg) -> Check:
    target = cfg.gguf_dir
    # real value from settings.json, else the configured lmstudio_dir fallback
    detected = lmstudio.detect_models_dir() or cfg.lmstudio_dir
    if same_path(detected, target):
        return Check("LM Studio dir", OK, f"{detected} == gguf_dir")
    return Check(
        "LM Studio dir", WARN, f"{detected} != gguf_dir ({target})",
        fix=f"LM Studio > My Models > set folder to {target} for zero-duplication GGUF sharing.",
    )


def check_hf_home(cfg) -> Check:
    env = os.environ.get("HF_HOME")
    if not env:
        return Check(
            "HF_HOME env", WARN, "not set (transformers/vLLM won't share mdl's cache)",
            fix=f"{env_set_hint('HF_HOME', str(cfg.hf_home))}   then open a new shell",
        )
    env_path = expand_path(env)
    if same_path(env_path, cfg.hf_home):
        return Check("HF_HOME env", OK, f"{env_path} (matches config)")
    return Check(
        "HF_HOME env", WARN, f"{env_path} != config hf_home ({cfg.hf_home})",
        fix=env_set_hint("HF_HOME", str(cfg.hf_home)),
    )


def collect(cfg) -> list[Check]:
    return [
        check_drive(cfg, "hf_home", "Drive: raw / HF cache"),
        check_drive(cfg, "gguf_dir", "Drive: GGUF master"),
        check_hf(),
        check_auth(),
        check_ollama(cfg),
        check_ollama_models(cfg),
        check_llamacpp(cfg),
        check_lmstudio(cfg),
        check_hf_home(cfg),
    ]


def run_doctor(cfg) -> bool:
    """Print the status table + suggested fixes. Returns False if any check FAILs."""
    checks = collect(cfg)
    table = Table(title="mdl doctor", title_style="bold", show_lines=False)
    table.add_column("Check", style="bold", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Details", overflow="fold")
    for c in checks:
        table.add_row(c.name, f"[{_STYLE[c.status]}]{c.status}[/]", c.detail)
    console.print(table)

    fixes = [c for c in checks if c.fix and c.status != OK]
    if fixes:
        console.print("\n[bold]Suggested fixes[/]")
        for c in fixes:
            console.print(f"  [{_STYLE[c.status]}]-[/] [bold]{c.name}[/]: {c.fix}")
    return not any(c.status == FAIL for c in checks)
