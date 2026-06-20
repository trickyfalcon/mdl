"""The ``mdl`` command-line interface (Typer + Rich)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from . import __version__
from . import hub
from .add import add_model
from .config import CONFIG_PATH, DEFAULTS, PATH_KEYS, Config
from .console import console, err_console, flags, info, success
from .convert import convert_model
from .doctor import run_doctor
from .errors import MdlError
from .library import Library, inventory
from .ops import apply_gc, apply_removal, build_gc_plan, build_removal_plan, render_gc_plan, render_removal_plan, sync_all
from .paths import drive_letter, expand_path, human_size, lmstudio_target_dir, split_repo_id
from .verify import verify_model

app = typer.Typer(
    name="mdl",
    help="Manage a local LLM library across HF transformers/vLLM, llama.cpp, Ollama, and LM Studio (Windows).",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)
config_app = typer.Typer(help="Show or edit configuration.", invoke_without_command=True)
app.add_typer(config_app, name="config")


# -- global options ------------------------------------------------------------------------
def _version_callback(value: bool) -> None:
    if value:
        console.print(f"mdl {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    dry_run: bool = typer.Option(False, "--dry-run", help="Print planned actions without doing them."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Stream subprocess output."),
    version: bool = typer.Option(
        False, "--version", help="Show version and exit.", is_eager=True, callback=_version_callback
    ),
) -> None:
    flags.dry_run = dry_run
    flags.verbose = verbose


def _load_config() -> Config:
    cfg = Config.load(create=True)
    if cfg.just_created:
        info(f"[green]created default config[/] at {CONFIG_PATH}")
        hub.print_hf_home_hint(cfg)
    return cfg


# -- add -----------------------------------------------------------------------------------
@app.command()
def add(
    repo: str = typer.Argument(..., help="Hugging Face repo id, e.g. Qwen/Qwen3-32B"),
    gguf_repo: Optional[str] = typer.Option(None, "--gguf-repo", help="Repo holding GGUFs (e.g. bartowski/<model>-GGUF)."),
    quant: Optional[str] = typer.Option(None, "--quant", help="Quant to pull/build (default from config)."),
    raw: bool = typer.Option(True, "--raw/--no-raw", help="Also download full safetensors into HF_HOME (H:)."),
    gguf: bool = typer.Option(True, "--gguf/--no-gguf", help="Download/place the GGUF (D:)."),
    convert: bool = typer.Option(False, "--convert", help="Build the GGUF locally if no prebuilt repo exists."),
    remote: Optional[str] = typer.Option(None, "--remote", help="With --convert: stream weights from this Hub repo."),
    register: str = typer.Option("ollama,lmstudio", "--register", help="Runtimes to wire up (csv): ollama,lmstudio."),
    force: bool = typer.Option(False, "--force", help="Re-download/verify even if files are already present."),
    retries: int = typer.Option(0, "--retries", help="Retry a failed download N times (resuming) before giving up."),
) -> None:
    """Download a model once and wire it up to every runtime."""
    cfg = _load_config()
    lib = Library.load()
    add_model(
        cfg, lib, repo,
        gguf_repo=gguf_repo, quant=quant, raw=raw, gguf=gguf,
        convert=convert, register=register, remote=remote, force=force, retries=retries,
    )


# -- list ----------------------------------------------------------------------------------
def _format_cell(fi) -> str:
    """Render a format cell: drive + size, plus a completeness marker.

    Without ``--check`` we only have the local ``*.incomplete`` signal -> ``(partial:N)``.
    With ``--check`` we have the Hub-verified state -> ``OK`` (complete) or ``NN%`` (partial).
    """
    if not fi.present and not fi.incomplete:
        return "[dim]-[/]"
    loc = f"{fi.drive} " if fi.drive else ""  # POSIX has no drive letter
    cell = f"{loc}{human_size(fi.size)}"
    if fi.state == "complete":
        cell += " [green]OK[/]"
    elif fi.state == "partial":
        if fi.expected:
            cell += f" [yellow]{int(100 * fi.size / fi.expected)}%[/]"
        else:
            cell += " [yellow]partial[/]"
    elif fi.incomplete:
        cell += f" [yellow](partial:{fi.incomplete})[/]"
    return cell


@app.command("list")
def list_cmd(
    check: bool = typer.Option(
        False, "--check", help="Reconcile against the Hub to show true completeness (slower, needs network)."
    ),
) -> None:
    """Show the library: formats, quants, runtimes, drives and sizes."""
    cfg = _load_config()
    lib = Library.load()
    rows = inventory(cfg, lib, check=check)
    if not rows:
        info("library is empty. Add one with [cyan]mdl add <repo>[/].")
        return

    table = Table(show_lines=False)
    table.add_column("Model", style="bold", overflow="fold")
    table.add_column("Raw (drive/size)")
    table.add_column("GGUF (drive/size)")
    table.add_column("Quant(s)")
    table.add_column("Runtimes")

    raw_total = gguf_total = 0
    for row in rows:
        raw_total += row.raw.size
        gguf_total += row.gguf.size
        raw_cell = _format_cell(row.raw)
        gguf_cell = _format_cell(row.gguf)
        runtimes = []
        if row.ollama:
            runtimes.append("ollama: " + ", ".join(row.ollama))
        if row.lmstudio:
            runtimes.append("lmstudio")
        table.add_row(
            row.model, raw_cell, gguf_cell,
            ", ".join(row.quants) if row.quants else "[dim]-[/]",
            "; ".join(runtimes) if runtimes else "[dim]none[/]",
        )
    console.print(table)
    console.print(
        f"[bold]totals[/]  raw [{drive_letter(cfg.hf_home) or 'local'}] {human_size(raw_total)}   "
        f"gguf [{drive_letter(cfg.gguf_dir) or 'local'}] {human_size(gguf_total)}   "
        f"grand {human_size(raw_total + gguf_total)}"
    )
    if not check:
        console.print("[dim]tip: `mdl list --check` verifies completeness against the Hub; `mdl verify <model>` repairs gaps.[/]")


# -- rm ------------------------------------------------------------------------------------
@app.command("rm")
def rm_cmd(
    model: str = typer.Argument(..., help="Model id / name as shown by `mdl list`."),
    fmt: Optional[str] = typer.Option(None, "--format", help="raw | gguf | all  (default: all)."),
    from_: Optional[str] = typer.Option(None, "--from", help="Runtimes to deregister, csv: ollama,lmstudio."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Remove a model across stores and/or runtimes."""
    cfg = _load_config()
    lib = Library.load()
    runtimes = None
    if from_ is not None:
        runtimes = {t.strip().lower() for t in from_.split(",") if t.strip()}
    removal = build_removal_plan(cfg, lib, model, fmt=fmt, from_runtimes=runtimes)
    if removal.is_empty():
        info("nothing matches that scope; nothing to remove.")
        return
    render_removal_plan(removal)
    if not yes and not flags.dry_run:
        if not typer.confirm("Proceed?"):
            info("aborted.")
            raise typer.Exit(1)
    apply_removal(cfg, lib, removal)


# -- sync ----------------------------------------------------------------------------------
@app.command()
def sync() -> None:
    """Re-apply all registrations from the current config (after moving the library)."""
    cfg = _load_config()
    lib = Library.load()
    sync_all(cfg, lib)


# -- gc ------------------------------------------------------------------------------------
@app.command()
def gc(
    model: Optional[str] = typer.Argument(None, help="Limit to one model (id/name); default sweeps everything."),
    locks: bool = typer.Option(False, "--locks", help="Also clear stale .lock files (unblocks a wedged cache)."),
    force: bool = typer.Option(False, "--force", help="Include partials touched recently (skips the active-download guard)."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Reclaim disk from abandoned ``*.incomplete`` download partials."""
    cfg = _load_config()
    lib = Library.load()
    plan_out = build_gc_plan(cfg, lib, model=model, locks=locks, force=force)
    if plan_out.is_empty():
        if plan_out.protected:
            info(f"nothing to reclaim ({len(plan_out.protected)} partial(s) protected as recently active; --force to include).")
        else:
            info("nothing to reclaim: no abandoned partials found.")
        return
    render_gc_plan(plan_out)
    if not yes and not flags.dry_run:
        if not typer.confirm("Proceed?"):
            info("aborted.")
            raise typer.Exit(1)
    apply_gc(cfg, plan_out)


# -- verify --------------------------------------------------------------------------------
@app.command()
def verify(
    model: str = typer.Argument(..., help="Model id / name as shown by `mdl list` (or a repo id)."),
    repair: bool = typer.Option(False, "--repair", help="Re-download any incomplete format (resumes from disk)."),
    retries: int = typer.Option(0, "--retries", help="With --repair: retry a failed download N times."),
) -> None:
    """Check a model's files against the Hub and optionally repair gaps."""
    cfg = _load_config()
    lib = Library.load()
    ok = verify_model(cfg, lib, model, repair=repair, retries=retries)
    raise typer.Exit(0 if ok else 1)


# -- convert -------------------------------------------------------------------------------
@app.command()
def convert(
    source: str = typer.Argument(..., help="HF repo id or local model directory."),
    quant: Optional[str] = typer.Option(None, "--quant", help="Target quant (default from config)."),
    remote: bool = typer.Option(False, "--remote", help="Stream weights from the Hub (source must be a repo id)."),
    out_as: Optional[str] = typer.Option(None, "--as", help="publisher/model to use for the output folder."),
) -> None:
    """Standalone safetensors -> GGUF conversion."""
    cfg = _load_config()
    q = quant or cfg.default_quant
    local = expand_path(source)
    if local.exists():
        model_name = local.name
        target_dir = lmstudio_target_dir(cfg.gguf_dir, out_as or f"_local/{model_name}")
        convert_model(cfg, source=local, quant=q, target_dir=target_dir, model_name=model_name, remote=False)
        return
    if "/" not in source:
        raise MdlError(f"'{source}' is neither an existing path nor a valid repo id.", hint="Use owner/name or a folder path.")
    _owner, name = split_repo_id(source)
    target_dir = lmstudio_target_dir(cfg.gguf_dir, out_as or source)
    if remote:
        convert_model(cfg, source=source, quant=q, target_dir=target_dir, model_name=name, remote=True)
        return
    hub.download_raw(cfg, source)
    snap = hub.snapshot_path(cfg, source)
    if snap is None and not flags.dry_run:
        raise MdlError(
            f"downloaded '{source}' but could not locate its snapshot in the HF cache.",
            hint="Check that the download completed and HF_HOME points at the right cache.",
        )
    convert_model(
        cfg,
        source=snap or (hub.cache_dir(cfg, source) / "snapshots"),
        quant=q,
        target_dir=target_dir,
        model_name=name,
        remote=False,
    )


# -- config --------------------------------------------------------------------------------
@config_app.callback()
def config_main(ctx: typer.Context) -> None:
    """Show the current configuration (use `mdl config set <key> <value>` to edit)."""
    if ctx.invoked_subcommand is not None:
        return
    cfg = _load_config()
    table = Table(show_lines=False, title=f"config: {CONFIG_PATH}")
    table.add_column("Key", style="bold", no_wrap=True)
    table.add_column("Value (raw)", overflow="fold")
    table.add_column("Expanded", overflow="fold")
    for key in DEFAULTS:
        raw = cfg.raw(key)
        expanded = str(cfg.expanded(key)) if key in PATH_KEYS else ""
        table.add_row(key, raw, expanded)
    console.print(table)


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Config key (e.g. gguf_dir)."),
    value: str = typer.Argument(..., help="New value (Windows paths are stored as-is)."),
) -> None:
    """Set a config key."""
    cfg = _load_config()
    cfg.set(key, value)
    cfg.save()
    success(f"set [bold]{key}[/] = {value}")
    if key in PATH_KEYS:
        console.print(f"  expanded: {cfg.expanded(key)}")


# -- doctor --------------------------------------------------------------------------------
@app.command()
def doctor() -> None:
    """Windows-aware status check of drives, tools, env vars and runtimes."""
    cfg = _load_config()
    ok = run_doctor(cfg)
    raise typer.Exit(0 if ok else 1)


def main() -> None:
    try:
        app()
    except MdlError as exc:
        err_console.print(f"[red]error:[/] {exc.message}")
        if exc.hint:
            err_console.print(f"[dim]hint:[/] {exc.hint}")
        raise SystemExit(1)
    except KeyboardInterrupt:
        err_console.print("[yellow]aborted.[/]")
        raise SystemExit(130)


if __name__ == "__main__":
    main()
