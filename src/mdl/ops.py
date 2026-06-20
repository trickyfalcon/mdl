"""``rm`` and ``sync`` operations.

* ``rm`` deletes formats across stores (raw HF cache on H:, GGUF master on D:) and/or removes
  runtime registrations (``ollama rm``; LM Studio needs no deletion -- dropping the GGUF makes
  it disappear from LM Studio's list). It always shows *what* will go and on *which drive*.
* ``sync`` re-applies every recorded registration from the current config -- the thing to run
  after moving the library (new drive letters, a NAS): repoint config, then ``sync``.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .add import pick_primary_gguf
from .console import console, info, is_dry, plan, step, success, warn
from .errors import MdlError
from .hub import cache_dir
from .library import Library, ModelRecord
from .paths import drive_letter, human_size, path_size
from .registry import lmstudio, ollama

_VALID_RUNTIMES = {"ollama", "lmstudio"}
_VALID_FORMATS = {"raw", "gguf", "all"}


@dataclass
class RemovalItem:
    kind: str
    path: Path
    drive: str
    size: int


@dataclass
class RemovalPlan:
    record: ModelRecord
    fmt: str
    runtimes: set[str]
    files: list[RemovalItem]
    ollama_names: list[str]
    unset_lmstudio: bool

    def is_empty(self) -> bool:
        return not self.files and not self.ollama_names and not self.unset_lmstudio


def build_removal_plan(
    cfg, library: Library, query: str, *, fmt: str | None, from_runtimes: set[str] | None
) -> RemovalPlan:
    rec = library.find(query)
    if rec is None:
        raise MdlError(f"'{query}' is not in the mdl library.", hint="Run `mdl list` to see what's tracked.")

    full = fmt is None and from_runtimes is None
    fmt_eff = (fmt or ("all" if full else "none")).lower()
    if fmt_eff not in _VALID_FORMATS | {"none"}:
        raise MdlError(f"Invalid --format '{fmt}'.", hint="Use raw, gguf, or all.")
    runtimes = set(from_runtimes) if from_runtimes is not None else (set(_VALID_RUNTIMES) if full else set())

    files: list[RemovalItem] = []
    if fmt_eff in ("raw", "all") and rec.raw_repo:
        d = cache_dir(cfg, rec.raw_repo)
        if d.exists():
            files.append(RemovalItem("raw (HF cache)", d, drive_letter(cfg.hf_home), path_size(d)))
    if fmt_eff in ("gguf", "all"):
        d = rec.gguf_dir_for(cfg.gguf_dir)
        if d.exists():
            files.append(RemovalItem("gguf master", d, drive_letter(cfg.gguf_dir), path_size(d)))

    ollama_names = list(rec.ollama) if "ollama" in runtimes else []
    unset_lmstudio = "lmstudio" in runtimes
    return RemovalPlan(rec, fmt_eff, runtimes, files, ollama_names, unset_lmstudio)


def render_removal_plan(removal: RemovalPlan) -> None:
    console.print(f"[bold]Will remove for[/] {removal.record.model}:")
    total = 0
    for item in removal.files:
        total += item.size
        console.print(f"  [red]delete[/] {item.kind:<16} [{item.drive}] {item.path}  ({human_size(item.size)})")
    for name in removal.ollama_names:
        console.print(f"  [red]ollama rm[/] {name}")
    if removal.unset_lmstudio:
        console.print("  [yellow]lmstudio[/] drop registration flag (delete the GGUF to remove it from LM Studio)")
    if removal.files:
        console.print(f"  [bold]total on-disk freed:[/] {human_size(total)}")


def apply_removal(cfg, library: Library, removal: RemovalPlan) -> None:
    rec = removal.record
    for item in removal.files:
        if is_dry():
            plan(f"delete {item.path}  ({human_size(item.size)} on {item.drive})")
            continue
        try:
            shutil.rmtree(item.path)
            success(f"deleted {item.kind}: {item.path}")
        except OSError as exc:
            warn(f"could not delete {item.path}: {exc}")

    for name in removal.ollama_names:
        ollama.remove(cfg, name)

    if is_dry():
        return

    rec.ollama = [n for n in rec.ollama if n not in removal.ollama_names]
    if removal.unset_lmstudio:
        rec.lmstudio = False
    raw_present = bool(rec.raw_repo) and cache_dir(cfg, rec.raw_repo).exists()
    gguf_present = rec.gguf_dir_for(cfg.gguf_dir).exists()
    if not raw_present and not gguf_present and not rec.ollama and not rec.lmstudio:
        library.remove(rec.model)
        info(f"removed '{rec.model}' from the library (nothing left).")
    else:
        rec.updated_at = datetime.now().isoformat(timespec="seconds")
    library.save()


def sync_all(cfg, library: Library) -> None:
    if not library.records:
        info("library is empty; nothing to sync.")
        return
    for rec in library.records.values():
        console.rule(rec.model)
        gdir = rec.gguf_dir_for(cfg.gguf_dir)
        quant = rec.quants[0] if rec.quants else ""
        primary = pick_primary_gguf(gdir, quant)

        if rec.lmstudio:
            if gdir.exists():
                lmstudio.register(cfg, gdir)
            else:
                warn(f"{rec.model}: GGUF dir missing ({gdir}); can't verify for LM Studio.")

        for oname in rec.ollama:
            if primary is not None:
                step(f"re-import {oname} into ollama")
                ollama.import_gguf(cfg, primary, oname)
            else:
                warn(f"{rec.model}: no GGUF on disk to re-import into ollama as '{oname}'.")

    if not is_dry():
        library.save()
    success("sync complete.")
