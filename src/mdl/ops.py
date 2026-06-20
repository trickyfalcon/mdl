"""``rm`` and ``sync`` operations.

* ``rm`` deletes formats across stores (raw HF cache on H:, GGUF master on D:) and/or removes
  runtime registrations (``ollama rm``; LM Studio needs no deletion -- dropping the GGUF makes
  it disappear from LM Studio's list). It always shows *what* will go and on *which drive*.
* ``sync`` re-applies every recorded registration from the current config -- the thing to run
  after moving the library (new drive letters, a NAS): repoint config, then ``sync``.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .add import pick_primary_gguf
from .console import console, info, is_dry, plan, step, success, warn
from .errors import MdlError
from .hub import cache_dir
from .library import Library, ModelRecord
from .paths import drive_letter, human_size, lmstudio_target_dir, path_size
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


# -- gc ------------------------------------------------------------------------------------
@dataclass
class GcItem:
    kind: str
    path: Path
    drive: str
    size: int


@dataclass
class GcPlan:
    items: list[GcItem] = field(default_factory=list)
    protected: list[Path] = field(default_factory=list)  # recently-active -> skipped for safety

    def total(self) -> int:
        return sum(i.size for i in self.items)

    def is_empty(self) -> bool:
        return not self.items


def _scoped_roots(cfg, library: Library, model: str | None) -> list[Path]:
    """Directories to sweep: a single model's stores, or the whole raw + gguf trees."""
    if not model:
        return [cfg.hf_home, cfg.gguf_dir]
    rec = library.find(model)
    if rec is not None:
        roots = []
        if rec.raw_repo:
            roots.append(cache_dir(cfg, rec.raw_repo))
        roots.append(rec.gguf_dir_for(cfg.gguf_dir))
        return roots
    if "/" in model:
        return [cache_dir(cfg, model), lmstudio_target_dir(cfg.gguf_dir, model)]
    raise MdlError(f"'{model}' is not in the library and is not a repo id.", hint="Run `mdl list`.")


def build_gc_plan(cfg, library: Library, *, model: str | None = None, locks: bool = False, force: bool = False) -> GcPlan:
    """Collect abandoned ``*.incomplete`` partials (and, with ``locks``, stale ``.lock`` files).

    A partial touched within ``download_stall_timeout`` seconds is assumed to belong to a
    live download and is *protected* (skipped) unless ``force`` -- so ``gc`` can't yank the
    floor out from under a running ``mdl add``.
    """
    plan_out = GcPlan()
    cutoff = time.time() - cfg.download_stall_timeout
    seen: set[Path] = set()
    for root in _scoped_roots(cfg, library, model):
        if not root.exists():
            continue
        try:
            partials = list(root.rglob("*.incomplete"))
        except OSError:
            partials = []
        for f in partials:
            if f in seen:
                continue
            seen.add(f)
            try:
                st = f.stat()
            except OSError:
                continue
            if not force and st.st_mtime > cutoff:
                plan_out.protected.append(f)
                continue
            plan_out.items.append(GcItem("incomplete", f, drive_letter(f), st.st_size))
        if locks:
            for lk in (root.rglob("*.lock") if root.exists() else []):
                if lk in seen:
                    continue
                seen.add(lk)
                try:
                    plan_out.items.append(GcItem("lock", lk, drive_letter(lk), lk.stat().st_size))
                except OSError:
                    continue
    return plan_out


def render_gc_plan(plan_out: GcPlan) -> None:
    console.print("[bold]Will reclaim:[/]")
    for item in plan_out.items:
        console.print(f"  [red]delete[/] {item.kind:<11} [{item.drive}] {item.path}  ({human_size(item.size)})")
    if plan_out.items:
        console.print(f"  [bold]total freed:[/] {human_size(plan_out.total())}")
    for p in plan_out.protected:
        console.print(f"  [yellow]skip[/] (recently active) {p}")


def apply_gc(cfg, plan_out: GcPlan) -> None:
    freed = 0
    for item in plan_out.items:
        if is_dry():
            plan(f"delete {item.path}  ({human_size(item.size)} on {item.drive})")
            continue
        try:
            item.path.unlink()
            freed += item.size
        except OSError as exc:
            warn(f"could not delete {item.path}: {exc}")
    if not is_dry():
        success(f"reclaimed {human_size(freed)}.")


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
