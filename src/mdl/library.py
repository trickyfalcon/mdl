"""The mdl library manifest + live inventory.

mdl records what it has added in ``%APPDATA%\\mdl\\library.json`` -- which repos, which quants,
and which runtimes were wired up. That manifest is the source of truth for membership and for
``sync`` (re-apply registrations); the on-disk *presence*, *size* and *drive* of each format are
computed live by scanning the actual stores so ``list`` stays honest even if files were moved
or deleted out of band.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path

from .config import CONFIG_DIR
from .errors import MdlError
from .hub import cache_dir
from .paths import detect_quant, drive_letter, lmstudio_target_dir, path_size, split_repo_id

LIBRARY_PATH = CONFIG_DIR / "library.json"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class ModelRecord:
    model: str
    raw_repo: str | None = None
    gguf_repo: str | None = None
    quants: list[str] = field(default_factory=list)
    ollama: list[str] = field(default_factory=list)
    lmstudio: bool = False
    added_at: str | None = None
    updated_at: str | None = None

    def display_name(self) -> str:
        _owner, name = split_repo_id(self.model)
        return name or self.model

    def gguf_dir_for(self, gguf_root: Path) -> Path:
        ref = self.gguf_repo or self.raw_repo or self.model
        return lmstudio_target_dir(gguf_root, ref)


@dataclass
class FormatInfo:
    present: bool
    drive: str
    size: int
    path: Path


@dataclass
class Row:
    model: str
    raw: FormatInfo
    gguf: FormatInfo
    quants: list[str]
    ollama: list[str]
    lmstudio: bool


class Library:
    def __init__(self, records: dict[str, ModelRecord] | None = None, path: Path = LIBRARY_PATH):
        self.records: dict[str, ModelRecord] = records or {}
        self.path = path

    # -- persistence ----------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path = LIBRARY_PATH) -> "Library":
        if not path.exists():
            return cls({}, path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls({}, path)
        records: dict[str, ModelRecord] = {}
        allowed = {f.name for f in fields(ModelRecord)}
        try:
            for model, raw in data.get("models", {}).items():
                clean = {k: v for k, v in raw.items() if k in allowed}
                clean["model"] = model
                records[model] = ModelRecord(**clean)
        except (AttributeError, TypeError) as exc:
            raise MdlError(
                f"Could not read library manifest at {path}: {exc}",
                hint="Fix the file by hand or delete it to regenerate it.",
            ) from exc
        return cls(records, path)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "models": {m: {k: v for k, v in asdict(r).items() if k != "model"} for m, r in self.records.items()},
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # -- mutation -------------------------------------------------------------------------
    def upsert(
        self,
        model: str,
        *,
        raw_repo: str | None = None,
        gguf_repo: str | None = None,
        quants: list[str] | None = None,
        ollama: list[str] | None = None,
        lmstudio: bool | None = None,
    ) -> ModelRecord:
        rec = self.records.get(model)
        if rec is None:
            rec = ModelRecord(model=model, added_at=_now())
            self.records[model] = rec
        if raw_repo is not None:
            rec.raw_repo = raw_repo
        if gguf_repo is not None:
            rec.gguf_repo = gguf_repo
        if quants:
            rec.quants = sorted(set(rec.quants) | set(quants))
        if ollama:
            rec.ollama = sorted(set(rec.ollama) | set(ollama))
        if lmstudio is not None:
            rec.lmstudio = lmstudio
        rec.updated_at = _now()
        return rec

    def remove(self, model: str) -> None:
        self.records.pop(model, None)

    # -- lookup ---------------------------------------------------------------------------
    def find(self, query: str) -> ModelRecord | None:
        q = query.strip().lower()
        # exact id
        for model, rec in self.records.items():
            if model.lower() == q:
                return rec
        # name component / gguf repo / ollama tag
        for rec in self.records.values():
            if rec.display_name().lower() == q:
                return rec
            if rec.gguf_repo and rec.gguf_repo.lower() == q:
                return rec
            if any(name.lower() == q or name.split(":")[0].lower() == q for name in rec.ollama):
                return rec
        # last resort: unique substring on the model id
        matches = [rec for model, rec in self.records.items() if q in model.lower()]
        return matches[0] if len(matches) == 1 else None


def _scan_gguf(gdir: Path) -> list[Path]:
    """List .gguf files under a dir, tolerating a missing/locked/unreadable tree."""
    if not gdir.exists():
        return []
    try:
        return sorted(gdir.rglob("*.gguf"))
    except OSError:
        return []


def inventory(cfg, library: Library) -> list[Row]:
    """Build display rows by scanning the real stores for each recorded model."""
    rows: list[Row] = []
    for rec in library.records.values():
        # raw
        raw_present, raw_size, raw_path = False, 0, Path("")
        if rec.raw_repo:
            raw_path = cache_dir(cfg, rec.raw_repo)
            raw_size = path_size(raw_path)
            raw_present = raw_size > 0
        raw_info = FormatInfo(raw_present, drive_letter(cfg.hf_home), raw_size, raw_path)

        # gguf
        gdir = rec.gguf_dir_for(cfg.gguf_dir)
        gguf_files = _scan_gguf(gdir)
        gguf_size = path_size(gdir) if gguf_files else 0
        gguf_info = FormatInfo(bool(gguf_files), drive_letter(cfg.gguf_dir), gguf_size, gdir)

        quants = sorted(set(rec.quants) | {q for f in gguf_files if (q := detect_quant(f.name))})

        rows.append(
            Row(
                model=rec.model,
                raw=raw_info,
                gguf=gguf_info,
                quants=quants,
                ollama=list(rec.ollama),
                lmstudio=rec.lmstudio,
            )
        )
    return rows
