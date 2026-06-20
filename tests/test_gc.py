"""Tests for `mdl gc`: abandoned-partial collection + the active-download guard."""

import time

from mdl.config import DEFAULTS, Config
from mdl.ops import build_gc_plan
from mdl.library import Library


def _cfg(tmp_path):
    cfg = Config(dict(DEFAULTS))
    cfg.values["hf_home"] = str(tmp_path / "hf")
    cfg.values["gguf_dir"] = str(tmp_path / "gguf")
    cfg.values["download_stall_timeout"] = "300"
    return cfg


def _touch(path, *, size=1024, age_seconds=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    if age_seconds:
        old = time.time() - age_seconds
        import os
        os.utime(path, (old, old))


def test_gc_collects_old_incomplete_partials(tmp_path):
    cfg = _cfg(tmp_path)
    blobs = tmp_path / "hf" / "hub" / "models--owner--Model" / "blobs"
    _touch(blobs / "abc.def.incomplete", size=2048, age_seconds=10_000)
    plan = build_gc_plan(cfg, Library())
    assert len(plan.items) == 1 and plan.total() == 2048


def test_gc_protects_recently_active_partials(tmp_path):
    cfg = _cfg(tmp_path)
    blobs = tmp_path / "hf" / "hub" / "models--owner--Model" / "blobs"
    _touch(blobs / "fresh.incomplete", size=4096, age_seconds=0)  # just written
    plan = build_gc_plan(cfg, Library())
    assert plan.is_empty() and len(plan.protected) == 1


def test_gc_force_includes_recent_partials(tmp_path):
    cfg = _cfg(tmp_path)
    blobs = tmp_path / "hf" / "hub" / "models--owner--Model" / "blobs"
    _touch(blobs / "fresh.incomplete", size=4096, age_seconds=0)
    plan = build_gc_plan(cfg, Library(), force=True)
    assert len(plan.items) == 1 and not plan.protected


def test_gc_locks_only_with_flag(tmp_path):
    cfg = _cfg(tmp_path)
    locks = tmp_path / "hf" / "hub" / ".locks" / "models--owner--Model"
    _touch(locks / "abc.lock", size=0, age_seconds=10_000)
    assert build_gc_plan(cfg, Library()).is_empty()  # locks ignored by default
    plan = build_gc_plan(cfg, Library(), locks=True)
    assert any(i.kind == "lock" for i in plan.items)


def test_gc_ignores_non_incomplete_files(tmp_path):
    cfg = _cfg(tmp_path)
    snap = tmp_path / "hf" / "hub" / "models--owner--Model" / "snapshots" / "rev0"
    _touch(snap / "model.safetensors", size=9999, age_seconds=10_000)
    assert build_gc_plan(cfg, Library()).is_empty()
