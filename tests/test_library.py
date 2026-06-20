"""Tests for the library manifest: merge semantics + robust loading."""

import json

import pytest

from mdl.errors import MdlError
from mdl.library import Library


def test_upsert_lmstudio_none_preserves_existing(tmp_path):
    lib = Library(path=tmp_path / "library.json")
    lib.upsert("a/b", gguf_repo="a/b-GGUF", lmstudio=True)
    assert lib.records["a/b"].lmstudio is True
    # re-adding without lmstudio in scope (lmstudio=None) must NOT clear the flag
    lib.upsert("a/b", quants=["Q4_K_M"], lmstudio=None)
    assert lib.records["a/b"].lmstudio is True


def test_upsert_merges_ollama_and_quants(tmp_path):
    lib = Library(path=tmp_path / "library.json")
    lib.upsert("a/b", ollama=["x:q4"], quants=["Q4_K_M"])
    lib.upsert("a/b", ollama=["y:q5"], quants=["Q5_K_M"])
    assert lib.records["a/b"].ollama == ["x:q4", "y:q5"]
    assert lib.records["a/b"].quants == ["Q4_K_M", "Q5_K_M"]


def test_load_drops_unknown_keys(tmp_path):
    path = tmp_path / "library.json"
    path.write_text(
        json.dumps({"version": 1, "models": {"a/b": {"raw_repo": "a/b", "future_field": 123}}}),
        encoding="utf-8",
    )
    lib = Library.load(path=path)
    assert "a/b" in lib.records
    assert lib.records["a/b"].raw_repo == "a/b"


def test_load_malformed_raises_mdlerror(tmp_path):
    path = tmp_path / "library.json"
    path.write_text(json.dumps({"version": 1, "models": [1, 2, 3]}), encoding="utf-8")
    with pytest.raises(MdlError):
        Library.load(path=path)


def test_load_missing_returns_empty(tmp_path):
    assert Library.load(path=tmp_path / "nope.json").records == {}


def test_roundtrip_save_load(tmp_path):
    path = tmp_path / "library.json"
    lib = Library(path=path)
    lib.upsert("Qwen/Qwen3-32B", gguf_repo="bartowski/Qwen3-32B-GGUF", quants=["Q4_K_M"], ollama=["qwen3-32b:q4_k_m"])
    lib.save()
    reloaded = Library.load(path=path)
    rec = reloaded.records["Qwen/Qwen3-32B"]
    assert rec.gguf_repo == "bartowski/Qwen3-32B-GGUF"
    assert rec.quants == ["Q4_K_M"]
    assert rec.ollama == ["qwen3-32b:q4_k_m"]
