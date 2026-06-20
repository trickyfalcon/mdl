"""Tests for `mdl resume` param resolution (library-aware, flag overrides)."""

import pytest

from mdl.add import resume_params, _register_from_record
from mdl.errors import MdlError
from mdl.library import Library


def _lib_with(model, **fields):
    lib = Library()
    lib.upsert(model, **fields)
    return lib


# -- untracked repo id (the DeepSeek case) ------------------------------------------------
def test_resume_untracked_defaults_to_raw_only():
    p = resume_params(Library(), "deepseek-ai/DeepSeek-V4-Pro")
    assert p["raw_repo"] == "deepseek-ai/DeepSeek-V4-Pro"
    assert p["raw"] is True and p["gguf"] is False and p["register"] == "none"


def test_resume_untracked_honors_explicit_flags():
    p = resume_params(Library(), "owner/Model", gguf=True, register="ollama")
    assert p["gguf"] is True and p["register"] == "ollama"


def test_resume_untracked_non_repo_raises():
    with pytest.raises(MdlError):
        resume_params(Library(), "not-a-repo")


# -- tracked model: infer from the record -------------------------------------------------
def test_resume_tracked_infers_from_record():
    lib = _lib_with(
        "Qwen/Qwen3-32B",
        raw_repo="Qwen/Qwen3-32B",
        gguf_repo="bartowski/Qwen3-32B-GGUF",
        quants=["Q4_K_M"],
        ollama=["qwen3-32b:q4_k_m"],
        lmstudio=True,
    )
    p = resume_params(lib, "Qwen/Qwen3-32B")
    assert p["raw_repo"] == "Qwen/Qwen3-32B"
    assert p["gguf_repo"] == "bartowski/Qwen3-32B-GGUF"
    assert p["quant"] == "Q4_K_M"
    assert p["raw"] is True and p["gguf"] is True
    assert set(p["register"].split(",")) == {"ollama", "lmstudio"}


def test_resume_tracked_explicit_flag_overrides_record():
    lib = _lib_with("owner/Model", raw_repo="owner/Model", gguf_repo="owner/Model-GGUF", quants=["Q4_K_M"], lmstudio=True)
    p = resume_params(lib, "owner/Model", gguf=False, register="none")
    assert p["gguf"] is False and p["register"] == "none"


def test_resume_tracked_raw_only_record():
    lib = _lib_with("owner/Raw", raw_repo="owner/Raw")  # no gguf, no runtimes
    p = resume_params(lib, "owner/Raw")
    assert p["raw"] is True and p["gguf"] is False and p["register"] == "none"


# -- register reconstruction --------------------------------------------------------------
def test_register_from_record_variants():
    lib = _lib_with("a/b", ollama=["b:q4"], lmstudio=True)
    assert set(_register_from_record(lib.find("a/b")).split(",")) == {"ollama", "lmstudio"}
    lib2 = _lib_with("a/c")
    assert _register_from_record(lib2.find("a/c")) == "none"
