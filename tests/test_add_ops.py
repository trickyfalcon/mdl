"""Tests for add-side pure helpers: register parsing + primary GGUF selection."""

from mdl.add import parse_register, pick_primary_gguf


def test_parse_register_default():
    assert parse_register("ollama,lmstudio") == {"ollama", "lmstudio"}


def test_parse_register_filters_unknown_and_whitespace():
    assert parse_register("ollama, bogus ,lmstudio") == {"ollama", "lmstudio"}


def test_parse_register_none_and_empty():
    assert parse_register("none") == set()
    assert parse_register("") == set()


def test_pick_primary_prefers_requested_quant(tmp_path):
    (tmp_path / "Qwen3-32B-Q4_K_M.gguf").write_bytes(b"x")
    (tmp_path / "Qwen3-32B-Q8_0.gguf").write_bytes(b"x")
    assert pick_primary_gguf(tmp_path, "Q4_K_M").name == "Qwen3-32B-Q4_K_M.gguf"


def test_pick_primary_first_shard_when_only_split(tmp_path):
    (tmp_path / "m-Q4_K_M-00001-of-00002.gguf").write_bytes(b"x")
    (tmp_path / "m-Q4_K_M-00002-of-00002.gguf").write_bytes(b"x")
    assert "00001-of" in pick_primary_gguf(tmp_path, "Q4_K_M").name


def test_pick_primary_none_when_empty(tmp_path):
    assert pick_primary_gguf(tmp_path, "Q4_K_M") is None
