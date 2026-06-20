"""Tests for config literal-TOML writing and round-tripping."""

from pathlib import Path

from mdl.config import DEFAULTS, Config, _dump_value


def test_dump_value_uses_literal_string():
    assert _dump_value(r"D:\models\gguf") == r"'D:\models\gguf'"
    assert _dump_value("%USERPROFILE%\\.lmstudio\\models") == "'%USERPROFILE%\\.lmstudio\\models'"


def test_dump_value_falls_back_to_basic_string_on_quote():
    out = _dump_value("it's")
    assert out.startswith('"') and out.endswith('"')
    assert "it's" in out


def test_config_roundtrip_preserves_backslashes(tmp_path):
    path = tmp_path / "config.toml"
    Config.load(create=True, path=path)
    text = path.read_text(encoding="utf-8")
    # literal single-quoted, backslashes verbatim
    assert "gguf_dir = 'D:\\models\\gguf'" in text
    assert "lmstudio_dir = '%USERPROFILE%\\.lmstudio\\models'" in text

    reloaded = Config.load(path=path)
    assert reloaded.raw("gguf_dir") == r"D:\models\gguf"
    assert reloaded.raw("default_quant") == "Q4_K_M"


def test_config_set_persists(tmp_path):
    path = tmp_path / "config.toml"
    cfg = Config.load(create=True, path=path)
    cfg.set("gguf_dir", r"E:\g")
    cfg.save()
    assert Config.load(path=path).raw("gguf_dir") == r"E:\g"


def test_config_expanded_paths():
    cfg = Config(dict(DEFAULTS))
    assert cfg.expanded("gguf_dir") == Path(r"D:\models\gguf")
    assert cfg.ollama_bin == "ollama"
    assert cfg.default_quant == "Q4_K_M"


def test_config_rejects_unknown_key():
    cfg = Config(dict(DEFAULTS))
    import pytest

    from mdl.errors import ConfigError

    with pytest.raises(ConfigError):
        cfg.set("nonsense_key", "x")
