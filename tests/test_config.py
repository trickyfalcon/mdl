"""Tests for config literal-TOML writing and round-tripping."""

from mdl.config import DEFAULTS, Config, _dump_value
from mdl.osenv import IS_WINDOWS
from mdl.paths import expand_path


def test_dump_value_uses_literal_string():
    assert _dump_value(r"D:\models\gguf") == r"'D:\models\gguf'"
    assert _dump_value("%USERPROFILE%\\.lmstudio\\models") == "'%USERPROFILE%\\.lmstudio\\models'"


def test_dump_value_falls_back_to_basic_string_on_quote():
    out = _dump_value("it's")
    assert out.startswith('"') and out.endswith('"')
    assert "it's" in out


def test_config_roundtrip_preserves_paths(tmp_path):
    path = tmp_path / "config.toml"
    Config.load(create=True, path=path)
    text = path.read_text(encoding="utf-8")
    # literal single-quoted, written verbatim (backslashes on Windows are NOT escaped)
    assert f"gguf_dir = '{DEFAULTS['gguf_dir']}'" in text
    if IS_WINDOWS:
        assert "gguf_dir = 'D:\\models\\gguf'" in text  # backslashes survive

    reloaded = Config.load(path=path)
    assert reloaded.raw("gguf_dir") == DEFAULTS["gguf_dir"]
    assert reloaded.raw("default_quant") == "Q4_K_M"


def test_config_set_persists(tmp_path):
    path = tmp_path / "config.toml"
    cfg = Config.load(create=True, path=path)
    cfg.set("gguf_dir", r"E:\g")
    cfg.save()
    assert Config.load(path=path).raw("gguf_dir") == r"E:\g"


def test_config_expanded_paths():
    cfg = Config(dict(DEFAULTS))
    assert cfg.expanded("gguf_dir") == expand_path(DEFAULTS["gguf_dir"])
    assert cfg.ollama_bin == "ollama"
    assert cfg.default_quant == "Q4_K_M"


def test_download_timeout_accessors_and_fallbacks():
    cfg = Config(dict(DEFAULTS))
    assert cfg.download_timeout == 30
    assert cfg.download_stall_timeout == 300
    cfg.values["download_stall_timeout"] = "not-a-number"
    assert cfg.download_stall_timeout == 300  # bad value -> default
    cfg.values["download_timeout"] = "-5"
    assert cfg.download_timeout == 30  # negative -> default
    cfg.values["download_timeout"] = "120"
    assert cfg.download_timeout == 120  # valid override honoured


def test_config_rejects_unknown_key():
    cfg = Config(dict(DEFAULTS))
    import pytest

    from mdl.errors import ConfigError

    with pytest.raises(ConfigError):
        cfg.set("nonsense_key", "x")
