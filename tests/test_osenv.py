"""Tests for the per-OS abstraction (exe names, defaults, env-set hints)."""

from mdl import osenv
from mdl.config import PATH_KEYS


def test_exe_windows(monkeypatch):
    monkeypatch.setattr(osenv, "IS_WINDOWS", True)
    assert osenv.exe("hf") == "hf.exe"
    assert osenv.exe("llama-quantize") == "llama-quantize.exe"


def test_exe_posix(monkeypatch):
    monkeypatch.setattr(osenv, "IS_WINDOWS", False)
    assert osenv.exe("hf") == "hf"
    assert osenv.exe("uv") == "uv"


def test_default_config_same_keys_both_os(monkeypatch):
    monkeypatch.setattr(osenv, "IS_WINDOWS", True)
    win = osenv.default_config()
    monkeypatch.setattr(osenv, "IS_WINDOWS", False)
    posix = osenv.default_config()
    assert set(win) == set(posix)  # only values differ across OSes
    # every path key is present in both, and shared tunables match
    assert PATH_KEYS <= set(win)
    assert win["default_quant"] == posix["default_quant"] == "Q4_K_M"
    assert win["download_stall_timeout"] == posix["download_stall_timeout"]


def test_posix_defaults_are_home_rooted(monkeypatch):
    monkeypatch.setattr(osenv, "IS_WINDOWS", False)
    cfg = osenv.default_config()
    assert cfg["gguf_dir"].startswith("~/") and "\\" not in cfg["gguf_dir"]
    assert cfg["hf_home"] == "~/models/hf"
    assert not cfg["llama_quantize"].endswith(".exe")


def test_windows_defaults_use_drive_and_exe(monkeypatch):
    monkeypatch.setattr(osenv, "IS_WINDOWS", True)
    cfg = osenv.default_config()
    assert cfg["gguf_dir"][1:3] == r":\\"[:2]  # like "D:\"
    assert cfg["llama_quantize"].endswith(".exe")


def test_env_set_hint(monkeypatch):
    monkeypatch.setattr(osenv, "IS_WINDOWS", True)
    assert osenv.env_set_hint("HF_HOME", "X:\\m") == 'setx HF_HOME "X:\\m"'
    monkeypatch.setattr(osenv, "IS_WINDOWS", False)
    assert osenv.env_set_hint("HF_HOME", "/m").startswith('export HF_HOME="/m"')
