"""Tests for pure path/repo helpers."""

from pathlib import Path

from mdl.paths import (
    detect_quant,
    drive_letter,
    expand_path,
    human_size,
    lmstudio_target_dir,
    quant_glob,
    same_path,
    split_repo_id,
)


def test_expand_path_expands_percent_userprofile(monkeypatch):
    monkeypatch.setenv("USERPROFILE", r"C:\Users\tester")
    assert expand_path(r"%USERPROFILE%\.lmstudio\models") == Path(r"C:\Users\tester\.lmstudio\models")


def test_expand_path_expands_percent_appdata(monkeypatch):
    monkeypatch.setenv("APPDATA", r"C:\Users\tester\AppData\Roaming")
    assert expand_path(r"%APPDATA%\mdl") == Path(r"C:\Users\tester\AppData\Roaming\mdl")


def test_expand_path_expands_tilde(monkeypatch):
    monkeypatch.delenv("HOME", raising=False)  # HOME would win over USERPROFILE on Windows
    monkeypatch.setenv("USERPROFILE", r"C:\Users\tester")
    assert expand_path(r"~\models") == Path(r"C:\Users\tester\models")


def test_expand_path_leaves_plain_path():
    assert expand_path(r"D:\models\gguf") == Path(r"D:\models\gguf")


def test_expand_path_unknown_var_left_intact(monkeypatch):
    monkeypatch.delenv("DEFINITELY_NOT_SET", raising=False)
    assert "%DEFINITELY_NOT_SET%" in str(expand_path(r"%DEFINITELY_NOT_SET%\x"))


def test_split_repo_id():
    assert split_repo_id("Qwen/Qwen3-32B") == ("Qwen", "Qwen3-32B")
    assert split_repo_id("bartowski/Qwen3-32B-GGUF") == ("bartowski", "Qwen3-32B-GGUF")
    assert split_repo_id("noslash") == ("", "noslash")
    assert split_repo_id("  Qwen/Qwen3-32B  ") == ("Qwen", "Qwen3-32B")


def test_lmstudio_target_dir_from_gguf_repo():
    g = Path(r"D:\models\gguf")
    assert lmstudio_target_dir(g, "bartowski/Qwen3-32B-GGUF") == g / "bartowski" / "Qwen3-32B-GGUF"


def test_lmstudio_target_dir_from_raw_repo():
    g = Path(r"D:\models\gguf")
    assert lmstudio_target_dir(g, "Qwen/Qwen3-32B") == g / "Qwen" / "Qwen3-32B"


def test_lmstudio_target_dir_no_publisher_falls_back():
    g = Path(r"D:\models\gguf")
    assert lmstudio_target_dir(g, "loosemodel") == g / "_local" / "loosemodel"


def test_human_size():
    assert human_size(0) == "0 B"
    assert human_size(512) == "512 B"
    assert human_size(1024) == "1.0 KB"
    assert human_size(1536) == "1.5 KB"
    assert human_size(5 * 1024 ** 3) == "5.0 GB"


def test_drive_letter():
    assert drive_letter(r"D:\models\gguf") == "D:"
    assert drive_letter(r"h:\x") == "H:"


def test_same_path_case_and_sep_insensitive():
    assert same_path(r"D:\models\gguf", "d:\\models\\gguf")
    assert same_path("D:\\models\\gguf", "D:\\models\\gguf\\")
    assert same_path(r"D:\a\..\b", r"D:\b")
    assert not same_path(r"D:\models\gguf", r"C:\models\gguf")


def test_quant_glob():
    assert quant_glob("Q4_K_M") == "*Q4_K_M*"


def test_detect_quant():
    assert detect_quant("Qwen3-32B-Q4_K_M.gguf") == "Q4_K_M"
    assert detect_quant("model-Q8_0.gguf") == "Q8_0"
    assert detect_quant("model-IQ4_XS.gguf") == "IQ4_XS"
    assert detect_quant("model-f16.gguf") == "F16"
    assert detect_quant("plain-model.gguf") is None
