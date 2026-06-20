"""Tests for the convert/quantize command builders."""

from pathlib import Path

import mdl.convert as cv
from mdl.config import DEFAULTS, Config


def _cfg():
    return Config(dict(DEFAULTS))


def test_quantize_command():
    cfg = _cfg()
    cmd = cv.quantize_command(cfg, Path(r"D:\g\m.f16.gguf"), Path(r"D:\g\m-Q4_K_M.gguf"), "Q4_K_M")
    assert cmd == [str(cfg.llama_quantize), r"D:\g\m.f16.gguf", r"D:\g\m-Q4_K_M.gguf", "Q4_K_M"]


def test_convert_command_local(monkeypatch):
    cfg = _cfg()
    monkeypatch.setattr(cv, "python_runner", lambda c: ["python"])
    cmd = cv.convert_command(cfg, Path(r"H:\snap"), Path(r"D:\g\m.f16.gguf"), remote=False, outtype="f16")
    assert cmd == ["python", str(cfg.convert_script), "--outfile", r"D:\g\m.f16.gguf", "--outtype", "f16", r"H:\snap"]


def test_convert_command_remote_puts_repo_id_last(monkeypatch):
    cfg = _cfg()
    monkeypatch.setattr(cv, "python_runner", lambda c: ["python"])
    cmd = cv.convert_command(cfg, "org/model", Path(r"D:\g\m.f16.gguf"), remote=True)
    assert cmd[:2] == ["python", str(cfg.convert_script)]
    assert "--remote" in cmd
    assert cmd[-1] == "org/model"
