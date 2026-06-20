"""Tests for the Ollama import (streamed) and the doctor OLLAMA_MODELS daemon-mismatch check."""

import subprocess

import pytest

from mdl import doctor
from mdl.config import Config
from mdl.doctor import _ollama_has_data, _ollama_models_check
from mdl.errors import RegistrationError
from mdl.registry import ollama


# -- streamed `ollama create` -------------------------------------------------------------
def test_import_streams_create(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, **kw):
        calls.append((list(cmd), kw))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(ollama, "run", fake_run)
    monkeypatch.setattr(ollama, "model_exists", lambda cfg, name: False)
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    ollama.import_gguf(Config(), gguf, "test:q4")
    create = [c for c in calls if "create" in c[0]]
    assert create, "ollama create was never invoked"
    assert create[0][1].get("stream") is True  # the slow copy must stream so it isn't silent


def test_import_failure_probes_daemon(monkeypatch, tmp_path):
    monkeypatch.setattr(ollama, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", ""))
    monkeypatch.setattr(ollama, "model_exists", lambda cfg, name: False)
    monkeypatch.setattr(ollama, "ollama_running", lambda cfg: (False, "no daemon"))
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    with pytest.raises(RegistrationError, match="not running"):
        ollama.import_gguf(Config(), gguf, "test:q4")


# -- _ollama_has_data ---------------------------------------------------------------------
def test_ollama_has_data(tmp_path):
    assert not _ollama_has_data(tmp_path)
    (tmp_path / "blobs").mkdir()
    assert _ollama_has_data(tmp_path)


# -- doctor OLLAMA_MODELS check -----------------------------------------------------------
def _with_blobs(p):
    (p / "blobs").mkdir(parents=True)
    return p


def test_check_unset_notes_default_when_blobs_exist(tmp_path):
    default_loc = _with_blobs(tmp_path / "default")
    c = _ollama_models_check(None, tmp_path / "want", default_loc)
    assert c.status == "WARN" and str(default_loc) in c.detail


def test_check_detects_daemon_mismatch(tmp_path):
    # env set to an empty dir, but blobs live in the default location -> daemon predates env var
    default_loc = _with_blobs(tmp_path / "default")
    env = tmp_path / "D_models_ollama"  # set but no blobs yet
    c = _ollama_models_check(str(env), env, default_loc)
    assert c.status == "WARN"
    assert "predates the env var" in c.detail
    assert "quit & reopen" in c.fix.lower() or "reopen ollama" in c.fix.lower()


def test_check_ok_when_env_has_blobs(tmp_path, monkeypatch):
    monkeypatch.setattr(doctor, "IS_WINDOWS", False)  # tmp is on C: here; skip the C:-drive branch
    env = _with_blobs(tmp_path / "models")
    default_loc = tmp_path / "default"  # no blobs
    c = _ollama_models_check(str(env), env, default_loc)
    assert c.status == "OK" and "matches config" in c.detail


def test_check_no_mismatch_when_default_empty(tmp_path, monkeypatch):
    # fresh setup: nothing imported anywhere yet -> no false mismatch warning
    monkeypatch.setattr(doctor, "IS_WINDOWS", False)
    env = tmp_path / "models"
    default_loc = tmp_path / "default"
    c = _ollama_models_check(str(env), env, default_loc)
    assert c.status == "OK"
