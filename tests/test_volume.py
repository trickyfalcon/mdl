"""Tests for volume/NAS awareness: UNC detection, classification, marker + write guard."""

import os

import pytest

from mdl import volume
from mdl.errors import MdlError


# -- is_unc -------------------------------------------------------------------------------
def test_is_unc_true_for_unc_paths():
    assert volume.is_unc(r"\\synology\models\gguf")
    assert volume.is_unc("//synology/models")  # forward-slash form tolerated


def test_is_unc_false_for_local_paths():
    assert not volume.is_unc(r"D:\models\gguf")
    assert not volume.is_unc("/mnt/nas/models")


# -- kind ---------------------------------------------------------------------------------
def test_kind_unc_is_network_without_touching_disk():
    # short-circuits on the UNC string -> no API/filesystem call, no hang on a dead server
    assert volume.kind(r"\\dead-server\share\models") == "network"


def test_kind_local_tmp(tmp_path):
    # a pytest tmp dir lives on a normal local disk on any OS
    assert volume.kind(tmp_path) in {"local", "unknown"}


# -- marker round-trip --------------------------------------------------------------------
def test_mark_and_is_marked(tmp_path):
    assert not volume.is_marked(tmp_path)
    volume.mark(tmp_path)
    assert volume.is_marked(tmp_path)
    assert (tmp_path / volume.MARKER).exists()


# -- ensure_ready -------------------------------------------------------------------------
def test_ensure_ready_local_dir_no_marker(tmp_path, monkeypatch):
    # local volumes don't need a drop-detection marker -> none written
    monkeypatch.setattr(volume, "kind", lambda p: "local")
    dest = tmp_path / "models" / "hf"
    assert volume.ensure_ready(dest, "raw HF cache") == []
    assert not volume.is_marked(dest)


def test_ensure_ready_fresh_network_dir_writes_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(volume, "kind", lambda p: "network")
    dest = tmp_path / "models" / "hf"  # doesn't exist yet -> first use of a mounted share
    warnings = volume.ensure_ready(dest, "raw HF cache")
    assert warnings == []
    assert volume.is_marked(dest)


def test_ensure_ready_raises_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(volume, "kind", lambda p: "missing")
    with pytest.raises(MdlError, match="not mounted"):
        volume.ensure_ready(tmp_path / "x", "raw HF cache")


def test_ensure_ready_network_empty_unmarked_is_refused(tmp_path, monkeypatch):
    # simulate a dropped share: classified network, dir exists, empty, no marker
    monkeypatch.setattr(volume, "kind", lambda p: "network")
    dest = tmp_path / "share"
    dest.mkdir()
    with pytest.raises(MdlError, match="looks unmounted"):
        volume.ensure_ready(dest, "GGUF dir")


def test_ensure_ready_network_with_marker_ok(tmp_path, monkeypatch):
    monkeypatch.setattr(volume, "kind", lambda p: "network")
    dest = tmp_path / "share"
    volume.mark(dest)  # pretend we've used this mounted share before
    assert volume.ensure_ready(dest, "GGUF dir") == []


def test_ensure_ready_network_nonempty_ok(tmp_path, monkeypatch):
    # legacy data already present (pre-marker) -> not treated as a dropped mount
    monkeypatch.setattr(volume, "kind", lambda p: "network")
    dest = tmp_path / "share"
    dest.mkdir()
    (dest / "model.gguf").write_bytes(b"x")
    assert volume.ensure_ready(dest, "GGUF dir") == []


def test_status_returns_kind_and_reachable(tmp_path):
    k, reachable = volume.status(tmp_path)
    assert isinstance(k, str) and reachable is True
