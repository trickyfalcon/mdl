"""Tests for verify resolution/formatting and the list completeness cell."""

import pytest

from mdl.cli import _format_cell
from mdl.config import Config
from mdl.hub import DownloadStatus
from mdl.library import FormatInfo, Library
from mdl.errors import MdlError
from mdl.verify import _format_status, _resolve


def _status(state, present, expected, *, files=1, incomplete=0, verified=True):
    return DownloadStatus(state, present, files, expected, files, incomplete, verified)


# -- _resolve -----------------------------------------------------------------------------
def test_resolve_bare_repo_id():
    name, raw, gguf, quants = _resolve(Config(), Library(), "owner/Model")
    assert (name, raw, gguf, quants) == ("owner/Model", "owner/Model", None, [])


def test_resolve_library_record():
    lib = Library()
    lib.upsert("owner/Model", raw_repo="owner/Model", gguf_repo="owner/Model-GGUF", quants=["Q4_K_M"])
    name, raw, gguf, quants = _resolve(Config(), lib, "owner/Model")
    assert raw == "owner/Model" and gguf == "owner/Model-GGUF" and quants == ["Q4_K_M"]


def test_resolve_unknown_non_repo_raises():
    with pytest.raises(MdlError):
        _resolve(Config(), Library(), "not-a-repo")


# -- _format_status -----------------------------------------------------------------------
def test_format_status_complete_shows_percent():
    line = _format_status("raw", _status("complete", 1000, 1000))
    assert "complete" in line and "(100%)" in line


def test_format_status_partial_shows_incomplete_count():
    line = _format_status("raw", _status("partial", 500, 1000, incomplete=3))
    assert "partial" in line and "(50%)" in line and "3 partial file(s)" in line


def test_format_status_unverified_is_flagged():
    line = _format_status("raw", _status("partial", 500, None, verified=False))
    assert "unverified" in line


# -- cli _format_cell ---------------------------------------------------------------------
def test_format_cell_absent():
    fi = FormatInfo(present=False, drive="H:", size=0, path=None)
    assert _format_cell(fi) == "[dim]-[/]"


def test_format_cell_local_partial_marker():
    fi = FormatInfo(present=True, drive="H:", size=100, path=None, incomplete=2)
    cell = _format_cell(fi)
    assert "H: 100 B" in cell and "partial:2" in cell


def test_format_cell_checked_complete():
    fi = FormatInfo(present=True, drive="H:", size=100, path=None, expected=100, state="complete")
    assert "OK" in _format_cell(fi)


def test_format_cell_checked_percent():
    fi = FormatInfo(present=True, drive="D:", size=50, path=None, expected=200, state="partial")
    assert "25%" in _format_cell(fi)
