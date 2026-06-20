"""Tests for the advisory per-repo download lock."""

from mdl import locks
from mdl.locks import RepoLock, repo_lock


def _point_locks_at(tmp_path, monkeypatch):
    """Redirect the lock directory into a tmp dir for the duration of a test."""
    monkeypatch.setattr(locks, "CONFIG_DIR", tmp_path)


def test_second_acquire_fails_while_held(tmp_path, monkeypatch):
    _point_locks_at(tmp_path, monkeypatch)
    a = RepoLock("owner/Model")
    b = RepoLock("owner/Model")
    assert a.acquire() is True
    try:
        assert b.acquire() is False  # same repo already locked
    finally:
        a.release()


def test_release_allows_reacquire(tmp_path, monkeypatch):
    _point_locks_at(tmp_path, monkeypatch)
    a = RepoLock("owner/Model")
    assert a.acquire() is True
    a.release()
    b = RepoLock("owner/Model")
    assert b.acquire() is True  # freed -> available again
    b.release()


def test_different_repos_do_not_conflict(tmp_path, monkeypatch):
    _point_locks_at(tmp_path, monkeypatch)
    a, b = RepoLock("owner/A"), RepoLock("owner/B")
    assert a.acquire() and b.acquire()
    a.release()
    b.release()


def test_context_manager_releases(tmp_path, monkeypatch):
    _point_locks_at(tmp_path, monkeypatch)
    with repo_lock("owner/Model") as held:
        assert held is True
    # lock released on exit -> can take it again
    assert RepoLock("owner/Model").acquire() is True


def test_disabled_context_is_noop(tmp_path, monkeypatch):
    _point_locks_at(tmp_path, monkeypatch)
    with repo_lock("owner/Model", enabled=False) as held:
        assert held is True
        # disabled lock doesn't actually hold anything, so a real one still succeeds
        real = RepoLock("owner/Model")
        assert real.acquire() is True
        real.release()
