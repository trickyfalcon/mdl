"""Advisory per-repo lock so two ``mdl add`` runs can't fight over the same download.

The lock is an OS-level byte-range lock held by an open file handle: the kernel releases it
automatically when the process exits (even on a crash), so there are no stale lock files to
reap. It is *advisory at the mdl layer* -- hf still has its own internal file locks that keep
the cache itself consistent; this just turns "two runs racing the same repo" into a clean,
early error instead of a confusing wait.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

from .config import CONFIG_DIR
from .paths import split_repo_id


def _lock_path(repo_id: str) -> Path:
    owner, name = split_repo_id(repo_id)
    safe = f"{owner}--{name}".strip("-") or "model"
    return CONFIG_DIR / "locks" / f"{safe}.lock"


class RepoLock:
    """A non-blocking, OS-enforced lock keyed by a repo id."""

    def __init__(self, repo_id: str) -> None:
        self.path = _lock_path(repo_id)
        self._fh = None

    def acquire(self) -> bool:
        """Return True if the lock was taken, False if another handle already holds it."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+")
        try:
            fh.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False
        self._fh = fh
        return True

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            self._fh.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            self._fh.close()
            self._fh = None


@contextmanager
def repo_lock(repo_id: str, *, enabled: bool = True):
    """Context manager yielding True if the lock is held (or disabled), False if taken elsewhere."""
    if not enabled:
        yield True
        return
    lock = RepoLock(repo_id)
    acquired = lock.acquire()
    try:
        yield acquired
    finally:
        if acquired:
            lock.release()
