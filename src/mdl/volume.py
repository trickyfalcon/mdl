"""Volume / mount awareness, so mdl plays nicely with a NAS or other network storage.

Two jobs:
  * **classify** where a configured store lives -- local disk vs a network share -- so
    ``doctor`` and ``list`` can say so;
  * **guard** a large write: fail fast on an unmounted drive or a dropped share instead of
    silently filling the system disk or writing into a placeholder mount-point directory.

Detection is solid on Windows (``GetDriveType`` + UNC) and best-effort on POSIX (mount-point
heuristics + ``/proc`` fstype). When unsure we return ``"unknown"`` and never block a write.

The drop guard leans on a tiny ``.mdl-volume`` marker dropped in each store the first time we
use it: if a *network* store later exists but has lost its marker and is empty, the mount is
almost certainly down and we're looking at a local placeholder -- so we refuse.
"""

from __future__ import annotations

import os
from pathlib import Path

from .errors import MdlError

MARKER = ".mdl-volume"
_PROBE = ".mdl-write-test"

# POSIX network filesystems (exact match) -- anything ``fuse.*`` (rclone/sshfs/...) is also net.
_NETWORK_FSTYPES = {"nfs", "nfs4", "cifs", "smbfs", "smb", "afpfs", "9p", "ncpfs", "glusterfs"}
# where auto-mounts usually live, used for the unmounted-placeholder heuristic
_POSIX_MOUNT_PARENTS = ("/mnt", "/media", "/Volumes", "/run/media")


def is_unc(path: str | os.PathLike[str]) -> bool:
    """True for a Windows UNC path (``\\\\server\\share\\...``)."""
    s = str(path).replace("/", "\\")
    return s.startswith("\\\\")


def volume_root(path: str | os.PathLike[str]) -> Path:
    """The volume's root to probe: drive root (``D:\\``), UNC share root, or POSIX anchor."""
    p = Path(path)
    return Path(p.anchor) if p.anchor else p


def _nearest_existing(path: Path) -> Path:
    """Closest existing ancestor of ``path`` (or the path itself); for size/fstype probes."""
    p = path
    for _ in range(128):
        if p.exists():
            return p
        parent = p.parent
        if parent == p:
            return p
        p = parent
    return p


def _posix_fstype(path: Path) -> str | None:
    """Filesystem type of the mount that holds ``path`` on Linux (via /proc), else ``None``."""
    try:
        target = os.path.realpath(path)
        best_mp, best_type = "", None
        with open("/proc/self/mountinfo", encoding="utf-8") as fh:
            for line in fh:
                # ... <mount point> ... - <fstype> <source> ...
                parts = line.split()
                if " - " not in line or len(parts) < 5:
                    continue
                mp = parts[4]
                sep = parts.index("-")
                fstype = parts[sep + 1] if sep + 1 < len(parts) else None
                if (target == mp or target.startswith(mp.rstrip("/") + "/")) and len(mp) > len(best_mp):
                    best_mp, best_type = mp, fstype
        return best_type
    except OSError:
        return None


def kind(path: str | os.PathLike[str]) -> str:
    """``'local' | 'network' | 'removable' | 'missing' | 'unknown'`` for ``path``'s volume."""
    p = Path(path)
    if is_unc(p):
        return "network"
    if os.name == "nt":
        return _windows_kind(p)
    return _posix_kind(p)


def _windows_kind(p: Path) -> str:
    import ctypes  # Windows-only; imported lazily so the module loads on POSIX

    drive = p.drive
    if not drive:
        return "unknown"
    try:
        code = ctypes.windll.kernel32.GetDriveTypeW(drive + "\\")
    except Exception:
        return "unknown"
    # 0 unknown, 1 no-root, 2 removable, 3 fixed, 4 remote, 5 cdrom, 6 ramdisk
    return {0: "unknown", 1: "missing", 2: "removable", 3: "local", 4: "network", 5: "removable", 6: "local"}.get(
        code, "unknown"
    )


def _posix_kind(p: Path) -> str:
    fstype = _posix_fstype(_nearest_existing(p))
    if fstype is None:
        return "unknown"
    f = fstype.lower()
    if f in _NETWORK_FSTYPES or f.startswith("fuse."):
        return "network"
    return "local"


def is_marked(store: str | os.PathLike[str]) -> bool:
    return (Path(store) / MARKER).exists()


def mark(store: str | os.PathLike[str]) -> None:
    """Drop the ``.mdl-volume`` marker so a later dropped mount is detectable. Best-effort."""
    try:
        store = Path(store)
        store.mkdir(parents=True, exist_ok=True)
        (store / MARKER).write_text("mdl volume marker\n", encoding="utf-8")
    except OSError:
        pass


def _is_empty(store: Path) -> bool:
    try:
        return not any(p.name != MARKER for p in store.iterdir())
    except OSError:
        return False


def _posix_placeholder_warning(store: Path) -> str | None:
    """If ``store`` sits under a usual auto-mount parent that is NOT currently a mount point,
    it's probably an unmounted placeholder on the local disk. Heuristic -> a warning, not fatal."""
    if os.name == "nt":
        return None
    try:
        parts = Path(os.path.abspath(store)).parts
        for parent in _POSIX_MOUNT_PARENTS:
            pp = Path(parent)
            if len(parts) > len(pp.parts) and Path(*parts[: len(pp.parts) + 1]).parent == pp:
                mount_candidate = Path(*parts[: len(pp.parts) + 1])
                if mount_candidate.exists() and not os.path.ismount(mount_candidate):
                    return f"{mount_candidate} is not a mount point -- if this should be a NAS/USB mount, it looks unmounted."
    except OSError:
        return None
    return None


def status(store: str | os.PathLike[str]) -> tuple[str, bool]:
    """``(kind, reachable)`` for display. ``reachable`` = the volume root currently exists."""
    p = Path(store)
    root = volume_root(p)
    try:
        reachable = root.exists()
    except OSError:
        reachable = False
    return kind(p), reachable


def ensure_ready(store: str | os.PathLike[str], label: str) -> list[str]:
    """Make ``store`` ready for a large write. Raise :class:`MdlError` on a hard problem
    (unmounted drive, dropped share, not writable); return a list of soft warnings."""
    store = Path(store)
    warnings: list[str] = []
    k = kind(store)
    root = volume_root(store)

    if k == "missing":
        raise MdlError(
            f"{label}: volume {root} is not mounted.",
            hint="Mount the drive/NAS, or repoint it with `mdl config set`.",
        )

    # Dropped-share guard: a *network* store that exists but lost its marker and is empty is
    # almost certainly a local placeholder for a mount that's currently down.
    if k == "network" and store.exists() and not is_marked(store) and _is_empty(store):
        raise MdlError(
            f"{label} ({store}) is on a network volume but looks unmounted "
            f"(empty and missing its {MARKER} marker).",
            hint="Check the NAS/share is mounted. If this is genuinely a new location, create the folder and retry.",
        )

    placeholder = _posix_placeholder_warning(store)
    if placeholder:
        warnings.append(f"{label}: {placeholder}")

    # Universal proof: can we actually write here?
    try:
        store.mkdir(parents=True, exist_ok=True)
        probe = store / _PROBE
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise MdlError(
            f"{label} ({store}) is not writable: {type(exc).__name__}: {exc}",
            hint="Check the mount is online and you have write permission.",
        )
    if k == "network":
        mark(store)  # only network volumes need the dropped-mount marker
    return warnings
