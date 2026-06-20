"""Subprocess helper that understands ``--dry-run`` and ``--verbose``.

* dry-run: print the planned command (or a friendlier label) and run nothing.
* verbose: stream the child's output live *and* capture it, so callers can still scan it for
  known error signatures (e.g. a gated-repo 401) and raise a friendly error.
* otherwise: capture output silently for the same scanning.
"""

from __future__ import annotations

import re
import subprocess
import sys
from typing import Sequence

from .console import is_dry, is_verbose, plan, step

_TRAILING_BACKSLASHES = re.compile(r"(\\+)$")


def fmt_cmd(cmd: Sequence[object]) -> str:
    """Render an argv list as a copy-pasteable command line (Windows quoting)."""
    parts: list[str] = []
    for raw in cmd:
        s = str(raw)
        if s == "" or any(ch in s for ch in ' \t"*?<>|'):
            body = s.replace('"', '\\"')
            # double any trailing backslashes so they don't escape the closing quote
            body = _TRAILING_BACKSLASHES.sub(lambda m: m.group(1) * 2, body)
            parts.append('"' + body + '"')
        else:
            parts.append(s)
    return " ".join(parts)


def run(
    cmd: Sequence[object],
    *,
    env: dict | None = None,
    cwd: object | None = None,
    label: str | None = None,
    dry: bool | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess | None:
    """Run ``cmd``. Returns the CompletedProcess, or ``None`` when skipped by dry-run."""
    argv = [str(c) for c in cmd]
    do_dry = is_dry() if dry is None else dry
    if do_dry:
        plan(label or fmt_cmd(argv))
        return None
    if is_verbose():
        step(fmt_cmd(argv))
        # Tee: echo the merged output live while accumulating it, so error classification
        # (gated/401, convert tails) still works under --verbose.
        proc = subprocess.Popen(
            argv,
            env=env,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        captured: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            captured.append(line)
        returncode = proc.wait(timeout=timeout)
        return subprocess.CompletedProcess(argv, returncode, "".join(captured), "")
    # Decode as UTF-8 with replacement: external tools (ollama, hf) emit UTF-8, but the
    # Windows locale codec (cp1252) would crash the reader thread on bytes it can't map.
    return subprocess.run(
        argv,
        env=env,
        cwd=cwd,
        timeout=timeout,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def output_of(proc: subprocess.CompletedProcess | None) -> str:
    """Combined stdout+stderr of a run (empty for dry runs)."""
    if proc is None:
        return ""
    return (getattr(proc, "stdout", None) or "") + (getattr(proc, "stderr", None) or "")
