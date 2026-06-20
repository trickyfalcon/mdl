"""Subprocess helper that understands ``--dry-run`` and ``--verbose``.

* dry-run: print the planned command (or a friendlier label) and run nothing.
* stream: let the child inherit our terminal so tools that draw progress bars (``hf``/xet,
  llama.cpp) detect a real TTY and render live. Nothing is captured -- ``hf`` suppresses its
  progress bar whenever stdout is a pipe, so capturing and showing progress are mutually
  exclusive. Callers classify failures by return code (plus a metadata probe) instead.
* verbose: stream the child's output live *and* capture it, so callers can still scan it for
  known error signatures (e.g. a gated-repo 401) and raise a friendly error.
* otherwise: capture output silently for the same scanning.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from typing import Sequence

from .console import is_dry, is_verbose, plan, step, warn
from .paths import path_size

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
    stream: bool = False,
    watch_dir: object | None = None,
    stall_timeout: float | None = None,
) -> subprocess.CompletedProcess | None:
    """Run ``cmd``. Returns the CompletedProcess, or ``None`` when skipped by dry-run.

    ``watch_dir`` + ``stall_timeout`` arm a progress watchdog (stream mode only): if the
    total bytes under ``watch_dir`` don't grow for ``stall_timeout`` seconds, the child's
    process tree is killed and a non-zero return is reported -- so a hung transfer (e.g. a
    dead xet connection that never times out on its own) surfaces as a retryable failure.
    """
    argv = [str(c) for c in cmd]
    do_dry = is_dry() if dry is None else dry
    if do_dry:
        plan(label or fmt_cmd(argv))
        return None
    if stream:
        # Inherit our stdout/stderr so the child sees a real terminal and renders its own
        # live progress bar. We capture nothing (returncode only); callers that need to
        # classify a failure do so without the child's text.
        if is_verbose():
            step(fmt_cmd(argv))
        return _run_streamed(argv, env, cwd, timeout, watch_dir, stall_timeout)
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


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill ``proc`` and its descendants. ``hf.exe`` launches a Python child, so a plain
    terminate() would orphan the worker -- on Windows use ``taskkill /T`` to take the tree."""
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
            )
        else:
            proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:
            pass


def _run_streamed(
    argv: list[str],
    env: dict | None,
    cwd: object | None,
    timeout: float | None,
    watch_dir: object | None,
    stall_timeout: float | None,
) -> subprocess.CompletedProcess:
    """Run ``argv`` with inherited stdio (live progress bar), optionally watchdogged."""
    proc = subprocess.Popen(argv, env=env, cwd=cwd)
    if not (watch_dir and stall_timeout and stall_timeout > 0):
        rc = proc.wait(timeout=timeout)
        return subprocess.CompletedProcess(argv, rc, "", "")

    poll = max(1.0, min(15.0, float(stall_timeout)))
    last_size = -1
    last_progress = time.monotonic()
    while True:
        try:
            rc = proc.wait(timeout=poll)
            return subprocess.CompletedProcess(argv, rc, "", "")  # finished on its own
        except subprocess.TimeoutExpired:
            pass
        size = path_size(watch_dir)
        now = time.monotonic()
        if size > last_size:
            last_size, last_progress = size, now
        elif now - last_progress >= stall_timeout:
            warn(f"no download progress for {int(stall_timeout)}s -- aborting to resume ...")
            _kill_tree(proc)
            return subprocess.CompletedProcess(argv, proc.returncode or 1, "", "")


def output_of(proc: subprocess.CompletedProcess | None) -> str:
    """Combined stdout+stderr of a run (empty for dry runs)."""
    if proc is None:
        return ""
    return (getattr(proc, "stdout", None) or "") + (getattr(proc, "stderr", None) or "")
