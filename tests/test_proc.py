"""Tests for command-line rendering (Windows quoting) and the stall watchdog."""

import sys

from mdl import proc
from mdl.proc import fmt_cmd


def test_fmt_cmd_plain_args_unquoted():
    assert fmt_cmd(["hf", "download", "Qwen/Qwen3-32B"]) == "hf download Qwen/Qwen3-32B"


def test_fmt_cmd_quotes_spaces():
    assert fmt_cmd(["hf", "download", "--local-dir", r"D:\a b\c"]) == 'hf download --local-dir "D:\\a b\\c"'


def test_fmt_cmd_quotes_glob_include_pattern():
    assert fmt_cmd(["hf", "--include", "*Q4_K_M*"]) == 'hf --include "*Q4_K_M*"'


def test_fmt_cmd_doubles_trailing_backslash():
    # an argument that needs quoting AND ends in a backslash must have it doubled, else the
    # backslash would escape the closing quote and corrupt the command line.
    assert fmt_cmd(["x", "D:\\dir with space\\"]) == 'x "D:\\dir with space\\\\"'


# -- stall watchdog -----------------------------------------------------------------------
def test_watchdog_kills_stalled_process(tmp_path):
    # a child that makes no on-disk progress under watch_dir must be aborted (non-zero rc)
    cmd = [sys.executable, "-c", "import time; time.sleep(60)"]
    result = proc.run(cmd, stream=True, watch_dir=tmp_path, stall_timeout=2)
    assert result is not None and result.returncode != 0


def test_watchdog_lets_progressing_process_finish(tmp_path):
    # bytes grow under watch_dir each interval -> watchdog must NOT kill; process exits 0
    target = tmp_path / "f.bin"
    script = (
        "import time,sys\n"
        "for _ in range(6):\n"
        "    open(sys.argv[1],'ab').write(b'x'*4096)\n"
        "    time.sleep(0.5)\n"
    )
    cmd = [sys.executable, "-c", script, str(target)]
    result = proc.run(cmd, stream=True, watch_dir=tmp_path, stall_timeout=2)
    assert result is not None and result.returncode == 0


def test_stream_without_watchdog_runs_normally():
    result = proc.run([sys.executable, "-c", "print('hi')"], stream=True)
    assert result is not None and result.returncode == 0
