"""Tests for command-line rendering (Windows quoting)."""

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
