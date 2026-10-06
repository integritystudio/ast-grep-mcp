"""Tests that atomic line writes keep the original file's permissions."""

import stat

from ast_grep_mcp.utils.text import write_file_lines

SCRIPT_MODE = 0o755


def test_rewrite_preserves_mode(tmp_path) -> None:
    script = tmp_path / "run.sh"
    script.write_text("echo old\n")
    script.chmod(SCRIPT_MODE)

    write_file_lines(script, ["echo new\n"])

    assert script.read_text() == "echo new\n"
    assert stat.S_IMODE(script.stat().st_mode) == SCRIPT_MODE


def test_new_file_is_created(tmp_path) -> None:
    target = tmp_path / "new.py"
    write_file_lines(target, ["x = 1\n"])
    assert target.read_text() == "x = 1\n"
