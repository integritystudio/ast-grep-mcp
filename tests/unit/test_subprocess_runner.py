"""Tests for the shared subprocess-with-timeout runner (DRY-08)."""

import sys
from pathlib import Path

from ast_grep_mcp.utils.subprocess_runner import (
    TOOL_NOT_FOUND,
    TOOL_TIMEOUT,
    run_tool,
)


class TestRunTool:
    def test_successful_run_captures_stdout(self) -> None:
        result = run_tool([sys.executable, "-c", "print('hello')"], timeout_seconds=30)

        assert result.ok
        assert result.returncode == 0
        assert result.stdout.strip() == "hello"
        assert result.error is None

    def test_nonzero_exit_is_not_ok_but_completed(self) -> None:
        result = run_tool([sys.executable, "-c", "import sys; sys.exit(3)"], timeout_seconds=30)

        assert not result.ok
        assert result.returncode == 3
        assert result.error is None

    def test_stderr_is_captured(self) -> None:
        result = run_tool(
            [sys.executable, "-c", "import sys; sys.stderr.write('boom')"], timeout_seconds=30
        )

        assert result.stderr == "boom"

    def test_missing_binary_reports_not_found(self) -> None:
        result = run_tool(["definitely-not-a-real-binary-dry08"], timeout_seconds=5)

        assert result.error == TOOL_NOT_FOUND
        assert not result.ok
        assert result.returncode is None

    def test_timeout_reports_timeout(self) -> None:
        result = run_tool(
            [sys.executable, "-c", "import time; time.sleep(10)"], timeout_seconds=0.2
        )

        assert result.error == TOOL_TIMEOUT
        assert not result.ok

    def test_cwd_is_honored(self, tmp_path: Path) -> None:
        result = run_tool(
            [sys.executable, "-c", "import os; print(os.getcwd())"],
            timeout_seconds=30,
            cwd=tmp_path,
        )

        assert result.ok
        assert Path(result.stdout.strip()) == tmp_path
