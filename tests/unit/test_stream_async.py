"""Unit tests for async streaming in executor.py.

Instead of mocking asyncio process/stream objects, these tests patch
_prepare_stream_command to run a real python subprocess that emits fixture
output. This exercises the production code path end-to-end: real pipes,
real StreamReader chunked reads, real EOF, real termination signals, and
real return codes.
"""

import json
import sys
from unittest.mock import patch

import pytest

from ast_grep_mcp.core.exceptions import AstGrepExecutionError, AstGrepNotFoundError
from ast_grep_mcp.core.executor import (
    async_stream_ast_grep_results,
    stream_ast_grep_results,
)


def fake_ast_grep(stdout: str = "", stderr: str = "", exit_code: int = 0, linger: bool = False):
    """Patch command prep so the executor spawns a real subprocess emitting fixtures.

    linger keeps the child alive after writing output so early-termination
    paths exercise a real SIGTERM instead of racing process exit.
    """
    script = (
        "import sys, time\n"
        f"sys.stdout.write({stdout!r}); sys.stdout.flush()\n"
        f"sys.stderr.write({stderr!r}); sys.stderr.flush()\n"
        + ("time.sleep(30)\n" if linger else "")
        + f"sys.exit({exit_code})\n"
    )
    return patch(
        "ast_grep_mcp.core.executor._prepare_stream_command",
        return_value=[sys.executable, "-c", script],
    )


def json_stream(matches: list) -> str:
    return "".join(json.dumps(m) + "\n" for m in matches)


class TestAsyncStreamAstGrepResults:
    """Test async_stream_ast_grep_results streaming function."""

    @pytest.mark.asyncio
    async def test_yields_parsed_matches(self):
        """Verify async generator yields correctly parsed JSON matches."""
        matches = [
            {"file": "test.py", "line": 1},
            {"file": "test.py", "line": 2},
        ]

        with fake_ast_grep(stdout=json_stream(matches)):
            results = []
            async for match in async_stream_ast_grep_results(
                "run", ["--json=stream", "/test"], max_results=0
            ):
                results.append(match)

        assert len(results) == 2
        assert results[0]["file"] == "test.py"
        assert results[0]["line"] == 1
        assert results[1]["line"] == 2

    @pytest.mark.asyncio
    async def test_early_termination_at_max_results(self):
        """Verify streaming stops at max_results and terminates the process."""
        matches = [
            {"file": "a.py", "line": 1},
            {"file": "b.py", "line": 2},
            {"file": "c.py", "line": 3},
        ]

        with fake_ast_grep(stdout=json_stream(matches), linger=True):
            results = []
            async for match in async_stream_ast_grep_results(
                "run", ["--json=stream", "/test"], max_results=2
            ):
                results.append(match)

        assert len(results) == 2
        assert results[0]["file"] == "a.py"
        assert results[1]["file"] == "b.py"

    @pytest.mark.asyncio
    async def test_stderr_capture_and_error_propagation(self):
        """Verify return code 1 (no matches) is not treated as an error."""
        with fake_ast_grep(exit_code=1):
            results = []
            async for match in async_stream_ast_grep_results("run", ["--json=stream", "/test"]):
                results.append(match)

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_file_not_found_error_conversion(self):
        """Verify FileNotFoundError is converted to AstGrepNotFoundError."""
        with patch(
            "ast_grep_mcp.core.executor._prepare_stream_command",
            return_value=["/nonexistent/ast-grep-binary"],
        ):
            with pytest.raises(AstGrepNotFoundError):
                async for _ in async_stream_ast_grep_results("run", ["--json=stream", "/test"]):
                    pass

    @pytest.mark.asyncio
    async def test_execution_error_on_nonzero_return(self):
        """Verify AstGrepExecutionError is raised on non-acceptable return codes."""
        with fake_ast_grep(stderr="Some error occurred", exit_code=42):
            with pytest.raises(AstGrepExecutionError):
                async for _ in async_stream_ast_grep_results("run", ["--json=stream", "/test"]):
                    pass

    @pytest.mark.asyncio
    async def test_skips_invalid_json_lines(self):
        """Verify invalid JSON lines are skipped without error."""
        stdout = 'valid\n{"file": "a.py"}\ninvalid json\n{"file": "b.py"}\n'

        with fake_ast_grep(stdout=stdout):
            results = []
            async for match in async_stream_ast_grep_results("run", ["--json=stream", "/test"]):
                results.append(match)

        assert len(results) == 2
        assert results[0]["file"] == "a.py"
        assert results[1]["file"] == "b.py"

    @pytest.mark.asyncio
    async def test_line_larger_than_read_chunk(self):
        """Verify a JSON line spanning multiple 64KB read chunks is reassembled."""
        big_match = {"file": "big.py", "payload": "x" * 200_000}

        with fake_ast_grep(stdout=json_stream([big_match])):
            results = []
            async for match in async_stream_ast_grep_results("run", ["--json=stream", "/test"]):
                results.append(match)

        assert len(results) == 1
        assert results[0]["file"] == "big.py"
        assert len(results[0]["payload"]) == 200_000

    @pytest.mark.asyncio
    async def test_final_line_without_trailing_newline(self):
        """Verify a match on the last line without a trailing newline is flushed."""
        stdout = json_stream([{"file": "a.py"}]) + json.dumps({"file": "b.py"})

        with fake_ast_grep(stdout=stdout):
            results = []
            async for match in async_stream_ast_grep_results("run", ["--json=stream", "/test"]):
                results.append(match)

        assert len(results) == 2
        assert results[1]["file"] == "b.py"


class TestStreamAstGrepResultsSyncShim:
    """Test sync shim backward compatibility."""

    def test_sync_shim_delegates_to_async(self):
        """Verify sync function delegates to async implementation."""
        matches = [
            {"file": "test.py", "line": 1},
            {"file": "test.py", "line": 2},
        ]

        with fake_ast_grep(stdout=json_stream(matches)):
            results = list(stream_ast_grep_results("run", ["--json=stream", "/test"]))

        assert len(results) == 2
        assert results[0]["file"] == "test.py"

    @pytest.mark.asyncio
    async def test_sync_shim_works_inside_running_event_loop(self):
        """FastMCP calls sync tools on its event-loop thread; asyncio.run() must not be hit there."""
        matches = [{"file": "test.py", "line": 1}]

        with fake_ast_grep(stdout=json_stream(matches)):
            results = list(stream_ast_grep_results("run", ["--json=stream", "/test"]))

        assert results == matches

    def test_sync_shim_preserves_max_results(self):
        """Verify max_results is respected through sync shim."""
        matches = [{"file": f"test{i}.py"} for i in range(5)]

        with fake_ast_grep(stdout=json_stream(matches), linger=True):
            results = list(
                stream_ast_grep_results("run", ["--json=stream", "/test"], max_results=3)
            )

        assert len(results) == 3
