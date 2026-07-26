"""Shared subprocess-with-timeout runner for external tools (DRY-08)."""

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

TOOL_NOT_FOUND = "not_found"
TOOL_TIMEOUT = "timeout"
TOOL_FAILED = "failed"


@dataclass(frozen=True)
class ToolRunResult:
    """Outcome of run_tool: a completed process, or why it never completed."""

    returncode: int | None
    stdout: str
    stderr: str
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True when the tool ran to completion and exited 0."""
        return self.error is None and self.returncode == 0


def run_tool(
    cmd: Sequence[str],
    timeout_seconds: float,
    cwd: str | Path | None = None,
) -> ToolRunResult:
    """Run an external tool with captured text output and a timeout.

    Never raises for the common failure modes of optional tooling: a missing
    or unrunnable binary reports ``error=TOOL_NOT_FOUND``, a timeout
    ``error=TOOL_TIMEOUT``, and any other subprocess-level failure
    ``error=TOOL_FAILED``.
    """
    try:
        result = subprocess.run(
            list(cmd), cwd=cwd, capture_output=True, text=True, timeout=timeout_seconds
        )
    except subprocess.TimeoutExpired:
        return ToolRunResult(None, "", "", error=TOOL_TIMEOUT)
    except OSError:
        return ToolRunResult(None, "", "", error=TOOL_NOT_FOUND)
    except subprocess.SubprocessError:
        return ToolRunResult(None, "", "", error=TOOL_FAILED)
    return ToolRunResult(result.returncode, result.stdout or "", result.stderr or "")
