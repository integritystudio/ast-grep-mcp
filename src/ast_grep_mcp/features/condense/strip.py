"""Dead code stripping for the condense pipeline.

Removes console.log/print debug statements and debugger calls from source.
Empty-block and commented-out-code removal are not yet implemented.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from ...constants import CondenseDefaults
from ...core.logging import get_logger

logger = get_logger("condense.strip")

# JS/TS debug statement patterns
_JS_CONSOLE_LOG = re.compile(r"^\s*console\.(log|debug|warn|error|info|trace)\s*\(.*\);\s*$")
_JS_DEBUGGER = re.compile(r"^\s*debugger;\s*$")

# Python debug statement patterns
_PY_PRINT = re.compile(r"^\s*print\s*\(.*\)\s*$")
_PY_BREAKPOINT = re.compile(r"^\s*(?:breakpoint|pdb\.set_trace)\s*\(\s*\)\s*$")
_PY_IMPORT_PDB = re.compile(r"^\s*import\s+pdb\s*$")

# Note: empty-block and commented-out-code patterns are not yet applied.

# A removed statement that was a block's whole body is replaced by a no-op so the
# surrounding control flow survives: an empty Python block is a syntax error, and a
# braceless JS `if (x)` would otherwise capture the next statement as its body.
_PY_NOOP = "pass"
_JS_NOOP = ";"
_PY_TRAILING_COMMENT = re.compile(r"\s+#.*$")
_JS_BRACELESS_HEADER = re.compile(r"^\s*(?:(?:\}\s*)?else\s+if\s*\(.*\)|if\s*\(.*\)|for\s*\(.*\)|while\s*\(.*\)|(?:\}\s*)?else|do)\s*$")


def strip_dead_code(source: str, language: str) -> Tuple[str, int]:
    """Remove debug statements, empty blocks, and commented-out code.

    Args:
        source: Source code text (should be normalized first).
        language: Language identifier.

    Returns:
        Tuple of (stripped_source, lines_removed_count).
    """
    lines = source.splitlines()

    if language in ("typescript", "javascript"):
        kept, removed = _strip_js_ts(lines)
    elif language == "python":
        kept, removed = _strip_python(lines)
    else:
        kept, removed = lines, 0

    return "\n".join(kept), removed


def _strip_js_ts(lines: List[str]) -> Tuple[List[str], int]:
    """Strip JS/TS debug statements and empty blocks."""
    kept: List[str] = []
    removed = 0

    for line in lines:
        is_console = CondenseDefaults.STRIP_CONSOLE_LOG and _JS_CONSOLE_LOG.match(line)
        is_debugger = CondenseDefaults.STRIP_DEBUG_STATEMENTS and _JS_DEBUGGER.match(line)
        if not (is_console or is_debugger):
            kept.append(line)
        elif _JS_BRACELESS_HEADER.match(_last_significant(kept, "//") or ""):
            kept.append(_indent_of(line) + _JS_NOOP)
        else:
            removed += 1

    return kept, removed


def _indent_of(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _last_significant(kept: List[str], comment_prefix: str) -> str | None:
    """Last kept line that is neither blank nor a whole-line comment."""
    for line in reversed(kept):
        stripped = line.strip()
        if stripped and not stripped.startswith(comment_prefix):
            return line
    return None


def _is_debug_python(line: str) -> bool:
    return any(p.match(line) for p in _PY_DEBUG_PATTERNS)


def _next_kept_indent(lines: List[str], start: int) -> int:
    """Indent width of the next line that will survive stripping; -1 at end of input."""
    for line in lines[start:]:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not _is_debug_python(line):
            return len(_indent_of(line))
    return -1


def _empties_python_block(kept: List[str], lines: List[str], index: int) -> bool:
    """True if dropping lines[index] leaves the block opened by the last kept line without a body."""
    opener = _last_significant(kept, "#")
    if opener is None or not _PY_TRAILING_COMMENT.sub("", opener).rstrip().endswith(":"):
        return False
    opener_indent = len(_indent_of(opener))
    return len(_indent_of(lines[index])) > opener_indent >= _next_kept_indent(lines, index + 1)


_PY_DEBUG_PATTERNS = [_PY_PRINT, _PY_BREAKPOINT, _PY_IMPORT_PDB]


def _strip_python(lines: List[str]) -> Tuple[List[str], int]:
    """Strip Python debug statements and import pdb."""
    if not CondenseDefaults.STRIP_DEBUG_STATEMENTS:
        return lines, 0

    kept: List[str] = []
    removed = 0
    for i, line in enumerate(lines):
        if not _is_debug_python(line):
            kept.append(line)
        elif _empties_python_block(kept, lines, i):
            kept.append(_indent_of(line) + _PY_NOOP)
        else:
            removed += 1
    return kept, removed
