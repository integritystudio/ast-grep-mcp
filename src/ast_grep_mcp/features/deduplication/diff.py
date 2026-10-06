"""Diff utilities for code comparison and visualization.

This module provides utilities for building diff trees, formatting diffs,
and generating various diff representations for duplicate code analysis.
"""

import difflib
import re
from pathlib import Path
from typing import Any, Callable

from ...constants import UnifiedDiffRegexGroups


class _DiffCounts:
    """Accumulator for diff change counts."""

    __slots__ = ("additions", "deletions", "modifications")

    def __init__(self) -> None:
        self.additions = 0
        self.deletions = 0
        self.modifications = 0

    def to_summary(self) -> dict[str, int]:
        return {
            "additions": self.additions,
            "deletions": self.deletions,
            "modifications": self.modifications,
            "total_changes": self.additions + self.deletions + self.modifications,
        }


def _collect_run(diff_lines: list[str], start: int, prefix: str) -> tuple[list[str], int]:
    """Collect a contiguous run of lines starting with ``prefix``; return contents and next index."""
    run: list[str] = []
    i = start
    while i < len(diff_lines) and diff_lines[i].startswith(prefix):
        run.append(diff_lines[i][1:].rstrip("\n"))
        i += 1
    return run, i


def _record_paired_changes(
    counts: _DiffCounts,
    changes: list[dict[str, Any]],
    del_lines: list[str],
    add_lines: list[str],
) -> None:
    """Pair deletions with additions 1-to-1 as modifications; record the excess as pure changes."""
    paired = min(len(del_lines), len(add_lines))
    for old, new in zip(del_lines[:paired], add_lines[:paired]):
        counts.modifications += 1
        changes.append({"type": "modification", "old": old, "new": new})

    for old in del_lines[paired:]:
        counts.deletions += 1
        changes.append({"type": "deletion", "content": old})

    for new in add_lines[paired:]:
        counts.additions += 1
        changes.append({"type": "addition", "content": new})


def _parse_unified_diff_lines(diff_lines: list[str]) -> tuple[_DiffCounts, list[dict[str, Any]]]:
    """Parse unified diff lines into counts and a changes list.

    BUGL-03: unified diffs emit all ``-`` lines then all ``+`` lines within a
    hunk, so a 2-line replacement looks like ``[-a, -b, +c, +d]`` — the naïve
    approach of pairing each ``-`` with the immediately-following ``+`` missed
    the multi-line case and miscounted modifications/deletions/additions.

    This implementation collects a complete run of deletions, then the
    following run of additions, and pairs them 1-to-1 as modifications, with
    any excess lines counted as pure deletions or additions.
    """
    counts = _DiffCounts()
    changes: list[dict[str, Any]] = []
    i = 0

    while i < len(diff_lines):
        line = diff_lines[i]

        if line.startswith(("---", "+++", "@@")):
            i += 1
            continue

        if line.startswith("-"):
            del_lines, i = _collect_run(diff_lines, i, "-")
            add_lines, i = _collect_run(diff_lines, i, "+")
            _record_paired_changes(counts, changes, del_lines, add_lines)
        elif line.startswith("+"):
            counts.additions += 1
            changes.append({"type": "addition", "content": line[1:].rstrip("\n")})
            i += 1
        else:
            i += 1

    return counts, changes


def _build_nested_structure(changes: list[dict[str, Any]], language: str | None) -> dict[str, Any]:
    """Wrap changes into a nested diff root node."""
    return {
        "root": {
            "type": "diff_root",
            "language": language,
            "children": [{"type": c["type"], "data": c} for c in changes],
        }
    }


_EMPTY_NESTED_RESULT: dict[str, Any] = {
    "nested_diff": {},
    "summary": {"additions": 0, "deletions": 0, "modifications": 0},
    "changes": [],
}


def build_nested_diff_tree(code1: str, code2: str, language: str | None = None) -> dict[str, Any]:
    """Build a nested diff tree from two code snippets.

    Creates a hierarchical representation of differences between two code blocks,
    organizing changes by type (additions, deletions, modifications).

    Args:
        code1: First code snippet (original)
        code2: Second code snippet (modified)
        language: Optional language hint for language-specific parsing

    Returns:
        Nested dictionary containing:
        - nested_diff: The hierarchical diff structure
        - summary: Statistics about the changes
        - changes: List of individual changes with context
    """
    if not code1 and not code2:
        return dict(_EMPTY_NESTED_RESULT)

    lines1 = code1.splitlines(keepends=True) if code1 else []
    lines2 = code2.splitlines(keepends=True) if code2 else []

    diff_lines = list(difflib.unified_diff(lines1, lines2, lineterm=""))
    counts, changes = _parse_unified_diff_lines(diff_lines)

    return {
        "nested_diff": _build_nested_structure(changes, language),
        "summary": counts.to_summary(),
        "changes": changes,
    }


def _append_lines_as_ops(diff_ops: list[dict[str, Any]], lines: list[str], op_type: str) -> None:
    """Append lines to diff_ops with the specified operation type."""
    for line in lines:
        diff_ops.append({"type": op_type, "content": line})


def _process_opcode(
    diff_ops: list[dict[str, Any]],
    tag: str,
    lines1: list[str],
    lines2: list[str],
    i1: int,
    i2: int,
    j1: int,
    j2: int,
) -> None:
    """Process a single diff opcode and append operations to diff_ops."""
    opcode_handlers: dict[str, Callable[[], None]] = {
        "equal": lambda: _append_lines_as_ops(diff_ops, lines1[i1:i2], "equal"),
        "delete": lambda: _append_lines_as_ops(diff_ops, lines1[i1:i2], "delete"),
        "insert": lambda: _append_lines_as_ops(diff_ops, lines2[j1:j2], "insert"),
    }

    handler = opcode_handlers.get(tag)
    if handler:
        handler()
    elif tag == "replace":
        _append_lines_as_ops(diff_ops, lines1[i1:i2], "delete")
        _append_lines_as_ops(diff_ops, lines2[j1:j2], "insert")


def _count_changes(diff_ops: list[dict[str, Any]]) -> int:
    """Count non-equal diff operations."""
    return sum(1 for op in diff_ops if op["type"] != "equal")


def build_diff_tree(code1: str, code2: str, language: str | None = None) -> dict[str, Any]:
    """Build a diff tree from two code snippets.

    Creates a simplified diff representation showing changes between two code blocks.

    Args:
        code1: First code snippet (original)
        code2: Second code snippet (modified)
        language: Optional language hint

    Returns:
        Dictionary containing diff info: diff ops, language, change summary.
    """
    if not code1 and not code2:
        return {"diff": [], "language": language, "summary": {"changes": 0}}

    lines1 = (code1 or "").splitlines()
    lines2 = (code2 or "").splitlines()

    matcher = difflib.SequenceMatcher(None, lines1, lines2)
    diff_ops: list[dict[str, Any]] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        _process_opcode(diff_ops, tag, lines1, lines2, i1, i2, j1, j2)

    return {
        "diff": diff_ops,
        "language": language,
        "summary": {"changes": _count_changes(diff_ops)},
    }


def _format_diff_alignment(alignment: dict[str, Any], lines: list[str]) -> None:
    """Handle diff-type alignment entries (old/new pair)."""
    old_val = alignment.get("old", "")
    new_val = alignment.get("new", "")
    if old_val:
        lines.append(f"- {old_val}")
    if new_val:
        lines.append(f"+ {new_val}")


def _format_alignment_entry(alignment: dict[str, Any], lines: list[str]) -> None:
    """Format a single alignment entry and append to lines."""
    align_type = alignment.get("type", "unknown")
    if align_type == "match":
        lines.append(f"  {alignment.get('value', '')}")
    elif align_type == "diff":
        _format_diff_alignment(alignment, lines)
    elif align_type == "delete":
        lines.append(f"- {alignment.get('value', alignment.get('content', ''))}")
    elif align_type == "insert":
        lines.append(f"+ {alignment.get('value', alignment.get('content', ''))}")


def _format_alignments(alignments: list[dict[str, Any]]) -> str:
    """Format alignments list into diff string."""
    lines: list[str] = []
    for alignment in alignments:
        _format_alignment_entry(alignment, lines)
    return "\n".join(lines)


def _format_diff_op(op: dict[str, Any]) -> str:
    """Format a single diff operation."""
    op_type = op.get("type", "equal")
    content = op.get("content", "")
    prefix_map = {"equal": "  ", "delete": "- ", "insert": "+ "}
    prefix = prefix_map.get(op_type, "  ")
    return f"{prefix}{content}"


def _format_diff_ops(diff_ops: list[dict[str, Any]]) -> str:
    """Format diff operations list into string."""
    return "\n".join(_format_diff_op(op) for op in diff_ops)


def _format_change_entry(change: dict[str, Any], lines: list[str]) -> None:
    """Format a single change entry and append to lines."""
    change_type = change.get("type", "")
    if change_type == "modification":
        lines.append(f"- {change.get('old', '')}")
        lines.append(f"+ {change.get('new', '')}")
    elif change_type == "deletion":
        lines.append(f"- {change.get('content', '')}")
    elif change_type == "addition":
        lines.append(f"+ {change.get('content', '')}")


def _format_changes(changes: list[dict[str, Any]]) -> str:
    """Format changes list into diff string."""
    lines: list[str] = []
    for change in changes:
        _format_change_entry(change, lines)
    return "\n".join(lines)


def format_alignment_diff(diff_data: dict[str, Any]) -> str | dict[str, Any]:
    """Format alignment diff for display.

    Converts alignment diff data into a human-readable format.

    Args:
        diff_data: Dictionary containing alignment information with optional keys:
            - alignments: List of alignment entries (type, value, old, new)
            - diff: List of diff operations
            - changes: List of change entries

    Returns:
        Formatted string representation of the diff, or the original dict if
        no recognized format is found.
    """
    if not diff_data:
        return ""

    if "alignments" in diff_data:
        return _format_alignments(diff_data["alignments"])

    if "diff" in diff_data and isinstance(diff_data["diff"], list):
        return _format_diff_ops(diff_data["diff"])

    if "changes" in diff_data and isinstance(diff_data["changes"], list):
        return _format_changes(diff_data["changes"])

    return diff_data


_HUNK_HEADER_PATTERN = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")


def _parse_file_header(line: str, prefix: str) -> str | None:
    """Parse a file header line and extract the filename."""
    if not line.startswith(prefix):
        return None
    filename = line[len(prefix) :].split("\t")[0]
    # Remove git prefixes
    if filename.startswith("a/") or filename.startswith("b/"):
        filename = filename[2:]
    return filename


def _parse_hunk_header(line: str) -> dict[str, Any] | None:
    """Parse a hunk header line and return hunk dict."""
    match = _HUNK_HEADER_PATTERN.match(line)
    if not match:
        return None
    return {
        "old_start": int(match.group(UnifiedDiffRegexGroups.OLD_START)),
        "old_count": int(match.group(UnifiedDiffRegexGroups.OLD_COUNT)) if match.group(UnifiedDiffRegexGroups.OLD_COUNT) else 1,
        "new_start": int(match.group(UnifiedDiffRegexGroups.NEW_START)),
        "new_count": int(match.group(UnifiedDiffRegexGroups.NEW_COUNT)) if match.group(UnifiedDiffRegexGroups.NEW_COUNT) else 1,
        "context": match.group(UnifiedDiffRegexGroups.CONTEXT).strip(),
        "changes": [],
    }


def _parse_diff_line(line: str) -> dict[str, str]:
    """Parse a diff content line and return change dict."""
    line_type_map = {"-": "deletion", "+": "addition", " ": "context"}
    first_char = line[0] if line else ""
    line_type = line_type_map.get(first_char, "other")
    content = line[1:] if first_char in line_type_map else line
    return {"type": line_type, "content": content}


class _DiffPreviewState:
    """Accumulator for unified diff parsing state."""

    __slots__ = ("hunks", "current_hunk", "old_file", "new_file")

    def __init__(self) -> None:
        self.hunks: list[dict[str, Any]] = []
        self.current_hunk: dict[str, Any] | None = None
        self.old_file: str | None = None
        self.new_file: str | None = None

    def flush_hunk(self) -> None:
        """Append current_hunk to hunks and reset."""
        if self.current_hunk:
            self.hunks.append(self.current_hunk)
            self.current_hunk = None


def _process_preview_line(state: _DiffPreviewState, line: str) -> None:
    """Process a single line of unified diff text, updating state."""
    if line.startswith("--- "):
        state.old_file = _parse_file_header(line, "--- ")
        return
    if line.startswith("+++ "):
        state.new_file = _parse_file_header(line, "+++ ")
        return
    if line.startswith("@@"):
        new_hunk = _parse_hunk_header(line)
        if new_hunk:
            state.flush_hunk()
            state.current_hunk = new_hunk
        return
    if state.current_hunk is not None:
        state.current_hunk["changes"].append(_parse_diff_line(line))


def diff_preview_to_dict(diff_text: str) -> dict[str, Any]:
    """Convert unified diff text to a dictionary representation.

    Parses unified diff format and extracts structured information.

    Args:
        diff_text: Unified diff text string

    Returns:
        Dictionary containing:
        - hunks: List of parsed hunk dictionaries with line changes
        - files: Tuple of (old_file, new_file) names if present
        - raw: Original diff text
    """
    if not diff_text or not diff_text.strip():
        return {"hunks": [], "files": (None, None), "raw": diff_text}

    state = _DiffPreviewState()
    for line in diff_text.strip().split("\n"):
        _process_preview_line(state, line)
    state.flush_hunk()

    return {
        "hunks": state.hunks,
        "files": (state.old_file, state.new_file),
        "raw": diff_text,
    }


def _ensure_trailing_newline(lines: list[str]) -> None:
    """Ensure the last line ends with a newline, in-place."""
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"


def generate_file_diff(old_content: str, new_content: str, filename: str) -> str:
    """Generate unified diff for a single file.

    Args:
        old_content: Original file content
        new_content: Modified file content
        filename: Name of the file for diff headers

    Returns:
        Unified diff string
    """
    old_lines = old_content.splitlines(keepends=True) if old_content else []
    new_lines = new_content.splitlines(keepends=True) if new_content else []
    return "".join(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
        )
    )


def generate_multi_file_diff(changes: list[dict[str, Any]]) -> str:
    """Generate unified diff for multiple files.

    Args:
        changes: List of change dictionaries, each containing:
            - file: Filename
            - old_content: Original content
            - new_content: Modified content

    Returns:
        Combined unified diff string for all files
    """
    diffs: list[str] = []
    for change in changes:
        filename = change.get("file", "unknown")
        old_content = change.get("old_content", "")
        new_content = change.get("new_content", "")
        diff = generate_file_diff(old_content, new_content, filename)
        if diff:
            diffs.append(diff)
    return "\n".join(diffs)


def generate_diff_from_file_paths(old_path: str, new_path: str) -> str:
    """Generate unified diff between two files by their paths.

    Args:
        old_path: Path to the original file
        new_path: Path to the modified file

    Returns:
        Unified diff string, or error message if files cannot be read
    """
    try:
        old_file = Path(old_path)
        new_file = Path(new_path)

        if not old_file.exists():
            return f"Error: File not found: {old_path}"
        if not new_file.exists():
            return f"Error: File not found: {new_path}"

        old_content = old_file.read_text(encoding="utf-8")
        new_content = new_file.read_text(encoding="utf-8")

        # Use the new file's name for the diff header
        filename = new_file.name
        return generate_file_diff(old_content, new_content, filename)

    except (OSError, UnicodeDecodeError) as e:
        return f"Error reading files: {e}"
