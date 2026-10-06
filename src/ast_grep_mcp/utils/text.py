"""Text processing and string manipulation utilities.

This module provides utilities for normalizing code, calculating similarity,
and other text processing operations.
"""

import difflib
import os
import shutil
import tempfile
from typing import Union

__all__ = [
    "normalize_code",
    "calculate_similarity",
    "clean_template_whitespace",
    "_clean_template_whitespace",
    "indent_lines",
    "read_file_lines",
    "write_file_lines",
    "FilePath",
]


def normalize_code(code: str, language: str | None = None) -> str:
    """Normalize code for comparison by removing whitespace and comments.

    Args:
        code: Code string to normalize
        language: Optional language hint (for future language-specific normalization)

    Returns:
        Normalized code string
    """
    lines = []
    for line in code.split("\n"):
        # Remove leading/trailing whitespace
        stripped = line.strip()
        # Skip empty lines and simple comments
        if stripped and not stripped.startswith("#") and not stripped.startswith("//"):
            lines.append(stripped)
    return "\n".join(lines)


def calculate_similarity(code1: str, code2: str, language: str | None = None) -> float:
    """Calculate similarity ratio between two code snippets.

    Uses SequenceMatcher for structural similarity comparison.

    Args:
        code1: First code snippet
        code2: Second code snippet
        language: Optional language hint (for future language-specific comparison)

    Returns:
        Similarity ratio between 0 and 1
    """
    if not code1 or not code2:
        return 0.0

    # Normalize code for comparison
    norm1 = normalize_code(code1, language)
    norm2 = normalize_code(code2, language)

    # Use difflib SequenceMatcher for similarity
    matcher = difflib.SequenceMatcher(None, norm1, norm2)
    return matcher.ratio()


def _trim_surrounding_blanks(lines: list[str]) -> list[str]:
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def clean_template_whitespace(template: str) -> str:
    """Clean and normalize whitespace in code templates.

    Removes excessive blank lines, normalizes indentation, and trims
    trailing whitespace while preserving code structure.

    Args:
        template: Code template string to clean

    Returns:
        Cleaned template with normalized whitespace
    """
    if not template:
        return ""
    lines = [line.rstrip() for line in template.split("\n")]
    lines = _trim_surrounding_blanks(lines)
    return "\n".join(_collapse_blank_lines(lines))


def _collapse_blank_lines(lines: list[str]) -> list[str]:
    result: list[str] = []
    prev_blank = False
    for line in lines:
        is_blank = not line.strip()
        if is_blank and not prev_blank:
            result.append("")
        elif not is_blank:
            result.append(line)
        prev_blank = is_blank
    return result


def indent_lines(text: str, prefix: str = "    ") -> list[str]:
    """Indent non-empty lines with the given prefix, leave blank lines empty.

    Args:
        text: Text to indent
        prefix: Indentation prefix (default 4 spaces)

    Returns:
        List of indented lines
    """
    return [f"{prefix}{line}" if line.strip() else "" for line in text.split("\n")]


FilePath = Union[str, "os.PathLike[str]"]


def read_file_lines(file_path: FilePath, errors: str | None = None) -> list[str]:
    """Read a file and return its lines (including newlines).

    Args:
        file_path: Path to the file (str or PathLike)
        errors: Decoding error handling passed to open() (e.g. "ignore");
            None uses the default strict handling

    Returns:
        List of lines with trailing newlines preserved

    Raises:
        OSError: If the file cannot be read, with the path in the message
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors=errors) as f:
            return f.readlines()
    except OSError as e:
        raise OSError(f"Failed to read {file_path}: {e}") from e


def write_file_lines(file_path: FilePath, lines: list[str]) -> None:
    """Write lines to a file atomically via temp-file-then-rename.

    Args:
        file_path: Path to the file (str or PathLike)
        lines: Lines to write (should include trailing newlines)

    Raises:
        OSError: If the file cannot be written, with the path in the message
    """
    target = str(file_path)
    dir_name = os.path.dirname(target) or "."
    try:
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.writelines(lines)
            # mkstemp creates 0600; keep the original file's mode (e.g. exec bit, group read)
            if os.path.exists(target):
                shutil.copymode(target, tmp_path)
            os.replace(tmp_path, target)
        except BaseException:
            os.unlink(tmp_path)
            raise
    except OSError as e:
        raise OSError(f"Failed to write {file_path}: {e}") from e


# Alias for backward compatibility
_clean_template_whitespace = clean_template_whitespace
