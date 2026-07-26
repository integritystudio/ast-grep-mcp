"""Shared source-file discovery helpers (DRY-07)."""

from collections.abc import Iterable, Sequence
from pathlib import Path

from ast_grep_mcp.constants import LANGUAGE_EXTENSIONS


def language_extensions(language: str, fallback: Sequence[str] | None = None) -> list[str]:
    """Dotted source extensions for a language.

    Falls back to ``[f".{language}"]`` unless a custom fallback is given.
    """
    default = list(fallback) if fallback is not None else [f".{language}"]
    return LANGUAGE_EXTENSIONS.get(language, default)


def find_source_files(
    root: str | Path,
    language: str,
    exclude_patterns: Sequence[str] = (),
    *,
    extensions: Sequence[str] | None = None,
    skip_dir_names: Iterable[str] | None = None,
) -> list[Path]:
    """Recursively discover source files under ``root`` by extension.

    Extensions default to the language's known extensions. Files are dropped
    when any path component is in ``skip_dir_names`` or when the path matches
    any glob in ``exclude_patterns`` (``Path.match`` semantics).

    Returns a sorted, de-duplicated list; empty if ``root`` is not a directory.
    """
    folder = Path(root)
    if not folder.is_dir():
        return []
    exts = list(extensions) if extensions is not None else language_extensions(language)
    skip = frozenset(skip_dir_names or ())
    files = {f for ext in exts for f in folder.rglob(f"*{ext}")}
    return [
        f
        for f in sorted(files)
        if not (skip and any(part in skip for part in f.parts))
        and not any(f.match(p) for p in exclude_patterns)
    ]
