"""Shared source-file discovery helpers (DRY-07)."""

from collections.abc import Iterable, Sequence
from pathlib import Path, PurePath

from ast_grep_mcp.constants import LANGUAGE_EXTENSIONS


def language_extensions(language: str, fallback: Sequence[str] | None = None) -> list[str]:
    """Dotted source extensions for a language.

    Falls back to ``[f".{language}"]`` unless a custom fallback is given.
    """
    default = list(fallback) if fallback is not None else [f".{language}"]
    return LANGUAGE_EXTENSIONS.get(language, default)


def matches_glob(path: str | PurePath, pattern: str, root: str | PurePath | None = None) -> bool:
    """Glob-match ``path`` with real ``**`` semantics (``PurePath.full_match``).

    The path is made relative to ``root`` when it lies under it, so directories
    above the project (e.g. ``~/code/build/proj``) never trigger a match. A
    pattern without ``/`` also matches the file's basename, gitignore-style.
    """
    rel = PurePath(path)
    if root is not None and rel.is_relative_to(root):
        rel = rel.relative_to(root)
    pattern = pattern.lstrip("/")
    if rel.full_match(pattern):
        return True
    return "/" not in pattern and PurePath(rel.name).full_match(pattern)


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
        if not (skip and any(part in skip for part in f.relative_to(folder).parts))
        and not any(f.match(p) for p in exclude_patterns)
    ]
