"""Core condensation logic for the condense feature.

Provides extract_surface_impl, condense_pack_impl, and supporting helpers.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...constants import (
    CondenseDefaults,
    CondenseFileRouting,
    CondenseParsing,
    ConversionFactors,
    IndentationDefaults,
)
from ...core.logging import get_logger
from ...models.condense import LanguageCondenseStats
from .estimator import _collect_files
from .normalizer import normalize_source
from .strip import strip_dead_code

logger = get_logger("condense.service")

# Language-specific surface extraction patterns for ast-grep
_SURFACE_PATTERNS: Dict[str, List[str]] = {
    "typescript": [
        "export function $NAME($$$PARAMS): $RET { $$$ }",
        "export function $NAME($$$PARAMS) { $$$ }",
        "export async function $NAME($$$PARAMS): $RET { $$$ }",
        "export async function $NAME($$$PARAMS) { $$$ }",
        "export class $NAME $$$BODY",
        "export interface $NAME { $$$ }",
        "export type $NAME = $$$",
        "export const $NAME: $TYPE = $$$",
        "export const $NAME = $$$",
        "export default $$$",
    ],
    "javascript": [
        "export function $NAME($$$PARAMS) { $$$ }",
        "export async function $NAME($$$PARAMS) { $$$ }",
        "export class $NAME $$$BODY",
        "export const $NAME = $$$",
        "export default $$$",
        "module.exports = $$$",
    ],
    "python": [
        "def $NAME($$$PARAMS): $$$",
        "async def $NAME($$$PARAMS): $$$",
        "class $NAME($$$BASES): $$$",
        "class $NAME: $$$",
        "@$DECORATOR\ndef $NAME($$$): $$$",
    ],
    "rust": [
        "pub fn $NAME($$$) -> $RET { $$$ }",
        "pub fn $NAME($$$) { $$$ }",
        "pub async fn $NAME($$$) -> $RET { $$$ }",
        "pub struct $NAME { $$$ }",
        "pub trait $NAME { $$$ }",
        "pub enum $NAME { $$$ }",
        "pub type $NAME = $$$;",
    ],
    "go": [
        "func $NAME($$$) $RET { $$$ }",
        "func ($RECV) $NAME($$$) $RET { $$$ }",
        "type $NAME struct { $$$ }",
        "type $NAME interface { $$$ }",
    ],
    "java": [
        "public $RET $NAME($$$) { $$$ }",
        "public class $NAME $$$BODY",
        "public interface $NAME { $$$ }",
        "public enum $NAME { $$$ }",
    ],
}


def _extract_surface_for_file(
    fp: Path,
    language: str,
    include_docstrings: bool,
) -> Tuple[str, int, int] | None:
    """Extract surface for a single file.

    Returns (header, original_len, condensed_len) or None on read failure.
    """
    try:
        source = fp.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    condensed = _extract_file_surface(
        source=source,
        file_path=str(fp),
        language=language,
        include_docstrings=include_docstrings,
    )
    return f"# {fp}\n{condensed}", len(source), len(condensed)


def _compute_reduction_pct(condensed: int, original: int) -> float:
    """Compute reduction percentage, clamped to [0, 100]."""
    if original <= 0:
        return 0.0
    return max(0.0, round((1.0 - condensed / original) * ConversionFactors.PERCENT_MULTIPLIER, 1))


def _accumulate_surface_results(
    files: List[Path],
    language: str,
    include_docstrings: bool,
) -> Tuple[List[str], int, int]:
    """Process files and accumulate surface extraction results.

    Returns (output_parts, total_original, total_condensed).
    """
    output_parts: List[str] = []
    total_original = 0
    total_condensed = 0

    for fp in files:
        result = _extract_surface_for_file(fp, language, include_docstrings)
        if result is None:
            continue
        header, orig_len, cond_len = result
        total_original += orig_len
        total_condensed += cond_len
        output_parts.append(header)

    return output_parts, total_original, total_condensed


def extract_surface_impl(
    path: str,
    language: str,
    include_docstrings: bool = CondenseDefaults.INCLUDE_DOCSTRINGS,
    complexity_guided: bool = False,
    complexity_threshold: int = CondenseDefaults.COMPLEXITY_STRIP_THRESHOLD,
) -> Dict[str, Any]:
    """Extract public API surface (exports, signatures, types) from source files.

    Args:
        path: Directory or file path to analyze.
        language: Programming language (e.g. "python", "typescript").
        include_docstrings: Whether to include docstrings in output.
        complexity_guided: Reserved for future use. When True, will integrate
            with features/complexity to vary extraction depth per function.
            Currently has no effect — all functions use signature+docstring mode.
        complexity_threshold: Reserved for future use alongside complexity_guided.

    Returns:
        Dict with condensed_source (str), files_processed (int),
        condensed_lines (int), and reduction_pct (float).
    """
    root = Path(path)
    if not root.exists():
        return {"error": f"Path does not exist: {path}"}

    files = [root] if root.is_file() else _collect_files(root, language)
    output_parts, total_original, total_condensed = _accumulate_surface_results(
        files,
        language,
        include_docstrings,
    )

    condensed_source = "\n\n".join(output_parts)
    patterns_matched = condensed_source.count("\n")
    reduction_pct = _compute_reduction_pct(total_condensed, total_original)

    logger.info(
        "extract_surface_complete",
        files=len(files),
        patterns_matched=patterns_matched,
        reduction_pct=reduction_pct,
    )

    return {
        "condensed_source": condensed_source,
        "files_processed": len(files),
        "condensed_lines": patterns_matched,
        "reduction_pct": reduction_pct,
    }


def _extract_file_surface(
    source: str,
    file_path: str,
    language: str,
    include_docstrings: bool,
) -> str:
    """Extract the public API surface from a single file's source text.

    Falls back to returning first N lines (signature-only mode) for languages
    without explicit patterns, using a line-based heuristic.
    """
    lines = source.splitlines()
    kept: List[str] = []
    lang = language.lower()

    if lang == "python":
        kept = _extract_python_surface(lines, include_docstrings)
    elif lang in ("typescript", "javascript"):
        kept = _extract_js_ts_surface(lines, include_docstrings)
    else:
        # Generic: keep all lines that look like declarations
        kept = _extract_generic_surface(lines)

    return "\n".join(kept)


_DECL_PREFIXES = ("def ", "async def ", "class ", "@")


def _handle_python_declaration(
    lines: List[str],
    i: int,
    include_docstrings: bool,
    indent: int,
) -> Tuple[List[str], int, int]:
    """Handle a Python declaration line (def/class/@decorator).

    Returns (kept_lines, body_indent, lines_consumed).
    """
    kept = [lines[i].rstrip()]
    consumed = 0
    if include_docstrings:
        doc_lines, advance = _collect_docstring(lines, i + 1)
        kept.extend(doc_lines)
        consumed = advance
    body_indent = indent + IndentationDefaults.SPACES_PER_LEVEL
    return kept, body_indent, consumed


def _extract_python_surface(lines: List[str], include_docstrings: bool) -> List[str]:
    """Extract Python surface: class/def signatures plus optional docstrings."""
    kept: List[str] = []
    i = 0
    body_indent: int | None = None

    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        if stripped.startswith(_DECL_PREFIXES):
            decl_kept, body_indent, consumed = _handle_python_declaration(
                lines,
                i,
                include_docstrings,
                indent,
            )
            kept.extend(decl_kept)
            i += consumed
        elif body_indent is not None and stripped and indent >= body_indent:
            pass  # skip body line
        elif not stripped:
            # Blank lines do not end a body; keep at most one in a row
            if kept and kept[-1]:
                kept.append("")
        else:
            if body_indent is not None and indent < body_indent:
                body_indent = None
            kept.append(line.rstrip())
        i += 1

    return kept


def _collect_docstring(lines: List[str], start: int) -> Tuple[List[str], int]:
    """Collect a docstring starting at `start`.

    Returns (docstring_lines, lines_consumed).
    Returns ([], 0) if no docstring at start.
    """
    if start >= len(lines):
        return [], 0
    first = lines[start].lstrip()
    if not first.startswith(('"""', "'''")):
        return [], 0

    triple = first[:3]
    doc_lines = [lines[start].rstrip()]
    # Single-line docstring
    if doc_lines[0].count(triple) >= 2:
        return doc_lines, 1

    # Multi-line: scan until closing triple
    j = start + 1
    while j < len(lines) and triple not in lines[j]:
        doc_lines.append(lines[j].rstrip())
        j += 1
    if j < len(lines):
        doc_lines.append(lines[j].rstrip())
        return doc_lines, j - start + 1

    return doc_lines, j - start


def _skip_string_literal(line: str, start: int) -> int:
    """Skip past a string literal starting at `start` (quote char position).

    Returns the index of the closing quote (or end of line if unterminated).
    """
    quote = line[start]
    i = start + 1
    n = len(line)
    while i < n:
        if line[i] == "\\" and i + 1 < n:
            i += 2
            continue
        if line[i] == quote:
            return i
        i += 1
    return i


def _count_structural_braces(line: str) -> int:
    """Count net structural braces on a line, skipping strings and comments.

    Returns opens minus closes for structural braces only.
    """
    net = 0
    i = 0
    n = len(line)
    while i < n:
        c = line[i]
        if c == "/" and i + 1 < n and line[i + 1] == "/":
            break
        if c in CondenseParsing.QUOTE_CHARS:
            i = _skip_string_literal(line, i)
        else:
            net += CondenseParsing.BRACE_DELTA.get(c, 0)
        i += 1
    return net


def _extract_js_ts_surface(lines: List[str], include_docstrings: bool) -> List[str]:
    """Extract JS/TS surface: export declarations only.

    Uses character-level brace counting with string/comment awareness.
    Falls back to returning all lines if no exports are found.
    """
    kept: List[str] = []
    brace_depth = 0
    in_export = False
    export_brace_start = 0

    for line in lines:
        if line.strip().startswith("export ") and not in_export:
            in_export = True
            export_brace_start = brace_depth

        delta = _count_structural_braces(line)
        brace_depth += delta

        if not in_export:
            continue
        kept.append(line.rstrip())
        # Ends on the closing brace, or on a statement end once back at the starting depth
        if brace_depth <= export_brace_start and (delta < 0 or line.rstrip().endswith((";", "}"))):
            in_export = False

    return kept if kept else lines


def _extract_generic_surface(lines: List[str]) -> List[str]:
    """Generic surface extraction: keep declaration-looking lines."""
    declaration_keywords = (
        "func ",
        "fn ",
        "def ",
        "class ",
        "struct ",
        "interface ",
        "type ",
        "enum ",
        "pub ",
        "export ",
        "module ",
        "namespace ",
    )
    return [
        line.rstrip()
        for line in lines
        if any(line.lstrip().startswith(kw) for kw in declaration_keywords) or not line.strip()  # preserve blank lines for readability
    ]


def _process_single_file(
    fp: Path,
    root: Path,
    strategy: str,
    file_type_routing: bool,
) -> Optional[Dict[str, Any]]:
    """Process one file through the condense pipeline.

    Returns None if the file should be skipped, otherwise a dict with:
    original_bytes, condensed_bytes, norm_count, removed_lines, header,
    lang, original_lines, condensed_lines.
    """
    try:
        source = fp.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    file_size = len(source.encode("utf-8"))
    if file_size > CondenseDefaults.MAX_FILE_SIZE_BYTES:
        return None

    lang = _detect_language(fp)
    rel_path = fp.relative_to(root) if fp.is_relative_to(root) else fp
    effective_strategy = _route_strategy(rel_path, strategy, file_type_routing)
    if effective_strategy == "exclude":
        return None

    normalized, norm_count = normalize_source(source, lang)
    stripped, removed_lines = strip_dead_code(normalized, lang)
    condensed = _apply_strategy(stripped, lang, effective_strategy)

    rel = fp.relative_to(root) if root.is_dir() else fp
    return {
        "original_bytes": file_size,
        "condensed_bytes": len(condensed.encode("utf-8")),
        "norm_count": norm_count,
        "removed_lines": removed_lines,
        "header": f"// file: {rel}\n{condensed}",
        "lang": lang,
        "original_lines": source.count("\n") + 1,
        "condensed_lines": condensed.count("\n") + 1,
    }


def _update_language_stats(
    per_language: Dict[str, LanguageCondenseStats],
    file_result: Dict[str, Any],
) -> None:
    """Accumulate per-language stats from a single file result."""
    lang = file_result["lang"]
    if lang not in per_language:
        per_language[lang] = LanguageCondenseStats(
            language=lang,
            files_processed=0,
            original_lines=0,
            condensed_lines=0,
            patterns_matched=0,
        )
    stats = per_language[lang]
    stats.files_processed += 1
    stats.original_lines += file_result["original_lines"]
    stats.condensed_lines += file_result["condensed_lines"]
    stats.original_bytes += file_result["original_bytes"]
    stats.condensed_bytes += file_result["condensed_bytes"]


def _serialize_language_stats(per_language: Dict[str, LanguageCondenseStats]) -> Dict[str, Any]:
    """Serialize per-language stats to dicts with reduction_pct."""
    return {
        lang: {
            "files_processed": s.files_processed,
            "original_lines": s.original_lines,
            "condensed_lines": s.condensed_lines,
            "original_bytes": s.original_bytes,
            "condensed_bytes": s.condensed_bytes,
            "reduction_pct": _compute_reduction_pct(s.condensed_bytes, s.original_bytes),
        }
        for lang, s in per_language.items()
    }


def _accumulate_pack_results(
    all_files: List[Path],
    root: Path,
    strategy: str,
    file_type_routing: bool,
) -> Dict[str, Any]:
    """Process all files and accumulate condensation stats.

    Returns dict with output_parts, files_processed, files_skipped,
    total_original_bytes, total_condensed_bytes, normalizations_applied,
    dead_code_removed_lines, per_language.
    """
    output_parts: List[str] = []
    files_processed = 0
    files_skipped = 0
    total_original_bytes = 0
    total_condensed_bytes = 0
    normalizations_applied = 0
    dead_code_removed_lines = 0
    per_language: Dict[str, LanguageCondenseStats] = {}

    for fp in all_files:
        file_result = _process_single_file(fp, root, strategy, file_type_routing)
        if file_result is None:
            files_skipped += 1
            continue

        total_original_bytes += file_result["original_bytes"]
        total_condensed_bytes += file_result["condensed_bytes"]
        normalizations_applied += file_result["norm_count"]
        dead_code_removed_lines += file_result["removed_lines"]
        output_parts.append(file_result["header"])
        files_processed += 1
        _update_language_stats(per_language, file_result)

    return {
        "output_parts": output_parts,
        "files_processed": files_processed,
        "files_skipped": files_skipped,
        "total_original_bytes": total_original_bytes,
        "total_condensed_bytes": total_condensed_bytes,
        "normalizations_applied": normalizations_applied,
        "dead_code_removed_lines": dead_code_removed_lines,
        "per_language": per_language,
    }


def _build_pack_result(acc: Dict[str, Any], strategy: str) -> Dict[str, Any]:
    """Build the final condense_pack result dict from accumulated stats."""
    orig = acc["total_original_bytes"]
    cond = acc["total_condensed_bytes"]
    return {
        "condensed_output": "\n\n".join(acc["output_parts"]),
        "strategy": strategy,
        "files_processed": acc["files_processed"],
        "files_skipped": acc["files_skipped"],
        "reduction_pct": _compute_reduction_pct(cond, orig),
        "original_bytes": orig,
        "condensed_bytes": cond,
        "original_tokens_est": int(orig * CondenseDefaults.AVG_TOKENS_PER_BYTE),
        "condensed_tokens_est": int(cond * CondenseDefaults.AVG_TOKENS_PER_BYTE),
        "normalizations_applied": acc["normalizations_applied"],
        "dead_code_removed_lines": acc["dead_code_removed_lines"],
        "per_language_stats": _serialize_language_stats(acc["per_language"]),
    }


def condense_pack_impl(
    path: str,
    language: str | None = None,
    strategy: str = CondenseDefaults.DEFAULT_STRATEGY,
    file_type_routing: bool = True,
    exclude_patterns: list[str] | None = None,
) -> Dict[str, Any]:
    """Chain normalize -> strip -> extract into a single condensation pipeline.

    Returns:
        Dict with condensed_output (str), strategy, files_processed (int),
        files_skipped (int), and a CondenseResult-compatible stats dict.
    """
    root = Path(path)
    if not root.exists():
        return {"error": f"Path does not exist: {path}"}

    all_files = [root] if root.is_file() else _collect_files(root, language)
    if exclude_patterns:
        exclusion_set = set(exclude_patterns)
        all_files = [fp for fp in all_files if not _path_matches_any(fp, exclusion_set)]

    acc = _accumulate_pack_results(all_files, root, strategy, file_type_routing)

    logger.info(
        "condense_pack_complete",
        strategy=strategy,
        files_processed=acc["files_processed"],
        files_skipped=acc["files_skipped"],
        reduction_pct=_compute_reduction_pct(acc["total_condensed_bytes"], acc["total_original_bytes"]),
    )

    return _build_pack_result(acc, strategy)


def _apply_strategy(source: str, language: str, strategy: str) -> str:
    """Apply the named strategy to a (normalized, stripped) source string.

    Strategies:
    - ai_chat: lossy surface extraction (signatures + docstrings only)
    - ai_analysis: lossless (normalized + dead-code-stripped source)
    - archival: lossless (same as ai_analysis at the per-file level)
    - polyglot: ai_chat for code, ai_analysis for config/text
    """
    if strategy in ("ai_chat", "polyglot"):
        # polyglot: for code languages use ai_chat surface extraction;
        # config/text files fall through to ai_analysis (pass-through).
        if language in ("python", "typescript", "javascript", "rust", "go", "java", "ruby", "php", "swift", "kotlin", "csharp", "cpp", "c"):
            return _extract_file_surface(
                source=source,
                file_path="",
                language=language,
                include_docstrings=True,
            )
    # ai_analysis / archival: return normalized+stripped source unchanged
    return source


def _detect_language(fp: Path) -> str:
    """Map file extension to language string."""
    ext_to_lang = {
        ".py": "python",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".js": "javascript",
        ".jsx": "javascript",
        ".rs": "rust",
        ".go": "go",
        ".java": "java",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".cs": "csharp",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".c": "c",
        ".h": "c",
        ".md": "markdown",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
    }
    return ext_to_lang.get(fp.suffix.lower(), "unknown")


_TEST_DIR_NAMES = frozenset({"test", "tests", "__tests__", "spec"})
_TEST_FILE_STEM = re.compile(r"^test_|_test$|\.(?:test|spec)$")


def _is_test_path(rel_path: Path) -> bool:
    """Test file by convention: in a test directory or named test_x / x_test / x.test / x.spec."""
    in_test_dir = any(part.lower() in _TEST_DIR_NAMES for part in rel_path.parts[:-1])
    return in_test_dir or bool(_TEST_FILE_STEM.search(rel_path.stem.lower()))


def _route_strategy(fp: Path, strategy: str, file_type_routing: bool) -> str:
    """Determine effective strategy for this file."""
    if not file_type_routing:
        return strategy

    suffix = fp.suffix.lower()
    name = fp.name.lower()

    # Binary/image — exclude
    if suffix in CondenseFileRouting.IMAGE_EXTENSIONS:
        return "exclude"

    # Lock files — exclude
    if name in {"package-lock.json", "yarn.lock", "poetry.lock", "uv.lock"}:
        return "exclude"

    # Config files — pass-through (no condensation)
    if suffix in CondenseFileRouting.CONFIG_EXTENSIONS:
        return "archival"

    # Text/docs — light strip only
    if suffix in CondenseFileRouting.TEXT_EXTENSIONS:
        return "archival"

    # Test files — signature extraction
    if _is_test_path(fp):
        return "ai_chat"

    return strategy


def _path_matches_any(fp: Path, patterns: set[str]) -> bool:
    """Check if fp matches any exclusion pattern using Path.match (supports globs)."""
    return any(fp.match(p) for p in patterns)
