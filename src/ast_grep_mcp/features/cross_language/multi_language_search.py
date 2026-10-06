"""Multi-language search implementation.

This module provides functionality to search across multiple programming
languages simultaneously using semantic patterns.
"""

import json as json_module
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ast_grep_mcp.constants import LANGUAGE_EXTENSIONS, ConversionFactors, CrossLanguageDefaults, SubprocessDefaults
from ast_grep_mcp.core.executor import run_ast_grep
from ast_grep_mcp.core.logging import get_logger
from ast_grep_mcp.models.cross_language import (
    SUPPORTED_LANGUAGES,
    MultiLanguageMatch,
    MultiLanguageSearchResult,
)
from ast_grep_mcp.utils.futures import WaitTimeoutError, map_with_per_item_timeout

logger = get_logger(__name__)

# Semantic pattern to ast-grep pattern mapping
SEMANTIC_TO_AST_GREP: Dict[str, Dict[str, str]] = {
    "function": {
        "python": "def $NAME($$$PARAMS): $$$BODY",
        "typescript": "function $NAME($$$PARAMS) { $$$BODY }",
        "javascript": "function $NAME($$$PARAMS) { $$$BODY }",
        "java": "$MODIFIER $TYPE $NAME($$$PARAMS) { $$$BODY }",
        "go": "func $NAME($$$PARAMS) $RET { $$$BODY }",
        "rust": "fn $NAME($$$PARAMS) -> $RET { $$$BODY }",
    },
    "async_function": {
        "python": "async def $NAME($$$PARAMS): $$$BODY",
        "typescript": "async function $NAME($$$PARAMS) { $$$BODY }",
        "javascript": "async function $NAME($$$PARAMS) { $$$BODY }",
    },
    "class": {
        "python": "class $NAME: $$$BODY",
        "typescript": "class $NAME { $$$BODY }",
        "javascript": "class $NAME { $$$BODY }",
        "java": "class $NAME { $$$BODY }",
        "go": "type $NAME struct { $$$BODY }",
        "rust": "struct $NAME { $$$BODY }",
    },
    "try_catch": {
        "python": "try: $$$TRY except $$$EXCEPT: $$$HANDLER",
        "typescript": "try { $$$TRY } catch ($ERR) { $$$HANDLER }",
        "javascript": "try { $$$TRY } catch ($ERR) { $$$HANDLER }",
        "java": "try { $$$TRY } catch ($TYPE $ERR) { $$$HANDLER }",
        "go": "if $ERR != nil { $$$HANDLER }",
        "rust": "match $EXPR { Ok($VAL) => $$$OK, Err($ERR) => $$$ERR }",
    },
    "import": {
        "python": "import $MODULE",
        "typescript": "import $$$IMPORTS from $MODULE",
        "javascript": "import $$$IMPORTS from $MODULE",
        "java": "import $PACKAGE.$CLASS",
        "go": "import $$$IMPORTS",
        "rust": "use $$$PATH",
    },
    "if_statement": {
        "python": "if $COND: $$$BODY",
        "typescript": "if ($COND) { $$$BODY }",
        "javascript": "if ($COND) { $$$BODY }",
        "java": "if ($COND) { $$$BODY }",
        "go": "if $COND { $$$BODY }",
        "rust": "if $COND { $$$BODY }",
    },
    "for_loop": {
        "python": "for $VAR in $ITER: $$$BODY",
        "typescript": "for ($INIT; $COND; $UPDATE) { $$$BODY }",
        "javascript": "for ($INIT; $COND; $UPDATE) { $$$BODY }",
        "java": "for ($INIT; $COND; $UPDATE) { $$$BODY }",
        "go": "for $INIT; $COND; $UPDATE { $$$BODY }",
        "rust": "for $VAR in $ITER { $$$BODY }",
    },
    "foreach": {
        "python": "for $VAR in $ITER: $$$BODY",
        "typescript": "for (const $VAR of $ITER) { $$$BODY }",
        "javascript": "for (const $VAR of $ITER) { $$$BODY }",
        "java": "for ($TYPE $VAR : $ITER) { $$$BODY }",
        "go": "for _, $VAR := range $ITER { $$$BODY }",
        "rust": "for $VAR in $ITER { $$$BODY }",
    },
    "arrow_function": {
        "python": "lambda $$$PARAMS: $BODY",
        "typescript": "($$$PARAMS) => $BODY",
        "javascript": "($$$PARAMS) => $BODY",
        "java": "($$$PARAMS) -> $BODY",
        "rust": "|$$$PARAMS| $BODY",
    },
}


# =============================================================================
# Semantic Group Detection
# =============================================================================

# Keywords mapped to semantic groups for O(1) lookup
SEMANTIC_GROUP_KEYWORDS = {
    "async_operations": ["async", "await"],
    "error_handling": ["try", "catch", "except"],
    "class_definitions": ["class", "struct"],
    "function_definitions": ["def ", "function", "func "],
    "imports": ["import", "require", "use "],
    "conditionals": ["if ", "else"],
    "loops": ["for ", "while "],
}

MAX_LANGUAGE_SEARCH_WORKERS = 5


def _detect_semantic_group(snippet: str) -> str:
    """Detect semantic group from code snippet using keyword matching."""
    snippet_lower = snippet.lower()

    for group, keywords in SEMANTIC_GROUP_KEYWORDS.items():
        for keyword in keywords:
            if keyword in snippet_lower:
                return group

    return "other"


def _group_by_semantic(
    matches: List[MultiLanguageMatch],
    semantic_query: str,
) -> List[MultiLanguageMatch]:
    """Group matches by semantic similarity."""
    for match in matches:
        match.semantic_group = _detect_semantic_group(match.code_snippet)
    return matches


# =============================================================================
# Search Helpers
# =============================================================================


def _detect_languages(project_folder: str) -> List[str]:
    """Detect programming languages present in a project."""
    detected = set()
    project_path = Path(project_folder)

    for lang, extensions in LANGUAGE_EXTENSIONS.items():
        for ext in extensions:
            if list(project_path.rglob(f"*{ext}"))[:1]:
                detected.add(lang)
                break

    return list(detected)


def _get_ast_grep_pattern(semantic: str, language: str) -> Optional[str]:
    """Get ast-grep pattern for a semantic concept in a language."""
    patterns = SEMANTIC_TO_AST_GREP.get(semantic, {})
    return patterns.get(language)


def _parse_match(match_data: Dict[str, Any], language: str) -> MultiLanguageMatch:
    """Parse a single match from ast-grep JSON output."""
    return MultiLanguageMatch(
        language=language,
        file_path=match_data.get("file", ""),
        line_number=match_data.get("range", {}).get("start", {}).get("line", 0),
        code_snippet=match_data.get("text", ""),
        semantic_group="",
        confidence=1.0,
    )


def _search_language(
    project_folder: str,
    language: str,
    pattern: str,
    max_results: int,
) -> List[MultiLanguageMatch]:
    """Search a single language for pattern matches."""
    extensions = LANGUAGE_EXTENSIONS.get(language, [])
    if not extensions:
        return []

    args = ["--pattern", pattern, "--lang", language, "--json", project_folder]

    try:
        result = run_ast_grep("run", args)
    except Exception as e:
        logger.warning("search_language_failed", language=language, error=str(e)[:100])
        return []

    if result.returncode != 0 or not result.stdout:
        return []

    try:
        data = json_module.loads(result.stdout)
    except json_module.JSONDecodeError:
        return []

    if not isinstance(data, list):
        return []

    return [_parse_match(m, language) for m in data[:max_results]]


# =============================================================================
# Query Parsing
# =============================================================================

# Term to pattern key mappings for O(1) lookup
QUERY_TERM_MAPPINGS = {
    "async": "async_function",
    "await": "async_function",
    "asynchronous": "async_function",
    "function": "function",
    "method": "function",
    "def": "function",
    "class": "class",
    "struct": "class",
    "try": "try_catch",
    "catch": "try_catch",
    "except": "try_catch",
    "error": "try_catch",
    "exception": "try_catch",
    "import": "import",
    "require": "import",
    "use": "import",
    "if": "if_statement",
    "conditional": "if_statement",
    "for": "for_loop",
    "loop": "for_loop",
    "foreach": "foreach",
    "iterate": "foreach",
    "lambda": "arrow_function",
    "arrow": "arrow_function",
    "closure": "arrow_function",
}


def _parse_semantic_query(query: str) -> str:
    """Parse a semantic query to find the best matching pattern key."""
    query_lower = query.lower()

    # Whole words (plural allowed): "foreach" is not "for", "modify" is not "if"
    for term, pattern_key in QUERY_TERM_MAPPINGS.items():
        if re.search(rf"\b{re.escape(term)}(?:e?s)?\b", query_lower):
            return pattern_key

    return "function"


# =============================================================================
# Main Implementation
# =============================================================================


def _resolve_languages(project_folder: str, languages: Optional[List[str]]) -> List[str]:
    """Resolve and filter languages to search."""
    if not languages or languages == ["auto"]:
        languages = _detect_languages(project_folder)
        logger.info("auto_detected_languages", languages=languages)
    return [lang for lang in languages if lang in SUPPORTED_LANGUAGES]


def _run_parallel_search(
    project_folder: str,
    languages: List[str],
    semantic_key: str,
    max_results_per_language: int,
) -> tuple[List[MultiLanguageMatch], Dict[str, int]]:
    """Execute parallel search across all languages, one bounded wait per language.

    Uses the shared per-item-timeout helper: a hung language search is
    abandoned after its wait instead of blocking the whole call at executor
    shutdown (the previous `with ThreadPoolExecutor` form waited for every
    worker, hung ones included, before returning).
    """
    searchable = [(lang, _get_ast_grep_pattern(semantic_key, lang)) for lang in languages]
    items: List[Tuple[str, str]] = [(lang, pattern) for lang, pattern in searchable if pattern]

    all_matches: List[MultiLanguageMatch] = []
    matches_by_language: Dict[str, int] = {}
    if not items:
        return all_matches, matches_by_language

    def search_one(item: Tuple[str, str]) -> List[MultiLanguageMatch]:
        lang, pattern = item
        return _search_language(project_folder, lang, pattern, max_results_per_language)

    def on_success(item: Tuple[str, str], matches: List[MultiLanguageMatch]) -> None:
        all_matches.extend(matches)
        matches_by_language[item[0]] = len(matches)

    def on_error(item: Tuple[str, str], error: Exception) -> None:
        if isinstance(error, WaitTimeoutError):
            logger.warning(
                "language_search_timed_out", language=item[0], timeout_seconds=SubprocessDefaults.AST_GREP_TIMEOUT_SECONDS
            )
        else:
            logger.warning("language_search_failed", language=item[0], error=(str(error) or type(error).__name__)[:100])
        matches_by_language[item[0]] = 0

    map_with_per_item_timeout(
        items,
        search_one,
        timeout_seconds=SubprocessDefaults.AST_GREP_TIMEOUT_SECONDS,
        max_workers=min(len(items), MAX_LANGUAGE_SEARCH_WORKERS),
        on_success=on_success,
        on_error=on_error,
    )
    return all_matches, matches_by_language


def search_multi_language_impl(
    project_folder: str,
    semantic_pattern: str,
    languages: Optional[List[str]] = None,
    group_by: str = "semantic",
    max_results_per_language: int = CrossLanguageDefaults.MAX_RESULTS_PER_LANGUAGE,
) -> MultiLanguageSearchResult:
    """Search across multiple languages for semantically equivalent patterns."""
    start_time = time.time()

    if not os.path.isdir(project_folder):
        raise ValueError(f"Project folder not found: {project_folder}")

    languages = _resolve_languages(project_folder, languages)
    if not languages:
        return MultiLanguageSearchResult(
            query=semantic_pattern,
            languages_searched=[],
            matches=[],
            total_matches=0,
            execution_time_ms=int((time.time() - start_time) * ConversionFactors.MILLISECONDS_PER_SECOND),
        )

    semantic_key = _parse_semantic_query(semantic_pattern)
    all_matches, matches_by_language = _run_parallel_search(project_folder, languages, semantic_key, max_results_per_language)

    if group_by == "semantic":
        all_matches = _group_by_semantic(all_matches, semantic_pattern)

    semantic_groups = list(set(m.semantic_group for m in all_matches if m.semantic_group))

    return MultiLanguageSearchResult(
        query=semantic_pattern,
        languages_searched=languages,
        matches=all_matches,
        total_matches=len(all_matches),
        matches_by_language=matches_by_language,
        semantic_groups=semantic_groups,
        execution_time_ms=int((time.time() - start_time) * ConversionFactors.MILLISECONDS_PER_SECOND),
    )
