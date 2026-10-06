"""Search feature service - implements the core search functionality."""

import contextlib
import json
import re
import time
from typing import Any, Dict, Generator, List, Literal, Optional, Union, cast

import sentry_sdk
import yaml

from ast_grep_mcp.constants import (
    CacheDefaults,
    CodeAnalysisDefaults,
    ConversionFactors,
    DisplayDefaults,
    FormattingDefaults,
    PatternSuggestionConfidence,
    RegexCaptureGroups,
    SemanticVolumeDefaults,
    StreamDefaults,
)
from ast_grep_mcp.core.cache import get_query_cache
from ast_grep_mcp.core.config import CACHE_ENABLED
from ast_grep_mcp.core.exceptions import InvalidYAMLError, NoMatchesError
from ast_grep_mcp.core.executor import (
    filter_files_by_size,
    run_ast_grep,
    stream_ast_grep_results,
)
from ast_grep_mcp.core.logging import get_logger
from ast_grep_mcp.models.base import DumpFormat
from ast_grep_mcp.models.pattern_debug import (
    AstComparison,
    IssueCategory,
    IssueSeverity,
    MatchAttempt,
    MetavariableInfo,
    PatternDebugResult,
    PatternIssue,
)
from ast_grep_mcp.models.pattern_develop import (
    CodeAnalysis,
    PatternDevelopResult,
    PatternSuggestion,
    RefinementStep,
    SuggestionType,
)
from ast_grep_mcp.utils.formatters import format_matches_as_text


@contextlib.contextmanager
def _op_error_handler(
    op_name: str,
    logger: Any,
    start_time: float,
    sentry_extras: Dict[str, Any],
    passthrough: tuple[type, ...] = (),
) -> Generator[None, None, None]:
    """Context manager: log error, capture in Sentry, re-raise on exception.

    Exceptions listed in *passthrough* are re-raised without logging or Sentry capture.
    """
    try:
        yield
    except Exception as e:
        if passthrough and isinstance(e, passthrough):
            raise
        execution_time = time.time() - start_time
        logger.error(
            f"{op_name}_failed",
            execution_time_seconds=round(execution_time, FormattingDefaults.ROUNDING_PRECISION),
            error=str(e)[: DisplayDefaults.ERROR_OUTPUT_PREVIEW_LENGTH],
            status="failed",
        )
        sentry_sdk.capture_exception(
            e,
            extras={
                **sentry_extras,
                "execution_time_seconds": round(execution_time, FormattingDefaults.ROUNDING_PRECISION),
            },
        )
        raise


def dump_syntax_tree_impl(code: str, language: str, format: DumpFormat = "cst") -> str:
    """Dump code's syntax structure. format: pattern|ast|cst."""
    logger = get_logger("search.dump_syntax_tree")
    start_time = time.time()
    logger.info("dump_syntax_tree_started", language=language, format=format, code_length=len(code))

    with _op_error_handler(
        "dump_syntax_tree",
        logger,
        start_time,
        {"function": "dump_syntax_tree_impl", "language": language, "format": format, "code_length": len(code)},
    ):
        result = run_ast_grep("run", ["--pattern", code, "--lang", language, f"--debug-query={format}"])
        output = result.stderr.strip()
        execution_time = time.time() - start_time
        logger.info(
            "dump_syntax_tree_completed",
            execution_time_seconds=round(execution_time, FormattingDefaults.ROUNDING_PRECISION),
            output_length=len(output),
            status="success",
        )
        return output


def _run_scan_against_code(code: str, yaml_rule: str, start_time: float, logger: Any, parsed_yaml: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Execute ast-grep scan and return matches, raising on failure."""
    with _op_error_handler(
        "test_match_code_rule",
        logger,
        start_time,
        {
            "function": "test_match_code_rule_impl",
            "rule_id": parsed_yaml.get("id"),
            "language": parsed_yaml.get("language"),
            "code_length": len(code),
        },
        passthrough=(InvalidYAMLError, NoMatchesError),
    ):
        result = run_ast_grep("scan", ["--inline-rules", yaml_rule, "--json", "--stdin"], input_text=code)
        matches = cast(List[Dict[str, Any]], json.loads(result.stdout.strip()))
        execution_time = time.time() - start_time
        logger.info(
            "test_match_code_rule_completed",
            execution_time_seconds=round(execution_time, FormattingDefaults.ROUNDING_PRECISION),
            match_count=len(matches),
            status="success",
        )
        if not matches:
            raise NoMatchesError("No matches found for the given code and rule")
        return matches


def test_match_code_rule_impl(code: str, yaml_rule: str) -> List[Dict[str, Any]]:
    """Test code against an ast-grep YAML rule. Returns list of matches."""
    logger = get_logger("search.test_match_code_rule")
    start_time = time.time()

    parsed_yaml = _validate_yaml_rule(yaml_rule)

    logger.info(
        "test_match_code_rule_started",
        rule_id=parsed_yaml.get("id"),
        language=parsed_yaml.get("language"),
        code_length=len(code),
        yaml_length=len(yaml_rule),
    )

    return _run_scan_against_code(code, yaml_rule, start_time, logger, parsed_yaml)


def _prepare_search_targets(project_folder: str, max_file_size_mb: int, language: str, logger: Any) -> List[str]:
    """Return search targets (directories or filtered files) based on file size limit."""
    if max_file_size_mb <= 0:
        return [project_folder]

    files_to_search, skipped_files = filter_files_by_size(
        project_folder, max_size_mb=max_file_size_mb, language=language if language else None
    )

    if files_to_search:
        logger.info(
            "file_size_filtering_applied",
            files_to_search=len(files_to_search),
            files_skipped=len(skipped_files),
            max_size_mb=max_file_size_mb,
        )
        return files_to_search

    if skipped_files:
        logger.warning("all_files_skipped_by_size", total_files=len(skipped_files), max_size_mb=max_file_size_mb)
        return []  # All files exceeded size limit

    # No files found at all, continue with directory search
    return [project_folder]


def _build_search_args(pattern: str, language: str, workers: int, search_targets: List[str]) -> List[str]:
    """Build ast-grep command arguments."""
    args = ["--pattern", pattern]
    if language:
        args.extend(["--lang", language])
    if workers > 0:
        args.extend(["--threads", str(workers)])

    return args + ["--json=stream"] + search_targets


def _cache_args_with_globs(stream_args: List[str], language_globs: Optional[Dict[str, List[str]]]) -> List[str]:
    """Return cache key args that include a stable serialization of language_globs."""
    if not language_globs:
        return stream_args
    import json as _json

    globs_key = _json.dumps(language_globs, sort_keys=True)
    return stream_args + [f"__language_globs__={globs_key}"]


def _check_cache(
    cache: Any,
    stream_args: List[str],
    project_folder: str,
    max_results: int,
    output_format: str,
    logger: Any,
    language_globs: Optional[Dict[str, List[str]]] = None,
) -> Union[str, List[Dict[str, Any]], None]:
    """Return formatted cached results, or None on a cache miss."""
    if not cache or max_results != 0:
        return None

    cache_args = _cache_args_with_globs(stream_args, language_globs)
    cached_result = cache.get("run", cache_args, project_folder)
    if cached_result is None:
        logger.info("find_code_cache_miss")
        return None

    # Apply max_results limit to cached results
    matches = cached_result[:max_results] if max_results > 0 else cached_result
    logger.info("find_code_cache_hit", cache_size=len(cache.cache), cached_results=len(matches))

    return _format_search_results(matches, output_format)


def _execute_search(
    stream_args: List[str],
    max_results: int,
    cache: Any,
    project_folder: str,
    logger: Any,
    language_globs: Optional[Dict[str, List[str]]] = None,
) -> List[Dict[str, Any]]:
    """Execute the search and optionally cache results.

    Follows an all-or-nothing contract: if an exception is raised at any point
    during streaming, the partially accumulated ``matches`` list is discarded
    and the exception is re-raised to the caller.  Partial results are only
    logged as a warning for diagnostic purposes and are never returned.
    Callers should not expect a partial result set on failure.
    """
    # Accumulate incrementally so partial results can be logged if streaming fails mid-stream.
    matches: List[Dict[str, Any]] = []
    try:
        for match in stream_ast_grep_results(
            "run",
            stream_args,
            max_results=max_results,
            progress_interval=StreamDefaults.PROGRESS_INTERVAL,
            language_globs=language_globs,
        ):
            matches.append(match)
    except Exception:
        if matches:
            logger.warning("find_code_partial_results", partial_match_count=len(matches))
        raise

    # Store in cache if available
    if cache and max_results == 0:
        cache_args = _cache_args_with_globs(stream_args, language_globs)
        cache.put("run", cache_args, project_folder, matches)
        logger.info("find_code_cache_stored", stored_results=len(matches), cache_size=len(cache.cache))

    return matches


def _format_search_results(matches: List[Dict[str, Any]], output_format: str) -> Union[str, List[Dict[str, Any]]]:
    """Format search results based on output format."""
    if output_format == "text":
        if not matches:
            return "No matches found"
        text_output = format_matches_as_text(matches)
        header = f"Found {len(matches)} matches"
        return header + ":\n\n" + text_output

    return matches


def _validate_output_format(output_format: str) -> None:
    """Raise ValueError if output_format is not 'text' or 'json'."""
    if output_format not in ["text", "json"]:
        raise ValueError(f"Invalid output_format: {output_format}. Must be 'text' or 'json'.")


def _handle_empty_search_targets(search_targets: List[str], output_format: str) -> Union[str, List[Any], None]:
    """Return empty result when all files were size-filtered, or None if targets remain."""
    if not search_targets:
        return "No matches found (all files exceeded size limit)" if output_format == "text" else []
    return None


def _run_find_code_search(
    project_folder: str,
    pattern: str,
    language: str,
    max_results: int,
    output_format: str,
    workers: int,
    search_targets: List[str],
    logger: Any,
    start_time: float,
    language_globs: Optional[Dict[str, List[str]]] = None,
) -> Union[str, List[Dict[str, Any]]]:
    """Execute find_code search with pre-validated targets."""
    stream_args = _build_search_args(pattern, language, workers, search_targets)
    cache = get_query_cache()
    cached_result = _check_cache(cache, stream_args, project_folder, max_results, output_format, logger, language_globs=language_globs)
    if cached_result is not None:
        return cached_result

    matches = _execute_search(stream_args, max_results, cache, project_folder, logger, language_globs=language_globs)
    result = _format_search_results(matches, output_format)
    execution_time = time.time() - start_time
    logger.info(
        "find_code_completed",
        execution_time_seconds=round(execution_time, FormattingDefaults.ROUNDING_PRECISION),
        match_count=len(matches),
        status="success",
    )
    return result


def find_code_impl(
    project_folder: str,
    pattern: str,
    language: str = "",
    max_results: int = 0,
    output_format: Literal["text", "json"] = "text",
    max_file_size_mb: int = 0,
    workers: int = 0,
    language_globs: Optional[Dict[str, List[str]]] = None,
) -> Union[str, List[Dict[str, Any]]]:
    """Find code in a project folder using a pattern.

    Args:
        language_globs: Optional mapping of language name to glob patterns, e.g.
            ``{"javascript": ["**/*.cjs"], "json": [".eslintrc", "*.schema.json"]}``.
            When provided a temporary sgconfig.yml is written and passed to ast-grep
            via ``--config``, enabling non-standard file extensions to be parsed with
            the correct Tree-sitter grammar.
    """
    logger = get_logger("search.find_code")
    start_time = time.time()
    size_display: Union[int, str] = max_file_size_mb if max_file_size_mb > 0 else "unlimited"
    workers_display: Union[int, str] = workers if workers > 0 else "auto"
    logger.info(
        "find_code_started",
        project_folder=project_folder,
        pattern=pattern[:100],
        language=language or "auto",
        max_results=max_results,
        output_format=output_format,
        max_file_size_mb=size_display,
        workers=workers_display,
    )
    with _op_error_handler(
        "find_code",
        logger,
        start_time,
        {"function": "find_code_impl", "project_folder": project_folder, "pattern": pattern[:100], "language": language},
    ):
        _validate_output_format(output_format)
        search_targets = _prepare_search_targets(project_folder, max_file_size_mb, language, logger)
        empty_result = _handle_empty_search_targets(search_targets, output_format)
        if empty_result is not None:
            return empty_result
        return _run_find_code_search(
            project_folder,
            pattern,
            language,
            max_results,
            output_format,
            workers,
            search_targets,
            logger,
            start_time,
            language_globs=language_globs,
        )


def _validate_yaml_rule(yaml_rule: str) -> Dict[str, Any]:
    """Parse and validate YAML rule; raise InvalidYAMLError if invalid."""
    try:
        parsed_yaml = yaml.safe_load(yaml_rule)
        if not isinstance(parsed_yaml, dict):
            raise InvalidYAMLError("YAML must be a dictionary", yaml_rule)

        required_fields = ["id", "language", "rule"]
        for field in required_fields:
            if field not in parsed_yaml:
                raise InvalidYAMLError(f"Missing required field '{field}'", yaml_rule)

        return parsed_yaml
    except yaml.YAMLError as e:
        raise InvalidYAMLError(f"YAML parsing failed: {e}", yaml_rule) from e


# Relational rules that require stopBy for deep tree traversal
RELATIONAL_RULES = ["inside", "has", "follows", "precedes"]


def _check_relational_entry(rule_obj: Dict[str, Any], rel_rule: str, path: str) -> List[str]:
    """Check a single relational rule entry for missing stopBy."""
    warnings: List[str] = []
    if rel_rule not in rule_obj:
        return warnings
    rel_content = rule_obj[rel_rule]
    if not isinstance(rel_content, dict):
        return warnings
    if "stopBy" not in rel_content:
        warnings.append(
            f"⚠️  Missing 'stopBy' in '{rel_rule}' at {path}.{rel_rule}. "
            f"Default is 'neighbor' (immediate parent only). "
            f"Add 'stopBy: end' to search the entire tree."
        )
    warnings.extend(_check_relational_rule_for_stopby(rel_content, f"{path}.{rel_rule}"))
    return warnings


def _check_composite_entries(rule_obj: Dict[str, Any], path: str) -> List[str]:
    """Check composite rule entries (all, any, not) recursively."""
    warnings: List[str] = []
    for composite in ["all", "any"]:
        if composite in rule_obj and isinstance(rule_obj[composite], list):
            for i, sub_rule in enumerate(rule_obj[composite]):
                warnings.extend(_check_relational_rule_for_stopby(sub_rule, f"{path}.{composite}[{i}]"))
    if "not" in rule_obj and isinstance(rule_obj["not"], dict):
        warnings.extend(_check_relational_rule_for_stopby(rule_obj["not"], f"{path}.not"))
    return warnings


def _check_relational_rule_for_stopby(rule_obj: Any, path: str = "rule") -> List[str]:
    """Recursively check a rule object for relational rules missing stopBy; return warnings."""
    if not isinstance(rule_obj, dict):
        return []
    warnings: List[str] = []
    for rel_rule in RELATIONAL_RULES:
        warnings.extend(_check_relational_entry(rule_obj, rel_rule, path))
    warnings.extend(_check_composite_entries(rule_obj, path))
    return warnings


def _check_yaml_rule_for_common_mistakes(parsed_yaml: Dict[str, Any]) -> List[str]:
    """Check parsed YAML rule for common mistakes; return list of warning messages."""
    warnings: List[str] = []

    rule = parsed_yaml.get("rule", {})
    if isinstance(rule, dict):
        warnings.extend(_check_relational_rule_for_stopby(rule))

    # Check for lowercase metavariables in pattern
    pattern = rule.get("pattern", "") if isinstance(rule, dict) else ""
    if isinstance(pattern, str) and re.search(r"\$[a-z]", pattern):
        warnings.append("⚠️  Pattern may contain lowercase metavariable (e.g., $name). Use UPPERCASE: $NAME, $ARGS, etc.")

    return warnings


def _execute_rule_search(
    project_folder: str, yaml_rule: str, max_results: int, output_format: str, cache: Any, logger: Any
) -> Union[str, List[Dict[str, Any]]]:
    """Execute search with YAML rule."""
    json_arg = ["--json"] if output_format == "json" else []

    if max_results > 0:
        # Streaming parses one JSON object per line, so it always needs --json=stream
        matches = list(
            stream_ast_grep_results("scan", ["--inline-rules", yaml_rule, "--json=stream", project_folder], max_results=max_results)
        )

        if output_format == "text":
            return format_matches_as_text(matches)
        return matches

    # Use non-streaming for all results
    cmd_result = run_ast_grep("scan", ["--inline-rules", yaml_rule, *json_arg, project_folder])

    if output_format == "json":
        return json.loads(cmd_result.stdout.strip()) if cmd_result.stdout.strip() else []

    return cmd_result.stdout.strip()


def _prepend_warnings_to_result(
    result: Union[str, List[Dict[str, Any]]], warnings: List[str], output_format: str
) -> Union[str, List[Dict[str, Any]], Dict[str, Any]]:
    """Prepend warnings to result; for JSON wraps as {warnings, matches} dict."""
    if not warnings:
        return result

    if output_format == "text":
        warning_block = "\n".join(warnings) + "\n\n"
        return warning_block + (result if isinstance(result, str) else "")

    # For JSON, add warnings as metadata
    if isinstance(result, list):
        return {
            "warnings": warnings,
            "matches": result,
        }
    return result


def _check_rule_cache(
    cache: Any, cache_key_parts: List[str], project_folder: str, max_results: int, logger: Any, rule_id: Optional[str]
) -> Optional[Any]:
    """Check cache for existing rule search results."""
    if not CACHE_ENABLED or not cache or max_results != 0:
        return None

    cached_result = cache.get("scan", cache_key_parts, project_folder)
    if cached_result is not None:
        logger.info("find_code_by_rule_cache_hit", rule_id=rule_id)
    return cached_result


def _store_rule_result_in_cache(cache: Any, cache_key_parts: List[str], project_folder: str, result: Any, max_results: int) -> None:
    """Store rule search result in cache if applicable."""
    if CACHE_ENABLED and cache and max_results == 0 and isinstance(result, list):
        cache.put("scan", cache_key_parts, project_folder, result)


def _log_rule_warnings(warnings: List[str], parsed_yaml: Dict[str, Any], logger: Any) -> None:
    """Log warnings about common mistakes in the YAML rule."""
    if warnings:
        logger.warning(
            "yaml_rule_warnings",
            rule_id=parsed_yaml.get("id"),
            warning_count=len(warnings),
            warnings=warnings,
        )


def _run_rule_search_with_cache(
    project_folder: str,
    yaml_rule: str,
    max_results: int,
    output_format: str,
    warnings: List[str],
    parsed_yaml: Dict[str, Any],
    logger: Any,
    start_time: float,
) -> Union[str, List[Dict[str, Any]], Dict[str, Any]]:
    """Execute rule search, using/updating cache, and prepend warnings."""
    cache = get_query_cache()
    cache_key_parts = ["scan", yaml_rule, output_format, project_folder]

    cached_result = _check_rule_cache(cache, cache_key_parts, project_folder, max_results, logger, parsed_yaml.get("id"))
    if cached_result is not None:
        return _prepend_warnings_to_result(cached_result, warnings, output_format)

    result = _execute_rule_search(project_folder, yaml_rule, max_results, output_format, cache, logger)
    _store_rule_result_in_cache(cache, cache_key_parts, project_folder, result, max_results)

    execution_time = time.time() - start_time
    match_count = len(result) if isinstance(result, list) else result.count("\n")
    logger.info(
        "find_code_by_rule_completed",
        execution_time_seconds=round(execution_time, FormattingDefaults.ROUNDING_PRECISION),
        match_count=match_count,
        status="success",
    )
    return _prepend_warnings_to_result(result, warnings, output_format)


def find_code_by_rule_impl(
    project_folder: str, yaml_rule: str, max_results: int = 0, output_format: Literal["text", "json"] = "text"
) -> Union[str, List[Dict[str, Any]], Dict[str, Any]]:
    """Find code using a YAML rule. Returns str, list, or dict with warnings."""
    logger = get_logger("search.find_code_by_rule")
    start_time = time.time()

    parsed_yaml = _validate_yaml_rule(yaml_rule)
    warnings = _check_yaml_rule_for_common_mistakes(parsed_yaml)
    _log_rule_warnings(warnings, parsed_yaml, logger)

    logger.info(
        "find_code_by_rule_started",
        project_folder=project_folder,
        rule_id=parsed_yaml.get("id"),
        language=parsed_yaml.get("language"),
        max_results=max_results,
        output_format=output_format,
    )

    with _op_error_handler(
        "find_code_by_rule",
        logger,
        start_time,
        {
            "function": "find_code_by_rule_impl",
            "project_folder": project_folder,
            "rule_id": parsed_yaml.get("id"),
            "language": parsed_yaml.get("language"),
        },
        passthrough=(InvalidYAMLError,),
    ):
        return _run_rule_search_with_cache(project_folder, yaml_rule, max_results, output_format, warnings, parsed_yaml, logger, start_time)


# =============================================================================
# Rule Builder Implementation
# =============================================================================


def _build_relational_rule(pattern_or_kind: str, stop_by: str, is_kind: bool = False) -> Dict[str, Any]:
    """Build a relational rule dict with stopBy; uses 'kind' key when is_kind=True."""
    rule: Dict[str, Any] = {"stopBy": stop_by}
    if is_kind:
        rule["kind"] = pattern_or_kind
    else:
        rule["pattern"] = pattern_or_kind
    return rule


def _add_relational_to_rule(
    rule_obj: Dict[str, Any],
    relational_type: str,
    value: Optional[str],
    stop_by: str,
) -> None:
    """Add a relational rule entry to rule_obj if value is provided."""
    if value:
        rule_obj[relational_type] = _build_relational_rule(value, stop_by)


def _generate_rule_id(pattern: str, language: str, inside: Optional[str], has: Optional[str]) -> str:
    """Generate a deterministic rule ID from pattern components."""
    import hashlib

    hash_input = f"{pattern}{language}{inside}{has}"
    return f"rule-{hashlib.sha256(hash_input.encode()).hexdigest()[: CacheDefaults.RULE_ID_HASH_LENGTH]}"


def _build_rule_object(
    pattern: str,
    inside: Optional[str],
    has: Optional[str],
    follows: Optional[str],
    precedes: Optional[str],
    inside_kind: Optional[str],
    has_kind: Optional[str],
    stop_by: str,
) -> Dict[str, Any]:
    """Build the rule object with relational constraints."""
    rule_obj: Dict[str, Any] = {"pattern": pattern}
    _add_relational_to_rule(rule_obj, "inside", inside, stop_by)
    _add_relational_to_rule(rule_obj, "has", has, stop_by)
    _add_relational_to_rule(rule_obj, "follows", follows, stop_by)
    _add_relational_to_rule(rule_obj, "precedes", precedes, stop_by)
    if inside_kind:
        rule_obj["inside"] = _build_relational_rule(inside_kind, stop_by, is_kind=True)
    if has_kind:
        rule_obj["has"] = _build_relational_rule(has_kind, stop_by, is_kind=True)
    return rule_obj


def _build_yaml_object(
    rule_id: str,
    language: str,
    rule_obj: Dict[str, Any],
    message: Optional[str],
    severity: Optional[str],
    fix: Optional[str],
) -> Dict[str, Any]:
    """Build the full YAML structure with optional fields."""
    yaml_obj: Dict[str, Any] = {"id": rule_id, "language": language, "rule": rule_obj}
    if message:
        yaml_obj["message"] = message
    if severity:
        yaml_obj["severity"] = severity
    if fix is not None:
        yaml_obj["fix"] = fix
    return yaml_obj


def build_rule_impl(
    pattern: str,
    language: str,
    rule_id: Optional[str] = None,
    inside: Optional[str] = None,
    has: Optional[str] = None,
    follows: Optional[str] = None,
    precedes: Optional[str] = None,
    inside_kind: Optional[str] = None,
    has_kind: Optional[str] = None,
    stop_by: str = "end",
    message: Optional[str] = None,
    severity: Optional[str] = None,
    fix: Optional[str] = None,
) -> str:
    """Build a YAML rule from components. Returns YAML string."""
    if not rule_id:
        rule_id = _generate_rule_id(pattern, language, inside, has)
    rule_obj = _build_rule_object(pattern, inside, has, follows, precedes, inside_kind, has_kind, stop_by)
    yaml_obj = _build_yaml_object(rule_id, language, rule_obj, message, severity, fix)
    return str(yaml.dump(yaml_obj, default_flow_style=False, sort_keys=False, allow_unicode=True))


# =============================================================================
# Pattern Debugging Implementation
# =============================================================================

# Regex patterns for metavariable detection
METAVAR_SINGLE = re.compile(r"\$([A-Z][A-Z0-9_]*)")  # $NAME, $VAR1
METAVAR_MULTI = re.compile(r"\$\$\$([A-Z][A-Z0-9_]*)?")  # $$$, $$$ARGS
METAVAR_NON_CAPTURING = re.compile(r"\$_([A-Z][A-Z0-9_]*)?")  # $_, $_NAME
METAVAR_UNNAMED = re.compile(r"\$\$([A-Z][A-Z0-9_]*)")  # $$VAR

# Invalid metavariable patterns with their error messages
INVALID_METAVAR_PATTERNS = [
    (re.compile(r"\$([a-z][a-zA-Z0-9_]*)"), "lowercase", "Metavariable must use UPPERCASE"),
    (re.compile(r"\$(\d[A-Z0-9_]*)"), "digit_start", "Metavariable cannot start with digit"),
    (re.compile(r"\$([A-Z][A-Z0-9]*-[A-Z0-9_-]*)"), "hyphen", "Metavariable cannot contain hyphens"),
]


def _invalid_metavar_issue(match: re.Match[str], error_type: str, base_message: str) -> MetavariableInfo:
    """Build a MetavariableInfo for an invalid metavariable match."""
    name = f"${match.group(1)}"
    if error_type == "lowercase":
        issue = f"{base_message}: '{name}' should be '${match.group(1).upper()}'"
    else:
        issue = f"{base_message}: '{name}'"
    return MetavariableInfo(name=name, type="invalid", valid=False, occurrences=1, issue=issue)


def _extract_invalid_metavariables(pattern: str) -> List[MetavariableInfo]:
    """Extract invalid metavariables from pattern."""
    metavars: List[MetavariableInfo] = []
    for regex, error_type, base_message in INVALID_METAVAR_PATTERNS:
        for match in regex.finditer(pattern):
            metavars.append(_invalid_metavar_issue(match, error_type, base_message))
    return metavars


def _add_metavar_if_new(
    display_name: str,
    mv_type: str,
    seen: Dict[str, int],
    metavars: List[MetavariableInfo],
) -> None:
    """Append metavar to metavars if unseen; otherwise increment its occurrence count."""
    if display_name in seen:
        seen[display_name] += 1
    else:
        seen[display_name] = 1
        metavars.append(
            MetavariableInfo(
                name=display_name,
                type=mv_type,
                valid=True,
                occurrences=1,
            )
        )


def _extract_multi_metavariables(pattern: str, seen: Dict[str, int], metavars: List[MetavariableInfo]) -> None:
    """Extract $$$ multi-node metavariables."""
    for match in METAVAR_MULTI.finditer(pattern):
        name = match.group(1) or ""
        display_name = f"$$${name}" if name else "$$$"
        _add_metavar_if_new(display_name, "multi", seen, metavars)


def _extract_unnamed_metavariables(pattern: str, seen: Dict[str, int], metavars: List[MetavariableInfo]) -> None:
    """Extract $$ unnamed node metavariables."""
    for match in METAVAR_UNNAMED.finditer(pattern):
        start = match.start()
        if start > 0 and pattern[start - 1] == "$":
            continue  # Skip $$$ patterns
        name = f"$${match.group(1)}"
        _add_metavar_if_new(name, "unnamed", seen, metavars)


def _extract_non_capturing_metavariables(pattern: str, seen: Dict[str, int], metavars: List[MetavariableInfo]) -> None:
    """Extract $_ non-capturing metavariables."""
    for match in METAVAR_NON_CAPTURING.finditer(pattern):
        name_part = match.group(1) or ""
        display_name = f"$_{name_part}" if name_part else "$_"
        _add_metavar_if_new(display_name, "non_capturing", seen, metavars)


def _extract_single_metavariables(pattern: str, seen: Dict[str, int], metavars: List[MetavariableInfo]) -> None:
    """Extract $NAME single-node metavariables."""
    for match in METAVAR_SINGLE.finditer(pattern):
        start = match.start()
        # Skip if part of $$$ or $$ or $_
        if start > 0 and pattern[start - 1] == "$":
            continue
        if start + 1 < len(pattern) and pattern[start + 1] == "_":
            continue
        name = f"${match.group(1)}"
        _add_metavar_if_new(name, "single", seen, metavars)


def _extract_metavariables(pattern: str) -> List[MetavariableInfo]:
    """Extract and validate all metavariables from a pattern."""
    metavars: List[MetavariableInfo] = []
    seen: Dict[str, int] = {}

    # Check for invalid metavariables first
    metavars.extend(_extract_invalid_metavariables(pattern))

    # Extract valid metavariables by type
    _extract_multi_metavariables(pattern, seen, metavars)
    _extract_unnamed_metavariables(pattern, seen, metavars)
    _extract_non_capturing_metavariables(pattern, seen, metavars)
    _extract_single_metavariables(pattern, seen, metavars)

    # Update occurrence counts
    for mv in metavars:
        if mv.name in seen:
            mv.occurrences = seen[mv.name]

    return metavars


# Fragment patterns that indicate incomplete code
FRAGMENT_INDICATORS = [
    (r"^\s*\.\w+", "Pattern starts with a method call - ensure full expression context"),
    (r"^\s*:\s*\w+", "Pattern starts with a type annotation - ensure full expression context"),
    (r"^\s*=\s*", "Pattern starts with assignment operator - include left-hand side"),
]


def _metavar_suggestion(mv: MetavariableInfo) -> str:
    """Build suggestion text for an invalid metavariable."""
    if "lowercase" in (mv.issue or "").lower():
        return f"Use uppercase letters: ${mv.name[1:].upper()}"
    return "Fix the metavariable syntax"


def _check_invalid_metavar_issues(metavars: List[MetavariableInfo]) -> List[PatternIssue]:
    """Check for invalid metavariable issues."""
    return [
        PatternIssue(
            severity=IssueSeverity.ERROR,
            category=IssueCategory.METAVARIABLE,
            message=mv.issue,
            suggestion=_metavar_suggestion(mv),
            location=mv.name,
        )
        for mv in metavars
        if not mv.valid and mv.issue
    ]


def _single_arg_issue(content: str) -> PatternIssue:
    """Build a PatternIssue for a single metavar in function args."""
    return PatternIssue(
        severity=IssueSeverity.INFO,
        category=IssueCategory.BEST_PRACTICE,
        message=f"Single metavariable '{content}' in function arguments may not match multiple arguments",
        suggestion=f"Use '$$$ARGS' or '$$${content[1:]}' to match zero or more arguments",
        location=content,
    )


def _check_single_arg_metavar_issues(pattern: str) -> List[PatternIssue]:
    """Check for single metavar in function arguments that may need $$$."""
    if "(" not in pattern or ")" not in pattern:
        return []
    paren_content = re.findall(r"\(([^)]+)\)", pattern)
    return [_single_arg_issue(content.strip()) for content in paren_content if re.match(r"^\$[A-Z][A-Z0-9_]*$", content.strip())]


_FRAGMENT_SUGGESTION = "Patterns must be valid, parseable code. Wrap in full expression or use YAML rule with 'context' and 'selector'"


def _check_fragment_issues(pattern: str) -> List[PatternIssue]:
    """Check for incomplete code fragment patterns."""
    return [
        PatternIssue(
            severity=IssueSeverity.WARNING,
            category=IssueCategory.SYNTAX,
            message=message,
            suggestion=_FRAGMENT_SUGGESTION,
        )
        for indicator_pattern, message in FRAGMENT_INDICATORS
        if re.match(indicator_pattern, pattern)
    ]


def _check_pattern_issues(pattern: str, metavars: List[MetavariableInfo]) -> List[PatternIssue]:
    """Check for common pattern issues and return a list of PatternIssue objects."""
    issues: List[PatternIssue] = []

    issues.extend(_check_invalid_metavar_issues(metavars))
    issues.extend(_check_single_arg_metavar_issues(pattern))
    issues.extend(_check_fragment_issues(pattern))

    return issues


def _extract_root_kind(ast_output: str) -> Optional[str]:
    """Extract the root node kind from AST output, or None if not found."""
    kind_match = re.search(r"kind:\s*(\w+)", ast_output)
    if kind_match:
        return kind_match.group(1)

    cst_match = re.search(r"\((\w+)\)|\[(\w+)\]|^(\w+)\s", ast_output)
    if cst_match:
        return (
            cst_match.group(RegexCaptureGroups.FIRST)
            or cst_match.group(RegexCaptureGroups.SECOND)
            or cst_match.group(RegexCaptureGroups.THIRD)
        )

    first_line = ast_output.split("\n")[0] if ast_output else ""
    word_match = re.search(r"^(\w+)", first_line.strip())
    return word_match.group(1) if word_match else None


def _truncate_ast(ast_output: str) -> str:
    """Truncate AST output to display limit."""
    if len(ast_output) > DisplayDefaults.AST_TRUNCATION_LENGTH:
        return ast_output[: DisplayDefaults.AST_TRUNCATION_LENGTH]
    return ast_output


def _collect_ast_differences(pattern_root: Optional[str], code_root: Optional[str], pattern_ast: str, code_ast: str) -> List[str]:
    """Collect structural differences between two ASTs."""
    differences: List[str] = []
    if pattern_root and code_root and pattern_root != code_root:
        differences.append(f"Root node mismatch: pattern has '{pattern_root}', code has '{code_root}'")
    if "ERROR" in pattern_ast.upper():
        differences.append("Pattern contains parse errors - pattern may not be valid code")
    if "ERROR" in code_ast.upper():
        differences.append("Code contains parse errors - code may have syntax issues")
    return differences


def _compare_asts(pattern_ast: str, code_ast: str) -> AstComparison:
    """Compare pattern and code ASTs; return AstComparison with structural differences."""
    pattern_root = _extract_root_kind(pattern_ast)
    code_root = _extract_root_kind(code_ast)
    kinds_match = pattern_root == code_root if (pattern_root and code_root) else False
    differences = _collect_ast_differences(pattern_root, code_root, pattern_ast, code_ast)

    return AstComparison(
        pattern_root_kind=pattern_root,
        code_root_kind=code_root,
        kinds_match=kinds_match,
        pattern_structure=_truncate_ast(pattern_ast),
        code_structure=_truncate_ast(code_ast),
        structural_differences=differences,
    )


def _attempt_match(pattern: str, code: str, language: str) -> MatchAttempt:
    """Try to match the pattern against the code; return a MatchAttempt."""
    logger = get_logger("search.debug_pattern")

    try:
        # Create a minimal YAML rule to test matching
        yaml_rule = f"""
id: debug-pattern-test
language: {language}
rule:
  pattern: |
    {pattern}
"""
        result = run_ast_grep("scan", ["--inline-rules", yaml_rule, "--json", "--stdin"], input_text=code)
        matches = json.loads(result.stdout.strip()) if result.stdout.strip() else []

        return MatchAttempt(
            matched=len(matches) > 0,
            match_count=len(matches),
            matches=matches[: SemanticVolumeDefaults.TOP_RESULTS_LIMIT],  # Limit matches for debugging
        )
    except Exception as e:
        logger.debug("match_attempt_failed", error=str(e))
        return MatchAttempt(
            matched=False,
            match_count=0,
            partial_matches=[f"Match attempt failed: {str(e)[:100]}"],
        )


_SEVERITY_PREFIX: Dict[IssueSeverity, str] = {
    IssueSeverity.ERROR: "ERROR",
    IssueSeverity.WARNING: "WARNING",
    IssueSeverity.INFO: "TIP",
}


def _add_suggestions_by_severity(
    severity: IssueSeverity,
    issues: List[PatternIssue],
    suggestions: List[str],
) -> None:
    """Add suggestions filtered by severity level.

    The display prefix is derived from *severity* via ``_SEVERITY_PREFIX``
    (INFO maps to "TIP" rather than "INFO").
    """
    prefix = _SEVERITY_PREFIX[severity]
    for issue in issues:
        if issue.severity == severity:
            suggestions.append(f"[{prefix}] {issue.suggestion}")


def _add_structural_suggestions(ast_comparison: AstComparison, suggestions: List[str]) -> None:
    """Add structural mismatch suggestions (Priority 2)."""
    has_root_mismatch = not ast_comparison.kinds_match and ast_comparison.pattern_root_kind and ast_comparison.code_root_kind
    if has_root_mismatch:
        suggestions.append(
            f"[STRUCTURE] Pattern root is '{ast_comparison.pattern_root_kind}' but code root is "
            f"'{ast_comparison.code_root_kind}'. Adjust pattern to match code structure."
        )

    for diff in ast_comparison.structural_differences:
        if "ERROR" in diff:
            suggestions.append(f"[PARSE ERROR] {diff}")


def _add_debug_suggestions(
    issues: List[PatternIssue],
    ast_comparison: AstComparison,
    match_attempt: MatchAttempt,
    suggestions: List[str],
) -> None:
    """Add debugging suggestions when pattern doesn't match (Priority 3)."""
    has_errors = any(i.severity == IssueSeverity.ERROR for i in issues)

    if match_attempt.matched or has_errors:
        return

    suggestions.append(
        "[DEBUG] Pattern is valid but doesn't match. Use 'dump_syntax_tree' with format='cst' "
        "on both pattern and code to compare their AST structures."
    )

    if ast_comparison.pattern_root_kind != ast_comparison.code_root_kind:
        suggestions.append(f"[TIP] Try using 'kind: {ast_comparison.code_root_kind}' in a YAML rule instead of pattern matching.")


def _add_default_suggestions(match_attempt: MatchAttempt, suggestions: List[str]) -> None:
    """Add default guidance if no specific suggestions were generated."""
    if suggestions:
        return

    if match_attempt.matched:
        suggestions.append("[SUCCESS] Pattern matches the code. No issues found.")
    else:
        suggestions.append(
            "[HELP] No obvious issues found. Consider:\n"
            "  1. Check if pattern is valid syntax for the language\n"
            "  2. Use dump_syntax_tree to compare AST structures\n"
            "  3. Try a simpler pattern and gradually add complexity\n"
            "  4. Use YAML rule with 'context' and 'selector' for sub-expressions"
        )


def _generate_suggestions(
    pattern: str,
    code: str,
    language: str,
    issues: List[PatternIssue],
    ast_comparison: AstComparison,
    match_attempt: MatchAttempt,
) -> List[str]:
    """Generate prioritized suggestions for fixing pattern issues."""
    suggestions: List[str] = []

    _add_suggestions_by_severity(IssueSeverity.ERROR, issues, suggestions)
    _add_structural_suggestions(ast_comparison, suggestions)
    _add_debug_suggestions(issues, ast_comparison, match_attempt, suggestions)
    _add_suggestions_by_severity(IssueSeverity.WARNING, issues, suggestions)
    _add_suggestions_by_severity(IssueSeverity.INFO, issues, suggestions)
    _add_default_suggestions(match_attempt, suggestions)

    return suggestions


def _get_pattern_ast(pattern: str, language: str, issues: List[PatternIssue]) -> tuple[str, bool]:
    """Try to get pattern AST, return (ast_text, is_valid)."""
    try:
        return dump_syntax_tree_impl(pattern, language, "pattern"), True
    except Exception as e:
        issues.append(
            PatternIssue(
                severity=IssueSeverity.ERROR,
                category=IssueCategory.SYNTAX,
                message=f"Pattern failed to parse: {str(e)[:100]}",
                suggestion="Ensure pattern is valid, parseable code for the target language",
            )
        )
        return f"Error parsing pattern: {str(e)[: DisplayDefaults.ERROR_OUTPUT_PREVIEW_LENGTH]}", False


def _get_code_ast(code: str, language: str, issues: List[PatternIssue]) -> str:
    """Try to get code AST, return ast_text."""
    try:
        return dump_syntax_tree_impl(code, language, "cst")
    except Exception as e:
        issues.append(
            PatternIssue(
                severity=IssueSeverity.WARNING,
                category=IssueCategory.SYNTAX,
                message=f"Code failed to parse: {str(e)[:100]}",
                suggestion="Check that the code is valid syntax",
            )
        )
        return f"Error parsing code: {str(e)[: DisplayDefaults.ERROR_OUTPUT_PREVIEW_LENGTH]}"


def _run_debug_analysis(pattern: str, code: str, language: str, start_time: float, logger: Any) -> PatternDebugResult:
    """Execute the pattern debug analysis and build result."""
    metavars = _extract_metavariables(pattern)
    issues = _check_pattern_issues(pattern, metavars)
    pattern_ast, pattern_valid = _get_pattern_ast(pattern, language, issues)
    code_ast = _get_code_ast(code, language, issues)
    ast_comparison = _compare_asts(pattern_ast, code_ast)
    match_attempt = _attempt_match(pattern, code, language)
    suggestions = _generate_suggestions(pattern, code, language, issues, ast_comparison, match_attempt)
    execution_time = time.time() - start_time
    result = PatternDebugResult(
        pattern=pattern,
        code=code,
        language=language,
        pattern_valid=pattern_valid,
        pattern_ast=pattern_ast,
        code_ast=code_ast,
        ast_comparison=ast_comparison,
        metavariables=metavars,
        issues=issues,
        suggestions=suggestions,
        match_attempt=match_attempt,
        execution_time_ms=int(execution_time * ConversionFactors.MILLISECONDS_PER_SECOND),
    )
    logger.info(
        "debug_pattern_completed",
        execution_time_seconds=round(execution_time, FormattingDefaults.ROUNDING_PRECISION),
        pattern_valid=pattern_valid,
        issues_found=len(issues),
        matched=match_attempt.matched,
        status="success",
    )
    return result


def debug_pattern_impl(
    pattern: str,
    code: str,
    language: str,
) -> PatternDebugResult:
    """Debug why a pattern doesn't match code. Returns PatternDebugResult."""
    logger = get_logger("search.debug_pattern")
    start_time = time.time()
    logger.info("debug_pattern_started", language=language, pattern_length=len(pattern), code_length=len(code))
    with _op_error_handler(
        "debug_pattern",
        logger,
        start_time,
        {"function": "debug_pattern_impl", "language": language, "pattern_length": len(pattern), "code_length": len(code)},
    ):
        return _run_debug_analysis(pattern, code, language, start_time, logger)


# =============================================================================
# Pattern Development Implementation
# =============================================================================

# Common identifier patterns by language
IDENTIFIER_PATTERNS: Dict[str, re.Pattern[str]] = {
    "javascript": re.compile(r"\b([a-zA-Z_$][a-zA-Z0-9_$]*)\b"),
    "typescript": re.compile(r"\b([a-zA-Z_$][a-zA-Z0-9_$]*)\b"),
    "python": re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b"),
    "go": re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b"),
    "rust": re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b"),
    "java": re.compile(r"\b([a-zA-Z_$][a-zA-Z0-9_$]*)\b"),
    "ruby": re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b"),
    "c": re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b"),
    "cpp": re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b"),
}

# Keywords by language (subset of most common)
LANGUAGE_KEYWORDS: Dict[str, set[str]] = {
    "javascript": {
        "function",
        "const",
        "let",
        "var",
        "return",
        "if",
        "else",
        "for",
        "while",
        "class",
        "async",
        "await",
        "import",
        "export",
        "from",
        "new",
        "this",
        "true",
        "false",
        "null",
        "undefined",
    },
    "typescript": {
        "function",
        "const",
        "let",
        "var",
        "return",
        "if",
        "else",
        "for",
        "while",
        "class",
        "async",
        "await",
        "import",
        "export",
        "from",
        "new",
        "this",
        "true",
        "false",
        "null",
        "undefined",
        "interface",
        "type",
        "enum",
    },
    "python": {
        "def",
        "class",
        "return",
        "if",
        "elif",
        "else",
        "for",
        "while",
        "import",
        "from",
        "as",
        "try",
        "except",
        "finally",
        "with",
        "async",
        "await",
        "True",
        "False",
        "None",
        "and",
        "or",
        "not",
        "in",
        "is",
        "lambda",
    },
    "go": {
        "func",
        "package",
        "import",
        "return",
        "if",
        "else",
        "for",
        "range",
        "switch",
        "case",
        "type",
        "struct",
        "interface",
        "var",
        "const",
        "go",
        "defer",
        "chan",
        "select",
        "true",
        "false",
        "nil",
    },
    "rust": {
        "fn",
        "let",
        "mut",
        "const",
        "return",
        "if",
        "else",
        "for",
        "while",
        "loop",
        "match",
        "struct",
        "enum",
        "impl",
        "trait",
        "use",
        "mod",
        "pub",
        "async",
        "await",
        "true",
        "false",
        "self",
        "Self",
    },
    "java": {
        "public",
        "private",
        "protected",
        "class",
        "interface",
        "void",
        "int",
        "boolean",
        "return",
        "if",
        "else",
        "for",
        "while",
        "new",
        "this",
        "static",
        "final",
        "import",
        "package",
        "try",
        "catch",
        "throw",
        "true",
        "false",
        "null",
    },
    "ruby": {
        "def",
        "end",
        "class",
        "module",
        "return",
        "if",
        "elsif",
        "else",
        "unless",
        "while",
        "for",
        "do",
        "begin",
        "rescue",
        "require",
        "include",
        "true",
        "false",
        "nil",
        "self",
    },
    "c": {
        "void",
        "int",
        "char",
        "float",
        "double",
        "return",
        "if",
        "else",
        "for",
        "while",
        "switch",
        "case",
        "struct",
        "typedef",
        "const",
        "static",
        "sizeof",
        "NULL",
    },
    "cpp": {
        "void",
        "int",
        "char",
        "float",
        "double",
        "return",
        "if",
        "else",
        "for",
        "while",
        "switch",
        "case",
        "class",
        "struct",
        "public",
        "private",
        "protected",
        "const",
        "static",
        "virtual",
        "override",
        "nullptr",
        "new",
        "delete",
        "template",
        "auto",
    },
}

# Literal patterns for string and number extraction
STRING_LITERAL_PATTERN = re.compile(r'(?:"[^"]*"|\'[^\']*\'|`[^`]*`)')
NUMBER_LITERAL_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")


def _get_identifier_pattern(language: str) -> re.Pattern[str]:
    """Get identifier pattern for language, with fallback."""
    return IDENTIFIER_PATTERNS.get(language, IDENTIFIER_PATTERNS["javascript"])


def _get_keywords(language: str) -> set[str]:
    """Get keywords for language, with fallback."""
    return LANGUAGE_KEYWORDS.get(language, set())


def _extract_identifiers(code: str, language: str) -> List[str]:
    """Extract unique identifiers from code, excluding keywords."""
    pattern = _get_identifier_pattern(language)
    keywords = _get_keywords(language)
    matches = pattern.findall(code)
    # Filter out keywords and deduplicate while preserving order
    seen: set[str] = set()
    result: List[str] = []
    for m in matches:
        if m not in keywords and m not in seen:
            seen.add(m)
            result.append(m)
    return result


def _extract_literals(code: str) -> List[str]:
    """Extract string and number literals from code."""
    strings = STRING_LITERAL_PATTERN.findall(code)
    numbers = NUMBER_LITERAL_PATTERN.findall(code)
    return strings + numbers


def _count_ast_depth(ast_output: str) -> int:
    """Estimate AST depth from output indentation."""
    max_indent = 0
    for line in ast_output.split("\n"):
        stripped = line.lstrip()
        if stripped:
            indent = len(line) - len(stripped)
            max_indent = max(max_indent, indent)
    return max_indent // 2  # Rough estimate


def _determine_complexity(ast_output: str, code: str) -> str:
    """Determine code complexity from AST structure."""
    depth = _count_ast_depth(ast_output)
    lines = len(code.strip().split("\n"))

    if depth <= CodeAnalysisDefaults.SIMPLE_CODE_DEPTH_THRESHOLD and lines <= CodeAnalysisDefaults.SIMPLE_CODE_LINES_THRESHOLD:
        return "simple"
    elif depth <= CodeAnalysisDefaults.MEDIUM_CODE_DEPTH_THRESHOLD and lines <= CodeAnalysisDefaults.MEDIUM_CODE_LINES_THRESHOLD:
        return "medium"
    else:
        return "complex"


def _extract_child_kinds(ast_output: str) -> List[str]:
    """Extract child node kinds from AST output."""
    # Look for kind: patterns in the AST output
    kinds = re.findall(r"kind:\s*(\w+)", ast_output)
    # Deduplicate while preserving order
    seen: set[str] = set()
    result: List[str] = []
    for k in kinds:
        if k not in seen:
            seen.add(k)
            result.append(k)
    return result[: DisplayDefaults.MAX_CHILD_KINDS]  # Limit to first 10


def _analyze_code(code: str, language: str, ast_output: str) -> CodeAnalysis:
    """Analyze code structure for pattern development."""
    root_kind = _extract_root_kind(ast_output) or "unknown"
    child_kinds = _extract_child_kinds(ast_output)
    identifiers = _extract_identifiers(code, language)
    literals = _extract_literals(code)
    keywords_found = [k for k in _get_keywords(language) if k in code.split()]
    complexity = _determine_complexity(ast_output, code)

    # Create simplified AST preview
    ast_lines = ast_output.split("\n")[: DisplayDefaults.AST_PREVIEW_MAX_LINES]
    ast_preview = "\n".join(ast_lines)
    if len(ast_output.split("\n")) > DisplayDefaults.AST_PREVIEW_MAX_LINES:
        ast_preview += "\n... (truncated)"

    return CodeAnalysis(
        root_kind=root_kind,
        child_kinds=child_kinds,
        identifiers=identifiers[: DisplayDefaults.MAX_IDENTIFIERS],
        literals=literals[: DisplayDefaults.MAX_LITERALS],
        keywords=keywords_found[: DisplayDefaults.MAX_IDENTIFIERS],
        complexity=complexity,
        ast_preview=ast_preview,
    )


_COMMON_IDENTS = {"name", "id", "value", "result", "data", "item", "obj"}


def _pick_metavar_name(ident: str, used_names: set[str]) -> str:
    """Pick a unique metavariable name for an identifier."""
    if ident.lower() in _COMMON_IDENTS or len(ident) <= DisplayDefaults.SHORT_IDENTIFIER_THRESHOLD:
        base = f"${ident.upper()}"
    else:
        base = f"${ident[:3].upper()}"
    metavar = base
    while metavar in used_names:
        metavar += "1"
    return metavar


def _generate_generalized_pattern(code: str, identifiers: List[str], literals: List[str]) -> str:
    """Generate pattern by replacing identifiers and literals with metavariables."""
    pattern = code

    for i, lit in enumerate(literals[: DisplayDefaults.MAX_PATTERN_REPLACEMENTS]):
        escaped = re.escape(lit)
        metavar = f"$LITERAL{i + 1}" if i > 0 else "$LITERAL"
        pattern = re.sub(escaped, metavar, pattern, count=1)

    used_names: set[str] = set()
    for ident in identifiers[: DisplayDefaults.MAX_PATTERN_IDENTIFIERS]:
        if ident in used_names:
            continue
        metavar = _pick_metavar_name(ident, used_names)
        used_names.add(metavar)
        pattern = re.sub(rf"\b{re.escape(ident)}\b", metavar, pattern)

    return pattern


def _exact_confidence(analysis: CodeAnalysis) -> float:
    """Return confidence level for exact pattern based on code complexity."""
    if analysis.complexity == "simple":
        return PatternSuggestionConfidence.EXACT_SIMPLE
    return PatternSuggestionConfidence.EXACT_COMPLEX


def _generalized_suggestion(code: str, analysis: CodeAnalysis) -> Optional[PatternSuggestion]:
    """Build a generalized pattern suggestion, or None if unchanged."""
    if not (analysis.identifiers or analysis.literals):
        return None
    generalized = _generate_generalized_pattern(code.strip(), analysis.identifiers, analysis.literals)
    if generalized == code.strip():
        return None
    return PatternSuggestion(
        pattern=generalized,
        description="Generalized - matches similar code with different names/values",
        type=SuggestionType.GENERALIZED,
        confidence=PatternSuggestionConfidence.GENERALIZED,
        notes="Metavariables ($NAME) match any identifier. Use $$$ARGS for multiple items.",
    )


def _generate_pattern_suggestions(
    code: str,
    language: str,
    analysis: CodeAnalysis,
) -> List[PatternSuggestion]:
    """Generate pattern suggestions with varying generalization levels."""
    suggestions: List[PatternSuggestion] = [
        PatternSuggestion(
            pattern=code.strip(),
            description="Exact match - matches this specific code only",
            type=SuggestionType.EXACT,
            confidence=_exact_confidence(analysis),
            notes="Good for finding exact duplicates",
        )
    ]

    gen = _generalized_suggestion(code, analysis)
    if gen is not None:
        suggestions.append(gen)

    if analysis.root_kind and analysis.root_kind != "unknown":
        suggestions.append(
            PatternSuggestion(
                pattern=f"# For YAML rule: kind: {analysis.root_kind}",
                description=f"Structural - matches any '{analysis.root_kind}' node",
                type=SuggestionType.STRUCTURAL,
                confidence=PatternSuggestionConfidence.STRUCTURAL,
                notes="Use with find_code_by_rule. Combine with 'has' or 'inside' for precision.",
            )
        )

    return suggestions


def _matched_refinement_steps(analysis: CodeAnalysis) -> List[RefinementStep]:
    """Return refinement steps when pattern already matches."""
    steps: List[RefinementStep] = []
    priority = 1
    if analysis.complexity != "simple":
        steps.append(
            RefinementStep(
                action="Simplify pattern",
                pattern="Focus on the key structural element you want to match",
                explanation="Complex patterns may miss variations. Start simple, add constraints.",
                priority=priority,
            )
        )
        priority += 1
    steps.append(
        RefinementStep(
            action="Add constraints with YAML rule",
            pattern="Use 'inside' with stopBy: end to limit scope",
            explanation="Wrap in YAML rule to add context (e.g., only inside functions)",
            priority=priority,
        )
    )
    return steps


def _unmatched_refinement_steps(code: str, language: str, analysis: CodeAnalysis) -> List[RefinementStep]:
    """Return refinement steps when pattern doesn't match."""
    code_preview = code[: DisplayDefaults.CONTENT_PREVIEW_LENGTH]
    dump_cmd = f'dump_syntax_tree(code="{code_preview}...", language="{language}", format="cst")'
    return [
        RefinementStep(
            action="Check metavariable syntax",
            pattern="Ensure $NAME uses UPPERCASE, use $$$ARGS for multiple items",
            explanation="Common mistake: lowercase metavariables don't work",
            priority=1,
        ),
        RefinementStep(
            action="Use dump_syntax_tree",
            pattern=dump_cmd,
            explanation="Compare pattern AST with code AST to find structural mismatches",
            priority=2,
        ),
        RefinementStep(
            action="Try kind-based matching",
            pattern=f"rule:\n  kind: {analysis.root_kind}",
            explanation="Match by node type instead of exact syntax",
            priority=3,
        ),
    ]


def _generate_refinement_steps(
    code: str,
    language: str,
    analysis: CodeAnalysis,
    pattern_matches: bool,
) -> List[RefinementStep]:
    """Generate steps to refine the pattern."""
    if pattern_matches:
        return _matched_refinement_steps(analysis)
    return _unmatched_refinement_steps(code, language, analysis)


def _generate_yaml_template(pattern: str, language: str, analysis: CodeAnalysis) -> str:
    """Generate a YAML rule template with the pattern."""
    # Escape the pattern for YAML
    if "\n" in pattern:
        pattern_yaml = f"|\n    {pattern.replace(chr(10), chr(10) + '    ')}"
    else:
        pattern_yaml = pattern

    template = f"""id: custom-pattern-rule
language: {language}
message: "Found matching code"
rule:
  pattern: {pattern_yaml}"""

    # Add inside constraint suggestion for complex patterns
    if analysis.complexity != "simple":
        template += f"""
  # Optional: Add context constraint
  # inside:
  #   stopBy: end
  #   kind: {analysis.child_kinds[0] if analysis.child_kinds else "function_declaration"}"""

    return template


_NEXT_STEPS_MATCH = [
    "1. Pattern matches! Use find_code() with this pattern to search your project",
    "2. Consider using build_rule() to add constraints (inside, has, follows)",
    "3. Test with test_match_code_rule() on edge cases before deploying",
]

_NEXT_STEPS_NO_MATCH = [
    "1. Use debug_pattern() to diagnose why the pattern doesn't match",
    "2. Check if pattern is valid parseable code for the target language",
    "3. Use dump_syntax_tree() to compare pattern and code AST structures",
    "4. Try the generalized pattern suggestion with metavariables",
]

_NEXT_STEPS_COMMON = [
    "5. For complex matching, use find_code_by_rule() with the YAML template provided",
    "6. See get_ast_grep_docs(topic='workflow') for the full development workflow",
]


def _generate_next_steps(pattern_matches: bool, analysis: CodeAnalysis) -> List[str]:
    """Generate guidance for next steps."""
    base = _NEXT_STEPS_MATCH if pattern_matches else _NEXT_STEPS_NO_MATCH
    return list(base) + _NEXT_STEPS_COMMON


def _test_pattern_match(pattern: str, code: str, language: str) -> tuple[bool, int]:
    """Test if pattern matches the code."""
    try:
        yaml_rule = f"""
id: develop-pattern-test
language: {language}
rule:
  pattern: |
    {pattern}
"""
        result = run_ast_grep("scan", ["--inline-rules", yaml_rule, "--json", "--stdin"], input_text=code)
        matches = json.loads(result.stdout.strip()) if result.stdout.strip() else []
        return len(matches) > 0, len(matches)
    except Exception:
        return False, 0


def _select_best_pattern(suggestions: List[PatternSuggestion], code: str, language: str) -> tuple[str, bool, int]:
    """Select and test the best pattern from suggestions. Returns (best_pattern, matches, count)."""
    if len(suggestions) >= 2 and suggestions[1].type == SuggestionType.GENERALIZED:
        best_pattern = suggestions[1].pattern
    else:
        best_pattern = suggestions[0].pattern

    pattern_matches, match_count = _test_pattern_match(best_pattern, code, language)
    if not pattern_matches and suggestions:
        exact_pattern = suggestions[0].pattern
        exact_matches, exact_count = _test_pattern_match(exact_pattern, code, language)
        if exact_matches:
            return exact_pattern, exact_matches, exact_count

    return best_pattern, pattern_matches, match_count


def _build_develop_result(
    code: str,
    language: str,
    analysis: CodeAnalysis,
    suggestions: List[PatternSuggestion],
    best_pattern: str,
    pattern_matches: bool,
    match_count: int,
    execution_time: float,
) -> PatternDevelopResult:
    """Build the PatternDevelopResult."""
    return PatternDevelopResult(
        code=code,
        language=language,
        code_analysis=analysis,
        suggested_patterns=suggestions,
        best_pattern=best_pattern,
        pattern_matches=pattern_matches,
        match_count=match_count,
        refinement_steps=_generate_refinement_steps(code, language, analysis, pattern_matches),
        yaml_rule_template=_generate_yaml_template(best_pattern, language, analysis),
        next_steps=_generate_next_steps(pattern_matches, analysis),
        execution_time_ms=int(execution_time * ConversionFactors.MILLISECONDS_PER_SECOND),
    )


def _run_develop_analysis(code: str, language: str, start_time: float, logger: Any) -> PatternDevelopResult:
    """Execute pattern development analysis and return result."""
    code_ast = dump_syntax_tree_impl(code, language, "cst")
    analysis = _analyze_code(code, language, code_ast)
    suggestions = _generate_pattern_suggestions(code, language, analysis)
    best_pattern, pattern_matches, match_count = _select_best_pattern(suggestions, code, language)
    execution_time = time.time() - start_time
    result = _build_develop_result(code, language, analysis, suggestions, best_pattern, pattern_matches, match_count, execution_time)
    logger.info(
        "develop_pattern_completed",
        execution_time_seconds=round(execution_time, FormattingDefaults.ROUNDING_PRECISION),
        pattern_matches=pattern_matches,
        suggestion_count=len(suggestions),
        status="success",
    )
    return result


def develop_pattern_impl(
    code: str,
    language: str,
    goal: Optional[str] = None,
) -> PatternDevelopResult:
    """Analyze code and develop ast-grep patterns to match it."""
    logger = get_logger("search.develop_pattern")
    start_time = time.time()
    logger.info("develop_pattern_started", language=language, code_length=len(code), has_goal=goal is not None)
    with _op_error_handler(
        "develop_pattern",
        logger,
        start_time,
        {"function": "develop_pattern_impl", "language": language, "code_length": len(code)},
    ):
        return _run_develop_analysis(code, language, start_time, logger)
