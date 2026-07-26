"""Documentation sync checker service.

This module provides functionality for keeping documentation
synchronized with code changes.
"""

import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import sentry_sdk

from ast_grep_mcp.constants import ConversionFactors, FilePatterns, RegexCaptureGroups, SeverityRankingDefaults
from ast_grep_mcp.core.logging import get_logger
from ast_grep_mcp.models.documentation import (
    DocSyncIssue,
    DocSyncResult,
    FunctionSignature,
)
from ast_grep_mcp.utils.file_discovery import find_source_files
from ast_grep_mcp.utils.text import read_file_lines

from .docstring_generator import FunctionSignatureParser

logger = get_logger(__name__)


# =============================================================================
# Docstring Analysis
# =============================================================================


def _extract_google_style_params(docstring: str) -> List[str]:
    args_match = re.search(r"Args:\s*\n((?:\s+\w+.*\n?)+)", docstring)
    if not args_match:
        return []
    return [
        m.group(RegexCaptureGroups.FIRST)
        for line in args_match.group(RegexCaptureGroups.FIRST).split("\n")
        if (m := re.match(r"\s+(\w+)(?:\s*\(|\s*:)", line))
    ]


def _extract_numpy_style_params(docstring: str) -> List[str]:
    params_match = re.search(r"Parameters\s*\n-+\s*\n((?:.*\n?)+?)(?:\n\w|\Z)", docstring)
    if not params_match:
        return []
    return [
        m.group(RegexCaptureGroups.FIRST)
        for line in params_match.group(RegexCaptureGroups.FIRST).split("\n")
        if (m := re.match(r"(\w+)\s*:", line))
    ]


def _extract_python_docstring_params(docstring: str) -> List[str]:
    """Extract parameter names from Python docstrings.

    Supports Google, NumPy, and Sphinx styles.

    Args:
        docstring: Docstring content

    Returns:
        List of parameter names
    """
    params = _extract_google_style_params(docstring) + _extract_numpy_style_params(docstring)
    params += [m.group(RegexCaptureGroups.FIRST) for m in re.finditer(r":param\s+(\w+):", docstring)]
    return params


def _extract_js_docstring_params(docstring: str) -> List[str]:
    """Extract parameter names from JSDoc/Javadoc docstrings.

    Args:
        docstring: Docstring content

    Returns:
        List of parameter names
    """
    return [m.group(RegexCaptureGroups.FIRST) for m in re.finditer(r"@param\s+(?:\{(?:[^{}]|\{[^}]*\})*\}\s+)?\[?(\w+)", docstring)]


def _extract_docstring_params(docstring: str, language: str) -> List[str]:
    """Extract documented parameter names from a docstring.

    Args:
        docstring: Docstring content
        language: Programming language

    Returns:
        List of documented parameter names
    """
    if language == "python":
        return _extract_python_docstring_params(docstring)
    elif language in ("typescript", "javascript", "java"):
        return _extract_js_docstring_params(docstring)
    return []


_RETURN_PATTERNS: Dict[str, List[str]] = {
    "python": [r"Returns:\s*\n", r"Returns\s*\n-+", r":returns?:"],
    "typescript": [r"@returns?\s"],
    "javascript": [r"@returns?\s"],
    "java": [r"@return\s"],
}


def _extract_docstring_return(docstring: str, language: str) -> bool:
    """Check if docstring documents a return value.

    Args:
        docstring: Docstring content
        language: Programming language

    Returns:
        True if return is documented
    """
    return any(re.search(p, docstring) for p in _RETURN_PATTERNS.get(language, []))


def _make_issue(
    func: FunctionSignature,
    issue_type: str,
    description: str,
    severity: str,
    suggested_fix: Optional[str] = None,
) -> DocSyncIssue:
    return DocSyncIssue(
        issue_type=issue_type,
        file_path=func.file_path,
        line_number=func.start_line,
        function_name=func.name,
        description=description,
        suggested_fix=suggested_fix,
        severity=severity,
    )


def _check_param_sync(func: FunctionSignature, language: str) -> List[DocSyncIssue]:
    doc_params = set(_extract_docstring_params(func.existing_docstring, language))  # type: ignore[arg-type]
    actual_params = {(p.name[:-1] if p.name.endswith("?") else p.name) for p in func.parameters if p.name not in ("self", "cls")}
    issues = [
        _make_issue(func, "mismatch", f"Parameter '{p}' is not documented", "info", f"Add documentation for parameter '{p}'")
        for p in (actual_params - doc_params)
    ]
    issues += [
        _make_issue(
            func,
            "stale",
            f"Documented parameter '{p}' does not exist in function signature",
            "warning",
            f"Remove documentation for parameter '{p}'",
        )
        for p in (doc_params - actual_params)
    ]
    return issues


def _check_return_sync(func: FunctionSignature, language: str) -> List[DocSyncIssue]:
    has_return = func.return_type and func.return_type.lower() not in ("none", "void")
    doc_has_return = _extract_docstring_return(func.existing_docstring, language)  # type: ignore[arg-type]
    if has_return and not doc_has_return:
        msg = f"Return value of type '{func.return_type}' is not documented"
        return [_make_issue(func, "mismatch", msg, "info", "Add return value documentation")]
    if not has_return and doc_has_return:
        msg = "Documented return value but function has no return type"
        fix = "Remove return value documentation or add return type annotation"
        return [_make_issue(func, "stale", msg, "info", fix)]
    return []


def _check_docstring_sync(
    func: FunctionSignature,
    language: str,
) -> List[DocSyncIssue]:
    """Check if a function's docstring is in sync with its signature.

    Args:
        func: Function signature
        language: Programming language

    Returns:
        List of issues found
    """
    if not func.existing_docstring:
        return [_make_issue(func, "undocumented", f"Function '{func.name}' is missing documentation", "warning")]
    return _check_param_sync(func, language) + _check_return_sync(func, language)


# =============================================================================
# Link Checking
# =============================================================================


_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
# Skip external schemes, fragments, and absolute-path URLs (web-root convention, not repo-relative).
_SKIP_URL_PREFIXES = ("http://", "https://", "mailto:", "#", "/", "~")


def _resolve_link_path(url: str, file_path: str, project_folder: str) -> str:
    return os.path.normpath(os.path.join(os.path.dirname(file_path), url.split("#")[0]))


def _check_line_links(line: str, line_num: int, file_path: str, project_folder: str) -> List[DocSyncIssue]:
    issues: List[DocSyncIssue] = []
    for match in _LINK_PATTERN.finditer(line):
        url = match.group(RegexCaptureGroups.SECOND)
        if url.startswith(_SKIP_URL_PREFIXES):
            continue
        full_path = _resolve_link_path(url, file_path, project_folder)
        if not os.path.exists(full_path):
            issues.append(
                DocSyncIssue(
                    issue_type="broken_link",
                    file_path=file_path,
                    line_number=line_num,
                    description=f"Broken link to '{url}'",
                    suggested_fix=f"Update or remove link to '{url}'",
                    severity="warning",
                )
            )
    return issues


def _check_markdown_links(file_path: str, project_folder: str) -> List[DocSyncIssue]:
    """Check for broken links in markdown files.

    Args:
        file_path: Path to markdown file
        project_folder: Project root

    Returns:
        List of broken link issues
    """
    try:
        lines = read_file_lines(file_path)
    except OSError:
        return []

    issues: List[DocSyncIssue] = []
    for i, line in enumerate(lines):
        issues.extend(_check_line_links(line, i + 1, file_path, project_folder))
    return issues


# =============================================================================
# Main Sync Check
# =============================================================================


_DEFAULT_SOURCE_PATTERNS: Dict[str, List[str]] = {
    "python": ["**/*.py"],
    "typescript": ["**/*.ts", "**/*.tsx"],
    "javascript": ["**/*.js", "**/*.jsx"],
    "java": ["**/*.java"],
}



def _resolve_include_patterns(language: str, include_patterns: List[str]) -> List[str]:
    if not include_patterns or include_patterns == ["all"]:
        return _DEFAULT_SOURCE_PATTERNS.get(language, ["**/*"])
    return include_patterns


def _is_excluded(path: str, project_folder: str, exclude_patterns: List[str]) -> bool:
    rel_path = os.path.relpath(path, project_folder)
    return any(pattern.replace("**/", "").replace("/**", "") in rel_path for pattern in exclude_patterns)


def _find_source_files(
    project_folder: str,
    language: str,
    include_patterns: List[str],
    exclude_patterns: List[str],
) -> List[str]:
    """Find source files to check.

    Args:
        project_folder: Project root
        language: Programming language
        include_patterns: Glob patterns to include
        exclude_patterns: Glob patterns to exclude

    Returns:
        List of file paths
    """
    import glob

    include_patterns = _resolve_include_patterns(language, include_patterns)
    exclude_patterns = FilePatterns.normalize_excludes(exclude_patterns)

    files: List[str] = []
    for pattern in include_patterns:
        files.extend(glob.glob(os.path.join(project_folder, pattern), recursive=True))

    return [f for f in set(files) if not _is_excluded(f, project_folder, exclude_patterns)]


def _find_markdown_files(project_folder: str) -> List[str]:
    """Find markdown documentation files.

    Args:
        project_folder: Project root

    Returns:
        List of markdown file paths
    """
    return [
        str(f)
        for f in find_source_files(
            project_folder,
            "markdown",
            extensions=[".md", ".markdown"],
            skip_dir_names=FilePatterns.SKIP_DIR_NAMES,
        )
    ]


def _check_function_docstring(
    func: FunctionSignature,
    language: str,
) -> Tuple[List[DocSyncIssue], bool, bool, Optional[Dict[str, Any]]]:
    """Check a single function's docstring.

    Args:
        func: Function signature
        language: Programming language

    Returns:
        Tuple of (issues, is_documented, is_stale, suggestion)
    """
    # Skip private functions (single underscore prefix)
    if func.name.startswith("_") and not func.name.startswith("__"):
        return [], False, False, None

    issues = _check_docstring_sync(func, language)
    is_documented = bool(func.existing_docstring)
    is_stale = is_documented and any(i.issue_type == "stale" for i in issues)

    suggestion = None
    if not func.existing_docstring:
        suggestion = {
            "file": func.file_path,
            "line": func.start_line,
            "function": func.name,
            "action": "add_docstring",
        }

    return issues, is_documented, is_stale, suggestion


def _accumulate_func_result(
    issues: List[DocSyncIssue],
    is_documented: bool,
    is_stale: bool,
    suggestion: Optional[Dict[str, Any]],
    all_issues: List[DocSyncIssue],
    suggestions: List[Dict[str, Any]],
) -> Tuple[int, int, int]:
    all_issues.extend(issues)
    if is_documented:
        return 1, 0, 1 if is_stale else 0
    if suggestion:
        suggestions.append(suggestion)
        return 0, 1, 0
    return 0, 0, 0


def _process_file_docstrings(
    file_path: str,
    language: str,
    parser: Any,
    all_issues: List[DocSyncIssue],
    suggestions: List[Dict[str, Any]],
) -> Tuple[int, int, int, int]:
    functions = parser.parse_file(file_path)
    total = len(functions)
    documented = undocumented = stale = 0
    for func in functions:
        doc_delta, undoc_delta, stale_delta = _accumulate_func_result(*_check_function_docstring(func, language), all_issues, suggestions)
        documented += doc_delta
        undocumented += undoc_delta
        stale += stale_delta
    return total, documented, undocumented, stale


def _check_docstrings_in_files(
    project_folder: str,
    language: str,
    include_patterns: List[str],
    exclude_patterns: List[str],
) -> Tuple[List[DocSyncIssue], int, int, int, int, List[Dict[str, Any]]]:
    """Check docstrings across source files.

    Returns:
        Tuple of (issues, total, documented, undocumented, stale, suggestions)
    """
    source_files = _find_source_files(project_folder, language, include_patterns, exclude_patterns)
    parser = FunctionSignatureParser(language)

    all_issues: List[DocSyncIssue] = []
    total_functions = documented_functions = undocumented_functions = stale_docstrings = 0
    suggestions: List[Dict[str, Any]] = []

    for file_path in source_files:
        try:
            total, doc, undoc, stale = _process_file_docstrings(file_path, language, parser, all_issues, suggestions)
            total_functions += total
            documented_functions += doc
            undocumented_functions += undoc
            stale_docstrings += stale
        except Exception as e:
            logger.warning("file_parse_error", file=file_path, error=str(e))
            sentry_sdk.capture_exception(e)

    return all_issues, total_functions, documented_functions, undocumented_functions, stale_docstrings, suggestions


def _check_markdown_link_issues(project_folder: str) -> List[DocSyncIssue]:
    """Check all markdown files for broken links.

    Args:
        project_folder: Project root

    Returns:
        List of broken link issues
    """
    all_issues: List[DocSyncIssue] = []
    md_files = _find_markdown_files(project_folder)

    for file_path in md_files:
        try:
            link_issues = _check_markdown_links(file_path, project_folder)
            all_issues.extend(link_issues)
        except Exception as e:
            logger.warning("link_check_error", file=file_path, error=str(e))
            sentry_sdk.capture_exception(e)

    return all_issues


def _collect_issues(
    project_folder: str,
    language: str,
    doc_types: List[str],
    include_patterns: List[str],
    exclude_patterns: List[str],
) -> Tuple[List[DocSyncIssue], int, int, int, int, List[Dict[str, Any]]]:
    all_issues: List[DocSyncIssue] = []
    total_functions = documented_functions = undocumented_functions = stale_docstrings = 0
    suggestions: List[Dict[str, Any]] = []

    if "all" in doc_types or "docstrings" in doc_types:
        issues, total_functions, documented_functions, undocumented_functions, stale_docstrings, suggestions = _check_docstrings_in_files(
            project_folder, language, include_patterns, exclude_patterns
        )
        all_issues.extend(issues)

    if "all" in doc_types or "links" in doc_types:
        all_issues.extend(_check_markdown_link_issues(project_folder))

    severity_order = SeverityRankingDefaults.DOC_SYNC_SORT_ORDER
    all_issues.sort(key=lambda i: (severity_order.get(i.severity, SeverityRankingDefaults.FALLBACK_RANK), i.file_path, i.line_number))
    return all_issues, total_functions, documented_functions, undocumented_functions, stale_docstrings, suggestions


def sync_documentation_impl(
    project_folder: str,
    language: str,
    doc_types: Optional[List[str]] = None,
    check_only: bool = True,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> DocSyncResult:
    """Synchronize documentation with code.

    Args:
        project_folder: Root folder of the project
        language: Programming language
        doc_types: Types to check ('docstrings', 'links', 'all')
        check_only: If True, only check without making changes
        include_patterns: File patterns to include
        exclude_patterns: File patterns to exclude

    Returns:
        DocSyncResult with sync status
    """
    start_time = time.time()
    doc_types = doc_types or ["all"]
    include_patterns = include_patterns or ["all"]
    exclude_patterns = exclude_patterns or []

    logger.info("sync_documentation_started", project_folder=project_folder, language=language, doc_types=doc_types, check_only=check_only)

    all_issues, total_functions, documented_functions, undocumented_functions, stale_docstrings, suggestions = _collect_issues(
        project_folder, language, doc_types, include_patterns, exclude_patterns
    )

    execution_time = int((time.time() - start_time) * ConversionFactors.MILLISECONDS_PER_SECOND)
    logger.info(
        "sync_documentation_completed",
        total_functions=total_functions,
        documented=documented_functions,
        undocumented=undocumented_functions,
        stale=stale_docstrings,
        issues_found=len(all_issues),
        execution_time_ms=execution_time,
    )

    return DocSyncResult(
        total_functions=total_functions,
        documented_functions=documented_functions,
        undocumented_functions=undocumented_functions,
        stale_docstrings=stale_docstrings,
        issues=all_issues,
        suggestions=suggestions,
        files_updated=[],
        check_only=check_only,
        execution_time_ms=execution_time,
    )
