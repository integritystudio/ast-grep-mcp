"""
Complexity analysis MCP tools.

This module provides MCP tool definitions for code complexity analysis
and Sentry integration testing.
"""

import os
import time
from typing import Any, Callable, Dict, List, Literal

import sentry_sdk
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from ast_grep_mcp.constants import (
    CodeQualityDefaults,
    ComplexityDefaults,
    ComplexityStorageDefaults,
    ConversionFactors,
    FilePatterns,
    FormattingDefaults,
    ParallelProcessing,
)
from ast_grep_mcp.core.logging import get_logger
from ast_grep_mcp.models.complexity import ComplexityThresholds
from ast_grep_mcp.utils.tool_context import ToolRun, tool_context

from .complexity_analyzer import ParallelComplexityAnalyzer
from .complexity_file_finder import ComplexityFileFinder
from .complexity_statistics import ComplexityStatisticsAggregator

# Note: detect_code_smells_impl is imported inside detect_code_smells_tool()
# to avoid circular import (quality.smells imports from complexity.analyzer)

_COMPLEXITY_EXCLUDE_DEFAULTS = ["**/node_modules/**", "**/__pycache__/**", "**/venv/**", "**/.venv/**", "**/site-packages/**"]
_SMELLS_EXCLUDE_DEFAULTS = [*_COMPLEXITY_EXCLUDE_DEFAULTS, "**/test*/**", "**/*test*"]


# Helper functions extracted from analyze_complexity_tool


def _validate_inputs(language: str) -> None:
    """Validate input parameters for complexity analysis.

    Args:
        language: The programming language to validate

    Raises:
        ValueError: If the language is not supported
    """
    supported_langs = ["python", "typescript", "javascript", "java"]
    if language.lower() not in supported_langs:
        raise ValueError(f"Unsupported language '{language}'. Supported: {', '.join(supported_langs)}")


def _normalize_complexity_exclude_patterns(exclude_patterns: List[str] | None) -> List[str]:
    """Normalize exclude patterns and enforce virtualenv exclusions."""
    return FilePatterns.normalize_excludes(exclude_patterns, defaults=_COMPLEXITY_EXCLUDE_DEFAULTS)


def _find_files_to_analyze(
    project_folder: str, language: str, include_patterns: List[str], exclude_patterns: List[str], logger: Any
) -> tuple[List[str], ComplexityFileFinder]:
    """Find files to analyze based on patterns.

    Args:
        project_folder: The project folder to analyze
        language: The programming language
        include_patterns: Glob patterns for files to include
        exclude_patterns: Glob patterns for files to exclude
        logger: Logger instance

    Returns:
        Tuple of (files to analyze, file finder instance)
    """
    file_finder = ComplexityFileFinder()
    files_to_analyze = file_finder.find_files(project_folder, language, include_patterns, exclude_patterns)

    logger.info("files_found", total_files=len(files_to_analyze), include_patterns=include_patterns, exclude_patterns=exclude_patterns)

    return files_to_analyze, file_finder


def _analyze_files_parallel(
    files_to_analyze: List[str], language: str, thresholds: ComplexityThresholds, max_threads: int
) -> tuple[List[Any], List[Any], ParallelComplexityAnalyzer]:
    """Analyze files in parallel for complexity metrics.

    Args:
        files_to_analyze: List of files to analyze
        language: The programming language
        thresholds: Complexity thresholds
        max_threads: Number of parallel threads

    Returns:
        Tuple of (all functions, exceeding functions, analyzer instance)
    """
    analyzer = ParallelComplexityAnalyzer()

    # Analyze files in parallel
    all_functions = analyzer.analyze_files(files_to_analyze, language, thresholds, max_threads)

    # Filter exceeding functions
    exceeding_functions = analyzer.filter_exceeding_functions(all_functions)

    return all_functions, exceeding_functions, analyzer


def _calculate_summary_statistics(
    all_functions: List[Any], exceeding_functions: List[Any], total_files: int, execution_time: float
) -> tuple[Dict[str, Any], ComplexityStatisticsAggregator]:
    """Calculate summary statistics from analysis results.

    Args:
        all_functions: All analyzed functions
        exceeding_functions: Functions exceeding thresholds
        total_files: Total number of files analyzed
        execution_time: Analysis execution time

    Returns:
        Tuple of (summary dictionary, statistics instance)
    """
    statistics = ComplexityStatisticsAggregator()
    summary = statistics.calculate_summary(all_functions, exceeding_functions, total_files, execution_time)

    return summary, statistics


def _store_and_generate_trends(
    store_results: bool,
    include_trends: bool,
    project_folder: str,
    summary: Dict[str, Any],
    all_functions: List[Any],
    statistics: ComplexityStatisticsAggregator,
) -> tuple[Any, Any, Any]:
    """Store results and generate trends if requested.

    Args:
        store_results: Whether to store results
        include_trends: Whether to include trends
        project_folder: The project folder
        summary: Summary statistics
        all_functions: All analyzed functions
        statistics: Statistics aggregator instance

    Returns:
        Tuple of (run_id, stored_at, trends)
    """
    run_id = None
    stored_at = None
    trends = None

    if store_results:
        run_id, stored_at = statistics.store_results(project_folder, summary, all_functions)

    if include_trends:
        trends = statistics.get_trends(project_folder, days=ComplexityStorageDefaults.TRENDS_LOOKBACK_DAYS)

    return run_id, stored_at, trends


def _format_response(
    summary: Dict[str, Any],
    thresholds_dict: Dict[str, int],
    exceeding_functions: List[Any],
    run_id: Any,
    stored_at: Any,
    trends: Any,
    statistics: ComplexityStatisticsAggregator,
) -> Dict[str, Any]:
    """Format the final response dictionary.

    Args:
        summary: Summary statistics
        thresholds_dict: Complexity thresholds used
        exceeding_functions: Functions exceeding thresholds
        run_id: Storage run ID
        stored_at: Storage location
        trends: Trend data
        statistics: Statistics aggregator instance

    Returns:
        Formatted response dictionary
    """
    return statistics.format_response(summary, thresholds_dict, exceeding_functions, run_id, stored_at, trends)


def _handle_no_files_found(language: str, execution_time: float) -> Dict[str, Any]:
    """Handle the case when no files are found to analyze.

    Args:
        language: The programming language
        execution_time: Time taken for the analysis attempt

    Returns:
        Response dictionary for no files found case
    """
    return {
        "summary": {
            "total_functions": 0,
            "total_files": 0,
            "exceeding_threshold": 0,
            "analysis_time_seconds": round(execution_time, FormattingDefaults.ROUNDING_PRECISION),
        },
        "functions": [],
        "message": f"No {language} files found in project matching the include patterns",
    }


def _thresholds_to_dict(thresholds: ComplexityThresholds) -> Dict[str, int]:
    return {
        "cyclomatic": thresholds.cyclomatic,
        "cognitive": thresholds.cognitive,
        "nesting_depth": thresholds.nesting_depth,
        "length": thresholds.lines,
    }


def _execute_analysis(
    project_folder: str,
    language: str,
    thresholds: ComplexityThresholds,
    files_to_analyze: List[str],
    store_results: bool,
    include_trends: bool,
    max_threads: int,
    run: ToolRun,
) -> Dict[str, Any]:
    all_functions, exceeding_functions, _ = _analyze_files_parallel(files_to_analyze, language, thresholds, max_threads)
    execution_time = time.time() - run.start_time
    summary, statistics = _calculate_summary_statistics(all_functions, exceeding_functions, len(files_to_analyze), execution_time)
    run_id, stored_at, trends = _store_and_generate_trends(
        store_results, include_trends, project_folder, summary, all_functions, statistics
    )
    run.add_completion_fields(
        total_functions=summary["total_functions"],
        exceeding_threshold=len(exceeding_functions),
    )
    return _format_response(summary, _thresholds_to_dict(thresholds), exceeding_functions, run_id, stored_at, trends, statistics)


def analyze_complexity_tool(
    project_folder: str,
    language: str,
    include_patterns: List[str] | None = None,
    exclude_patterns: List[str] | None = None,
    cyclomatic_threshold: int = ComplexityDefaults.CYCLOMATIC_THRESHOLD,
    cognitive_threshold: int = ComplexityDefaults.COGNITIVE_THRESHOLD,
    nesting_threshold: int = ComplexityDefaults.NESTING_THRESHOLD,
    length_threshold: int = ComplexityDefaults.LENGTH_THRESHOLD,
    store_results: bool = True,
    include_trends: bool = False,
    max_threads: int = ParallelProcessing.DEFAULT_WORKERS,
) -> Dict[str, Any]:
    """Analyze cyclomatic, cognitive, nesting, and length complexity for all functions in a project.

    Returns summary with only functions exceeding configured thresholds.
    """
    if include_patterns is None:
        include_patterns = ["**/*"]
    exclude_patterns = _normalize_complexity_exclude_patterns(exclude_patterns)
    logger = get_logger("tool.analyze_complexity")
    logger.info(
        "tool_invoked",
        tool="analyze_complexity",
        project_folder=project_folder,
        language=language,
        cyclomatic_threshold=cyclomatic_threshold,
        cognitive_threshold=cognitive_threshold,
        nesting_threshold=nesting_threshold,
        length_threshold=length_threshold,
        max_threads=max_threads,
    )

    with tool_context("analyze_complexity", project_folder=project_folder, language=language) as run:
        _validate_inputs(language)
        thresholds = ComplexityThresholds(
            cyclomatic=cyclomatic_threshold, cognitive=cognitive_threshold, nesting_depth=nesting_threshold, lines=length_threshold
        )
        files_to_analyze, _ = _find_files_to_analyze(project_folder, language, include_patterns, exclude_patterns, logger)
        if not files_to_analyze:
            return _handle_no_files_found(language, time.time() - run.start_time)
        return _execute_analysis(
            project_folder, language, thresholds, files_to_analyze, store_results, include_trends, max_threads, run
        )


def _sentry_test_error(message: str, result: Dict[str, Any]) -> None:
    """Trigger a test exception for Sentry."""
    try:
        raise ValueError(f"Sentry integration test error: {message}")
    except ValueError as e:
        sentry_sdk.capture_exception(e, extras={"test": True, "tool": "test_sentry_integration", "message": message})
        result["message"] = "Test exception captured and sent to Sentry"
        result["exception_type"] = "ValueError"


def _sentry_test_warning(message: str, result: Dict[str, Any]) -> None:
    """Send a test warning message to Sentry."""
    sentry_sdk.capture_message(
        f"Sentry integration test warning: {message}", level="warning", extras={"test": True, "tool": "test_sentry_integration"}
    )
    result["message"] = "Test warning message sent to Sentry"


def _sentry_test_breadcrumb(message: str, result: Dict[str, Any]) -> None:
    """Add test breadcrumbs and send to Sentry."""
    sentry_sdk.add_breadcrumb(
        message=f"Test breadcrumb 1: {message}", category="test.breadcrumb", level="info", data={"test": True, "sequence": 1}
    )
    sentry_sdk.add_breadcrumb(
        message="Test breadcrumb 2: Sequence item", category="test.breadcrumb", level="info", data={"test": True, "sequence": 2}
    )
    sentry_sdk.capture_message(
        "Test breadcrumb context (check breadcrumb trail)", level="info", extras={"test": True, "tool": "test_sentry_integration"}
    )
    result["message"] = "Test breadcrumbs added and sent to Sentry (check breadcrumb trail in event)"
    result["breadcrumb_count"] = 2


def _sentry_test_span(message: str, result: Dict[str, Any]) -> None:
    """Create a test performance span in Sentry."""
    with sentry_sdk.start_span(op="test.operation", name=f"Test span: {message}") as span:
        span.set_data("test", True)
        span.set_data("message", message)
        span.set_data("tool", "test_sentry_integration")
        time.sleep(0.1)
    sentry_sdk.capture_message(
        "Test span completed (check performance monitoring)", level="info", extras={"test": True, "tool": "test_sentry_integration"}
    )
    result["message"] = "Test performance span created and sent to Sentry"


_SENTRY_TEST_HANDLERS: Dict[str, Callable[[str, Dict[str, Any]], None]] = {
    "error": _sentry_test_error,
    "warning": _sentry_test_warning,
    "breadcrumb": _sentry_test_breadcrumb,
    "span": _sentry_test_span,
}


def test_sentry_integration_tool(
    test_type: Literal["error", "warning", "breadcrumb", "span"] = "breadcrumb", message: str = "Test message"
) -> Dict[str, Any]:
    """Test Sentry integration by triggering different event types.

    Args:
        test_type: Type of Sentry test ('error', 'warning', 'breadcrumb', or 'span')
        message: Custom test message

    Returns:
        Information about what was sent to Sentry
    """
    logger = get_logger("tool.test_sentry_integration")
    logger.info("tool_invoked", tool="test_sentry_integration", test_type=test_type)

    with tool_context("test_sentry_integration", test_type=test_type) as run:
        if not os.getenv("SENTRY_DSN"):
            return {"status": "skipped", "message": "Sentry not configured (SENTRY_DSN not set)", "test_type": test_type}
        result: Dict[str, Any] = {"status": "success", "test_type": test_type}
        _SENTRY_TEST_HANDLERS[test_type](message, result)
        run.add_completion_fields(test_type=test_type)
        result["execution_time_seconds"] = round(time.time() - run.start_time, FormattingDefaults.ROUNDING_PRECISION)
        result["sentry_configured"] = True
        return result


_SMELL_EXCLUDE_DEFAULTS = FilePatterns.DEFAULT_EXCLUDE + FilePatterns.TEST_EXCLUDE


def _prepare_smell_detection_params(include_patterns: List[str] | None, exclude_patterns: List[str] | None) -> tuple[List[str], List[str]]:
    """Prepare and validate parameters for smell detection.

    Args:
        include_patterns: Glob patterns for files to include
        exclude_patterns: Glob patterns for files to exclude

    Returns:
        Tuple of (include_patterns, exclude_patterns) with defaults applied
    """
    if include_patterns is None:
        include_patterns = ["**/*"]
    exclude_patterns = FilePatterns.normalize_excludes(exclude_patterns, defaults=_SMELL_EXCLUDE_DEFAULTS)
    return include_patterns, exclude_patterns


def _process_smell_detection_result(result: Dict[str, Any], run: ToolRun) -> Dict[str, Any]:
    """Add execution time and attach completion metrics.

    Args:
        result: Smell detection result dictionary
        run: ToolRun handle from the enclosing tool_context

    Returns:
        Result dictionary with execution_time_ms added
    """
    result["execution_time_ms"] = round((time.time() - run.start_time) * ConversionFactors.MILLISECONDS_PER_SECOND)
    run.add_completion_fields(
        files_analyzed=result.get("files_analyzed", 0),
        total_smells=result.get("total_smells", 0),
    )
    return result


def detect_code_smells_tool(
    project_folder: str,
    language: str,
    include_patterns: List[str] | None = None,
    exclude_patterns: List[str] | None = None,
    long_function_lines: int = CodeQualityDefaults.LONG_FUNCTION_LINES,
    parameter_count: int = CodeQualityDefaults.PARAMETER_COUNT,
    nesting_depth: int = CodeQualityDefaults.NESTING_DEPTH,
    class_lines: int = CodeQualityDefaults.CLASS_LINES,
    class_methods: int = CodeQualityDefaults.CLASS_METHODS,
    detect_magic_numbers: bool = True,
    severity_filter: str = "all",
    max_threads: int = ParallelProcessing.DEFAULT_WORKERS,
) -> Dict[str, Any]:
    """Detect long functions, parameter bloat, deep nesting, large classes, and magic numbers."""
    # Import here to avoid circular import with quality.smells
    from ast_grep_mcp.features.quality.smells import detect_code_smells_impl

    logger = get_logger("tool.detect_code_smells")
    include_patterns, exclude_patterns = _prepare_smell_detection_params(include_patterns, exclude_patterns)
    logger.info("tool_invoked", tool="detect_code_smells", project_folder=project_folder, language=language)

    with tool_context("detect_code_smells", project_folder=project_folder, language=language) as run:
        result = detect_code_smells_impl(
            project_folder=project_folder,
            language=language,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            long_function_lines=long_function_lines,
            parameter_count=parameter_count,
            nesting_depth=nesting_depth,
            class_lines=class_lines,
            class_methods=class_methods,
            detect_magic_numbers=detect_magic_numbers,
            severity_filter=severity_filter,
            max_threads=max_threads,
        )
        return _process_smell_detection_result(result, run)


def _register_analyze_complexity(mcp: FastMCP) -> None:
    @mcp.tool()
    def analyze_complexity(
        project_folder: str = Field(description="The absolute path to the project folder to analyze"),
        language: str = Field(description="The programming language (python, typescript, javascript, java)"),
        include_patterns: List[str] = Field(default_factory=lambda: ["**/*"], description="Glob patterns for files to include"),
        exclude_patterns: List[str] = Field(
            default_factory=lambda: list(_COMPLEXITY_EXCLUDE_DEFAULTS),
            description="Glob patterns for files to exclude",
        ),
        cyclomatic_threshold: int = Field(default=ComplexityDefaults.CYCLOMATIC_THRESHOLD, description="Cyclomatic complexity threshold"),
        cognitive_threshold: int = Field(default=ComplexityDefaults.COGNITIVE_THRESHOLD, description="Cognitive complexity threshold"),
        nesting_threshold: int = Field(default=ComplexityDefaults.NESTING_THRESHOLD, description="Maximum nesting depth threshold"),
        length_threshold: int = Field(default=ComplexityDefaults.LENGTH_THRESHOLD, description="Function length threshold in lines"),
        store_results: bool = Field(default=True, description="Store results in database for trend tracking"),
        include_trends: bool = Field(default=False, description="Include historical trend data in response"),
        max_threads: int = Field(default=ParallelProcessing.DEFAULT_WORKERS, description="Number of parallel threads for analysis"),
    ) -> Dict[str, Any]:
        """Analyze cyclomatic, cognitive, nesting, and length complexity for all functions in a project."""
        return analyze_complexity_tool(
            project_folder=project_folder,
            language=language,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            cyclomatic_threshold=cyclomatic_threshold,
            cognitive_threshold=cognitive_threshold,
            nesting_threshold=nesting_threshold,
            length_threshold=length_threshold,
            store_results=store_results,
            include_trends=include_trends,
            max_threads=max_threads,
        )


def _register_test_sentry(mcp: FastMCP) -> None:
    @mcp.tool()
    def test_sentry_integration(
        test_type: Literal["error", "warning", "breadcrumb", "span"] = Field(
            default="breadcrumb",
            description="Type of Sentry test: 'error' (exception), 'warning' (capture_message), 'breadcrumb', or 'span' (performance)",
        ),
        message: str = Field(default="Test message", description="Custom test message"),
    ) -> Dict[str, Any]:
        """Test Sentry integration by triggering different event types."""
        return test_sentry_integration_tool(test_type=test_type, message=message)


def _register_detect_smells(mcp: FastMCP) -> None:
    @mcp.tool()
    def detect_code_smells(
        project_folder: str = Field(description="The absolute path to the project folder to analyze"),
        language: str = Field(description="The programming language (python, typescript, javascript, java)"),
        include_patterns: List[str] = Field(default_factory=lambda: ["**/*"], description="Glob patterns for files to include"),
        exclude_patterns: List[str] = Field(
            default_factory=lambda: list(_SMELLS_EXCLUDE_DEFAULTS),
            description="Glob patterns for files to exclude",
        ),
        long_function_lines: int = Field(
            default=CodeQualityDefaults.LONG_FUNCTION_LINES,
            description="Line count threshold for long function smell",
        ),
        parameter_count: int = Field(
            default=CodeQualityDefaults.PARAMETER_COUNT,
            description="Parameter count threshold for parameter bloat",
        ),
        nesting_depth: int = Field(default=CodeQualityDefaults.NESTING_DEPTH, description="Nesting depth threshold for deep nesting smell"),
        class_lines: int = Field(default=CodeQualityDefaults.CLASS_LINES, description="Line count threshold for large class smell"),
        class_methods: int = Field(default=CodeQualityDefaults.CLASS_METHODS, description="Method count threshold for large class smell"),
        detect_magic_numbers: bool = Field(default=True, description="Whether to detect magic number smells"),
        severity_filter: str = Field(default="all", description="Filter by severity: 'all', 'high', 'medium', 'low'"),
        max_threads: int = Field(default=ParallelProcessing.DEFAULT_WORKERS, description="Number of parallel threads for analysis"),
    ) -> Dict[str, Any]:
        """Detect long functions, parameter bloat, deep nesting, large classes, and magic numbers."""
        return detect_code_smells_tool(
            project_folder=project_folder,
            language=language,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            long_function_lines=long_function_lines,
            parameter_count=parameter_count,
            nesting_depth=nesting_depth,
            class_lines=class_lines,
            class_methods=class_methods,
            detect_magic_numbers=detect_magic_numbers,
            severity_filter=severity_filter,
            max_threads=max_threads,
        )


def register_complexity_tools(mcp: FastMCP) -> None:
    """Register complexity analysis tools with the MCP server."""
    _register_analyze_complexity(mcp)
    _register_test_sentry(mcp)
    _register_detect_smells(mcp)
