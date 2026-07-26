"""MCP tool wrappers for deduplication features.

This module provides the high-level tool interfaces exposed via MCP.
These functions wrap the underlying deduplication modules to provide
a clean API for the MCP server.
"""

from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from ...constants import DeduplicationDefaults, FilePatterns
from ...core.logging import get_logger
from .analysis_orchestrator import DeduplicationAnalysisOrchestrator
from .applicator import DeduplicationApplicator
from .benchmark import DeduplicationBenchmark
from .detector import DuplicationDetector
from .similarity import (
    HybridSimilarity,
    SemanticSimilarity,
    SemanticSimilarityConfig,
)


def find_duplication_tool(
    project_folder: str,
    language: str,
    min_similarity: float = DeduplicationDefaults.MIN_SIMILARITY,
    min_lines: int = DeduplicationDefaults.MIN_LINES,
    exclude_patterns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Find duplicate functions/classes/methods in a codebase.

    This is the main entry point for the find_duplication MCP tool.

    Args:
        project_folder: Path to the project folder
        language: Programming language
        min_similarity: Minimum similarity threshold (0-1)
        min_lines: Minimum lines to consider
        exclude_patterns: Path patterns to exclude

    Returns:
        Dictionary with duplication results
    """
    logger = get_logger("deduplication.tool.find")

    exclude_patterns = FilePatterns.normalize_excludes(exclude_patterns)

    detector = DuplicationDetector(language=language)
    results = detector.find_duplication(
        project_folder=project_folder,
        construct_type="function_definition",  # Default to functions
        min_similarity=min_similarity,
        min_lines=min_lines,
        exclude_patterns=exclude_patterns,
    )

    summary = results.get("summary", {})
    logger.info(
        "find_duplication_complete",
        duplicate_groups=summary.get("duplicate_groups", 0),
        total_duplicated_lines=summary.get("total_duplicated_lines", 0),
        potential_line_savings=summary.get("potential_line_savings", 0),
    )

    return results


def analyze_deduplication_candidates_tool(
    project_path: str,
    language: str,
    min_similarity: float = DeduplicationDefaults.MIN_SIMILARITY,
    include_test_coverage: bool = True,
    min_lines: int = DeduplicationDefaults.MIN_LINES,
    max_candidates: int = DeduplicationDefaults.MAX_CANDIDATES,
    exclude_patterns: Optional[List[str]] = None,
    timeout_per_candidate: Optional[float] = None,
) -> Dict[str, Any]:
    """Analyze a project for deduplication candidates and return ranked results."""
    logger = get_logger("deduplication.tool.analyze")

    exclude_patterns = FilePatterns.normalize_excludes(exclude_patterns)

    # Delegate to orchestrator
    orchestrator = DeduplicationAnalysisOrchestrator()
    result = orchestrator.analyze_candidates(
        project_path=project_path,
        language=language,
        min_similarity=min_similarity,
        include_test_coverage=include_test_coverage,
        min_lines=min_lines,
        max_candidates=max_candidates,
        exclude_patterns=exclude_patterns,
        timeout_per_candidate=timeout_per_candidate,
    )

    logger.info(
        "analyze_candidates_complete",
        total_groups=result.get("total_groups_analyzed", 0),
        returned_candidates=len(result.get("candidates", [])),
        total_savings_potential=result.get("top_candidates_savings_potential", 0),
        include_test_coverage=include_test_coverage,
    )

    return result


def apply_deduplication_tool(
    project_folder: str,
    group_id: int,
    refactoring_plan: Dict[str, Any],
    dry_run: bool = True,
    backup: bool = True,
    extract_to_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Apply automated deduplication refactoring with pre/post validation and auto-rollback."""
    logger = get_logger("deduplication.tool.apply")

    applicator = DeduplicationApplicator()
    result = applicator.apply_deduplication(
        project_folder=project_folder,
        group_id=group_id,
        refactoring_plan=refactoring_plan,
        dry_run=dry_run,
        backup=backup,
        extract_to_file=extract_to_file,
    )

    logger.info(
        "apply_deduplication_complete",
        status=result.get("status"),
        group_id=group_id,
        dry_run=dry_run,
        files_modified=len(result.get("files_modified", [])),
    )

    return result


def benchmark_deduplication_tool(iterations: int = 10, save_baseline: bool = False, check_regression: bool = True) -> Dict[str, Any]:
    """Run performance benchmarks for deduplication functions.

    Benchmarks the following operations:
    - **scoring**: calculate_deduplication_score (should be < 1ms)
    - **pattern_analysis**: rank_deduplication_candidates and analyze variations
    - **code_generation**: generate_deduplication_recommendation
    - **full_workflow**: create_enhanced_duplication_response

    Args:
        iterations: Number of iterations per benchmark (default: 10)
        save_baseline: Save results as new baseline for regression detection
        check_regression: Check results against baseline for performance regressions

    Returns:
        Dictionary with benchmark results including:
        - total_benchmarks: Number of benchmarks run
        - results: List of benchmark results with statistics
        - regression_detected: Whether any regressions were found
        - regression_errors: List of specific regression failures
    """
    logger = get_logger("deduplication.tool.benchmark")

    benchmark = DeduplicationBenchmark()
    results = benchmark.benchmark_deduplication(iterations=iterations, save_baseline=save_baseline, check_regression=check_regression)

    logger.info(
        "benchmark_complete",
        total_benchmarks=results.get("total_benchmarks"),
        regression_detected=results.get("regression_detected"),
        execution_time_seconds=results.get("execution_time_seconds"),
    )

    return results


def calculate_ast_similarity_tool(code1: str, code2: str) -> Dict[str, Any]:
    """Score two code snippets purely on AST/structural similarity.

    Uses HybridSimilarity's Stage 2 algorithm: SequenceMatcher on
    normalized code (or Jaccard of structural keyword patterns for
    snippets exceeding max_lines_for_full_ast).
    """
    logger = get_logger("deduplication.tool.ast_similarity")
    hybrid = HybridSimilarity()
    lines1 = len(code1.split("\n"))
    lines2 = len(code2.split("\n"))
    max_lines = hybrid.hybrid_config.max_lines_for_full_ast
    method = "simplified_jaccard" if max(lines1, lines2) > max_lines else "sequence_matcher"
    similarity = hybrid._calculate_ast_similarity(code1, code2)

    logger.info("ast_similarity_complete", similarity=similarity, method=method)
    return {
        "similarity": similarity,
        "method": method,
        "code1_lines": lines1,
        "code2_lines": lines2,
    }


def calculate_semantic_similarity_tool(
    code1: str,
    code2: str,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Score two code snippets on CodeBERT semantic (embedding) similarity.

    Requires optional deps: transformers + torch. Returns an error dict
    if the deps are not installed.
    """
    logger = get_logger("deduplication.tool.semantic_similarity")

    if not SemanticSimilarity.is_available():
        return {
            "error": "semantic_dependencies_missing",
            "message": "CodeBERT semantic similarity requires `transformers` and `torch`. Install with `pip install transformers torch`.",
        }

    config = SemanticSimilarityConfig(model_name=model_name) if model_name else None
    calculator = SemanticSimilarity(config=config)
    result = calculator.calculate_similarity_detailed(code1, code2)

    logger.info(
        "semantic_similarity_complete",
        similarity=result.similarity,
        model=result.model_used,
    )
    return result.to_dict()


def _register_find_duplication(mcp: FastMCP) -> None:
    from pydantic import Field

    @mcp.tool()
    def find_duplication(
        project_folder: str = Field(description="Path to the project folder"),
        language: str = Field(description="Programming language"),
        min_similarity: float = Field(default=DeduplicationDefaults.MIN_SIMILARITY, description="Minimum similarity threshold (0-1)"),
        min_lines: int = Field(default=DeduplicationDefaults.MIN_LINES, description="Minimum lines to consider"),
        exclude_patterns: Optional[List[str]] = Field(default=None, description="Path patterns to exclude"),
    ) -> Dict[str, Any]:
        """Find duplicate functions/classes/methods in a codebase."""
        return find_duplication_tool(
            project_folder=project_folder,
            language=language,
            min_similarity=min_similarity,
            min_lines=min_lines,
            exclude_patterns=exclude_patterns,
        )


def _register_analyze_candidates(mcp: FastMCP) -> None:
    from pydantic import Field

    @mcp.tool()
    def analyze_deduplication_candidates(
        project_path: str = Field(description="The absolute path to the project folder to analyze"),
        language: str = Field(description="The target language"),
        min_similarity: float = Field(default=DeduplicationDefaults.MIN_SIMILARITY, description="Minimum similarity threshold (0.0-1.0)"),
        include_test_coverage: bool = Field(default=True, description="Whether to check test coverage for prioritization"),
        min_lines: int = Field(default=DeduplicationDefaults.MIN_LINES, description="Minimum number of lines to consider for duplication"),
        max_candidates: int = Field(default=DeduplicationDefaults.MAX_CANDIDATES, description="Maximum number of candidates to return"),
        exclude_patterns: Optional[List[str]] = Field(default=None, description="Path patterns to exclude from analysis"),
        timeout_per_candidate: Optional[float] = Field(
            default=None,
            description="Per-candidate enrichment timeout in seconds (default: 30s). Raise this if enrichment is spuriously timing out.",
        ),
    ) -> Dict[str, Any]:
        """Analyze a project for deduplication candidates and return ranked results."""
        return analyze_deduplication_candidates_tool(
            project_path=project_path,
            language=language,
            min_similarity=min_similarity,
            include_test_coverage=include_test_coverage,
            min_lines=min_lines,
            max_candidates=max_candidates,
            exclude_patterns=exclude_patterns,
            timeout_per_candidate=timeout_per_candidate,
        )


def _register_apply_deduplication(mcp: FastMCP) -> None:
    from pydantic import Field

    @mcp.tool()
    def apply_deduplication(
        project_folder: str = Field(description="The absolute path to the project folder"),
        group_id: int = Field(description="The duplication group ID from find_duplication results"),
        refactoring_plan: Dict[str, Any] = Field(
            description="The refactoring plan with generated_code, files_affected, strategy, language"
        ),
        dry_run: bool = Field(default=True, description="Preview changes without applying (default: true for safety)"),
        backup: bool = Field(default=True, description="Create backup before applying changes (default: true)"),
        extract_to_file: Optional[str] = Field(default=None, description="Where to place extracted function (auto-detect if None)"),
    ) -> Dict[str, Any]:
        """Apply automated deduplication refactoring with pre/post validation and auto-rollback."""
        return apply_deduplication_tool(
            project_folder=project_folder,
            group_id=group_id,
            refactoring_plan=refactoring_plan,
            dry_run=dry_run,
            backup=backup,
            extract_to_file=extract_to_file,
        )


def _register_benchmark_deduplication(mcp: FastMCP) -> None:
    from pydantic import Field

    @mcp.tool()
    def benchmark_deduplication(
        iterations: int = Field(default=10, description="Number of iterations per benchmark (default: 10)"),
        save_baseline: bool = Field(default=False, description="Save results as new baseline for regression detection"),
        check_regression: bool = Field(default=True, description="Check results against baseline for performance regressions"),
    ) -> Dict[str, Any]:
        """Run performance benchmarks for deduplication functions."""
        return benchmark_deduplication_tool(iterations=iterations, save_baseline=save_baseline, check_regression=check_regression)


def _register_calculate_ast_similarity(mcp: FastMCP) -> None:
    from pydantic import Field

    @mcp.tool()
    def calculate_ast_similarity(
        code1: str = Field(description="First code snippet"),
        code2: str = Field(description="Second code snippet"),
    ) -> Dict[str, Any]:
        """Score two code snippets purely on AST/structural similarity (Stage 2 only)."""
        return calculate_ast_similarity_tool(code1=code1, code2=code2)


def _register_calculate_semantic_similarity(mcp: FastMCP) -> None:
    from pydantic import Field

    @mcp.tool()
    def calculate_semantic_similarity(
        code1: str = Field(description="First code snippet"),
        code2: str = Field(description="Second code snippet"),
        model_name: Optional[str] = Field(
            default=None,
            description="Optional CodeBERT-compatible model name (defaults to microsoft/codebert-base)",
        ),
    ) -> Dict[str, Any]:
        """Score two code snippets on CodeBERT semantic similarity (Stage 3 only)."""
        return calculate_semantic_similarity_tool(code1=code1, code2=code2, model_name=model_name)


def register_deduplication_tools(mcp: FastMCP) -> None:
    """Register all deduplication tools with the MCP server."""
    _register_find_duplication(mcp)
    _register_analyze_candidates(mcp)
    _register_apply_deduplication(mcp)
    _register_benchmark_deduplication(mcp)
    _register_calculate_ast_similarity(mcp)
    _register_calculate_semantic_similarity(mcp)
