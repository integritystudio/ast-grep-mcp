"""Orchestration for deduplication candidate analysis.

This module handles the multi-step process of finding duplicates,
ranking them, checking test coverage, and generating recommendations.
"""

import copy
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from ...constants import CodeAnalysisDefaults, DeduplicationDefaults, ParallelProcessing
from ...core.logging import get_logger
from ...utils.futures import CancelledBeforeStartError, WaitTimeoutError, map_with_per_item_timeout
from .config import AnalysisConfig
from .coverage import CoverageDetector
from .detector import DuplicationDetector
from .ranker import DuplicationRanker
from .recommendations import RecommendationEngine

# Type alias for progress callback function
# Signature: (stage_name: str, progress_percent: float) -> None
ProgressCallback = Callable[[str, float], None]


class DeduplicationAnalysisOrchestrator:
    """Orchestrates the complete deduplication candidate analysis workflow."""

    def __init__(self) -> None:
        """Initialize the orchestrator with lazy component initialization."""
        self.logger = get_logger("deduplication.analysis_orchestrator")
        # Components are lazily initialized via properties to reduce initialization overhead

    @property
    def detector(self) -> DuplicationDetector:
        """Get or create DuplicationDetector instance (lazy initialization)."""
        if not hasattr(self, "_detector"):
            self._detector = DuplicationDetector()
        return self._detector

    @detector.setter
    def detector(self, value: DuplicationDetector) -> None:
        """Set DuplicationDetector instance (for testing/dependency injection)."""
        self._detector = value

    @property
    def ranker(self) -> DuplicationRanker:
        """Get or create DuplicationRanker instance (lazy initialization)."""
        if not hasattr(self, "_ranker"):
            self._ranker = DuplicationRanker()
        return self._ranker

    @ranker.setter
    def ranker(self, value: DuplicationRanker) -> None:
        """Set DuplicationRanker instance (for testing/dependency injection)."""
        self._ranker = value

    @property
    def coverage_detector(self) -> CoverageDetector:
        """Get or create CoverageDetector instance (lazy initialization)."""
        if not hasattr(self, "_coverage_detector"):
            self._coverage_detector = CoverageDetector()
        return self._coverage_detector

    @coverage_detector.setter
    def coverage_detector(self, value: CoverageDetector) -> None:
        """Set CoverageDetector instance (for testing/dependency injection)."""
        self._coverage_detector = value

    @property
    def recommendation_engine(self) -> RecommendationEngine:
        """Get or create RecommendationEngine instance (lazy initialization)."""
        if not hasattr(self, "_recommendation_engine"):
            self._recommendation_engine = RecommendationEngine()
        return self._recommendation_engine

    @recommendation_engine.setter
    def recommendation_engine(self, value: RecommendationEngine) -> None:
        """Set RecommendationEngine instance (for testing/dependency injection)."""
        self._recommendation_engine = value

    def analyze_candidates(
        self,
        project_path: str,
        language: str,
        min_similarity: float = DeduplicationDefaults.MIN_SIMILARITY,
        include_test_coverage: bool = True,
        min_lines: int = DeduplicationDefaults.MIN_LINES,
        max_candidates: int = DeduplicationDefaults.MAX_CANDIDATES,
        exclude_patterns: List[str] | None = None,
        progress_callback: Optional[ProgressCallback] = None,
        timeout_per_candidate: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Analyze a project for deduplication candidates (legacy interface).

        Converts individual parameters to AnalysisConfig and delegates to
        analyze_candidates_with_config(). New code should use that method directly.

        Raises:
            ValueError: If input parameters are invalid
        """
        config = AnalysisConfig(
            project_path=project_path,
            language=language,
            min_similarity=min_similarity,
            include_test_coverage=include_test_coverage,
            min_lines=min_lines,
            max_candidates=max_candidates,
            exclude_patterns=exclude_patterns,
            progress_callback=progress_callback,
            timeout_per_candidate=timeout_per_candidate,
        )
        return self.analyze_candidates_with_config(config)

    def analyze_candidates_with_config(self, config: AnalysisConfig) -> Dict[str, Any]:
        """Analyze a project for deduplication candidates (recommended interface).

        Orchestrates a 4-step workflow: find duplicates, rank candidates,
        check test coverage, generate recommendations.

        Args:
            config: Analysis configuration object

        Raises:
            ValueError: If configuration is invalid
        """
        self._validate_analysis_inputs(config.project_path, config.language, config.min_similarity, config.min_lines, config.max_candidates)

        # Helper function for progress reporting
        def report_progress(stage: str, percent: float) -> None:
            """Report progress if callback is provided."""
            if config.progress_callback:
                config.progress_callback(stage, percent)

        self.logger.info("analysis_start", **config.to_dict())

        # Step 1: Find duplicates (0% -> 25%)
        # Ensure the detector is initialised with the correct language so the
        # detector_initialized log entry is accurate (TSD-05).  The lazy
        # property defaulted to language="python" and then mutated language
        # after the fact, causing every non-Python analysis to log python.
        # We only recreate a real DuplicationDetector; injected test doubles
        # (isinstance check = False) are left untouched.
        report_progress("Finding duplicate code", 0.0)
        if not hasattr(self, "_detector") or (
            isinstance(self._detector, DuplicationDetector) and self._detector.language != config.language
        ):
            self._detector = DuplicationDetector(language=config.language)
        duplication_results = self.detector.find_duplication(
            project_folder=config.project_path,
            construct_type="function_definition",
            min_similarity=config.min_similarity,
            min_lines=config.min_lines,
            exclude_patterns=config.exclude_patterns or [],
        )

        # Step 2: Rank candidates by refactoring value (25% -> 40%)
        # Pass max_candidates for early exit optimization (avoids unnecessary rank numbering)
        report_progress("Ranking candidates by value", DeduplicationDefaults.PROGRESS_RANKING)
        ranked_candidates = self.ranker.rank_deduplication_candidates(
            duplication_results.get("duplication_groups", []), max_results=config.max_candidates
        )

        # Step 3-5: Enrich and summarize top candidates (40% -> 100%)
        report_progress("Enriching candidates", DeduplicationDefaults.PROGRESS_ENRICHING)
        result = self._enrich_and_summarize_with_config(ranked_candidates, config, progress_callback=report_progress)

        # Complete
        report_progress("Analysis complete", 1.0)
        return result

    _SUPPORTED_LANGUAGES = ["python", "javascript", "typescript", "java", "go", "rust", "cpp", "c", "ruby"]

    def _validate_analysis_inputs(
        self, project_path: str, language: str, min_similarity: float, min_lines: int, max_candidates: int
    ) -> None:
        """Validate analysis inputs. Raises ValueError on invalid parameters."""
        self._validate_project_path(project_path)
        if not 0.0 <= min_similarity <= 1.0:
            raise ValueError(f"min_similarity must be between 0.0 and 1.0, got {min_similarity}")
        if min_lines < 1:
            raise ValueError(f"min_lines must be a positive integer, got {min_lines}")
        if max_candidates < 1:
            raise ValueError(f"max_candidates must be a positive integer, got {max_candidates}")
        self._warn_unsupported_language(language)

    def _validate_project_path(self, project_path: str) -> None:
        """Raise ValueError if project_path is missing or not a directory."""
        if not os.path.exists(project_path):
            raise ValueError(f"Project path does not exist: {project_path}")
        if not os.path.isdir(project_path):
            raise ValueError(f"Project path is not a directory: {project_path}")

    def _warn_unsupported_language(self, language: str) -> None:
        """Log a warning if language is not in the supported list."""
        if language.lower() not in self._SUPPORTED_LANGUAGES:
            self.logger.warning(
                "unsupported_language",
                language=language,
                supported=self._SUPPORTED_LANGUAGES,
                message=f"Language '{language}' may not be fully supported",
            )

    def _get_top_candidates(self, candidates: List[Dict[str, Any]], max_count: int) -> List[Dict[str, Any]]:
        """Get top N candidates from the list.

        Args:
            candidates: All candidates
            max_count: Maximum candidates to return

        Returns:
            Sliced list of top candidates
        """
        return candidates[:max_count]

    def _build_analysis_metadata(
        self, language: str, min_similarity: float, min_lines: int, include_test_coverage: bool, project_path: str
    ) -> Dict[str, Any]:
        """Build analysis metadata dictionary (legacy interface).

        Maintained for backward compatibility. New code should use
        _build_analysis_metadata_from_config().

        Args:
            language: Programming language
            min_similarity: Minimum similarity threshold
            min_lines: Minimum lines to consider
            include_test_coverage: Whether test coverage was included
            project_path: Project folder path

        Returns:
            Analysis metadata dictionary
        """
        return {
            "language": language,
            "min_similarity": min_similarity,
            "min_lines": min_lines,
            "include_test_coverage": include_test_coverage,
            "project_path": project_path,
        }

    def _build_analysis_metadata_from_config(self, config: AnalysisConfig) -> Dict[str, Any]:
        """Build analysis metadata dictionary from config.

        Args:
            config: Analysis configuration object

        Returns:
            Analysis metadata dictionary
        """
        return {
            "language": config.language,
            "min_similarity": config.min_similarity,
            "min_lines": config.min_lines,
            "include_test_coverage": config.include_test_coverage,
            "project_path": config.project_path,
        }

    def _enrich_and_summarize(
        self,
        ranked_candidates: List[Dict[str, Any]],
        max_candidates: int,
        include_test_coverage: bool,
        language: str,
        project_path: str,
        min_similarity: float,
        min_lines: int,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Dict[str, Any]:
        """Enrich top candidates and generate summary (legacy interface).

        Maintained for backward compatibility. New code should use
        _enrich_and_summarize_with_config().

        Args:
            ranked_candidates: All ranked candidates
            max_candidates: Maximum candidates to return
            include_test_coverage: Whether to add test coverage
            language: Programming language
            project_path: Project folder path
            min_similarity: Minimum similarity threshold
            min_lines: Minimum lines to consider
            progress_callback: Optional progress reporting callback

        Returns:
            Analysis results with enriched candidates and metadata
        """
        # Convert to config and call new method
        config = AnalysisConfig(
            project_path=project_path,
            language=language,
            min_similarity=min_similarity,
            include_test_coverage=include_test_coverage,
            min_lines=min_lines,
            max_candidates=max_candidates,
            progress_callback=progress_callback,
        )
        return self._enrich_and_summarize_with_config(ranked_candidates, config, progress_callback=progress_callback)

    def _empty_enrich_result(self, config: AnalysisConfig) -> Dict[str, Any]:
        """Return an empty analysis result when there are no candidates."""
        return {
            "candidates": [],
            "total_groups_analyzed": 0,
            "top_candidates_count": 0,
            "top_candidates_savings_potential": 0,
            "analysis_metadata": self._build_analysis_metadata_from_config(config),
        }

    def _build_enrich_result(
        self, top_candidates: List[Dict[str, Any]], ranked_candidates: List[Dict[str, Any]], config: AnalysisConfig
    ) -> Dict[str, Any]:
        """Build the final enrichment result dict."""
        savings = self._calculate_total_savings(top_candidates)
        self.logger.info(
            "analysis_complete",
            total_groups_analyzed=len(ranked_candidates),
            top_candidates_count=len(top_candidates),
            top_candidates_savings_potential=savings,
        )
        return {
            "candidates": top_candidates,
            "total_groups_analyzed": len(ranked_candidates),
            "top_candidates_count": len(top_candidates),
            "top_candidates_savings_potential": savings,
            "analysis_metadata": self._build_analysis_metadata_from_config(config),
        }

    def _enrich_and_summarize_with_config(
        self,
        ranked_candidates: List[Dict[str, Any]],
        config: AnalysisConfig,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Dict[str, Any]:
        """Enrich top candidates and generate summary using config object."""

        def report(stage: str, percent: float) -> None:
            if progress_callback:
                progress_callback(stage, percent)

        if not ranked_candidates:
            self.logger.warning("no_candidates_to_enrich")
            return self._empty_enrich_result(config)

        report("Selecting top candidates", DeduplicationDefaults.PROGRESS_SELECTION)
        top_candidates = self._get_top_candidates(ranked_candidates, config.max_candidates)

        if config.include_test_coverage:
            report("Checking test coverage", DeduplicationDefaults.PROGRESS_COVERAGE_CHECK)
            self._add_test_coverage_batch(
                top_candidates,
                config.language,
                config.project_path,
                parallel=config.parallel,
                max_workers=config.max_workers,
                timeout_per_candidate=config.timeout_per_candidate,
            )
            report("Test coverage complete", DeduplicationDefaults.PROGRESS_COVERAGE_COMPLETE)

        report("Generating recommendations", DeduplicationDefaults.PROGRESS_RECOMMENDATIONS)
        self._add_recommendations(
            top_candidates,
            parallel=config.parallel,
            max_workers=config.max_workers,
            timeout_per_candidate=config.timeout_per_candidate,
        )

        report("Calculating statistics", DeduplicationDefaults.PROGRESS_STATISTICS)
        return self._build_enrich_result(top_candidates, ranked_candidates, config)

    def _enrich_with_test_coverage(self, candidate: Dict[str, Any], language: str, project_path: str) -> None:
        """Enrich a single candidate with test coverage data.

        Args:
            candidate: Candidate to enrich
            language: Programming language
            project_path: Project folder path

        Note:
            This method is used for sequential processing. For batch parallel
            processing, use _add_test_coverage_batch() instead.
        """
        files = candidate.get("files", [])
        if files:
            coverage_map = self.coverage_detector.get_test_coverage_for_files(files, language, project_path)
            candidate["test_coverage"] = coverage_map
            candidate["has_tests"] = any(coverage_map.values())

    def _enrich_with_recommendation(self, candidate: Dict[str, Any]) -> None:
        """Enrich a single candidate with recommendation.

        Args:
            candidate: Candidate to enrich
        """
        recommendation = self.recommendation_engine.generate_deduplication_recommendation(
            score=candidate.get("score", 0),
            complexity=candidate.get("complexity_score", CodeAnalysisDefaults.DEFAULT_COMPLEXITY_SCORE),
            lines_saved=candidate.get("lines_saved", 0),
            has_tests=candidate.get("has_tests", False),
            affected_files=len(candidate.get("files", [])),
        )
        candidate["recommendation"] = recommendation

    def _handle_enrichment_error(
        self,
        candidate: Dict[str, Any],
        error: Exception,
        operation_name: str,
        error_field: str,
        default_error_value: Any,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        """Handle enrichment errors by logging and updating candidate.

        Args:
            candidate: The candidate that failed
            error: The exception that occurred (or TimeoutError)
            operation_name: Name of operation for logging
            error_field: Field name to store error message
            default_error_value: Default value to set on error
            timeout_seconds: Resolved timeout value for WaitTimeoutError log and
                message (a number, not a string). None when the error is not a
                wait timeout.
        """
        # Log the error. Only a helper wait expiry is a "timeout" here — a
        # TimeoutError raised by the enrichment itself (socket.timeout,
        # OSError(ETIMEDOUT)) is a worker failure with its own message.
        # CancelledBeforeStartError is a subtype of WaitTimeoutError that means
        # the item was never dequeued (starved behind hung workers), distinct
        # from a running worker that exceeded the deadline (CR-07).
        if isinstance(error, CancelledBeforeStartError):
            self.logger.warning(
                f"{operation_name}_never_started",
                candidate_id=candidate.get("id", "unknown"),
                timeout_seconds=timeout_seconds,
            )
            candidate[error_field] = "Cancelled before start"
        elif isinstance(error, WaitTimeoutError):
            self.logger.error(
                f"{operation_name}_timeout",
                candidate_id=candidate.get("id", "unknown"),
                timeout_seconds=timeout_seconds,
            )
            candidate[error_field] = f"Operation timed out after {timeout_seconds:g}s"
        else:
            error_text = str(error) or type(error).__name__
            self.logger.error(f"{operation_name}_enrichment_failed", candidate_id=candidate.get("id", "unknown"), error=error_text)
            candidate[error_field] = error_text

        # Set default error value
        if isinstance(default_error_value, dict):
            for key, value in default_error_value.items():
                candidate[key] = value

    def _make_enrichment_callbacks(
        self,
        operation_name: str,
        error_field: str,
        default_error_value: Any,
        timeout_seconds: float,
        failed_candidates: List[Dict[str, Any]],
    ) -> Tuple[
        Callable[[Tuple[Dict[str, Any], Dict[str, Any]], None], None],
        Callable[[Tuple[Dict[str, Any], Dict[str, Any]], Exception], None],
    ]:
        """Build on_success/on_error callbacks for map_with_per_item_timeout.

        Items are (original, working) pairs: workers enrich the private working
        copy, which is merged onto the original only on success — an abandoned
        hung worker can never mutate the dict the caller already received back,
        and an enrich_func that mutates then raises leaks no partial writes.
        Keys deleted by enrich_func are not propagated.
        """

        def on_success(pair: Tuple[Dict[str, Any], Dict[str, Any]], _result: None) -> None:
            original, working = pair
            original.update(working)

        def on_error(pair: Tuple[Dict[str, Any], Dict[str, Any]], error: Exception) -> None:
            original, _working = pair
            ts = timeout_seconds if isinstance(error, WaitTimeoutError) else None
            self._handle_enrichment_error(original, error, operation_name, error_field, default_error_value, timeout_seconds=ts)
            failed_candidates.append(original)

        return on_success, on_error

    def _process_parallel_enrichment(
        self,
        candidates: List[Dict[str, Any]],
        enrich_func: Callable[..., None],
        operation_name: str,
        error_field: str,
        default_error_value: Any,
        max_workers: int,
        timeout_seconds: float,
        kwargs: Dict[str, Any],
        total_timeout_seconds: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Process enrichment in parallel with per-candidate timeouts.

        Args:
            candidates: List of candidates to enrich
            enrich_func: Function to call for each candidate
            operation_name: Name of operation for logging
            error_field: Field name to store error message
            default_error_value: Default value to set on error
            max_workers: Maximum number of threads
            timeout_seconds: Timeout per candidate
            kwargs: Additional arguments for enrich_func
            total_timeout_seconds: Shared wall-clock deadline across all candidates.
                Items reached after the deadline expires are immediately marked as
                timed out without waiting. Defaults to None (no shared deadline).

        Returns:
            List of failed candidates
        """
        failed_candidates: List[Dict[str, Any]] = []
        on_success, on_error = self._make_enrichment_callbacks(
            operation_name, error_field, default_error_value, timeout_seconds, failed_candidates
        )

        pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        for candidate in candidates:
            try:
                pairs.append((candidate, copy.deepcopy(candidate)))
            except Exception as error:
                # An uncopyable candidate fails alone rather than aborting the stage.
                on_error((candidate, candidate), error)

        map_with_per_item_timeout(
            pairs,
            lambda pair: enrich_func(pair[1], **kwargs),
            timeout_seconds=timeout_seconds,
            max_workers=max_workers,
            on_success=on_success,
            on_error=on_error,
            total_timeout_seconds=total_timeout_seconds,
        )

        return failed_candidates

    def _process_sequential_enrichment(
        self,
        candidates: List[Dict[str, Any]],
        enrich_func: Callable[..., None],
        operation_name: str,
        error_field: str,
        default_error_value: Any,
        timeout_seconds: float,
        kwargs: Dict[str, Any],
        total_timeout_seconds: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Process enrichment one candidate at a time with per-candidate timeout.

        Each candidate gets its own single-worker pool (one helper call each) so
        a hung enrichment is abandoned after timeout_seconds without blocking the
        next candidate. Previously this path called enrich_func inline with no
        timeout, so timeout_per_candidate was silently ignored.

        When ``total_timeout_seconds`` is set, a shared deadline is computed at
        entry and the remaining time is passed to each per-candidate call so the
        entire stage returns within the deadline regardless of candidate count.

        Args:
            candidates: List of candidates to enrich
            enrich_func: Function to call for each candidate
            operation_name: Name of operation for logging
            error_field: Field name to store error message
            default_error_value: Default value to set on error
            timeout_seconds: Maximum seconds to wait per candidate
            kwargs: Additional arguments for enrich_func
            total_timeout_seconds: Shared wall-clock deadline across all candidates.

        Returns:
            List of failed candidates
        """
        failed_candidates: List[Dict[str, Any]] = []
        on_success, on_error = self._make_enrichment_callbacks(
            operation_name, error_field, default_error_value, timeout_seconds, failed_candidates
        )

        deadline: Optional[float] = (time.monotonic() + total_timeout_seconds) if total_timeout_seconds is not None else None

        for candidate in candidates:
            try:
                pair = (candidate, copy.deepcopy(candidate))
            except Exception as error:
                # An uncopyable candidate fails alone rather than aborting the stage.
                on_error((candidate, candidate), error)
                continue

            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    on_error((candidate, candidate), WaitTimeoutError("Global deadline exceeded"))
                    continue
                call_total: Optional[float] = remaining
            else:
                call_total = None

            map_with_per_item_timeout(
                [pair],
                lambda pair: enrich_func(pair[1], **kwargs),
                timeout_seconds=timeout_seconds,
                max_workers=1,
                on_success=on_success,
                on_error=on_error,
                total_timeout_seconds=call_total,
            )

        return failed_candidates

    def _resolve_timeout(self, timeout_per_candidate: Optional[float]) -> float:
        """Return timeout_per_candidate, defaulting to ParallelProcessing.DEFAULT_TIMEOUT_PER_CANDIDATE_SECONDS.

        A value of 0 or negative is treated the same as None (use the default)
        to prevent ``future.result(timeout=0)`` from instantly failing every
        running future — a behaviour that was previously unreachable dead code
        (CR-11).
        """
        if timeout_per_candidate is not None and timeout_per_candidate > 0:
            return float(timeout_per_candidate)
        return float(ParallelProcessing.DEFAULT_TIMEOUT_PER_CANDIDATE_SECONDS)

    def _resolve_total_timeout(self, total_timeout_seconds: Optional[float]) -> float:
        """Return total_timeout_seconds, defaulting to ParallelProcessing.MAX_TIMEOUT_SECONDS."""
        if total_timeout_seconds is not None:
            return total_timeout_seconds
        return float(ParallelProcessing.MAX_TIMEOUT_SECONDS)

    def _parallel_enrich(
        self,
        candidates: List[Dict[str, Any]],
        enrich_func: Callable[..., None],
        operation_name: str,
        error_field: str,
        default_error_value: Any,
        parallel: bool = True,
        max_workers: int = ParallelProcessing.DEFAULT_WORKERS,
        timeout_per_candidate: Optional[float] = None,
        total_timeout_seconds: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Generic parallel enrichment helper.

        Args:
            candidates: Candidates to enrich
            enrich_func: Called per candidate as func(candidate, **kwargs)
            operation_name: Name used in log keys
            error_field: Field to store error message on failure
            default_error_value: Value(s) set on candidate when enrichment fails
            parallel: Use parallel execution when True and len(candidates) > 1
            max_workers: Thread pool size
            timeout_per_candidate: Per-candidate wait window in seconds, as a
                float (default 30s). Zero or negative is treated as None (use
                default). The wait clock starts when the loop *reaches* the
                future, not when the candidate started executing, so a candidate
                running behind hung workers may complete in more than
                ``timeout_per_candidate`` wall-clock seconds and still pass.
                Use ``total_timeout_seconds`` for a hard wall-clock cap.
            total_timeout_seconds: Shared wall-clock deadline across all candidates
                in seconds. Items reached after the deadline expires are immediately
                marked as timed out. Defaults to ParallelProcessing.MAX_TIMEOUT_SECONDS
                (300s), enforcing the documented 5-minute cap.
            **kwargs: Forwarded to enrich_func

        Returns:
            List of candidates that failed enrichment
        """
        timeout_seconds = self._resolve_timeout(timeout_per_candidate)
        resolved_total = self._resolve_total_timeout(total_timeout_seconds)
        if parallel and len(candidates) > 1:
            failed_candidates = self._process_parallel_enrichment(
                candidates, enrich_func, operation_name, error_field, default_error_value, max_workers, timeout_seconds, kwargs,
                total_timeout_seconds=resolved_total,
            )
        else:
            # Sequential path also enforces the per-candidate timeout; previously
            # timeout_per_candidate was silently ignored on this branch.
            failed_candidates = self._process_sequential_enrichment(
                candidates, enrich_func, operation_name, error_field, default_error_value, timeout_seconds, kwargs,
                total_timeout_seconds=resolved_total,
            )
        self.logger.info(f"{operation_name}_added", candidate_count=len(candidates), failed_count=len(failed_candidates), parallel=parallel)
        return failed_candidates

    def _collect_unique_files(self, candidates: List[Dict[str, Any]]) -> List[str]:
        """Collect all unique file paths referenced across candidates, preserving order."""
        all_files: List[str] = []
        for candidate in candidates:
            all_files.extend(candidate.get("files", []))
        return list(dict.fromkeys(all_files))

    def _distribute_coverage(self, candidates: List[Dict[str, Any]], coverage_map: Dict[str, Any]) -> None:
        """Apply a pre-computed coverage_map to each candidate in-place."""
        for candidate in candidates:
            files = candidate.get("files", [])
            if files:
                candidate_coverage = {f: coverage_map.get(f, False) for f in files}
                candidate["test_coverage"] = candidate_coverage
                candidate["has_tests"] = any(candidate_coverage.values())
            else:
                candidate["test_coverage"] = {}
                candidate["has_tests"] = False

    def _add_test_coverage_batch(
        self,
        candidates: List[Dict[str, Any]],
        language: str,
        project_path: str,
        parallel: bool = True,
        max_workers: int = ParallelProcessing.DEFAULT_WORKERS,
        timeout_per_candidate: Optional[float] = None,
    ) -> None:
        """Add test coverage via optimized batch processing (60-80% faster than per-candidate).

        Collects all unique files once, runs batch detection, then distributes results.
        """
        if not candidates:
            return

        unique_files = self._collect_unique_files(candidates)
        if not unique_files:
            self.logger.debug("no_files_for_coverage_check")
            return

        self.logger.debug(
            "batch_coverage_start",
            candidate_count=len(candidates),
            unique_file_count=len(unique_files),
            parallel=parallel,
        )

        coverage_map = self.coverage_detector.get_test_coverage_for_files_batch(
            unique_files, language, project_path, parallel=parallel, max_workers=max_workers, timeout_per_file=timeout_per_candidate
        )
        self._distribute_coverage(candidates, coverage_map)
        self.logger.info("batch_coverage_added", candidate_count=len(candidates), unique_files_checked=len(unique_files), parallel=parallel)

    def _add_test_coverage(
        self,
        candidates: List[Dict[str, Any]],
        language: str,
        project_path: str,
        parallel: bool = True,
        max_workers: int = ParallelProcessing.DEFAULT_WORKERS,
    ) -> List[Dict[str, Any]]:
        """Add test coverage information to candidates (legacy method).

        Args:
            candidates: List of candidates to enrich
            language: Programming language
            project_path: Project folder path
            parallel: Whether to use parallel execution (default: True)
            max_workers: Maximum number of threads for parallel execution

        Returns:
            List of candidates that failed enrichment (for monitoring)

        Note:
            This is the legacy implementation. Use _add_test_coverage_batch()
            for better performance (60-80% faster). Now refactored to use
            _parallel_enrich() to reduce code duplication.
        """
        return self._parallel_enrich(
            candidates=candidates,
            enrich_func=self._enrich_with_test_coverage,
            operation_name="test_coverage",
            error_field="test_coverage_error",
            default_error_value={"test_coverage": {}, "has_tests": False},
            parallel=parallel,
            max_workers=max_workers,
            language=language,
            project_path=project_path,
        )

    def _add_recommendations(
        self,
        candidates: List[Dict[str, Any]],
        parallel: bool = True,
        max_workers: int = ParallelProcessing.DEFAULT_WORKERS,
        timeout_per_candidate: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Add recommendations to candidates.

        Args:
            candidates: List of candidates to enrich
            parallel: Whether to use parallel execution (default: True)
            max_workers: Maximum number of threads for parallel execution
            timeout_per_candidate: Timeout in seconds for each candidate (default: 30s)

        Returns:
            List of candidates that failed enrichment (for monitoring)

        Note:
            Now refactored to use _parallel_enrich() to reduce code duplication.
        """
        return self._parallel_enrich(
            candidates=candidates,
            enrich_func=self._enrich_with_recommendation,
            operation_name="recommendations",
            error_field="recommendation_error",
            default_error_value={
                "recommendation": {"action": "error", "reasoning": "Failed to generate recommendation", "priority": "low"}
            },
            timeout_per_candidate=timeout_per_candidate,
            parallel=parallel,
            max_workers=max_workers,
        )

    def _calculate_total_savings(self, candidates: List[Dict[str, Any]]) -> int:
        """Calculate total potential line savings.

        Args:
            candidates: List of candidates

        Returns:
            Total potential lines saved
        """
        total = sum(c.get("lines_saved", 0) * len(c.get("files", [])) for c in candidates)
        return int(total)
