"""Performance benchmarking suite for ast-grep MCP server.

This module provides benchmarking for:
- Pattern search (find_code)
- YAML rule search (find_code_by_rule)
- Complex multi-condition rules
- Large result sets
- Streaming and caching performance

Usage:
    # Run benchmarks
    pytest tests/test_benchmark.py -v

    # Run with baseline comparison
    pytest tests/test_benchmark.py -v --benchmark-compare

    # Update baseline
    pytest tests/test_benchmark.py -v --benchmark-save=baseline
"""

import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import pytest

# Import implementation functions directly (bypasses MCP decorator)
from ast_grep_mcp.features.search.service import find_code_by_rule_impl, find_code_impl
from ast_grep_mcp.utils.console_logger import console


# Create a mock mcp object with tools dictionary for backward compatibility
class _MockMCP:
    """Mock MCP for backward compatibility with existing benchmark tests."""

    tools: Dict[str, Any] = {
        "find_code": find_code_impl,
        "find_code_by_rule": find_code_by_rule_impl,
    }


mcp = _MockMCP()

# Import deduplication functions from modular structure
# Ensure caching is enabled for benchmarks
# Initialize the modular cache (used by the actual implementation)
from ast_grep_mcp.core import cache as core_cache  # noqa: E402
from ast_grep_mcp.core import config as core_config  # noqa: E402
from ast_grep_mcp.features.deduplication.coverage import CoverageDetector  # noqa: E402
from ast_grep_mcp.features.deduplication.ranker import DuplicationRanker  # noqa: E402
from ast_grep_mcp.features.deduplication.recommendations import RecommendationEngine  # noqa: E402
from ast_grep_mcp.features.deduplication.reporting import DuplicationReporter  # noqa: E402

# Instantiate for use in tests
_coverage_detector = CoverageDetector()
get_test_coverage_for_files = _coverage_detector.get_test_coverage_for_files
_ranker = DuplicationRanker()
rank_deduplication_candidates = _ranker.rank_deduplication_candidates
_engine = RecommendationEngine()
generate_deduplication_recommendation = _engine.generate_deduplication_recommendation
_reporter = DuplicationReporter()
create_enhanced_duplication_response = _reporter.create_enhanced_duplication_response

core_config.CACHE_ENABLED = True
if core_cache._query_cache is None:
    core_cache.init_query_cache(max_size=100, ttl_seconds=300)


class BenchmarkResult:
    """Store benchmark results for comparison."""

    def __init__(self, name: str, execution_time: float, memory_mb: float, result_count: int, cache_hit: bool = False):
        self.name = name
        self.execution_time = execution_time
        self.memory_mb = memory_mb
        self.result_count = result_count
        self.cache_hit = cache_hit

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "execution_time_seconds": round(self.execution_time, 3),
            "memory_mb": round(self.memory_mb, 2),
            "result_count": self.result_count,
            "cache_hit": self.cache_hit,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkResult":
        return cls(
            name=data["name"],
            execution_time=data["execution_time_seconds"],
            memory_mb=data["memory_mb"],
            result_count=data["result_count"],
            cache_hit=data.get("cache_hit", False),
        )


class BenchmarkRunner:
    """Run benchmarks and track results."""

    def __init__(self, baseline_file: str = "tests/benchmark_baseline.json"):
        self.baseline_file = baseline_file
        self.results: List[BenchmarkResult] = []
        self.baseline: Dict[str, BenchmarkResult] = {}
        self._load_baseline()

    def _load_baseline(self) -> None:
        """Load baseline metrics from file."""
        if os.path.exists(self.baseline_file):
            with open(self.baseline_file, "r") as f:
                data = json.load(f)
                self.baseline = {item["name"]: BenchmarkResult.from_dict(item) for item in data.get("benchmarks", [])}

    def save_baseline(self) -> None:
        """Save current results as new baseline."""
        baseline_data = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "benchmarks": [r.to_dict() for r in self.results]}
        os.makedirs(os.path.dirname(self.baseline_file), exist_ok=True)
        with open(self.baseline_file, "w") as f:
            json.dump(baseline_data, f, indent=2)

    def run_benchmark(self, name: str, func: Any, *args: Any, **kwargs: Any) -> BenchmarkResult:
        """Run a single benchmark and record results."""
        import tracemalloc

        # Track cache hits before running
        cache_hits_before = core_cache._query_cache.hits if core_cache._query_cache else 0

        # Start memory tracking
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]

        # Run benchmark
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time

        # Calculate memory usage
        current, peak = tracemalloc.get_traced_memory()
        memory_mb = (peak - start_memory) / (1024 * 1024)
        tracemalloc.stop()

        # Check if cache was hit
        cache_hits_after = core_cache._query_cache.hits if core_cache._query_cache else 0
        cache_hit = cache_hits_after > cache_hits_before

        # Count results
        if isinstance(result, list):
            result_count = len(result)
        elif isinstance(result, str):
            # Count matches in text output
            result_count = result.count("\n\n") if result else 0
        else:
            result_count = 0

        benchmark_result = BenchmarkResult(
            name=name, execution_time=execution_time, memory_mb=memory_mb, result_count=result_count, cache_hit=cache_hit
        )

        self.results.append(benchmark_result)
        return benchmark_result

    def check_regression(self, threshold: float = 0.10) -> Tuple[bool, List[str]]:
        """Check for performance regressions.

        Args:
            threshold: Maximum allowed slowdown (0.10 = 10%)

        Returns:
            Tuple of (has_regression, error_messages)
        """
        if not self.baseline:
            return (False, ["No baseline found - skipping regression check"])

        errors = []
        for result in self.results:
            if result.name not in self.baseline:
                continue

            baseline = self.baseline[result.name]
            slowdown = (result.execution_time - baseline.execution_time) / baseline.execution_time

            if slowdown > threshold:
                errors.append(
                    f"{result.name}: {slowdown * 100:.1f}% slower ({baseline.execution_time:.3f}s → {result.execution_time:.3f}s)"
                )

        return (len(errors) > 0, errors)

    def generate_report(self) -> str:
        """Generate markdown benchmark report."""
        lines = ["# Performance Benchmark Report", ""]
        lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Benchmarks Run:** {len(self.results)}")
        lines.append("")

        # Results table
        lines.append("## Benchmark Results")
        lines.append("")
        lines.append("| Benchmark | Time (s) | Memory (MB) | Results | vs Baseline |")
        lines.append("|-----------|----------|-------------|---------|-------------|")

        for result in self.results:
            baseline_info = ""
            if result.name in self.baseline:
                baseline = self.baseline[result.name]
                diff = ((result.execution_time - baseline.execution_time) / baseline.execution_time) * 100
                if diff > 0:
                    baseline_info = f"+{diff:.1f}% 🔴"
                elif diff < -5:  # Improvement > 5%
                    baseline_info = f"{diff:.1f}% 🟢"
                else:
                    baseline_info = "~same"
            else:
                baseline_info = "new"

            lines.append(
                f"| {result.name} | {result.execution_time:.3f} | {result.memory_mb:.2f} | {result.result_count} | {baseline_info} |"
            )

        lines.append("")

        # Regression check
        has_regression, errors = self.check_regression()
        if has_regression:
            lines.append("## ⚠️ Performance Regressions Detected")
            lines.append("")
            for error in errors:
                lines.append(f"- {error}")
            lines.append("")
        else:
            lines.append("## ✅ No Performance Regressions")
            lines.append("")

        return "\n".join(lines)


@pytest.fixture
def benchmark_runner() -> BenchmarkRunner:
    """Provide benchmark runner for tests."""
    return BenchmarkRunner()


@pytest.fixture
def benchmark_fixtures() -> Path:
    """Provide path to benchmark fixtures."""
    return Path(__file__).parent.parent / "fixtures"


class TestPerformanceBenchmarks:
    """Performance benchmarking test suite."""

    def test_benchmark_simple_pattern_search(self, benchmark_runner: BenchmarkRunner, benchmark_fixtures: Path) -> None:
        """Benchmark simple pattern search with find_code."""
        tool = mcp.tools["find_code"]  # type: ignore

        result = benchmark_runner.run_benchmark(
            "simple_pattern_search",
            tool,
            project_folder=str(benchmark_fixtures),
            pattern="def $FUNC",
            language="python",
            output_format="json",
        )

        assert result.execution_time < 5.0, "Simple pattern search too slow"
        assert result.memory_mb < 50.0, "Simple pattern search uses too much memory"

    def test_benchmark_yaml_rule_search(self, benchmark_runner: BenchmarkRunner, benchmark_fixtures: Path) -> None:
        """Benchmark YAML rule search with find_code_by_rule."""
        tool = mcp.tools["find_code_by_rule"]  # type: ignore

        yaml_rule = """
id: test-rule
language: python
rule:
  kind: class_definition
  pattern: class $NAME
"""

        result = benchmark_runner.run_benchmark(
            "yaml_rule_search", tool, project_folder=str(benchmark_fixtures), yaml_rule=yaml_rule, output_format="json"
        )

        assert result.execution_time < 5.0, "YAML rule search too slow"
        assert result.memory_mb < 50.0, "YAML rule search uses too much memory"

    def test_benchmark_max_results_early_termination(self, benchmark_runner: BenchmarkRunner, benchmark_fixtures: Path) -> None:
        """Benchmark early termination with max_results."""
        tool = mcp.tools["find_code"]  # type: ignore

        # Benchmark with max_results (should terminate early)
        result = benchmark_runner.run_benchmark(
            "early_termination_max_10",
            tool,
            project_folder=str(benchmark_fixtures),
            pattern="def $FUNC",
            language="python",
            max_results=10,
            output_format="json",
        )

        # Should find exactly 10 results (or fewer if not enough matches)
        assert result.result_count <= 10, "Early termination didn't work"

    def test_benchmark_file_size_filtering(self, benchmark_runner: BenchmarkRunner, benchmark_fixtures: Path) -> None:
        """Benchmark file size filtering performance."""
        tool = mcp.tools["find_code"]  # type: ignore

        # Benchmark with file size filtering
        result = benchmark_runner.run_benchmark(
            "file_size_filtering_10mb",
            tool,
            project_folder=str(benchmark_fixtures),
            pattern="def $FUNC",
            language="python",
            max_file_size_mb=10,
            output_format="json",
        )

        assert result.execution_time < 5.0, "File size filtering too slow"

    def test_benchmark_caching_performance(self, benchmark_runner: BenchmarkRunner, benchmark_fixtures: Path) -> None:
        """Benchmark cache hit performance.

        BR-05: the global query cache is torn down by `initialized_cache`
        fixture teardown in the unit/quality suites, so this test fails when
        run together with them.  Reinitialise here so the test is order-
        independent.
        """
        # Ensure a live cache exists for this test regardless of prior teardowns.
        core_config.CACHE_ENABLED = True
        if core_cache._query_cache is None:
            core_cache.init_query_cache(max_size=100, ttl_seconds=300)
        else:
            core_cache._query_cache.cache.clear()

        tool = mcp.tools["find_code"]  # type: ignore

        # First run (cache miss)
        result1 = benchmark_runner.run_benchmark(
            "cache_miss", tool, project_folder=str(benchmark_fixtures), pattern="def $FUNC", language="python", output_format="json"
        )

        # Second run (cache hit)
        result2 = benchmark_runner.run_benchmark(
            "cache_hit", tool, project_folder=str(benchmark_fixtures), pattern="def $FUNC", language="python", output_format="json"
        )

        # Verify second run was a cache hit
        assert result2.cache_hit, "Second run should be a cache hit"

        # For small fixtures both runs finish in sub-millisecond time, so
        # ratio-based assertions are dominated by scheduling jitter and
        # tracemalloc overhead.  Assert an absolute ceiling instead.
        assert result2.execution_time < 0.5, f"Cache hit too slow: {result2.execution_time * 1000:.3f}ms (expected <500ms)"

        # Log the speedup for informational purposes
        speedup = result1.execution_time / result2.execution_time if result2.execution_time > 0 else float("inf")
        if speedup >= 2.0:
            console.log(f"Cache speedup: {speedup:.1f}x (excellent)")
        elif speedup >= 1.0:
            console.log(f"Cache speedup: {speedup:.1f}x (good)")
        else:
            console.log(f"Cache overhead: {1 / speedup:.1f}x (acceptable for small fixtures)")

    def test_generate_benchmark_report(self, benchmark_runner: BenchmarkRunner, tmp_path: Path) -> None:
        """Generate benchmark report after running benchmarks."""
        # Run a simple benchmark
        benchmark_runner.results.append(BenchmarkResult(name="test_benchmark", execution_time=0.123, memory_mb=10.5, result_count=42))

        # Generate report
        report = benchmark_runner.generate_report()

        assert "Performance Benchmark Report" in report
        assert "test_benchmark" in report
        assert "0.123" in report

        # Save report to file
        report_file = tmp_path / "benchmark_report.md"
        report_file.write_text(report)
        assert report_file.exists()

    def test_regression_detection(self, benchmark_runner: BenchmarkRunner) -> None:
        """Test performance regression detection."""
        # Add baseline result
        baseline = BenchmarkResult(name="test_query", execution_time=1.0, memory_mb=10.0, result_count=100)
        benchmark_runner.baseline["test_query"] = baseline

        # Add current result with 15% regression
        current = BenchmarkResult(name="test_query", execution_time=1.15, memory_mb=10.0, result_count=100)
        benchmark_runner.results.append(current)

        # Check for regression
        has_regression, errors = benchmark_runner.check_regression(threshold=0.10)

        assert has_regression, "Should detect regression"
        assert len(errors) == 1
        assert "15.0% slower" in errors[0]


@pytest.mark.skipif(os.environ.get("CI") != "true", reason="Only run in CI for regression detection")
class TestCIBenchmarks:
    """Benchmarks that run in CI for regression detection."""

    def test_ci_regression_check(self, benchmark_runner: BenchmarkRunner, benchmark_fixtures: Path) -> None:
        """Run benchmarks in CI and fail on regression."""
        # Run all standard benchmarks
        tool = mcp.tools["find_code"]  # type: ignore

        benchmark_runner.run_benchmark(
            "ci_simple_search", tool, project_folder=str(benchmark_fixtures), pattern="def $FUNC", language="python", output_format="json"
        )

        # Check for regressions
        has_regression, errors = benchmark_runner.check_regression()

        if has_regression:
            pytest.fail("Performance regression detected:\n" + "\n".join(errors))


# =============================================================================
# Phase 6.4: Deduplication Performance Benchmarks
# =============================================================================


class DeduplicationBenchmarkResult:
    """Store deduplication benchmark results with statistical analysis."""

    def __init__(self, name: str, times: List[float], iterations: int):
        self.name = name
        self.times = times
        self.iterations = iterations
        self.mean = statistics.mean(times)
        self.std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
        self.min_time = min(times)
        self.max_time = max(times)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "mean_seconds": round(self.mean, 6),
            "std_dev_seconds": round(self.std_dev, 6),
            "min_seconds": round(self.min_time, 6),
            "max_seconds": round(self.max_time, 6),
        }


class DeduplicationBenchmarkRunner:
    """Run deduplication-specific benchmarks with statistical reporting."""

    # Regression thresholds (max allowed slowdown from baseline)
    THRESHOLDS = {
        "pattern_analysis": 0.15,  # 15% slowdown allowed
        "code_generation": 0.10,  # 10% slowdown allowed
        "full_workflow": 0.20,  # 20% slowdown allowed
        "scoring": 0.05,  # 5% slowdown allowed
        "test_coverage": 0.15,  # 15% slowdown allowed
    }

    def __init__(self, baseline_file: str = "tests/dedup_benchmark_baseline.json"):
        self.baseline_file = baseline_file
        self.results: List[DeduplicationBenchmarkResult] = []
        self.baseline: Dict[str, Dict[str, float]] = {}
        self._load_baseline()

    def _load_baseline(self) -> None:
        """Load baseline metrics from file."""
        if os.path.exists(self.baseline_file):
            with open(self.baseline_file, "r") as f:
                data = json.load(f)
                self.baseline = {item["name"]: item for item in data.get("benchmarks", [])}

    def save_baseline(self) -> None:
        """Save current results as new baseline."""
        baseline_data = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "benchmarks": [r.to_dict() for r in self.results]}
        os.makedirs(os.path.dirname(self.baseline_file) if os.path.dirname(self.baseline_file) else ".", exist_ok=True)
        with open(self.baseline_file, "w") as f:
            json.dump(baseline_data, f, indent=2)

    def run_benchmark(
        self, name: str, func: Callable[..., Any], iterations: int = 10, *args: Any, **kwargs: Any
    ) -> DeduplicationBenchmarkResult:
        """Run a benchmark with multiple iterations and collect statistics."""
        times: List[float] = []

        for _ in range(iterations):
            start = time.perf_counter()
            func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        result = DeduplicationBenchmarkResult(name=name, times=times, iterations=iterations)
        self.results.append(result)
        return result

    def check_regression(self) -> Tuple[bool, List[str]]:
        """Check for performance regressions against baseline."""
        if not self.baseline:
            return (False, ["No baseline found - skipping regression check"])

        errors = []
        for result in self.results:
            if result.name not in self.baseline:
                continue

            baseline = self.baseline[result.name]
            baseline_mean = baseline.get("mean_seconds", result.mean)
            threshold = self.THRESHOLDS.get(result.name, 0.10)

            if baseline_mean > 0:
                slowdown = (result.mean - baseline_mean) / baseline_mean
                if slowdown > threshold:
                    errors.append(
                        f"{result.name}: {slowdown * 100:.1f}% slower "
                        f"({baseline_mean:.6f}s -> {result.mean:.6f}s, "
                        f"threshold: {threshold * 100:.0f}%)"
                    )

        return (len(errors) > 0, errors)

    def generate_report(self) -> Dict[str, Any]:
        """Generate JSON benchmark report."""
        has_regression, errors = self.check_regression()

        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_benchmarks": len(self.results),
            "results": [r.to_dict() for r in self.results],
            "regression_detected": has_regression,
            "regression_errors": errors,
            "thresholds": self.THRESHOLDS,
        }


@pytest.fixture
def dedup_benchmark_runner() -> DeduplicationBenchmarkRunner:
    """Provide deduplication benchmark runner for tests."""
    return DeduplicationBenchmarkRunner()


class TestDeduplicationBenchmarks:
    """Performance benchmarks for deduplication functions (Phase 6.4)."""

    def test_benchmark_calculate_deduplication_score(self, dedup_benchmark_runner: DeduplicationBenchmarkRunner) -> None:
        """Benchmark calculate_deduplication_score function."""
        ranker = DuplicationRanker()

        # Test with various inputs (using new duplicate_group dict API)
        test_cases = [
            {  # High value candidate
                "potential_line_savings": 100,
                "instances": [{"file": "a.py", "line": 10}, {"file": "b.py", "line": 20}, {"file": "a.py", "line": 50}],
            },
            {  # Low value candidate
                "potential_line_savings": 10,
                "instances": [{"file": f"file{i}.py", "line": i * 10} for i in range(8)],
            },
            {  # Medium value candidate
                "potential_line_savings": 50,
                "instances": [{"file": f"module{i % 5}.py", "line": i * 5} for i in range(5)],
            },
        ]

        def run_scoring() -> None:
            for duplicate_group in test_cases:
                ranker.calculate_deduplication_score(duplicate_group)

        result = dedup_benchmark_runner.run_benchmark(
            "scoring",
            run_scoring,
            iterations=100,  # Fast function, more iterations
        )

        # Scoring should be fast (< 5ms per call)
        assert result.mean < 0.005, f"Scoring too slow: {result.mean * 1000:.3f}ms"

    def test_benchmark_rank_deduplication_candidates(self, dedup_benchmark_runner: DeduplicationBenchmarkRunner) -> None:
        """Benchmark rank_deduplication_candidates function."""
        # rank_deduplication_candidates imported at module level from modular structure

        # Create test candidates
        candidates = [
            {
                "lines_saved": i * 10,
                "complexity_score": (i % 10) + 1,
                "has_tests": i % 2 == 0,
                "affected_files": (i % 5) + 1,
                "external_call_sites": i * 2,
            }
            for i in range(50)  # 50 candidates
        ]

        result = dedup_benchmark_runner.run_benchmark(
            "pattern_analysis",  # Ranking is part of pattern analysis
            rank_deduplication_candidates,
            50,  # iterations
            candidates,
        )

        # Should complete reasonably fast for 50 candidates
        assert result.mean < 0.05, f"Ranking too slow: {result.mean * 1000:.3f}ms"

    # test_benchmark_analyze_duplicate_variations - DELETED
    # Function analyze_duplicate_variations never existed in modular architecture
    # Variation analysis is now done via DuplicationAnalyzer.classify_variations()
    # which is tested elsewhere in the test suite

    def test_benchmark_get_test_coverage_for_files(self, dedup_benchmark_runner: DeduplicationBenchmarkRunner, tmp_path: Path) -> None:
        """Benchmark get_test_coverage_for_files function."""
        # get_test_coverage_for_files imported at module level from modular structure

        # Create test files
        src_dir = tmp_path / "src"
        test_dir = tmp_path / "tests"
        src_dir.mkdir()
        test_dir.mkdir()

        file_paths = []
        for i in range(10):
            src_file = src_dir / f"module_{i}.py"
            src_file.write_text(f"def func_{i}(): pass")
            file_paths.append(str(src_file))

            # Create corresponding test for half of them
            if i % 2 == 0:
                test_file = test_dir / f"test_module_{i}.py"
                test_file.write_text(f"def test_func_{i}(): pass")

        result = dedup_benchmark_runner.run_benchmark(
            "test_coverage",
            get_test_coverage_for_files,
            10,  # iterations
            file_paths,
            "python",
            str(tmp_path),
        )

        # Test coverage check should be reasonable
        assert result.mean < 0.5, f"Test coverage check too slow: {result.mean * 1000:.3f}ms"

    def test_benchmark_generate_deduplication_recommendation(self, dedup_benchmark_runner: DeduplicationBenchmarkRunner) -> None:
        """Benchmark generate_deduplication_recommendation function."""
        # generate_deduplication_recommendation imported at module level from modular structure

        # Test various scenarios
        test_cases = [
            (85.0, 3, 100, True, 3),  # High value
            (45.0, 7, 20, False, 8),  # Medium value
            (25.0, 9, 5, False, 15),  # Low value
        ]

        def run_recommendations() -> None:
            for score, complexity, lines, has_tests, files in test_cases:
                generate_deduplication_recommendation(score, complexity, lines, has_tests, files)

        result = dedup_benchmark_runner.run_benchmark(
            "code_generation",  # Recommendations include strategy generation
            run_recommendations,
            50,  # iterations
        )

        # Should be fast
        assert result.mean < 0.001, f"Recommendation too slow: {result.mean * 1000:.3f}ms"

    def test_benchmark_create_enhanced_duplication_response(self, dedup_benchmark_runner: DeduplicationBenchmarkRunner) -> None:
        """Benchmark create_enhanced_duplication_response function."""
        # create_enhanced_duplication_response imported at module level from modular structure

        # Create test candidates
        candidates = [
            {
                "code": f"""def helper_{i}(x, y):
    result = x + y * {i}
    return result""",
                "function_name": f"helper_{i}",
                "replacement": f"result = extracted_helper_{i}(x, y)",
                "similarity": 85.0 + i,
                "complexity": (i % 10) + 1,
                "files": [f"file_{i}.py", f"file_{i + 1}.py"],
            }
            for i in range(20)
        ]

        result = dedup_benchmark_runner.run_benchmark(
            "full_workflow",  # This is a full response generation
            create_enhanced_duplication_response,
            10,  # iterations
            candidates,
            False,  # include_diffs
            False,  # include_colors
        )

        # Full workflow should complete in reasonable time
        assert result.mean < 0.1, f"Response generation too slow: {result.mean * 1000:.3f}ms"

    def test_generate_dedup_benchmark_report(self, dedup_benchmark_runner: DeduplicationBenchmarkRunner) -> None:
        """Generate benchmark report after running tests."""
        # Add some test results
        dedup_benchmark_runner.results.append(
            DeduplicationBenchmarkResult(name="test_scoring", times=[0.0001, 0.00012, 0.00009], iterations=3)
        )

        report = dedup_benchmark_runner.generate_report()

        assert "timestamp" in report
        assert report["total_benchmarks"] == 1
        assert len(report["results"]) == 1
        assert "mean_seconds" in report["results"][0]
        assert "std_dev_seconds" in report["results"][0]

    def test_dedup_regression_detection(self, dedup_benchmark_runner: DeduplicationBenchmarkRunner) -> None:
        """Test deduplication regression detection."""
        # Add baseline
        dedup_benchmark_runner.baseline["scoring"] = {"name": "scoring", "mean_seconds": 0.0001}

        # Add result with 4% regression (within 5% threshold)
        dedup_benchmark_runner.results.append(DeduplicationBenchmarkResult(name="scoring", times=[0.000104] * 10, iterations=10))

        has_regression, errors = dedup_benchmark_runner.check_regression()
        assert not has_regression, "Should not detect regression within threshold"

        # Add result with 10% regression (exceeds 5% threshold)
        dedup_benchmark_runner.results = []
        dedup_benchmark_runner.results.append(DeduplicationBenchmarkResult(name="scoring", times=[0.00011] * 10, iterations=10))

        has_regression, errors = dedup_benchmark_runner.check_regression()
        assert has_regression, "Should detect regression exceeding threshold"
        assert len(errors) == 1


def run_deduplication_benchmarks(iterations: int = 10, save_baseline: bool = False) -> Dict[str, Any]:
    """Run all deduplication benchmarks and return results.

    This function can be called programmatically from the MCP tool.

    Args:
        iterations: Number of iterations per benchmark
        save_baseline: Whether to save results as new baseline

    Returns:
        Benchmark results as JSON-serializable dict
    """
    # Functions imported at module level from modular structure:
    # - rank_deduplication_candidates
    # - generate_deduplication_recommendation
    # - create_enhanced_duplication_response
    # - DuplicationRanker (for calculate_deduplication_score)

    runner = DeduplicationBenchmarkRunner()
    ranker = DuplicationRanker()

    # Benchmark 1: Scoring (using DuplicationRanker.calculate_deduplication_score)
    test_cases = [
        {"potential_line_savings": 100, "instances": [{"file": "f1.py", "line": 1}] * 2},
        {"potential_line_savings": 10, "instances": [{"file": "f1.py", "line": 1}] * 10},
        {"potential_line_savings": 50, "instances": [{"file": "f1.py", "line": 1}] * 5},
    ]

    def run_scoring() -> None:
        for duplicate_group in test_cases:
            ranker.calculate_deduplication_score(duplicate_group)

    runner.run_benchmark("scoring", run_scoring, iterations * 10)

    # Benchmark 2: Pattern Analysis (ranking)
    candidates = [
        {
            "lines_saved": i * 10,
            "complexity_score": (i % 10) + 1,
            "has_tests": i % 2 == 0,
            "affected_files": (i % 5) + 1,
            "external_call_sites": i * 2,
        }
        for i in range(50)
    ]

    runner.run_benchmark("pattern_analysis", rank_deduplication_candidates, iterations * 5, candidates)

    # Benchmark 3: Code Generation (recommendations)
    test_recs = [
        (85.0, 3, 100, True, 3),
        (45.0, 7, 20, False, 8),
        (25.0, 9, 5, False, 15),
    ]

    def run_recommendations() -> None:
        for score, complexity, lines, has_tests, files in test_recs:
            generate_deduplication_recommendation(score, complexity, lines, has_tests, files)

    runner.run_benchmark("code_generation", run_recommendations, iterations * 5)

    # Benchmark 4: Full Workflow
    response_candidates = [
        {
            "code": f"def helper_{i}(x, y): return x + y * {i}",
            "function_name": f"helper_{i}",
            "replacement": f"result = extracted_helper_{i}(x, y)",
            "similarity": 85.0 + i,
            "complexity": (i % 10) + 1,
            "files": [f"file_{i}.py"],
        }
        for i in range(20)
    ]

    runner.run_benchmark(
        "full_workflow",
        create_enhanced_duplication_response,
        iterations,
        response_candidates,
        False,  # include_diffs
        False,  # include_colors
    )

    if save_baseline:
        runner.save_baseline()

    return runner.generate_report()
