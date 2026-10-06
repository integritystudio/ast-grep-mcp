"""Test coverage detection for deduplication analysis."""

import glob as glob_module
import os
import re as regex_module
from typing import Callable, Dict, List, Optional, Set, Tuple

from ...constants import ParallelProcessing
from ...core.logging import get_logger
from ...utils.futures import WaitTimeoutError, map_with_per_item_timeout

__all__ = [
    "CoverageDetector",
]


# Helper function for JavaScript-like languages
def _get_javascript_patterns(source_name: str) -> List[str]:
    """Get import patterns for JavaScript-like languages."""
    return [
        f"from ['\"].*{source_name}['\"]",
        f"require\\(['\"].*{source_name}['\"]\\)",
        f"import.*{source_name}",
    ]


# Helper function for Ruby-like patterns
def _get_ruby_patterns(source_name: str) -> List[str]:
    """Get import patterns for Ruby."""
    return [f"require.*{source_name}"]


# Configuration-driven pattern for import patterns
IMPORT_PATTERN_CONFIG: Dict[str, Callable[[str], List[str]]] = {
    "python": lambda source_name: [
        f"from {source_name} import",
        f"from .{source_name} import",
        f"import {source_name}",
        f"from.*{source_name}.*import",
    ],
    "javascript": _get_javascript_patterns,
    "js": _get_javascript_patterns,
    "typescript": _get_javascript_patterns,
    "ts": _get_javascript_patterns,
    "tsx": _get_javascript_patterns,
    "java": lambda source_name: [
        f"import.*\\.{source_name};",
        f"\\b{source_name}\\b",  # Direct class usage
    ],
    "ruby": _get_ruby_patterns,
    "rb": _get_ruby_patterns,
}


_GENERIC_TEST_PATTERNS: List[str] = [
    "**/test*",
    "**/tests/*",
    "**/*test*",
    "**/*spec*",
]

_TEST_FILE_PATTERNS: Dict[str, List[str]] = {
    "python": [
        "test_*.py",
        "*_test.py",
        "tests/*.py",
        "**/tests/*.py",
        "**/test/*.py",
        "**/*_test.py",
        "**/test_*.py",
    ],
    "javascript": [
        "*.test.js",
        "*.spec.js",
        "__tests__/*.js",
        "**/__tests__/*.js",
        "**/tests/*.js",
        "**/test/*.js",
        "**/*.test.js",
        "**/*.spec.js",
    ],
    "typescript": [
        "*.test.ts",
        "*.test.tsx",
        "*.spec.ts",
        "*.spec.tsx",
        "__tests__/*.ts",
        "__tests__/*.tsx",
        "**/__tests__/*.ts",
        "**/__tests__/*.tsx",
        "**/tests/*.ts",
        "**/tests/*.tsx",
        "**/*.test.ts",
        "**/*.test.tsx",
        "**/*.spec.ts",
        "**/*.spec.tsx",
    ],
    "java": [
        "*Test.java",
        "*Tests.java",
        "**/*Test.java",
        "**/*Tests.java",
        "**/test/**/*.java",
        "src/test/**/*.java",
    ],
    "go": [
        "*_test.go",
        "**/*_test.go",
    ],
    "ruby": [
        "*_test.rb",
        "*_spec.rb",
        "test/*.rb",
        "spec/*.rb",
        "**/test/*.rb",
        "**/spec/*.rb",
        "**/*_test.rb",
        "**/*_spec.rb",
    ],
    "rust": [
        "**/tests/*.rs",
        "**/tests/**/*.rs",
    ],
    "csharp": [
        "*Tests.cs",
        "*Test.cs",
        "**/*Tests.cs",
        "**/*Test.cs",
        "**/Tests/**/*.cs",
    ],
}


# Language aliases
def _python_test_paths(name: str, dir_path: str, root: str, _ext: str) -> List[str]:
    return [
        os.path.join(root, dir_path, f"test_{name}.py"),
        os.path.join(root, dir_path, f"{name}_test.py"),
        os.path.join(root, "tests", f"test_{name}.py"),
        os.path.join(root, "tests", dir_path, f"test_{name}.py"),
        os.path.join(root, "test", f"test_{name}.py"),
        os.path.join(root, "tests", "unit", f"test_{name}.py"),
        os.path.join(root, "tests", "integration", f"test_{name}.py"),
    ]


def _js_test_paths(name: str, dir_path: str, root: str, _ext: str) -> List[str]:
    return [
        os.path.join(root, dir_path, f"{name}.test.js"),
        os.path.join(root, dir_path, f"{name}.spec.js"),
        os.path.join(root, dir_path, "__tests__", f"{name}.js"),
        os.path.join(root, "tests", f"{name}.test.js"),
        os.path.join(root, "__tests__", f"{name}.js"),
    ]


def _ts_test_paths(name: str, dir_path: str, root: str, ext: str) -> List[str]:
    ts_ext = ".tsx" if ext == ".tsx" else ".ts"
    return [
        os.path.join(root, dir_path, f"{name}.test{ts_ext}"),
        os.path.join(root, dir_path, f"{name}.spec{ts_ext}"),
        os.path.join(root, dir_path, "__tests__", f"{name}{ts_ext}"),
        os.path.join(root, "tests", f"{name}.test{ts_ext}"),
        os.path.join(root, "__tests__", f"{name}{ts_ext}"),
    ]


def _java_test_paths(name: str, dir_path: str, root: str, _ext: str) -> List[str]:
    maven_dir = dir_path.replace("src/main/java/", "").replace("src\\main\\java\\", "")
    return [
        os.path.join(root, dir_path, f"{name}Test.java"),
        os.path.join(root, dir_path, f"{name}Tests.java"),
        os.path.join(root, "src", "test", "java", maven_dir, f"{name}Test.java"),
    ]


def _go_test_paths(name: str, dir_path: str, root: str, _ext: str) -> List[str]:
    return [os.path.join(root, dir_path, f"{name}_test.go")]


def _ruby_test_paths(name: str, dir_path: str, root: str, _ext: str) -> List[str]:
    return [
        os.path.join(root, dir_path, f"{name}_test.rb"),
        os.path.join(root, dir_path, f"{name}_spec.rb"),
        os.path.join(root, "test", f"{name}_test.rb"),
        os.path.join(root, "spec", f"{name}_spec.rb"),
    ]


_TEST_PATH_BUILDERS: Dict[str, Callable[[str, str, str, str], List[str]]] = {
    "python": _python_test_paths,
    "javascript": _js_test_paths,
    "js": _js_test_paths,
    "typescript": _ts_test_paths,
    "ts": _ts_test_paths,
    "tsx": _ts_test_paths,
    "java": _java_test_paths,
    "go": _go_test_paths,
    "ruby": _ruby_test_paths,
    "rb": _ruby_test_paths,
}

_TEST_FILE_PATTERNS["js"] = _TEST_FILE_PATTERNS["javascript"]
_TEST_FILE_PATTERNS["ts"] = _TEST_FILE_PATTERNS["typescript"]
_TEST_FILE_PATTERNS["tsx"] = _TEST_FILE_PATTERNS["typescript"]
_TEST_FILE_PATTERNS["rb"] = _TEST_FILE_PATTERNS["ruby"]
_TEST_FILE_PATTERNS["rs"] = _TEST_FILE_PATTERNS["rust"]
_TEST_FILE_PATTERNS["cs"] = _TEST_FILE_PATTERNS["csharp"]
_TEST_FILE_PATTERNS["c#"] = _TEST_FILE_PATTERNS["csharp"]


class CoverageDetector:
    """Detects test coverage for source files to assess refactoring risk."""

    def __init__(self) -> None:
        """Initialize the test coverage detector."""
        self.logger = get_logger("deduplication.test_coverage")

    def find_test_file_patterns(self, language: str) -> List[str]:
        """Get test file patterns for a given programming language.

        Args:
            language: Programming language (python, javascript, typescript, java, etc.)

        Returns:
            List of glob patterns for test files in that language
        """
        return _TEST_FILE_PATTERNS.get(language.lower(), _GENERIC_TEST_PATTERNS)

    def _get_potential_test_paths(self, file_path: str, language: str, project_root: str) -> List[str]:
        """Generate potential test file paths for a source file.

        Args:
            file_path: Path to the source file
            language: Programming language
            project_root: Root directory of the project

        Returns:
            List of potential test file paths that could contain tests for this file
        """
        try:
            rel_path = os.path.relpath(file_path, project_root)
        except ValueError:
            rel_path = os.path.basename(file_path)

        basename = os.path.basename(file_path)
        name_without_ext, ext = os.path.splitext(basename)
        dir_path = os.path.dirname(rel_path)

        builder = _TEST_PATH_BUILDERS.get(language.lower())
        if builder is None:
            return []
        return [os.path.normpath(p) for p in builder(name_without_ext, dir_path, project_root, ext)]

    def _read_file_content(self, file_path: str) -> Optional[str]:
        """Read file content safely.

        Args:
            file_path: Path to the file

        Returns:
            File content or None if file cannot be read
        """
        if not os.path.exists(file_path):
            return None

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except (IOError, OSError):
            return None

    def _check_import_patterns(self, patterns: List[str], content: str) -> bool:
        """Check if any import pattern matches in content.

        Args:
            patterns: List of regex patterns to check
            content: File content to search in

        Returns:
            True if any pattern matches
        """
        for pattern in patterns:
            if regex_module.search(pattern, content, regex_module.IGNORECASE):
                return True
        return False

    def _check_go_same_directory(self, test_file_path: str, source_file_path: str) -> bool:
        """Check if Go files are in the same directory (automatic access).

        Args:
            test_file_path: Path to test file
            source_file_path: Path to source file

        Returns:
            True if files are in the same directory
        """
        source_dir = os.path.dirname(source_file_path)
        test_dir = os.path.dirname(test_file_path)
        return os.path.normpath(source_dir) == os.path.normpath(test_dir)

    def _check_test_file_references_source(
        self, test_file_path: str, source_file_path: str, language: str, content: Optional[str] = None
    ) -> bool:
        """Check if a test file references/imports the source file.

        Args:
            test_file_path: Path to the test file
            source_file_path: Path to the source file being tested
            language: Programming language
            content: Pre-read test file content; read from disk when omitted

        Returns:
            True if the test file appears to test the source file
        """
        # Early return: read file content
        if content is None:
            content = self._read_file_content(test_file_path)
        if content is None:
            return False

        # Get the module/class name from source file
        source_name = os.path.splitext(os.path.basename(source_file_path))[0]
        # File names may contain regex metacharacters (e.g. "util(1).py")
        name_pattern = regex_module.escape(source_name)
        lang = language.lower()

        # Special case: Go files in same directory have automatic access
        if lang == "go":
            if self._check_go_same_directory(test_file_path, source_file_path):
                return True
            # Check for package import
            if regex_module.search(f"import.*{name_pattern}", content, regex_module.IGNORECASE):
                return True

        # Configuration-driven pattern checking
        if lang in IMPORT_PATTERN_CONFIG:
            patterns = IMPORT_PATTERN_CONFIG[lang](name_pattern)
            if self._check_import_patterns(patterns, content):
                return True

        # Fallback: source file name appears as a whole word anywhere in the test
        if regex_module.search(rf"\b{name_pattern}\b", content, regex_module.IGNORECASE):
            return True

        return False

    def has_test_coverage(self, file_path: str, language: str, project_root: str) -> bool:
        """Check if a source file has corresponding test coverage.

        Args:
            file_path: Path to the source file
            language: Programming language
            project_root: Root directory of the project

        Returns:
            True if test coverage exists for the file
        """
        # Get potential test file paths
        potential_tests = self._get_potential_test_paths(file_path, language, project_root)

        # Check if any potential test file exists
        for test_path in potential_tests:
            if os.path.exists(test_path):
                self.logger.debug("found_test_file", source_file=file_path, test_file=test_path)
                return True

        # Also search using glob patterns for more flexible matching
        patterns = self.find_test_file_patterns(language)

        for pattern in patterns:
            full_pattern = os.path.join(project_root, pattern)
            try:
                matches = glob_module.glob(full_pattern, recursive=True)
                for match in matches:
                    # Check if this test file references our source
                    if self._check_test_file_references_source(match, file_path, language):
                        self.logger.debug("found_test_by_reference", source_file=file_path, test_file=match)
                        return True
            except Exception as e:
                self.logger.warning("glob_search_failed", pattern=pattern, error=str(e))
                continue

        self.logger.debug("no_test_coverage", source_file=file_path)
        return False

    def _find_all_test_files(self, language: str, project_root: str) -> Set[str]:
        """Find all test files in the project (cached for batch operations).

        Args:
            language: Programming language
            project_root: Root directory of the project

        Returns:
            Set of normalized test file paths
        """
        test_files: Set[str] = set()
        patterns = self.find_test_file_patterns(language)

        for pattern in patterns:
            full_pattern = os.path.join(project_root, pattern)
            try:
                matches = glob_module.glob(full_pattern, recursive=True)
                # Normalize paths for consistent comparison
                test_files.update(os.path.normpath(match) for match in matches)
            except Exception as e:
                self.logger.warning("glob_search_failed", pattern=pattern, error=str(e))
                continue

        self.logger.debug("test_files_discovered", count=len(test_files), language=language)

        return test_files

    def _read_test_file_contents(self, test_files: Set[str]) -> Dict[str, Optional[str]]:
        """Read all test file contents once for reuse across a batch (BUG-11).

        Args:
            test_files: Set of test file paths

        Returns:
            Mapping of test file path to its content (None if unreadable)
        """
        return {test_file: self._read_file_content(test_file) for test_file in test_files}

    def _has_test_coverage_optimized(
        self,
        file_path: str,
        language: str,
        project_root: str,
        test_files: Set[str],
        content_cache: Optional[Dict[str, Optional[str]]] = None,
    ) -> bool:
        """Optimized test coverage check using pre-computed test file set.

        Args:
            file_path: Path to the source file
            language: Programming language
            project_root: Root directory of the project
            test_files: Pre-computed set of all test files
            content_cache: Pre-read test file contents keyed by path; files are
                read from disk on each call when omitted

        Returns:
            True if test coverage exists for the file
        """
        # Get potential test file paths
        potential_tests = self._get_potential_test_paths(file_path, language, project_root)

        # Check if any potential test file exists in our pre-computed set
        for test_path in potential_tests:
            normalized_test = os.path.normpath(test_path)
            if normalized_test in test_files:
                self.logger.debug("found_test_file", source_file=file_path, test_file=normalized_test)
                return True

        # Check if any test file references our source
        for test_file in test_files:
            if content_cache is not None:
                content = content_cache.get(test_file)
                if content is None:  # unreadable at cache-build time; don't re-read per file
                    continue
            else:
                content = None
            if self._check_test_file_references_source(test_file, file_path, language, content=content):
                self.logger.debug("found_test_by_reference", source_file=file_path, test_file=test_file)
                return True

        self.logger.debug("no_test_coverage", source_file=file_path)
        return False

    def get_test_coverage_for_files(self, file_paths: List[str], language: str, project_root: str) -> Dict[str, bool]:
        """Get test coverage status for multiple files (sequential).

        Args:
            file_paths: List of source file paths
            language: Programming language
            project_root: Root directory of the project

        Returns:
            Dictionary mapping file paths to their test coverage status

        Note:
            For better performance with many files, use
            get_test_coverage_for_files_batch() with parallel=True.
        """
        coverage_map: Dict[str, bool] = {}
        covered_count = 0

        for file_path in file_paths:
            has_coverage = self.has_test_coverage(file_path, language, project_root)
            coverage_map[file_path] = has_coverage
            if has_coverage:
                covered_count += 1

        self.logger.info(
            "test_coverage_analysis_complete",
            total_files=len(file_paths),
            files_with_coverage=covered_count,
            files_without_coverage=len(file_paths) - covered_count,
        )

        return coverage_map

    def get_test_coverage_for_files_batch(
        self,
        file_paths: List[str],
        language: str,
        project_root: str,
        parallel: bool = True,
        max_workers: int = ParallelProcessing.DEFAULT_WORKERS,
        timeout_per_file: Optional[float] = None,
    ) -> Dict[str, bool]:
        """Get test coverage status for multiple files with batch optimization.

        This method provides significant performance improvements over the
        sequential version by:
        1. Pre-computing all test files once (instead of per-file glob searches)
        2. Optionally using parallel execution for file processing
        3. Reusing the test file set across all files

        Args:
            file_paths: List of source file paths
            language: Programming language
            project_root: Root directory of the project
            parallel: Whether to use parallel execution (default: True)
            max_workers: Maximum number of threads for parallel execution
            timeout_per_file: Per-file timeout in seconds; defaults to
                ParallelProcessing.DEFAULT_TIMEOUT_PER_CANDIDATE_SECONDS

        Returns:
            Dictionary mapping file paths to their test coverage status

        Performance:
            - Sequential: O(n * m) where n=files, m=test patterns
            - Batch: O(m + n) - 60-80% faster for large file sets
        """
        if not file_paths:
            return {}

        # Pre-compute all test files and their contents once (major optimization)
        test_files = self._find_all_test_files(language, project_root)
        content_cache = self._read_test_file_contents(test_files)

        # Process files using appropriate strategy
        if parallel and len(file_paths) > 1:
            coverage_map, covered_count = self._process_parallel_batch(
                file_paths, language, project_root, test_files, max_workers, timeout_per_file, content_cache
            )
        else:
            coverage_map, covered_count = self._process_sequential_batch(
                file_paths, language, project_root, test_files, content_cache
            )

        # Log final results
        self._log_batch_results(len(file_paths), covered_count, parallel, len(test_files))

        return coverage_map

    def _process_file_coverage(
        self,
        file_path: str,
        language: str,
        project_root: str,
        test_files: Set[str],
        content_cache: Optional[Dict[str, Optional[str]]] = None,
    ) -> bool:
        """Process coverage check for a single file with error handling.

        Args:
            file_path: Path to the source file
            language: Programming language
            project_root: Root directory
            test_files: Pre-computed set of test files
            content_cache: Pre-read test file contents keyed by path

        Returns:
            True if test coverage exists, False otherwise or on error
        """
        try:
            return self._has_test_coverage_optimized(file_path, language, project_root, test_files, content_cache)
        except Exception as e:
            self.logger.error("test_coverage_check_failed", file_path=file_path, error=str(e))
            return False

    def _process_parallel_batch(
        self,
        file_paths: List[str],
        language: str,
        project_root: str,
        test_files: Set[str],
        max_workers: int,
        timeout_per_file: Optional[float] = None,
        content_cache: Optional[Dict[str, Optional[str]]] = None,
    ) -> Tuple[Dict[str, bool], int]:
        """Process files in parallel for coverage checking with per-file timeout.

        Args:
            file_paths: List of source file paths
            language: Programming language
            project_root: Root directory
            test_files: Pre-computed set of test files
            max_workers: Maximum number of threads
            timeout_per_file: Per-file timeout in seconds; defaults to
                ParallelProcessing.DEFAULT_TIMEOUT_PER_CANDIDATE_SECONDS
            content_cache: Pre-read test file contents keyed by path

        Returns:
            Tuple of (coverage_map, covered_count)
        """
        self.logger.debug("batch_coverage_parallel_start", file_count=len(file_paths), max_workers=max_workers)

        timeout = timeout_per_file if timeout_per_file is not None else ParallelProcessing.DEFAULT_TIMEOUT_PER_CANDIDATE_SECONDS
        coverage_map: Dict[str, bool] = {}

        def check_file(file_path: str) -> bool:
            return self._has_test_coverage_optimized(file_path, language, project_root, test_files, content_cache)

        def on_success(file_path: str, has_coverage: bool) -> None:
            coverage_map[file_path] = has_coverage

        def on_error(file_path: str, error: Exception) -> None:
            if isinstance(error, WaitTimeoutError):
                self.logger.error("test_coverage_check_timed_out", file_path=file_path, timeout_seconds=timeout)
            else:
                self.logger.error("test_coverage_check_failed", file_path=file_path, error=str(error) or type(error).__name__)
            coverage_map[file_path] = False

        map_with_per_item_timeout(
            file_paths,
            check_file,
            timeout_seconds=timeout,
            max_workers=max_workers,
            on_success=on_success,
            on_error=on_error,
        )

        covered_count = sum(1 for has_coverage in coverage_map.values() if has_coverage)
        return coverage_map, covered_count

    def _process_sequential_batch(
        self,
        file_paths: List[str],
        language: str,
        project_root: str,
        test_files: Set[str],
        content_cache: Optional[Dict[str, Optional[str]]] = None,
    ) -> Tuple[Dict[str, bool], int]:
        """Process files sequentially for coverage checking.

        Args:
            file_paths: List of source file paths
            language: Programming language
            project_root: Root directory
            test_files: Pre-computed set of test files
            content_cache: Pre-read test file contents keyed by path

        Returns:
            Tuple of (coverage_map, covered_count)
        """
        self.logger.debug("batch_coverage_sequential_start", file_count=len(file_paths))

        coverage_map: Dict[str, bool] = {}
        covered_count = 0

        for file_path in file_paths:
            has_coverage = self._process_file_coverage(file_path, language, project_root, test_files, content_cache)
            coverage_map[file_path] = has_coverage
            if has_coverage:
                covered_count += 1

        return coverage_map, covered_count

    def _log_batch_results(self, total_files: int, covered_count: int, parallel: bool, test_files_count: int) -> None:
        """Log the final batch processing results.

        Args:
            total_files: Total number of files processed
            covered_count: Number of files with coverage
            parallel: Whether parallel processing was used
            test_files_count: Number of test files found
        """
        self.logger.info(
            "batch_coverage_analysis_complete",
            total_files=total_files,
            files_with_coverage=covered_count,
            files_without_coverage=total_files - covered_count,
            parallel=parallel,
            test_files_found=test_files_count,
        )
