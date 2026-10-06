"""Orphan code detection service.

This module provides functionality to detect orphan files and functions
in a codebase using dependency graph analysis and verification.
"""

import ast
import fnmatch
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ast_grep_mcp.constants import ConversionFactors, FilePatterns, SemanticVolumeDefaults, SubprocessDefaults
from ast_grep_mcp.core.logging import get_logger
from ast_grep_mcp.models.orphan import (
    DependencyEdge,
    DependencyGraph,
    OrphanAnalysisConfig,
    OrphanAnalysisResult,
    OrphanFile,
    OrphanFunction,
    VerificationStatus,
)

_GREP_EXCLUDE_DIRS = ("node_modules", ".git", "dist", "build", "__pycache__", ".venv", "venv", "coverage", ".next", ".turbo")


class OrphanDetector:
    """Detects orphan files and functions in a codebase."""

    def __init__(self, config: Optional[OrphanAnalysisConfig] = None) -> None:
        """Initialize the detector with optional configuration."""
        self.config = config or OrphanAnalysisConfig()
        self.logger = get_logger("orphan_detector")

    def analyze(self, project_folder: str) -> OrphanAnalysisResult:
        """Analyze a project for orphan code.

        Args:
            project_folder: Path to the project root

        Returns:
            OrphanAnalysisResult with detected orphans
        """
        start_time = time.time()
        base_path = Path(project_folder)
        self.logger.info("orphan_analysis_started", project=project_folder)

        graph = self._build_dependency_graph(base_path)
        self._identify_entry_points(base_path, graph)
        orphan_files = self._find_orphan_files(base_path, graph)
        if self.config.verify_with_grep:
            orphan_files = self._verify_orphans_with_grep(base_path, orphan_files)

        orphan_functions: List[OrphanFunction] = []
        total_functions = 0
        if self.config.analyze_functions:
            orphan_functions, total_functions = self._find_orphan_functions(base_path, graph)

        elapsed_ms = int((time.time() - start_time) * ConversionFactors.MILLISECONDS_PER_SECOND)
        result = OrphanAnalysisResult(
            orphan_files=orphan_files,
            orphan_functions=orphan_functions,
            total_files_analyzed=len(graph.files),
            total_functions_analyzed=total_functions,
            dependency_graph=graph,
            analysis_time_ms=elapsed_ms,
            config=self.config,
        )
        self.logger.info(
            "orphan_analysis_complete",
            orphan_files=len(orphan_files),
            orphan_functions=len(orphan_functions),
            total_files=len(graph.files),
            elapsed_ms=elapsed_ms,
        )
        return result

    def _build_dependency_graph(self, base_path: Path) -> DependencyGraph:
        """Build the import dependency graph for the project.

        Two-pass approach: first collect all files, then extract imports.
        This ensures target files are in graph.files before edge resolution.
        """
        graph = DependencyGraph()

        # Pass 1: collect all files
        all_files: List[Tuple[Path, str]] = []
        for pattern in self.config.include_patterns:
            glob_pattern = pattern.split("**/")[-1] if "**/" in pattern else pattern
            for file_path in base_path.rglob(glob_pattern):
                if self._should_exclude(file_path, base_path) or not file_path.is_file():
                    continue
                rel_path = str(file_path.relative_to(base_path))
                graph.files.add(rel_path)
                all_files.append((file_path, rel_path))

        # Pass 2: extract imports (all targets now resolvable)
        for file_path, rel_path in all_files:
            if file_path.suffix == ".py":
                self._extract_python_imports(file_path, rel_path, base_path, graph)
            elif file_path.suffix in (".ts", ".js", ".tsx", ".jsx"):
                self._extract_js_imports(file_path, rel_path, base_path, graph)

        graph.build_index()
        self.logger.debug("dependency_graph_built", files=len(graph.files), edges=len(graph.edges))
        return graph

    def _should_exclude(self, file_path: Path, base_path: Path) -> bool:
        """Check if a file should be excluded from analysis."""
        rel_path = str(file_path.relative_to(base_path))
        return any(self._matches_exclude_pattern(file_path, rel_path, p) for p in self.config.exclude_patterns)

    def _matches_exclude_pattern(self, file_path: Path, rel_path: str, pattern: str) -> bool:
        """Return True if the file matches a single exclude pattern."""
        if fnmatch.fnmatch(rel_path, pattern.lstrip("**/")):
            return True
        if fnmatch.fnmatch(str(file_path), pattern):
            return True
        clean = pattern.replace("**/", "").replace("/**", "")
        return any(fnmatch.fnmatch(part, clean) for part in file_path.parts)

    def _extract_python_imports(self, file_path: Path, rel_path: str, base_path: Path, graph: DependencyGraph) -> None:
        """Extract imports from a Python file."""
        tree = self._parse_python_file(file_path, rel_path)
        if tree is None:
            return
        external_imports: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self._process_import_node(node, rel_path, base_path, graph, external_imports)
            elif isinstance(node, ast.ImportFrom):
                self._process_import_from_node(node, file_path, rel_path, base_path, graph, external_imports)
        if external_imports:
            graph.external_imports[rel_path] = external_imports

    def _parse_python_file(self, file_path: Path, rel_path: str) -> Optional[ast.AST]:
        """Parse a Python file, returning None on failure."""
        try:
            return ast.parse(file_path.read_text(encoding="utf-8"))
        except Exception as e:
            self.logger.debug("parse_failed", file=rel_path, error=str(e))
            return None

    def _process_import_node(
        self,
        node: ast.Import,
        rel_path: str,
        base_path: Path,
        graph: DependencyGraph,
        external_imports: Set[str],
    ) -> None:
        """Process an ast.Import node."""
        for alias in node.names:
            self._register_import(alias.name, rel_path, f"import {alias.name}", "absolute", base_path, graph, external_imports)

    def _register_import(
        self,
        module_name: str,
        rel_path: str,
        import_stmt: str,
        import_type: str,
        base_path: Path,
        graph: DependencyGraph,
        external_imports: Set[str],
    ) -> None:
        """Add a resolved import edge or record it as external."""
        target = self._resolve_python_import(module_name, base_path)
        if target and target in graph.files:
            graph.edges.append(DependencyEdge(source=rel_path, target=target, import_type=import_type, import_statement=import_stmt))
        else:
            external_imports.add(module_name.split(".")[0])

    def _process_import_from_node(
        self,
        node: ast.ImportFrom,
        file_path: Path,
        rel_path: str,
        base_path: Path,
        graph: DependencyGraph,
        external_imports: Set[str],
    ) -> None:
        """Process an ast.ImportFrom node."""
        if not node.module:
            self._process_bare_relative_import(node, file_path, rel_path, base_path, graph)
            return
        import_type = "relative" if node.level > 0 else "absolute"
        target = self._resolve_import_from_target(node, file_path, base_path)
        if target and target in graph.files:
            graph.edges.append(
                DependencyEdge(
                    source=rel_path,
                    target=target,
                    import_type=import_type,
                    import_statement=f"from {node.module} import ...",
                )
            )
        elif node.module:
            external_imports.add(node.module.split(".")[0])

    def _process_bare_relative_import(
        self, node: ast.ImportFrom, file_path: Path, rel_path: str, base_path: Path, graph: DependencyGraph
    ) -> None:
        """`from . import sibling` names modules directly; each alias may be a file in the package."""
        package_dir = self._ascend_dirs(file_path.parent, node.level - 1)
        for alias in node.names:
            target = self._resolve_relative_parts([alias.name], package_dir, base_path)
            if target and target in graph.files:
                stmt = f"from {'.' * node.level} import {alias.name}"
                graph.edges.append(DependencyEdge(source=rel_path, target=target, import_type="relative", import_statement=stmt))

    def _resolve_import_from_target(self, node: ast.ImportFrom, file_path: Path, base_path: Path) -> Optional[str]:
        """Resolve the target file for an ImportFrom node."""
        if node.level > 0:
            return self._resolve_relative_import(file_path, node.module or "", node.level, base_path)
        return self._resolve_python_import(node.module or "", base_path)

    def _resolve_python_import(self, module_name: str, base_path: Path) -> Optional[str]:
        """Resolve a Python module name to a file path."""
        parts = module_name.split(".")

        pkg_path = base_path / "/".join(parts) / "__init__.py"
        if pkg_path.exists():
            return str(pkg_path.relative_to(base_path))

        mod_path = base_path / "/".join(parts[:-1]) / f"{parts[-1]}.py" if len(parts) > 1 else base_path / f"{parts[0]}.py"
        if mod_path.exists():
            return str(mod_path.relative_to(base_path))

        return self._resolve_python_import_with_prefix(parts, base_path)

    def _resolve_python_import_with_prefix(self, parts: List[str], base_path: Path) -> Optional[str]:
        """Try resolving a Python module under src/ and root prefixes."""
        for prefix in ["src/", ""]:
            result = self._resolve_module_with_one_prefix(parts, base_path, prefix)
            if result:
                return result
        return None

    def _resolve_module_with_one_prefix(self, parts: List[str], base_path: Path, prefix: str) -> Optional[str]:
        """Try package then module resolution under a single prefix."""
        pkg_path = base_path / prefix / "/".join(parts) / "__init__.py"
        if pkg_path.exists():
            return str(pkg_path.relative_to(base_path))
        mod_path = (base_path / prefix / "/".join(parts)).with_suffix(".py")
        if mod_path.exists():
            return str(mod_path.relative_to(base_path))
        return None

    def _resolve_relative_import(self, file_path: Path, module: str, level: int, base_path: Path) -> Optional[str]:
        """Resolve a relative Python import."""
        parts = module.split(".") if module else []
        if not parts:
            return None
        return self._resolve_relative_parts(parts, self._ascend_dirs(file_path.parent, level - 1), base_path)

    @staticmethod
    def _resolve_relative_parts(parts: List[str], current_dir: Path, base_path: Path) -> Optional[str]:
        """Try module file then package path relative to current_dir."""
        joined = "/".join(parts)
        mod_path = current_dir / "/".join(parts[:-1]) / f"{parts[-1]}.py" if len(parts) > 1 else current_dir / f"{parts[0]}.py"
        if mod_path.exists():
            return str(mod_path.relative_to(base_path))
        pkg_path = current_dir / joined / "__init__.py"
        if pkg_path.exists():
            return str(pkg_path.relative_to(base_path))
        return None

    @staticmethod
    def _ascend_dirs(directory: Path, levels: int) -> Path:
        """Walk up `levels` parent directories."""
        for _ in range(levels):
            directory = directory.parent
        return directory

    def _extract_js_imports(self, file_path: Path, rel_path: str, base_path: Path, graph: DependencyGraph) -> None:
        """Extract imports from a JavaScript/TypeScript file."""
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            self.logger.debug("read_failed", file=rel_path, error=str(e))
            return

        external_imports: Set[str] = set()

        # Match ES6 imports and re-exports: import/export X from 'path', import { X } from 'path'
        import_pattern = r"(?:import|export)\s+(?:[\w\s{},*]+\s+from\s+)?['\"]([^'\"]+)['\"]"

        # Match require('path') and dynamic import('path')
        require_pattern = r"(?:require|import)\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"

        for pattern in [import_pattern, require_pattern]:
            self._process_js_pattern_matches(pattern, content, file_path, rel_path, base_path, graph, external_imports)

        if external_imports:
            graph.external_imports[rel_path] = external_imports

    def _process_js_pattern_matches(
        self,
        pattern: str,
        content: str,
        file_path: Path,
        rel_path: str,
        base_path: Path,
        graph: DependencyGraph,
        external_imports: Set[str],
    ) -> None:
        """Process all matches of a JS import pattern."""
        for match in re.finditer(pattern, content):
            import_path = match.group(1)
            self._process_single_js_import(import_path, match.group(0), file_path, rel_path, base_path, graph, external_imports)

    def _process_single_js_import(
        self,
        import_path: str,
        import_statement: str,
        file_path: Path,
        rel_path: str,
        base_path: Path,
        graph: DependencyGraph,
        external_imports: Set[str],
    ) -> None:
        """Process a single JS import path."""
        if not import_path.startswith("."):
            external_imports.add(import_path.split("/")[0])
            return
        target = self._resolve_js_relative_import(file_path, import_path, base_path)
        if target and target in graph.files:
            graph.edges.append(DependencyEdge(source=rel_path, target=target, import_type="relative", import_statement=import_statement))

    _JS_EXTENSIONS = [".ts", ".tsx", ".js", ".jsx", "/index.ts", "/index.js"]

    def _resolve_js_relative_import(self, file_path: Path, import_path: str, base_path: Path) -> Optional[str]:
        """Resolve a relative JavaScript/TypeScript import.

        Handles: extensionless imports, .js→.ts remapping (TS ESM convention),
        and index file resolution.
        """
        target_path = (file_path.parent / import_path).resolve()
        candidates = [Path(str(target_path) + ext) for ext in self._JS_EXTENSIONS] + [target_path]
        # TS ESM convention: source imports use .js but files on disk are .ts
        if import_path.endswith(".js"):
            ts_path = (file_path.parent / import_path[:-3]).resolve()
            candidates = [Path(str(ts_path) + ext) for ext in (".ts", ".tsx")] + candidates
        elif import_path.endswith(".jsx"):
            tsx_path = (file_path.parent / import_path[:-4]).resolve()
            candidates = [Path(str(tsx_path) + ".tsx")] + candidates
        return self._first_relative_path(candidates, base_path)

    def _first_relative_path(self, candidates: List[Path], base_path: Path) -> Optional[str]:
        """Return the relative path string for the first existing candidate."""
        for path in candidates:
            rel = self._safe_relative(path, base_path)
            if rel is not None:
                return rel
        return None

    @staticmethod
    def _safe_relative(path: Path, base_path: Path) -> Optional[str]:
        """Return path relative to base_path, or None if outside base or non-existent."""
        if not path.exists():
            return None
        try:
            return str(path.relative_to(base_path))
        except ValueError:
            return None

    def _identify_entry_points(self, base_path: Path, graph: DependencyGraph) -> None:
        """Identify entry point files in the project."""
        for file_path in graph.files:
            full_path = base_path / file_path
            if full_path.name == "__init__.py" or self._matches_entry_point_pattern(file_path, full_path):
                graph.entry_points.add(file_path)
        self.logger.debug("entry_points_identified", count=len(graph.entry_points))

    def _matches_entry_point_pattern(self, file_path: str, full_path: Path) -> bool:
        """Return True if the file matches any configured entry point pattern."""
        return any(self._file_matches_ep(file_path, full_path, p) for p in self.config.entry_point_patterns)

    @staticmethod
    def _file_matches_ep(file_path: str, full_path: Path, pattern: str) -> bool:
        """Check one entry-point pattern against a file."""
        stripped = pattern.removeprefix("**/")
        return fnmatch.fnmatch(file_path, stripped) or fnmatch.fnmatch(full_path.name, stripped.split("/")[-1])

    _LANGUAGE_MAP: Dict[str, str] = {".py": "python", ".js": "javascript", ".ts": "typescript", ".tsx": "typescript", ".jsx": "javascript"}

    def _find_orphan_files(self, base_path: Path, graph: DependencyGraph) -> List[OrphanFile]:
        """Find files with no incoming imports."""
        orphans: List[OrphanFile] = []
        for file_path in graph.files:
            if file_path in graph.entry_points or Path(file_path).name == "__init__.py":
                continue
            if not graph.get_importers(file_path):
                orphans.append(self._make_orphan_file(file_path, base_path))
        return orphans

    def _make_orphan_file(self, file_path: str, base_path: Path) -> OrphanFile:
        """Construct an OrphanFile record for an un-imported file."""
        full_path = base_path / file_path
        # Unparseable files stay in the graph, so a non-UTF-8 one must not abort the run
        lines = len(full_path.read_text(encoding="utf-8", errors="replace").split("\n"))
        suffix = Path(file_path).suffix
        language = self._LANGUAGE_MAP.get(suffix, "typescript")
        return OrphanFile(
            file_path=file_path,
            absolute_path=str(full_path),
            lines=lines,
            language=language,
            status=VerificationStatus.LIKELY,
            reason="No direct imports found",
        )

    @staticmethod
    def _grep_exclude_args() -> List[str]:
        """Build --exclude-dir args to skip heavy directories."""
        args: List[str] = []
        for d in _GREP_EXCLUDE_DIRS:
            args.extend(["--exclude-dir", d])
        return args

    def _verify_orphans_with_grep(self, base_path: Path, orphans: List[OrphanFile]) -> List[OrphanFile]:
        """Verify orphan candidates using parallel grep with exclude-dirs."""
        if not orphans:
            return orphans

        exclude_args = self._grep_exclude_args()
        max_workers = min(8, len(orphans))

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(self._verify_single_orphan, base_path, orphan, exclude_args): orphan for orphan in orphans}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    orphan = futures[future]
                    self.logger.debug("grep_verify_failed", file=orphan.file_path, error=str(e))

        return orphans

    def _verify_single_orphan(self, base_path: Path, orphan: OrphanFile, exclude_args: List[str]) -> OrphanFile:
        """Run grep verification for one orphan with exclude-dirs."""
        try:
            result = subprocess.run(
                ["grep", "-r", "-l", *exclude_args, Path(orphan.file_path).stem, str(base_path)],
                capture_output=True,
                text=True,
                timeout=SubprocessDefaults.GREP_TIMEOUT_SECONDS,
            )
            refs = [ln for ln in result.stdout.strip().split("\n") if ln and not ln.endswith(orphan.file_path)]
            self._apply_grep_refs(orphan, refs)
        except subprocess.TimeoutExpired:
            orphan.status = VerificationStatus.UNCERTAIN
            orphan.reason = "Verification timed out"
        except Exception as e:
            self.logger.debug("grep_failed", file=orphan.file_path, error=str(e))
        return orphan

    @staticmethod
    def _apply_grep_refs(orphan: OrphanFile, references: List[str]) -> None:
        """Update orphan status based on grep reference results."""
        if not references:
            orphan.status = VerificationStatus.CONFIRMED
            orphan.reason = "No imports or string references found"
        else:
            orphan.status = VerificationStatus.UNCERTAIN
            orphan.reason = f"Found {len(references)} possible string references"
            orphan.importers = references[: SemanticVolumeDefaults.TOP_RESULTS_LIMIT]

    _FRAMEWORK_DECORATORS = frozenset(
        {
            "tool",
            "command",
            "route",
            "get",
            "post",
            "put",
            "delete",
            "patch",
            "property",
            "staticmethod",
            "classmethod",
            "abstractmethod",
            "fixture",
            "parametrize",
            "cached_property",
            "validator",
            "field_validator",
            "model_validator",
            "app",
            "api",
        }
    )

    @classmethod
    def _has_framework_decorator(cls, node: ast.FunctionDef) -> bool:
        """Return True if the function has a decorator that implies framework invocation."""
        for dec in node.decorator_list:
            name = cls._decorator_name(dec)
            if name and name in cls._FRAMEWORK_DECORATORS:
                return True
        return False

    @staticmethod
    def _decorator_name(dec: ast.expr) -> Optional[str]:
        """Extract the leaf name from a decorator node."""
        if isinstance(dec, ast.Name):
            return dec.id
        if isinstance(dec, ast.Attribute):
            return dec.attr
        if isinstance(dec, ast.Call):
            func = dec.func
            if isinstance(func, ast.Name):
                return func.id
            if isinstance(func, ast.Attribute):
                return func.attr
        return None

    def _find_orphan_functions(self, base_path: Path, graph: DependencyGraph) -> Tuple[List[OrphanFunction], int]:
        """Find functions with no call sites using parallel grep."""
        candidates, total_functions = self._collect_function_candidates(base_path, graph)
        if not candidates:
            return [], total_functions

        exclude_args = self._grep_exclude_args()
        src_path = base_path / "src"
        search_path = str(src_path) if src_path.is_dir() else str(base_path)
        orphan_functions = self._grep_for_uncalled_functions(candidates, search_path, exclude_args)
        return orphan_functions, total_functions

    def _collect_function_candidates(self, base_path: Path, graph: DependencyGraph) -> Tuple[List[Tuple[ast.FunctionDef, str]], int]:
        """Collect candidate functions from all Python files, skipping non-candidates."""
        candidates: List[Tuple[ast.FunctionDef, str]] = []
        total_functions = 0

        for file_path in graph.files:
            if not file_path.endswith(".py"):
                continue
            full_path = base_path / file_path
            try:
                tree = ast.parse(full_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            func_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            total_functions += len(func_nodes)
            for node in func_nodes:
                if not self._should_skip_function(node):
                    candidates.append((node, file_path))

        return candidates, total_functions

    @classmethod
    def _should_skip_function(cls, node: ast.FunctionDef) -> bool:
        """Return True if the function should be excluded from orphan analysis."""
        if node.name.startswith("__") and node.name.endswith("__"):
            return True
        if node.name.startswith("test_"):
            return True
        return cls._has_framework_decorator(node)

    def _grep_for_uncalled_functions(
        self,
        candidates: List[Tuple[ast.FunctionDef, str]],
        search_path: str,
        exclude_args: List[str],
    ) -> List[OrphanFunction]:
        """Run parallel grep to find functions with no call sites."""
        orphan_functions: List[OrphanFunction] = []
        max_workers = min(8, len(candidates))

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self._check_function_called_grep, node.name, search_path, rel_path, exclude_args): (node, rel_path)
                for node, rel_path in candidates
            }
            for future in as_completed(futures):
                node, rel_path = futures[future]
                try:
                    is_called = future.result()
                except Exception:
                    is_called = True  # Assume called on error
                if not is_called:
                    orphan_functions.append(
                        OrphanFunction(
                            name=node.name,
                            file_path=rel_path,
                            line_start=node.lineno,
                            line_end=node.end_lineno or node.lineno,
                            status=VerificationStatus.LIKELY,
                            reason="No call sites found in codebase",
                            is_private=node.name.startswith("_"),
                        )
                    )

        return orphan_functions

    def _check_function_called_grep(self, func_name: str, search_path: str, source_file: str, exclude_args: List[str]) -> bool:
        """Check if a function is called anywhere via grep (thread-safe)."""
        try:
            result = subprocess.run(
                ["grep", "-r", "-E", *exclude_args, rf"{func_name}\s*\(", search_path],
                capture_output=True,
                text=True,
                timeout=SubprocessDefaults.GREP_TIMEOUT_SECONDS,
            )
            return self._has_call_site(result.stdout, source_file, func_name)
        except Exception:
            return True

    def _has_call_site(self, grep_output: str, source_file: str, func_name: str) -> bool:
        """Return True if grep output contains a call site (not just the definition)."""
        def_pattern = re.compile(rf"\bdef\s+{re.escape(func_name)}\s*\(")
        for line in grep_output.strip().split("\n"):
            if not line or line.strip().startswith("#"):
                continue
            # For same-file matches, skip the definition line itself
            if source_file in line:
                # Extract the code portion after the file path
                code_part = line.split(":", 1)[-1] if ":" in line else line
                if def_pattern.search(code_part):
                    continue
            return True
        return False


def _build_orphan_config(
    include_patterns: Optional[List[str]],
    exclude_patterns: Optional[List[str]],
    analyze_functions: bool,
    verify_with_grep: bool,
) -> OrphanAnalysisConfig:
    config = OrphanAnalysisConfig(analyze_functions=analyze_functions, verify_with_grep=verify_with_grep)
    if include_patterns:
        config.include_patterns = include_patterns
    if exclude_patterns:
        merged = list(config.exclude_patterns)
        merged.extend(exclude_patterns)
        config.exclude_patterns = list(dict.fromkeys(merged))
    config.exclude_patterns = FilePatterns.merge_with_venv_excludes(config.exclude_patterns)
    return config


def detect_orphans_impl(
    project_folder: str,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    analyze_functions: bool = True,
    verify_with_grep: bool = True,
) -> Dict[str, Any]:
    """Detect orphan files and functions in a project.

    Args:
        project_folder: Path to the project root
        include_patterns: Glob patterns for files to analyze
        exclude_patterns: Glob patterns for files to exclude
        analyze_functions: Whether to analyze function-level orphans
        verify_with_grep: Whether to verify findings with grep

    Returns:
        Dictionary with orphan analysis results
    """
    config = _build_orphan_config(include_patterns, exclude_patterns, analyze_functions, verify_with_grep)
    return OrphanDetector(config).analyze(project_folder).to_dict()
