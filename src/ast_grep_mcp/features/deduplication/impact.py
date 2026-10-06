"""Impact analysis for deduplication refactoring."""

import json
import os
import re
import subprocess
from typing import Any, Dict, List, Tuple, TypedDict, Union, cast

from ...constants import SemanticVolumeDefaults
from ...core import run_ast_grep
from ...core.logging import get_logger

__all__ = [
    "ImpactAnalyzer",
]

EXTRACTED_FUNCTION_GROWTH_FACTOR = 1.2
EXTERNAL_CALL_RISK_CAP = 3
REEXPORT_FILE_RISK_SCORE = 3
MEDIUM_RISK_MAX_SCORE = 4


class _RiskLevelConfig(TypedDict):
    """Type definition for risk level configuration."""

    max_score: Union[int, float]
    recommendations: List[str]


_LANGUAGE_NAME_PATTERNS: Dict[str, List[str]] = {
    "python": [
        r"\bdef\s+(\w+)\s*\(",
        r"\bclass\s+(\w+)",
    ],
    "javascript": [
        r"\bfunction\s+(\w+)\s*\(",
        r"\b(?:const|let|var)\s+(\w+)\s*=\s*(?:\([^)]*\)|[^=])*=>",
        r"^\s*(\w+)\s*\([^)]*\)\s*\{",
    ],
    "typescript": [
        r"\bfunction\s+(\w+)\s*\(",
        r"\b(?:const|let|var)\s+(\w+)\s*=\s*(?:\([^)]*\)|[^=])*=>",
        r"^\s*(\w+)\s*\([^)]*\)\s*\{",
    ],
    "jsx": [
        r"\bfunction\s+(\w+)\s*\(",
        r"\b(?:const|let|var)\s+(\w+)\s*=\s*(?:\([^)]*\)|[^=])*=>",
        r"^\s*(\w+)\s*\([^)]*\)\s*\{",
    ],
    "tsx": [
        r"\bfunction\s+(\w+)\s*\(",
        r"\b(?:const|let|var)\s+(\w+)\s*=\s*(?:\([^)]*\)|[^=])*=>",
        r"^\s*(\w+)\s*\([^)]*\)\s*\{",
    ],
    "java": [
        r"\b(?:public|private|protected|static|\w+)\s+(\w+)\s*\([^)]*\)\s*\{",
        r"\bclass\s+(\w+)",
    ],
    "csharp": [
        r"\b(?:public|private|protected|static|\w+)\s+(\w+)\s*\([^)]*\)\s*\{",
        r"\bclass\s+(\w+)",
    ],
    "cpp": [
        r"\b(?:public|private|protected|static|\w+)\s+(\w+)\s*\([^)]*\)\s*\{",
        r"\bclass\s+(\w+)",
    ],
    "c": [
        r"\b(?:public|private|protected|static|\w+)\s+(\w+)\s*\([^)]*\)\s*\{",
        r"\bclass\s+(\w+)",
    ],
    "go": [
        r"\bfunc\s+(?:\([^)]*\)\s+)?(\w+)\s*\(",
    ],
    "rust": [
        r"\bfn\s+(\w+)\s*[<(]",
        r"\bstruct\s+(\w+)",
    ],
}

_RISK_LEVEL_CONFIG: Dict[str, _RiskLevelConfig] = {
    "low": {
        "max_score": 1,
        "recommendations": ["Safe to proceed with standard review", "Run tests after applying changes"],
    },
    "medium": {
        "max_score": MEDIUM_RISK_MAX_SCORE,
        "recommendations": [
            "Review external call sites before applying",
            "Consider updating call sites in the same commit",
            "Run comprehensive test suite after changes",
        ],
    },
    "high": {
        "max_score": float("inf"),
        "recommendations": [
            "Carefully review all external references",
            "Consider deprecating old functions instead of removing",
            "Update external call sites first",
            "May require coordinated changes across modules",
            "Consider feature flag for gradual rollout",
        ],
    },
}

_FILTER_COMMON_WORDS = {"new", "get", "set", "if", "for", "while", "return", "main", "init", "test"}


class ImpactAnalyzer:
    """Analyzes the impact of applying deduplication to code."""

    def __init__(self) -> None:
        """Initialize the impact analyzer."""
        self.logger = get_logger("deduplication.impact_analysis")

    def analyze_deduplication_impact(self, duplicate_group: Dict[str, Any], project_root: str, language: str) -> Dict[str, Any]:
        """Analyze the impact of applying deduplication to a duplicate group."""
        locations = duplicate_group.get("locations", [])
        sample_code = duplicate_group.get("sample_code", "")
        duplicate_count = duplicate_group.get("duplicate_count", 0)
        lines_per_duplicate = duplicate_group.get("lines_per_duplicate", 0)

        files_in_group = self._parse_files_from_locations(locations)
        function_names = self._extract_function_names_from_code(sample_code, language)

        self.logger.info(
            "impact_analysis_start",
            duplicate_count=duplicate_count,
            files_in_group=len(files_in_group),
            function_names=function_names[: SemanticVolumeDefaults.TOP_RESULTS_LIMIT] if function_names else [],
        )

        all_external_refs = self._find_all_external_refs(function_names, project_root, language, files_in_group)

        lines_changed = self._estimate_lines_changed(
            duplicate_count=duplicate_count, lines_per_duplicate=lines_per_duplicate, external_call_sites=len(all_external_refs)
        )
        breaking_change_risk = self._assess_breaking_change_risk(
            function_names=function_names,
            files_in_group=files_in_group,
            external_call_sites=all_external_refs,
            project_root=project_root,
            language=language,
        )
        files_affected = len(files_in_group) + len({s["file"] for s in all_external_refs if "file" in s})

        result = {
            "files_affected": files_affected,
            "lines_changed": lines_changed,
            "external_call_sites": all_external_refs,
            "breaking_change_risk": breaking_change_risk,
        }

        self.logger.info(
            "impact_analysis_complete",
            files_affected=files_affected,
            external_references=len(all_external_refs),
            risk_level=breaking_change_risk.get("level", "unknown"),
        )

        return result

    def _parse_files_from_locations(self, locations: List[Any]) -> List[str]:
        """Extract unique absolute file paths from location entries.

        BUGL-08: locations from the detector are dicts ``{"file": ..., "line":
        ...}``, not ``"file:line"`` strings, so the old ``":" in loc`` guard
        silently discarded every entry and ``files_in_group`` was always empty.
        The function now handles both formats and absolutises relative paths so
        they match the absolute paths produced by ``_to_call_site_record``.
        """
        files: List[str] = []
        for loc in locations:
            if isinstance(loc, dict):
                file_path = loc.get("file", "")
            elif isinstance(loc, str) and ":" in loc:
                file_path = loc.split(":")[0]
            else:
                continue
            if not file_path:
                continue
            abs_path = os.path.abspath(file_path)
            if abs_path not in files:
                files.append(abs_path)
        return files

    def _find_all_external_refs(
        self, function_names: List[str], project_root: str, language: str, exclude_files: List[str]
    ) -> List[Dict[str, Any]]:
        """Find all external call sites and import references combined."""
        call_sites = self._find_external_call_sites(
            function_names=function_names, project_root=project_root, language=language, exclude_files=exclude_files
        )
        import_sites = self._find_import_references(
            function_names=function_names, project_root=project_root, language=language, exclude_files=exclude_files
        )
        return call_sites + import_sites

    def _extract_function_names_from_code(self, code: str, language: str) -> List[str]:
        """Extract function/method/class names from code sample.

        Args:
            code: Code sample to analyze
            language: Programming language

        Returns:
            List of extracted names
        """
        if not code:
            return []

        # Get language-specific patterns and extract names
        patterns = self._get_language_patterns(language.lower())
        names = self._apply_extraction_patterns(code, patterns)

        # Filter and deduplicate
        return self._filter_extracted_names(names)

    def _get_language_patterns(self, language: str) -> List[str]:
        """Get regex patterns for extracting names from code in a specific language."""
        return _LANGUAGE_NAME_PATTERNS.get(language, [])

    def _apply_extraction_patterns(self, code: str, patterns: List[str]) -> List[str]:
        """Apply regex patterns to extract names from code.

        Args:
            code: Source code to analyze
            patterns: List of regex patterns to apply

        Returns:
            List of all extracted names (may contain duplicates)
        """
        names: List[str] = []

        for pattern in patterns:
            # Use MULTILINE flag for patterns that match line beginnings
            flags = re.MULTILINE if pattern.startswith("^") else 0
            matches = re.findall(pattern, code, flags)
            names.extend(matches)

        return names

    def _filter_extracted_names(self, names: List[str]) -> List[str]:
        """Filter and deduplicate extracted function/class names."""
        seen: set[str] = set()
        result = []
        for name in names:
            if name and name not in seen and name.lower() not in _FILTER_COMMON_WORDS:
                seen.add(name)
                result.append(name)
        return result

    def _find_external_call_sites(
        self, function_names: List[str], project_root: str, language: str, exclude_files: List[str]
    ) -> List[Dict[str, Any]]:
        """Find call sites for functions outside the duplicate locations."""
        call_sites: List[Dict[str, Any]] = []
        for func_name in function_names[:10]:  # Limit to prevent too many searches
            try:
                call_sites.extend(self._search_call_sites_for_name(func_name, project_root, language, exclude_files))
            except (json.JSONDecodeError, subprocess.SubprocessError) as e:
                self.logger.debug("call_site_search_error", function=func_name, error=str(e))
        return call_sites

    def _search_call_sites_for_name(
        self, func_name: str, project_root: str, language: str, exclude_files: List[str]
    ) -> List[Dict[str, Any]]:
        """Run ast-grep for one function name and return filtered call-site records."""
        args = ["--pattern", f"{func_name}($$$)", "--lang", language, "--json", project_root]
        result = run_ast_grep("run", args)
        if not (result.returncode == 0 and result.stdout.strip()):
            return []
        return [
            self._to_call_site_record(match, func_name, project_root)
            for match in json.loads(result.stdout)
            if match.get("file", "") not in exclude_files
        ]

    def _to_call_site_record(self, match: Dict[str, Any], func_name: str, project_root: str) -> Dict[str, Any]:
        """Shape a single ast-grep match into a call-site dict."""
        file_path = match.get("file", "")
        if not os.path.isabs(file_path):
            file_path = os.path.join(project_root, file_path)
        return {
            "file": file_path,
            "line": match.get("range", {}).get("start", {}).get("line", 0) + 1,
            "column": match.get("range", {}).get("start", {}).get("column", 0),
            "function_called": func_name,
            "context": match.get("text", "")[:100],
            "type": "function_call",
        }

    def _find_import_references(
        self, function_names: List[str], project_root: str, language: str, exclude_files: List[str]
    ) -> List[Dict[str, Any]]:
        """Find import statements that reference the duplicated code."""
        if not function_names:
            return []

        import_refs: List[Dict[str, Any]] = []
        lang = language.lower()

        # Process up to 10 function names
        for func_name in function_names[:10]:
            patterns = self._get_import_patterns(func_name, lang)
            if not patterns:
                continue

            refs = self._search_import_patterns(patterns, func_name, language, project_root, exclude_files)
            import_refs.extend(refs)

        return import_refs

    def _get_import_patterns(self, func_name: str, language: str) -> List[str]:
        """Get language-specific import patterns for a function name.

        Args:
            func_name: Function name to search for
            language: Language (lowercase)

        Returns:
            List of ast-grep patterns to search for
        """
        # Configuration-driven pattern mapping
        pattern_map = {
            "python": [f"from $MODULE import {func_name}", f"from $MODULE import $$$, {func_name}, $$$"],
            "javascript": [f"import {{ {func_name} }} from $MODULE", f"import {{ $$$, {func_name}, $$$ }} from $MODULE"],
            "typescript": [f"import {{ {func_name} }} from $MODULE", f"import {{ $$$, {func_name}, $$$ }} from $MODULE"],
            "jsx": [f"import {{ {func_name} }} from $MODULE", f"import {{ $$$, {func_name}, $$$ }} from $MODULE"],
            "tsx": [f"import {{ {func_name} }} from $MODULE", f"import {{ $$$, {func_name}, $$$ }} from $MODULE"],
            "java": [f"import $$$$.{func_name}"],
            "go": [],  # Go imports are package-level, not function-level
        }

        return pattern_map.get(language, [])

    def _search_import_patterns(
        self, patterns: List[str], func_name: str, language: str, project_root: str, exclude_files: List[str]
    ) -> List[Dict[str, Any]]:
        """Search for import patterns using ast-grep.

        Args:
            patterns: List of patterns to search for
            func_name: Function name being searched
            language: Programming language
            project_root: Project root path
            exclude_files: Files to exclude from results

        Returns:
            List of import reference info dicts
        """
        import_refs = []

        for pattern in patterns:
            matches = self._execute_import_search(pattern, language, project_root)
            if not matches:
                continue

            refs = self._process_import_matches(matches, func_name, project_root, exclude_files)
            import_refs.extend(refs)

        return import_refs

    def _execute_import_search(self, pattern: str, language: str, project_root: str) -> List[Dict[str, Any]]:
        """Execute ast-grep search for a specific pattern.

        Args:
            pattern: ast-grep pattern to search for
            language: Programming language
            project_root: Project root path

        Returns:
            List of match dictionaries from ast-grep
        """
        try:
            args = ["--pattern", pattern, "--lang", language, "--json", project_root]
            result = run_ast_grep("run", args)

            if result.returncode != 0 or not result.stdout.strip():
                return []

            return cast(List[Dict[str, Any]], json.loads(result.stdout))

        except (json.JSONDecodeError, subprocess.SubprocessError):
            return []

    def _process_import_matches(
        self, matches: List[Dict[str, Any]], func_name: str, project_root: str, exclude_files: List[str]
    ) -> List[Dict[str, Any]]:
        """Process ast-grep matches into import reference info.

        Args:
            matches: List of matches from ast-grep
            func_name: Function name that was searched
            project_root: Project root path
            exclude_files: Files to exclude from results

        Returns:
            List of import reference info dicts
        """
        import_refs = []

        for match in matches:
            file_path = match.get("file", "")

            # Skip excluded files
            if file_path in exclude_files:
                continue

            # Ensure absolute path
            if not os.path.isabs(file_path):
                file_path = os.path.join(project_root, file_path)

            import_ref = self._create_import_ref(match, file_path, func_name)
            import_refs.append(import_ref)

        return import_refs

    def _create_import_ref(self, match: Dict[str, Any], file_path: str, func_name: str) -> Dict[str, Any]:
        """Create an import reference info dictionary from a match.

        Args:
            match: Match data from ast-grep
            file_path: Absolute file path
            func_name: Function name that was imported

        Returns:
            Import reference info dict
        """
        range_data = match.get("range", {})
        start_data = range_data.get("start", {})

        return {
            "file": file_path,
            "line": start_data.get("line", 0) + 1,
            "column": start_data.get("column", 0),
            "imported_name": func_name,
            "context": match.get("text", "")[:100],
            "type": "import",
        }

    def _estimate_lines_changed(self, duplicate_count: int, lines_per_duplicate: int, external_call_sites: int) -> Dict[str, Any]:
        """Estimate the number of lines that would change during deduplication.

        Args:
            duplicate_count: Number of duplicate instances
            lines_per_duplicate: Lines in each duplicate
            external_call_sites: Number of external references

        Returns:
            Dict with additions and deletions estimates
        """
        # Deletions: All duplicate code except one instance
        deletions = (duplicate_count - 1) * lines_per_duplicate

        # Additions:
        # - One extracted function (slightly more lines due to parameterization)
        extracted_function_lines = int(lines_per_duplicate * EXTRACTED_FUNCTION_GROWTH_FACTOR)

        # - Import statements for each file (1 line per file)
        import_lines = duplicate_count - 1  # minus the file with the extracted function

        # - Replacement calls (1 line each)
        replacement_calls = duplicate_count

        # - Updates to external call sites (minimal, usually 0 or 1 line each)
        external_updates = external_call_sites

        additions = extracted_function_lines + import_lines + replacement_calls + external_updates

        # Net change
        net_change = additions - deletions

        return {
            "additions": additions,
            "deletions": deletions,
            "net_change": net_change,
            "breakdown": {
                "extracted_function": extracted_function_lines,
                "new_imports": import_lines,
                "replacement_calls": replacement_calls,
                "external_call_updates": external_updates,
            },
        }

    def _assess_breaking_change_risk(
        self,
        function_names: List[str],
        files_in_group: List[str],
        external_call_sites: List[Dict[str, Any]],
        project_root: str,
        language: str,
    ) -> Dict[str, Any]:
        """Assess the risk of breaking changes from deduplication."""
        risk_factors = []
        risk_score = 0

        # Calculate risk from each factor
        score, factors = self._calculate_external_reference_risk(external_call_sites)
        risk_score += score
        risk_factors.extend(factors)

        score, factors = self._calculate_public_api_risk(function_names, language)
        risk_score += score
        risk_factors.extend(factors)

        score, factors = self._calculate_cross_module_risk(files_in_group)
        risk_score += score
        risk_factors.extend(factors)

        score, factors = self._calculate_test_file_risk(files_in_group)
        risk_score += score
        risk_factors.extend(factors)

        score, factors = self._calculate_reexport_risk(files_in_group)
        risk_score += score
        risk_factors.extend(factors)

        # Determine risk level and recommendations
        level, recommendations = self._determine_risk_level(risk_score)

        return {
            "level": level,
            "score": risk_score,
            "factors": risk_factors,
            "recommendations": recommendations,
            "external_reference_count": len(external_call_sites),
        }

    def _calculate_external_reference_risk(self, external_call_sites: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
        """Calculate risk from external references."""
        if not external_call_sites:
            return 0, []

        factors = []
        score = 0

        call_count = len([s for s in external_call_sites if s.get("type") == "function_call"])
        import_count = len([s for s in external_call_sites if s.get("type") == "import"])

        if call_count > 0:
            factors.append(f"Found {call_count} external call site(s) that may need updates")
            score += min(call_count, EXTERNAL_CALL_RISK_CAP)

        if import_count > 0:
            factors.append(f"Found {import_count} import statement(s) referencing the code")
            score += min(import_count, 2)  # Cap at 2

        return score, factors

    def _calculate_public_api_risk(self, function_names: List[str], language: str) -> Tuple[int, List[str]]:
        """Calculate risk from public API exposure."""
        lang_lower = language.lower()

        for name in function_names:
            if self._is_public_api(name, lang_lower):
                return 2, [f"Function '{name}' appears to be public API"]

        return 0, []

    def _is_public_api(self, name: str, language: str) -> bool:
        """Check if a function name appears to be public API."""
        # Python: no underscore prefix
        if language == "python" and not name.startswith("_"):
            return True

        # Java/C#: typically PascalCase for public
        if language in ("java", "csharp") and name[0].isupper():
            return True

        return False

    def _calculate_cross_module_risk(self, files_in_group: List[str]) -> Tuple[int, List[str]]:
        """Calculate risk from cross-module dependencies."""
        if len(files_in_group) <= 1:
            return 0, []

        directories = {os.path.dirname(f) for f in files_in_group}

        if len(directories) > 1:
            factor = f"Duplicates span {len(directories)} different modules/directories"
            return 2, [factor]

        return 0, []

    def _calculate_test_file_risk(self, files_in_group: List[str]) -> Tuple[int, List[str]]:
        """Calculate risk adjustment for test files."""
        test_files = [f for f in files_in_group if "test" in f.lower() or "spec" in f.lower()]

        if test_files:
            factor = f"{len(test_files)} test file(s) contain duplicates - lower risk"
            return -1, [factor]  # Negative score reduces risk

        return 0, []

    def _calculate_reexport_risk(self, files_in_group: List[str]) -> Tuple[int, List[str]]:
        """Calculate risk from re-export files."""
        reexport_filenames = {"__init__.py", "index.ts", "index.js", "mod.rs"}
        reexport_files = [f for f in files_in_group if os.path.basename(f) in reexport_filenames]

        if reexport_files:
            return REEXPORT_FILE_RISK_SCORE, ["Code is in module export file - higher breakage risk"]

        return 0, []

    def _determine_risk_level(self, risk_score: int) -> Tuple[str, List[str]]:
        """Determine risk level and recommendations based on score."""
        for level_name, config in _RISK_LEVEL_CONFIG.items():
            if risk_score <= config["max_score"]:
                return level_name, config["recommendations"]
        return "high", _RISK_LEVEL_CONFIG["high"]["recommendations"]
