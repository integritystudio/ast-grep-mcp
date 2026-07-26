#!/usr/bin/env python3
"""
Comprehensive codebase analysis using all MCP tools.
This script analyzes the ast-grep-mcp codebase for:
- Code complexity issues
- Code smells
- Duplication opportunities
- Security vulnerabilities
- Code quality standards
"""

import argparse
import re
import sys
import traceback
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ast_grep_mcp.constants import (
    DeduplicationDefaults,
    FilePatterns,
    FormattingDefaults,
    SemanticVolumeDefaults,
    SubprocessDefaults,
)
from ast_grep_mcp.features.complexity.analyzer import analyze_file_complexity
from ast_grep_mcp.features.complexity.tools import analyze_complexity_tool, detect_code_smells_tool
from ast_grep_mcp.features.deduplication.scoring_scales import AnalyzeCodebaseTopN
from ast_grep_mcp.features.deduplication.tools import analyze_deduplication_candidates_tool
from ast_grep_mcp.features.quality.security_scanner import detect_security_issues_impl
from ast_grep_mcp.features.quality.tools import apply_standards_fixes_tool, enforce_standards_tool, generate_quality_report_tool
from ast_grep_mcp.models.complexity import ComplexityThresholds, FunctionComplexity
from ast_grep_mcp.utils.console_logger import console
from ast_grep_mcp.utils.file_discovery import find_source_files, language_extensions
from ast_grep_mcp.utils.slicing import take_top_n
from ast_grep_mcp.utils.subprocess_runner import TOOL_NOT_FOUND, run_tool
from scripts.analysis_output_helpers import log_count_breakdown, print_section_header

DEFAULT_PROJECT_FOLDER = "src/ast_grep_mcp"
DEFAULT_LANGUAGE = "python"
EXCLUDE_PATTERNS = FilePatterns.DEFAULT_EXCLUDE + FilePatterns.TEST_EXCLUDE + FilePatterns.MINIFIED_EXCLUDE
DUPLICATION_MIN_LINES = 10


def out(message: object = "") -> None:
    """Emit CLI output via shared console logger."""
    console.log(str(message))


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print_section_header(out, title, width=FormattingDefaults.WIDE_SECTION_WIDTH)



def _language_include_patterns(language: str) -> list[str]:
    """Build MCP tool include_patterns for a language's source extensions."""
    return [f"**/*{ext}" for ext in language_extensions(language)]


def _report_phase_exception(phase: str, exc: Exception) -> None:
    """Log an exception that escaped a phase and emit a traceback."""
    out(f"Exception during {phase}: {exc}")
    traceback.print_exc()



def analyze_individual_files(project_folder: str, language: str) -> None:
    """Analyze the top most complex files individually."""
    print_section("PHASE 1: Individual File Complexity Analysis")

    source_files = find_source_files(project_folder, language, EXCLUDE_PATTERNS)
    if not source_files:
        out(f"\nNo {language} source files found in {project_folder}")
        return

    thresholds = ComplexityThresholds()
    file_functions: dict[str, list[FunctionComplexity]] = {}
    for f in source_files:
        try:
            file_functions[str(f)] = analyze_file_complexity(str(f), language, thresholds)
        except Exception:
            continue

    ranked = sorted(
        file_functions.items(),
        key=lambda item: max((fn.metrics.cognitive for fn in item[1]), default=0),
        reverse=True,
    )
    top_files = take_top_n(ranked, AnalyzeCodebaseTopN.TOP_FILES)

    out(f"\nAnalyzing top {len(top_files)} most complex files out of {len(source_files)} total:")

    for file_path, functions in top_files:
        out(f"\n--- {file_path} ---")
        critical = [f for f in functions if f.metrics.cyclomatic > thresholds.cyclomatic or f.metrics.cognitive > thresholds.cognitive]

        out(f"Total functions: {len(functions)}")
        out(f"Critical functions: {len(critical)}")

        if critical:
            out("\nWorst offenders:")
            for func in take_top_n(
                sorted(critical, key=lambda x: x.metrics.cognitive, reverse=True),
                AnalyzeCodebaseTopN.WORST_OFFENDERS,
            ):
                out(f"  - {func.function_name} (line {func.start_line})")
                out(
                    f"    Cyclomatic: {func.metrics.cyclomatic}, Cognitive: {func.metrics.cognitive}, "
                    f"Nesting: {func.metrics.nesting_depth}, Lines: {func.metrics.lines}"
                )


def analyze_project_complexity(project_folder: str, language: str) -> None:
    """Run project-wide complexity analysis."""
    print_section("PHASE 2: Project-Wide Complexity Analysis")

    try:
        result = analyze_complexity_tool(
            project_folder=project_folder,
            language=language,
            include_patterns=_language_include_patterns(language),
            exclude_patterns=EXCLUDE_PATTERNS,
            store_results=False,
            include_trends=False,
        )

        summary = result.get("summary", {})
        total_functions = summary.get("total_functions", 0)
        exceeding_count = summary.get("exceeding_threshold", 0)
        pct = (exceeding_count / total_functions * 100) if total_functions else 0
        out(f"\nTotal functions analyzed: {total_functions}")
        out(f"Functions exceeding thresholds: {exceeding_count}")
        out(f"Percentage over threshold: {pct:.1f}%")
        out(f"\nAverage cyclomatic complexity: {summary.get('avg_cyclomatic', 0):.2f}")
        out(f"Average cognitive complexity: {summary.get('avg_cognitive', 0):.2f}")
        out(f"Max nesting depth: {summary.get('max_nesting', 0)}")

        exceeding = result.get("functions", [])
        if not exceeding:
            return

        out("\nTop 10 most complex functions by cognitive complexity:")
        for i, func in enumerate(
            take_top_n(
                sorted(exceeding, key=lambda x: x.get("cognitive", 0), reverse=True),
                AnalyzeCodebaseTopN.TOP_COMPLEX_FUNCTIONS,
            ),
            1,
        ):
            out(f"  {i}. {func['file']}:{func['name']} (lines {func['lines']})")
            out(
                f"     Cyclomatic: {func['cyclomatic']}, Cognitive: {func['cognitive']}, "
                f"Nesting: {func['nesting_depth']}, Lines: {func['length']}"
            )
    except Exception as e:
        _report_phase_exception("project complexity analysis", e)


def detect_code_smells(project_folder: str, language: str) -> None:
    """Run code smell detection."""
    print_section("PHASE 3: Code Smell Detection")

    try:
        result = detect_code_smells_tool(
            project_folder=project_folder,
            language=language,
            include_patterns=_language_include_patterns(language),
            exclude_patterns=EXCLUDE_PATTERNS,
        )

        summary = result.get("summary", {})
        out(f"\nTotal files analyzed: {result.get('files_analyzed', 0)}")
        out(f"Total smells found: {result.get('total_smells', 0)}")

        by_severity = summary.get("by_severity", {})
        out("\nBy severity:")
        log_count_breakdown(out, by_severity, order=["high", "medium", "low"], indent="  ", capitalize_labels=True)

        smells = result.get("smells", [])
        if not smells:
            return

        out("\nTop 10 code smells:")
        for smell in take_top_n(smells, AnalyzeCodebaseTopN.TOP_SMELLS_PREVIEW):
            out(f"  - [{smell.get('severity', 'unknown').upper()}] {smell.get('type', 'unknown')}")
            out(f"    File: {smell.get('file', 'unknown')}:{smell.get('line', '?')}")
            out(f"    {smell.get('message', 'No message')}")
    except Exception as e:
        _report_phase_exception("code smell detection", e)


def detect_security_issues(project_folder: str, language: str) -> None:
    """Run security vulnerability scanning."""
    print_section("PHASE 4: Security Vulnerability Scanning")

    try:
        result = detect_security_issues_impl(
            project_folder=project_folder,
            language=language,
        )

        summary = result.summary
        out(f"\nTotal files scanned: {summary.get('total_files', 0)}")
        out(f"Total issues found: {summary.get('total_issues', 0)}")

        by_severity = summary.get("by_severity", {})
        out("\nBy severity:")
        log_count_breakdown(
            out,
            by_severity,
            order=["critical", "high", "medium", "low"],
            indent="  ",
            capitalize_labels=True,
        )

        by_category = summary.get("by_category", {})
        if by_category:
            out("\nBy category:")
            for category, count in sorted(by_category.items(), key=lambda x: x[1], reverse=True):
                out(f"  {category}: {count}")

        if result.issues:
            out("\nTop 10 security issues:")
            for issue in take_top_n(result.issues, AnalyzeCodebaseTopN.TOP_SECURITY_ISSUES_PREVIEW):
                out(f"  - [{issue.severity.upper()}] {issue.issue_type}")
                out(f"    File: {issue.file}:{issue.line}")
                out(f"    {issue.description}")
    except Exception as e:
        _report_phase_exception("security scanning", e)


def analyze_duplication(project_folder: str, language: str) -> None:
    """Analyze code duplication opportunities."""
    print_section("PHASE 5: Code Duplication Analysis")

    try:
        result = analyze_deduplication_candidates_tool(
            project_path=project_folder,
            language=language,
            min_similarity=DeduplicationDefaults.MIN_SIMILARITY,
            min_lines=DUPLICATION_MIN_LINES,
            exclude_patterns=EXCLUDE_PATTERNS,
        )

        total_groups = result.get("total_groups_analyzed", 0)
        candidates = result.get("candidates", [])
        out(f"\nDuplication groups analyzed: {total_groups}")
        out(f"Top candidates: {result.get('top_candidates_count', len(candidates))}")
        out(f"Estimated LOC savings (top candidates): {result.get('top_candidates_savings_potential', 0)}")

        if not candidates:
            out("\nNo duplication candidates found.")
            return

        top_limit = SemanticVolumeDefaults.TOP_RESULTS_LIMIT
        out(f"\nTop {top_limit} duplication candidates by potential savings:")
        for i, group in enumerate(candidates[:top_limit], 1):
            instances = group.get("instances", [])
            similarity = group.get("similarity_score", group.get("similarity", 0))
            out(f"  {i}. Group with {len(instances)} instances ({similarity:.1%} similar)")
            out(f"     Potential LOC savings: {group.get('potential_line_savings', 0)} lines")
            if instances:
                out("     Locations:")
                for inst in take_top_n(instances, AnalyzeCodebaseTopN.DUPLICATION_LOCATION_PREVIEW):
                    out(f"       - {inst.get('file', 'unknown')}:{inst.get('lines', '?')}")
    except Exception as e:
        _report_phase_exception("duplication analysis", e)


def _iter_summary_section(report: str, limit: int) -> Iterator[str]:
    """Yield lines of the Summary section, stopping at the next top-level header."""
    in_summary = False
    for line in report.split("\n")[:limit]:
        if "## Summary" in line or "## Executive Summary" in line:
            in_summary = True
        elif in_summary and line.startswith("##"):
            return
        if in_summary:
            yield line


def generate_summary_report(project_folder: str, language: str, apply_fixes: bool = False) -> None:
    """Generate comprehensive quality report and optionally apply fixes."""
    print_section("PHASE 6: Generate Comprehensive Quality Report")

    try:
        enforcement_result = enforce_standards_tool(
            project_folder=project_folder,
            language=language,
            include_patterns=_language_include_patterns(language),
            exclude_patterns=EXCLUDE_PATTERNS,
        )

        result = generate_quality_report_tool(
            enforcement_result=enforcement_result,
            project_name="ast-grep-mcp",
            output_format="markdown",
            save_to_file="QUALITY_REPORT.md",
        )

        out("\nQuality report generated successfully!")
        report_path = result.get("saved_to")
        if report_path:
            out(f"Report saved to: {report_path}")

        report_content = result.get("content", "")
        if isinstance(report_content, str) and report_content:
            out("\nReport Summary:")
            for line in _iter_summary_section(report_content, SemanticVolumeDefaults.SUMMARY_PREVIEW_LIMIT):
                out(line)

        if apply_fixes:
            _apply_fixes(enforcement_result, language, project_folder=project_folder)

    except Exception as e:
        _report_phase_exception("report generation", e)


def _run_tsc_check(project_folder: str) -> bool:
    """Run tsc --noEmit to verify no type errors after fixes.

    Returns True if tsc passes (or is not available), False if errors found.
    """
    tsconfig = Path(project_folder) / "tsconfig.json"
    if not tsconfig.exists():
        return True

    out("\nRunning tsc --noEmit to verify fixes...")
    result = run_tool(
        ["npx", "tsc", "--noEmit"],
        SubprocessDefaults.TSC_NOEMIT_TIMEOUT_SECONDS,
        cwd=project_folder,
    )
    if result.error == TOOL_NOT_FOUND:
        out("tsc not found, skipping type check")
        return True
    if result.error is not None:
        out(f"tsc --noEmit timed out after {SubprocessDefaults.TSC_NOEMIT_TIMEOUT_SECONDS}s, skipping")
        return True
    if result.returncode == 0:
        out("tsc --noEmit: PASSED (no type errors)")
        return True

    error_lines = result.stdout.strip().splitlines() if result.stdout else []
    error_count = sum(1 for line in error_lines if ": error TS" in line)
    out(f"tsc --noEmit: FAILED ({error_count} type errors)")
    for line in error_lines[: SemanticVolumeDefaults.DETAIL_RESULTS_LIMIT]:
        if ": error TS" in line:
            out(f"  {line}")
    if error_count > SemanticVolumeDefaults.DETAIL_RESULTS_LIMIT:
        out(f"  ... and {error_count - SemanticVolumeDefaults.DETAIL_RESULTS_LIMIT} more errors")
    return False


_CLI_ENTRY_POINT_RE = re.compile(r'\bif\s+__name__\s*==\s*["\']__main__["\']')


def _is_cli_entry_point(file_path: str) -> bool:
    """Check if a file is a CLI entry point (has if __name__ == '__main__')."""
    try:
        with open(file_path, encoding="utf-8") as f:
            return bool(_CLI_ENTRY_POINT_RE.search(f.read()))
    except (OSError, UnicodeDecodeError):
        return False


# Rules that delete lines rather than replacing them — dangerous for CLI entry points.
# Directory/file-name patterns (scripts/, bin/, cli/, *_runner.py, etc.) are handled
# by each rule's exclude_files in rules.py.  This filter catches the remaining case:
# files containing `if __name__ == '__main__'` which can't be expressed as a glob.
_DESTRUCTIVE_RULES = {"no-print-production", "no-console-log", "no-system-out"}


def _filter_destructive_violations(violations: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Filter out removal-rule violations targeting CLI entry points.

    Skips violations from _DESTRUCTIVE_RULES when the target file contains
    ``if __name__ == '__main__'``, indicating intentional user-facing output.
    Directory and filename-based exclusions are handled at detection time by
    each rule's ``exclude_files`` patterns.

    Returns:
        Tuple of (filtered_violations, skipped_count)
    """
    filtered = []
    skipped = 0
    cli_cache: dict[str, bool] = {}

    for v in violations:
        rule_id = v.get("rule_id", "")
        file_path = v.get("file", "")

        if rule_id in _DESTRUCTIVE_RULES:
            if file_path not in cli_cache:
                cli_cache[file_path] = _is_cli_entry_point(file_path)
            if cli_cache[file_path]:
                skipped += 1
                continue

        filtered.append(v)

    return filtered, skipped


def _apply_fixes(enforcement_result: dict[str, Any], language: str, project_folder: str = "") -> None:
    """Apply automatic standards fixes from enforcement violations."""
    print_section("PHASE 7: Apply Standards Fixes")

    violations = enforcement_result.get("violations", [])
    if not violations:
        out("\nNo violations to fix.")
        return

    violations, skipped = _filter_destructive_violations(violations)
    if skipped:
        out(f"\nSkipped {skipped} violations in CLI/test files (removal rules would delete intentional output)")
    if not violations:
        out("No remaining violations to fix after filtering.")
        return

    try:
        dry_result = apply_standards_fixes_tool(
            violations=violations,
            language=language,
            fix_types=["safe"],
            dry_run=True,
            create_backup=True,
        )

        dry_summary = dry_result.get("summary", {})
        fixable = dry_summary.get("fixes_successful", 0)
        out(f"\nDry run: {fixable} of {dry_summary.get('total_violations', 0)} violations can be auto-fixed (safe fixes only)")

        if fixable == 0:
            out("No auto-fixable violations found.")
            return

        fix_result = apply_standards_fixes_tool(
            violations=violations,
            language=language,
            fix_types=["safe"],
            dry_run=False,
            create_backup=True,
        )

        fix_summary = fix_result.get("summary", {})
        out(f"\nFixed: {fix_summary.get('fixes_successful', 0)} violations")
        out(f"Failed: {fix_summary.get('fixes_failed', 0)}")
        out(f"Files modified: {fix_summary.get('files_modified', 0)}")
        backup_id = fix_result.get("backup_id")
        if backup_id:
            out(f"Backup ID: {backup_id}")

        if language == "typescript" and project_folder:
            if not _run_tsc_check(project_folder):
                out(f"\nWARNING: Type errors detected after fixes. Backup available: {backup_id}")
                out("Review errors above and restore from backup if needed.")

    except Exception as e:
        _report_phase_exception("fix application", e)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Comprehensive codebase analysis using MCP tools.")
    parser.add_argument(
        "project_folder",
        nargs="?",
        default=DEFAULT_PROJECT_FOLDER,
        help=f"Path to the project folder to analyze (default: {DEFAULT_PROJECT_FOLDER})",
    )
    parser.add_argument(
        "-l",
        "--language",
        default=DEFAULT_LANGUAGE,
        help=f"Source language (default: {DEFAULT_LANGUAGE})",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply safe auto-fixes for standards violations",
    )
    return parser.parse_args()


def main() -> None:
    """Run all analyses."""
    args = parse_args()
    project_folder = args.project_folder
    language = args.language

    if not Path(project_folder).is_dir():
        out(f"Error: '{project_folder}' is not a valid directory")
        sys.exit(1)

    out("=" * FormattingDefaults.WIDE_SECTION_WIDTH)
    out(f" COMPREHENSIVE CODEBASE ANALYSIS - {project_folder}")
    out("=" * FormattingDefaults.WIDE_SECTION_WIDTH)
    out(f"\nTarget: {project_folder} ({language})")
    out("This analysis uses MCP tools to evaluate code quality,")
    out("complexity, security, and duplication opportunities.\n")

    analyze_individual_files(project_folder, language)
    analyze_project_complexity(project_folder, language)
    detect_code_smells(project_folder, language)
    detect_security_issues(project_folder, language)
    analyze_duplication(project_folder, language)
    generate_summary_report(project_folder, language, apply_fixes=args.fix)

    print_section("ANALYSIS COMPLETE")
    out("\nAll phases completed. Review the output above and QUALITY_REPORT.md")
    out("for detailed findings and recommendations.")


if __name__ == "__main__":
    main()
