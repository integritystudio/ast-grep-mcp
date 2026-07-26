"""Standards enforcement engine for linting rules.

This module provides functionality to execute linting rules against a codebase:
- Rule set loading (built-in and custom)
- Parallel rule execution with ThreadPoolExecutor
- Violation collection and grouping
- Severity filtering
- Human-readable reporting
"""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Dict, List, Set

import sentry_sdk
import yaml

from ast_grep_mcp.constants import ConversionFactors, FormattingDefaults, RuleSetPriority, SeverityRankingDefaults, StreamDefaults
from ast_grep_mcp.core.executor import stream_ast_grep_results
from ast_grep_mcp.core.logging import get_logger
from ast_grep_mcp.features.quality.rules import RULE_TEMPLATES, load_rules_from_project
from ast_grep_mcp.models.standards import EnforcementResult, LintingRule, RuleExecutionContext, RuleSet, RuleTemplate, RuleViolation
from ast_grep_mcp.utils.file_discovery import find_source_files

# =============================================================================
# Built-in Rule Sets
# =============================================================================

RULE_SETS: Dict[str, Dict[str, Any]] = {
    "recommended": {
        "description": "General best practices for clean, maintainable code",
        "priority": RuleSetPriority.RECOMMENDED,
        "rules": [
            "no-var",
            "no-console-log",
            "no-double-equals",
            "no-empty-catch",
            "prefer-const",
            "no-bare-except",
            "no-mutable-defaults",
            "no-print-production",
            "no-debugger",
            "no-fixme-comments",
        ],
    },
    "security": {
        "description": "Security-focused rules to detect vulnerabilities",
        "priority": RuleSetPriority.SECURITY,  # Higher priority = run first
        "rules": [
            "no-eval-exec",
            "no-hardcoded-credentials",
            "no-sql-injection",
            "no-double-equals",
            "no-empty-catch",
            "no-bare-except",
            "no-string-exception",
            "no-assert-production",
            "proper-exception-handling",
        ],
    },
    "performance": {
        "description": "Performance anti-patterns and optimization opportunities",
        "priority": RuleSetPriority.PERFORMANCE,
        "rules": [
            "no-magic-numbers"  # Placeholder - will expand with performance rules in future phases
        ],
    },
    "style": {
        "description": "Code style and formatting rules",
        "priority": RuleSetPriority.STYLE,
        "rules": [
            "no-var",
            "prefer-const",
            "no-console-log",
            "no-print-production",
            "no-system-out",
            "no-any-type",
            "no-magic-numbers",
            "require-type-hints",
            "no-todo-comments",
        ],
    },
}


# =============================================================================
# Rule Set Loading
# =============================================================================


def template_to_linting_rule(template: RuleTemplate) -> LintingRule:
    """Convert RuleTemplate to LintingRule.

    Args:
        template: RuleTemplate object from RULE_TEMPLATES

    Returns:
        LintingRule object ready for execution
    """
    return LintingRule(
        id=template.id,
        language=template.language,
        severity=template.severity,
        message=template.message,
        pattern=template.pattern,
        note=template.note,
        fix=template.fix,
        constraints=template.constraints,
        exclude_files=template.exclude_files,
    )


def load_custom_rules(project_folder: str, language: str) -> List[LintingRule]:
    """Load custom rules from .ast-grep-rules/ directory.

    Args:
        project_folder: Project root directory
        language: Target language for filtering

    Returns:
        List of LintingRule objects loaded from YAML files
    """
    logger = get_logger("load_custom_rules")

    # Use the rules.py function
    all_rules = load_rules_from_project(project_folder)

    # Filter by language
    filtered_rules = [r for r in all_rules if r.language == language]

    logger.info("custom_rules_loaded", count=len(filtered_rules), language=language)
    return filtered_rules


def _load_rules_from_templates(rule_ids: Set[str], language: str) -> List[LintingRule]:
    """Load and filter rules from RULE_TEMPLATES by language.

    Args:
        rule_ids: Set of rule IDs to load
        language: Target language for filtering

    Returns:
        List of LintingRule objects matching the language
    """
    rules: List[LintingRule] = []
    for rule_id in rule_ids:
        if rule_id in RULE_TEMPLATES:
            template = RULE_TEMPLATES[rule_id]
            if template.language == language:
                rules.append(template_to_linting_rule(template))
    return rules


def _load_all_rules(language: str, logger: Any) -> RuleSet:
    """Load 'all' rule set - combines all built-in rule sets.

    Args:
        language: Target language for filtering
        logger: Logger instance

    Returns:
        RuleSet containing all built-in rules for the language
    """
    all_rule_ids: Set[str] = set()
    for set_config in RULE_SETS.values():
        all_rule_ids.update(set_config["rules"])

    rules = _load_rules_from_templates(all_rule_ids, language)

    logger.info("rule_set_loaded", rule_set="all", language=language, rules_count=len(rules))
    return RuleSet(name="all", description=f"All built-in rules for {language}", rules=rules, priority=RuleSetPriority.RECOMMENDED)


def _load_custom_rule_set(project_folder: str, language: str, logger: Any) -> RuleSet:
    """Load custom rule set from .ast-grep-rules/ directory.

    Args:
        project_folder: Project root for loading custom rules
        language: Target language for filtering
        logger: Logger instance

    Returns:
        RuleSet containing custom rules
    """
    custom_rules = load_custom_rules(project_folder, language)
    logger.info("rule_set_loaded", rule_set="custom", language=language, rules_count=len(custom_rules))
    return RuleSet(name="custom", description="Custom rules from .ast-grep-rules/", rules=custom_rules, priority=RuleSetPriority.CUSTOM)


def _load_builtin_rule_set(rule_set_name: str, language: str, logger: Any) -> RuleSet:
    """Load a built-in rule set by name.

    Args:
        rule_set_name: Name of built-in rule set
        language: Target language for filtering
        logger: Logger instance

    Returns:
        RuleSet object with loaded rules

    Raises:
        ValueError: If rule set not found
    """
    if rule_set_name not in RULE_SETS:
        available = ", ".join(list(RULE_SETS.keys()) + ["custom", "all"])
        raise ValueError(f"Rule set '{rule_set_name}' not found. Available: {available}")

    set_config = RULE_SETS[rule_set_name]
    rule_ids = set(set_config["rules"])
    rules = _load_rules_from_templates(rule_ids, language)

    logger.info("rule_set_loaded", rule_set=rule_set_name, language=language, rules_count=len(rules))

    return RuleSet(name=rule_set_name, description=set_config["description"], rules=rules, priority=set_config["priority"])


def load_rule_set(rule_set_name: str, project_folder: str, language: str) -> RuleSet:
    """Load a built-in or custom rule set.

    Args:
        rule_set_name: Name of rule set ('recommended', 'security', 'custom', 'all')
        project_folder: Project root for loading custom rules
        language: Target language for filtering rules

    Returns:
        RuleSet object with loaded rules

    Raises:
        ValueError: If rule set not found or language unsupported
    """
    logger = get_logger("load_rule_set")

    # Dispatch to appropriate loader based on rule set type
    if rule_set_name == "all":
        return _load_all_rules(language, logger)

    if rule_set_name == "custom":
        return _load_custom_rule_set(project_folder, language, logger)

    return _load_builtin_rule_set(rule_set_name, language, logger)


# =============================================================================
# Rule Execution
# =============================================================================


def _extract_single_meta_vars(meta_data: Dict[str, Any]) -> Dict[str, Any]:
    single = meta_data.get("single", {})
    if not isinstance(single, dict):
        return {}
    return {k: v["text"] for k, v in single.items() if isinstance(v, dict) and "text" in v}


def _extract_multi_meta_vars(meta_data: Dict[str, Any]) -> Dict[str, Any]:
    multi = meta_data.get("multi", {})
    if not isinstance(multi, dict):
        return {}
    return {k: [v.get("text", "") for v in vl if isinstance(v, dict)] for k, vl in multi.items() if isinstance(vl, list)}


def _extract_meta_vars(match: Dict[str, Any]) -> Any:
    """Extract metavariables from ast-grep match."""
    if "metaVariables" not in match:
        return None
    meta_data = match["metaVariables"]
    return {**_extract_single_meta_vars(meta_data), **_extract_multi_meta_vars(meta_data)}


def parse_match_to_violation(match: Dict[str, Any], rule: LintingRule) -> RuleViolation:
    """Parse ast-grep JSON match into RuleViolation.

    Args:
        match: JSON match object from ast-grep
        rule: Rule that generated this match

    Returns:
        RuleViolation object
    """
    range_info = match.get("range", {})
    start = range_info.get("start", {})
    end = range_info.get("end", {})
    return RuleViolation(
        file=match.get("file", ""),
        line=start.get("line", 0) + 1,
        column=start.get("column", 0) + 1,
        end_line=end.get("line", 0) + 1,
        end_column=end.get("column", 0) + 1,
        severity=rule.severity,
        rule_id=rule.id,
        message=rule.message,
        code_snippet=match.get("text", ""),
        fix_suggestion=rule.fix,
        meta_vars=_extract_meta_vars(match),
    )


def _matches_glob_pattern(file_path: str, pattern: str) -> bool:
    """Check if file_path matches a single glob pattern."""
    if "**" in pattern:
        parts = [p.strip("/") for p in pattern.split("**") if p.strip("/")]
        return any(part in file_path for part in parts)
    return fnmatch(file_path, pattern) or fnmatch(str(Path(file_path).name), pattern)


def should_exclude_file(file_path: str, exclude_patterns: List[str]) -> bool:
    """Check if file should be excluded based on patterns.

    Args:
        file_path: Absolute file path
        exclude_patterns: List of glob patterns to exclude

    Returns:
        True if file should be excluded
    """
    return any(_matches_glob_pattern(file_path, pattern) for pattern in exclude_patterns)


def _is_excluded(file_path: str, context_patterns: List[str], rule_patterns: List[str]) -> bool:
    if should_exclude_file(file_path, context_patterns):
        return True
    return bool(rule_patterns) and should_exclude_file(file_path, rule_patterns)


def _collect_violations(matches: List[Dict[str, Any]], rule: LintingRule, context: RuleExecutionContext) -> List[RuleViolation]:
    """Filter and collect violations from matches, applying exclude patterns and limit."""
    violations: List[RuleViolation] = []
    rule_excludes = rule.exclude_files or []
    for match in matches:
        violation = parse_match_to_violation(match, rule)
        if _is_excluded(violation.file, context.exclude_patterns, rule_excludes):
            continue
        violations.append(violation)
        if context.max_violations > 0 and len(violations) >= context.max_violations:
            break
    return violations


def _run_ast_grep_scan(rule: LintingRule, context: RuleExecutionContext) -> List[Dict[str, Any]]:
    yaml_str = yaml.safe_dump(rule.to_yaml_dict())
    args = ["--inline-rules", yaml_str, "--json=stream", context.project_folder]
    max_results = context.max_violations if context.max_violations > 0 else 0
    with sentry_sdk.start_span(op="execute_rule", name=f"Rule: {rule.id}"):
        return list(stream_ast_grep_results("scan", args, max_results=max_results, progress_interval=StreamDefaults.PROGRESS_INTERVAL))


def execute_rule(rule: LintingRule, context: RuleExecutionContext) -> List[RuleViolation]:
    """Execute a single rule and return violations.

    Args:
        rule: LintingRule to execute
        context: Execution context with project settings

    Returns:
        List of RuleViolation objects found by this rule
    """
    logger = context.logger
    try:
        matches = _run_ast_grep_scan(rule, context)
        violations = _collect_violations(matches, rule, context)
        logger.info("rule_executed", rule_id=rule.id, violations_found=len(violations))
        return violations
    except Exception as e:
        logger.error("rule_execution_failed", rule_id=rule.id, error=str(e))
        sentry_sdk.capture_exception(e, extras={"rule_id": rule.id})
        return []


def _should_stop_execution(violations_count: int, max_violations: int) -> bool:
    """Check if execution should stop due to violation limit."""
    return max_violations > 0 and violations_count >= max_violations


def _execute_rule_with_limit(
    rule: LintingRule, context: RuleExecutionContext, all_violations: List[RuleViolation], violations_lock: threading.Lock
) -> List[RuleViolation]:
    """Execute a single rule with violation limit checking.

    Args:
        rule: The linting rule to execute
        context: Execution context
        all_violations: Shared list of all violations (for limit checking)
        violations_lock: Lock for thread-safe access to all_violations

    Returns:
        List of violations found by this rule
    """
    # Check limit before executing
    with violations_lock:
        if _should_stop_execution(len(all_violations), context.max_violations):
            context.logger.info("max_violations_reached", current=len(all_violations))
            return []

    # Execute the rule
    return execute_rule(rule, context)


def _cancel_pending_futures(futures: Dict[Any, LintingRule]) -> None:
    for f in futures:
        if not f.done():
            f.cancel()


def _process_rule_result(
    future: Any,
    futures: Dict[Any, LintingRule],
    all_violations: List[RuleViolation],
    violations_lock: threading.Lock,
    context: RuleExecutionContext,
) -> bool:
    """Process the result from a completed rule execution future.

    Returns True if execution should continue, False if max violations reached.
    """
    try:
        violations = future.result()
        with violations_lock:
            all_violations.extend(violations)
            if _should_stop_execution(len(all_violations), context.max_violations):
                _cancel_pending_futures(futures)
                return False
    except Exception as e:
        rule = futures[future]
        context.logger.warning("rule_batch_execution_failed", rule_id=rule.id, error=str(e))
    return True


def execute_rules_batch(rules: List[LintingRule], context: RuleExecutionContext) -> List[RuleViolation]:
    """Execute multiple rules in parallel.

    Args:
        rules: List of LintingRule objects to execute
        context: Execution context

    Returns:
        Combined list of all violations found
    """
    all_violations: List[RuleViolation] = []
    violations_lock = threading.Lock()

    # Execute rules in parallel
    with ThreadPoolExecutor(max_workers=context.max_threads) as executor:
        # Submit all rules for execution
        futures = {executor.submit(_execute_rule_with_limit, rule, context, all_violations, violations_lock): rule for rule in rules}

        # Process results as they complete
        for future in as_completed(futures):
            should_continue = _process_rule_result(future, futures, all_violations, violations_lock, context)

            if not should_continue:
                break

    return all_violations


# =============================================================================
# Violation Processing
# =============================================================================


def group_violations_by_file(violations: List[RuleViolation]) -> Dict[str, List[RuleViolation]]:
    """Group violations by file path.

    Args:
        violations: List of all violations

    Returns:
        Dictionary mapping file paths to their violations
    """
    grouped: Dict[str, List[RuleViolation]] = {}

    for violation in violations:
        if violation.file not in grouped:
            grouped[violation.file] = []
        grouped[violation.file].append(violation)

    # Sort violations within each file by line number
    for file_path in grouped:
        grouped[file_path].sort(key=lambda v: (v.line, v.column))

    return grouped


def group_violations_by_severity(violations: List[RuleViolation]) -> Dict[str, List[RuleViolation]]:
    """Group violations by severity level.

    Args:
        violations: List of all violations

    Returns:
        Dictionary mapping severity levels to their violations
    """
    grouped: Dict[str, List[RuleViolation]] = {"error": [], "warning": [], "info": []}

    for violation in violations:
        if violation.severity in grouped:
            grouped[violation.severity].append(violation)

    return grouped


def group_violations_by_rule(violations: List[RuleViolation]) -> Dict[str, List[RuleViolation]]:
    """Group violations by rule ID.

    Args:
        violations: List of all violations

    Returns:
        Dictionary mapping rule IDs to their violations
    """
    grouped: Dict[str, List[RuleViolation]] = {}

    for violation in violations:
        if violation.rule_id not in grouped:
            grouped[violation.rule_id] = []
        grouped[violation.rule_id].append(violation)

    return grouped


def filter_violations_by_severity(violations: List[RuleViolation], severity_threshold: str) -> List[RuleViolation]:
    """Filter violations to only include >= severity threshold.

    Args:
        violations: All violations
        severity_threshold: Minimum severity ('error', 'warning', 'info')

    Returns:
        Filtered list of violations
    """
    severity_order = SeverityRankingDefaults.ENFORCER_THRESHOLD_ORDER
    min_level = severity_order.get(severity_threshold, 0)

    return [v for v in violations if severity_order.get(v.severity, 0) >= min_level]


# =============================================================================
# Reporting
# =============================================================================


def _format_severity_section(summary: Dict[str, Any]) -> List[str]:
    lines: List[str] = ["Violations by Severity:"]
    for severity in ["error", "warning", "info"]:
        count = summary["by_severity"].get(severity, 0)
        if count > 0:
            lines.append(f"  {severity.upper()}: {count}")
    return lines


def _format_file_section(violations_by_file: Dict[str, Any]) -> List[str]:
    if not violations_by_file:
        return ["\nNo violations found!"]
    lines: List[str] = ["Violations by File:", "-" * FormattingDefaults.WIDE_SECTION_WIDTH]
    for file_path, violations in sorted(violations_by_file.items()):
        lines.append(f"\n{file_path} ({len(violations)} violations)")
        for v in violations:
            lines.append(f"  Line {v.line}:{v.column} [{v.severity.upper()}] {v.rule_id}")
            lines.append(f"    {v.message}")
            if v.fix_suggestion:
                lines.append(f"    Fix: {v.fix_suggestion}")
            lines.append("")
    return lines


def format_violation_report(result: EnforcementResult) -> str:
    """Format enforcement result as human-readable text report.

    Args:
        result: EnforcementResult object

    Returns:
        Formatted text report
    """
    sep = "=" * FormattingDefaults.WIDE_SECTION_WIDTH
    summary = result.summary
    lines: List[str] = [
        sep,
        "CODE STANDARDS ENFORCEMENT REPORT",
        sep,
        "",
        f"Files Scanned: {summary['files_scanned']}",
        f"Rules Executed: {len(result.rules_executed)}",
        f"Total Violations: {summary['total_violations']}",
        f"Execution Time: {summary['execution_time_ms']}ms",
        "",
    ]
    lines.extend(_format_severity_section(summary))
    lines.append("")
    lines.extend(_format_file_section(result.violations_by_file))
    lines.append(sep)
    return "\n".join(lines)


# =============================================================================
# Main Enforcement Function
# =============================================================================


def _load_custom_rule_set_by_ids(project_path: Path, language: str, custom_rules: List[str]) -> RuleSet:
    """Load a custom rule set filtered to specific rule IDs."""
    all_custom = load_custom_rules(str(project_path), language)
    rules = [r for r in all_custom if r.id in custom_rules]
    if not rules:
        raise ValueError(f"No custom rules found matching IDs: {custom_rules}. Available: {[r.id for r in all_custom]}")
    return RuleSet(name="custom", description=f"Custom rules: {', '.join(custom_rules)}", rules=rules, priority=RuleSetPriority.CUSTOM)


def _empty_enforcement_result() -> EnforcementResult:
    return EnforcementResult(
        summary={
            "total_violations": 0,
            "by_severity": {"error": 0, "warning": 0, "info": 0},
            "by_file": {},
            "files_scanned": 0,
            "rules_executed": 0,
            "execution_time_ms": 0,
        },
        violations=[],
        violations_by_file={},
        violations_by_severity={"error": [], "warning": [], "info": []},
        violations_by_rule={},
        rules_executed=[],
        execution_time_ms=0,
        files_scanned=0,
    )


def _build_enforcement_result(
    filtered_violations: List[RuleViolation],
    rule_set_obj: RuleSet,
    violations_by_file: Dict[str, List[RuleViolation]],
    violations_by_severity: Dict[str, List[RuleViolation]],
    violations_by_rule: Dict[str, List[RuleViolation]],
    duration_ms: int,
    files_scanned: int,
) -> EnforcementResult:
    summary = {
        "total_violations": len(filtered_violations),
        "by_severity": {sev: len(violations_by_severity[sev]) for sev in ("error", "warning", "info")},
        "by_file": {fp: len(vs) for fp, vs in violations_by_file.items()},
        "files_scanned": files_scanned,
        "rules_executed": len(rule_set_obj.rules),
        "execution_time_ms": duration_ms,
    }
    return EnforcementResult(
        summary=summary,
        violations=filtered_violations,
        violations_by_file=violations_by_file,
        violations_by_severity=violations_by_severity,
        violations_by_rule=violations_by_rule,
        rules_executed=[r.id for r in rule_set_obj.rules],
        execution_time_ms=duration_ms,
        files_scanned=files_scanned,
    )


def _run_enforcement(
    rule_set_obj: RuleSet,
    context: RuleExecutionContext,
    severity_threshold: str,
    start_time: float,
) -> EnforcementResult:
    with sentry_sdk.start_span(op="execute_rules", name=f"Rules: {len(rule_set_obj.rules)}"):
        all_violations = execute_rules_batch(rule_set_obj.rules, context)
    filtered = filter_violations_by_severity(all_violations, severity_threshold)
    by_file = group_violations_by_file(filtered)
    by_severity = group_violations_by_severity(filtered)
    by_rule = group_violations_by_rule(filtered)
    files_scanned = len(find_source_files(context.project_folder, context.language, context.exclude_patterns))
    execution_time = __import__("time").time() - start_time
    duration_ms = int(execution_time * ConversionFactors.MILLISECONDS_PER_SECOND)
    result = _build_enforcement_result(filtered, rule_set_obj, by_file, by_severity, by_rule, duration_ms, files_scanned)
    context.logger.info(
        "enforcement_completed",
        total_violations=len(filtered),
        files_scanned=files_scanned,
        execution_time_seconds=round(execution_time, FormattingDefaults.ROUNDING_PRECISION),
    )
    return result


def enforce_standards_impl(
    project_folder: str,
    language: str,
    rule_set: str,
    custom_rules: List[str],
    include_patterns: List[str],
    exclude_patterns: List[str],
    severity_threshold: str,
    max_violations: int,
    max_threads: int,
) -> EnforcementResult:
    """Enforce coding standards by executing linting rules against a project."""
    import time

    logger = get_logger("enforce_standards")
    start_time = time.time()

    project_path = Path(project_folder).resolve()
    if not project_path.exists():
        raise ValueError(f"Project folder does not exist: {project_folder}")

    with sentry_sdk.start_span(op="load_rule_set", name=f"Set: {rule_set}"):
        if rule_set == "custom" and custom_rules:
            rule_set_obj = _load_custom_rule_set_by_ids(project_path, language, custom_rules)
        else:
            rule_set_obj = load_rule_set(rule_set, str(project_path), language)

    if not rule_set_obj.rules:
        return _empty_enforcement_result()

    logger.info("rules_loaded", rule_set=rule_set, rules_count=len(rule_set_obj.rules))

    context = RuleExecutionContext(
        project_folder=str(project_path),
        language=language,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        max_violations=max_violations,
        max_threads=max_threads,
        logger=logger,
    )

    return _run_enforcement(rule_set_obj, context, severity_threshold, start_time)
