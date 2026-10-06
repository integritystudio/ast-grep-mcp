"""Auto-fix system for code quality violations.

This module provides functionality to automatically fix code quality violations:
- Safe fix application (guaranteed-safe fixes only)
- Suggested fix application (may need review)
- Fix validation (syntax checking, behavior preservation)
- Multi-fix coordination (batch operations with rollback)
"""

import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import sentry_sdk

from ast_grep_mcp.constants import ConversionFactors, PatternSuggestionConfidence, SecurityScanDefaults
from ast_grep_mcp.core.logging import get_logger
from ast_grep_mcp.features.rewrite.backup import create_backup
from ast_grep_mcp.features.rewrite.service import validate_syntax
from ast_grep_mcp.models.standards import (
    FixBatchResult,
    FixResult,
    FixValidation,
    RuleViolation,
)
from ast_grep_mcp.utils.text import read_file_lines, write_file_lines

logger = get_logger(__name__)
REVIEW_EMPTY_CATCH_CONFIDENCE = 0.75

# =============================================================================
# Fix Classification - Determine Safety Level
# =============================================================================

# Safe patterns that can be auto-applied without review
SAFE_FIX_PATTERNS: Dict[str, Dict[str, Any]] = {
    # JavaScript/TypeScript
    "no-var": {"confidence": 1.0, "reason": "var → const/let is safe"},
    "no-console-log": {"confidence": SecurityScanDefaults.VERY_HIGH_CONFIDENCE, "reason": "console.log removal is usually safe"},
    "no-debugger": {"confidence": 1.0, "reason": "debugger removal is always safe"},
    "prefer-const": {"confidence": 1.0, "reason": "let → const is safe when variable not reassigned"},
    "no-double-equals": {"confidence": SecurityScanDefaults.HIGH_CONFIDENCE, "reason": "== → === is usually safe"},
    # Python
    "no-print-production": {"confidence": SecurityScanDefaults.HIGH_CONFIDENCE, "reason": "print() removal is usually safe"},
    "no-bare-except": {
        "confidence": SecurityScanDefaults.ELEVATED_CONFIDENCE,
        "reason": "except: → except Exception: is usually safe",
    },
    "no-mutable-defaults": {"confidence": SecurityScanDefaults.VERY_HIGH_CONFIDENCE, "reason": "Mutable default fix is usually safe"},
    # Java
    "no-system-out": {"confidence": SecurityScanDefaults.HIGH_CONFIDENCE, "reason": "System.out removal is usually safe"},
}

# Patterns that require review (lower confidence)
REVIEW_REQUIRED_PATTERNS: Dict[str, Dict[str, Any]] = {
    "no-eval-exec": {
        "confidence": PatternSuggestionConfidence.STRUCTURAL,
        "reason": "eval/exec removal may break functionality",
    },
    "no-sql-injection": {"confidence": SecurityScanDefaults.MEDIUM_CONFIDENCE, "reason": "SQL parameterization needs careful review"},
    "no-empty-catch": {"confidence": REVIEW_EMPTY_CATCH_CONFIDENCE, "reason": "Empty catch replacement may change behavior"},
    "proper-exception-handling": {"confidence": SecurityScanDefaults.MEDIUM_CONFIDENCE, "reason": "Exception handling changes need review"},
}


def classify_fix_safety(rule_id: str, violation: RuleViolation) -> FixValidation:
    """Classify a fix as safe, suggested, or requiring review.

    Args:
        rule_id: ID of the rule being fixed
        violation: The violation to be fixed

    Returns:
        FixValidation with safety assessment
    """
    # Check if this is a known safe pattern
    if rule_id in SAFE_FIX_PATTERNS:
        pattern_info = SAFE_FIX_PATTERNS[rule_id]
        return FixValidation(is_safe=True, confidence=float(pattern_info["confidence"]), warnings=[], errors=[], requires_review=False)

    # Check if this requires review
    if rule_id in REVIEW_REQUIRED_PATTERNS:
        pattern_info = REVIEW_REQUIRED_PATTERNS[rule_id]
        return FixValidation(
            is_safe=False,
            confidence=float(pattern_info["confidence"]),
            warnings=[str(pattern_info["reason"])],
            errors=[],
            requires_review=True,
        )

    # Unknown pattern - conservative approach
    return FixValidation(
        is_safe=False,
        confidence=PatternSuggestionConfidence.UNKNOWN_FIX,
        warnings=["Unknown fix pattern - manual review recommended"],
        errors=[],
        requires_review=True,
    )


# =============================================================================
# Fix Application - Apply Fixes to Files
# =============================================================================


def _splice_fixed_code(lines: List[str], start_line: int, end_line: int, snippet: str, fixed_code: str) -> List[str]:
    """Replace a range of file lines with fixed code, preserving indentation.

    ast-grep snippets strip the leading indent from the first line but keep
    absolute indentation on subsequent lines.  This helper re-applies the
    stripped indent so the replacement integrates cleanly into the file.

    Each replacement line is emitted as a separate list entry (one ``\\n``
    per entry) so that line indices remain valid for subsequent operations
    on the same ``lines`` list.

    Args:
        lines: File lines (with line endings, from ``str.splitlines(keepends=True)``).
        start_line: 0-indexed first line of the span to replace.
        end_line: 0-indexed last line (inclusive) of the span to replace.
        snippet: The original ``code_snippet`` returned by ast-grep.
        fixed_code: The transformed code to splice in.

    Returns:
        A new list of lines with the replacement applied.
    """
    result = list(lines)

    if start_line == end_line:
        # Single-line — the snippet is a substring of the line, so a simple
        # str.replace preserves surrounding whitespace.
        result[start_line] = result[start_line].replace(snippet, fixed_code)
    else:
        # Multi-line — compute the indent that ast-grep stripped from line 0.
        file_first = result[start_line]
        leading_indent = file_first[: len(file_first) - len(file_first.lstrip())]

        fixed_lines = fixed_code.split("\n")
        fixed_lines[0] = leading_indent + fixed_lines[0]

        # Emit each line as a separate entry with its own newline so that
        # downstream line-index arithmetic stays correct.
        replacement = [fl + "\n" for fl in fixed_lines]
        result[start_line : end_line + 1] = replacement

    return result


def apply_pattern_fix(file_path: str, violation: RuleViolation, fix_pattern: str, language: str) -> FixResult:
    """Apply a pattern-based fix to a single violation.

    Args:
        file_path: Path to the file to fix
        violation: The violation to fix
        fix_pattern: The fix pattern (can be a literal string or metavar replacement)
        language: Programming language for syntax validation

    Returns:
        FixResult with fix outcome
    """
    try:
        # Read the file
        original_lines = read_file_lines(file_path)
        lines = list(original_lines)

        # Get the original code snippet
        start_line = violation.line - 1  # Convert to 0-indexed
        end_line = violation.end_line - 1

        if start_line < 0 or end_line >= len(lines):
            return FixResult(
                violation=violation,
                success=False,
                file_modified=False,
                original_code=violation.code_snippet,
                error="Line numbers out of range",
            )

        original_code = violation.code_snippet

        # Apply fix pattern
        fixed_code = _apply_fix_pattern(original_code, fix_pattern, violation.meta_vars or {})

        # If no change, skip
        if fixed_code == original_code:
            return FixResult(
                violation=violation,
                success=True,
                file_modified=False,
                original_code=original_code,
                fixed_code=fixed_code,
                fix_type="pattern",
            )

        # Replace in file (indent-aware)
        lines = _splice_fixed_code(lines, start_line, end_line, original_code, fixed_code)

        # Write back
        write_file_lines(file_path, lines)

        # Validate syntax
        validation = validate_syntax(file_path, language)
        syntax_valid = validation["valid"]

        if not syntax_valid:
            # Rollback the change
            write_file_lines(file_path, original_lines)

            return FixResult(
                violation=violation,
                success=False,
                file_modified=False,
                original_code=original_code,
                fixed_code=fixed_code,
                syntax_valid=False,
                error=f"Syntax validation failed: {validation['error']}",
                fix_type="pattern",
            )

        return FixResult(
            violation=violation,
            success=True,
            file_modified=True,
            original_code=original_code,
            fixed_code=fixed_code,
            syntax_valid=True,
            fix_type="pattern",
        )

    except Exception as e:
        sentry_sdk.capture_exception(e)
        return FixResult(
            violation=violation,
            success=False,
            file_modified=False,
            original_code=violation.code_snippet,
            error=f"Fix application error: {str(e)}",
            fix_type="pattern",
        )


def _apply_fix_pattern(code: str, fix_pattern: str, meta_vars: Dict[str, "str | list[str]"]) -> str:
    """Apply a fix pattern with metavariable substitution.

    Args:
        code: Original code
        fix_pattern: Fix pattern (may contain $VAR placeholders)
        meta_vars: Metavariables captured during pattern matching

    Returns:
        Fixed code with metavariables substituted
    """
    fixed = fix_pattern

    # Replace metavariables (e.g., $VAR, $ARGS, $$$)
    for var_name, var_value in meta_vars.items():
        if isinstance(var_value, list):
            var_value = " ".join(var_value)
        fixed = fixed.replace(f"${var_name}", var_value)

    return fixed


def apply_removal_fix(file_path: str, violation: RuleViolation, language: str) -> FixResult:
    """Apply a removal fix (delete the violating code).

    Args:
        file_path: Path to the file to fix
        violation: The violation to fix (will be removed)
        language: Programming language for syntax validation

    Returns:
        FixResult with fix outcome
    """
    try:
        # Read the file
        original_lines = read_file_lines(file_path)
        lines = list(original_lines)

        # Get the original code snippet
        start_line = violation.line - 1  # Convert to 0-indexed
        end_line = violation.end_line - 1

        if start_line < 0 or end_line >= len(lines):
            return FixResult(
                violation=violation,
                success=False,
                file_modified=False,
                original_code=violation.code_snippet,
                error="Line numbers out of range",
            )

        original_code = violation.code_snippet

        # Remove the lines
        del lines[start_line : end_line + 1]

        # Write back
        write_file_lines(file_path, lines)

        # Validate syntax
        validation = validate_syntax(file_path, language)
        syntax_valid = validation["valid"]

        if not syntax_valid:
            # Rollback the change
            write_file_lines(file_path, original_lines)

            return FixResult(
                violation=violation,
                success=False,
                file_modified=False,
                original_code=original_code,
                fixed_code="",
                syntax_valid=False,
                error=f"Syntax validation failed: {validation['error']}",
                fix_type="removal",
            )

        return FixResult(
            violation=violation,
            success=True,
            file_modified=True,
            original_code=original_code,
            fixed_code="",
            syntax_valid=True,
            fix_type="removal",
        )

    except Exception as e:
        sentry_sdk.capture_exception(e)
        return FixResult(
            violation=violation,
            success=False,
            file_modified=False,
            original_code=violation.code_snippet,
            error=f"Removal error: {str(e)}",
            fix_type="removal",
        )


# =============================================================================
# Batch Fix Coordinator - Apply Multiple Fixes
# =============================================================================


def apply_fixes_batch(
    violations: List[RuleViolation],
    language: str,
    project_folder: str,
    fix_types: List[str] = ["safe"],
    dry_run: bool = True,
    create_backup_flag: bool = True,
) -> FixBatchResult:
    """Apply fixes to multiple violations in a coordinated manner.

    Args:
        violations: List of violations to fix
        language: Programming language for syntax validation
        project_folder: Project root folder
        fix_types: Types of fixes to apply ('safe', 'suggested', 'all')
        dry_run: If True, preview fixes without applying
        create_backup_flag: If True, create backup before applying fixes

    Returns:
        FixBatchResult with batch fix outcome
    """
    start_time = time.time()

    # Filter violations by fix type
    fixable_violations = _filter_violations_by_fix_type(violations, fix_types)

    if dry_run:
        result = _execute_dry_run(fixable_violations, start_time)
        return result

    # Real run - apply fixes
    backup_id = _create_backup_if_needed(fixable_violations, project_folder, create_backup_flag)

    # Group violations by file
    violations_by_file = _group_violations_by_file(fixable_violations)

    # Apply fixes and collect results
    fix_results, files_modified, fixes_successful, fixes_failed, validation_passed = _execute_real_run(violations_by_file, language)

    # If any fixes failed and we have a backup, offer rollback
    if fixes_failed > 0 and backup_id:
        logger.warning(f"{fixes_failed} fixes failed. Backup {backup_id} available for rollback")

    return _build_batch_result(
        fixable_violations, fix_results, files_modified, fixes_successful, fixes_failed, validation_passed, backup_id, start_time
    )


def _execute_dry_run(fixable_violations: List[RuleViolation], start_time: float) -> FixBatchResult:
    """Execute a dry run preview of fixes without applying them.

    Args:
        fixable_violations: Violations to preview
        start_time: Batch operation start time

    Returns:
        FixBatchResult with preview information
    """
    results = []
    for violation in fixable_violations:
        validation = classify_fix_safety(violation.rule_id, violation)

        # Create a preview fix result
        results.append(
            FixResult(
                violation=violation,
                success=validation.is_safe,
                file_modified=False,
                original_code=violation.code_snippet,
                fixed_code=violation.fix_suggestion or "(fix pattern not specified)",
                syntax_valid=True,
                fix_type="safe" if validation.is_safe else "suggested",
            )
        )

    execution_time = int((time.time() - start_time) * ConversionFactors.MILLISECONDS_PER_SECOND)

    return FixBatchResult(
        total_violations=len(fixable_violations),
        fixes_attempted=0,
        fixes_successful=0,
        fixes_failed=0,
        files_modified=[],
        validation_passed=True,
        results=results,
        execution_time_ms=execution_time,
    )


def _create_backup_if_needed(fixable_violations: List[RuleViolation], project_folder: str, create_backup_flag: bool) -> Optional[str]:
    """Create backup of files if requested.

    Args:
        fixable_violations: Violations with files to backup
        project_folder: Project root folder
        create_backup_flag: Whether to create backup

    Returns:
        Backup ID if created, None otherwise
    """
    if not create_backup_flag:
        return None

    # Get unique file paths
    file_paths = list(set(v.file for v in fixable_violations))
    if not file_paths:
        return None

    # Create backup
    backup_id = create_backup(file_paths, project_folder)
    logger.info(f"Created backup {backup_id} for {len(file_paths)} files")
    return backup_id


def _group_violations_by_file(fixable_violations: List[RuleViolation]) -> Dict[str, List[RuleViolation]]:
    """Group violations by file path.

    Args:
        fixable_violations: Violations to group

    Returns:
        Dictionary mapping file paths to their violations
    """
    violations_by_file: Dict[str, List[RuleViolation]] = {}
    for violation in fixable_violations:
        if violation.file not in violations_by_file:
            violations_by_file[violation.file] = []
        violations_by_file[violation.file].append(violation)
    return violations_by_file


def _execute_real_run(
    violations_by_file: Dict[str, List[RuleViolation]], language: str
) -> Tuple[List[FixResult], Set[str], int, int, bool]:
    """Execute actual fix application.

    Args:
        violations_by_file: Violations grouped by file
        language: Programming language for validation

    Returns:
        Tuple of (results, files_modified, fixes_successful, fixes_failed, validation_passed)
    """
    fix_results: List[FixResult] = []
    files_modified: Set[str] = set()
    fixes_successful = 0
    fixes_failed = 0
    validation_passed = True

    for file_path, file_violations in violations_by_file.items():
        # Sort by line number (reverse order to avoid line number shifts)
        file_violations.sort(key=lambda v: v.line, reverse=True)

        for violation in file_violations:
            result = _apply_single_fix(file_path, violation, language)
            fix_results.append(result)

            # Update counters based on result
            fixes_successful, fixes_failed, validation_passed = _process_fix_result(
                result, file_path, fixes_successful, fixes_failed, validation_passed, files_modified
            )

    return fix_results, files_modified, fixes_successful, fixes_failed, validation_passed


# Pattern matching assignment operators after a variable name.
# Covers: =, +=, -=, *=, /=, %=, <<=, >>=, >>>=, &=, |=, ^=, &&=, ||=, ??=
_ASSIGNMENT_OPS = re.compile(
    r"(?:\+\+|--"  # postfix increment/decrement
    r"|\?\?=|\|\|=|&&="  # logical assignment (check before single-char)
    r"|>>>=|<<=|>>="  # shift assignment
    r"|[+\-*/%&|^]=)"  # compound assignment
    r"|(?<![=!<>])=[^=>]"  # plain assignment (not ==, !=, <=, >=, =>)
)


def _line_reassigns_var(line: str, var_pattern: re.Pattern[str]) -> bool:
    """Check if a single line reassigns the variable matched by var_pattern."""
    for match in var_pattern.finditer(line):
        after = line[match.end() :].lstrip()
        before = line[: match.start()].rstrip()

        if _ASSIGNMENT_OPS.match(after):
            return True
        if before.endswith("++") or before.endswith("--"):
            return True

    return False


def _decl_line_has_reassignment(line: str, var_pattern: re.Pattern[str]) -> bool:
    """Check if a declaration line also reassigns the variable (e.g., for-loop iterators).

    Skips the first occurrence (the declaration) and checks subsequent occurrences.
    """
    first = var_pattern.search(line)
    if not first:
        return False
    rest = line[first.end() :]
    return var_pattern.search(rest) is not None and _line_reassigns_var(rest, var_pattern)


def _line_is_comment(line: str) -> bool:
    """Return True if the line is a single-line comment (JS/TS style)."""
    stripped = line.lstrip()
    return stripped.startswith("//") or stripped.startswith("*")


def _check_line_for_reassignment(
    i: int,
    line: str,
    decl_idx: int,
    var_pattern: "re.Pattern[str]",
) -> Optional[bool]:
    """Check one source line for a variable reassignment.

    Returns:
        True  — reassignment found, stop scanning.
        False — skip this line (comment or no pattern match).
        None  — this is the declaration line with no reassignment; continue.
    """
    if _line_is_comment(line) or not var_pattern.search(line):
        return False
    if i == decl_idx:
        return True if _decl_line_has_reassignment(line, var_pattern) else None
    return _line_reassigns_var(line, var_pattern)


def _is_variable_reassigned(file_path: str, var_name: str, decl_line: int) -> bool:
    """Check if a let-declared variable is reassigned after its declaration.

    Args:
        file_path: Path to the source file
        var_name: Variable name to check
        decl_line: 1-indexed line number of the declaration

    Returns:
        True if the variable appears to be reassigned
    """
    try:
        lines = read_file_lines(file_path)
    except OSError:
        return True  # Assume reassigned if we can't read

    var_pattern = re.compile(rf"\b{re.escape(var_name)}\b")
    decl_idx = decl_line - 1  # Convert to 0-indexed

    for i, line in enumerate(lines):
        if i < decl_idx:
            continue
        result = _check_line_for_reassignment(i, line, decl_idx, var_pattern)
        if result is True:
            return True

    return False


def _extract_var_name_from_let(code_snippet: str) -> Optional[str]:
    """Extract variable name from a let declaration snippet.

    Returns None for destructuring patterns (too complex to verify).
    """
    m = re.match(r"\s*let\s+(\w+)\s*[=;:,)]", code_snippet)
    return m.group(1) if m else None


# == null / != null is intentional TypeScript idiom (checks both null and undefined).
# Lookahead/lookbehind prevents matching === or !==.
_NULL_EQUALITY_RE = re.compile(
    r"(?<![=!])(==|!=)(?!=)\s*(null|undefined)\b"
    r"|\b(null|undefined)\s*(?<![=!])(==|!=)(?!=)"
)
_LOOSE_EQ_RE = re.compile(r"(?<![=!])==(?!=)")
_LOOSE_NEQ_RE = re.compile(r"(?<![=!])!=(?!=)")


def _is_null_equality_check(code: str) -> bool:
    """Return True if code is a == null or == undefined idiom."""
    return bool(_NULL_EQUALITY_RE.search(code))


def _fix_double_equals(code: str) -> str:
    """Fix loose equality, preserving == null/undefined idiom."""
    if _is_null_equality_check(code):
        return code
    if _LOOSE_EQ_RE.search(code):
        return _LOOSE_EQ_RE.sub("===", code, count=1)
    if _LOOSE_NEQ_RE.search(code):
        return _LOOSE_NEQ_RE.sub("!==", code, count=1)
    return code


# Rule-specific code transformations for rules whose fix field is a
# human-readable description rather than an ast-grep rewrite pattern.
# Each handler takes original code and returns transformed code.
_RULE_CODE_TRANSFORMS: Dict[str, Any] = {
    "prefer-const": lambda code: code.replace("let ", "const ", 1),
    "no-var": lambda code: code.replace("var ", "const ", 1),
    "no-double-equals": _fix_double_equals,
    "no-bare-except": lambda code: code.replace("except:", "except Exception:", 1),
}

# Rules that should use line removal instead of pattern replacement.
_REMOVAL_RULES = {"no-console-log", "no-debugger", "no-print-production", "no-system-out"}


def _apply_single_fix(file_path: str, violation: RuleViolation, language: str) -> FixResult:
    """Apply a single fix to a violation.

    Args:
        file_path: Path to file containing violation
        violation: The violation to fix
        language: Programming language for validation

    Returns:
        FixResult with outcome
    """
    # 1. Check for a known code transformation
    if violation.rule_id in _RULE_CODE_TRANSFORMS:
        # prefer-const: verify the variable is not reassigned before fixing
        if violation.rule_id == "prefer-const":
            var_name = _extract_var_name_from_let(violation.code_snippet)
            if var_name is None or _is_variable_reassigned(file_path, var_name, violation.line):
                logger.debug(f"Skipping prefer-const for reassigned variable '{var_name}' at {file_path}:{violation.line}")
                return FixResult(
                    violation=violation,
                    success=True,
                    file_modified=False,
                    original_code=violation.code_snippet,
                    fix_type="skipped",
                )

        transform = _RULE_CODE_TRANSFORMS[violation.rule_id]
        fixed_code = transform(violation.code_snippet)
        return apply_pattern_fix(file_path, violation, fixed_code, language)

    # 2. Check for removal rules (no-console-log, no-debugger, etc.)
    if violation.rule_id in _REMOVAL_RULES:
        return apply_removal_fix(file_path, violation, language)

    # No known fix: deleting the flagged code (e.g. eval, SQL concat) would silently change behavior.
    if not violation.fix_suggestion:
        logger.debug(f"Skipping {violation.rule_id} at {file_path}:{violation.line}: no fix available")
        return FixResult(
            violation=violation,
            success=True,
            file_modified=False,
            original_code=violation.code_snippet,
            fix_type="skipped",
        )

    # 3. Fallback: use the fix_suggestion as a literal pattern
    return apply_pattern_fix(file_path, violation, violation.fix_suggestion, language)


def _process_fix_result(
    result: FixResult, file_path: str, fixes_successful: int, fixes_failed: int, validation_passed: bool, files_modified: Set[str]
) -> Tuple[int, int, bool]:
    """Process a single fix result and update counters.

    Args:
        result: The fix result to process
        file_path: Path to the modified file
        fixes_successful: Current successful fix count
        fixes_failed: Current failed fix count
        validation_passed: Current validation status
        files_modified: Set of modified file paths

    Returns:
        Tuple of (updated_fixes_successful, updated_fixes_failed, updated_validation_passed)
    """
    if result.success and result.file_modified:
        fixes_successful += 1
        files_modified.add(file_path)

        if not result.syntax_valid:
            validation_passed = False
    elif not result.success:
        fixes_failed += 1
        validation_passed = False

    return fixes_successful, fixes_failed, validation_passed


def _build_batch_result(
    fixable_violations: List[RuleViolation],
    fix_results: List[FixResult],
    files_modified: Set[str],
    fixes_successful: int,
    fixes_failed: int,
    validation_passed: bool,
    backup_id: Optional[str],
    start_time: float,
) -> FixBatchResult:
    """Build the final batch result.

    Args:
        fixable_violations: All violations that were fixable
        fix_results: Results from all fix attempts
        files_modified: Set of modified file paths
        fixes_successful: Number of successful fixes
        fixes_failed: Number of failed fixes
        validation_passed: Whether all validations passed
        backup_id: Backup ID if created
        start_time: Batch operation start time

    Returns:
        FixBatchResult with complete batch information
    """
    execution_time = int((time.time() - start_time) * ConversionFactors.MILLISECONDS_PER_SECOND)

    return FixBatchResult(
        total_violations=len(fixable_violations),
        fixes_attempted=fixes_successful + fixes_failed,
        fixes_successful=fixes_successful,
        fixes_failed=fixes_failed,
        files_modified=list(files_modified),
        backup_id=backup_id,
        validation_passed=validation_passed,
        results=fix_results,
        execution_time_ms=execution_time,
    )


def _filter_violations_by_fix_type(violations: List[RuleViolation], fix_types: List[str]) -> List[RuleViolation]:
    """Filter violations based on fix type preference.

    Args:
        violations: All violations
        fix_types: Allowed fix types ('safe', 'suggested', 'all')

    Returns:
        Filtered list of violations that can be fixed
    """
    if "all" in fix_types:
        return violations

    fixable = []
    for violation in violations:
        validation = classify_fix_safety(violation.rule_id, violation)

        if "safe" in fix_types and validation.is_safe:
            fixable.append(violation)
        elif "suggested" in fix_types and not validation.is_safe:
            fixable.append(violation)

    return fixable


# =============================================================================
# Fix Preview - Generate Diff Without Applying
# =============================================================================


def preview_fix(violation: RuleViolation) -> Dict[str, Any]:
    """Preview a fix without applying it.

    Args:
        violation: The violation to preview fix for

    Returns:
        Dictionary with fix preview information
    """
    validation = classify_fix_safety(violation.rule_id, violation)

    return {
        "violation": {
            "file": violation.file,
            "line": violation.line,
            "rule_id": violation.rule_id,
            "message": violation.message,
            "code": violation.code_snippet,
        },
        "fix": {
            "original": violation.code_snippet,
            "fixed": violation.fix_suggestion or "(removal)",
            "is_safe": validation.is_safe,
            "confidence": validation.confidence,
            "requires_review": validation.requires_review,
            "warnings": validation.warnings,
        },
    }
