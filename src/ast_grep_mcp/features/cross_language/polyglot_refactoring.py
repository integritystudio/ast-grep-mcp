"""Polyglot refactoring implementation.

This module provides functionality to refactor code across
multiple programming languages atomically.
"""

import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ast_grep_mcp.constants import LANGUAGE_EXTENSIONS, BackupDefaults, ConversionFactors, DisplayDefaults, FilePatterns
from ast_grep_mcp.core.logging import get_logger
from ast_grep_mcp.features.rewrite.backup import create_backup
from ast_grep_mcp.models.cross_language import (
    SUPPORTED_LANGUAGES,
    PolyglotChange,
    PolyglotRefactoringPlan,
    PolyglotRefactoringResult,
    RefactoringType,
)
from ast_grep_mcp.utils.file_discovery import find_source_files
from ast_grep_mcp.utils.text import read_file_lines, write_file_lines

logger = get_logger(__name__)

# API symbol patterns by language
# Note: Use {{}} to escape literal braces in .format() templates
API_SYMBOL_PATTERNS: Dict[str, Dict[str, str]] = {
    "python": {
        "route": r'@app\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)["\']',
        "function": r"def\s+{symbol}\s*\(",
        "constant": r"{symbol}\s*=",
        "class": r"class\s+{symbol}\s*[\(:]",
    },
    "typescript": {
        "route": r'app\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)["\']',
        "function": r"(?:function|const|let|var)\s+{symbol}\s*[=\(]",
        "constant": r"(?:const|let|var)\s+{symbol}\s*=",
        "class": r"class\s+{symbol}\s*[{{<]",
        "interface": r"interface\s+{symbol}\s*[{{<]",
        "type": r"type\s+{symbol}\s*=",
    },
    "javascript": {
        "route": r'app\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)["\']',
        "function": r"(?:function|const|let|var)\s+{symbol}\s*[=\(]",
        "constant": r"(?:const|let|var)\s+{symbol}\s*=",
        "class": r"class\s+{symbol}\s*[{{<]",
    },
    "java": {
        "route": r'@(?:Get|Post|Put|Delete|Patch)Mapping\s*\(\s*["\']([^"\']+)["\']',
        "function": r"(?:public|private|protected)\s+\w+\s+{symbol}\s*\(",
        "constant": r"(?:public|private|protected)?\s*(?:static)?\s*final\s+\w+\s+{symbol}\s*=",
        "class": r"(?:public|private|protected)?\s*(?:abstract)?\s*class\s+{symbol}\s*[{{<]",
    },
    "go": {
        "route": r'(?:r|router)\.(Get|Post|Put|Delete|Patch)\s*\(\s*["\']([^"\']+)["\']',
        "function": r"func\s+(?:\([^)]+\)\s+)?{symbol}\s*\(",
        "constant": r"(?:const|var)\s+{symbol}\s*=",
        "type": r"type\s+{symbol}\s+struct",
    },
}

# Common identifiers to warn about
COMMON_IDENTIFIERS = {"id", "type", "class", "name", "value"}

# Default languages for refactoring
DEFAULT_LANGUAGES = ["python", "typescript", "javascript", "java", "go"]
LARGE_REFACTORING_CHANGE_THRESHOLD = 50

# Accepted aliases for refactoring type values
REFACTORING_TYPE_ALIASES: Dict[str, str] = {
    "rename": "rename_api",
}


# =============================================================================
# File Discovery
# =============================================================================


# Vendored code, environments and our own backups must never be rewritten.
_SKIP_DIR_NAMES = FilePatterns.SKIP_DIR_NAMES | {BackupDefaults.DIR_NAME}


def _find_files_with_language(
    project_folder: str,
    languages: List[str],
) -> Dict[str, List[str]]:
    """Find files for each language in a project."""
    result: Dict[str, List[str]] = {}
    for lang in languages:
        extensions = LANGUAGE_EXTENSIONS.get(lang, [])
        files = [str(f) for f in find_source_files(project_folder, lang, extensions=extensions, skip_dir_names=_SKIP_DIR_NAMES)]
        if files:
            result[lang] = files

    return result


# =============================================================================
# Symbol Detection
# =============================================================================


def _match_pattern_type(line: str, patterns: Dict[str, str], symbol: str) -> Optional[str]:
    """Check if line matches any pattern type for the symbol."""
    for pattern_type, pattern_template in patterns.items():
        pattern = pattern_template.format(symbol=re.escape(symbol))
        if re.search(pattern, line):
            return pattern_type
    return None


def _find_symbol_occurrences(
    file_path: str,
    symbol: str,
    language: str,
) -> List[Tuple[int, str, str]]:
    """Find occurrences of a symbol in a file."""
    try:
        lines = read_file_lines(file_path, errors="ignore")
    except Exception as e:
        logger.warning("file_read_error", file=file_path, error=str(e)[: DisplayDefaults.CONTENT_PREVIEW_LENGTH])
        return []

    patterns = API_SYMBOL_PATTERNS.get(language, {})
    word_pattern = rf"\b{re.escape(symbol)}\b"
    occurrences = []

    for i, line in enumerate(lines, 1):
        matched_type = _match_pattern_type(line, patterns, symbol)
        if not matched_type and re.search(word_pattern, line):
            matched_type = "reference"
        if matched_type:
            occurrences.append((i, line.rstrip(), matched_type))

    return occurrences


# =============================================================================
# Change Creation
# =============================================================================


def _create_rename_change(
    file_path: str,
    line_number: int,
    original_line: str,
    symbol: str,
    new_name: str,
    language: str,
) -> PolyglotChange:
    """Create a change object for renaming a symbol."""
    new_line = re.sub(rf"\b{re.escape(symbol)}\b", new_name, original_line)

    return PolyglotChange(
        language=language,
        file_path=file_path,
        line_number=line_number,
        original_code=original_line,
        new_code=new_line,
        change_type="modify",
    )


def _collect_changes_for_file(
    file_path: str,
    symbol_name: str,
    new_name: str,
    language: str,
) -> List[PolyglotChange]:
    """Collect all changes for a single file."""
    changes = []
    occurrences = _find_symbol_occurrences(file_path, symbol_name, language)
    for line_num, line_content, _ in occurrences:
        change = _create_rename_change(file_path, line_num, line_content, symbol_name, new_name, language)
        changes.append(change)
    return changes


def _collect_all_changes(
    files_by_language: Dict[str, List[str]],
    symbol_name: str,
    new_name: Optional[str],
) -> List[PolyglotChange]:
    """Collect changes across all files and languages."""
    if not new_name:
        return []

    all_changes: List[PolyglotChange] = []
    for language, files in files_by_language.items():
        for file_path in files:
            changes = _collect_changes_for_file(file_path, symbol_name, new_name, language)
            all_changes.extend(changes)
    return all_changes


# =============================================================================
# Risk Analysis
# =============================================================================


def _analyze_risks(
    changes: List[PolyglotChange],
    symbol: str,
    new_name: str,
) -> List[str]:
    """Analyze risks of the refactoring."""
    risks = []

    if len(changes) > LARGE_REFACTORING_CHANGE_THRESHOLD:
        risks.append(f"Large refactoring: {len(changes)} changes across multiple files")

    languages = set(c.language for c in changes)
    if len(languages) > 1:
        risks.append(f"Multi-language refactoring: {', '.join(languages)}")

    if new_name.lower() in COMMON_IDENTIFIERS:
        risks.append(f"'{new_name}' is a common identifier - verify no conflicts")

    api_changes = sum(1 for c in changes if "route" in c.original_code.lower())
    if api_changes > 0:
        risks.append(f"{api_changes} API route changes - clients may need updates")

    return risks


# =============================================================================
# Manual Review Detection
# =============================================================================

CONFIG_PATTERNS = ["README.md", "CHANGELOG.md", "*.yaml", "*.yml", "*.json", "*.toml"]


def _check_config_file(file_path: Path, symbol: str) -> Optional[str]:
    """Return file path string if symbol found in file, else None."""
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        return str(file_path) if symbol in content else None
    except Exception:
        return None


def _identify_manual_review(
    project_folder: str,
    symbol: str,
    languages: List[str],
) -> List[str]:
    """Identify files that need manual review."""
    manual_review = []
    project_path = Path(project_folder)

    for pattern in CONFIG_PATTERNS:
        for file_path in project_path.rglob(pattern):
            result = _check_config_file(file_path, symbol)
            if result:
                manual_review.append(result)

    return manual_review[:10]


# =============================================================================
# Change Application
# =============================================================================


def _apply_changes_to_file(file_path: str, file_changes: List[PolyglotChange]) -> bool:
    """Apply changes to a single file. Returns True on success."""
    try:
        lines = read_file_lines(file_path)

        file_changes.sort(key=lambda c: c.line_number, reverse=True)
        for change in file_changes:
            line_idx = change.line_number - 1
            if 0 <= line_idx < len(lines):
                lines[line_idx] = change.new_code + "\n"

        write_file_lines(file_path, lines)
        return True
    except Exception as e:
        logger.error("apply_change_failed", file=file_path, error=str(e)[:100])
        return False


def _apply_changes(changes: List[PolyglotChange], project_folder: str) -> Tuple[List[str], str]:
    """Back up the affected files, then apply changes. Returns (modified files, backup id)."""
    changes_by_file: Dict[str, List[PolyglotChange]] = {}
    for change in changes:
        changes_by_file.setdefault(change.file_path, []).append(change)

    backup_id = create_backup(list(changes_by_file), project_folder)
    modified_files = []
    for file_path, file_changes in changes_by_file.items():
        if _apply_changes_to_file(file_path, file_changes):
            modified_files.append(file_path)

    return modified_files, backup_id


# =============================================================================
# Validation
# =============================================================================

BRACKET_PAIRS = [("(", ")"), ("[", "]"), ("{", "}")]


def _validate_changes(
    changes: List[PolyglotChange],
    project_folder: str,
) -> Tuple[bool, List[str]]:
    """Validate that changes would result in valid code."""
    errors = []

    for change in changes:
        code = change.new_code
        for open_b, close_b in BRACKET_PAIRS:
            if code.count(open_b) != code.count(close_b):
                errors.append(f"{change.file_path}:{change.line_number} - Unbalanced {open_b}{close_b}")

    return len(errors) == 0, errors


# =============================================================================
# Input Validation
# =============================================================================


def _validate_inputs(
    project_folder: str,
    refactoring_type: str,
    new_name: Optional[str],
) -> RefactoringType:
    """Validate inputs and return parsed refactoring type."""
    if not os.path.isdir(project_folder):
        raise ValueError(f"Project folder not found: {project_folder}")

    refactoring_type = REFACTORING_TYPE_ALIASES.get(refactoring_type, refactoring_type)
    try:
        refactor_type = RefactoringType(refactoring_type)
    except ValueError:
        raise ValueError(f"Invalid refactoring type: {refactoring_type}. Valid types: {[t.value for t in RefactoringType]}")

    if refactor_type == RefactoringType.RENAME_API and not new_name:
        raise ValueError("new_name is required for rename_api refactoring")

    return refactor_type


def _normalize_languages(affected_languages: Optional[List[str]]) -> List[str]:
    """Normalize and filter the affected languages list."""
    if not affected_languages or affected_languages == ["all"]:
        languages = DEFAULT_LANGUAGES
    else:
        languages = affected_languages
    return [lang for lang in languages if lang in SUPPORTED_LANGUAGES]


# =============================================================================
# Main Implementation
# =============================================================================


def refactor_polyglot_impl(
    project_folder: str,
    refactoring_type: str,
    symbol_name: str,
    new_name: Optional[str] = None,
    affected_languages: Optional[List[str]] = None,
    dry_run: bool = True,
) -> PolyglotRefactoringResult:
    """Refactor across multiple languages."""
    start_time = time.time()

    refactor_type = _validate_inputs(project_folder, refactoring_type, new_name)
    languages = _normalize_languages(affected_languages)
    files_by_language = _find_files_with_language(project_folder, languages)
    all_changes = _collect_all_changes(files_by_language, symbol_name, new_name)

    risks = _analyze_risks(all_changes, symbol_name, new_name or "")
    manual_review = _identify_manual_review(project_folder, symbol_name, languages)

    plan = PolyglotRefactoringPlan(
        refactoring_type=refactor_type,
        symbol_name=symbol_name,
        new_name=new_name,
        affected_languages=languages,
        changes=all_changes,
        risks=risks,
        requires_manual_review=manual_review,
    )

    validation_passed, validation_errors = _validate_changes(all_changes, project_folder)

    files_modified: List[str] = []
    backup_id: Optional[str] = None
    should_apply = not dry_run and validation_passed and all_changes
    if should_apply:
        files_modified, backup_id = _apply_changes(all_changes, project_folder)

    return PolyglotRefactoringResult(
        plan=plan,
        dry_run=dry_run,
        changes_made=all_changes if not dry_run else [],
        files_modified=files_modified,
        validation_passed=validation_passed,
        validation_errors=validation_errors,
        execution_time_ms=int((time.time() - start_time) * ConversionFactors.MILLISECONDS_PER_SECOND),
        backup_id=backup_id,
    )
