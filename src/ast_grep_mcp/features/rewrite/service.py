"""Rewrite feature service - implements code transformation functionality."""

import json
import os
import re
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple

import sentry_sdk
import yaml

from ast_grep_mcp.constants import DisplayDefaults, FormattingDefaults, SyntaxValidationDefaults
from ast_grep_mcp.core.exceptions import InvalidYAMLError
from ast_grep_mcp.core.executor import (
    filter_files_by_size,
    run_ast_grep,
    stream_ast_grep_results,
)
from ast_grep_mcp.core.logging import get_logger
from ast_grep_mcp.features.rewrite.backup import (
    create_backup,
    list_available_backups,
    restore_backup,
)
from ast_grep_mcp.utils.subprocess_runner import TOOL_NOT_FOUND, run_tool

_TSC_ERROR_PATTERN: Pattern[str] = re.compile(SyntaxValidationDefaults.TSC_SYNTAX_ERROR_PATTERN)
_JSX_NODE_KINDS = ["jsx_element", "jsx_self_closing_element"]


def _validate_python_syntax(content: str, file_path: str) -> Dict[str, Any]:
    """Validate Python syntax.

    Args:
        content: File content
        file_path: Path to file (for error messages)

    Returns:
        Dict with 'valid' and 'error' keys
    """
    try:
        compile(content, file_path, "exec")
        return {"valid": True, "error": None}
    except SyntaxError as e:
        return {"valid": False, "error": f"Line {e.lineno}: {e.msg}"}


def _parse_node_error(stderr: str) -> str:
    if stderr:
        return stderr.strip().split("\n")[0]
    return "Syntax error"


def _run_node_check(tmp_path: str) -> Dict[str, Any]:
    """Run node --check on a file and return validity result."""
    result = run_tool(["node", "--check", tmp_path], SyntaxValidationDefaults.NODE_TIMEOUT_SECONDS)
    if result.error is not None:
        return {"valid": True, "error": "JavaScript validation failed"}
    if result.returncode != 0:
        return {"valid": False, "error": _parse_node_error(result.stderr)}
    return {"valid": True, "error": None}


def _count_tree_sitter_nodes(file_path: str, kinds: List[str]) -> int:
    """Count nodes of the given kinds in file_path, parsed with ast-grep's JavaScript (JSX-aware) grammar."""
    rule = yaml.dump({"id": "syntax-check", "language": "javascript", "rule": {"any": [{"kind": k} for k in kinds]}})
    output = run_ast_grep("scan", ["--inline-rules", rule, "--json", file_path]).stdout.strip()
    return len(json.loads(output)) if output else 0


def _validate_jsx_syntax(file_path: str) -> Dict[str, Any]:
    """Validate JSX via tree-sitter ERROR nodes; node --check cannot parse JSX."""
    if _count_tree_sitter_nodes(file_path, ["ERROR"]):
        return {"valid": False, "error": "Syntax error (tree-sitter ERROR node)"}
    return {"valid": True, "error": None}


def _validate_javascript_file(content: str, file_path: str) -> Dict[str, Any]:
    """node --check, deferring to the JSX-aware check when the file contains JSX."""
    validation = _validate_javascript_syntax(content)
    if not validation["valid"] and _count_tree_sitter_nodes(file_path, _JSX_NODE_KINDS):
        return _validate_jsx_syntax(file_path)
    return validation


def _validate_javascript_syntax(content: str) -> Dict[str, Any]:
    """Validate JavaScript syntax using Node.js with ESM support.

    Writes content to a temp .mjs file and uses node --check, which
    correctly handles ESM import/export statements.

    Args:
        content: File content

    Returns:
        Dict with 'valid' and 'error' keys
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = os.path.join(tmpdir, "check.mjs")
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(content)
            return _run_node_check(tmp_path)
    except FileNotFoundError:
        return {"valid": True, "error": "JavaScript validation skipped (node not available)"}


def _extract_tsc_syntax_error(combined: str) -> Optional[str]:
    """Return the first TS1xxx syntax error line from tsc output, or None."""
    for line in combined.split("\n"):
        if _TSC_ERROR_PATTERN.search(line):
            return line.strip()
    return None


def _validate_typescript_syntax(file_path: str) -> Dict[str, Any]:
    """Validate TypeScript syntax using tsc.

    Runs tsc --noEmit on the file and filters output for TS1xxx
    syntax errors only, ignoring type errors (TS2xxx+) that may
    arise from missing modules or type definitions.

    Args:
        file_path: Path to the TypeScript file

    Returns:
        Dict with 'valid' and 'error' keys
    """
    result = run_tool(
        [
            "tsc",
            "--noEmit",
            "--noResolve",
            "--skipLibCheck",
            "--module",
            "esnext",
            "--target",
            "esnext",
            "--moduleResolution",
            "bundler",
            file_path,
        ],
        SyntaxValidationDefaults.TSC_TIMEOUT_SECONDS,
    )
    if result.error == TOOL_NOT_FOUND:
        return {"valid": True, "error": "TypeScript validation skipped (tsc not available)"}
    if result.error is not None:
        return {"valid": True, "error": "TypeScript validation skipped (tsc timed out)"}
    combined = result.stdout + result.stderr
    error_line = _extract_tsc_syntax_error(combined)
    if error_line:
        return {"valid": False, "error": error_line}
    return {"valid": True, "error": None}


def _validate_java_syntax(file_path: str) -> Dict[str, Any]:
    """Validate Java syntax using javac.

    Args:
        file_path: Path to Java file

    Returns:
        Dict with 'valid' and 'error' keys
    """
    javac_result = run_tool(["javac", "-Xlint:none", file_path], SyntaxValidationDefaults.JAVAC_TIMEOUT_SECONDS)
    if javac_result.error is not None:
        return {"valid": True, "error": "Java validation skipped (javac not available)"}
    if javac_result.returncode != 0:
        return {"valid": False, "error": javac_result.stderr[: SyntaxValidationDefaults.JAVAC_ERROR_PREVIEW_LENGTH]}

    # Clean up .class file if compilation succeeded
    class_file = file_path.replace(".java", ".class")
    if os.path.exists(class_file):
        os.remove(class_file)
    return {"valid": True, "error": None}


def validate_syntax(file_path: str, language: str) -> Dict[str, Any]:
    """Validate syntax of a rewritten file.

    Args:
        file_path: Absolute path to the file to validate
        language: Programming language (python, javascript, typescript, etc.)

    Returns:
        Dict with 'valid' (bool), 'error' (str if invalid), 'language' (str)
    """
    result: Dict[str, Any] = {"file": file_path, "language": language, "valid": True, "error": None}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Language-specific validators
        validators: Dict[str, Callable[[], Dict[str, Any]]] = {
            "python": lambda: _validate_python_syntax(content, file_path),
            "javascript": lambda: _validate_javascript_file(content, file_path),
            "typescript": lambda: _validate_typescript_syntax(file_path),
            "tsx": lambda: _validate_typescript_syntax(file_path),
            "jsx": lambda: _validate_jsx_syntax(file_path),
            "java": lambda: _validate_java_syntax(file_path),
        }

        # Get validator for language
        validator = validators.get(language)
        if validator:
            validation = validator()
            result["valid"] = validation["valid"]
            result["error"] = validation["error"]
        else:
            result["valid"] = True
            result["error"] = f"Syntax validation not supported for {language}"

    except Exception as e:
        result["valid"] = False
        result["error"] = f"Validation error: {str(e)}"

    return result


def validate_rewrites(modified_files: List[str], language: str) -> Dict[str, Any]:
    """Validate syntax of all rewritten files.

    Args:
        modified_files: List of file paths that were modified
        language: Programming language

    Returns:
        Dict with validation summary and results per file
    """
    validation_results = []
    failed_count = 0
    skipped_count = 0

    for file_path in modified_files:
        result = validate_syntax(file_path, language)
        validation_results.append(result)

        error = result["error"] or ""
        if "not supported" in error or "skipped" in error:
            skipped_count += 1
        elif not result["valid"]:
            failed_count += 1

    return {
        "validated": len(modified_files),
        "passed": len(modified_files) - failed_count - skipped_count,
        "failed": failed_count,
        "skipped": skipped_count,
        "results": validation_results,
    }


def _validate_yaml_rule(yaml_rule: str) -> Dict[str, Any]:
    """Validate and parse YAML rule.

    Args:
        yaml_rule: YAML rule string

    Returns:
        Parsed rule data dictionary

    Raises:
        InvalidYAMLError: If YAML is invalid
        ValueError: If required fields are missing
    """
    try:
        rule_data = yaml.safe_load(yaml_rule)
    except yaml.YAMLError as e:
        raise InvalidYAMLError(f"Invalid YAML rule: {e}", yaml_rule) from e

    if not isinstance(rule_data, dict):
        raise InvalidYAMLError("Rule must be a YAML dictionary", yaml_rule)

    if "fix" not in rule_data:
        raise ValueError("Rule must include a 'fix' field for code rewriting")

    if "language" not in rule_data:
        raise ValueError("Rule must include a 'language' field")

    return rule_data





def _build_command_args(
    rule_file: str, project_folder: str, max_file_size_mb: int, workers: int, language: str
) -> Tuple[List[str], List[str]]:
    """Build command arguments for ast-grep.

    Args:
        rule_file: Path to rule file
        project_folder: Project folder path
        max_file_size_mb: Max file size in MB
        workers: Number of worker threads
        language: Programming language

    Returns:
        Tuple of (args, search_targets)
    """
    args = ["--rule", rule_file]

    # Handle file size filtering
    if max_file_size_mb > 0:
        files_to_search, _ = filter_files_by_size(project_folder, max_size_mb=max_file_size_mb, language=language)
        if files_to_search:
            search_targets = files_to_search
        else:
            search_targets = []
    else:
        search_targets = [project_folder]

    # Add worker threads
    if workers > 0:
        args.extend(["--threads", str(workers)])

    return args, search_targets


def _match_to_change_entry(match: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "file": match.get("file", "unknown"),
        "line": match.get("range", {}).get("start", {}).get("line", 0),
        "original": match.get("text", ""),
        "replacement": match["replacement"],
        "rule_id": match.get("ruleId", "unknown"),
    }


def _format_dry_run_changes(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract change preview entries from stream matches."""
    return [_match_to_change_entry(m) for m in matches if "replacement" in m]


def _log_dry_run_completed(logger: Any, start_time: float, changes_found: int) -> None:
    execution_time = time.time() - start_time
    logger.info(
        "rewrite_code_completed",
        execution_time_seconds=round(execution_time, FormattingDefaults.ROUNDING_PRECISION),
        dry_run=True,
        changes_found=changes_found,
        status="success",
    )


def _perform_dry_run(args: List[str], search_targets: List[str], logger: Any, start_time: float) -> Dict[str, Any]:
    """Perform dry run to preview changes.

    Args:
        args: Command arguments
        search_targets: Files/folders to search
        logger: Logger instance
        start_time: Start time for execution timing

    Returns:
        Dict with dry run results
    """
    if not search_targets:
        return {"message": "No files to rewrite (all exceeded size limit)", "changes": []}

    full_args = args + ["--json=stream"] + search_targets
    matches = list(stream_ast_grep_results("scan", full_args, max_results=0))

    if not matches:
        _log_dry_run_completed(logger, start_time, 0)
        return {"dry_run": True, "message": "No matches found - no changes would be applied", "changes": []}

    changes = _format_dry_run_changes(matches)
    _log_dry_run_completed(logger, start_time, len(changes))
    return {"dry_run": True, "message": f"Found {len(changes)} change(s) - set dry_run=false to apply", "changes": changes}


def _maybe_create_backup(files_to_modify: List[str], project_folder: str, backup: bool, logger: Any) -> Optional[str]:
    if not backup:
        return None
    backup_id = create_backup(files_to_modify, project_folder)
    logger.info("backup_created", backup_id=backup_id, file_count=len(files_to_modify))
    return backup_id


def _run_rewrite_command(rule_file: str, search_targets: List[str], workers: int) -> Any:
    rewrite_args = ["--rule", rule_file, "--update-all"] + search_targets
    if workers > 0:
        rewrite_args.insert(0, "--threads")
        rewrite_args.insert(1, str(workers))
    return run_ast_grep("scan", rewrite_args)


def _build_rewrite_response(
    files_to_modify: List[str],
    backup_id: Optional[str],
    result: Any,
    validation_summary: Dict[str, Any],
) -> Dict[str, Any]:
    response: Dict[str, Any] = {
        "dry_run": False,
        "message": f"Applied changes to {len(files_to_modify)} file(s)",
        "modified_files": files_to_modify,
        "backup_id": backup_id,
        "output": result.stdout,
        "validation": validation_summary,
    }
    if validation_summary["failed"] > 0:
        response["warning"] = (
            f"{validation_summary['failed']} file(s) failed syntax validation. "
            f"Use rollback_rewrite(backup_id='{backup_id}') to restore if needed."
        )
    return response


def _log_rewrite_completed(
    logger: Any,
    start_time: float,
    files_to_modify: List[str],
    backup_id: Optional[str],
    validation_summary: Dict[str, Any],
) -> None:
    logger.info(
        "syntax_validation",
        validated=validation_summary["validated"],
        passed=validation_summary["passed"],
        failed=validation_summary["failed"],
        skipped=validation_summary["skipped"],
    )
    execution_time = time.time() - start_time
    logger.info(
        "rewrite_code_completed",
        execution_time_seconds=round(execution_time, FormattingDefaults.ROUNDING_PRECISION),
        dry_run=False,
        modified_files=len(files_to_modify),
        backup_id=backup_id,
        validation_failed=validation_summary["failed"],
        status="success",
    )


def _apply_rewrites(
    args: List[str],
    search_targets: List[str],
    rule_file: str,
    project_folder: str,
    backup: bool,
    workers: int,
    language: str,
    logger: Any,
    start_time: float,
) -> Dict[str, Any]:
    """Apply actual code rewrites.

    Args:
        args: Command arguments
        search_targets: Files/folders to search
        rule_file: Path to rule file
        project_folder: Project folder path
        backup: Whether to create backup
        workers: Number of worker threads
        language: Programming language
        logger: Logger instance
        start_time: Start time for execution timing

    Returns:
        Dict with rewrite results
    """
    if not search_targets:
        return {"dry_run": False, "message": "No files to rewrite (all exceeded size limit)", "modified_files": [], "backup_id": None}

    preview_args = args + ["--json=stream"] + search_targets
    preview_matches = list(stream_ast_grep_results("scan", preview_args, max_results=0))
    files_to_modify: List[str] = [f for f in set(m.get("file") for m in preview_matches if m.get("file")) if f is not None]

    if not files_to_modify:
        return {"dry_run": False, "message": "No changes applied - no matches found", "modified_files": [], "backup_id": None}

    backup_id = _maybe_create_backup(files_to_modify, project_folder, backup, logger)
    result = _run_rewrite_command(rule_file, search_targets, workers)
    validation_summary = validate_rewrites(files_to_modify, language)
    _log_rewrite_completed(logger, start_time, files_to_modify, backup_id, validation_summary)
    return _build_rewrite_response(files_to_modify, backup_id, result, validation_summary)


def _handle_rewrite_error(e: Exception, logger: Any, start_time: float, project_folder: str, dry_run: bool) -> None:
    execution_time = time.time() - start_time
    logger.error(
        "rewrite_code_failed",
        execution_time_seconds=round(execution_time, FormattingDefaults.ROUNDING_PRECISION),
        error=str(e)[: DisplayDefaults.ERROR_OUTPUT_PREVIEW_LENGTH],
        status="failed",
    )
    sentry_sdk.capture_exception(
        e,
        extras={
            "function": "rewrite_code_impl",
            "project_folder": project_folder,
            "dry_run": dry_run,
            "execution_time_seconds": round(execution_time, FormattingDefaults.ROUNDING_PRECISION),
        },
    )


def _execute_rewrite(
    project_folder: str,
    yaml_rule: str,
    rule_file: str,
    dry_run: bool,
    backup: bool,
    max_file_size_mb: int,
    workers: int,
    logger: Any,
    start_time: float,
) -> Dict[str, Any]:
    """Validate and dispatch rewrite."""
    rule_data = _validate_yaml_rule(yaml_rule)
    language = rule_data.get("language", "unknown")
    args, search_targets = _build_command_args(rule_file, project_folder, max_file_size_mb, workers, language)
    if dry_run:
        return _perform_dry_run(args, search_targets, logger, start_time)
    return _apply_rewrites(args, search_targets, rule_file, project_folder, backup, workers, language, logger, start_time)


def rewrite_code_impl(
    project_folder: str, yaml_rule: str, dry_run: bool = True, backup: bool = True, max_file_size_mb: int = 0, workers: int = 0
) -> Dict[str, Any]:
    """
    Implementation of rewrite_code.

    Rewrite code using ast-grep fix rules. Apply automated code transformations safely.

    Args:
        project_folder: The absolute path to the project folder
        yaml_rule: YAML rule with 'fix' field for code transformation
        dry_run: Preview changes without applying (default: true for safety)
        backup: Create backup before applying changes (default: true)
        max_file_size_mb: Skip files larger than this (0 = unlimited)
        workers: Number of worker threads (0 = auto)

    Returns:
        Dict with rewrite results

    Raises:
        InvalidYAMLError: If YAML is invalid
        ValueError: If rule is missing required fields
    """
    logger = get_logger("rewrite.rewrite_code")
    start_time = time.time()

    logger.info("rewrite_code_started", project_folder=project_folder, dry_run=dry_run, backup=backup, workers=workers)

    with tempfile.TemporaryDirectory() as tmpdir:
        rule_file = os.path.join(tmpdir, "rule.yml")
        with open(rule_file, "w", encoding="utf-8") as f:
            f.write(yaml_rule)
        try:
            return _execute_rewrite(project_folder, yaml_rule, rule_file, dry_run, backup, max_file_size_mb, workers, logger, start_time)
        except Exception as e:
            _handle_rewrite_error(e, logger, start_time, project_folder, dry_run)
            raise


def _build_rollback_response(result: Dict[str, Any]) -> Dict[str, Any]:
    if result["success"]:
        return {
            "success": True,
            "message": f"Successfully restored {len(result['restored_files'])} file(s)",
            "restored_files": result["restored_files"],
        }
    return {
        "success": False,
        "message": "Rollback encountered errors",
        "restored_files": result["restored_files"],
        "errors": result["errors"],
    }


def rollback_rewrite_impl(backup_id: str, project_folder: str) -> Dict[str, Any]:
    """
    Implementation of rollback_rewrite.

    Restore files from a backup created during rewrite operations.

    Args:
        backup_id: The unique identifier of the backup to restore
        project_folder: The absolute path to the project folder

    Returns:
        Dict with rollback results
    """
    logger = get_logger("rewrite.rollback")
    start_time = time.time()

    logger.info("rollback_started", backup_id=backup_id, project_folder=project_folder)

    try:
        result = restore_backup(backup_id, project_folder)

        execution_time = time.time() - start_time
        logger.info(
            "rollback_completed",
            execution_time_seconds=round(execution_time, FormattingDefaults.ROUNDING_PRECISION),
            success=result["success"],
            restored_files=len(result["restored_files"]),
            status="success" if result["success"] else "partial",
        )

        return _build_rollback_response(result)

    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(
            "rollback_failed",
            execution_time_seconds=round(execution_time, FormattingDefaults.ROUNDING_PRECISION),
            error=str(e)[: DisplayDefaults.ERROR_OUTPUT_PREVIEW_LENGTH],
            status="failed",
        )
        sentry_sdk.capture_exception(
            e,
            extras={
                "function": "rollback_rewrite_impl",
                "backup_id": backup_id,
                "project_folder": project_folder,
                "execution_time_seconds": round(execution_time, FormattingDefaults.ROUNDING_PRECISION),
            },
        )
        raise


def list_backups_impl(project_folder: str) -> List[Dict[str, Any]]:
    """
    Implementation of list_backups.

    List all available backups in the project.

    Args:
        project_folder: The absolute path to the project folder

    Returns:
        List of backup information dictionaries
    """
    logger = get_logger("rewrite.list_backups")

    logger.info("list_backups_started", project_folder=project_folder)

    try:
        backups = list_available_backups(project_folder)

        logger.info("list_backups_completed", backup_count=len(backups))

        return backups

    except Exception as e:
        logger.error("list_backups_failed", error=str(e)[: DisplayDefaults.ERROR_OUTPUT_PREVIEW_LENGTH])
        sentry_sdk.capture_exception(e, extras={"function": "list_backups_impl", "project_folder": project_folder})
        raise
