"""Command execution and ast-grep interface for ast-grep MCP server."""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import AsyncGenerator
from typing import Any, Dict, Generator, List, Optional, Tuple, cast

import sentry_sdk
import yaml

from ast_grep_mcp.constants import DisplayDefaults, ExecutorDefaults, FileConstants, FilePatterns, FormattingDefaults, StreamDefaults
from ast_grep_mcp.core.config import CONFIG_PATH
from ast_grep_mcp.core.exceptions import (
    AstGrepExecutionError,
    AstGrepNotFoundError,
)
from ast_grep_mcp.core.logging import get_logger
from ast_grep_mcp.utils.tool_context import tool_context


def _load_custom_languages() -> List[str]:
    """Load custom language names from sgconfig.yml.

    Returns:
        List of custom language names, or empty list if none configured or on error.
    """
    # https://ast-grep.github.io/advanced/custom-language.html#register-language-in-sgconfig-yml
    if not CONFIG_PATH or not os.path.exists(CONFIG_PATH):
        return []
    try:
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
            if config and "customLanguages" in config:
                return list(config["customLanguages"].keys())
    except Exception as e:
        get_logger("executor").debug("custom_language_load_error", config_path=CONFIG_PATH, error=str(e))
    return []


def get_supported_languages() -> List[str]:
    """Get all supported languages as a field description string."""
    languages = [  # https://ast-grep.github.io/reference/languages.html
        "bash",
        "c",
        "cpp",
        "csharp",
        "css",
        "elixir",
        "go",
        "haskell",
        "html",
        "java",
        "javascript",
        "json",
        "jsx",
        "kotlin",
        "lua",
        "nix",
        "php",
        "python",
        "ruby",
        "rust",
        "scala",
        "solidity",
        "swift",
        "tsx",
        "typescript",
        "yaml",
    ]
    languages += _load_custom_languages()
    return sorted(set(languages))


def _execute_subprocess(
    args: List[str], input_text: Optional[str], allow_nonzero: bool, *, use_shell: bool
) -> subprocess.CompletedProcess[str]:
    """Run subprocess wrapped in a Sentry span.

    Args:
        args: Command arguments list
        input_text: Optional stdin input
        allow_nonzero: If True, don't raise on non-zero exit codes
        use_shell: Whether to use shell execution (Windows ast-grep)

    Returns:
        CompletedProcess instance
    """
    with sentry_sdk.start_span(op="subprocess.run", name=f"Running {args[0]}") as span:
        span.set_data("command", args[0])
        span.set_data("has_stdin", input_text is not None)
        try:
            result = subprocess.run(
                args,
                capture_output=True,
                input=input_text,
                text=True,
                check=not allow_nonzero,
                shell=use_shell,
            )
        except subprocess.CalledProcessError as e:
            span.set_data("returncode", e.returncode)
            raise
        span.set_data("returncode", result.returncode)
    return result


def run_command(args: List[str], input_text: Optional[str] = None, *, allow_nonzero: bool = False) -> subprocess.CompletedProcess[str]:
    """Execute a command with proper error handling.

    Args:
        args: Command arguments list
        input_text: Optional stdin input
        allow_nonzero: If True, don't raise on non-zero exit codes

    Returns:
        CompletedProcess instance

    Raises:
        AstGrepNotFoundError: If command binary not found
        AstGrepExecutionError: If command execution fails (unless allow_nonzero)
    """
    logger = get_logger("subprocess")
    has_stdin = input_text is not None
    logger.info("executing_command", command=args[0], args=args[1:], has_stdin=has_stdin)

    with tool_context("run_command", command=" ".join(args), has_stdin=has_stdin) as run:
        try:
            use_shell = sys.platform == "win32" and args[0] == ExecutorDefaults.AST_GREP_COMMAND
            result = _execute_subprocess(args, input_text, allow_nonzero, use_shell=use_shell)
            run.add_completion_fields(command=args[0], returncode=result.returncode)
            return result
        except subprocess.CalledProcessError as e:
            stderr_msg = e.stderr.strip() if e.stderr else ""
            raise AstGrepExecutionError(command=args, returncode=e.returncode, stderr=stderr_msg) from e
        except FileNotFoundError as e:
            if args[0] == ExecutorDefaults.AST_GREP_COMMAND:
                raise AstGrepNotFoundError() from e
            raise AstGrepNotFoundError(f"Command '{args[0]}' not found") from e


def _get_language_extensions(language: str) -> Optional[List[str]]:
    """Get file extensions for a language.

    Args:
        language: Programming language name

    Returns:
        List of extensions or None if language not found
    """
    lang_map = {
        "python": [".py", ".pyi"],
        "javascript": [".js", ".jsx", ".mjs"],
        "typescript": [".ts", ".tsx"],
        "java": [".java"],
        "rust": [".rs"],
        "go": [".go"],
        "c": [".c", ".h"],
        "cpp": [".cpp", ".hpp", ".cc", ".cxx", ".h"],
        "ruby": [".rb"],
        "php": [".php"],
        "swift": [".swift"],
        "kotlin": [".kt", ".kts"],
    }
    return lang_map.get(language.lower())


def _should_skip_directory(dirname: str) -> bool:
    """Check if directory should be skipped.

    Args:
        dirname: Directory name

    Returns:
        True if should skip, False otherwise
    """
    if dirname.startswith("."):
        return True
    return dirname in FilePatterns.SKIP_DIR_NAMES


def _process_file(
    file: str, root: str, lang_extensions: Optional[List[str]], max_size_bytes: int, logger: Any
) -> Tuple[Optional[str], Optional[str]]:
    """Process a single file for size filtering.

    Args:
        file: File name
        root: Root directory path
        lang_extensions: Language-specific extensions
        max_size_bytes: Maximum size in bytes
        logger: Logger instance

    Returns:
        Tuple of (file_to_search, skipped_file) - only one will be non-None
    """
    # Skip hidden files
    if file.startswith("."):
        return (None, None)

    # Check language filter
    if lang_extensions and not any(file.endswith(ext) for ext in lang_extensions):
        return (None, None)

    file_path = os.path.join(root, file)

    try:
        file_size = os.path.getsize(file_path)

        if file_size > max_size_bytes:
            logger.debug(
                "file_skipped_size",
                file=file_path,
                size_mb=round(file_size / FileConstants.BYTES_PER_MB, 2),
                max_size_mb=max_size_bytes / FileConstants.BYTES_PER_MB,
            )
            return (None, file_path)

        return (file_path, None)

    except OSError as e:
        logger.debug("file_stat_error", file=file_path, error=str(e))
        return (None, None)


def _walk_and_classify(
    directory: str, lang_extensions: Optional[List[str]], max_size_bytes: int, logger: Any
) -> Tuple[List[str], List[str]]:
    """Walk directory tree and classify files by size.

    Returns:
        Tuple of (files_to_search, skipped_files)
    """
    files_to_search: List[str] = []
    skipped_files: List[str] = []

    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if not _should_skip_directory(d)]

        for file in files:
            found, skipped = _process_file(file, root, lang_extensions, max_size_bytes, logger)
            if found:
                files_to_search.append(found)
            elif skipped:
                skipped_files.append(skipped)

    return files_to_search, skipped_files


def filter_files_by_size(directory: str, max_size_mb: Optional[int] = None, language: Optional[str] = None) -> Tuple[List[str], List[str]]:
    """Filter files in directory by size.

    Args:
        directory: Directory to search
        max_size_mb: Maximum file size in megabytes. Must be > 0 to apply filtering.
            Pass None or <= 0 only when the caller will handle the unlimited case itself
            (returns ([], []) so callers can detect the no-op and fall back to their
            own file-discovery logic).
        language: Optional language filter for file extensions

    Returns:
        Tuple of (files_to_search, skipped_files)
        - files_to_search: List of file paths under size limit
        - skipped_files: List of file paths that exceeded the limit
        When max_size_mb is None or <= 0 returns ([], []) as a no-op signal.
    """
    if max_size_mb is None or max_size_mb <= 0:
        return ([], [])

    logger = get_logger("file_filter")
    max_size_bytes = max_size_mb * FileConstants.BYTES_PER_MB
    lang_extensions = _get_language_extensions(language) if language else None

    files_to_search, skipped_files = _walk_and_classify(directory, lang_extensions, max_size_bytes, logger)

    if skipped_files:
        logger.info(
            "files_filtered_by_size",
            total_files=len(files_to_search) + len(skipped_files),
            files_to_search=len(files_to_search),
            skipped_files=len(skipped_files),
            max_size_mb=max_size_mb,
        )

    return (files_to_search, skipped_files)


def _write_language_globs_config(language_globs: Dict[str, List[str]], tmpdir: str) -> str:
    """Write a temporary sgconfig.yml with languageGlobs and return its path.

    Args:
        language_globs: Mapping of language name to glob patterns.
        tmpdir: Temporary directory to write the config file into.

    Returns:
        Absolute path to the written sgconfig.yml.
    """
    config_path = os.path.join(tmpdir, "sgconfig.yml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump({"languageGlobs": language_globs}, f)
    return config_path


def run_ast_grep(
    command: str,
    args: List[str],
    input_text: Optional[str] = None,
    language_globs: Optional[Dict[str, List[str]]] = None,
) -> subprocess.CompletedProcess[str]:
    """Execute ast-grep command with optional config.

    Args:
        command: ast-grep subcommand (run, scan, etc.)
        args: Command arguments
        input_text: Optional stdin input
        language_globs: Optional mapping of language → glob patterns written to a
            temporary sgconfig.yml and passed via --config.  When provided, takes
            precedence over the global CONFIG_PATH.

    Returns:
        CompletedProcess instance
    """
    if language_globs:
        tmpdir = tempfile.mkdtemp()
        try:
            config_path = _write_language_globs_config(language_globs, tmpdir)
            effective_args = ["--config", config_path] + args
            allow_nonzero = any(arg.startswith("--debug-query") for arg in effective_args)
            return run_command([ExecutorDefaults.AST_GREP_COMMAND, command] + effective_args, input_text, allow_nonzero=allow_nonzero)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    if CONFIG_PATH:
        args = ["--config", CONFIG_PATH] + args
    # --debug-query outputs to stderr and returns exit code 1 even on success
    allow_nonzero = any(arg.startswith("--debug-query") for arg in args)
    return run_command([ExecutorDefaults.AST_GREP_COMMAND, command] + args, input_text, allow_nonzero=allow_nonzero)


def _prepare_stream_command(command: str, args: List[str], config_override: Optional[str] = None) -> List[str]:
    """Prepare the full ast-grep command with optional config.

    Args:
        command: ast-grep subcommand
        args: Command arguments
        config_override: Path to a config file that takes precedence over CONFIG_PATH.

    Returns:
        Full command list
    """
    final_args = args.copy()
    effective_config = config_override or CONFIG_PATH
    if effective_config:
        final_args = ["--config", effective_config] + final_args
    return [ExecutorDefaults.AST_GREP_COMMAND, command] + final_args





def _parse_json_line(line: str, logger: Any) -> Optional[Dict[str, Any]]:
    """Parse a line of JSON output.

    Args:
        line: Line to parse
        logger: Logger instance

    Returns:
        Parsed JSON dict or None if invalid
    """
    line = line.strip()
    if not line:
        return None

    try:
        return cast(Dict[str, Any], json.loads(line))
    except json.JSONDecodeError as e:
        logger.warning("stream_json_parse_error", line_preview=line[: FileConstants.LINE_PREVIEW_LENGTH], error=str(e))
        sentry_sdk.capture_exception(e)
        sentry_sdk.add_breadcrumb(
            message="JSON parse error in ast-grep stream",
            category="ast-grep.stream",
            level="warning",
            data={"line_preview": line[: FileConstants.LINE_PREVIEW_LENGTH]},
        )
        return None


def _should_log_progress(match_count: int, last_progress_log: int, progress_interval: int) -> bool:
    """Check if progress should be logged.

    Args:
        match_count: Current match count
        last_progress_log: Last logged count
        progress_interval: Interval for logging

    Returns:
        True if should log progress
    """
    if progress_interval <= 0:
        return False
    return match_count - last_progress_log >= progress_interval





def _handle_stream_error(
    returncode: int, stderr_output: str, full_command: List[str], start_time: float, match_count: int, logger: Any
) -> None:
    """Handle non-zero return codes from the process.

    Args:
        returncode: Process return code
        stderr_output: Captured stderr text
        full_command: Command that was run
        start_time: Start time for execution
        match_count: Number of matches found
        logger: Logger instance

    Raises:
        AstGrepExecutionError: If process failed
    """
    # SIGTERM from early termination is not an error
    if returncode in (0, 1, StreamDefaults.SIGTERM_RETURN_CODE):
        return

    execution_time = time.time() - start_time

    logger.error(
        "stream_failed",
        returncode=returncode,
        stderr=stderr_output[: DisplayDefaults.ERROR_OUTPUT_PREVIEW_LENGTH],
        execution_time_seconds=round(execution_time, FormattingDefaults.ROUNDING_PRECISION),
    )

    error = AstGrepExecutionError(command=full_command, returncode=returncode, stderr=stderr_output)
    sentry_sdk.capture_exception(
        error,
        extras={
            "command": " ".join(full_command),
            "returncode": returncode,
            "match_count": match_count,
        },
    )
    raise error











def _log_stream_completion(match_count: int, start_time: float, max_results: int, logger: Any) -> None:
    """Log final stream metrics."""
    execution_time = time.time() - start_time
    logger.info(
        "stream_completed",
        total_matches=match_count,
        execution_time_seconds=round(execution_time, FormattingDefaults.ROUNDING_PRECISION),
        early_terminated=max_results > 0 and match_count >= max_results,
    )


def _raise_not_found_error(full_command: List[str], cause: Exception) -> None:
    """Raise an AstGrepNotFoundError with Sentry capture."""
    if full_command[0] == ExecutorDefaults.AST_GREP_COMMAND:
        error = AstGrepNotFoundError()
    else:
        error = AstGrepNotFoundError(f"Command '{full_command[0]}' not found")
    sentry_sdk.capture_exception(error, extras={"command": " ".join(full_command)})
    raise error from cause


async def _async_terminate_process(
    process: asyncio.subprocess.Process, logger: Any, reason: str
) -> None:
    """Terminate a process gracefully, then forcefully if needed."""
    logger.info(f"stream_{reason}")
    process.terminate()
    try:
        await asyncio.wait_for(
            process.wait(), timeout=StreamDefaults.PROCESS_TERMINATE_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        process.kill()
        try:
            await asyncio.wait_for(
                process.wait(), timeout=StreamDefaults.PROCESS_KILL_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            logger.error("process_kill_timeout", pid=process.pid)


def _log_stream_progress(
    match_count: int,
    last_progress_log: int,
    progress_interval: int,
    start_time: float,
    logger: Any,
) -> int:
    """Log streaming progress at the configured interval.

    Returns the updated last-progress-log watermark.
    """
    if not _should_log_progress(match_count, last_progress_log, progress_interval):
        return last_progress_log
    logger.info(
        "stream_progress",
        matches_found=match_count,
        execution_time_seconds=round(
            time.time() - start_time, FormattingDefaults.ROUNDING_PRECISION
        ),
    )
    return match_count


async def _terminate_stream_early(
    process: "asyncio.subprocess.Process",
    match_count: int,
    max_results: int,
    logger: Any,
) -> None:
    """Log and terminate the process once max_results is reached."""
    logger.info(
        "stream_early_termination",
        matches_found=match_count,
        max_results=max_results,
    )
    await _async_terminate_process(process, logger, "early_termination")


async def _async_iter_stdout_matches(
    process: "asyncio.subprocess.Process",
    max_results: int,
    progress_interval: int,
    start_time: float,
    logger: Any,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Iterate over stdout lines, yielding parsed matches.

    Reads raw bytes to avoid asyncio StreamReader buffer limits (8KB default).
    Manually handles newline splitting to support large JSON outputs.
    """
    if not process.stdout:
        return

    match_count = 0
    last_progress_log = 0
    buffer = b''

    # Read in 64KB chunks to handle large lines
    chunk_size = 65536

    while True:
        chunk = await process.stdout.read(chunk_size)
        if not chunk:
            # Flush remaining buffer
            match = _parse_json_line(buffer.decode(), logger) if buffer else None
            if match:
                yield match
            break

        buffer += chunk
        lines = buffer.split(b'\n')
        buffer = lines[-1]  # Keep incomplete line

        for line in lines[:-1]:
            match = _parse_json_line(line.decode(), logger) if line else None
            if not match:
                continue

            match_count += 1
            last_progress_log = _log_stream_progress(
                match_count, last_progress_log, progress_interval, start_time, logger
            )

            yield match

            if max_results > 0 and match_count >= max_results:
                await _terminate_stream_early(process, match_count, max_results, logger)
                return


async def async_stream_ast_grep_results(
    command: str,
    args: List[str],
    max_results: int = 0,
    progress_interval: int = StreamDefaults.PROGRESS_INTERVAL,
    language_globs: Optional[Dict[str, List[str]]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Async version of stream_ast_grep_results using asyncio.subprocess.

    Eliminates threading.Thread stderr drain; concurrent stdout/stderr via asyncio tasks.
    Adds wall-clock timeout via asyncio.wait_for.
    """
    logger = get_logger("stream_results")
    start_time = time.time()

    tmpdir: Optional[str] = None
    config_override: Optional[str] = None
    if language_globs:
        tmpdir = tempfile.mkdtemp()
        config_override = _write_language_globs_config(language_globs, tmpdir)

    full_command = _prepare_stream_command(command, args, config_override=config_override)
    logger.info(
        "stream_started",
        command=command,
        max_results=max_results,
        progress_interval=progress_interval,
    )

    process: Optional[asyncio.subprocess.Process] = None
    try:
        process = await asyncio.create_subprocess_exec(
            *full_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        match_count = 0
        if process.stderr is None:  # guaranteed non-None by asyncio.subprocess.PIPE
            raise RuntimeError("subprocess stderr pipe unavailable")
        stderr_task = asyncio.create_task(process.stderr.read())

        async for match in _async_iter_stdout_matches(
            process, max_results, progress_interval, start_time, logger
        ):
            match_count += 1
            yield match

        returncode = await process.wait()
        stderr_output = (await stderr_task).decode(errors="replace")
        _handle_stream_error(
            returncode, stderr_output, full_command, start_time, match_count, logger
        )
        _log_stream_completion(match_count, start_time, max_results, logger)

    except FileNotFoundError as e:
        logger.error("stream_command_not_found", command=full_command[0])
        _raise_not_found_error(full_command, e)

    finally:
        if process and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(
                    process.wait(), timeout=StreamDefaults.PROCESS_KILL_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                process.kill()
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


def stream_ast_grep_results(
    command: str,
    args: List[str],
    max_results: int = 0,
    progress_interval: int = StreamDefaults.PROGRESS_INTERVAL,
    language_globs: Optional[Dict[str, List[str]]] = None,
) -> Generator[Dict[str, Any], None, None]:
    """Sync shim — delegates to async_stream_ast_grep_results via a new event loop."""
    async def _collect() -> List[Dict[str, Any]]:
        return [
            m async for m in async_stream_ast_grep_results(
                command, args, max_results, progress_interval, language_globs
            )
        ]

    yield from asyncio.run(_collect())
