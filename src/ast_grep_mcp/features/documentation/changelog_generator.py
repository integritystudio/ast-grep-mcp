"""Changelog generation service.

This module provides functionality for generating changelogs
from git commits using conventional commit format.
"""

import os
import re
import subprocess
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

from ast_grep_mcp.constants import ChangelogDefaults, ConversionFactors, FormattingDefaults, RegexCaptureGroups
from ast_grep_mcp.core.logging import get_logger
from ast_grep_mcp.models.documentation import (
    ChangelogEntry,
    ChangelogResult,
    ChangelogVersion,
    ChangeType,
    CommitInfo,
)

logger = get_logger(__name__)

_GIT_FIELD_SEP = "\x1f"
_GIT_RECORD_SEP = "\x1e"
_GIT_FIELD_SEP_FORMAT = "%x1f"
_GIT_RECORD_SEP_FORMAT = "%x1e"


# =============================================================================
# Git Operations
# =============================================================================


def _run_git_command(project_folder: str, args: List[str]) -> Tuple[bool, str]:
    """Run a git command and return output.

    Args:
        project_folder: Project root
        args: Git command arguments

    Returns:
        Tuple of (success, output)
    """
    if not os.path.isdir(project_folder):
        return False, f"Directory not found: {project_folder}"
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=project_folder,
            capture_output=True,
            text=True,
            check=True,
        )
        return True, result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip()
    except FileNotFoundError:
        return False, "Git not found"


def _resolve_version_ref(project_folder: str, version: str) -> str:
    """Resolve a version string to a git ref.

    Tries v-prefixed tag first, then bare ref, falls back to HEAD.

    Args:
        project_folder: Project root
        version: Version string (tag name, commit, or HEAD)

    Returns:
        Resolved git ref
    """
    if version.upper() == "HEAD":
        return "HEAD"

    # Try v-prefixed tag
    success, _ = _run_git_command(project_folder, ["rev-parse", f"v{version}"])
    if success:
        return f"v{version}"

    # Try bare ref
    success, _ = _run_git_command(project_folder, ["rev-parse", version])
    if success:
        return version

    return "HEAD"


def _get_first_commit(project_folder: str) -> str | None:
    """Get the first commit hash in the repository.

    Args:
        project_folder: Project root

    Returns:
        First commit hash, or None on failure
    """
    success, first_commit = _run_git_command(project_folder, ["rev-list", "--max-parents=0", "HEAD"])
    return first_commit if success else None


def _find_previous_tag(project_folder: str, exclude_ref: str) -> str | None:
    """Find the most recent v-prefixed tag, excluding a given ref.

    Falls back to the first commit if no suitable tag is found.

    Args:
        project_folder: Project root
        exclude_ref: Ref to skip (e.g., current version tag)

    Returns:
        Tag name, first commit hash, or None on failure
    """
    success, tags = _run_git_command(project_folder, ["tag", "--sort=-version:refname", "-l", "v*"])
    if success and tags:
        for tag in tags.split("\n"):
            if tag and tag != exclude_ref:
                return tag

    return _get_first_commit(project_folder)


def _get_commit_range(
    project_folder: str,
    from_version: str | None,
    to_version: str,
) -> Tuple[str, str]:
    """Determine commit range for changelog.

    Args:
        project_folder: Project root
        from_version: Starting version (tag or commit)
        to_version: Ending version (tag, commit, or HEAD)

    Returns:
        Tuple of (from_ref, to_ref)
    """
    to_ref = _resolve_version_ref(project_folder, to_version)

    if from_version:
        from_ref = _resolve_version_ref(project_folder, from_version)
    else:
        previous_tag = _find_previous_tag(project_folder, to_ref)
        if previous_tag is None:
            logger.warning("from_ref_fallback_to_full_history", to_ref=to_ref)
        from_ref = previous_tag or ""

    return from_ref, to_ref


def _get_commits(
    project_folder: str,
    from_ref: str,
    to_ref: str,
) -> List[CommitInfo]:
    """Get commits in range.

    Args:
        project_folder: Project root
        from_ref: Starting reference
        to_ref: Ending reference

    Returns:
        List of CommitInfo objects
    """
    commits: list[CommitInfo] = []

    # Fields: hash, full_hash, author, email, date, subject, body. ASCII unit/record
    # separators cannot occur in commit text, unlike "|" in a subject or author name.
    log_format = _GIT_FIELD_SEP_FORMAT.join(["%h", "%H", "%an", "%ae", "%aI", "%s", "%b"])

    if from_ref:
        range_arg = f"{from_ref}..{to_ref}"
    else:
        range_arg = to_ref

    success, output = _run_git_command(project_folder, ["log", range_arg, f"--format={log_format}{_GIT_RECORD_SEP_FORMAT}"])

    if not success:
        logger.warning("git_log_failed", output=output)
        return commits

    # Parse commits
    for commit_str in output.split(_GIT_RECORD_SEP):
        commit_str = commit_str.strip()
        if not commit_str:
            continue

        parts = commit_str.split(_GIT_FIELD_SEP, ChangelogDefaults.COMMIT_PARTS_COUNT)
        if len(parts) < ChangelogDefaults.COMMIT_PARTS_COUNT:
            continue

        hash_short, hash_full, author, email, date, subject = parts[: ChangelogDefaults.COMMIT_PARTS_COUNT]
        body = parts[ChangelogDefaults.COMMIT_PARTS_COUNT] if len(parts) > ChangelogDefaults.COMMIT_PARTS_COUNT else ""

        # Parse conventional commit format
        parsed = _parse_conventional_commit(subject, body)

        commit = CommitInfo(
            hash=hash_short,
            full_hash=hash_full,
            message=subject,
            body=body,
            author=author,
            author_email=email,
            date=date,
            change_type=parsed.get("type"),
            scope=parsed.get("scope"),
            is_breaking=parsed.get("is_breaking", False),
            issues=parsed.get("issues", []),
            prs=parsed.get("prs", []),
        )
        commits.append(commit)

    return commits


def _parse_conventional_commit(subject: str, body: str) -> Dict[str, Any]:
    """Parse conventional commit format.

    Format: type(scope)!: description

    Args:
        subject: Commit subject line
        body: Commit body

    Returns:
        Dict with type, scope, is_breaking, issues, prs
    """
    result: Dict[str, Any] = {
        "type": None,
        "scope": None,
        "is_breaking": False,
        "issues": [],
        "prs": [],
    }

    # Parse type(scope)!: pattern
    pattern = re.compile(r"^(\w+)(?:\(([^)]+)\))?(!)?:\s*(.+)$")
    match = pattern.match(subject)

    if match:
        result["type"] = match.group(RegexCaptureGroups.FIRST).lower()
        result["scope"] = match.group(RegexCaptureGroups.SECOND)
        result["is_breaking"] = bool(match.group(RegexCaptureGroups.THIRD))

    # Check for BREAKING CHANGE in body
    if "BREAKING CHANGE" in body or "BREAKING-CHANGE" in body:
        result["is_breaking"] = True

    # Extract issue references
    issue_pattern = re.compile(r"#(\d+)")
    all_text = subject + " " + body
    result["issues"] = list(set(issue_pattern.findall(all_text)))

    # Extract PR references (common formats)
    pr_pattern = re.compile(r"(?:pull request|pr|merge request|mr)\s*#?(\d+)", re.IGNORECASE)
    result["prs"] = list(set(pr_pattern.findall(all_text)))

    return result


# =============================================================================
# Changelog Formatting
# =============================================================================


def _map_commit_type_to_change_type(commit_type: str | None) -> ChangeType:
    """Map conventional commit type to changelog change type.

    Args:
        commit_type: Conventional commit type (feat, fix, etc.)

    Returns:
        ChangeType enum
    """
    type_map = {
        "feat": ChangeType.ADDED,
        "feature": ChangeType.ADDED,
        "add": ChangeType.ADDED,
        "fix": ChangeType.FIXED,
        "bugfix": ChangeType.FIXED,
        "bug": ChangeType.FIXED,
        "docs": ChangeType.CHANGED,
        "doc": ChangeType.CHANGED,
        "style": ChangeType.CHANGED,
        "refactor": ChangeType.CHANGED,
        "perf": ChangeType.CHANGED,
        "test": ChangeType.CHANGED,
        "chore": ChangeType.CHANGED,
        "build": ChangeType.CHANGED,
        "ci": ChangeType.CHANGED,
        "deprecate": ChangeType.DEPRECATED,
        "deprecated": ChangeType.DEPRECATED,
        "remove": ChangeType.REMOVED,
        "removed": ChangeType.REMOVED,
        "delete": ChangeType.REMOVED,
        "security": ChangeType.SECURITY,
        "sec": ChangeType.SECURITY,
    }

    if commit_type:
        return type_map.get(commit_type.lower(), ChangeType.CHANGED)

    return ChangeType.CHANGED


def _group_commits_by_version(
    commits: List[CommitInfo],
    project_folder: str,
    to_version: str,
) -> List[ChangelogVersion]:
    """Group commits into versions.

    Args:
        commits: List of commits
        project_folder: Project root
        to_version: Target version

    Returns:
        List of ChangelogVersion objects
    """
    # For now, create a single version for all commits
    # A more sophisticated implementation would detect tags and group by them

    if not commits:
        return []

    # Determine version info
    if to_version.upper() == "HEAD":
        version_str = "Unreleased"
        date_str = datetime.now().strftime("%Y-%m-%d")
        is_unreleased = True
    else:
        version_str = to_version
        # Try to get tag date
        success, tag_date = _run_git_command(project_folder, ["log", "-1", "--format=%aI", f"v{to_version}"])
        if success and tag_date:
            date_str = tag_date[: FormattingDefaults.ISO_DATE_LENGTH]  # YYYY-MM-DD
        else:
            date_str = datetime.now().strftime("%Y-%m-%d")
        is_unreleased = False

    # Group entries by change type
    entries: Dict[ChangeType, List[ChangelogEntry]] = {}

    for commit in commits:
        change_type = _map_commit_type_to_change_type(commit.change_type)

        # If breaking change, might want to categorize differently
        if commit.is_breaking:
            # Keep track but don't change category
            pass

        entry = ChangelogEntry(
            change_type=change_type,
            description=commit.message,
            commit_hash=commit.hash,
            scope=commit.scope,
            is_breaking=commit.is_breaking,
            issues=commit.issues,
            prs=commit.prs,
        )

        if change_type not in entries:
            entries[change_type] = []
        entries[change_type].append(entry)

    return [
        ChangelogVersion(
            version=version_str,
            date=date_str,
            entries=entries,
            is_unreleased=is_unreleased,
        )
    ]


# Keep a Changelog section order
_KEEPACHANGELOG_SECTION_ORDER = [
    ChangeType.ADDED,
    ChangeType.CHANGED,
    ChangeType.DEPRECATED,
    ChangeType.REMOVED,
    ChangeType.FIXED,
    ChangeType.SECURITY,
]


def _format_changelog_entry(entry: ChangelogEntry) -> str:
    """Format a single changelog entry.

    Args:
        entry: Changelog entry

    Returns:
        Formatted entry string
    """
    # Remove conventional commit prefix if present
    msg = re.sub(r"^(\w+)(?:\([^)]+\))?!?:\s*", "", entry.description)

    # Add scope if present
    if entry.scope:
        msg = f"**{entry.scope}:** {msg}"

    # Mark breaking changes
    if entry.is_breaking:
        msg = f"**BREAKING:** {msg}"

    # Add references
    refs = []
    if entry.issues:
        refs.extend([f"#{i}" for i in entry.issues])
    if entry.prs:
        refs.extend([f"PR #{p}" for p in entry.prs])
    if entry.commit_hash:
        refs.append(f"({entry.commit_hash})")

    if refs:
        msg = f"{msg} {' '.join(refs)}"

    return f"- {msg}"


def _format_keepachangelog_version(version: ChangelogVersion) -> List[str]:
    """Format a single version for Keep a Changelog.

    Args:
        version: Version to format

    Returns:
        List of formatted lines
    """
    lines = []

    # Version header
    if version.is_unreleased:
        lines.append("## [Unreleased]")
    else:
        lines.append(f"## [{version.version}] - {version.date}")
    lines.append("")

    # Sections in order
    for change_type in _KEEPACHANGELOG_SECTION_ORDER:
        entries = version.entries.get(change_type, [])
        if not entries:
            continue

        lines.append(f"### {change_type.value}")
        lines.append("")
        lines.extend([_format_changelog_entry(e) for e in entries])
        lines.append("")

    return lines


def _format_keepachangelog(versions: List[ChangelogVersion], project_name: str = "") -> str:
    """Format changelog in Keep a Changelog format.

    Args:
        versions: List of versions with entries
        project_name: Project name for header

    Returns:
        Markdown string
    """
    lines = [
        "# Changelog",
        "",
        "All notable changes to this project will be documented in this file.",
        "",
        "The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),",
        "and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).",
        "",
    ]

    for version in versions:
        lines.extend(_format_keepachangelog_version(version))

    return "\n".join(lines)


# Conventional changelog type names
_CONVENTIONAL_TYPE_NAMES: Dict[ChangeType, str] = {
    ChangeType.ADDED: "Features",
    ChangeType.FIXED: "Bug Fixes",
    ChangeType.CHANGED: "Changes",
    ChangeType.REMOVED: "Removed",
    ChangeType.SECURITY: "Security",
}

# Conventional changelog section order
_CONVENTIONAL_SECTION_ORDER = [
    ChangeType.ADDED,
    ChangeType.FIXED,
    ChangeType.CHANGED,
    ChangeType.REMOVED,
    ChangeType.SECURITY,
]


def _format_conventional_entry(entry: ChangelogEntry) -> str:
    """Format a single entry for conventional changelog.

    Args:
        entry: Changelog entry

    Returns:
        Formatted entry string (without bullet)
    """
    msg = re.sub(r"^(\w+)(?:\([^)]+\))?!?:\s*", "", entry.description)
    commit_ref = f" ({entry.commit_hash})" if entry.commit_hash else ""
    if entry.is_breaking:
        msg = f"**BREAKING:** {msg}"
    return f"{msg}{commit_ref}"


def _format_conventional_section(
    change_type: ChangeType,
    entries: List[ChangelogEntry],
) -> List[str]:
    """Format a section of the conventional changelog.

    Args:
        change_type: The type of changes
        entries: Entries for this type

    Returns:
        List of formatted lines
    """
    if not entries:
        return []

    type_name = _CONVENTIONAL_TYPE_NAMES.get(change_type, "Other")
    lines = [f"### {type_name}", ""]

    # Group by scope
    by_scope: Dict[str, List[ChangelogEntry]] = {}
    for entry in entries:
        scope = entry.scope or "general"
        by_scope.setdefault(scope, []).append(entry)

    for scope, scope_entries in sorted(by_scope.items()):
        if len(by_scope) > 1:
            lines.append(f"* **{scope}**")
            for entry in scope_entries:
                lines.append(f"  * {_format_conventional_entry(entry)}")
        else:
            for entry in scope_entries:
                lines.append(f"* {_format_conventional_entry(entry)}")

    lines.append("")
    return lines


def _format_conventional(versions: List[ChangelogVersion], project_name: str = "") -> str:
    """Format changelog in Conventional Changelog format.

    Args:
        versions: List of versions with entries
        project_name: Project name for header

    Returns:
        Markdown string
    """
    lines = [f"# {project_name or 'Project'} Changelog", ""]

    for version in versions:
        # Version header
        if version.is_unreleased:
            lines.append("## Unreleased")
        else:
            lines.append(f"## {version.version} ({version.date})")
        lines.append("")

        # Sections in order
        for change_type in _CONVENTIONAL_SECTION_ORDER:
            entries = version.entries.get(change_type, [])
            lines.extend(_format_conventional_section(change_type, entries))

    return "\n".join(lines)


# =============================================================================
# Main Generator
# =============================================================================


def generate_changelog_impl(
    project_folder: str,
    from_version: str | None = None,
    to_version: str = "HEAD",
    changelog_format: str = "keepachangelog",
    group_by: str = "type",
) -> ChangelogResult:
    """Generate changelog from git commits.

    Args:
        project_folder: Root folder of the project
        from_version: Starting version (tag or None for last tag)
        to_version: Ending version (tag or HEAD)
        changelog_format: Output format ('keepachangelog', 'conventional', 'json')
        group_by: How to group entries ('type', 'scope')

    Returns:
        ChangelogResult with generated changelog
    """
    start_time = time.time()

    logger.info(
        "generate_changelog_started",
        project_folder=project_folder,
        from_version=from_version,
        to_version=to_version,
        format=changelog_format,
    )

    # Check if git repo
    success, _ = _run_git_command(project_folder, ["rev-parse", "--git-dir"])
    if not success:
        logger.warning("not_a_git_repository")
        return ChangelogResult(
            versions=[],
            markdown="# Changelog\n\nNot a git repository.",
            commits_processed=0,
            commits_skipped=0,
            execution_time_ms=int((time.time() - start_time) * ConversionFactors.MILLISECONDS_PER_SECOND),
        )

    # Determine commit range
    from_ref, to_ref = _get_commit_range(project_folder, from_version, to_version)

    # Get commits
    commits = _get_commits(project_folder, from_ref, to_ref)

    # Count commits with/without conventional format
    commits_processed = len(commits)
    commits_skipped = sum(1 for c in commits if not c.change_type)

    # Group into versions
    versions = _group_commits_by_version(commits, project_folder, to_version)

    # Format output
    project_name = os.path.basename(project_folder)

    if changelog_format == "keepachangelog":
        markdown = _format_keepachangelog(versions, project_name)
    elif changelog_format == "conventional":
        markdown = _format_conventional(versions, project_name)
    else:  # json format handled by tool layer
        markdown = _format_keepachangelog(versions, project_name)

    execution_time = int((time.time() - start_time) * ConversionFactors.MILLISECONDS_PER_SECOND)

    logger.info(
        "generate_changelog_completed",
        commits_processed=commits_processed,
        commits_skipped=commits_skipped,
        versions=len(versions),
        execution_time_ms=execution_time,
    )

    return ChangelogResult(
        versions=versions,
        markdown=markdown,
        commits_processed=commits_processed,
        commits_skipped=commits_skipped,
        execution_time_ms=execution_time,
    )
