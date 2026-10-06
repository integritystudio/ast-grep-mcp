"""Shared backup primitives for rewrite and deduplication modules."""

import hashlib
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict

from ast_grep_mcp.constants import FormattingDefaults


def generate_backup_timestamp() -> str:
    """Return a millisecond-precision timestamp string for backup IDs.

    Format: YYYYMMDD-HHMMSS-mmm (last 3 microsecond digits trimmed).
    """
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")[: -FormattingDefaults.TIMESTAMP_MS_TRIM]


def get_file_hash(file_path: str) -> str:
    """Calculate SHA-256 hash of a file's contents.

    Args:
        file_path: Absolute path to the file

    Returns:
        Hex digest of the file's SHA-256 hash, or empty string on error.
    """
    try:
        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except (OSError, IOError):
        return ""


def resolve_backup_dir(prefix: str, timestamp: str, backup_base_dir: Path) -> tuple[str, Path]:
    """Generate a unique backup ID and directory path, handling collisions.

    Args:
        prefix: ID prefix (e.g. "backup", "dedup-backup")
        timestamp: Formatted timestamp string
        backup_base_dir: Base directory containing all backups

    Returns:
        Tuple of (backup_id, backup_dir_path).
    """
    backup_id = f"{prefix}-{timestamp}"
    backup_dir = backup_base_dir / backup_id

    counter = 1
    while backup_dir.exists():
        backup_id = f"{prefix}-{timestamp}-{counter}"
        backup_dir = backup_base_dir / backup_id
        counter += 1

    return backup_id, backup_dir


_EXTERNAL_BACKUP_DIR = "_external"


def copy_file_to_backup(
    file_path: str,
    project_folder: str,
    backup_dir: Path,
    original_hashes: Dict[str, str] | None = None,
) -> Dict[str, str] | None:
    """Copy a single file into a backup directory.

    Args:
        file_path: Source file to back up
        project_folder: Project root for computing relative paths
        backup_dir: Destination backup directory
        original_hashes: Optional dict of file-path → hash.
            When provided, an ``original_hash`` key is added to the
            returned entry (defaulting to ``""`` if the file is missing
            from the dict).  When ``None``, the key is omitted.

    Returns:
        Metadata entry dict, or ``None`` if the source file doesn't exist.
    """
    if not os.path.exists(file_path):
        return None

    rel_path = os.path.relpath(file_path, project_folder)
    if rel_path == os.pardir or rel_path.startswith(os.pardir + os.sep):
        # Outside the project: "../x" would land outside backup_dir and could overwrite a real file
        abs_path = Path(os.path.abspath(file_path))
        rel_path = str(Path(_EXTERNAL_BACKUP_DIR, *abs_path.parts[1:]))
    backup_file_path = backup_dir / rel_path
    backup_file_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(file_path, backup_file_path)

    entry: Dict[str, str] = {
        "original": file_path,
        "relative": rel_path,
        "backup": str(backup_file_path),
    }
    if original_hashes is not None:
        entry["original_hash"] = original_hashes.get(file_path, "")

    return entry


def restore_file_from_backup(
    backup_path: str,
    original_path: str,
    make_parents: bool = False,
) -> str | None:
    """Restore a single file from backup.

    Args:
        backup_path: Path to the backed-up copy
        original_path: Destination path to restore to
        make_parents: Create parent directories if missing

    Returns:
        ``original_path`` on success, ``None`` if backup file is missing.

    Raises:
        Any OS-level error from the copy (caller should handle).
    """
    if not os.path.exists(backup_path):
        return None

    if make_parents:
        os.makedirs(os.path.dirname(original_path), exist_ok=True)

    shutil.copy2(backup_path, original_path)
    return original_path
