"""Tests for DeduplicationBackupManager - backup management for deduplication operations.

Tests cover:
- Initialization
- Backup creation with metadata and file hashing
- Rollback/restore from backups
- Old backup cleanup
- File hash calculation
- Backup listing
- Error handling and edge cases
"""

import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from ast_grep_mcp.features.deduplication.applicator_backup import (
    DeduplicationBackupManager,
)
from ast_grep_mcp.utils.backup import copy_file_to_backup


class TestDeduplicationBackupManagerInit:
    """Tests for DeduplicationBackupManager initialization."""

    def test_init_sets_project_folder(self):
        """Test that initialization sets project folder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            assert manager.project_folder == tmpdir

    def test_init_sets_backup_base_dir(self):
        """Test that initialization sets backup base directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            expected = Path(tmpdir) / ".ast-grep-backups"
            assert manager.backup_base_dir == expected

    def test_init_creates_logger(self):
        """Test that initialization creates a logger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            assert manager.logger is not None


class TestCreateBackup:
    """Tests for create_backup method."""

    def test_creates_backup_directory(self):
        """Test that backup creates the backup directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            # Create a file to backup
            file_path = os.path.join(tmpdir, "test.py")
            with open(file_path, "w") as f:
                f.write("original content")

            backup_id = manager.create_backup([file_path], {"test": "metadata"})

            backup_dir = manager.backup_base_dir / backup_id
            assert backup_dir.exists()

    def test_returns_backup_id(self):
        """Test that create_backup returns a backup ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            file_path = os.path.join(tmpdir, "test.py")
            with open(file_path, "w") as f:
                f.write("content")

            backup_id = manager.create_backup([file_path], {})

            assert backup_id.startswith("dedup-backup-")

    def test_copies_files_to_backup(self):
        """Test that files are copied to backup directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            file_path = os.path.join(tmpdir, "module.py")
            with open(file_path, "w") as f:
                f.write("def func(): pass")

            backup_id = manager.create_backup([file_path], {})

            # Check file exists in backup
            backup_dir = manager.backup_base_dir / backup_id
            backup_file = backup_dir / "module.py"
            assert backup_file.exists()

            with open(backup_file) as f:
                assert f.read() == "def func(): pass"

    def test_preserves_directory_structure(self):
        """Test that directory structure is preserved in backup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            # Create nested file
            nested_dir = os.path.join(tmpdir, "src", "utils")
            os.makedirs(nested_dir)
            file_path = os.path.join(nested_dir, "helper.py")
            with open(file_path, "w") as f:
                f.write("content")

            backup_id = manager.create_backup([file_path], {})

            backup_dir = manager.backup_base_dir / backup_id
            backup_file = backup_dir / "src" / "utils" / "helper.py"
            assert backup_file.exists()

    def test_saves_metadata_file(self):
        """Test that metadata file is saved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            file_path = os.path.join(tmpdir, "test.py")
            with open(file_path, "w") as f:
                f.write("content")

            backup_id = manager.create_backup([file_path], {"custom": "data"})

            metadata_path = manager.backup_base_dir / backup_id / "backup-metadata.json"
            assert metadata_path.exists()

            with open(metadata_path) as f:
                metadata = json.load(f)

            assert metadata["backup_id"] == backup_id
            assert metadata["backup_type"] == "deduplication"
            assert "timestamp" in metadata
            assert metadata["deduplication_metadata"]["custom"] == "data"

    def test_stores_file_hashes(self):
        """Test that file hashes are stored in metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            file_path = os.path.join(tmpdir, "test.py")
            content = "test content for hashing"
            with open(file_path, "w") as f:
                f.write(content)

            expected_hash = hashlib.sha256(content.encode()).hexdigest()

            backup_id = manager.create_backup([file_path], {})

            metadata_path = manager.backup_base_dir / backup_id / "backup-metadata.json"
            with open(metadata_path) as f:
                metadata = json.load(f)

            assert file_path in metadata["deduplication_metadata"]["original_hashes"]
            assert metadata["deduplication_metadata"]["original_hashes"][file_path] == expected_hash

    def test_skips_nonexistent_files(self):
        """Test that nonexistent files are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            # Create one real file
            real_file = os.path.join(tmpdir, "real.py")
            with open(real_file, "w") as f:
                f.write("real content")

            fake_file = os.path.join(tmpdir, "nonexistent.py")

            backup_id = manager.create_backup([real_file, fake_file], {})

            metadata_path = manager.backup_base_dir / backup_id / "backup-metadata.json"
            with open(metadata_path) as f:
                metadata = json.load(f)

            # Only real file should be in backup
            assert len(metadata["files"]) == 1
            assert metadata["files"][0]["original"] == real_file

    def test_handles_backup_id_collision(self):
        """Test that backup ID collision is handled by appending counter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            file_path = os.path.join(tmpdir, "test.py")
            with open(file_path, "w") as f:
                f.write("content")

            # The code does: timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
            # So strftime returns e.g. "20240101-120000-000123" and [:-3] gives "20240101-120000-000"
            strftime_return = "20240101-120000-000123"  # Will be sliced to "20240101-120000-000"
            expected_timestamp = strftime_return[:-3]  # "20240101-120000-000"

            # Create directories that would collide
            collision_dir1 = manager.backup_base_dir / f"dedup-backup-{expected_timestamp}"
            collision_dir1.mkdir(parents=True, exist_ok=True)

            collision_dir2 = manager.backup_base_dir / f"dedup-backup-{expected_timestamp}-1"
            collision_dir2.mkdir(parents=True, exist_ok=True)

            # Now create a backup with the same timestamp - should get -2 suffix
            with (
                patch("ast_grep_mcp.utils.backup.datetime") as mock_ts_dt,
                patch("ast_grep_mcp.features.deduplication.applicator_backup.datetime") as mock_dt,
            ):
                mock_ts_dt.now.return_value.strftime.return_value = strftime_return
                mock_dt.now.return_value.isoformat.return_value = "2024-01-01T12:00:00"

                backup_id = manager.create_backup([file_path], {})

            # Backup should have counter suffix -2 (since -1 also exists)
            assert backup_id == f"dedup-backup-{expected_timestamp}-2"

    def test_multiple_files_backup(self):
        """Test backing up multiple files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            files = []
            for i in range(3):
                file_path = os.path.join(tmpdir, f"file{i}.py")
                with open(file_path, "w") as f:
                    f.write(f"content {i}")
                files.append(file_path)

            backup_id = manager.create_backup(files, {})

            metadata_path = manager.backup_base_dir / backup_id / "backup-metadata.json"
            with open(metadata_path) as f:
                metadata = json.load(f)

            assert len(metadata["files"]) == 3


class TestRollback:
    """Tests for rollback method."""

    def test_restores_files(self):
        """Test that rollback restores files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            file_path = os.path.join(tmpdir, "test.py")
            original_content = "original content"
            with open(file_path, "w") as f:
                f.write(original_content)

            backup_id = manager.create_backup([file_path], {})

            # Modify file
            with open(file_path, "w") as f:
                f.write("modified content")

            # Rollback
            restored = manager.rollback(backup_id)

            assert file_path in restored
            with open(file_path) as f:
                assert f.read() == original_content

    def test_returns_restored_files_list(self):
        """Test that rollback returns list of restored files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            files = []
            for i in range(2):
                file_path = os.path.join(tmpdir, f"file{i}.py")
                with open(file_path, "w") as f:
                    f.write(f"content {i}")
                files.append(file_path)

            backup_id = manager.create_backup(files, {})

            # Modify files
            for fp in files:
                with open(fp, "w") as f:
                    f.write("modified")

            restored = manager.rollback(backup_id)

            assert len(restored) == 2

    def test_raises_for_invalid_backup_id(self):
        """Test that rollback raises for invalid backup ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            with pytest.raises(ValueError, match="not found"):
                manager.rollback("nonexistent-backup-id")

    def test_skips_missing_backup_files(self):
        """Test that rollback skips missing backup files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            file_path = os.path.join(tmpdir, "test.py")
            with open(file_path, "w") as f:
                f.write("content")

            backup_id = manager.create_backup([file_path], {})

            # Delete backup file
            backup_dir = manager.backup_base_dir / backup_id
            backup_file = backup_dir / "test.py"
            os.unlink(backup_file)

            # Should not raise, just skip
            restored = manager.rollback(backup_id)

            assert len(restored) == 0

    def test_handles_restore_error(self):
        """Test that restore errors are logged but don't crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            file_path = os.path.join(tmpdir, "test.py")
            with open(file_path, "w") as f:
                f.write("content")

            backup_id = manager.create_backup([file_path], {})

            # Mock shutil.copy2 to raise an exception
            with patch("shutil.copy2", side_effect=PermissionError("No permission")):
                restored = manager.rollback(backup_id)

            # Should return empty list due to error
            assert len(restored) == 0


class TestCleanupOldBackups:
    """Tests for cleanup_old_backups method."""

    def test_removes_old_backups(self):
        """Test that old backups are removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            file_path = os.path.join(tmpdir, "test.py")
            with open(file_path, "w") as f:
                f.write("content")

            backup_id = manager.create_backup([file_path], {})

            # Manually modify timestamp to make it old
            metadata_path = manager.backup_base_dir / backup_id / "backup-metadata.json"
            with open(metadata_path) as f:
                metadata = json.load(f)

            old_date = datetime.now() - timedelta(days=60)
            metadata["timestamp"] = old_date.isoformat()

            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

            removed = manager.cleanup_old_backups(days=30)

            assert removed == 1
            assert not (manager.backup_base_dir / backup_id).exists()

    def test_keeps_recent_backups(self):
        """Test that recent backups are kept."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            file_path = os.path.join(tmpdir, "test.py")
            with open(file_path, "w") as f:
                f.write("content")

            backup_id = manager.create_backup([file_path], {})

            removed = manager.cleanup_old_backups(days=30)

            assert removed == 0
            assert (manager.backup_base_dir / backup_id).exists()

    def test_returns_zero_when_no_backup_dir(self):
        """Test that cleanup returns 0 when backup dir doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            removed = manager.cleanup_old_backups(days=30)

            assert removed == 0

    def test_skips_non_directories(self):
        """Test that non-directory entries are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            # Create backup base dir with a file (not directory)
            manager.backup_base_dir.mkdir(parents=True)
            stray_file = manager.backup_base_dir / "stray_file.txt"
            stray_file.write_text("stray")

            removed = manager.cleanup_old_backups(days=30)

            assert removed == 0

    def test_skips_dirs_without_metadata(self):
        """Test that directories without metadata are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            # Create backup dir without metadata
            fake_backup = manager.backup_base_dir / "fake-backup"
            fake_backup.mkdir(parents=True)

            removed = manager.cleanup_old_backups(days=30)

            assert removed == 0

    def test_skips_empty_timestamp(self):
        """Test that backups without timestamp are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            # Create backup with empty timestamp
            backup_dir = manager.backup_base_dir / "backup-no-timestamp"
            backup_dir.mkdir(parents=True)
            metadata_path = backup_dir / "backup-metadata.json"
            with open(metadata_path, "w") as f:
                json.dump({"timestamp": ""}, f)

            removed = manager.cleanup_old_backups(days=30)

            assert removed == 0

    def test_handles_cleanup_errors(self):
        """Test that cleanup errors are handled gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            file_path = os.path.join(tmpdir, "test.py")
            with open(file_path, "w") as f:
                f.write("content")

            backup_id = manager.create_backup([file_path], {})

            # Make backup old
            metadata_path = manager.backup_base_dir / backup_id / "backup-metadata.json"
            with open(metadata_path) as f:
                metadata = json.load(f)
            old_date = datetime.now() - timedelta(days=60)
            metadata["timestamp"] = old_date.isoformat()
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

            # Mock shutil.rmtree to raise
            with patch("shutil.rmtree", side_effect=PermissionError("No permission")):
                removed = manager.cleanup_old_backups(days=30)

            # Should not crash, just skip
            assert removed == 0


class TestGetFileHash:
    """Tests for get_file_hash method."""

    def test_returns_sha256_hash(self):
        """Test that correct SHA-256 hash is returned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            file_path = os.path.join(tmpdir, "test.txt")
            content = b"test content for hashing"
            with open(file_path, "wb") as f:
                f.write(content)

            expected = hashlib.sha256(content).hexdigest()
            result = manager.get_file_hash(file_path)

            assert result == expected

    def test_returns_empty_string_on_error(self):
        """Test that empty string is returned on error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            result = manager.get_file_hash("/nonexistent/file.py")

            assert result == ""

    def test_handles_io_error(self):
        """Test that IOError is handled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            file_path = os.path.join(tmpdir, "test.py")
            with open(file_path, "w") as f:
                f.write("content")

            with patch("builtins.open", side_effect=IOError("Read error")):
                result = manager.get_file_hash(file_path)

            assert result == ""


class TestListBackups:
    """Tests for list_backups method."""

    def test_returns_empty_list_when_no_backups(self):
        """Test that empty list is returned when no backups exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            backups = manager.list_backups()

            assert backups == []

    def test_returns_backup_metadata(self):
        """Test that backup metadata is returned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            file_path = os.path.join(tmpdir, "test.py")
            with open(file_path, "w") as f:
                f.write("content")

            backup_id = manager.create_backup([file_path], {"key": "value"})

            backups = manager.list_backups()

            assert len(backups) == 1
            assert backups[0]["backup_id"] == backup_id

    def test_sorts_by_timestamp_newest_first(self):
        """Test that backups are sorted by timestamp (newest first)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            file_path = os.path.join(tmpdir, "test.py")
            with open(file_path, "w") as f:
                f.write("content")

            backup_id1 = manager.create_backup([file_path], {})
            time.sleep(0.01)  # Small delay to ensure different timestamps
            backup_id2 = manager.create_backup([file_path], {})

            backups = manager.list_backups()

            assert len(backups) == 2
            assert backups[0]["backup_id"] == backup_id2  # Newest first
            assert backups[1]["backup_id"] == backup_id1

    def test_skips_non_directories(self):
        """Test that non-directory entries are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            # Create backup base dir with a file
            manager.backup_base_dir.mkdir(parents=True)
            stray_file = manager.backup_base_dir / "stray.txt"
            stray_file.write_text("stray")

            backups = manager.list_backups()

            assert backups == []

    def test_skips_dirs_without_metadata(self):
        """Test that directories without metadata are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            # Create fake backup dir without metadata
            fake_backup = manager.backup_base_dir / "fake-backup"
            fake_backup.mkdir(parents=True)

            backups = manager.list_backups()

            assert backups == []

    def test_handles_metadata_read_errors(self):
        """Test that metadata read errors are handled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            # Create backup dir with invalid JSON metadata
            backup_dir = manager.backup_base_dir / "invalid-backup"
            backup_dir.mkdir(parents=True)
            metadata_path = backup_dir / "backup-metadata.json"
            with open(metadata_path, "w") as f:
                f.write("invalid json{{{")

            backups = manager.list_backups()

            # Should not crash, just skip invalid backup
            assert backups == []


class TestGenerateBackupId:
    """Tests for _generate_backup_id helper."""

    def test_returns_prefixed_id(self):
        """Test that generated ID has the expected prefix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            backup_id, backup_dir = manager._generate_backup_id()

            assert backup_id.startswith("dedup-backup-")
            assert backup_dir == manager.backup_base_dir / backup_id

    def test_no_collision_suffix_when_dir_absent(self):
        """Test that no counter suffix is added when no collision exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            backup_id, _backup_dir = manager._generate_backup_id()

            # Should not have a trailing -N counter
            parts = backup_id.split("-")
            # Format: dedup-backup-YYYYMMDD-HHMMSS-fff
            assert len(parts) == 5


class TestComputeFileHashes:
    """Tests for _compute_file_hashes helper."""

    def test_returns_hashes_for_existing_files(self):
        """Test that hashes are computed for existing files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            fp = os.path.join(tmpdir, "a.py")
            with open(fp, "w") as f:
                f.write("hello")

            hashes = manager._compute_file_hashes([fp])

            assert fp in hashes
            assert len(hashes[fp]) == 64  # SHA-256 hex length

    def test_skips_nonexistent_files(self):
        """Test that nonexistent files are excluded from result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            hashes = manager._compute_file_hashes(["/nonexistent/file.py"])

            assert hashes == {}

    def test_mixed_existing_and_missing(self):
        """Test with a mix of real and missing files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            real = os.path.join(tmpdir, "real.py")
            with open(real, "w") as f:
                f.write("data")

            hashes = manager._compute_file_hashes([real, "/no/such/file.py"])

            assert len(hashes) == 1
            assert real in hashes


class TestCopyFileToBackup:
    """Tests for shared copy_file_to_backup utility."""

    def test_returns_none_for_missing_file(self):
        """Test that None is returned when source file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backup_dir = Path(tmpdir) / "backup"
            backup_dir.mkdir()

            result = copy_file_to_backup("/no/such/file.py", tmpdir, backup_dir, {})

            assert result is None

    def test_returns_entry_dict_with_correct_keys(self):
        """Test that returned dict has the expected keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backup_dir = Path(tmpdir) / "backup"
            backup_dir.mkdir()

            fp = os.path.join(tmpdir, "mod.py")
            with open(fp, "w") as f:
                f.write("code")

            result = copy_file_to_backup(fp, tmpdir, backup_dir, {fp: "abc123"})

            assert result is not None
            assert set(result.keys()) == {"original", "relative", "backup", "original_hash"}
            assert result["original"] == fp
            assert result["original_hash"] == "abc123"

    def test_falls_back_to_empty_hash_when_missing(self):
        """Test that original_hash defaults to empty string when file not in hashes dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backup_dir = Path(tmpdir) / "backup"
            backup_dir.mkdir()

            fp = os.path.join(tmpdir, "mod.py")
            with open(fp, "w") as f:
                f.write("code")

            result = copy_file_to_backup(fp, tmpdir, backup_dir, {})

            assert result is not None
            assert result["original_hash"] == ""

    def test_omits_hash_key_when_hashes_none(self):
        """Test that original_hash key is omitted when original_hashes is None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backup_dir = Path(tmpdir) / "backup"
            backup_dir.mkdir()

            fp = os.path.join(tmpdir, "mod.py")
            with open(fp, "w") as f:
                f.write("code")

            result = copy_file_to_backup(fp, tmpdir, backup_dir)

            assert result is not None
            assert "original_hash" not in result


class TestRestoreSingleFile:
    """Tests for _restore_single_file helper."""

    def test_returns_path_on_success(self):
        """Test that original path is returned after successful restore."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            original = os.path.join(tmpdir, "orig.py")
            backup = os.path.join(tmpdir, "bak.py")

            with open(backup, "w") as f:
                f.write("original content")
            with open(original, "w") as f:
                f.write("modified")

            result = manager._restore_single_file({"original": original, "backup": backup})

            assert result == original
            with open(original) as f:
                assert f.read() == "original content"

    def test_returns_none_when_backup_missing(self):
        """Test that None is returned when backup file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            result = manager._restore_single_file({"original": "/some/file.py", "backup": "/no/such/backup.py"})

            assert result is None

    def test_returns_none_on_copy_error(self):
        """Test that None is returned when copy2 raises."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            backup = os.path.join(tmpdir, "bak.py")
            with open(backup, "w") as f:
                f.write("data")

            with patch("ast_grep_mcp.features.deduplication.applicator_backup.shutil.copy2", side_effect=PermissionError("denied")):
                result = manager._restore_single_file({"original": "/dst/file.py", "backup": backup})

            assert result is None


class TestTryCleanupBackup:
    """Tests for _try_cleanup_backup helper."""

    def test_returns_false_for_no_metadata(self):
        """Test that False is returned when metadata is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)
            empty_dir = Path(tmpdir) / "empty"
            empty_dir.mkdir()

            result = manager._try_cleanup_backup(empty_dir, datetime.now())

            assert result is False

    def test_returns_false_for_empty_timestamp(self):
        """Test that False is returned when timestamp is empty string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)
            bdir = Path(tmpdir) / "bak"
            bdir.mkdir()
            with open(bdir / "backup-metadata.json", "w") as f:
                json.dump({"timestamp": ""}, f)

            result = manager._try_cleanup_backup(bdir, datetime.now())

            assert result is False

    def test_returns_false_for_malformed_timestamp(self):
        """Test that False is returned when timestamp is not valid ISO format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)
            bdir = Path(tmpdir) / "bak"
            bdir.mkdir()
            with open(bdir / "backup-metadata.json", "w") as f:
                json.dump({"timestamp": "not-a-date"}, f)

            result = manager._try_cleanup_backup(bdir, datetime.now())

            assert result is False
            assert bdir.exists()  # directory not removed

    def test_returns_false_for_recent_backup(self):
        """Test that False is returned and dir kept when backup is recent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)
            bdir = Path(tmpdir) / "bak"
            bdir.mkdir()
            with open(bdir / "backup-metadata.json", "w") as f:
                json.dump({"timestamp": datetime.now().isoformat()}, f)

            cutoff = datetime.now() - timedelta(days=30)
            result = manager._try_cleanup_backup(bdir, cutoff)

            assert result is False
            assert bdir.exists()

    def test_returns_true_and_removes_old_backup(self):
        """Test that True is returned and dir removed when backup is old."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)
            bdir = Path(tmpdir) / "bak"
            bdir.mkdir()
            old_date = datetime.now() - timedelta(days=60)
            with open(bdir / "backup-metadata.json", "w") as f:
                json.dump({"timestamp": old_date.isoformat(), "backup_id": "test"}, f)

            cutoff = datetime.now() - timedelta(days=30)
            result = manager._try_cleanup_backup(bdir, cutoff)

            assert result is True
            assert not bdir.exists()


class TestCreateBackupEdgeCases:
    """Edge case tests for create_backup."""

    def test_empty_file_list(self):
        """Test backup with no files produces valid metadata with empty files list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            backup_id = manager.create_backup([], {"reason": "test"})

            metadata_path = manager.backup_base_dir / backup_id / "backup-metadata.json"
            with open(metadata_path) as f:
                metadata = json.load(f)

            assert metadata["files"] == []
            assert metadata["deduplication_metadata"]["original_hashes"] == {}


class TestRollbackEdgeCases:
    """Edge case tests for rollback."""

    def test_empty_files_in_metadata(self):
        """Test rollback with backup that has empty files list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            backup_id = manager.create_backup([], {})

            restored = manager.rollback(backup_id)

            assert restored == []


class TestCleanupMultiple:
    """Tests for cleanup_old_backups with multiple backups."""

    def test_removes_multiple_old_keeps_recent(self):
        """Test that multiple old backups are removed while recent ones kept."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            fp = os.path.join(tmpdir, "test.py")
            with open(fp, "w") as f:
                f.write("content")

            # Create 3 backups
            ids = []
            for _ in range(3):
                ids.append(manager.create_backup([fp], {}))
                time.sleep(0.01)

            # Make first two old
            old_date = datetime.now() - timedelta(days=60)
            for bid in ids[:2]:
                mp = manager.backup_base_dir / bid / "backup-metadata.json"
                with open(mp) as f:
                    meta = json.load(f)
                meta["timestamp"] = old_date.isoformat()
                with open(mp, "w") as f:
                    json.dump(meta, f)

            removed = manager.cleanup_old_backups(days=30)

            assert removed == 2
            assert not (manager.backup_base_dir / ids[0]).exists()
            assert not (manager.backup_base_dir / ids[1]).exists()
            assert (manager.backup_base_dir / ids[2]).exists()


class TestIntegration:
    """Integration tests for backup workflow."""

    def test_full_backup_and_restore_workflow(self):
        """Test complete backup and restore workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            # Create files
            files = []
            for i in range(3):
                file_path = os.path.join(tmpdir, f"module{i}.py")
                with open(file_path, "w") as f:
                    f.write(f"original content {i}")
                files.append(file_path)

            # Create backup
            backup_id = manager.create_backup(files, {"operation": "test"})

            # Modify files
            for fp in files:
                with open(fp, "w") as f:
                    f.write("modified content")

            # Verify modification
            for fp in files:
                with open(fp) as f:
                    assert f.read() == "modified content"

            # Restore
            restored = manager.rollback(backup_id)

            # Verify restoration
            assert len(restored) == 3
            for i, fp in enumerate(files):
                with open(fp) as f:
                    assert f.read() == f"original content {i}"

    def test_backup_list_and_cleanup(self):
        """Test backup listing and cleanup workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            file_path = os.path.join(tmpdir, "test.py")
            with open(file_path, "w") as f:
                f.write("content")

            # Create multiple backups
            backup_ids = []
            for _ in range(3):
                backup_id = manager.create_backup([file_path], {})
                backup_ids.append(backup_id)
                time.sleep(0.01)

            # List backups
            backups = manager.list_backups()
            assert len(backups) == 3

            # Make first backup old
            first_backup_dir = manager.backup_base_dir / backup_ids[0]
            metadata_path = first_backup_dir / "backup-metadata.json"
            with open(metadata_path) as f:
                metadata = json.load(f)
            old_date = datetime.now() - timedelta(days=60)
            metadata["timestamp"] = old_date.isoformat()
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

            # Cleanup old backups
            removed = manager.cleanup_old_backups(days=30)
            assert removed == 1

            # Verify remaining backups
            backups = manager.list_backups()
            assert len(backups) == 2


class TestCreatedFilesTracking:
    """Tests for rollback of newly created files (BUG-08)."""

    def test_create_backup_stores_created_files_in_metadata(self):
        """created_files list is persisted in backup metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            existing = os.path.join(tmpdir, "existing.py")
            with open(existing, "w") as f:
                f.write("original")

            new_file = os.path.join(tmpdir, "_extracted_utils.py")

            backup_id = manager.create_backup(
                files=[existing],
                metadata={},
                created_files=[new_file],
            )

            metadata_path = manager.backup_base_dir / backup_id / "backup-metadata.json"
            with open(metadata_path) as f:
                metadata = json.load(f)

            assert metadata["created_files"] == [new_file]

    def test_create_backup_created_files_defaults_to_empty(self):
        """created_files defaults to [] when not supplied."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            fp = os.path.join(tmpdir, "mod.py")
            with open(fp, "w") as f:
                f.write("code")

            backup_id = manager.create_backup(files=[fp], metadata={})

            metadata_path = manager.backup_base_dir / backup_id / "backup-metadata.json"
            with open(metadata_path) as f:
                metadata = json.load(f)

            assert metadata["created_files"] == []

    def test_rollback_deletes_newly_created_file(self):
        """Rollback deletes a file that was created during refactoring."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            existing = os.path.join(tmpdir, "source.py")
            with open(existing, "w") as f:
                f.write("original content")

            new_file = os.path.join(tmpdir, "_extracted_utils.py")

            backup_id = manager.create_backup(
                files=[existing],
                metadata={},
                created_files=[new_file],
            )

            # Simulate the refactoring: modify existing file and create new file
            with open(existing, "w") as f:
                f.write("modified content")
            with open(new_file, "w") as f:
                f.write("def extracted(): pass")

            assert os.path.exists(new_file)

            manager.rollback(backup_id)

            # Existing file should be restored
            with open(existing) as f:
                assert f.read() == "original content"

            # Newly created file should be deleted
            assert not os.path.exists(new_file)

    def test_rollback_restores_appended_target_file(self):
        """Rollback restores an existing file that was appended to during extract."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            shared_utils = os.path.join(tmpdir, "utils.py")
            with open(shared_utils, "w") as f:
                f.write("def helper(): pass\n")

            source = os.path.join(tmpdir, "source.py")
            with open(source, "w") as f:
                f.write("duplicate code here\n")

            # Both the source and the append-target are in the backup set
            backup_id = manager.create_backup(
                files=[source, shared_utils],
                metadata={},
                created_files=[],
            )

            # Simulate append: add extracted function to the shared utils file
            with open(shared_utils, "a") as f:
                f.write("\ndef extracted(): pass\n")
            with open(source, "w") as f:
                f.write("from utils import extracted\nextracted()\n")

            manager.rollback(backup_id)

            with open(shared_utils) as f:
                assert f.read() == "def helper(): pass\n"
            with open(source) as f:
                assert f.read() == "duplicate code here\n"

    def test_rollback_skips_already_deleted_created_file(self):
        """Rollback does not error when a tracked created file no longer exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            fp = os.path.join(tmpdir, "mod.py")
            with open(fp, "w") as f:
                f.write("code")

            ghost_file = os.path.join(tmpdir, "ghost.py")

            backup_id = manager.create_backup(
                files=[fp],
                metadata={},
                created_files=[ghost_file],
            )

            # ghost_file was never created; rollback should not raise
            restored = manager.rollback(backup_id)
            assert isinstance(restored, list)

    def test_rollback_handles_delete_error_gracefully(self):
        """Rollback logs but does not raise when a created file cannot be deleted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeduplicationBackupManager(tmpdir)

            fp = os.path.join(tmpdir, "mod.py")
            with open(fp, "w") as f:
                f.write("code")

            new_file = os.path.join(tmpdir, "new.py")
            backup_id = manager.create_backup(
                files=[fp],
                metadata={},
                created_files=[new_file],
            )

            with open(new_file, "w") as f:
                f.write("def fn(): pass")

            with patch("os.remove", side_effect=PermissionError("denied")):
                # Should not raise
                restored = manager.rollback(backup_id)

            assert isinstance(restored, list)
            assert os.path.exists(new_file)


class TestCopyFileToBackupContainment:
    """A file outside the project must still be backed up inside backup_dir."""

    def test_outside_file_stays_inside_backup_dir(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        backup_dir = project / ".backups" / "b1"
        victim = tmp_path / "victim.txt"
        outside = tmp_path / "elsewhere" / "victim.txt"
        project.mkdir()
        outside.parent.mkdir()
        victim.write_text("do not overwrite")
        outside.write_text("outside content")

        entry = copy_file_to_backup(str(outside), str(project), backup_dir)

        assert entry is not None
        assert Path(entry["backup"]).resolve().is_relative_to(backup_dir.resolve())
        assert Path(entry["backup"]).read_text() == "outside content"
        assert victim.read_text() == "do not overwrite"
