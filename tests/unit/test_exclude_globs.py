"""Tests that exclude globs match whole path segments relative to the project root."""

import pytest

from ast_grep_mcp.features.complexity.complexity_file_finder import ComplexityFileFinder
from ast_grep_mcp.features.quality.enforcer import should_exclude_file
from ast_grep_mcp.features.quality.security_scanner import _should_skip_file
from ast_grep_mcp.utils.file_discovery import matches_glob


@pytest.mark.parametrize(
    ("path", "pattern", "expected"),
    [
        ("venv/a.py", "**/venv/**", True),
        ("src/venv/a.py", "**/venv/**", True),
        ("devenv/a.py", "**/venv/**", False),
        ("src/build_utils.py", "**/build/**", False),
        ("web/app.min.js", "**/*.min.js", True),
        ("src/latest.py", "**/test*/**", False),
        ("src/test.tmp", "*.tmp", True),
    ],
)
def test_matches_glob_uses_segment_semantics(path: str, pattern: str, expected: bool) -> None:
    assert matches_glob(path, pattern) is expected


def test_directories_above_the_root_are_ignored() -> None:
    assert matches_glob("/home/u/build/proj/src/a.py", "**/build/**", "/home/u/build/proj") is False
    assert should_exclude_file("/home/u/build/proj/src/a.py", ["**/build/**"], "/home/u/build/proj") is False


def test_file_finder_keeps_project_under_lookalike_dir(tmp_path) -> None:
    project = tmp_path / "devenv"
    (project / "src").mkdir(parents=True)
    (project / "venv").mkdir()
    (project / "src" / "app.py").write_text("x = 1\n")
    (project / "venv" / "lib.py").write_text("y = 2\n")

    files = ComplexityFileFinder().find_files(str(project), "python", ["**/*"], ["**/venv/**"])

    assert [p.rsplit("/", 1)[-1] for p in files] == ["app.py"]


def test_secret_scan_skips_dirs_by_segment(tmp_path) -> None:
    assert _should_skip_file(tmp_path / "distance.py", tmp_path) is False
    assert _should_skip_file(tmp_path / "node_modules" / "x.js", tmp_path) is True
