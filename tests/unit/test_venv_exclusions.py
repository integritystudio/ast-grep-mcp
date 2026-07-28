"""Tests for enforcing virtualenv exclusions across tool entry points."""

from ast_grep_mcp.constants import FilePatterns
from ast_grep_mcp.features.complexity.tools import (
    _normalize_complexity_exclude_patterns,
    _prepare_smell_detection_params,
)
from ast_grep_mcp.features.deduplication.detector import DuplicationDetector
from ast_grep_mcp.features.documentation.sync_checker import _find_source_files


def test_file_patterns_merge_with_venv_excludes_adds_required_patterns() -> None:
    merged = FilePatterns.merge_with_venv_excludes(["**/generated/**"])

    assert "**/generated/**" in merged
    for pattern in FilePatterns.VENV_EXCLUDE:
        assert pattern in merged


def test_complexity_exclude_normalization_preserves_custom_and_adds_venv() -> None:
    normalized = _normalize_complexity_exclude_patterns(["**/generated/**"])

    assert "**/generated/**" in normalized
    for pattern in FilePatterns.VENV_EXCLUDE:
        assert pattern in normalized


def test_prepare_smell_detection_params_enforces_venv_excludes() -> None:
    include_patterns, exclude_patterns = _prepare_smell_detection_params(["**/*.py"], ["**/generated/**"])

    assert include_patterns == ["**/*.py"]
    assert "**/generated/**" in exclude_patterns
    for pattern in FilePatterns.VENV_EXCLUDE:
        assert pattern in exclude_patterns


def test_documentation_source_scan_excludes_venv_even_with_custom_excludes(tmp_path) -> None:
    app_file = tmp_path / "src" / "app.py"
    app_file.parent.mkdir(parents=True)
    app_file.write_text("print('ok')\n", encoding="utf-8")

    venv_file = tmp_path / "venv" / "lib" / "python3.12" / "site-packages" / "pkg.py"
    venv_file.parent.mkdir(parents=True)
    venv_file.write_text("print('ignore')\n", encoding="utf-8")

    files = _find_source_files(
        project_folder=str(tmp_path),
        language="python",
        include_patterns=["**/*.py"],
        exclude_patterns=["**/generated/**"],
    )

    assert str(app_file) in files
    assert str(venv_file) not in files


def test_duplication_detector_enforces_venv_excludes_with_custom_patterns(monkeypatch, tmp_path) -> None:
    detector = DuplicationDetector(language="python")

    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(detector, "_validate_parameters", lambda *args, **kwargs: None)
    monkeypatch.setattr(detector, "_get_construct_pattern", lambda *args, **kwargs: "pattern")

    def _fake_find_constructs(project_folder, pattern, max_constructs, exclude_patterns):
        captured["exclude_patterns"] = list(exclude_patterns)
        return [], 0

    monkeypatch.setattr(detector, "_find_constructs", _fake_find_constructs)
    detector.find_duplication(project_folder=str(tmp_path), exclude_patterns=["node_modules"])

    for pattern in ["site-packages", ".venv", "venv", "virtualenv"]:
        assert pattern in captured["exclude_patterns"]


def test_dedup_tools_normalization_enforces_venv_patterns() -> None:
    normalized = FilePatterns.normalize_excludes(["**/node_modules/**"])

    assert "**/node_modules/**" in normalized
    for pattern in FilePatterns.VENV_EXCLUDE:
        assert pattern in normalized
