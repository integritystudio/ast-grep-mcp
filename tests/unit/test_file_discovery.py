"""Tests for the shared source-file discovery helper (DRY-07)."""

from pathlib import Path

from ast_grep_mcp.utils.file_discovery import find_source_files, language_extensions


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x = 1\n", encoding="utf-8")
    return path


class TestLanguageExtensions:
    def test_known_language(self) -> None:
        assert ".py" in language_extensions("python")

    def test_unknown_language_falls_back_to_dotted_name(self) -> None:
        assert language_extensions("dart") == [".dart"]

    def test_custom_fallback(self) -> None:
        assert language_extensions("unknown", fallback=[".py", ".js"]) == [".py", ".js"]


class TestFindSourceFiles:
    def test_discovers_by_language_extension(self, tmp_path: Path) -> None:
        py = _touch(tmp_path / "src" / "app.py")
        _touch(tmp_path / "src" / "notes.txt")

        assert find_source_files(tmp_path, "python") == [py]

    def test_exclude_patterns_drop_matches(self, tmp_path: Path) -> None:
        kept = _touch(tmp_path / "src" / "app.py")
        _touch(tmp_path / "src" / "app_test.py")

        files = find_source_files(tmp_path, "python", ["**/*_test.py"])

        assert files == [kept]

    def test_skip_dir_names_filter_path_components(self, tmp_path: Path) -> None:
        kept = _touch(tmp_path / "src" / "app.py")
        _touch(tmp_path / "node_modules" / "pkg" / "index.py")
        distribution = _touch(tmp_path / "distribution" / "mod.py")

        files = find_source_files(tmp_path, "python", skip_dir_names={"node_modules", "dist"})

        assert files == [distribution, kept] or files == sorted([kept, distribution])

    def test_custom_extensions_override_language(self, tmp_path: Path) -> None:
        md = _touch(tmp_path / "README.md")
        _touch(tmp_path / "app.py")

        assert find_source_files(tmp_path, "python", extensions=[".md"]) == [md]

    def test_missing_root_returns_empty(self, tmp_path: Path) -> None:
        assert find_source_files(tmp_path / "absent", "python") == []

    def test_result_is_sorted_and_deduplicated(self, tmp_path: Path) -> None:
        b = _touch(tmp_path / "b.py")
        a = _touch(tmp_path / "a.pyi")

        files = find_source_files(tmp_path, "python")

        assert files == [a, b]
