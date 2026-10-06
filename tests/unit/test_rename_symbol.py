"""Tests for symbol renaming functionality."""

from unittest.mock import Mock, patch

import pytest

from ast_grep_mcp.features.refactoring.rename_coordinator import RenameCoordinator
from ast_grep_mcp.features.refactoring.renamer import SymbolRenamer
from ast_grep_mcp.features.refactoring.tools import rename_symbol_tool
from ast_grep_mcp.models.refactoring import ScopeInfo, SymbolReference


@pytest.fixture
def python_renamer():
    """Create a SymbolRenamer for Python."""

    return SymbolRenamer("python")


@pytest.fixture
def typescript_renamer():
    """Create a SymbolRenamer for TypeScript."""
    return SymbolRenamer("typescript")


@pytest.fixture
def python_coordinator():
    """Create a RenameCoordinator for Python."""
    return RenameCoordinator("python")


class TestSymbolRenamer:
    """Tests for SymbolRenamer class."""

    def test_find_symbol_references_simple(self, python_renamer, tmp_path):
        """Test finding simple symbol references in a file."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def calculate_total(items):
    total = 0
    for item in items:
        total += item
    return total
""")

        with patch("ast_grep_mcp.features.refactoring.renamer.run_ast_grep") as mock_run:
            # ast-grep returns absolute paths
            mock_run.return_value = Mock(
                returncode=0,
                stdout=f'[{{"file": "{str(test_file)}", "range": {{"start": {{"line": 2, "column": 4}}}}, "lines": "total = 0"}},'
                f'{{"file": "{str(test_file)}", "range": {{"start": {{"line": 4, "column": 8}}}}, "lines": "total += item"}},'
                f'{{"file": "{str(test_file)}", "range": {{"start": {{"line": 5, "column": 11}}}}, "lines": "return total"}}]',
            )

            references = python_renamer.find_symbol_references(
                project_folder=str(tmp_path),
                symbol_name="total",
            )

            assert len(references) == 3
            assert all(ref.file_path.endswith("test.py") for ref in references)
            assert references[0].line == 3  # 0-indexed + 1 = 1-indexed
            assert references[1].line == 5
            assert references[2].line == 6

    def test_find_symbol_references_no_matches(self, python_renamer, tmp_path):
        """Test finding references when symbol doesn't exist."""
        with patch("ast_grep_mcp.features.refactoring.renamer.run_ast_grep") as mock_run:
            mock_run.return_value = Mock(returncode=1, stdout="")

            references = python_renamer.find_symbol_references(
                project_folder=str(tmp_path),
                symbol_name="nonexistent",
            )

            assert len(references) == 0

    def test_build_scope_tree_python_simple(self, python_renamer, tmp_path):
        """Test building scope tree for simple Python file."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
x = 1

def foo():
    y = 2
    def bar():
        z = 3
    return y

class MyClass:
    def method(self):
        a = 4
""")

        scopes = python_renamer.build_scope_tree(str(test_file))

        # Should have: module, foo, bar, MyClass, MyClass.method
        assert len(scopes) >= 4

        # Check module scope exists
        module_scopes = [s for s in scopes if s.scope_type == "module"]
        assert len(module_scopes) > 0

        # Check function scopes exist
        function_scopes = [s for s in scopes if s.scope_type == "function"]
        assert len(function_scopes) >= 2  # foo and bar

        # Check class scope exists
        class_scopes = [s for s in scopes if s.scope_type == "class"]
        assert len(class_scopes) >= 1

    def test_build_scope_tree_nested_functions(self, python_renamer, tmp_path):
        """Test building scope tree with nested functions."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def outer():
    x = 1
    def inner():
        y = 2
        def innermost():
            z = 3
        return y
    return x
""")

        scopes = python_renamer.build_scope_tree(str(test_file))

        function_scopes = [s for s in scopes if s.scope_type == "function"]
        assert len(function_scopes) >= 3  # outer, inner, innermost

        # Check nesting relationships
        outer_scope = next((s for s in function_scopes if s.scope_name == "outer"), None)
        inner_scope = next((s for s in function_scopes if s.scope_name == "inner"), None)

        assert outer_scope is not None
        assert inner_scope is not None
        assert inner_scope.parent_scope is not None

    def test_check_naming_conflicts_no_conflict(self, python_renamer, tmp_path):
        """Test conflict detection when no conflicts exist."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def foo():
    x = 1
    return x
""")

        scopes = python_renamer.build_scope_tree(str(test_file))

        references = [
            SymbolReference(
                file_path=str(test_file),
                line=2,
                column=4,
                context="x = 1",
                scope="foo",
            )
        ]

        scope_trees = {str(test_file): scopes}
        conflicts = python_renamer.check_naming_conflicts(
            references=references,
            new_name="y",  # No conflict with y
            scopes=scope_trees,
        )

        assert len(conflicts) == 0

    def test_check_naming_conflicts_with_conflict(self, python_renamer, tmp_path):
        """Test conflict detection when conflicts exist."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def foo():
    x = 1
    y = 2
    return x + y
""")

        python_renamer.build_scope_tree(str(test_file))

        # Create scope with defined symbols
        foo_scope = ScopeInfo(
            scope_type="function",
            scope_name="foo",
            start_line=1,
            end_line=4,
            defined_symbols={"x", "y"},
        )

        references = [
            SymbolReference(
                file_path=str(test_file),
                line=2,
                column=4,
                context="x = 1",
                scope="foo",
            )
        ]

        scope_trees = {str(test_file): [foo_scope]}
        conflicts = python_renamer.check_naming_conflicts(
            references=references,
            new_name="y",  # Conflict with existing y
            scopes=scope_trees,
        )

        assert len(conflicts) > 0
        assert "y" in conflicts[0]

    def test_classify_reference_definition(self, python_renamer):
        """Test classifying a definition reference."""
        ref = SymbolReference(
            file_path="/test.py",
            line=1,
            column=0,
            context="def foo():",
            scope="",
        )

        python_renamer._classify_reference(ref)

        assert ref.is_definition
        assert not ref.is_import

    def test_classify_reference_import(self, python_renamer):
        """Test classifying an import reference."""
        ref = SymbolReference(
            file_path="/test.py",
            line=1,
            column=0,
            context="from module import foo",
            scope="",
        )

        python_renamer._classify_reference(ref)

        assert ref.is_import
        assert not ref.is_definition

    def test_classify_reference_export(self, typescript_renamer):
        """Test classifying an export reference."""
        ref = SymbolReference(
            file_path="/test.ts",
            line=1,
            column=0,
            context="export function foo() {}",
            scope="",
        )

        typescript_renamer._classify_reference(ref)

        assert ref.is_export

    def test_classify_reference_usage(self, python_renamer):
        """Test classifying a usage reference."""
        ref = SymbolReference(
            file_path="/test.py",
            line=1,
            column=0,
            context="result = foo()",
            scope="",
        )

        python_renamer._classify_reference(ref)

        assert not ref.is_definition
        assert not ref.is_import
        assert not ref.is_export


class TestScopeTreeSymbols:
    """build_scope_tree must record what each scope defines, or conflict checks never fire."""

    def test_python_conflict_detected_from_real_scope_tree(self, python_renamer, tmp_path):
        test_file = tmp_path / "test.py"
        test_file.write_text("import os\n\n\ndef foo(arg):\n    x = 1\n    y = 2\n    return x + y + arg\n")

        scopes = python_renamer.build_scope_tree(str(test_file))
        by_name = {s.scope_name: s.defined_symbols for s in scopes}
        assert by_name["<module>"] == {"os", "foo"}
        assert by_name["foo"] == {"arg", "x", "y"}

        ref = SymbolReference(file_path=str(test_file), line=5, column=4, context="x = 1", scope="foo")
        conflicts = python_renamer.check_naming_conflicts([ref], "y", {str(test_file): scopes})
        assert len(conflicts) == 1

    def test_typescript_declarations_recorded(self, typescript_renamer, tmp_path):
        test_file = tmp_path / "test.ts"
        test_file.write_text("const total = 1;\nexport function sum(a: number) {\n  let acc = 0;\n  return acc + a;\n}\n")

        scopes = typescript_renamer.build_scope_tree(str(test_file))
        by_name = {s.scope_name: s.defined_symbols for s in scopes}
        assert by_name["<module>"] == {"total", "sum"}
        assert by_name["sum"] == {"acc"}


class TestRenameCoordinator:
    """Tests for RenameCoordinator class."""

    def test_rename_symbol_dry_run(self, python_coordinator, tmp_path):
        """Test renaming a symbol in dry-run mode."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def calculate(x):
    result = x * 2
    return result
""")

        with patch("ast_grep_mcp.features.refactoring.renamer.run_ast_grep") as mock_run:
            # ast-grep returns absolute paths
            mock_run.return_value = Mock(
                returncode=0,
                stdout=f'[{{"file": "{str(test_file)}", "range": {{"start": {{"line": 2, "column": 4}}}}, "lines": "result = x * 2"}},'
                f'{{"file": "{str(test_file)}", "range": {{"start": {{"line": 3, "column": 11}}}}, "lines": "return result"}}]',
            )

            result = python_coordinator.rename_symbol(
                project_folder=str(tmp_path),
                old_name="result",
                new_name="output",
                dry_run=True,
            )

            assert result.success
            assert result.old_name == "result"
            assert result.new_name == "output"
            assert result.references_found == 2
            assert result.references_updated == 0  # Dry run
            assert result.backup_id is None  # Dry run
            assert result.diff_preview is not None

    def test_rename_symbol_no_references(self, python_coordinator, tmp_path):
        """Test renaming when symbol has no references."""
        with patch("ast_grep_mcp.features.refactoring.renamer.run_ast_grep") as mock_run:
            mock_run.return_value = Mock(returncode=1, stdout="")

            result = python_coordinator.rename_symbol(
                project_folder=str(tmp_path),
                old_name="nonexistent",
                new_name="whatever",
                dry_run=True,
            )

            assert not result.success
            assert "No references found" in result.error

    def test_rename_symbol_with_conflicts(self, python_coordinator, tmp_path):
        """Test renaming when naming conflicts exist."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def foo():
    x = 1
    y = 2
    return x + y
""")

        with patch("ast_grep_mcp.features.refactoring.renamer.run_ast_grep") as mock_run:
            # ast-grep returns absolute paths
            mock_run.return_value = Mock(
                returncode=0, stdout=f'[{{"file": "{str(test_file)}", "range": {{"start": {{"line": 2, "column": 4}}}}, "lines": "x = 1"}}]'
            )

            # Mock build_scope_tree to return a scope with 'y' already defined
            with patch.object(python_coordinator.renamer, "build_scope_tree") as mock_scope:
                mock_scope.return_value = [
                    ScopeInfo(
                        scope_type="function",
                        scope_name="foo",
                        start_line=1,
                        end_line=4,
                        defined_symbols={"x", "y"},
                    )
                ]

                with patch.object(python_coordinator.renamer, "check_naming_conflicts") as mock_check:
                    mock_check.return_value = ["test.py:2 - 'y' already defined in scope 'foo'"]

                    result = python_coordinator.rename_symbol(
                        project_folder=str(tmp_path),
                        old_name="x",
                        new_name="y",  # Conflict!
                        dry_run=True,
                    )

                    assert not result.success
                    assert len(result.conflicts) > 0
                    assert "Naming conflicts detected" in result.error

    def test_rename_symbol_apply(self, python_coordinator, tmp_path):
        """Test actually applying a rename (not dry-run)."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def calculate(x):
    result = x * 2
    return result
""")

        with patch("ast_grep_mcp.features.refactoring.renamer.run_ast_grep") as mock_run:
            # ast-grep returns absolute paths
            mock_run.return_value = Mock(
                returncode=0,
                stdout=f'[{{"file": "{str(test_file)}", "range": {{"start": {{"line": 2, "column": 4}}}}, "lines": "result = x * 2"}},'
                f'{{"file": "{str(test_file)}", "range": {{"start": {{"line": 3, "column": 11}}}}, "lines": "return result"}}]',
            )

            with patch("ast_grep_mcp.features.refactoring.rename_coordinator.create_backup") as mock_backup:
                mock_backup.return_value = "backup-123"

                result = python_coordinator.rename_symbol(
                    project_folder=str(tmp_path),
                    old_name="result",
                    new_name="output",
                    dry_run=False,
                )

                assert result.success
                assert result.references_updated == 2
                assert result.backup_id == "backup-123"
                assert len(result.files_modified) == 1

                # Verify file was actually modified
                modified_content = test_file.read_text()
                assert "output" in modified_content
                assert "result" not in modified_content.split("\n")[1]  # Check line 2

    def test_rename_in_file_word_boundary(self, python_coordinator, tmp_path):
        """Test that rename respects word boundaries."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def foo():
    result = 1
    results = [result]  # Should not rename 'results'
    return result
""")

        references = [
            SymbolReference(
                file_path=str(test_file),
                line=3,
                column=4,
                context="result = 1",
                scope="foo",
            ),
            SymbolReference(
                file_path=str(test_file),
                line=4,
                column=15,
                context="results = [result]",
                scope="foo",
            ),
            SymbolReference(
                file_path=str(test_file),
                line=5,
                column=11,
                context="return result",
                scope="foo",
            ),
        ]

        python_coordinator._rename_in_file(
            file_path=str(test_file),
            references=references,
            old_name="result",
            new_name="output",
        )

        modified_content = test_file.read_text()
        assert "output = 1" in modified_content
        assert "return output" in modified_content
        assert "results = [output]" in modified_content  # 'results' stayed, but 'result' inside changed

    def test_generate_diff_preview(self, python_coordinator, tmp_path):
        """Test diff preview generation."""
        test_file = tmp_path / "test.py"

        references = [
            SymbolReference(
                file_path=str(test_file),
                line=2,
                column=4,
                context="result = x * 2",
                scope="calculate",
            ),
            SymbolReference(
                file_path=str(test_file),
                line=3,
                column=11,
                context="return result",
                scope="calculate",
            ),
        ]

        diff = python_coordinator._generate_diff_preview(
            references=references,
            old_name="result",
            new_name="output",
        )

        assert "result = x * 2" in diff
        assert "output = x * 2" in diff
        assert "return result" in diff
        assert "return output" in diff
        assert "Line 2:" in diff
        assert "Line 3:" in diff


class TestRenameSymbolTool:
    """Tests for rename_symbol MCP tool."""

    def test_rename_symbol_tool_dry_run(self, tmp_path):
        """Test rename_symbol tool in dry-run mode."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def calculate(x):
    result = x * 2
    return result
""")

        with patch("ast_grep_mcp.features.refactoring.renamer.run_ast_grep") as mock_run:
            # ast-grep returns absolute paths
            mock_run.return_value = Mock(
                returncode=0,
                stdout=f'[{{"file": "{str(test_file)}", "range": {{"start": {{"line": 2, "column": 4}}}}, "lines": "result = x * 2"}},'
                f'{{"file": "{str(test_file)}", "range": {{"start": {{"line": 3, "column": 11}}}}, "lines": "return result"}}]',
            )

            result = rename_symbol_tool(
                project_folder=str(tmp_path),
                symbol_name="result",
                new_name="output",
                language="python",
                dry_run=True,
            )

            assert result["success"]
            assert result["old_name"] == "result"
            assert result["new_name"] == "output"
            assert result["references_found"] == 2
            assert result["references_updated"] == 0
            assert result["backup_id"] is None
            assert result["diff_preview"] is not None

    def test_rename_symbol_tool_error_handling(self, tmp_path):
        """Test error handling in rename_symbol tool."""
        # Test when renamer itself throws an exception during coordination
        with patch("ast_grep_mcp.features.refactoring.rename_coordinator.RenameCoordinator.rename_symbol") as mock_rename:
            mock_rename.side_effect = Exception("Coordinator failed")

            result = rename_symbol_tool(
                project_folder=str(tmp_path),
                symbol_name="foo",
                new_name="bar",
                language="python",
                dry_run=True,
            )

            assert not result["success"]
            assert "Rename symbol failed" in result["error"]
            assert result["old_name"] == "foo"
            assert result["new_name"] == "bar"

    def test_rename_symbol_tool_with_file_filter(self, tmp_path):
        """Test rename_symbol tool with file filter."""
        with patch("ast_grep_mcp.features.refactoring.renamer.run_ast_grep") as mock_run:
            mock_run.return_value = Mock(returncode=1, stdout="")

            result = rename_symbol_tool(
                project_folder=str(tmp_path),
                symbol_name="foo",
                new_name="bar",
                language="python",
                file_filter="*.py",
                dry_run=True,
            )

            # Should fail due to no references, but file_filter should be passed through
            assert not result["success"]


class TestMultiFileRename:
    """Integration tests for multi-file symbol renaming."""

    def test_rename_across_multiple_files(self, python_coordinator, tmp_path):
        """Test renaming a symbol used across multiple files."""
        # Create multiple files
        file1 = tmp_path / "module1.py"
        file1.write_text("""
def process_data(data):
    return data * 2
""")

        file2 = tmp_path / "module2.py"
        file2.write_text("""
from module1 import process_data

def main():
    result = process_data(10)
    return result
""")

        with patch("ast_grep_mcp.features.refactoring.renamer.run_ast_grep") as mock_run:
            # ast-grep returns absolute paths
            mock_run.return_value = Mock(
                returncode=0,
                stdout=(
                    f'[{{"file": "{str(file1)}", "range": {{"start": {{"line": 1, "column": 4}}}},'
                    f' "lines": "def process_data(data):"}},'
                    f'{{"file": "{str(file2)}", "range": {{"start": {{"line": 1, "column": 19}}}},'
                    f' "lines": "from module1 import process_data"}},'
                    f'{{"file": "{str(file2)}", "range": {{"start": {{"line": 4, "column": 13}}}},'
                    f' "lines": "result = process_data(10)"}}]'
                ),
            )

            with patch("ast_grep_mcp.features.refactoring.rename_coordinator.create_backup") as mock_backup:
                mock_backup.return_value = "backup-multi"

                result = python_coordinator.rename_symbol(
                    project_folder=str(tmp_path),
                    old_name="process_data",
                    new_name="transform_data",
                    dry_run=False,
                )

                assert result.success
                assert result.references_updated == 3
                assert len(result.files_modified) == 2
                assert result.backup_id == "backup-multi"

    def test_rollback_on_failure(self, python_coordinator, tmp_path):
        """Test that rename rolls back on failure."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def foo():
    x = 1
    return x
""")

        with patch("ast_grep_mcp.features.refactoring.renamer.run_ast_grep") as mock_run:
            # ast-grep returns absolute paths
            mock_run.return_value = Mock(
                returncode=0, stdout=f'[{{"file": "{str(test_file)}", "range": {{"start": {{"line": 2, "column": 4}}}}, "lines": "x = 1"}}]'
            )

            with patch("ast_grep_mcp.features.refactoring.rename_coordinator.create_backup") as mock_backup:
                mock_backup.return_value = "backup-fail"

                with patch("ast_grep_mcp.features.refactoring.rename_coordinator.restore_backup") as mock_restore:
                    with patch.object(python_coordinator, "_rename_in_file") as mock_rename:
                        mock_rename.side_effect = Exception("File write failed")

                        result = python_coordinator.rename_symbol(
                            project_folder=str(tmp_path),
                            old_name="x",
                            new_name="y",
                            dry_run=False,
                        )

                        assert not result.success
                        assert "File write failed" in result.error
                        # Verify rollback was called
                        mock_restore.assert_called_once_with("backup-fail", str(tmp_path))
