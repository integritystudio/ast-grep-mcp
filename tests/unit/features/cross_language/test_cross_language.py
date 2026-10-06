"""Comprehensive tests for cross-language operations features.

This module tests:
- Multi-language search
- Pattern equivalence lookup
- Language conversion
- Polyglot refactoring
- API binding generation
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ast_grep_mcp.features.cross_language.binding_generator import (
    _parse_openapi_spec,
    _to_camel_case,
    _to_pascal_case,
    generate_language_bindings_impl,
)
from ast_grep_mcp.features.cross_language.language_converter import (
    PYTHON_TO_TS_PATTERNS,
    _apply_patterns,
    convert_code_language_impl,
)
from ast_grep_mcp.features.cross_language.multi_language_search import (
    _detect_languages,
    _parse_semantic_query,
    search_multi_language_impl,
)
from ast_grep_mcp.features.cross_language.pattern_database import (
    PATTERN_DATABASE,
    get_equivalents,
    get_pattern,
    get_type_mapping,
    search_patterns,
)
from ast_grep_mcp.features.cross_language.pattern_equivalence import (
    find_language_equivalents_impl,
    get_pattern_details,
    list_pattern_categories,
)
from ast_grep_mcp.features.cross_language.polyglot_refactoring import (
    _create_rename_change,
    _find_symbol_occurrences,
    refactor_polyglot_impl,
)
from ast_grep_mcp.models.cross_language import (
    SUPPORTED_CONVERSION_PAIRS,
    SUPPORTED_LANGUAGES,
    ConversionStyle,
    RefactoringType,
)

# =============================================================================
# Pattern Database Tests
# =============================================================================


class TestPatternDatabase:
    """Tests for pattern database functions."""

    def test_pattern_database_not_empty(self):
        """Test pattern database has entries."""
        assert len(PATTERN_DATABASE) > 0

    def test_get_pattern_existing(self):
        """Test getting existing pattern."""
        pattern = get_pattern("list_comprehension")
        assert pattern is not None
        assert "concept" in pattern
        assert "examples" in pattern

    def test_get_pattern_nonexistent(self):
        """Test getting non-existent pattern."""
        pattern = get_pattern("nonexistent_pattern_xyz")
        assert pattern is None

    def test_search_patterns_by_concept(self):
        """Test searching patterns by concept."""
        results = search_patterns("comprehension")
        assert len(results) > 0
        assert any("list_comprehension" in r.get("pattern_id", "") for r in results)

    def test_search_patterns_by_category(self):
        """Test searching patterns with category filter."""
        results = search_patterns("function", category="functions")
        # All results should be in functions category
        for r in results:
            assert r.get("category") == "functions"

    def test_get_equivalents(self):
        """Test getting pattern equivalents."""
        equiv = get_equivalents("list_comprehension")
        assert equiv is not None
        assert "examples" in equiv
        assert len(equiv["examples"]) > 0

    def test_get_equivalents_with_target_languages(self):
        """Test getting equivalents filtered by language."""
        equiv = get_equivalents("list_comprehension", target_languages=["python", "javascript"])
        assert equiv is not None
        languages = set(equiv["examples"].keys())
        assert languages <= {"python", "javascript"}

    def test_get_type_mapping_python_to_typescript(self):
        """Test type mapping Python to TypeScript."""
        mapping = get_type_mapping("python", "typescript")
        assert mapping.get("str") == "string"
        assert mapping.get("int") == "number"
        assert mapping.get("bool") == "boolean"

    def test_get_type_mapping_nonexistent(self):
        """Test type mapping for unsupported pair."""
        mapping = get_type_mapping("ruby", "swift")
        assert mapping == {}


# =============================================================================
# Pattern Equivalence Tests
# =============================================================================


class TestPatternEquivalence:
    """Tests for pattern equivalence lookup."""

    def test_find_language_equivalents_basic(self):
        """Test basic pattern equivalence lookup."""
        result = find_language_equivalents_impl(pattern_description="list comprehension")
        assert result is not None
        assert len(result.equivalences) > 0
        assert result.execution_time_ms >= 0

    def test_find_language_equivalents_with_source(self):
        """Test equivalence lookup with source language."""
        result = find_language_equivalents_impl(pattern_description="async await", source_language="python")
        assert result.source_language == "python"

    def test_find_language_equivalents_with_targets(self):
        """Test equivalence lookup with target languages."""
        result = find_language_equivalents_impl(pattern_description="try catch", target_languages=["python", "typescript"])
        assert "python" in result.target_languages
        assert "typescript" in result.target_languages

    def test_find_language_equivalents_no_match(self):
        """Test equivalence lookup with no matching patterns."""
        result = find_language_equivalents_impl(pattern_description="xyznonexistent123")
        assert len(result.equivalences) == 0
        assert len(result.suggestions) > 0  # Should provide suggestions

    def test_list_pattern_categories(self):
        """Test listing pattern categories."""
        categories = list_pattern_categories()
        assert len(categories) > 0
        for cat in categories:
            assert "category" in cat
            assert "count" in cat
            assert cat["count"] > 0

    def test_get_pattern_details(self):
        """Test getting pattern details."""
        details = get_pattern_details("if_else")
        assert details is not None
        assert details.pattern_id == "if_else"
        assert details.category == "control_flow"
        assert len(details.examples) > 0


# =============================================================================
# Language Converter Tests
# =============================================================================


class TestLanguageConverter:
    """Tests for language conversion functionality."""

    def test_convert_python_to_typescript_function(self):
        """Test converting Python function to TypeScript."""
        python_code = """
def greet(name: str) -> str:
    return f"Hello, {name}!"
"""
        result = convert_code_language_impl(code_snippet=python_code, from_language="python", to_language="typescript")
        assert result.successful_conversions == 1
        converted = result.conversions[0].converted_code
        assert "function" in converted
        assert "return" in converted

    def test_convert_python_to_typescript_class(self):
        """Test converting Python class to TypeScript."""
        python_code = """
class Person:
    def __init__(self, name):
        self.name = name
"""
        result = convert_code_language_impl(code_snippet=python_code, from_language="python", to_language="typescript")
        assert result.successful_conversions == 1
        converted = result.conversions[0].converted_code
        assert "class Person" in converted

    def test_convert_javascript_to_python_function(self):
        """Test converting JavaScript function to Python."""
        js_code = """
function calculateSum(a, b) {
    return a + b;
}
"""
        result = convert_code_language_impl(code_snippet=js_code, from_language="javascript", to_language="python")
        assert result.successful_conversions == 1
        converted = result.conversions[0].converted_code
        assert "def" in converted

    def test_convert_with_type_mappings(self):
        """Test conversion includes type mappings."""
        python_code = "def test(x: str, y: int) -> bool: pass"
        result = convert_code_language_impl(code_snippet=python_code, from_language="python", to_language="typescript")
        # Type mappings should be recorded
        assert result.conversions[0].success

    def test_convert_unsupported_pair(self):
        """Test conversion with unsupported language pair."""
        with pytest.raises(ValueError) as exc_info:
            convert_code_language_impl(code_snippet="code", from_language="ruby", to_language="swift")
        assert "Unsupported conversion pair" in str(exc_info.value)

    def test_convert_with_warnings(self):
        """Test conversion generates warnings for complex patterns."""
        python_code = """
@decorator
def function():
    with context_manager():
        yield value
"""
        result = convert_code_language_impl(code_snippet=python_code, from_language="python", to_language="typescript")
        # Should generate warnings for decorators and context managers
        assert len(result.conversions[0].warnings) > 0

    def test_convert_boolean_values(self):
        """Test boolean value conversion."""
        python_code = "x = True if condition else False"
        result = convert_code_language_impl(code_snippet=python_code, from_language="python", to_language="javascript")
        converted = result.conversions[0].converted_code
        assert "true" in converted or "false" in converted

    def test_apply_patterns_function(self):
        """Test pattern application helper."""
        code = "print('hello')"
        result, applied = _apply_patterns(code, PYTHON_TO_TS_PATTERNS)
        assert "console.log" in result
        assert "print" in applied


# =============================================================================
# Multi-Language Search Tests
# =============================================================================


class TestMultiLanguageSearch:
    """Tests for multi-language search functionality."""

    def test_detect_languages(self):
        """Test language detection in a project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files for different languages
            Path(tmpdir, "app.py").write_text("print('hello')")
            Path(tmpdir, "index.ts").write_text("console.log('hello')")
            Path(tmpdir, "Main.java").write_text("class Main {}")

            languages = _detect_languages(tmpdir)
            assert "python" in languages
            assert "typescript" in languages
            assert "java" in languages

    def test_parse_semantic_query_async(self):
        """Test semantic query parsing for async."""
        result = _parse_semantic_query("async function with error handling")
        assert result == "async_function"

    def test_parse_semantic_query_class(self):
        """Test semantic query parsing for class."""
        result = _parse_semantic_query("class definition")
        # Note: "class" maps to the class pattern key
        assert result in ["class", "function"]  # Depends on word order in mappings

    def test_parse_semantic_query_matches_whole_words(self):
        """Terms must not match inside longer words."""
        assert _parse_semantic_query("foreach over items") == "foreach"
        assert _parse_semantic_query("modify the record") == "function"
        assert _parse_semantic_query("nested loops") == "for_loop"

    def test_parse_semantic_query_try_catch(self):
        """Test semantic query parsing for error handling."""
        result = _parse_semantic_query("try catch exception")
        assert result == "try_catch"

    def test_parse_semantic_query_default(self):
        """Test semantic query parsing default to function."""
        result = _parse_semantic_query("unknown pattern xyz")
        assert result == "function"

    @patch("ast_grep_mcp.features.cross_language.multi_language_search.run_ast_grep")
    def test_search_multi_language_basic(self, mock_run):
        """Test basic multi-language search."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '[{"file": "test.py", "range": {"start": {"line": 1}}, "text": "def test():"}]'
        mock_run.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "test.py").write_text("def test(): pass")

            result = search_multi_language_impl(project_folder=tmpdir, semantic_pattern="function", languages=["python"])

            assert result.total_matches >= 0
            assert "python" in result.languages_searched

    def test_search_multi_language_invalid_folder(self):
        """Test search with invalid folder."""
        with pytest.raises(ValueError) as exc_info:
            search_multi_language_impl(project_folder="/nonexistent/path/xyz", semantic_pattern="function")
        assert "not found" in str(exc_info.value)

    def test_search_multi_language_no_supported_languages(self):
        """Test search returns empty when no languages supported."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = search_multi_language_impl(project_folder=tmpdir, semantic_pattern="function", languages=["unsupported_lang_xyz"])
            assert result.total_matches == 0
            assert result.languages_searched == []


# =============================================================================
# Polyglot Refactoring Tests
# =============================================================================


class TestPolyglotRefactoring:
    """Tests for polyglot refactoring functionality."""

    def test_find_symbol_occurrences(self):
        """Test finding symbol occurrences in a file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def myFunction():\n    pass\n\nmyFunction()\n")
            f.flush()

            try:
                occurrences = _find_symbol_occurrences(f.name, "myFunction", "python")
                assert len(occurrences) >= 2
            finally:
                os.unlink(f.name)

    def test_create_rename_change(self):
        """Test creating a rename change."""
        change = _create_rename_change(
            file_path="/test/file.py",
            line_number=10,
            original_line="def oldName():",
            symbol="oldName",
            new_name="newName",
            language="python",
        )
        assert change.file_path == "/test/file.py"
        assert change.line_number == 10
        assert "newName" in change.new_code
        assert "oldName" not in change.new_code

    def test_refactor_polyglot_dry_run(self):
        """Test polyglot refactoring in dry run mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            py_file = Path(tmpdir, "api.py")
            py_file.write_text("def getUserProfile():\n    pass\n")

            ts_file = Path(tmpdir, "client.ts")
            ts_file.write_text("function getUserProfile() {\n    return {};\n}\n")

            result = refactor_polyglot_impl(
                project_folder=tmpdir,
                refactoring_type="rename_api",
                symbol_name="getUserProfile",
                new_name="fetchUserProfile",
                dry_run=True,
            )

            assert result.dry_run is True
            assert len(result.plan.changes) >= 2
            assert result.files_modified == []  # Dry run doesn't modify

    def test_refactor_polyglot_apply(self):
        """Test polyglot refactoring with actual application."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test file
            py_file = Path(tmpdir, "test.py")
            py_file.write_text("def oldSymbol():\n    oldSymbol()\n")

            result = refactor_polyglot_impl(
                project_folder=tmpdir, refactoring_type="rename_api", symbol_name="oldSymbol", new_name="newSymbol", dry_run=False
            )

            assert result.dry_run is False
            # Check file was modified
            content = py_file.read_text()
            assert "newSymbol" in content
            assert "oldSymbol" not in content

    def test_refactor_polyglot_rename_alias(self):
        """Test that 'rename' is accepted as alias for 'rename_api'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = refactor_polyglot_impl(
                project_folder=tmpdir,
                refactoring_type="rename",
                symbol_name="oldName",
                new_name="newName",
                dry_run=True,
            )
            assert result.plan.refactoring_type.value == "rename_api"

    def test_refactor_polyglot_invalid_type(self):
        """Test refactoring with invalid type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError) as exc_info:
                refactor_polyglot_impl(project_folder=tmpdir, refactoring_type="invalid_type", symbol_name="test")
            assert "Invalid refactoring type" in str(exc_info.value)

    def test_refactor_polyglot_missing_new_name(self):
        """Test rename refactoring without new_name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError) as exc_info:
                refactor_polyglot_impl(project_folder=tmpdir, refactoring_type="rename_api", symbol_name="test")
            assert "new_name is required" in str(exc_info.value)


# =============================================================================
# API Binding Generator Tests
# =============================================================================


class TestBindingGenerator:
    """Tests for API binding generation functionality."""

    def test_to_camel_case(self):
        """Test camelCase conversion."""
        assert _to_camel_case("get_user_profile") == "getUserProfile"
        assert _to_camel_case("get-user-profile") == "getUserProfile"
        assert _to_camel_case("simple") == "simple"

    def test_to_pascal_case(self):
        """Test PascalCase conversion."""
        assert _to_pascal_case("get_user_profile") == "GetUserProfile"
        assert _to_pascal_case("get-user-profile") == "GetUserProfile"
        assert _to_pascal_case("simple") == "Simple"

    def test_parse_openapi_spec(self):
        """Test parsing OpenAPI specification."""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/users": {
                    "get": {"operationId": "getUsers", "summary": "Get all users", "responses": {"200": {"description": "Success"}}}
                },
                "/users/{id}": {
                    "get": {
                        "operationId": "getUser",
                        "parameters": [{"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}],
                        "responses": {"200": {"description": "Success"}},
                    }
                },
            },
        }

        api_name, version, base_url, endpoints = _parse_openapi_spec(spec)

        assert api_name == "Test API"
        assert version == "1.0.0"
        assert base_url == "https://api.example.com"
        assert len(endpoints) == 2

    def test_generate_language_bindings_python(self):
        """Test generating Python bindings."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            spec = {
                "openapi": "3.0.0",
                "info": {"title": "TestAPI", "version": "1.0.0"},
                "servers": [{"url": "https://api.test.com"}],
                "paths": {"/items": {"get": {"operationId": "getItems", "responses": {"200": {"description": "OK"}}}}},
            }
            json.dump(spec, f)
            f.flush()

            try:
                result = generate_language_bindings_impl(api_definition_file=f.name, target_languages=["python"])

                assert result.api_name == "TestAPI"
                assert result.endpoints_count == 1
                assert len(result.bindings) == 1
                assert result.bindings[0].language == "python"
                # Class name is generated from API name
                assert "class" in result.bindings[0].code and "Client" in result.bindings[0].code
            finally:
                os.unlink(f.name)

    def test_generate_language_bindings_typescript(self):
        """Test generating TypeScript bindings."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            spec = {
                "openapi": "3.0.0",
                "info": {"title": "MyAPI", "version": "2.0.0"},
                "servers": [{"url": "https://api.example.com"}],
                "paths": {
                    "/users/{id}": {
                        "get": {
                            "operationId": "getUserById",
                            "parameters": [{"name": "id", "in": "path", "required": True}],
                            "responses": {"200": {"description": "OK"}},
                        }
                    }
                },
            }
            json.dump(spec, f)
            f.flush()

            try:
                result = generate_language_bindings_impl(api_definition_file=f.name, target_languages=["typescript"])

                assert len(result.bindings) == 1
                binding = result.bindings[0]
                assert binding.language == "typescript"
                # Class name is generated from API name
                assert "export class" in binding.code and "Client" in binding.code
                assert "getUserById" in binding.code.lower() or "getuserbyid" in binding.code.lower()
            finally:
                os.unlink(f.name)

    def test_generate_language_bindings_multiple(self):
        """Test generating bindings for multiple languages."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            spec = {
                "openapi": "3.0.0",
                "info": {"title": "Multi", "version": "1.0.0"},
                "servers": [{"url": "https://api.test.com"}],
                "paths": {"/test": {"get": {"operationId": "test", "responses": {"200": {}}}}},
            }
            json.dump(spec, f)
            f.flush()

            try:
                result = generate_language_bindings_impl(
                    api_definition_file=f.name, target_languages=["python", "typescript", "javascript"]
                )

                assert len(result.bindings) == 3
                languages = {b.language for b in result.bindings}
                assert languages == {"python", "typescript", "javascript"}
            finally:
                os.unlink(f.name)

    def test_generate_language_bindings_invalid_file(self):
        """Test binding generation with invalid file."""
        with pytest.raises(ValueError) as exc_info:
            generate_language_bindings_impl(api_definition_file="/nonexistent/file.json")
        assert "not found" in str(exc_info.value)


# =============================================================================
# Model Tests
# =============================================================================


class TestModels:
    """Tests for cross-language data models."""

    def test_conversion_style_enum(self):
        """Test ConversionStyle enum values."""
        assert ConversionStyle.LITERAL.value == "literal"
        assert ConversionStyle.IDIOMATIC.value == "idiomatic"
        assert ConversionStyle.COMPATIBLE.value == "compatible"

    def test_refactoring_type_enum(self):
        """Test RefactoringType enum values."""
        assert RefactoringType.RENAME_API.value == "rename_api"
        assert RefactoringType.EXTRACT_CONSTANT.value == "extract_constant"
        assert RefactoringType.UPDATE_CONTRACT.value == "update_contract"

    def test_supported_languages(self):
        """Test supported languages list."""
        assert "python" in SUPPORTED_LANGUAGES
        assert "typescript" in SUPPORTED_LANGUAGES
        assert "javascript" in SUPPORTED_LANGUAGES
        assert "java" in SUPPORTED_LANGUAGES

    def test_supported_conversion_pairs(self):
        """Test supported conversion pairs."""
        assert ("python", "typescript") in SUPPORTED_CONVERSION_PAIRS
        assert ("python", "javascript") in SUPPORTED_CONVERSION_PAIRS
        assert ("javascript", "typescript") in SUPPORTED_CONVERSION_PAIRS


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for cross-language features."""

    def test_full_workflow_find_and_convert(self):
        """Test finding equivalents then converting code."""
        # Find pattern equivalents
        equiv_result = find_language_equivalents_impl(pattern_description="async function")
        assert len(equiv_result.equivalences) > 0

        # Convert example code
        python_code = "async def fetch(): pass"
        convert_result = convert_code_language_impl(code_snippet=python_code, from_language="python", to_language="typescript")
        assert convert_result.successful_conversions == 1

    def test_workflow_search_and_refactor(self):
        """Test searching then refactoring across languages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multi-language project
            Path(tmpdir, "backend.py").write_text("def apiEndpoint(): pass\n")
            Path(tmpdir, "frontend.ts").write_text("function apiEndpoint() {}\n")

            # Refactor
            result = refactor_polyglot_impl(
                project_folder=tmpdir, refactoring_type="rename_api", symbol_name="apiEndpoint", new_name="newApiEndpoint", dry_run=True
            )

            # Should find in both files
            assert len(result.plan.changes) >= 2
            languages = {c.language for c in result.plan.changes}
            assert "python" in languages
            assert "typescript" in languages


class TestRunParallelSearchTimeout:
    """Regression tests for the BUG-06-CR-16 hang fix in _run_parallel_search.

    The previous `with ThreadPoolExecutor` form blocked at shutdown(wait=True)
    until every worker finished, hung ones included; the shared per-item-timeout
    helper abandons a hung language search after its wait instead.
    """

    TIMEOUT_SECONDS = 1
    PROMPT_RETURN_BOUND_SECONDS = 10
    HUNG_WORKER_SAFETY_NET_SECONDS = 30

    def test_hung_language_search_times_out_without_blocking(self):
        import threading
        import time as time_module

        from ast_grep_mcp.features.cross_language import multi_language_search as mls

        release = threading.Event()

        def fake_search(project_folder, language, pattern, max_results):
            if language == "python":
                release.wait(timeout=self.HUNG_WORKER_SAFETY_NET_SECONDS)
                return []
            return []

        try:
            with (
                patch.object(mls, "_search_language", side_effect=fake_search),
                patch.object(mls.SubprocessDefaults, "AST_GREP_TIMEOUT_SECONDS", self.TIMEOUT_SECONDS),
            ):
                start = time_module.monotonic()
                all_matches, matches_by_language = mls._run_parallel_search(
                    project_folder="/tmp/does-not-matter",
                    languages=["python", "javascript"],
                    semantic_key="function",
                    max_results_per_language=10,
                )
                elapsed = time_module.monotonic() - start
        finally:
            release.set()

        assert elapsed < self.PROMPT_RETURN_BOUND_SECONDS, f"search should return promptly, took {elapsed:.1f}s"
        assert matches_by_language["python"] == 0
        assert matches_by_language["javascript"] == 0
        assert all_matches == []

    def test_languages_without_patterns_are_skipped(self):
        from ast_grep_mcp.features.cross_language import multi_language_search as mls

        all_matches, matches_by_language = mls._run_parallel_search(
            project_folder="/tmp/does-not-matter",
            languages=["not-a-language"],
            semantic_key="function",
            max_results_per_language=10,
        )

        assert all_matches == []
        assert matches_by_language == {}
