"""Tests for function extraction refactoring."""

import pytest

from ast_grep_mcp.features.refactoring.analyzer import CodeSelectionAnalyzer
from ast_grep_mcp.features.refactoring.extractor import FunctionExtractor
from ast_grep_mcp.features.refactoring.tools import extract_function_tool
from ast_grep_mcp.models.refactoring import VariableType


class TestCodeSelectionAnalyzer:
    """Tests for CodeSelectionAnalyzer."""

    def test_analyze_python_simple_selection(self, tmp_path):
        """Test analyzing a simple Python code selection."""

        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def process_user(user):
    name = user['name']
    email = user['email']
    # Extract this block
    normalized_email = email.lower().strip()
    domain = normalized_email.split('@')[1]
    # End extraction
    return {'name': name, 'email': normalized_email, 'domain': domain}
""")

        analyzer = CodeSelectionAnalyzer("python")
        selection = analyzer.analyze_selection(
            file_path=str(test_file),
            start_line=5,
            end_line=6,
            project_folder=str(tmp_path),
        )

        assert selection.start_line == 5
        assert selection.end_line == 6
        assert selection.language == "python"

        # Check that email is detected as a parameter
        param_vars = selection.get_variables_by_type(VariableType.PARAMETER)
        assert any(v.name == "email" for v in param_vars)

        # Check that normalized_email is detected (assigned in selection)
        assert "normalized_email" in [v.name for v in selection.variables]

        # Note: domain is assigned but may be classified as LOCAL
        # What matters is that the function extraction works correctly

    def test_detect_indentation(self):
        """Test indentation detection."""
        analyzer = CodeSelectionAnalyzer("python")

        # Test with spaces
        lines = ["    def foo():", "        pass"]
        indent = analyzer._detect_indentation(lines)
        assert indent == "    "

        # Test with tabs
        lines = ["\tdef foo():", "\t\tpass"]
        indent = analyzer._detect_indentation(lines)
        assert indent == "\t"

    def test_has_early_returns_python(self):
        """Test detection of early returns in Python."""
        analyzer = CodeSelectionAnalyzer("python")

        content_with_return = "if x > 5:\n    return True"
        assert analyzer._has_early_returns(content_with_return)

        content_without_return = "x = 5\ny = 10"
        assert not analyzer._has_early_returns(content_without_return)

    def test_has_exception_handling_python(self):
        """Test detection of exception handling in Python."""
        analyzer = CodeSelectionAnalyzer("python")

        content_with_try = "try:\n    x = 1\nexcept:\n    pass"
        assert analyzer._has_exception_handling(content_with_try)

        content_with_raise = "if error:\n    raise ValueError('Error')"
        assert analyzer._has_exception_handling(content_with_raise)

        content_without = "x = 5\ny = 10"
        assert not analyzer._has_exception_handling(content_without)


class TestFunctionExtractor:
    """Tests for FunctionExtractor."""

    def test_generate_function_name(self):
        """Test automatic function name generation."""
        from ast_grep_mcp.models.refactoring import CodeSelection

        extractor = FunctionExtractor("python")

        # Test with validate in content
        selection = CodeSelection(
            file_path="test.py",
            start_line=1,
            end_line=2,
            language="python",
            content="if not email:\n    validate_error()",
        )
        name = extractor._generate_function_name(selection)
        assert "validate" in name

        # Test with process in content
        selection.content = "process_data(x)\ny = transform(x)"
        name = extractor._generate_function_name(selection)
        assert "process" in name or "transform" in name

    def test_generate_signature_python(self):
        """Test Python function signature generation."""
        from ast_grep_mcp.models.refactoring import CodeSelection

        extractor = FunctionExtractor("python")

        selection = CodeSelection(
            file_path="test.py",
            start_line=1,
            end_line=2,
            language="python",
            content="result = x + y",
            parameters_needed=["x", "y"],
            return_values=["result"],
        )

        signature = extractor._generate_signature(selection, "add_numbers")

        assert signature.name == "add_numbers"
        assert len(signature.parameters) == 2
        assert signature.parameters[0]["name"] == "x"
        assert signature.parameters[1]["name"] == "y"

    def test_generate_return_statement_python(self):
        """Test return statement generation for Python."""
        extractor = FunctionExtractor("python")

        # Single return value
        stmt = extractor._generate_return_statement(["result"])
        assert stmt == "return result"

        # Multiple return values
        stmt = extractor._generate_return_statement(["x", "y", "z"])
        assert stmt == "return x, y, z"

        # No return values
        stmt = extractor._generate_return_statement([])
        assert stmt == ""

    def test_generate_call_site_python(self):
        """Test call site generation for Python."""
        from ast_grep_mcp.models.refactoring import (
            CodeSelection,
            FunctionSignature,
        )

        extractor = FunctionExtractor("python")

        signature = FunctionSignature(
            name="calculate",
            parameters=[{"name": "x"}, {"name": "y"}],
        )

        # Single return value
        selection = CodeSelection(
            file_path="test.py",
            start_line=1,
            end_line=2,
            language="python",
            content="",
            indentation="    ",
            return_values=["result"],
        )

        call = extractor._generate_call_site(selection, signature)
        assert "result = calculate(x, y)" in call
        assert call.startswith("    ")  # Check indentation

        # Multiple return values
        selection.return_values = ["x", "y"]
        call = extractor._generate_call_site(selection, signature)
        assert "x, y = calculate(x, y)" in call


class TestExtractFunctionTool:
    """Integration tests for extract_function_tool."""

    def test_extract_function_dry_run(self, tmp_path):
        """Test extract function in dry-run mode."""
        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def calculate_total(items):
    total = 0
    for item in items:
        price = item['price']
        quantity = item['quantity']
        subtotal = price * quantity
        total += subtotal
    return total
""")

        result = extract_function_tool(
            project_folder=str(tmp_path),
            file_path=str(test_file),
            start_line=5,
            end_line=7,
            language="python",
            function_name="calculate_item_subtotal",
            dry_run=True,
        )

        assert result["success"]
        assert result["function_name"] == "calculate_item_subtotal"

        # item is the only parameter (price and quantity are assigned IN the selection)
        assert "item" in result["parameters"]

        # price and quantity are LOCAL (created and used within selection)
        # They don't need to be parameters

        # subtotal is returned (created in selection, needed outside)
        assert "subtotal" in result["return_values"]
        assert result["diff_preview"] is not None
        assert result["backup_id"] is None  # No backup in dry-run

    def test_extract_function_with_no_returns(self, tmp_path):
        """Test extracting function that doesn't return values."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def process():
    x = 5
    console.log(x)
    log_info("Processing")
""")

        result = extract_function_tool(
            project_folder=str(tmp_path),
            file_path=str(test_file),
            start_line=3,
            end_line=4,
            language="python",
            dry_run=True,
        )

        assert result["success"]
        # Should have no return values
        assert len(result["return_values"]) == 0

    def test_extract_function_apply(self, tmp_path):
        """Test applying function extraction (not dry-run)."""
        test_file = tmp_path / "test.py"
        original_content = """
def process_user(user):
    name = user['name']
    email = user['email']
    normalized_email = email.lower().strip()
    domain = normalized_email.split('@')[1]
    return {'name': name, 'email': normalized_email, 'domain': domain}
"""
        test_file.write_text(original_content)

        result = extract_function_tool(
            project_folder=str(tmp_path),
            file_path=str(test_file),
            start_line=5,
            end_line=6,
            language="python",
            function_name="normalize_email",
            dry_run=False,
        )

        assert result["success"]
        assert result["backup_id"] is not None

        # Verify file was modified
        modified_content = test_file.read_text()
        assert "def normalize_email" in modified_content
        assert "normalize_email(email)" in modified_content


    @pytest.mark.parametrize("location", ["before", "after"])
    def test_extracted_function_is_module_level_and_parses(self, tmp_path, location):
        """The function goes outside the enclosing def, dedented, and the call replaces the selection."""
        import ast

        test_file = tmp_path / "test.py"
        test_file.write_text(
            "\ndef process_user(user):\n"
            "    email = user['email']\n"
            "    normalized_email = email.lower().strip()\n"
            "    domain = normalized_email.split('@')[1]\n"
            "    return {'email': normalized_email, 'domain': domain}\n"
            "\n\ndef other():\n    return 1\n"
        )

        result = extract_function_tool(
            project_folder=str(tmp_path),
            file_path=str(test_file),
            start_line=4,
            end_line=5,
            language="python",
            function_name="normalize_email",
            extract_location=location,
            dry_run=False,
        )

        assert result["success"]
        tree = ast.parse(test_file.read_text())
        top_level = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
        expected = ["normalize_email", "process_user", "other"] if location == "before" else ["process_user", "normalize_email", "other"]
        assert top_level == expected
        process_user = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "process_user")
        assert "normalize_email(email)" in ast.unparse(process_user)


class TestJavaScriptExtraction:
    """Tests for JavaScript/TypeScript extraction."""

    def test_analyze_javascript_variables(self, tmp_path):
        """Test analyzing JavaScript variables."""
        test_file = tmp_path / "test.js"
        test_file.write_text("""
function processUser(user) {
    const name = user.name;
    const email = user.email;
    // Extract this
    const normalized = email.toLowerCase().trim();
    const domain = normalized.split('@')[1];
    // End
    return { name, email: normalized, domain };
}
""")

        analyzer = CodeSelectionAnalyzer("javascript")
        selection = analyzer.analyze_selection(
            file_path=str(test_file),
            start_line=5,
            end_line=6,
            project_folder=str(tmp_path),
        )

        assert selection.language == "javascript"

        # Check email is detected as parameter
        param_vars = selection.get_variables_by_type(VariableType.PARAMETER)
        assert any(v.name == "email" for v in param_vars)


# Fixtures for common test data
@pytest.fixture
def sample_python_code():
    """Sample Python code for testing."""
    return """
def calculate_discount(price, quantity):
    base_total = price * quantity
    if quantity > 10:
        discount_rate = 0.1
    elif quantity > 5:
        discount_rate = 0.05
    else:
        discount_rate = 0
    discount = base_total * discount_rate
    final_total = base_total - discount
    return final_total
"""


@pytest.fixture
def sample_typescript_code():
    """Sample TypeScript code for testing."""
    return """
function processOrder(order: Order): OrderResult {
    const items = order.items;
    let total = 0;
    for (const item of items) {
        const subtotal = item.price * item.quantity;
        total += subtotal;
    }
    const tax = total * 0.1;
    const finalTotal = total + tax;
    return { total, tax, finalTotal };
}
"""


class TestProcessScanLine:
    """Unit tests for FunctionExtractor._process_scan_line and _scan_imports."""

    def setup_method(self):
        self.extractor = FunctionExtractor("python")

    # --- _process_scan_line branch coverage ---

    def test_skip_blank_line(self):
        last, multi, should_break = self.extractor._process_scan_line("", 1, 0, False)
        assert last == 0
        assert multi is False
        assert should_break is False

    def test_skip_comment_line(self):
        last, multi, should_break = self.extractor._process_scan_line("# a comment", 1, 0, False)
        assert last == 0
        assert multi is False
        assert should_break is False

    def test_import_start_single_line(self):
        last, multi, should_break = self.extractor._process_scan_line("import os", 3, 0, False)
        assert last == 3
        assert multi is False
        assert should_break is False

    def test_from_import_single_line(self):
        last, multi, should_break = self.extractor._process_scan_line("from os import path", 5, 0, False)
        assert last == 5
        assert multi is False
        assert should_break is False

    def test_import_start_multiline_opens_paren(self):
        last, multi, should_break = self.extractor._process_scan_line("from ast_grep_mcp.models import (", 4, 0, False)
        assert last == 4
        assert multi is True
        assert should_break is False

    def test_multiline_continuation(self):
        last, multi, should_break = self.extractor._process_scan_line("    Foo,", 5, 3, True)
        assert last == 5
        assert multi is True
        assert should_break is False

    def test_multiline_closing_paren(self):
        last, multi, should_break = self.extractor._process_scan_line("    Bar,\n)", 6, 3, True)
        assert last == 6
        assert multi is False
        assert should_break is False

    def test_post_import_break(self):
        """Non-import line after imports have been seen triggers break."""
        last, multi, should_break = self.extractor._process_scan_line("class Foo:", 8, 5, False)
        assert last == 5
        assert should_break is True

    def test_no_imports_seen_no_break(self):
        """Non-import line with last_import_line==0 does not break."""
        last, multi, should_break = self.extractor._process_scan_line("class Foo:", 2, 0, False)
        assert last == 0
        assert should_break is False

    # --- _scan_imports integration ---

    def test_scan_imports_simple(self):
        lines = ["import os\n", "import sys\n", "\n", "def main():\n", "    pass\n"]
        last_line, in_multiline = self.extractor._scan_imports(lines)
        assert last_line == 2
        assert in_multiline is False

    def test_scan_imports_multiline(self):
        lines = [
            "from x import (\n",
            "    Foo,\n",
            "    Bar,\n",
            ")\n",
            "\n",
            "def main():\n",
        ]
        last_line, in_multiline = self.extractor._scan_imports(lines)
        assert last_line == 4
        assert in_multiline is False

    def test_scan_imports_stacked(self):
        lines = [
            "import os\n",
            "from pathlib import Path\n",
            "import sys\n",
            "class Foo:\n",
        ]
        last_line, in_multiline = self.extractor._scan_imports(lines)
        assert last_line == 3

    def test_scan_imports_no_imports(self):
        lines = ["class Foo:\n", "    pass\n"]
        last_line, _ = self.extractor._scan_imports(lines)
        assert last_line == 0

    def test_scan_imports_only_comments_and_blanks(self):
        lines = ["# header\n", "\n", "# comment\n"]
        last_line, _ = self.extractor._scan_imports(lines)
        assert last_line == 0
