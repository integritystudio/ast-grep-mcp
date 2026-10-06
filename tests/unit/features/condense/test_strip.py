"""Tests for dead code stripping."""

import ast

from ast_grep_mcp.features.condense.strip import (
    _strip_js_ts,
    _strip_python,
    strip_dead_code,
)


class TestStripDeadCode:
    def test_returns_tuple(self):
        result = strip_dead_code("console.log('hi');", "javascript")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_unknown_language_returns_source(self):
        source = "some_debug_call()"
        stripped, removed = strip_dead_code(source, "lua")
        assert stripped == source
        assert removed == 0


class TestStripJsTs:
    def test_removes_console_log(self):
        lines = ["const x = 1;", "console.log(x);", "return x;"]
        kept, removed = _strip_js_ts(lines)
        assert removed == 1
        assert all("console.log" not in line for line in kept)

    def test_removes_console_debug(self):
        lines = ["  console.debug('msg');"]
        kept, removed = _strip_js_ts(lines)
        assert removed == 1
        assert len(kept) == 0

    def test_removes_debugger(self):
        lines = ["function foo() {", "  debugger;", "  return 1;", "}"]
        kept, removed = _strip_js_ts(lines)
        assert removed == 1
        assert "debugger;" not in " ".join(kept)

    def test_preserves_real_code(self):
        lines = ["const x = 1;", "const y = x + 2;", "return y;"]
        kept, removed = _strip_js_ts(lines)
        assert removed == 0
        assert len(kept) == 3

    def test_console_log_in_string_not_removed(self):
        # console.log inside a string literal should not be stripped (not a statement)
        lines = ['const msg = "use console.log for debugging";']
        kept, removed = _strip_js_ts(lines)
        assert removed == 0

    def test_multiple_debug_statements(self):
        lines = [
            "console.log('a');",
            "const x = 1;",
            "console.warn('b');",
            "console.error('c');",
        ]
        kept, removed = _strip_js_ts(lines)
        assert removed == 3
        assert len(kept) == 1


class TestStripPython:
    def test_removes_print(self):
        lines = ["x = 1", "print(x)", "return x"]
        kept, removed = _strip_python(lines)
        assert removed == 1
        assert all("print" not in line for line in kept)

    def test_removes_breakpoint(self):
        lines = ["def foo():", "    breakpoint()", "    return 1"]
        kept, removed = _strip_python(lines)
        assert removed == 1

    def test_removes_pdb_set_trace(self):
        lines = ["  pdb.set_trace()"]
        kept, removed = _strip_python(lines)
        assert removed == 1

    def test_removes_import_pdb(self):
        lines = ["import pdb", "x = 1"]
        kept, removed = _strip_python(lines)
        assert removed == 1
        assert "import pdb" not in " ".join(kept)

    def test_preserves_real_code(self):
        lines = ["def foo():", "    x = 1", "    return x"]
        kept, removed = _strip_python(lines)
        assert removed == 0
        assert len(kept) == 3

    def test_print_in_assignment_not_removed(self):
        # Only bare print() calls as statements should be stripped
        # But the current regex matches "print(x)" standalone – confirm behavior
        lines = ["result = print_to_file(x)"]
        kept, removed = _strip_python(lines)
        assert removed == 0


class TestStripKeepsBlocksValid:
    """Removing a block's only statement must not break syntax or control flow."""

    def test_python_sole_print_in_block_becomes_pass(self) -> None:
        source = "def f(x):\n    if x:\n        print(x)\n    else:\n        return 1\n"
        stripped, _ = strip_dead_code(source, "python")
        ast.parse(stripped)
        assert "        pass" in stripped


    def test_python_block_of_only_debug_lines_at_eof(self) -> None:
        source = "for i in range(3):\n    print(i)\n    breakpoint()\n"
        stripped, _ = strip_dead_code(source, "python")
        ast.parse(stripped)


    def test_python_print_with_siblings_is_removed(self) -> None:
        source = "def f(x):\n    print(x)\n    return x\n"
        stripped, removed = strip_dead_code(source, "python")
        assert stripped == "def f(x):\n    return x"
        assert removed == 1


    def test_js_braceless_if_body_is_kept_as_empty_statement(self) -> None:
        source = "function f(err, v) {\n  if (err)\n    console.error(err);\n  return v;\n}"
        stripped, _ = strip_dead_code(source, "javascript")
        assert "  if (err)\n    ;\n  return v;" in stripped


    def test_js_console_in_braced_block_is_removed(self) -> None:
        source = "if (err) {\n  console.error(err);\n}\nreturn v;"
        stripped, removed = strip_dead_code(source, "javascript")
        assert stripped == "if (err) {\n}\nreturn v;"
        assert removed == 1
