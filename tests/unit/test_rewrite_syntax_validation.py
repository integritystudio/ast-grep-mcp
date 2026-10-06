"""Tests for post-rewrite syntax validation of JavaScript and JSX."""

import pytest

from ast_grep_mcp.features.rewrite.service import validate_rewrites, validate_syntax

VALID_JSX = 'const A = () => <div className="x">{y}</div>;\nexport default A;\n'
BROKEN_JSX = "const A = () => <div>{y</div>;\n"
BROKEN_JS = "function f( {\n"


@pytest.mark.parametrize(("language", "name"), [("jsx", "a.jsx"), ("javascript", "a.js")])
def test_valid_jsx_passes(tmp_path, language: str, name: str) -> None:
    path = tmp_path / name
    path.write_text(VALID_JSX)
    assert validate_syntax(str(path), language)["valid"] is True


def test_broken_jsx_fails(tmp_path) -> None:
    path = tmp_path / "a.jsx"
    path.write_text(BROKEN_JSX)
    assert validate_syntax(str(path), "jsx")["valid"] is False


def test_broken_plain_js_still_fails(tmp_path) -> None:
    path = tmp_path / "a.js"
    path.write_text(BROKEN_JS)
    assert validate_syntax(str(path), "javascript")["valid"] is False


def test_unsupported_language_counts_as_skipped(tmp_path) -> None:
    path = tmp_path / "a.rb"
    path.write_text("puts 1\n")
    summary = validate_rewrites([str(path)], "ruby")
    assert (summary["passed"], summary["skipped"], summary["failed"]) == (0, 1, 0)
