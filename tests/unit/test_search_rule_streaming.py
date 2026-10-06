"""Tests that limited (streamed) rule searches return the same matches as unlimited ones."""

import pytest

from ast_grep_mcp.features.search.service import find_code_by_rule_impl

RULE = "id: find-print\nlanguage: python\nrule:\n  pattern: print($A)\n"
SOURCE = "print(1)\nprint(2)\nprint(3)\n"
LIMIT = 2


@pytest.fixture
def project(tmp_path):
    (tmp_path / "app.py").write_text(SOURCE)
    return str(tmp_path)


def test_limited_json_search_returns_matches(project) -> None:
    matches = find_code_by_rule_impl(project, RULE, max_results=LIMIT, output_format="json")
    assert isinstance(matches, list)
    assert [m["text"] for m in matches] == ["print(1)", "print(2)"]


def test_limited_text_search_returns_matches(project) -> None:
    text = find_code_by_rule_impl(project, RULE, max_results=LIMIT, output_format="text")
    assert isinstance(text, str)
    assert "print(1)" in text and "print(2)" in text
    assert "print(3)" not in text
