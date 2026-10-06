"""Tests that bracket balance ignores brackets inside literals and comments."""

import pytest

from ast_grep_mcp.utils.syntax_validation import validate_bracket_balance, validate_code_for_language


@pytest.mark.parametrize(
    ("code", "language"),
    [
        ('const s = "(";', "javascript"),
        ("const re = `{${x}`;", "typescript"),
        ("// TODO: handle [ edge\nf(x);", "javascript"),
        ("/* { */ call();", "javascript"),
        ("char c = '{';", "java"),
        ('String s = "a\\"(";', "java"),
    ],
)
def test_brackets_in_literals_and_comments_are_ignored(code: str, language: str) -> None:
    assert validate_code_for_language(code, language) == (True, "")


def test_real_imbalance_is_still_reported() -> None:
    assert validate_bracket_balance('f("(", x;') == [("parentheses", "Mismatched parentheses")]
