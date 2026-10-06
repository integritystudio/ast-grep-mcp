"""Tests that AST-based security issues report the match's real location."""

from ast_grep_mcp.features.quality.security_scanner import _match_to_issue

PATTERN_DEF = {"severity": "high", "title": "eval", "description": "d", "remediation": "r"}


def test_issue_location_comes_from_ast_grep_range() -> None:
    # ast-grep JSON: no top-level line/column, 0-based range positions
    match = {
        "file": "app.py",
        "text": "eval(input())",
        "range": {"start": {"line": 1, "column": 4}, "end": {"line": 1, "column": 17}},
    }

    issue = _match_to_issue(match, PATTERN_DEF)

    assert (issue.line, issue.column, issue.end_line, issue.end_column) == (2, 5, 2, 18)
