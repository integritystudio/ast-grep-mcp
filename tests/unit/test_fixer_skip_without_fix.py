"""Tests that violations with no available fix are skipped, never deleted."""

import pytest

from ast_grep_mcp.features.quality.fixer import _apply_single_fix
from ast_grep_mcp.models.standards import RuleViolation

SOURCE = "import os\nx = eval(input())\nos.system('ls ' + x)\n"
FLAGGED_LINE = 2


@pytest.mark.parametrize("rule_id", ["no-eval-exec", "no-sql-injection", "no-hardcoded-credentials"])
def test_violation_without_fix_is_skipped(tmp_path, rule_id: str) -> None:
    path = tmp_path / "app.py"
    path.write_text(SOURCE)
    violation = RuleViolation(
        file=str(path),
        line=FLAGGED_LINE,
        column=5,
        end_line=FLAGGED_LINE,
        end_column=18,
        severity="error",
        rule_id=rule_id,
        message="flagged",
        code_snippet="eval(input())",
        fix_suggestion=None,
    )

    result = _apply_single_fix(str(path), violation, "python")

    assert result.fix_type == "skipped"
    assert result.file_modified is False
    assert path.read_text() == SOURCE
