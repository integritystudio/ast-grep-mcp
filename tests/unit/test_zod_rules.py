"""Tests that Zod rules match what they claim and never rewrite into each other."""

import json
import subprocess
from pathlib import Path

from ast_grep_mcp.features.rewrite.service import rewrite_code_impl
from ast_grep_mcp.features.rewrite.zod_templates import ZOD_IMPORT_RULES, ZOD_SCHEMA_RULES, get_zod_rule

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBPATH_RULE_ID = "import-zod-subpath-to-namespace"
IMPORTS = 'import { z } from "zod/v4";\nimport { z } from "zod";\n'


def test_subpath_template_rewrites_only_subpath_imports(tmp_path) -> None:
    source = tmp_path / "schema.ts"
    source.write_text(IMPORTS)

    rewrite_code_impl(str(tmp_path), get_zod_rule(SUBPATH_RULE_ID), dry_run=False, backup=False)

    assert source.read_text() == 'import * as z from "zod/v4";\nimport { z } from "zod";\n'


def test_subpath_rule_file_matches(tmp_path) -> None:
    (tmp_path / "schema.ts").write_text(IMPORTS)
    result = subprocess.run(
        ["ast-grep", "scan", "--rule", str(REPO_ROOT / "rules" / "import-zod-rules.yaml"), "--json", str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    hits = [m for m in json.loads(result.stdout) if m["ruleId"] == SUBPATH_RULE_ID]
    assert [m["text"] for m in hits] == ['import { z } from "zod/v4";']


def test_no_rewrite_introduces_z_any() -> None:
    for rule_id, rule in {**ZOD_SCHEMA_RULES, **ZOD_IMPORT_RULES}.items():
        fix = rule.split("\nfix:", 1)[1].split("\n", 1)[0]
        assert "z.any()" not in fix, rule_id


def test_zod_rule_files_load(tmp_path) -> None:
    (tmp_path / "a.ts").write_text("const user = z.object({});\n")
    for name in ("import-zod-rules.yaml", "zod-validation-rules.yaml"):
        rule_file = str(REPO_ROOT / "rules" / name)
        result = subprocess.run(["ast-grep", "scan", "--rule", rule_file, str(tmp_path)], capture_output=True, text=True)
        assert result.returncode == 0, f"{name}: {result.stderr}"
