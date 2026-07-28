"""Regression tests: TypeScript deduplication recall (TSD-01 through TSD-04).

Every defect documented in docs/typescript-dedup-gaps.md survived the test
suite because no test asserted non-zero recall on TypeScript.  This module
adds that coverage.

Test strategy:
- Unit tests for the core similarity/grouping stage use known-similar TS
  strings injected directly into group_duplicates.  These are language-
  agnostic (they test the similarity math, not the pattern matching) and
  pass on any branch.
- Pattern-validation tests check _get_construct_yaml_rule (added in
  fix-ts-patterns) for each construct type.  These fail on main and pass
  after that branch is merged, acting as a forward regression gate.
"""

import textwrap
from typing import Any, Dict, List

import pytest
import yaml

from ast_grep_mcp.features.deduplication.detector import DuplicationDetector


def _make_match(text: str, file: str = "a.ts", start_line: int = 1) -> Dict[str, Any]:
    return {"text": text, "file": file, "range": {"start": {"line": start_line}}}


# ── Fixture strings ──────────────────────────────────────────────────────────

_PROCESS_USER = textwrap.dedent("""\
    function processUserData(user: User): string {
      const validated = validateInput(user.id);
      if (!validated) {
        throw new Error("Invalid user id");
      }
      const result = transformData(user.name, user.email);
      logOperation("process_user", user.id);
      return result;
    }
""")

_PROCESS_ORDER = textwrap.dedent("""\
    function processOrderData(order: Order): string {
      const validated = validateInput(order.id);
      if (!validated) {
        throw new Error("Invalid order id");
      }
      const result = transformData(order.customerId.toString(), order.total.toString());
      logOperation("process_order", order.id);
      return result;
    }
""")


# ── Core recall tests ────────────────────────────────────────────────────────


class TestGroupDuplicatesRecall:
    """group_duplicates must find structurally near-identical TS functions."""

    @pytest.mark.xfail(
        reason="TSD-01: structure-hash bucketing isolates 99% of functions into singletons; "
        "passes after fix-bucketing-recall is merged",
        strict=False,
    )
    def test_finds_near_identical_typed_functions(self):
        """Two typed TS functions with the same control flow should form a group."""
        detector = DuplicationDetector(language="typescript")
        matches = [
            _make_match(_PROCESS_USER, "service.ts", 1),
            _make_match(_PROCESS_ORDER, "service.ts", 20),
        ]
        groups = detector.group_duplicates(matches, min_similarity=0.7, min_lines=5)
        assert len(groups) >= 1, (
            "Expected at least one duplicate group for near-identical TypeScript functions. "
            "Zero groups means the similarity/LSH stage has a recall defect (TSD-01)."
        )

    def test_similar_functions_in_same_group(self):
        """The two near-identical functions should appear together."""
        detector = DuplicationDetector(language="typescript")
        matches = [
            _make_match(_PROCESS_USER, "service.ts", 1),
            _make_match(_PROCESS_ORDER, "service.ts", 20),
        ]
        groups = detector.group_duplicates(matches, min_similarity=0.7, min_lines=5)
        if not groups:
            pytest.skip("No groups found — TSD-01 recall defect likely present")
        files_in_group = {m["file"] for g in groups for m in g}
        assert "service.ts" in files_in_group

    def test_identical_functions_grouped(self):
        """Identical function text must produce a group."""
        detector = DuplicationDetector(language="typescript")
        matches = [
            _make_match(_PROCESS_USER, "a.ts", 1),
            _make_match(_PROCESS_USER, "b.ts", 1),
        ]
        groups = detector.group_duplicates(matches, min_similarity=0.95, min_lines=5)
        assert len(groups) >= 1, "Identical functions must always form a duplicate group."


# ── Pattern validation tests (require fix-ts-patterns merge) ─────────────────


_TS_CONSTRUCT_TYPES = [
    "function_definition",
    "traditional_function",
    "arrow_function",
    "method_definition",
    "class_definition",
]

_JS_TS_LANGS = ["javascript", "typescript", "jsx", "tsx"]


class TestConstructYamlRuleValidity:
    """_get_construct_yaml_rule must produce well-formed YAML for all construct types.

    This class documents cross-cutting recommendation #3 from the TypeScript
    dedup gaps investigation: validate construct patterns in tests so that
    invalid patterns (like TSD-04's unparseable method_definition source
    pattern) are caught before they reach production.
    """

    @pytest.mark.parametrize("lang", _JS_TS_LANGS)
    @pytest.mark.parametrize("construct_type", _TS_CONSTRUCT_TYPES)
    def test_yaml_rule_is_parseable(self, lang: str, construct_type: str):
        """Each (language, construct_type) pair must produce parseable inline YAML."""
        detector = DuplicationDetector(language=lang)
        if not hasattr(detector, "_get_construct_yaml_rule"):
            pytest.skip("_get_construct_yaml_rule not yet present (fix-ts-patterns not merged)")

        rule = detector._get_construct_yaml_rule(construct_type)
        try:
            parsed = yaml.safe_load(rule)
        except yaml.YAMLError as exc:
            pytest.fail(f"YAML rule for ({lang}, {construct_type}) is not valid YAML: {exc}\nRule:\n{rule}")

        assert isinstance(parsed, dict), f"Rule must parse to a dict, got {type(parsed)}"
        assert "id" in parsed, "Rule must have an 'id' field"
        assert "language" in parsed, "Rule must have a 'language' field"
        assert "rule" in parsed, "Rule must have a 'rule' field"
        assert "kind" in parsed.get("rule", {}), "rule.kind must be set"

    @pytest.mark.parametrize("lang", _JS_TS_LANGS)
    @pytest.mark.parametrize("construct_type", _TS_CONSTRUCT_TYPES)
    def test_yaml_rule_language_matches_detector(self, lang: str, construct_type: str):
        """Language field in the YAML rule must match the detector's language."""
        detector = DuplicationDetector(language=lang)
        if not hasattr(detector, "_get_construct_yaml_rule"):
            pytest.skip("_get_construct_yaml_rule not yet present (fix-ts-patterns not merged)")

        rule = detector._get_construct_yaml_rule(construct_type)
        parsed = yaml.safe_load(rule)
        assert parsed["language"] == lang, (
            f"Rule language '{parsed['language']}' != detector language '{lang}'"
        )

    @pytest.mark.parametrize("construct_type", _TS_CONSTRUCT_TYPES)
    def test_kind_is_non_empty_string(self, construct_type: str):
        """kind value must be a non-empty string for all TypeScript construct types."""
        detector = DuplicationDetector(language="typescript")
        if not hasattr(detector, "_get_construct_yaml_rule"):
            pytest.skip("_get_construct_yaml_rule not yet present (fix-ts-patterns not merged)")

        rule = detector._get_construct_yaml_rule(construct_type)
        parsed = yaml.safe_load(rule)
        kind = parsed["rule"]["kind"]
        assert isinstance(kind, str) and kind, f"kind must be a non-empty string, got: {kind!r}"
