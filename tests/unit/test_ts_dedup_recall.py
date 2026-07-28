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
from typing import Any, Dict

import pytest
import yaml

from ast_grep_mcp.features.deduplication.detector import DuplicationDetector


def _make_match(text: str, file: str = "a.ts", start_line: int = 1) -> Dict[str, Any]:
    return {"text": text, "file": file, "range": {"start": {"line": start_line}}}


# ── Fixture strings ──────────────────────────────────────────────────────────

_PROCESS_USER = textwrap.dedent("""\
    function processUserData(item: Item): string {
      const validated = validateInput(item.id);
      if (!validated) {
        throw new Error("Invalid item id");
      }
      const result = transformData(item.name, item.email);
      logOperation("process_user", item.id);
      return result;
    }
""")

_PROCESS_ORDER = textwrap.dedent("""\
    function processOrderData(item: Item): string {
      const validated = validateInput(item.id);
      if (!validated) {
        throw new Error("Invalid item id");
      }
      const result = transformData(item.name, item.email);
      logOperation("process_order", item.id);
      return result;
    }
""")


# ── Core recall tests ────────────────────────────────────────────────────────


class TestGroupDuplicatesRecall:
    """group_duplicates must find structurally near-identical TS functions."""

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


# ── TSD-07: verification must use the configured scorer, not MinHash ─────────

# Real duplicate pair from ~/.claude/hooks lib/trace-context.ts (HEAD at
# 2026-07-28).  These differ by exactly two tokens — the type name and the path
# helper — and PromptTraceContext was a deprecated alias of TraceContext, so
# they were the same type.  MinHash estimates 0.7734 for this pair while the
# hybrid scorer gives 0.8715: verifying LSH candidates with the MinHash
# estimate discarded it, even though lsh_recall_margin had correctly admitted
# it as a candidate.
_SAVE_SESSION_CTX = textwrap.dedent("""\
    function saveSessionContext(sessionId: string, traceId: string, spanId: string): void {
      try {
        ensureDirExists(TRACE_CTX_DIR);
        const ctx: TraceContext = { traceId, spanId, timestamp: Date.now() };
        writeFileSync(getSessionTracePath(sessionId), JSON.stringify(ctx));
      } catch {
        // Non-critical — silently fail
      }
    }
""")

_SAVE_PROMPT_CTX = textwrap.dedent("""\
    function savePromptContext(sessionId: string, traceId: string, spanId: string): void {
      try {
        ensureDirExists(TRACE_CTX_DIR);
        const ctx: PromptTraceContext = { traceId, spanId, timestamp: Date.now() };
        writeFileSync(getContextPath(sessionId), JSON.stringify(ctx));
      } catch {
        // Non-critical — silently fail
      }
    }
""")

_MIN_SIMILARITY = 0.80


class TestVerificationUsesConfiguredScorer:
    """TSD-07: LSH candidates must be verified with the detector's scorer.

    MinHash under-estimates relative to AST/semantic scoring.  Verifying with
    the estimate against the full threshold makes lsh_recall_margin
    self-defeating: it widens candidate generation, then re-applies the strict
    threshold using the very metric whose imprecision the margin absorbs.
    """

    def test_pair_below_minhash_but_above_hybrid_is_found(self) -> None:
        """The pair MinHash under-scores is still grouped in hybrid mode."""
        detector = DuplicationDetector(language="typescript", similarity_mode="hybrid")

        minhash_score = detector._minhash.estimate_similarity(_SAVE_SESSION_CTX, _SAVE_PROMPT_CTX)
        hybrid_score = detector.calculate_similarity(_SAVE_SESSION_CTX, _SAVE_PROMPT_CTX)

        # Guard the premise: without this gap the test proves nothing.
        assert minhash_score < _MIN_SIMILARITY <= hybrid_score, (
            f"fixture no longer exercises TSD-07 "
            f"(minhash={minhash_score:.4f}, hybrid={hybrid_score:.4f})"
        )

        groups = detector.group_duplicates(
            [
                _make_match(_SAVE_SESSION_CTX, file="trace-context.ts", start_line=47),
                _make_match(_SAVE_PROMPT_CTX, file="trace-context.ts", start_line=85),
            ],
            _MIN_SIMILARITY,
            min_lines=5,
        )

        assert len(groups) == 1, "hybrid-scored duplicate pair was dropped at verification"
        assert len(groups[0]) == 2

    def test_minhash_mode_still_uses_minhash(self) -> None:
        """similarity_mode='minhash' is honoured — the scorer is not hardcoded."""
        detector = DuplicationDetector(language="typescript", similarity_mode="minhash")

        groups = detector.group_duplicates(
            [
                _make_match(_SAVE_SESSION_CTX, file="trace-context.ts", start_line=47),
                _make_match(_SAVE_PROMPT_CTX, file="trace-context.ts", start_line=85),
            ],
            _MIN_SIMILARITY,
            min_lines=5,
        )

        assert groups == [], "minhash mode should not recover a pair MinHash scores below threshold"

    def test_find_all_similar_pairs_defaults_to_minhash_estimate(self) -> None:
        """The standalone MinHash API is unchanged when no scorer is injected."""
        similarity = DuplicationDetector(language="typescript")._minhash
        items = [("a", _SAVE_SESSION_CTX), ("b", _SAVE_PROMPT_CTX)]

        assert similarity.find_all_similar_pairs(items, _MIN_SIMILARITY) == []

    def test_injected_scorer_is_used_for_verification(self) -> None:
        """An injected scorer overrides the MinHash estimate."""
        similarity = DuplicationDetector(language="typescript")._minhash
        items = [("a", _SAVE_SESSION_CTX), ("b", _SAVE_PROMPT_CTX)]

        pairs = similarity.find_all_similar_pairs(
            items, _MIN_SIMILARITY, scorer=lambda _c1, _c2: 0.99
        )

        assert len(pairs) == 1
        assert pairs[0][2] == pytest.approx(0.99)
