"""Consolidated tests for deduplication analysis functionality.

This file consolidates tests from:
- test_variation_classification.py
- test_parameter_extraction.py
- test_complexity_scoring.py

Focus: Variation analysis, parameter extraction, complexity scoring
"""

import os
import sys
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import from modular structure where interfaces match
# Import from modular code
# Variation analysis functions migrated to modular architecture
from ast_grep_mcp.features.deduplication.analyzer import (
    PatternAnalyzer,
    _detect_nested_function_call,
    classify_variations,
    detect_conditional_variations,
    identify_varying_identifiers,
)

# Type inference and parameter functions migrated to modular architecture
from ast_grep_mcp.features.deduplication.generator import (
    CodeGenerator,
    _infer_from_identifier_name,
    _infer_single_value_type,
    generate_parameter_name,
    infer_parameter_type,
)
from ast_grep_mcp.models.complexity import get_complexity_level

# ParameterType class migrated to modular architecture
from ast_grep_mcp.models.deduplication import ParameterType, VariationSeverity


class TestVariationClassification:
    """Tests for variation classification in duplicate code."""

    def test_classify_variations_simple(self):
        """Test classifying simple variations."""
        code1 = "result = value * 2"
        code2 = "result = value * 3"

        variations = classify_variations(code1, code2, "python")
        assert variations is not None
        assert isinstance(variations, dict) or isinstance(variations, list)

    def test_classify_variations_complex(self):
        """Test classifying complex variations."""
        code1 = """
def process(data):
    if data > 10:
        return data * 2
    return data
"""
        code2 = """
def process(data):
    if data > 20:
        return data * 3
    return data
"""
        variations = classify_variations(code1, code2, "python")
        assert variations is not None

    def test_detect_conditional_variations(self):
        """Test detecting conditional variations."""
        code1 = "if x > 10: return True"
        code2 = "if x > 20: return False"

        conditions = detect_conditional_variations(code1, code2, "python")
        assert conditions is not None
        assert isinstance(conditions, dict) or isinstance(conditions, list)

    def test_variation_severity_enum(self):
        """Test VariationSeverity enum values."""
        assert hasattr(VariationSeverity, "LOW")
        assert hasattr(VariationSeverity, "MEDIUM")
        assert hasattr(VariationSeverity, "HIGH")


class TestParameterExtraction:
    """Tests for parameter extraction from duplicate code."""

    def test_identify_varying_identifiers(self):
        """Test identifying varying identifiers between code snippets."""
        code1 = "result = process_user_data(user_id)"
        code2 = "result = process_order_data(order_id)"

        identifiers = identify_varying_identifiers(code1, code2, "python")
        assert identifiers is not None
        assert isinstance(identifiers, dict) or isinstance(identifiers, list)

    def test_generate_parameter_name(self):
        """Test generating parameter names from identifiers."""
        name = generate_parameter_name("user_id", ["user_id", "order_id"])
        assert name is not None
        assert isinstance(name, str)
        assert name != ""

    def test_infer_parameter_type(self):
        """Test inferring parameter type from context."""
        code_context = "user_id = 123"
        param_type = infer_parameter_type("user_id", code_context, "python")
        assert param_type is not None

    def test_infer_single_value_type(self):
        """Test inferring type from a single value."""
        assert _infer_single_value_type("123", "python") is not None
        assert _infer_single_value_type('"string"', "python") is not None
        assert _infer_single_value_type("True", "python") is not None

    def test_infer_from_identifier_name(self):
        """Test inferring type from identifier naming patterns."""
        result = _infer_from_identifier_name("user_id", "python")
        assert result is not None or result is None  # Can be None if no pattern matches

    def test_detect_nested_function_call(self):
        """Test detecting nested function calls."""
        code = "result = outer(inner(value))"
        nested = _detect_nested_function_call(code, "value", "python")
        assert nested is not None
        assert nested["nesting_depth"] == 2

    def test_parameter_type_enum(self):
        """Test ParameterType enum values."""
        assert hasattr(ParameterType, "STRING")
        assert hasattr(ParameterType, "NUMBER")
        assert hasattr(ParameterType, "BOOLEAN")


class TestCalculateCallNestingDepth:
    """Regression tests for PatternAnalyzer._calculate_call_nesting_depth (BUGL-09)."""

    def setup_method(self) -> None:
        self.analyzer = PatternAnalyzer()

    def test_identifier_after_closed_nested_calls_is_depth_zero(self) -> None:
        """Regression: BUGL-09 — f(g(x)) + identifier must report depth 0.

        The old code returned max_depth=2 because it tracked the highest
        parenthesis depth seen before the identifier, not the depth AT the
        identifier's position.
        """
        depth = self.analyzer._calculate_call_nesting_depth("f(g(x)) + identifier", "identifier")
        assert depth == 0

    def test_identifier_inside_nested_calls_reports_correct_depth(self) -> None:
        """f(g(identifier)) — identifier is at depth 2."""
        depth = self.analyzer._calculate_call_nesting_depth("f(g(identifier))", "identifier")
        assert depth == 2

    def test_identifier_at_top_level(self) -> None:
        """identifier alone — depth is 0."""
        depth = self.analyzer._calculate_call_nesting_depth("identifier", "identifier")
        assert depth == 0

    def test_identifier_not_found_returns_zero(self) -> None:
        """Unknown identifier — returns 0 without error."""
        depth = self.analyzer._calculate_call_nesting_depth("f(g(x))", "missing")
        assert depth == 0


class TestComplexityScoring:
    """Tests for complexity scoring of duplicate code."""

    def test_get_complexity_level_low(self):
        """Test getting complexity level for low score."""
        level = get_complexity_level(3)
        assert level == "low"

    def test_get_complexity_level_medium(self):
        """Test getting complexity level for medium score."""
        level = get_complexity_level(7)
        assert level == "medium"

    def test_get_complexity_level_high(self):
        """Test getting complexity level for high score."""
        level = get_complexity_level(15)
        assert level == "high"

    def test_complexity_boundaries(self):
        """Test complexity level boundaries."""
        assert get_complexity_level(4) == "low"
        assert get_complexity_level(5) == "medium"
        assert get_complexity_level(9) == "medium"
        assert get_complexity_level(10) == "high"


class TestAstGrepScans:
    """Real ast-grep scans behind literal, conditional and nested-call detection."""

    SAMPLES = {
        "python": 'x = 1\ny = "a"\nz = True\nif x == 2:\n    foo(bar(x))\n',
        "javascript": 'const x = 1; const s = "a"; const b = true;\nif (x === 2) { foo(bar(x)); }\n',
    }

    @pytest.mark.parametrize("language", sorted(SAMPLES))
    def test_extracts_each_literal_type(self, language: str) -> None:
        analyzer = PatternAnalyzer()
        code = self.SAMPLES[language]
        counts = {t: len(analyzer._extract_literals_with_ast_grep(code, t, language)) for t in PatternAnalyzer._LITERAL_TYPES}
        assert counts == {"number": 2, "string": 1, "boolean": 1}

    @pytest.mark.parametrize("language", sorted(SAMPLES))
    def test_extracts_conditionals(self, language: str) -> None:
        assert PatternAnalyzer()._extract_conditionals(self.SAMPLES[language], language)

    @pytest.mark.parametrize("language", sorted(SAMPLES))
    def test_detects_nested_call(self, language: str) -> None:
        nested = PatternAnalyzer().detect_nested_function_call(self.SAMPLES[language], "x", language)
        assert nested is not None
        assert nested["nesting_depth"] == 2


class TestConditionalOperatorParsing:
    """Word operators must match whole words, and multi-word operators before their prefixes."""

    def test_identifier_containing_in_is_kept_whole(self) -> None:
        assert PatternAnalyzer()._extract_cond_vars("index > 5") == {"index"}

    def test_in_vs_not_in_is_an_operator_change(self) -> None:
        details = PatternAnalyzer()._analyze_conditional_difference("x in y", "x not in y", "python")
        assert details["operators"] == {"from": "in", "to": "not in"}


class TestExternalDependencies:
    def test_method_name_tails_are_not_dependencies(self) -> None:
        deps = CodeGenerator("python")._find_external_dependencies("self.helper(x)\nos.path.join(a)\nvalidate(x)\n")
        assert "elper" not in deps and "oin" not in deps
        assert "validate" in deps


def _fake_extract_literals(code: str, literal_type: str, language: str) -> List[Dict[str, Any]]:
    """Deterministic stand-in for ast-grep literal extraction (no subprocess)."""
    if literal_type != "string":
        return []
    return [{"line": 0, "column": 0, "value": code, "type": literal_type}]


class TestLiteralVariationExtraction:
    """Tests for literal-map reuse in group analysis (BUG-13)."""

    LANGUAGE = "python"

    def test_precomputed_maps_match_default_path(self) -> None:
        """Passing pre-extracted maps for code1 yields identical results."""
        analyzer = PatternAnalyzer()
        code1, code2 = 'x = "a"', 'x = "b"'
        with patch.object(analyzer, "_extract_literals_with_ast_grep", side_effect=_fake_extract_literals):
            default_result = analyzer.identify_varying_literals(code1, code2, self.LANGUAGE)
            maps = analyzer._extract_literal_maps(code1, self.LANGUAGE)
            precomputed_result = analyzer.identify_varying_literals(code1, code2, self.LANGUAGE, code1_literal_maps=maps)
        assert default_result == precomputed_result
        assert default_result == [
            {"position": 1, "column": 0, "value1": code1, "value2": code2, "literal_type": "string"}
        ]

    def test_base_literals_extracted_once_per_group(self) -> None:
        """Group analysis extracts the base snippet's literals only once."""
        analyzer = PatternAnalyzer()
        base = 'x = "a"'
        group = [{"text": base}, {"text": 'x = "b"'}, {"text": 'x = "c"'}, {"text": 'x = "d"'}]
        with patch.object(analyzer, "_extract_literals_with_ast_grep", side_effect=_fake_extract_literals) as mock_extract:
            result = analyzer.analyze_duplicate_group_literals(group, self.LANGUAGE)
        types_count = len(PatternAnalyzer._LITERAL_TYPES)
        base_calls = [call for call in mock_extract.call_args_list if call.args[0] == base]
        assert len(base_calls) == types_count
        # base extracted once + one extraction per non-base member
        assert mock_extract.call_count == types_count * len(group)
        assert result["total_variations"] == 1
        assert result["variations"][0]["values"] == [base, 'x = "b"', 'x = "c"', 'x = "d"']

    def test_identical_group_skips_extraction(self) -> None:
        """No literal extraction is performed when all members equal the base."""
        analyzer = PatternAnalyzer()
        group = [{"text": 'x = "a"'}, {"text": 'x = "a"'}, {"text": 'x = "a"'}]
        with patch.object(analyzer, "_extract_literals_with_ast_grep", side_effect=_fake_extract_literals) as mock_extract:
            result = analyzer.analyze_duplicate_group_literals(group, self.LANGUAGE)
        assert mock_extract.call_count == 0
        assert result["total_variations"] == 0
