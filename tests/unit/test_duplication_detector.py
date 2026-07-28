"""Tests for DuplicationDetector - core duplication detection functionality.

Tests cover:
- Initialization with different similarity modes
- Parameter validation
- Construct pattern generation for multiple languages
- Similarity calculation (hybrid, minhash, sequence_matcher)
- Code normalization
- Duplicate grouping and bucket creation
- Group merging
- Refactoring suggestion generation
- Statistics calculation
- Result formatting
"""

import tempfile
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from ast_grep_mcp.constants import SemanticSimilarityDefaults
from ast_grep_mcp.features.deduplication.detector import (
    DuplicationDetector,
)


class TestDuplicationDetectorInit:
    """Tests for DuplicationDetector initialization."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        detector = DuplicationDetector()

        assert detector.language == "python"
        assert detector.similarity_mode == "hybrid"
        assert detector.use_minhash is True
        assert detector.logger is not None

    def test_init_with_language(self):
        """Test initialization with specific language."""
        detector = DuplicationDetector(language="javascript")

        assert detector.language == "javascript"

    def test_init_with_minhash_mode(self):
        """Test initialization with minhash mode."""
        detector = DuplicationDetector(similarity_mode="minhash")

        assert detector.similarity_mode == "minhash"
        assert detector.use_minhash is True

    def test_init_with_sequence_matcher_mode(self):
        """Test initialization with sequence_matcher mode."""
        detector = DuplicationDetector(similarity_mode="sequence_matcher")

        assert detector.similarity_mode == "sequence_matcher"
        assert detector.use_minhash is False

    def test_init_legacy_use_minhash_false(self):
        """Test that legacy use_minhash=False sets sequence_matcher mode."""
        detector = DuplicationDetector(use_minhash=False)

        assert detector.similarity_mode == "sequence_matcher"
        assert detector.use_minhash is False

    def test_init_creates_similarity_calculators(self):
        """Test that similarity calculators are created."""
        detector = DuplicationDetector()

        assert detector._minhash is not None
        assert detector._hybrid is not None
        assert detector._structure_hash is not None


class TestValidateParameters:
    """Tests for _validate_parameters method."""

    def test_valid_parameters(self):
        """Test that valid parameters pass validation."""
        detector = DuplicationDetector()

        # Should not raise
        detector._validate_parameters(0.8, 5, 100)

    def test_min_similarity_below_zero(self):
        """Test that min_similarity below 0 raises ValueError."""
        detector = DuplicationDetector()

        with pytest.raises(ValueError, match="min_similarity must be between"):
            detector._validate_parameters(-0.1, 5, 100)

    def test_min_similarity_above_one(self):
        """Test that min_similarity above 1 raises ValueError."""
        detector = DuplicationDetector()

        with pytest.raises(ValueError, match="min_similarity must be between"):
            detector._validate_parameters(1.5, 5, 100)

    def test_min_lines_below_one(self):
        """Test that min_lines below 1 raises ValueError."""
        detector = DuplicationDetector()

        with pytest.raises(ValueError, match="min_lines must be at least"):
            detector._validate_parameters(0.8, 0, 100)

    def test_max_constructs_negative(self):
        """Test that negative max_constructs raises ValueError."""
        detector = DuplicationDetector()

        with pytest.raises(ValueError, match="max_constructs must be"):
            detector._validate_parameters(0.8, 5, -1)

    def test_max_constructs_zero_allowed(self):
        """Test that max_constructs=0 (unlimited) is allowed."""
        detector = DuplicationDetector()

        # Should not raise
        detector._validate_parameters(0.8, 5, 0)


class TestGetConstructPattern:
    """Tests for _get_construct_pattern method."""

    def test_python_function_pattern(self):
        """Test Python function pattern."""
        detector = DuplicationDetector(language="python")

        pattern = detector._get_construct_pattern("function_definition")

        assert "def $NAME" in pattern

    def test_python_class_pattern(self):
        """Test Python class pattern."""
        detector = DuplicationDetector(language="python")

        pattern = detector._get_construct_pattern("class_definition")

        assert "class $NAME" in pattern

    def test_javascript_uses_yaml_rule_not_pattern(self):
        """JS/TS uses kind-based YAML rules, not source patterns (TSD-02/03/04)."""
        detector = DuplicationDetector(language="javascript")
        rule = detector._get_construct_yaml_rule("function_definition")
        assert "kind: function_declaration" in rule
        assert "language: javascript" in rule

    def test_typescript_yaml_rule_function_declaration(self):
        """TypeScript function_definition maps to function_declaration kind."""
        detector = DuplicationDetector(language="typescript")
        rule = detector._get_construct_yaml_rule("function_definition")
        assert "kind: function_declaration" in rule
        assert "language: typescript" in rule

    def test_typescript_yaml_rule_arrow_function(self):
        """TypeScript arrow_function kind."""
        detector = DuplicationDetector(language="typescript")
        rule = detector._get_construct_yaml_rule("arrow_function")
        assert "kind: arrow_function" in rule

    def test_typescript_yaml_rule_method_definition(self):
        """TypeScript method_definition uses method_definition kind (not a broken source pattern)."""
        detector = DuplicationDetector(language="typescript")
        rule = detector._get_construct_yaml_rule("method_definition")
        assert "kind: method_definition" in rule

    def test_typescript_yaml_rule_traditional_function(self):
        """Traditional function also maps to function_declaration kind."""
        detector = DuplicationDetector(language="typescript")
        rule = detector._get_construct_yaml_rule("traditional_function")
        assert "kind: function_declaration" in rule

    def test_typescript_yaml_rule_class_definition(self):
        """TypeScript class_definition maps to class_declaration kind."""
        detector = DuplicationDetector(language="typescript")
        rule = detector._get_construct_yaml_rule("class_definition")
        assert "kind: class_declaration" in rule

    def test_tsx_yaml_rule_function(self):
        """TSX uses kind-based YAML rule."""
        detector = DuplicationDetector(language="tsx")
        rule = detector._get_construct_yaml_rule("function_definition")
        assert "kind: function_declaration" in rule
        assert "language: tsx" in rule

    def test_yaml_rule_unknown_construct_falls_back_to_function_declaration(self):
        """Unknown construct type defaults to function_declaration kind."""
        detector = DuplicationDetector(language="typescript")
        rule = detector._get_construct_yaml_rule("unknown_type")
        assert "kind: function_declaration" in rule

    def test_java_function_pattern(self):
        """Test Java function pattern."""
        detector = DuplicationDetector(language="java")

        pattern = detector._get_construct_pattern("function_definition")

        assert "$TYPE $NAME" in pattern

    def test_csharp_function_pattern(self):
        """Test C# function pattern."""
        detector = DuplicationDetector(language="csharp")

        pattern = detector._get_construct_pattern("function_definition")

        assert "$TYPE $NAME" in pattern

    def test_unknown_construct_fallback(self):
        """Test that unknown construct type falls back to function_definition."""
        detector = DuplicationDetector(language="python")

        pattern = detector._get_construct_pattern("unknown_type")

        assert "def $NAME" in pattern

    def test_jsx_uses_yaml_rule(self):
        """JSX uses kind-based YAML rules for construct discovery."""
        detector = DuplicationDetector(language="jsx")
        rule = detector._get_construct_yaml_rule("function_definition")
        assert "kind: function_declaration" in rule
        assert "language: jsx" in rule

    def test_tsx_uses_yaml_rule(self):
        """TSX uses kind-based YAML rules for construct discovery."""
        detector = DuplicationDetector(language="tsx")
        rule = detector._get_construct_yaml_rule("function_definition")
        assert "kind: function_declaration" in rule
        assert "language: tsx" in rule

    def test_javascript_class_definition_uses_class_pattern(self):
        """Test that JS class_definition returns 'class $NAME', not const fallback (BUG-05)."""
        detector = DuplicationDetector(language="javascript")

        pattern = detector._get_construct_pattern("class_definition")

        assert "class $NAME" in pattern
        assert "const" not in pattern

    def test_typescript_class_definition_uses_class_pattern(self):
        """Test that TS class_definition returns 'class $NAME', not const fallback (BUG-05)."""
        detector = DuplicationDetector(language="typescript")

        pattern = detector._get_construct_pattern("class_definition")

        assert "class $NAME" in pattern
        assert "const" not in pattern

    def test_tsx_class_definition_uses_class_pattern(self):
        """Test that TSX class_definition returns 'class $NAME', not const fallback (BUG-05)."""
        detector = DuplicationDetector(language="tsx")

        pattern = detector._get_construct_pattern("class_definition")

        assert "class $NAME" in pattern
        assert "const" not in pattern

    def test_jsx_class_definition_uses_class_pattern(self):
        """Test that JSX class_definition returns 'class $NAME', not const fallback (BUG-05)."""
        detector = DuplicationDetector(language="jsx")

        pattern = detector._get_construct_pattern("class_definition")

        assert "class $NAME" in pattern
        assert "const" not in pattern


class TestCalculateSimilarity:
    """Tests for calculate_similarity method."""

    def test_empty_code_returns_zero(self):
        """Test that empty code returns 0.0 similarity."""
        detector = DuplicationDetector()

        assert detector.calculate_similarity("", "def func(): pass") == 0.0
        assert detector.calculate_similarity("def func(): pass", "") == 0.0
        assert detector.calculate_similarity("", "") == 0.0

    def test_identical_code_high_similarity(self):
        """Test that identical code has high similarity."""
        detector = DuplicationDetector()

        code = "def func():\n    return 42"
        similarity = detector.calculate_similarity(code, code)

        assert similarity >= 0.9

    def test_different_code_low_similarity(self):
        """Test that very different code has low similarity."""
        detector = DuplicationDetector()

        code1 = "def func1(): return 1"
        code2 = "class MyClass: pass"

        similarity = detector.calculate_similarity(code1, code2)

        assert similarity < 0.5

    def test_hybrid_mode_uses_hybrid(self):
        """Test that hybrid mode uses hybrid calculator."""
        detector = DuplicationDetector(similarity_mode="hybrid")

        with patch.object(
            detector._hybrid,
            "estimate_similarity",
            return_value=SemanticSimilarityDefaults.MEDIUM_SIMILARITY_THRESHOLD,
        ) as mock:
            result = detector.calculate_similarity("code1", "code2")

            mock.assert_called_once_with("code1", "code2")
            assert result == SemanticSimilarityDefaults.MEDIUM_SIMILARITY_THRESHOLD

    def test_minhash_mode_uses_minhash(self):
        """Test that minhash mode uses minhash calculator."""
        detector = DuplicationDetector(similarity_mode="minhash")

        with patch.object(detector._minhash, "estimate_similarity", return_value=0.75) as mock:
            result = detector.calculate_similarity("code1", "code2")

            mock.assert_called_once_with("code1", "code2")
            assert result == 0.75

    def test_sequence_matcher_mode(self):
        """Test sequence_matcher mode uses SequenceMatcher."""
        detector = DuplicationDetector(similarity_mode="sequence_matcher")

        code1 = "def func(): return 1"
        code2 = "def func(): return 1"

        similarity = detector.calculate_similarity(code1, code2)

        assert similarity >= 0.9


class TestCalculateSimilarityPrecise:
    """Tests for calculate_similarity_precise method."""

    def test_empty_code_returns_zero(self):
        """Test that empty code returns 0.0."""
        detector = DuplicationDetector()

        assert detector.calculate_similarity_precise("", "code") == 0.0
        assert detector.calculate_similarity_precise("code", "") == 0.0

    def test_identical_code_returns_one(self):
        """Test that identical code returns 1.0."""
        detector = DuplicationDetector()

        code = "def func(): return 42"

        assert detector.calculate_similarity_precise(code, code) == 1.0


class TestCalculateSimilarityDetailed:
    """Tests for calculate_similarity_detailed method."""

    def test_returns_hybrid_result(self):
        """Test that detailed calculation returns HybridSimilarityResult."""
        detector = DuplicationDetector()

        result = detector.calculate_similarity_detailed("def func1(): return 1", "def func2(): return 2")

        # Should return a result with similarity attribute
        assert hasattr(result, "similarity")
        assert 0.0 <= result.similarity <= 1.0


class TestNormalizeCode:
    """Tests for _normalize_code method."""

    def test_removes_empty_lines(self):
        """Test that empty lines are removed."""
        detector = DuplicationDetector()

        code = "def func():\n\n    return 1"
        normalized = detector._normalize_code(code)

        assert "\n\n" not in normalized

    def test_strips_trailing_whitespace(self):
        """Test that trailing whitespace is stripped."""
        detector = DuplicationDetector()

        code = "def func():   \n    return 1   "
        normalized = detector._normalize_code(code)

        assert not any(line.endswith(" ") for line in normalized.split("\n"))

    def test_normalizes_indentation(self):
        """Test that deep indentation is normalized."""
        detector = DuplicationDetector()

        code = "        deeply_indented()"
        normalized = detector._normalize_code(code)

        # Should have max 4 spaces of indentation
        indent = len(normalized) - len(normalized.lstrip())
        assert indent <= 4


class TestGroupDuplicates:
    """Tests for group_duplicates method."""

    def test_empty_matches_returns_empty(self):
        """Test that empty matches returns empty list."""
        detector = DuplicationDetector()

        result = detector.group_duplicates([], 0.8, 5)

        assert result == []

    def test_filters_by_min_lines(self):
        """Test that matches below min_lines are filtered."""
        detector = DuplicationDetector()

        matches = [
            {"text": "a\nb\nc", "file": "f1.py", "range": {"start": {"line": 1}}},  # 3 lines
            {"text": "a\nb\nc\nd\ne", "file": "f2.py", "range": {"start": {"line": 1}}},  # 5 lines
        ]

        result = detector.group_duplicates(matches, 0.8, 5)

        # Only one match >= 5 lines, so no groups
        assert result == []

    def test_groups_similar_code(self):
        """Test that similar code is grouped together."""
        detector = DuplicationDetector()

        code = "def func():\n    x = 1\n    y = 2\n    z = 3\n    return x + y + z"
        matches = [
            {"text": code, "file": "f1.py", "range": {"start": {"line": 1}}},
            {"text": code, "file": "f2.py", "range": {"start": {"line": 10}}},
        ]

        result = detector.group_duplicates(matches, 0.8, 3)

        assert len(result) == 1
        assert len(result[0]) == 2


class TestCreateHashBuckets:
    """Tests for _create_hash_buckets method."""

    def test_creates_buckets(self):
        """Test that hash buckets are created."""
        detector = DuplicationDetector()

        matches = [
            {"text": "def func1(): pass", "file": "f1.py"},
            {"text": "def func2(): pass", "file": "f2.py"},
        ]

        buckets = detector._create_hash_buckets(matches)

        assert isinstance(buckets, dict)
        # All matches should be in some bucket
        total_in_buckets = sum(len(b) for b in buckets.values())
        assert total_in_buckets == 2


class TestCalculateStructureHash:
    """Tests for _calculate_structure_hash method."""

    def test_returns_integer(self):
        """Test that hash is an integer."""
        detector = DuplicationDetector()

        hash_val = detector._calculate_structure_hash("def func(): pass")

        assert isinstance(hash_val, int)

    def test_similar_code_same_bucket(self):
        """Test that similar code gets similar hashes."""
        detector = DuplicationDetector()

        code1 = "def func1():\n    return 1"
        code2 = "def func2():\n    return 2"

        hash1 = detector._calculate_structure_hash(code1)
        hash2 = detector._calculate_structure_hash(code2)

        # Similar structure should have same hash
        assert hash1 == hash2


class TestFindSimilarInBucket:
    """Tests for _find_similar_in_bucket method."""

    def test_finds_similar_items(self):
        """Test finding similar items in bucket."""
        detector = DuplicationDetector()

        code = "def func():\n    x = 1\n    return x"
        bucket = [
            {"text": code, "file": "f1.py", "range": {"start": {"line": 1}}},
            {"text": code, "file": "f2.py", "range": {"start": {"line": 1}}},
        ]

        groups = detector._find_similar_in_bucket(bucket, 0.8)

        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_skips_dissimilar_items(self):
        """Test that dissimilar items are not grouped."""
        detector = DuplicationDetector()

        bucket = [
            {"text": "def func1(): pass", "file": "f1.py", "range": {"start": {"line": 1}}},
            {"text": "class MyClass:\n    x = 1\n    y = 2", "file": "f2.py", "range": {"start": {"line": 1}}},
        ]

        groups = detector._find_similar_in_bucket(bucket, 0.9)

        assert len(groups) == 0


class TestItemHelpers:
    """Tests for item helper methods."""

    def test_get_item_key(self):
        """Test _get_item_key generates unique key."""
        detector = DuplicationDetector()

        item = {"file": "/path/to/file.py", "range": {"start": {"line": 10}}}
        key = detector._get_item_key(item)

        assert key == "/path/to/file.py:10"

    def test_items_equal_same_items(self):
        """Test _items_equal returns True for same items."""
        detector = DuplicationDetector()

        item1 = {"file": "f.py", "range": {"start": {"line": 5}}}
        item2 = {"file": "f.py", "range": {"start": {"line": 5}}}

        assert detector._items_equal(item1, item2) is True

    def test_items_equal_different_items(self):
        """Test _items_equal returns False for different items."""
        detector = DuplicationDetector()

        item1 = {"file": "f.py", "range": {"start": {"line": 5}}}
        item2 = {"file": "f.py", "range": {"start": {"line": 10}}}

        assert detector._items_equal(item1, item2) is False


class TestMergeOverlappingGroups:
    """Tests for _merge_overlapping_groups method."""

    def test_empty_groups(self):
        """Test that empty groups returns empty list."""
        detector = DuplicationDetector()

        result = detector._merge_overlapping_groups([])

        assert result == []

    def test_merges_overlapping_groups(self):
        """Test that overlapping groups are merged."""
        detector = DuplicationDetector()

        item1 = {"file": "f1.py", "range": {"start": {"line": 1}}}
        item2 = {"file": "f2.py", "range": {"start": {"line": 1}}}
        item3 = {"file": "f3.py", "range": {"start": {"line": 1}}}

        # Group 1: item1, item2
        # Group 2: item2, item3
        # Should merge into one group with all three
        groups = [
            [item1, item2],
            [item2, item3],
        ]

        result = detector._merge_overlapping_groups(groups)

        assert len(result) == 1
        assert len(result[0]) == 3

    def test_keeps_separate_groups(self):
        """Test that non-overlapping groups stay separate."""
        detector = DuplicationDetector()

        groups = [
            [
                {"file": "f1.py", "range": {"start": {"line": 1}}},
                {"file": "f2.py", "range": {"start": {"line": 1}}},
            ],
            [
                {"file": "f3.py", "range": {"start": {"line": 1}}},
                {"file": "f4.py", "range": {"start": {"line": 1}}},
            ],
        ]

        result = detector._merge_overlapping_groups(groups)

        assert len(result) == 2


class TestBuildItemToGroupsMap:
    """Tests for _build_item_to_groups_map method."""

    def test_builds_mapping(self):
        """Test that mapping is built correctly."""
        detector = DuplicationDetector()

        groups = [
            [{"file": "f1.py", "range": {"start": {"line": 1}}}],
            [{"file": "f2.py", "range": {"start": {"line": 1}}}],
        ]

        mapping = detector._build_item_to_groups_map(groups)

        assert "f1.py:1" in mapping
        assert "f2.py:1" in mapping


class TestAddUniqueItems:
    """Tests for _add_unique_items method."""

    def test_adds_unique_items(self):
        """Test that unique items are added."""
        detector = DuplicationDetector()

        target: List[Dict[str, Any]] = []
        source = [
            {"file": "f1.py", "range": {"start": {"line": 1}}},
            {"file": "f2.py", "range": {"start": {"line": 1}}},
        ]

        detector._add_unique_items(target, source)

        assert len(target) == 2

    def test_skips_duplicates(self):
        """Test that duplicates are not added."""
        detector = DuplicationDetector()

        item = {"file": "f1.py", "range": {"start": {"line": 1}}}
        target = [item]
        source = [item, {"file": "f2.py", "range": {"start": {"line": 1}}}]

        detector._add_unique_items(target, source)

        assert len(target) == 2


class TestGenerateRefactoringSuggestions:
    """Tests for generate_refactoring_suggestions method."""

    def test_empty_groups_returns_empty(self):
        """Test that empty groups returns empty suggestions."""
        detector = DuplicationDetector()

        result = detector.generate_refactoring_suggestions([], "function_definition")

        assert result == []

    def test_single_item_groups_skipped(self):
        """Test that single-item groups are skipped."""
        detector = DuplicationDetector()

        groups = [[{"text": "def func(): pass", "file": "f.py", "range": {"start": {"line": 1}}}]]

        result = detector.generate_refactoring_suggestions(groups, "function_definition")

        assert result == []

    def test_generates_suggestions(self):
        """Test that suggestions are generated for valid groups."""
        detector = DuplicationDetector()

        code = "def func():\n    return 1"
        groups = [
            [
                {"text": code, "file": "f1.py", "range": {"start": {"line": 1}}},
                {"text": code, "file": "f2.py", "range": {"start": {"line": 10}}},
            ]
        ]

        result = detector.generate_refactoring_suggestions(groups, "function_definition")

        assert len(result) == 1
        assert result[0]["duplicate_count"] == 2
        assert result[0]["lines_per_duplicate"] == 2
        assert "refactoring_strategy" in result[0]

    def test_total_lines_uses_per_item_counts(self):
        """Regression test for BUGL-01: total_duplicated_lines must sum actual
        per-item line counts, not multiply group[0]'s count by len(group)."""
        detector = DuplicationDetector()

        short_code = "def a():\n    return 1"  # 2 lines
        long_code = "def a():\n    x = 1\n    y = 2\n    return x + y"  # 4 lines

        groups = [
            [
                {"text": short_code, "file": "f1.py", "range": {"start": {"line": 1}}},
                {"text": long_code, "file": "f2.py", "range": {"start": {"line": 10}}},
            ]
        ]

        result = detector.generate_refactoring_suggestions(groups, "function_definition")

        assert len(result) == 1
        # lines_per_duplicate is always from group[0]
        assert result[0]["lines_per_duplicate"] == 2
        # total must be sum of actual lengths (2 + 4), NOT 2 * 2 = 4
        assert result[0]["total_duplicated_lines"] == 6
        # savings = total - kept copy (group[0])
        assert result[0]["potential_line_savings"] == 4


class TestDetermineRefactoringStrategy:
    """Tests for _determine_refactoring_strategy method."""

    def test_small_function_extract_utility(self):
        """Test that small functions suggest extract utility."""
        detector = DuplicationDetector()

        group = [{"text": "def f():\n    return 1"}]  # 2 lines

        strategy = detector._determine_refactoring_strategy(group, "function_definition")

        assert strategy["type"] == "extract_utility_function"

    def test_large_function_extract_module(self):
        """Test that large functions suggest extract module."""
        detector = DuplicationDetector()

        # Create code with 15 lines
        lines = ["def func():"] + ["    x = 1"] * 14
        code = "\n".join(lines)
        group = [{"text": code}]

        strategy = detector._determine_refactoring_strategy(group, "function_definition")

        assert strategy["type"] == "extract_module"

    def test_class_extract_base_class(self):
        """Test that classes suggest extract base class."""
        detector = DuplicationDetector()

        group = [{"text": "class MyClass:\n    pass"}]

        strategy = detector._determine_refactoring_strategy(group, "class_definition")

        assert strategy["type"] == "extract_base_class"

    def test_method_extract_method(self):
        """Test that methods suggest extract method."""
        detector = DuplicationDetector()

        group = [{"text": "def method():\n    pass"}]

        strategy = detector._determine_refactoring_strategy(group, "method_definition")

        assert strategy["type"] == "extract_method"


class TestCalculateStatistics:
    """Tests for _calculate_statistics method."""

    def test_calculates_statistics(self):
        """Test that statistics are calculated correctly."""
        detector = DuplicationDetector()

        all_matches = [{"file": "f1.py"}, {"file": "f2.py"}]
        duplication_groups = [[{}, {}]]
        suggestions = [{"total_duplicated_lines": 20, "potential_line_savings": 10}]

        stats = detector._calculate_statistics(all_matches, duplication_groups, suggestions)

        assert stats["total_constructs"] == 2
        assert stats["duplicate_groups"] == 1
        assert stats["total_duplicated_lines"] == 20
        assert stats["potential_line_savings"] == 10


class TestEmptyResult:
    """Tests for _empty_result method."""

    def test_returns_empty_result_structure(self):
        """Test that empty result has correct structure."""
        detector = DuplicationDetector()

        result = detector._empty_result("function_definition", 1.5)

        assert result["summary"]["total_constructs"] == 0
        assert result["summary"]["duplicate_groups"] == 0
        assert result["duplication_groups"] == []
        assert result["refactoring_suggestions"] == []
        assert "function_definition" in result["message"]


class TestFormatResult:
    """Tests for _format_result method."""

    def test_formats_result(self):
        """Test that result is formatted correctly."""
        detector = DuplicationDetector()

        code = "def func(): pass"
        all_matches = [{"file": "f1.py"}]
        duplication_groups = [
            [
                {"file": "f1.py", "range": {"start": {"line": 0}, "end": {"line": 1}}, "text": code},
                {"file": "f2.py", "range": {"start": {"line": 5}, "end": {"line": 6}}, "text": code},
            ]
        ]
        suggestions = [{"group_id": 1, "potential_line_savings": 5}]
        stats = {"total_constructs": 1, "duplicate_groups": 1, "total_duplicated_lines": 10, "potential_line_savings": 5}

        result = detector._format_result(all_matches, duplication_groups, suggestions, stats, 0.5)

        assert "summary" in result
        assert "duplication_groups" in result
        assert "refactoring_suggestions" in result
        assert "message" in result
        assert result["summary"]["analysis_time_seconds"] == 0.5

    def test_formats_group_instances(self):
        """Test that group instances are formatted with file and line info."""
        detector = DuplicationDetector()

        code = "def func(): pass"
        groups = [
            [
                {"file": "/path/f1.py", "range": {"start": {"line": 9}, "end": {"line": 10}}, "text": code},
                {"file": "/path/f2.py", "range": {"start": {"line": 19}, "end": {"line": 20}}, "text": code},
            ]
        ]

        stats = {"total_constructs": 0, "duplicate_groups": 1, "total_duplicated_lines": 0, "potential_line_savings": 0}
        result = detector._format_result([], groups, [], stats, 0.1)

        assert len(result["duplication_groups"]) == 1
        assert len(result["duplication_groups"][0]["instances"]) == 2
        assert result["duplication_groups"][0]["instances"][0]["file"] == "/path/f1.py"
        assert result["duplication_groups"][0]["instances"][0]["lines"] == "10-11"

    def test_format_group_includes_files_field(self):
        """Regression test for BUG-04: _format_group must emit a 'files' field.

        Downstream ranking and enrichment read candidate.get('files', []) to
        compute test coverage, recommendations, and savings potential.  Without
        this field those steps silently no-op and return zero/empty values.
        """
        detector = DuplicationDetector()
        code = "def foo():\n    return 1"
        group = [
            {"file": "src/a.py", "range": {"start": {"line": 0}, "end": {"line": 1}}, "text": code},
            {"file": "src/b.py", "range": {"start": {"line": 5}, "end": {"line": 6}}, "text": code},
        ]

        formatted = detector._format_group(0, group)

        assert "files" in formatted, "'files' key missing from _format_group output"
        assert set(formatted["files"]) == {"src/a.py", "src/b.py"}

    def test_format_group_includes_potential_line_savings(self):
        """Regression test for BUG-04: _format_group must emit 'potential_line_savings'.

        The ranker's calculate_savings_score() reads potential_line_savings (40% weight).
        Without it, every candidate scores zero on savings and ranking is meaningless.
        """
        detector = DuplicationDetector()
        # 3-line snippet duplicated twice → saves 3 lines
        code = "def foo():\n    x = 1\n    return x"
        group = [
            {"file": "src/a.py", "range": {"start": {"line": 0}, "end": {"line": 2}}, "text": code},
            {"file": "src/b.py", "range": {"start": {"line": 10}, "end": {"line": 12}}, "text": code},
        ]

        formatted = detector._format_group(0, group)

        assert "potential_line_savings" in formatted, "'potential_line_savings' key missing"
        assert formatted["potential_line_savings"] > 0, "potential_line_savings should be non-zero for a duplicated group"


class TestFindDuplication:
    """Tests for find_duplication method."""

    def test_empty_project_returns_empty_result(self):
        """Test that empty project returns empty result."""
        detector = DuplicationDetector()

        with patch("ast_grep_mcp.features.deduplication.detector.stream_ast_grep_results") as mock_stream:
            mock_stream.return_value = iter([])

            with tempfile.TemporaryDirectory() as tmpdir:
                result = detector.find_duplication(tmpdir)

            assert result["summary"]["total_constructs"] == 0
            assert result["duplication_groups"] == []

    def test_filters_excluded_patterns(self):
        """Test that excluded patterns are filtered."""
        detector = DuplicationDetector()

        code = "def f():\n    pass\n    pass\n    pass\n    pass"
        matches = [
            {"file": "/project/src/main.py", "text": code, "range": {"start": {"line": 1}}},
            {"file": "/project/node_modules/lib.py", "text": code, "range": {"start": {"line": 1}}},
        ]

        with patch("ast_grep_mcp.features.deduplication.detector.stream_ast_grep_results") as mock_stream:
            mock_stream.return_value = iter(matches)

            with tempfile.TemporaryDirectory() as tmpdir:
                result = detector.find_duplication(tmpdir, exclude_patterns=["node_modules"])

            # node_modules file should be excluded
            assert result["summary"]["total_constructs"] == 1

    def test_propagates_exceptions(self):
        """Test that exceptions are propagated."""
        detector = DuplicationDetector()

        with patch("ast_grep_mcp.features.deduplication.detector.stream_ast_grep_results") as mock_stream:
            mock_stream.side_effect = RuntimeError("Test error")

            with pytest.raises(RuntimeError, match="Test error"):
                with tempfile.TemporaryDirectory() as tmpdir:
                    detector.find_duplication(tmpdir)


class TestProcessGroupConnections:
    """Tests for _process_group_connections method."""

    def test_processes_connections(self):
        """Test that group connections are processed."""
        detector = DuplicationDetector()

        item1 = {"file": "f1.py", "range": {"start": {"line": 1}}}
        item2 = {"file": "f2.py", "range": {"start": {"line": 1}}}

        groups = [[item1, item2]]
        item_to_groups = {
            "f1.py:1": [0],
            "f2.py:1": [0],
        }
        used_groups: set[int] = {0}
        to_merge: list[int] = []
        merged_group: list[Dict[str, Any]] = []

        detector._process_group_connections(0, groups, item_to_groups, used_groups, to_merge, merged_group)

        # No new groups to merge since group 0 is already used
        assert len(to_merge) == 0


class TestEdgeCases:
    """Tests for edge cases to improve coverage."""

    def test_javascript_unknown_construct_yaml_default(self):
        """Unknown JS construct type defaults to function_declaration kind."""
        detector = DuplicationDetector(language="javascript")
        rule = detector._get_construct_yaml_rule("unknown_custom_type")
        assert "kind: function_declaration" in rule

    def test_group_duplicates_all_below_min_lines(self):
        """Test group_duplicates when all matches are below min_lines."""
        detector = DuplicationDetector()

        # All matches have 2 lines, but min_lines is 5
        matches = [
            {"text": "line1\nline2", "file": "f1.py", "range": {"start": {"line": 1}}},
            {"text": "line1\nline2", "file": "f2.py", "range": {"start": {"line": 1}}},
        ]

        result = detector.group_duplicates(matches, 0.8, 5)

        assert result == []

    def test_find_similar_in_bucket_skips_used_items(self):
        """Test that _find_similar_in_bucket skips already used items."""
        detector = DuplicationDetector()

        code = "def func():\n    x = 1\n    y = 2\n    return x + y"
        # Create 3 items where all are similar
        bucket = [
            {"text": code, "file": "f1.py", "range": {"start": {"line": 1}}},
            {"text": code, "file": "f2.py", "range": {"start": {"line": 1}}},
            {"text": code, "file": "f3.py", "range": {"start": {"line": 1}}},
        ]

        groups = detector._find_similar_in_bucket(bucket, 0.8)

        # Should create one group with all 3 items
        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_find_constructs_logs_limit_reached(self):
        """Test that _find_constructs logs when limit is reached."""
        detector = DuplicationDetector()

        # Create exactly max_constructs matches
        matches = [{"file": f"/project/f{i}.py", "text": f"def func{i}(): pass", "range": {"start": {"line": 1}}} for i in range(5)]

        with patch("ast_grep_mcp.features.deduplication.detector.stream_ast_grep_results") as mock_stream:
            mock_stream.return_value = iter(matches)

            with tempfile.TemporaryDirectory() as tmpdir:
                result = detector._find_constructs(tmpdir, "def $NAME($$$)", 5, [])

            # Should have all 5 matches
            assert len(result) == 5

    def test_find_similar_in_bucket_inner_loop_continue(self):
        """Test that inner loop skips already used items (line 411).

        Scenario: 3 items where item1 and item3 are similar but item2 is different.
        - Processing item1 (i=0): matches item3, adds index 2 to used
        - Processing item2 (i=1): different from others, inner loop iterates to j=2
        - Since j=2 is in used, the inner loop continue (line 411) is hit
        """
        detector = DuplicationDetector()

        code_a = "def calculate():\n    x = 1\n    y = 2\n    z = 3\n    return x + y + z"
        code_b = "class Completely:\n    different = True\n    structure = False"
        code_c = "def calculate():\n    x = 1\n    y = 2\n    z = 3\n    return x + y + z"

        bucket = [
            {"text": code_a, "file": "f1.py", "range": {"start": {"line": 1}}},
            {"text": code_b, "file": "f2.py", "range": {"start": {"line": 1}}},
            {"text": code_c, "file": "f3.py", "range": {"start": {"line": 1}}},
        ]

        groups = detector._find_similar_in_bucket(bucket, 0.8)

        # Should create one group with items 1 and 3 (code_a and code_c)
        assert len(groups) == 1
        assert len(groups[0]) == 2


# ── Precision filter tests ──────────────────────────────────────


def _make_match(code: str, file: str = "a.py", line: int = 0) -> Dict[str, Any]:
    """Helper to build a match dict."""
    return {"text": code, "file": file, "range": {"start": {"line": line}}}


class TestTrivialConstructorFilter:
    """Tests for _is_trivial_constructor_group."""

    def test_short_init_detected(self):
        detector = DuplicationDetector()
        group = [
            _make_match("def __init__(self, lang):\n    self.lang = lang", "a.py"),
            _make_match("def __init__(self, lang):\n    self.lang = lang", "b.py"),
        ]
        assert detector._is_trivial_constructor_group(group) is True

    def test_long_init_not_filtered(self):
        detector = DuplicationDetector()
        body = "\n".join([f"    self.f{i} = f{i}" for i in range(12)])
        code = f"def __init__(self, **kw):\n{body}"
        group = [_make_match(code, "a.py"), _make_match(code, "b.py")]
        assert detector._is_trivial_constructor_group(group) is False

    def test_non_init_not_filtered(self):
        detector = DuplicationDetector()
        group = [
            _make_match("def process(self):\n    pass", "a.py"),
            _make_match("def process(self):\n    pass", "b.py"),
        ]
        assert detector._is_trivial_constructor_group(group) is False

    def test_mixed_init_and_regular(self):
        """Group with mix of __init__ and regular method is NOT trivial."""
        detector = DuplicationDetector()
        group = [
            _make_match("def __init__(self, x):\n    self.x = x", "a.py"),
            _make_match("def process(self, x):\n    self.x = x", "b.py"),
        ]
        assert detector._is_trivial_constructor_group(group) is False

    def test_js_constructor(self):
        detector = DuplicationDetector()
        group = [
            _make_match("function constructor(a) {\n  this.a = a;\n}", "a.js"),
            _make_match("function constructor(b) {\n  this.b = b;\n}", "b.js"),
        ]
        assert detector._is_trivial_constructor_group(group) is True

    def test_empty_group_returns_false(self):
        detector = DuplicationDetector()
        assert detector._is_trivial_constructor_group([]) is False


class TestDelegationWrapperFilter:
    """Tests for _is_delegation_wrapper_group."""

    def test_thin_wrapper_detected(self):
        detector = DuplicationDetector()
        group = [
            _make_match("def error(self, msg):\n    self._log(msg)", "a.py"),
            _make_match("def warning(self, msg):\n    self._log(msg)", "a.py", 10),
        ]
        assert detector._is_delegation_wrapper_group(group) is True

    def test_substantial_body_not_filtered(self):
        detector = DuplicationDetector()
        code = "def process(self, data):\n    x = validate(data)\n    y = transform(x)\n    z = enrich(y)\n    return z"
        group = [_make_match(code, "a.py"), _make_match(code, "b.py")]
        assert detector._is_delegation_wrapper_group(group) is False

    def test_super_call_wrapper(self):
        detector = DuplicationDetector()
        group = [
            _make_match("def detect(self, path):\n    super().detect(path)", "a.py"),
            _make_match("def detect(self, path):\n    super().detect(path)", "b.py"),
        ]
        assert detector._is_delegation_wrapper_group(group) is True

    def test_chained_attribute_call_detected(self):
        """self._logger.info(...) is a thin chained delegation wrapper."""
        detector = DuplicationDetector()
        group = [
            _make_match("def warn(self, msg):\n    self._logger.warn(msg)", "a.py"),
            _make_match("def info(self, msg):\n    self._logger.info(msg)", "a.py", 10),
        ]
        assert detector._is_delegation_wrapper_group(group) is True

    def test_empty_group_returns_false(self):
        detector = DuplicationDetector()
        assert detector._is_delegation_wrapper_group([]) is False


class TestParallelFormatterFilter:
    """Tests for _is_parallel_formatter_group."""

    def test_parallel_to_formatters_detected(self):
        detector = DuplicationDetector()
        group = [
            _make_match("def to_python(self):\n    return f'def {self.name}()'", "a.py"),
            _make_match("def to_typescript(self):\n    return f'function {self.name}()'", "a.py", 10),
        ]
        assert detector._is_parallel_formatter_group(group) is True

    def test_same_name_not_flagged(self):
        """Identical to_python in two files is real duplication, not parallel."""
        detector = DuplicationDetector()
        group = [
            _make_match("def to_python(self):\n    return f'def {self.name}()'", "a.py"),
            _make_match("def to_python(self):\n    return f'def {self.name}()'", "b.py"),
        ]
        assert detector._is_parallel_formatter_group(group) is False

    def test_non_formatter_not_flagged(self):
        detector = DuplicationDetector()
        group = [
            _make_match("def process_a(self):\n    pass", "a.py"),
            _make_match("def process_b(self):\n    pass", "b.py"),
        ]
        assert detector._is_parallel_formatter_group(group) is False


class TestMinSavingsFilter:
    """Tests for _meets_min_savings."""

    def test_high_savings_passes(self):
        detector = DuplicationDetector()
        code = "\n".join([f"    line_{i}" for i in range(25)])
        full = f"def big_func():\n{code}"
        group = [_make_match(full, "a.py"), _make_match(full, "b.py")]
        assert detector._meets_min_savings(group) is True

    def test_low_savings_filtered(self):
        detector = DuplicationDetector()
        code = "def small():\n    return 1\n    # pad\n    # pad2"
        group = [_make_match(code, "a.py"), _make_match(code, "b.py")]
        # 4 lines * 2 = 8 total, savings = 4 < 20
        assert detector._meets_min_savings(group) is False

    def test_many_copies_increase_savings(self):
        detector = DuplicationDetector()
        code = "def med():\n" + "\n".join([f"    x{i} = {i}" for i in range(7)])
        # 8 lines * 4 copies = 32, savings = 24 >= 20
        group = [_make_match(code, f"f{i}.py") for i in range(4)]
        assert detector._meets_min_savings(group) is True

    def test_empty_group(self):
        detector = DuplicationDetector()
        assert detector._meets_min_savings([]) is False

    def test_variable_length_members_use_sum_minus_min(self):
        """Near-duplicates with different lengths: savings = sum - min(lengths)."""
        detector = DuplicationDetector()
        # long: 22 lines, short: 5 lines → savings = 27 - 5 = 22 >= 20
        long_code = "def big():\n" + "\n".join([f"    x{i} = {i}" for i in range(20)])
        short_code = "def small():\n" + "\n".join([f"    x{i} = {i}" for i in range(3)])
        group = [_make_match(long_code, "a.py"), _make_match(short_code, "b.py")]
        assert detector._meets_min_savings(group) is True

    def test_savings_boundary_at_threshold(self):
        """savings == MIN_LINE_SAVINGS - 1 should fail; == MIN_LINE_SAVINGS should pass."""
        detector = DuplicationDetector()
        # 9 lines * 2 copies → savings = 9, below threshold of 20
        code_short = "def f():\n" + "\n".join([f"    x{i} = {i}" for i in range(8)])
        group = [_make_match(code_short, "a.py"), _make_match(code_short, "b.py")]
        assert detector._meets_min_savings(group) is False
        # 21 lines * 2 copies → savings = 21, at/above threshold
        code_long = "def g():\n" + "\n".join([f"    x{i} = {i}" for i in range(20)])
        group2 = [_make_match(code_long, "a.py"), _make_match(code_long, "b.py")]
        assert detector._meets_min_savings(group2) is True


class TestApplyPrecisionFilters:
    """Integration test for _apply_precision_filters combining all filters."""

    def test_all_false_positive_patterns_removed(self):
        """The 5 false-positive patterns from the investigation should be filtered."""
        detector = DuplicationDetector()

        # Group 1: trivial __init__
        init_group = [
            _make_match("def __init__(self, lang):\n    self.lang = lang", "a.py"),
            _make_match("def __init__(self, lang):\n    self.lang = lang", "b.py"),
        ]
        # Group 2: delegation wrappers
        wrapper_group = [
            _make_match("def error(self, msg):\n    self._log_to_stderr(msg)", "c.py"),
            _make_match("def warning(self, msg):\n    self._log_to_stderr(msg)", "c.py", 10),
        ]
        # Group 3: parallel formatters
        formatter_group = [
            _make_match("def to_python_signature(self):\n    return f'def {self.name}()'", "d.py"),
            _make_match("def to_typescript_signature(self):\n    return f'function {self.name}()'", "d.py", 10),
        ]
        # Group 4: below min savings (tiny functions)
        tiny_group = [
            _make_match("def query_a(self):\n    return self.db.query('SELECT * FROM a')", "e.py"),
            _make_match("def query_b(self):\n    return self.db.query('SELECT * FROM b')", "e.py", 10),
        ]

        groups = [init_group, wrapper_group, formatter_group, tiny_group]
        result = detector._apply_precision_filters(groups)
        assert result == []

    def test_legitimate_duplicates_preserved(self):
        """Real duplicates above savings threshold are kept."""
        detector = DuplicationDetector()
        body = "\n".join([f"    result.append(process(item_{i}))" for i in range(22)])
        code = f"def process_batch(items):\n{body}"
        group = [_make_match(code, "a.py"), _make_match(code, "b.py")]

        result = detector._apply_precision_filters([group])
        assert len(result) == 1


class TestFindConstructsBUG07:
    """Regression tests for BUG-07: exclude patterns must not consume the construct budget.

    Previously, max_constructs was passed as the stream limit before filtering.
    Excluded paths (.venv, node_modules) appearing early in the walk consumed
    the entire budget and were then discarded, leaving zero real matches.
    """

    def test_excluded_matches_do_not_consume_limit(self) -> None:
        """Real matches survive even when excluded matches would fill the old budget."""
        detector = DuplicationDetector()

        excluded = [
            {"file": f"/project/.venv/lib/f{i}.py", "text": f"def func{i}(): pass", "range": {"start": {"line": 1}}}
            for i in range(10)
        ]
        real = [
            {"file": f"/project/src/module{i}.py", "text": f"def real{i}(): pass", "range": {"start": {"line": 1}}}
            for i in range(3)
        ]
        # Stream returns excluded first, then real — old code would cap at 5 (all excluded)
        all_raw = excluded + real

        with patch("ast_grep_mcp.features.deduplication.detector.stream_ast_grep_results") as mock_stream:
            mock_stream.return_value = iter(all_raw)

            with tempfile.TemporaryDirectory() as tmpdir:
                result = detector._find_constructs(tmpdir, "def $NAME($$$)", max_constructs=5, exclude_patterns=[".venv"])

        assert len(result) == 3, "real matches must survive after excluded ones are filtered out"
        assert all("/project/src/" in m["file"] for m in result)

    def test_limit_applied_after_filtering(self) -> None:
        """max_constructs truncates kept matches, not raw matches."""
        detector = DuplicationDetector()

        real = [
            {"file": f"/project/src/m{i}.py", "text": f"def f{i}(): pass", "range": {"start": {"line": 1}}}
            for i in range(10)
        ]

        with patch("ast_grep_mcp.features.deduplication.detector.stream_ast_grep_results") as mock_stream:
            mock_stream.return_value = iter(real)

            with tempfile.TemporaryDirectory() as tmpdir:
                result = detector._find_constructs(tmpdir, "def $NAME($$$)", max_constructs=4, exclude_patterns=[])

        assert len(result) == 4

    def test_zero_max_constructs_returns_all(self) -> None:
        """max_constructs=0 returns all kept matches without truncation."""
        detector = DuplicationDetector()

        real = [
            {"file": f"/project/src/m{i}.py", "text": f"def f{i}(): pass", "range": {"start": {"line": 1}}}
            for i in range(8)
        ]

        with patch("ast_grep_mcp.features.deduplication.detector.stream_ast_grep_results") as mock_stream:
            mock_stream.return_value = iter(real)

            with tempfile.TemporaryDirectory() as tmpdir:
                result = detector._find_constructs(tmpdir, "def $NAME($$$)", max_constructs=0, exclude_patterns=[])

        assert len(result) == 8

    def test_stream_called_with_unlimited_max_results(self) -> None:
        """_find_constructs must pass max_results=0 to the stream (no pre-filter)."""
        detector = DuplicationDetector()

        with patch("ast_grep_mcp.features.deduplication.detector.stream_ast_grep_results") as mock_stream:
            mock_stream.return_value = iter([])

            with tempfile.TemporaryDirectory() as tmpdir:
                detector._find_constructs(tmpdir, "def $NAME($$$)", max_constructs=5, exclude_patterns=[])

        _, kwargs = mock_stream.call_args
        assert kwargs.get("max_results") == 0, "stream must be called with max_results=0 so excluded paths can't consume the budget"
