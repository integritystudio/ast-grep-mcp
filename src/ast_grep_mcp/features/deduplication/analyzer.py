"""
Pattern analysis module for deduplication.

This module provides functionality for analyzing code patterns,
identifying variations, and classifying differences between duplicate code blocks.
"""

import functools
import json
import os
import re
import subprocess
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

from ...constants import DifficultyThresholds, RegexCaptureGroups, SemanticVolumeDefaults, SubprocessDefaults
from ...core.logging import get_logger
from ...models.deduplication import VariationCategory, VariationSeverity
from .scoring_scales import VariationScoreCutoff, VariationScoreScale

IDENTIFIER_CONTEXT_WINDOW_CHARS = 20

_INLINE_RULE_ID = "dedup-analyzer-scan"


@functools.lru_cache(maxsize=None)
def _is_valid_kind(language: str, kind: str) -> bool:
    """Return True if ast-grep's grammar for language has a node kind named kind.

    ast-grep rejects a whole rule when any `kind` in it is unknown to the grammar,
    so multi-language `any` lists must be filtered per language first.
    """
    rule_yaml = yaml.dump({"id": _INLINE_RULE_ID, "language": language, "rule": {"kind": kind}})
    with tempfile.TemporaryDirectory() as empty_dir:
        result = subprocess.run(
            ["ast-grep", "scan", "--inline-rules", rule_yaml, empty_dir],
            capture_output=True,
            text=True,
            timeout=SubprocessDefaults.GREP_TIMEOUT_SECONDS,
        )
    return result.returncode == 0


def _kind_rule_for_language(rule: Dict[str, Any], language: str) -> Optional[Dict[str, Any]]:
    """Drop kinds the language's grammar lacks; None if no valid kind remains."""
    inner = rule["rule"]
    if "any" in inner:
        valid = [k for k in inner["any"] if _is_valid_kind(language, k["kind"])]
        return {"any": valid} if valid else None
    return inner if _is_valid_kind(language, inner["kind"]) else None

_LANGUAGE_KEYWORDS: Dict[str, set[str]] = {
    "python": {
        "def",
        "class",
        "if",
        "else",
        "elif",
        "for",
        "while",
        "return",
        "import",
        "from",
        "as",
        "try",
        "except",
        "finally",
        "with",
        "lambda",
        "yield",
        "raise",
        "pass",
        "break",
        "continue",
        "and",
        "or",
        "not",
        "in",
        "is",
        "None",
        "True",
        "False",
        "async",
        "await",
        "global",
        "nonlocal",
        "assert",
    },
    "javascript": {
        "function",
        "const",
        "let",
        "var",
        "if",
        "else",
        "for",
        "while",
        "return",
        "import",
        "export",
        "from",
        "class",
        "new",
        "this",
        "try",
        "catch",
        "finally",
        "throw",
        "async",
        "await",
        "true",
        "false",
        "null",
        "undefined",
    },
    "typescript": {
        "function",
        "const",
        "let",
        "var",
        "if",
        "else",
        "for",
        "while",
        "return",
        "import",
        "export",
        "from",
        "class",
        "new",
        "this",
        "try",
        "catch",
        "finally",
        "throw",
        "async",
        "await",
        "true",
        "false",
        "null",
        "undefined",
        "interface",
        "type",
        "enum",
        "implements",
        "extends",
    },
}


class PatternAnalyzer:
    """Analyzes patterns and variations in duplicate code."""

    def __init__(self) -> None:
        """Initialize the pattern analyzer."""
        self.logger = get_logger("deduplication.analyzer")

    def _compare_literal_maps(
        self,
        pos_map1: Dict[Tuple[int, int], Dict[str, Any]],
        pos_map2: Dict[Tuple[int, int], Dict[str, Any]],
        literal_type: str,
    ) -> List[Dict[str, Any]]:
        all_positions = set(pos_map1.keys()) | set(pos_map2.keys())
        result = []
        for pos in sorted(all_positions):
            lit1, lit2 = pos_map1.get(pos), pos_map2.get(pos)
            if lit1 and lit2 and lit1["value"] != lit2["value"]:
                result.append(
                    {
                        "position": pos[0] + 1,
                        "column": pos[1],
                        "value1": lit1["value"],
                        "value2": lit2["value"],
                        "literal_type": literal_type,
                    }
                )
        return result

    def _extract_literal_maps(self, code: str, language: str) -> Dict[str, Dict[Tuple[int, int], Dict[str, Any]]]:
        """Extract per-type literal position maps for a code snippet.

        Args:
            code: Code snippet to scan
            language: Programming language

        Returns:
            Mapping of literal type to {(line, column): literal} position map
        """
        return {
            literal_type: {
                (lit["line"], lit["column"]): lit
                for lit in self._extract_literals_with_ast_grep(code, literal_type, language)
            }
            for literal_type in self._LITERAL_TYPES
        }

    def identify_varying_literals(
        self,
        code1: str,
        code2: str,
        language: str = "python",
        code1_literal_maps: Optional[Dict[str, Dict[Tuple[int, int], Dict[str, Any]]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Identify varying literal values between two similar code blocks.

        Args:
            code1: First code snippet
            code2: Second code snippet
            language: Programming language
            code1_literal_maps: Optional pre-extracted literal maps for code1
                (from ``_extract_literal_maps``) to avoid re-parsing it

        Returns:
            List of varying literals with positions and values
        """
        self.logger.info("identifying_varying_literals", language=language)

        varying_literals = []

        maps1 = code1_literal_maps if code1_literal_maps is not None else self._extract_literal_maps(code1, language)
        maps2 = self._extract_literal_maps(code2, language)
        for literal_type in self._LITERAL_TYPES:
            varying_literals.extend(self._compare_literal_maps(maps1[literal_type], maps2[literal_type], literal_type))

        varying_literals.sort(key=lambda x: (x["position"], x.get("column", 0)))
        self.logger.info("varying_literals_found", count=len(varying_literals))
        return varying_literals

    _LITERAL_TYPES: Tuple[str, ...] = ("string", "number", "boolean")

    _LITERAL_EXT_MAP: Dict[str, str] = {
        "python": ".py",
        "javascript": ".js",
        "typescript": ".ts",
        "jsx": ".jsx",
        "tsx": ".tsx",
        "java": ".java",
        "csharp": ".cs",
        "cpp": ".cpp",
        "c": ".c",
    }

    _LITERAL_RULES: Dict[str, Dict[str, Any]] = {
        "number": {"rule": {"any": [{"kind": "integer"}, {"kind": "float"}, {"kind": "number"}]}},
        "string": {"rule": {"any": [{"kind": "string"}, {"kind": "string_literal"}]}},
        "boolean": {"rule": {"any": [{"kind": "true"}, {"kind": "false"}, {"kind": "none"}, {"kind": "null"}]}},
    }

    def _parse_literal_matches(self, stdout: str, literal_type: str) -> List[Dict[str, Any]]:
        """Parse ast-grep JSON output into literal match dicts."""
        try:
            matches = json.loads(stdout)
        except json.JSONDecodeError:
            self.logger.warning("literal_parse_error", literal_type=literal_type)
            return []
        return [
            {
                "line": m.get("range", {}).get("start", {}).get("line", 0),
                "column": m.get("range", {}).get("start", {}).get("column", 0),
                "value": m.get("text", ""),
                "type": literal_type,
            }
            for m in matches
        ]

    def _scan_with_kind_rule(self, temp_path: str, rule: Dict[str, Any], language: str) -> Optional[str]:
        """Run ast-grep scan with rule as an inline rule; return JSON stdout, or None on failure."""
        inner = _kind_rule_for_language(rule, language)
        if inner is None:
            self.logger.warning("ast_grep_scan_no_valid_kinds", language=language)
            return None
        rule_yaml = yaml.dump({"id": _INLINE_RULE_ID, "language": language, "rule": inner})
        result = subprocess.run(
            ["ast-grep", "scan", "--inline-rules", rule_yaml, "--json", temp_path],
            capture_output=True,
            text=True,
            timeout=SubprocessDefaults.GREP_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            self.logger.warning("ast_grep_scan_failed", language=language, returncode=result.returncode, stderr=result.stderr.strip())
            return None
        return result.stdout

    def _run_literal_scan(self, temp_path: str, rule: Dict[str, Any], literal_type: str, language: str) -> List[Dict[str, Any]]:
        try:
            stdout = self._scan_with_kind_rule(temp_path, rule, language)
            if stdout and stdout.strip():
                return self._parse_literal_matches(stdout, literal_type)
            return []
        except subprocess.TimeoutExpired:
            self.logger.warning("literal_extraction_timeout", literal_type=literal_type)
            return []
        except Exception as e:
            self.logger.error("literal_extraction_error", error=str(e), literal_type=literal_type)
            return []

    def _extract_literals_with_ast_grep(self, code: str, literal_type: str, language: str) -> List[Dict[str, Any]]:
        """Extract literals from code using ast-grep."""
        rule = self._LITERAL_RULES.get(literal_type)
        if rule is None:
            return []
        ext = self._LITERAL_EXT_MAP.get(language, ".py")
        with tempfile.NamedTemporaryFile(mode="w", suffix=ext, delete=False) as f:
            f.write(code)
            temp_path = f.name
        try:
            return self._run_literal_scan(temp_path, rule, literal_type, language)
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass

    def analyze_duplicate_group_literals(self, group: List[Dict[str, Any]], language: str = "python") -> Dict[str, Any]:
        """
        Analyze a group of duplicates to find all varying literals.

        Args:
            group: List of duplicate code matches
            language: Programming language

        Returns:
            Analysis results with variations and suggestions
        """
        if len(group) < 2:
            return {"total_variations": 0, "variations": [], "suggested_parameters": []}

        base_code = group[0].get("text", "")
        all_variations = self._accumulate_literal_variations(base_code, group[1:], language)
        formatted_variations, suggested_parameters = self._format_literal_variations(all_variations)
        return {
            "total_variations": len(formatted_variations),
            "variations": formatted_variations,
            "suggested_parameters": suggested_parameters,
        }

    def _accumulate_one_item(
        self,
        base_code: str,
        code: str,
        all_variations: Dict[tuple[Any, ...], List[str]],
        language: str,
        base_literal_maps: Optional[Dict[str, Dict[Tuple[int, int], Dict[str, Any]]]] = None,
    ) -> None:
        for var in self.identify_varying_literals(base_code, code, language, code1_literal_maps=base_literal_maps):
            key = (var["position"], var.get("column", 0), var["literal_type"])
            vals = all_variations.setdefault(key, [var["value1"]])
            if var["value2"] not in vals:
                vals.append(var["value2"])

    def _accumulate_literal_variations(
        self, base_code: str, items: List[Dict[str, Any]], language: str
    ) -> Dict[tuple[Any, ...], List[str]]:
        all_variations: Dict[tuple[Any, ...], List[str]] = {}
        base_literal_maps: Optional[Dict[str, Dict[Tuple[int, int], Dict[str, Any]]]] = None
        for item in items:
            code = item.get("text", "")
            if code != base_code:
                if base_literal_maps is None:
                    base_literal_maps = self._extract_literal_maps(base_code, language)
                self._accumulate_one_item(base_code, code, all_variations, language, base_literal_maps)
        return all_variations

    def _format_literal_variations(self, all_variations: Dict[tuple[Any, ...], List[str]]) -> tuple[List[Dict[str, Any]], List[str]]:
        formatted_variations: List[Dict[str, Any]] = []
        suggested_parameters: List[str] = []
        for (line, col, lit_type), values in sorted(all_variations.items()):
            formatted_variations.append(
                {
                    "position": {"line": line, "column": col},
                    "type": lit_type,
                    "values": values,
                    "unique_count": len(set(values)),
                }
            )
            param_name = self._suggest_parameter_name(lit_type, values[0])
            if param_name and param_name not in suggested_parameters:
                suggested_parameters.append(param_name)
        return formatted_variations, suggested_parameters

    def classify_variation(self, variation_type: str, old_value: str, new_value: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Classify a single variation between duplicate code blocks.

        Args:
            variation_type: Type hint from detector
            old_value: Original value
            new_value: New value
            context: Optional surrounding context

        Returns:
            Classification with category, severity, and metadata
        """
        category = self._determine_category(variation_type, old_value, new_value)
        severity = self._determine_severity(category, old_value, new_value)
        complexity = self._calculate_variation_complexity(category, severity, old_value, new_value)

        return {
            "category": category,
            "severity": severity,
            "old_value": old_value,
            "new_value": new_value,
            "context": context,
            "parameterizable": severity in [VariationSeverity.LOW, VariationSeverity.MEDIUM],
            "suggested_param_name": self._suggest_parameter_name(category, old_value),
            "complexity": complexity,
        }

    def _classify_difficulty(self, avg_complexity: float) -> str:
        if avg_complexity <= 2:
            return "trivial"
        if avg_complexity <= DifficultyThresholds.SIMPLE:
            return "simple"
        if avg_complexity <= DifficultyThresholds.MODERATE:
            return "moderate"
        if avg_complexity <= DifficultyThresholds.COMPLEX:
            return "complex"
        return "very_complex"

    def _aggregate_classifications(self, variations: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], int, int, List[str]]:
        classifications: List[Dict[str, Any]] = []
        total_complexity = 0
        parameterizable_count = 0
        param_suggestions: List[str] = []
        for var in variations:
            c = self.classify_variation(var.get("type", "unknown"), var.get("old_value", ""), var.get("new_value", ""), var.get("context"))
            classifications.append(c)
            total_complexity += c["complexity"]["score"]
            if c["parameterizable"]:
                parameterizable_count += 1
            name = c["suggested_param_name"]
            if name and name not in param_suggestions:
                param_suggestions.append(name)
        return classifications, total_complexity, parameterizable_count, param_suggestions

    def classify_variations(self, variations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Classify multiple variations and determine overall complexity.

        Args:
            variations: List of variations to classify

        Returns:
            Overall classification with complexity metrics
        """
        if not variations:
            return {
                "total_variations": 0,
                "complexity_score": 0,
                "refactoring_difficulty": "trivial",
                "classifications": [],
                "parameterizable_count": 0,
                "parameter_suggestions": [],
            }

        classifications, total_complexity, parameterizable_count, param_suggestions = self._aggregate_classifications(variations)
        avg_complexity = total_complexity / len(variations)

        return {
            "total_variations": len(variations),
            "complexity_score": round(avg_complexity, 2),
            "refactoring_difficulty": self._classify_difficulty(avg_complexity),
            "classifications": classifications,
            "parameterizable_count": parameterizable_count,
            "parameter_suggestions": param_suggestions[: SemanticVolumeDefaults.TOP_RESULTS_LIMIT],
        }

    _VARIATION_TYPE_MAP: Dict[str, str] = {
        **{k: VariationCategory.LITERAL for k in ("literal", "string", "number", "boolean")},
        **{k: VariationCategory.IDENTIFIER for k in ("identifier", "name", "variable", "function", "class")},
        **{k: VariationCategory.TYPE for k in ("type", "annotation", "type_hint")},
        **{k: VariationCategory.EXPRESSION for k in ("expression", "operator", "call")},
        **{k: VariationCategory.LOGIC for k in ("logic", "control", "flow", "condition")},
    }

    def _determine_category(self, variation_type: str, old_value: str, new_value: str) -> str:
        """Determine the category of a variation."""
        category = self._VARIATION_TYPE_MAP.get(variation_type.lower())
        if category is not None:
            return category
        return self._infer_category_from_content(old_value, new_value)

    _CONTROL_KEYWORDS = frozenset({"if", "else", "elif", "for", "while", "switch", "case", "try", "catch", "except"})
    _EXPRESSION_INDICATORS = frozenset({"(", ")", "+", "-", "*", "/", "==", "!=", "&&", "||", "and", "or"})

    def _matches_any_keyword(self, old_low: str, new_low: str, keywords: frozenset[str]) -> bool:
        return any(kw in old_low or kw in new_low for kw in keywords)

    def _infer_category_from_content(self, old_value: str, new_value: str) -> str:
        """Infer category from content when type hint is unavailable."""
        if self._is_literal(old_value) and self._is_literal(new_value):
            return VariationCategory.LITERAL
        if self._is_type_annotation(old_value) or self._is_type_annotation(new_value):
            return VariationCategory.TYPE
        old_low, new_low = old_value.lower(), new_value.lower()
        if self._matches_any_keyword(old_low, new_low, self._CONTROL_KEYWORDS):
            return VariationCategory.LOGIC
        if self._matches_any_keyword(old_value, new_value, self._EXPRESSION_INDICATORS):
            return VariationCategory.EXPRESSION
        return VariationCategory.IDENTIFIER

    _BOOLEAN_LITERALS = frozenset({"true", "false", "none", "null", "nil"})

    def _is_literal(self, value: str) -> bool:
        """Check if a value appears to be a literal."""
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            return True
        if value.lower() in self._BOOLEAN_LITERALS:
            return True
        try:
            float(value)
            return True
        except ValueError:
            return False

    def _is_type_annotation(self, value: str) -> bool:
        """Check if a value appears to be a type annotation."""
        type_indicators = [
            "->",
            ":",
            "Optional[",
            "List[",
            "Dict[",
            "Tuple[",
            "Set[",
            "Union[",
            "Any",
            "int",
            "str",
            "float",
            "bool",
            "None",
            "<",
            ">",  # Generics in other languages
        ]
        return any(ind in value for ind in type_indicators)

    def _severity_identifier(self, old_value: str, new_value: str) -> str:
        old_words = len(old_value.split("_")) + len(old_value.split())
        new_words = len(new_value.split("_")) + len(new_value.split())
        return VariationSeverity.MEDIUM if abs(old_words - new_words) > 2 else VariationSeverity.LOW

    def _severity_expression(self, old_value: str, new_value: str) -> str:
        if "(" in old_value and "(" in new_value:
            different_fn = old_value.split("(")[0] != new_value.split("(")[0]
            return VariationSeverity.HIGH if different_fn else VariationSeverity.MEDIUM
        return VariationSeverity.MEDIUM

    def _severity_type(self, old_value: str, new_value: str) -> str:
        return VariationSeverity.MEDIUM if "[" in old_value or "[" in new_value else VariationSeverity.LOW

    def _determine_severity(self, category: str, old_value: str, new_value: str) -> str:
        """Determine the severity of a variation."""
        severity_fns: Dict[str, Callable[[], str]] = {
            VariationCategory.LITERAL: lambda: VariationSeverity.LOW,
            VariationCategory.IDENTIFIER: lambda: self._severity_identifier(old_value, new_value),
            VariationCategory.TYPE: lambda: self._severity_type(old_value, new_value),
            VariationCategory.EXPRESSION: lambda: self._severity_expression(old_value, new_value),
            VariationCategory.LOGIC: lambda: VariationSeverity.HIGH,
        }
        fn = severity_fns.get(category)
        return fn() if fn else VariationSeverity.MEDIUM

    _PARAM_NAME_CATEGORY_MAP: Dict[str, Optional[str]] = {
        VariationCategory.IDENTIFIER: "name",
        VariationCategory.TYPE: "type_param",
        VariationCategory.EXPRESSION: "expression",
        VariationCategory.LOGIC: None,
    }

    def _suggest_literal_param_name(self, value: str) -> str:
        val_low = value.lower()
        if any(kw in val_low for kw in ("url", "path", "file")):
            return "target_path"
        if any(kw in val_low for kw in ("name", "id", "key")):
            return "identifier"
        if any(char.isdigit() for char in value):
            return "value"
        return "text_value"

    def _suggest_parameter_name(self, category: str, value: str) -> Optional[str]:
        """Suggest a parameter name for a variation."""
        if category == VariationCategory.LITERAL:
            return self._suggest_literal_param_name(value)
        return self._PARAM_NAME_CATEGORY_MAP.get(category)

    def _score_literal_variation(self) -> tuple[int, str]:
        """Score literal category variation."""
        return 1, "Simple value substitution"

    def _score_identifier_variation(self, severity: str) -> tuple[int, str]:
        """Score identifier category variation."""
        if severity == VariationSeverity.LOW:
            return 1, "Simple identifier rename"
        return 2, "Identifier with semantic differences"

    def _score_expression_variation(self, severity: str) -> tuple[int, str]:
        """Score expression category variation."""
        if severity == VariationSeverity.LOW:
            return int(VariationScoreScale.SCORE_3), "Minor expression variation"
        if severity == VariationSeverity.MEDIUM:
            return int(VariationScoreScale.SCORE_4), "Different operations or function calls"
        return int(VariationScoreScale.SCORE_4), "Significant expression restructuring"

    def _score_type_variation(self, severity: str) -> tuple[int, str]:
        """Score type category variation."""
        if severity == VariationSeverity.LOW:
            return int(VariationScoreScale.SCORE_4), "Simple type substitution"
        if severity == VariationSeverity.MEDIUM:
            return int(VariationScoreScale.SCORE_5), "Generic type variation"
        return int(VariationScoreScale.SCORE_6), "Complex type system changes"

    def _score_logic_variation(self, severity: str, old_value: str, new_value: str) -> tuple[int, str]:
        """Score logic category variation."""
        if "inserted" in str(old_value) or "deleted" in str(new_value):
            return int(VariationScoreScale.SCORE_7), "Added or removed logic branches"
        if severity == VariationSeverity.HIGH:
            return int(VariationScoreScale.SCORE_7), "Significant control flow differences"
        return int(VariationScoreScale.SCORE_5), "Conditional logic variation"

    def _get_complexity_level(self, score: int) -> str:
        """Get complexity level from score."""
        if score <= 2:
            return "trivial"
        if score <= int(VariationScoreCutoff.MODERATE_MAX):
            return "moderate"
        return "complex"

    def _calculate_variation_complexity(self, category: str, severity: str, old_value: str, new_value: str) -> Dict[str, Any]:
        """
        Calculate complexity score for a variation.

        Scores range from 1-7:
        - 1: Trivial (literal substitution)
        - 2-3: Simple (identifier/expression)
        - 4-5: Moderate (type/complex expression)
        - 6-7: Complex (logic/structure)
        """
        # Use a mapping to handle category scoring
        scorers: Dict[str, Callable[[], Tuple[int, str]]] = {
            VariationCategory.LITERAL: lambda: self._score_literal_variation(),
            VariationCategory.IDENTIFIER: lambda: self._score_identifier_variation(severity),
            VariationCategory.EXPRESSION: lambda: self._score_expression_variation(severity),
            VariationCategory.TYPE: lambda: self._score_type_variation(severity),
            VariationCategory.LOGIC: lambda: self._score_logic_variation(severity, old_value, new_value),
        }

        # Get score and reasoning for the category
        scorer = scorers.get(category)
        if scorer:
            score, reasoning = scorer()
        else:
            score, reasoning = int(VariationScoreScale.SCORE_3), "Unknown variation type"

        # Determine complexity level
        level = self._get_complexity_level(score)

        return {"score": score, "level": level, "reasoning": reasoning}

    def _compare_conditional_pair(
        self, i: int, cond1: Optional[Dict[str, Any]], cond2: Optional[Dict[str, Any]], language: str
    ) -> Optional[Dict[str, Any]]:
        if cond1 and cond2:
            if cond1["text"] == cond2["text"]:
                return None
            v: Dict[str, Any] = {
                "position": i + 1,
                "type": "modified",
                "condition1": cond1["text"],
                "condition2": cond2["text"],
                "line1": cond1.get("line", 0),
                "line2": cond2.get("line", 0),
            }
            v["details"] = self._analyze_conditional_difference(cond1["text"], cond2["text"], language)
            return v
        if cond1:
            return {"position": i + 1, "type": "removed", "condition1": cond1["text"], "condition2": None, "line1": cond1.get("line", 0)}
        if cond2:
            return {"position": i + 1, "type": "added", "condition1": None, "condition2": cond2["text"], "line2": cond2.get("line", 0)}
        return None

    def detect_conditional_variations(self, code1: str, code2: str, language: str = "python") -> List[Dict[str, Any]]:
        """
        Detect variations in conditional statements between two code blocks.

        Analyzes if/else conditions, comparison operators, and conditional values
        to identify differences that could be parameterized.

        Args:
            code1: First code snippet
            code2: Second code snippet
            language: Programming language (default: python)

        Returns:
            List of conditional variations with details about the differences
        """
        self.logger.info("detecting_conditional_variations", language=language)

        conditionals1 = self._extract_conditionals(code1, language)
        conditionals2 = self._extract_conditionals(code2, language)
        max_len = max(len(conditionals1), len(conditionals2), 0)

        variations: List[Dict[str, Any]] = []
        for i in range(max_len):
            c1 = conditionals1[i] if i < len(conditionals1) else None
            c2 = conditionals2[i] if i < len(conditionals2) else None
            v = self._compare_conditional_pair(i, c1, c2, language)
            if v is not None:
                variations.append(v)

        self.logger.info("conditional_variations_found", count=len(variations))
        return variations

    _CONDITIONAL_RULE_PYTHON = {"rule": {"any": [{"kind": "if_statement"}, {"kind": "elif_clause"}, {"kind": "comparison_operator"}]}}
    _CONDITIONAL_RULE_OTHER = {"rule": {"any": [{"kind": "if_statement"}, {"kind": "binary_expression"}, {"kind": "ternary_expression"}]}}
    _CONDITIONAL_EXT_MAP = {"python": ".py", "javascript": ".js", "typescript": ".ts", "java": ".java"}

    def _run_conditional_scan(self, temp_path: str, rule: Dict[str, Any], code: str, language: str) -> List[Dict[str, Any]]:
        try:
            stdout = self._scan_with_kind_rule(temp_path, rule, language)
            if stdout is None:
                return self._extract_conditionals_regex(code, language)
            if stdout.strip():
                return self._parse_conditional_matches(stdout, language)
            return []
        except subprocess.TimeoutExpired:
            self.logger.warning("conditional_extraction_timeout", language=language)
            return []
        except FileNotFoundError:
            return self._extract_conditionals_regex(code, language)
        except Exception as e:
            self.logger.error("conditional_extraction_error", error=str(e))
            return self._extract_conditionals_regex(code, language)

    def _extract_conditionals(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Extract conditional statements from code using ast-grep."""
        ext = self._CONDITIONAL_EXT_MAP.get(language, ".py")
        rule = self._CONDITIONAL_RULE_PYTHON if language == "python" else self._CONDITIONAL_RULE_OTHER
        with tempfile.NamedTemporaryFile(mode="w", suffix=ext, delete=False) as f:
            f.write(code)
            temp_path = f.name
        try:
            return self._run_conditional_scan(temp_path, rule, code, language)
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass

    def _parse_conditional_matches(self, stdout: str, language: str) -> List[Dict[str, Any]]:
        """Parse ast-grep JSON output into conditional match dicts."""
        try:
            matches = json.loads(stdout)
        except json.JSONDecodeError:
            self.logger.warning("conditional_parse_error", language=language)
            return []
        return [
            {
                "line": m.get("range", {}).get("start", {}).get("line", 0),
                "text": m.get("text", "").strip(),
                "kind": m.get("kind", "unknown"),
            }
            for m in matches
        ]

    def _extract_conditionals_regex(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Fallback regex-based extraction for conditionals."""
        import re

        pattern = r"(?:if|elif)\s+(.+?):" if language == "python" else r"(?:if|else\s+if)\s*\((.+?)\)"
        return [
            {"line": i, "text": m.group(0).strip(), "kind": "if_statement"}
            for i, line in enumerate(code.split("\n"), 1)
            for m in [re.search(pattern, line)]
            if m
        ]

    def _get_language_extension(self, language: str) -> str:
        """Get file extension for a programming language."""
        ext_map = {
            "python": ".py",
            "javascript": ".js",
            "typescript": ".ts",
            "java": ".java",
        }
        return ext_map.get(language, ".py")

    def _get_call_rule(self, language: str) -> Dict[str, Any]:
        """Get ast-grep rule for call expressions based on language."""
        if language == "python":
            return {"rule": {"kind": "call"}}
        return {"rule": {"kind": "call_expression"}}

    def _build_nested_call_result(self, match: Dict[str, Any], identifier: str, nesting_depth: int) -> Dict[str, Any]:
        return {
            "identifier": identifier,
            "nesting_depth": nesting_depth,
            "call_expression": match.get("text", ""),
            "line": match.get("range", {}).get("start", {}).get("line", 0) + 1,
        }

    def _find_nested_call_in_matches(self, matches: List[Dict[str, Any]], identifier: str) -> Optional[Dict[str, Any]]:
        """Find nested function call containing identifier in ast-grep matches."""
        for match in matches:
            text = match.get("text", "")
            if identifier not in text:
                continue
            depth = self._calculate_call_nesting_depth(text, identifier)
            if depth > 1:
                return self._build_nested_call_result(match, identifier, depth)
        return None

    def _run_ast_grep_for_calls(self, temp_path: str, language: str, identifier: str) -> Optional[Dict[str, Any]]:
        """Run ast-grep to find nested function calls."""
        stdout = self._scan_with_kind_rule(temp_path, self._get_call_rule(language), language)
        if not stdout or not stdout.strip():
            return None

        try:
            matches = json.loads(stdout)
            return self._find_nested_call_in_matches(matches, identifier)
        except json.JSONDecodeError:
            self.logger.warning("nested_call_parse_error", language=language)
            return None

    def _run_nested_call_detection(self, temp_path: str, code: str, identifier: str, language: str) -> Optional[Dict[str, Any]]:
        try:
            return self._run_ast_grep_for_calls(temp_path, language, identifier)
        except subprocess.TimeoutExpired:
            self.logger.warning("nested_call_detection_timeout", language=language)
            return None
        except FileNotFoundError:
            return self._detect_nested_call_regex(code, identifier)
        except Exception as e:
            self.logger.error("nested_call_detection_error", error=str(e))
            return None

    def detect_nested_function_call(self, code: str, identifier: str, language: str = "python") -> Optional[Dict[str, Any]]:
        """
        Detect if an identifier is used within nested function calls.

        Args:
            code: Source code to analyze
            identifier: The identifier to look for
            language: Programming language

        Returns:
            Dict with nesting info if found, None otherwise
        """
        self.logger.info("detecting_nested_function_call", identifier=identifier, language=language)
        ext = self._get_language_extension(language)
        with tempfile.NamedTemporaryFile(mode="w", suffix=ext, delete=False) as f:
            f.write(code)
            temp_path = f.name
        try:
            return self._run_nested_call_detection(temp_path, code, identifier, language)
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass

    def _calculate_call_nesting_depth(self, expression: str, identifier: str) -> int:
        """Calculate the nesting depth of function calls around an identifier."""
        # Count opening parentheses before the identifier
        idx = expression.find(identifier)
        if idx == -1:
            return 0

        depth = 0
        max_depth = 0
        for i, char in enumerate(expression[:idx]):
            if char == "(":
                depth += 1
                max_depth = max(max_depth, depth)
            elif char == ")":
                depth -= 1

        return depth

    def _detect_nested_call_regex(self, code: str, identifier: str) -> Optional[Dict[str, Any]]:
        """Fallback regex-based detection for nested function calls."""
        import re

        pattern = r"(\w+)\s*\(\s*(\w+)\s*\([^)]*" + re.escape(identifier) + r"[^)]*\)"
        hit = next(
            ((i, m) for i, line in enumerate(code.split("\n"), 1) for m in [re.search(pattern, line)] if m),
            None,
        )
        if hit is None:
            return None
        i, m = hit
        return {
            "identifier": identifier,
            "nesting_depth": 2,
            "call_expression": m.group(0),
            "line": i,
            "outer_function": m.group(RegexCaptureGroups.FIRST),
            "inner_function": m.group(RegexCaptureGroups.SECOND),
        }

    # Multi-word operators first; word operators need boundaries so "index" is not "in"
    _COMPARISON_OPERATOR_RE = re.compile(r"\bnot\s+in\b|\bis\s+not\b|\bin\b|\bis\b|>=|<=|!=|==|>|<")
    _COND_KEYWORD_EXCLUSIONS = frozenset({"if", "elif", "else"})

    def _find_operator(self, cond: str) -> Optional[str]:
        match = self._COMPARISON_OPERATOR_RE.search(cond)
        return " ".join(match.group(0).split()) if match else None

    def _extract_cond_vars(self, cond: str) -> set[str]:
        cond = self._COMPARISON_OPERATOR_RE.sub(" ", cond)
        return set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", cond)) - self._COND_KEYWORD_EXCLUSIONS

    def _analyze_conditional_difference(self, cond1: str, cond2: str, language: str) -> Dict[str, Any]:
        """Analyze the specific differences between two conditional expressions."""
        import re

        details: Dict[str, Any] = {"operator_changed": False, "value_changed": False, "variable_changed": False, "comparison_values": {}}

        op1, op2 = self._find_operator(cond1), self._find_operator(cond2)
        if op1 and op2 and op1 != op2:
            details["operator_changed"] = True
            details["operators"] = {"from": op1, "to": op2}

        nums1 = re.findall(r"\b\d+(?:\.\d+)?\b", cond1)
        nums2 = re.findall(r"\b\d+(?:\.\d+)?\b", cond2)
        if nums1 != nums2:
            details["value_changed"] = True
            details["comparison_values"] = {"from": nums1, "to": nums2}

        vars1, vars2 = self._extract_cond_vars(cond1), self._extract_cond_vars(cond2)
        if vars1 != vars2:
            details["variable_changed"] = True
            details["variables"] = {"from": list(vars1), "to": list(vars2)}

        return details


# Standalone function for backward compatibility
def detect_conditional_variations(code1: str, code2: str, language: str = "python") -> List[Dict[str, Any]]:
    """
    Detect variations in conditional statements between two code blocks.

    This is a convenience function that creates a PatternAnalyzer instance
    and calls its detect_conditional_variations method.

    Args:
        code1: First code snippet
        code2: Second code snippet
        language: Programming language (default: python)

    Returns:
        List of conditional variations with details about the differences
    """
    analyzer = PatternAnalyzer()
    return analyzer.detect_conditional_variations(code1, code2, language)


def _detect_nested_function_call(code: str, identifier: str, language: str = "python") -> Optional[Dict[str, Any]]:
    """
    Detect if an identifier is used within nested function calls.

    This is a convenience function that creates a PatternAnalyzer instance
    and calls its detect_nested_function_call method.

    Args:
        code: Source code to analyze
        identifier: The identifier to look for
        language: Programming language (default: python)

    Returns:
        Dict with nesting info if found, None otherwise
    """
    analyzer = PatternAnalyzer()
    return analyzer.detect_nested_function_call(code, identifier, language)


def _collect_all_variations(analyzer: "PatternAnalyzer", code1: str, code2: str, language: str) -> List[Dict[str, Any]]:
    literal_variations = analyzer.identify_varying_literals(code1, code2, language)
    cond_variations = analyzer.detect_conditional_variations(code1, code2, language)
    identifier_variations = identify_varying_identifiers(code1, code2, language)
    return (
        [
            {"type": "literal", "old_value": v.get("value1", ""), "new_value": v.get("value2", ""), "context": v.get("context", "")}
            for v in literal_variations
        ]
        + [
            {
                "type": "conditional",
                "old_value": v.get("condition1", ""),
                "new_value": v.get("condition2", ""),
                "context": f"line {v.get('line1', 0)}",
            }
            for v in cond_variations
        ]
        + [
            {
                "type": "identifier",
                "old_value": v.get("identifier1", ""),
                "new_value": v.get("identifier2", ""),
                "context": v.get("context", ""),
            }
            for v in identifier_variations
        ]
    )


def _determine_overall_severity(classification: Dict[str, Any], variation_count: int) -> str:
    classes = classification.get("classifications", [])
    high_count = sum(1 for v in classes if v.get("severity") == VariationSeverity.HIGH)
    medium_count = sum(1 for v in classes if v.get("severity") == VariationSeverity.MEDIUM)
    if high_count > 0:
        return VariationSeverity.HIGH
    if medium_count > variation_count // 2:
        return VariationSeverity.MEDIUM
    return VariationSeverity.LOW


def classify_variations(code1: str, code2: str, language: str = "python") -> Dict[str, Any]:
    """
    Classify variations between two code snippets.

    Compares two code blocks and identifies the types and severity of
    differences between them.

    Args:
        code1: First code snippet
        code2: Second code snippet
        language: Programming language (default: python)

    Returns:
        Dictionary containing:
        - severity: Overall severity level (low, medium, high)
        - variations: List of specific variations found
        - parameterizable: Whether differences can be parameterized
        - suggested_refactoring: Recommended refactoring approach
    """
    analyzer = PatternAnalyzer()
    variations = _collect_all_variations(analyzer, code1, code2, language)

    if not variations:
        return {
            "severity": VariationSeverity.LOW,
            "variations": [],
            "parameterizable": True,
            "suggested_refactoring": "extract_function",
            "complexity_score": 0,
            "parameter_suggestions": [],
        }

    classification = analyzer.classify_variations(variations)
    overall_severity = _determine_overall_severity(classification, len(variations))
    return {
        "severity": overall_severity,
        "variations": variations,
        "parameterizable": classification.get("parameterizable_count", 0) > 0,
        "suggested_refactoring": "extract_function" if overall_severity != VariationSeverity.HIGH else "manual_review",
        "complexity_score": classification.get("complexity_score", 0),
        "parameter_suggestions": classification.get("parameter_suggestions", []),
    }


def identify_varying_identifiers(code1: str, code2: str, language: str = "python") -> List[Dict[str, Any]]:
    """
    Identify identifiers that vary between two code snippets.

    Compares identifier usage in two code blocks to find names that
    differ but serve similar purposes (e.g., 'user_id' vs 'order_id').

    Args:
        code1: First code snippet
        code2: Second code snippet
        language: Programming language (default: python)

    Returns:
        List of varying identifier pairs with context
    """
    varying_identifiers: List[Dict[str, Any]] = []

    # Extract identifiers from both code blocks
    identifiers1 = _extract_identifiers_from_code(code1, language)
    identifiers2 = _extract_identifiers_from_code(code2, language)

    varying_identifiers = [
        {
            "identifier1": ident1["name"],
            "identifier2": identifiers2[pos]["name"],
            "position": pos,
            "context": ident1.get("context", ""),
            "usage_type": ident1.get("usage_type", "unknown"),
        }
        for pos, ident1 in identifiers1.items()
        if pos in identifiers2 and ident1["name"] != identifiers2[pos]["name"]
    ]

    return varying_identifiers


def _get_usage_type(context: str, name: str) -> str:
    if "def " in context or "function " in context:
        return "function"
    if "class " in context:
        return "class"
    if "(" in context[context.find(name) : context.find(name) + len(name) + 2]:
        return "function_call"
    return "variable"


def _extract_identifiers_from_code(code: str, language: str) -> Dict[int, Dict[str, Any]]:
    """
    Extract identifiers from code with position information.

    Args:
        code: Source code to analyze
        language: Programming language

    Returns:
        Dictionary mapping position to identifier info
    """
    import re

    excluded = _LANGUAGE_KEYWORDS.get(language, _LANGUAGE_KEYWORDS["python"])
    identifier_pattern = r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b"
    identifiers: Dict[int, Dict[str, Any]] = {}

    for idx, match in enumerate(re.finditer(identifier_pattern, code)):
        name = match.group(RegexCaptureGroups.FIRST)
        if name in excluded or name.isupper():
            continue
        start = max(0, match.start() - IDENTIFIER_CONTEXT_WINDOW_CHARS)
        end = min(len(code), match.end() + IDENTIFIER_CONTEXT_WINDOW_CHARS)
        context = code[start:end].strip()
        identifiers[idx] = {
            "name": name,
            "position": match.start(),
            "context": context,
            "usage_type": _get_usage_type(context, name),
        }

    return identifiers
