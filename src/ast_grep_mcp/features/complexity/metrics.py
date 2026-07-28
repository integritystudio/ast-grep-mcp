"""
Code complexity metrics calculation.

This module provides functions for calculating various code complexity metrics:
- Cyclomatic complexity (McCabe)
- Cognitive complexity (SonarSource)
- Nesting depth
- Pattern-based complexity analysis

All metrics are calculated using ast-grep patterns and text-based analysis.
"""

import json
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple, cast

from ast_grep_mcp.constants import IndentationDefaults, SubprocessDefaults

# =============================================================================
# COMPLEXITY PATTERNS
# =============================================================================

COMPLEXITY_PATTERNS: Dict[str, Dict[str, Any]] = {
    "python": {
        "function": "def $NAME($$$)",
        "async_function": "async def $NAME($$$)",
        "branches": [
            "if $COND:",
            "elif $COND:",
            "for $VAR in $ITER:",
            "while $COND:",
            "except $TYPE:",
            "except:",
            "with $CTX:",
            "case $PATTERN:",
        ],
        "logical_operators": [
            "$A and $B",
            "$A or $B",
        ],
        "nesting_constructs": ["if", "for", "while", "with", "try", "match"],
    },
    "typescript": {
        "function": "function $NAME($$$) { $$$ }",
        "arrow_function": "const $NAME = ($$$) => { $$$ }",
        "method": "$NAME($$$) { $$$ }",
        "branches": [
            "if ($COND) { $$$ }",
            "for ($INIT; $COND; $INC) { $$$ }",
            "for ($VAR of $ITER) { $$$ }",
            "for ($VAR in $OBJ) { $$$ }",
            "while ($COND) { $$$ }",
            "switch ($EXPR) { $$$ }",
            "case $VAL:",
            "catch ($ERR) { $$$ }",
            "$COND ? $A : $B",
        ],
        "logical_operators": [
            "$A && $B",
            "$A || $B",
            "$A ?? $B",
        ],
        "nesting_constructs": ["if", "for", "while", "switch", "try"],
    },
    "javascript": {
        "function": "function $NAME($$$) { $$$ }",
        "arrow_function": "const $NAME = ($$$) => { $$$ }",
        "method": "$NAME($$$) { $$$ }",
        "branches": [
            "if ($COND) { $$$ }",
            "for ($INIT; $COND; $INC) { $$$ }",
            "for ($VAR of $ITER) { $$$ }",
            "for ($VAR in $OBJ) { $$$ }",
            "while ($COND) { $$$ }",
            "switch ($EXPR) { $$$ }",
            "case $VAL:",
            "catch ($ERR) { $$$ }",
            "$COND ? $A : $B",
        ],
        "logical_operators": [
            "$A && $B",
            "$A || $B",
            "$A ?? $B",
        ],
        "nesting_constructs": ["if", "for", "while", "switch", "try"],
    },
    "java": {
        "function": "$TYPE $NAME($$$) { $$$ }",
        "branches": [
            "if ($COND) { $$$ }",
            "for ($INIT; $COND; $INC) { $$$ }",
            "for ($TYPE $VAR : $ITER) { $$$ }",
            "while ($COND) { $$$ }",
            "switch ($EXPR) { $$$ }",
            "case $VAL:",
            "catch ($TYPE $VAR) { $$$ }",
            "$COND ? $A : $B",
        ],
        "logical_operators": [
            "$A && $B",
            "$A || $B",
        ],
        "nesting_constructs": ["if", "for", "while", "switch", "try"],
    },
}

# Language-specific keyword and operator configurations
CYCLOMATIC_CONFIG = {
    "python": {"keywords": ["if ", "elif ", "for ", "while ", "except ", "except:", "with ", "case "], "operators": [" and ", " or "]},
    "typescript": {
        "keywords": ["if ", "if(", "for ", "for(", "while ", "while(", "switch ", "switch(", "case ", "catch ", "catch(", "? "],
        "operators": [" && ", " || ", " ?? "],
    },
    "javascript": {
        "keywords": ["if ", "if(", "for ", "for(", "while ", "while(", "switch ", "switch(", "case ", "catch ", "catch(", "? "],
        "operators": [" && ", " || ", " ?? "],
    },
    "java": {
        "keywords": ["if ", "if(", "for ", "for(", "while ", "while(", "switch ", "switch(", "case ", "catch ", "catch("],
        "operators": [" && ", " || "],
    },
}


def get_complexity_patterns(language: str) -> Dict[str, Any]:
    """Get AST patterns for a specific language.

    Args:
        language: Programming language name

    Returns:
        Dictionary of patterns for the language
    """
    lang_lower = language.lower()
    if lang_lower in COMPLEXITY_PATTERNS:
        return COMPLEXITY_PATTERNS[lang_lower]
    # Default to Python patterns
    return COMPLEXITY_PATTERNS["python"]


def count_pattern_matches(code: str, pattern: str, language: str) -> int:
    """Count occurrences of an AST pattern in code using ast-grep.

    Args:
        code: Source code to analyze
        pattern: ast-grep pattern to search for
        language: Programming language

    Returns:
        Number of matches found
    """
    try:
        result = subprocess.run(
            ["ast-grep", "run", "--pattern", pattern, "--lang", language, "--json"],
            input=code,
            capture_output=True,
            text=True,
            timeout=SubprocessDefaults.GREP_TIMEOUT_SECONDS,
        )
        if result.returncode == 0 and result.stdout.strip():
            matches = json.loads(result.stdout)
            return len(matches) if isinstance(matches, list) else 0
        return 0
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        return 0


def _get_cyclomatic_config(language: str) -> Dict[str, List[str]]:
    """Get cyclomatic complexity configuration for a language.

    Args:
        language: Programming language name

    Returns:
        Dictionary with keywords and operators lists
    """
    lang_lower = language.lower()
    if lang_lower in CYCLOMATIC_CONFIG:
        return CYCLOMATIC_CONFIG[lang_lower]
    # Default to Python configuration
    return CYCLOMATIC_CONFIG["python"]


# Comments and string literals are prose, not control flow.  Counting keywords
# inside them inflated every metric here — a long docstring using the words
# "for", "if", "and" or "or" read as decision points, which measured
# utils/futures.py:map_with_per_item_timeout at cyclomatic 24 when its body has
# 9 (14 of the 23 counted points came from the docstring alone).  Blanking
# happens before any counting, and preserves line count and indentation so the
# line-oriented cognitive and nesting passes still see the real code layout.
_TRIPLE_QUOTED = re.compile(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'')
_BLOCK_COMMENT = re.compile(r"/\*[\s\S]*?\*/")
_SINGLE_QUOTED = re.compile(r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'')
_LINE_COMMENT = re.compile(r"(#|//).*$", re.MULTILINE)


def _blank_preserving_newlines(match: "re.Match[str]") -> str:
    """Replace a matched span with its newlines, dropping all other characters."""
    return "\n" * match.group(0).count("\n")


def strip_non_code(code: str) -> str:
    """Blank out comments and string literals so only real code is counted.

    Multi-line spans collapse to their newlines and single-line spans to empty
    quotes, keeping every remaining code line at its original index and
    indentation.

    This is deliberately lexical rather than AST-based: the surrounding metrics
    are keyword heuristics applied to arbitrary extracted snippets, which are
    frequently not parseable on their own (a dedented method body, a fragment
    cut at a line boundary). Pathological cases — a quote inside a comment, a
    triple quote inside a single-quoted string — can still mis-span, so treat
    the output as a better estimate, not a guarantee.

    Args:
        code: Source code, possibly a fragment.

    Returns:
        The code with comment and string-literal content removed.
    """
    without_blocks = _TRIPLE_QUOTED.sub(_blank_preserving_newlines, code)
    without_blocks = _BLOCK_COMMENT.sub(_blank_preserving_newlines, without_blocks)
    # Strings before line comments: a '#' or '//' inside a string literal is
    # not a comment, and the literal is already gone by this point.
    without_strings = _SINGLE_QUOTED.sub('""', without_blocks)
    return _LINE_COMMENT.sub("", without_strings)


def _count_occurrences(code: str, items: List[str]) -> int:
    """Count occurrences of items in code.

    Args:
        code: Source code text
        items: List of strings to count

    Returns:
        Total count of all items
    """
    total = 0
    for item in items:
        total += code.count(item)
    return total


def calculate_cyclomatic_complexity(code: str, language: str) -> int:
    """Calculate McCabe cyclomatic complexity.

    Cyclomatic complexity = E - N + 2P
    Simplified: 1 + number of decision points

    Args:
        code: Function source code
        language: Programming language

    Returns:
        Cyclomatic complexity score (minimum 1)
    """
    complexity = 1  # Base complexity

    # Get language-specific configuration
    config = _get_cyclomatic_config(language)

    # Count decision keywords and logical operators in code only — keywords in
    # comments and string literals are prose, not branches.
    code = strip_non_code(code)
    complexity += _count_occurrences(code, config["keywords"])
    complexity += _count_occurrences(code, config["operators"])

    return complexity


# =============================================================================
# COGNITIVE COMPLEXITY HELPER FUNCTIONS
# =============================================================================


def _get_control_flow_keywords(language: str, patterns: Dict[str, Any]) -> List[str]:
    """Get control flow keywords for a specific language.

    Args:
        language: Programming language name
        patterns: Language complexity patterns

    Returns:
        List of control flow keywords that add complexity
    """
    lang_lower = language.lower()

    control_flow_map = {
        "python": ["if", "elif", "for", "while", "except", "with"],
        "typescript": ["if", "for", "while", "catch", "switch", "do"],
        "javascript": ["if", "for", "while", "catch", "switch", "do"],
        "java": ["if", "for", "while", "catch", "switch", "do"],
    }

    if lang_lower in control_flow_map:
        return control_flow_map[lang_lower]

    # Default to nesting constructs from patterns (may be empty list)
    return cast(List[str], patterns.get("nesting_constructs", []))


def _calculate_line_indentation(line: str, base_indent: Optional[int]) -> Tuple[int, Optional[int]]:
    """Calculate the nesting level from line indentation.

    Args:
        line: Source code line
        base_indent: Base indentation level (None if not yet determined)

    Returns:
        Tuple of (current_nesting_level, updated_base_indent)
    """
    stripped = line.lstrip()
    if not stripped:
        return 0, base_indent

    indent = len(line) - len(stripped)

    # Set base indent on first non-empty line
    if base_indent is None:
        return 0, indent

    # Calculate nesting level based on indentation
    indent_diff = indent - base_indent
    current_nesting = max(0, indent_diff // IndentationDefaults.SPACES_PER_LEVEL)

    return current_nesting, base_indent


def _is_comment_line(stripped: str) -> bool:
    """Check if a line is a comment.

    Args:
        stripped: Left-stripped line content

    Returns:
        True if line is a comment
    """
    return stripped.startswith("#") or stripped.startswith("//") or stripped.startswith("/*")


def _match_control_flow_keyword(stripped: str, control_flow: List[str]) -> Optional[str]:
    """Check if line starts with a control flow keyword.

    Args:
        stripped: Left-stripped line content
        control_flow: List of control flow keywords

    Returns:
        Matched keyword or None
    """
    for keyword in control_flow:
        # Match keyword at start of line (after stripping)
        # Use word boundary to avoid matching 'format' when looking for 'for'
        pattern = rf"^{keyword}(?:\s|\(|:)"
        if re.match(pattern, stripped):
            return keyword
    return None


def _calculate_keyword_complexity(keyword: str, stripped: str, current_nesting: int) -> int:
    """Calculate complexity increment for a control flow keyword.

    Args:
        keyword: Matched control flow keyword
        stripped: Left-stripped line content
        current_nesting: Current nesting level

    Returns:
        Complexity increment (1 + nesting penalty for most keywords)
    """
    # Handle else if / elif specially - they add +1 but no nesting penalty
    if keyword == "elif" or stripped.startswith("else if"):
        return 1

    # Base increment + nesting penalty for all other keywords
    return 1 + current_nesting


def _get_logical_operator_patterns(language: str) -> Tuple[str, str]:
    """Get regex patterns for logical operators by language.

    Args:
        language: Programming language

    Returns:
        Tuple of (and_pattern, or_pattern)
    """
    if language.lower() == "python":
        return r"\band\b", r"\bor\b"
    # C-style uses && and ||
    return r"&&", r"\|\|"


def _find_all_operators(stripped: str, and_pattern: str, or_pattern: str) -> List[Tuple[int, str]]:
    """Find all logical operator positions in a line.

    Args:
        stripped: Left-stripped line content
        and_pattern: Regex pattern for AND operator
        or_pattern: Regex pattern for OR operator

    Returns:
        List of (position, operator_type) tuples sorted by position
    """
    and_matches = list(re.finditer(and_pattern, stripped))
    or_matches = list(re.finditer(or_pattern, stripped))

    if not and_matches and not or_matches:
        return []

    # Combine and sort by position
    all_ops: List[Tuple[int, str]] = [(m.start(), "and") for m in and_matches] + [(m.start(), "or") for m in or_matches]
    all_ops.sort(key=lambda x: x[0])
    return all_ops


def _count_operator_sequences(all_ops: List[Tuple[int, str]]) -> int:
    """Count logical operator sequences (changes in operator type).

    Args:
        all_ops: List of (position, operator_type) tuples

    Returns:
        Number of sequences
    """
    if not all_ops:
        return 0

    sequences = 1  # First sequence
    for i in range(1, len(all_ops)):
        if all_ops[i][1] != all_ops[i - 1][1]:
            sequences += 1
    return sequences


def _count_logical_operator_sequences(stripped: str, language: str) -> int:
    """Count logical operator sequences in a line.

    Based on SonarSource spec: Each sequence of logical operators adds complexity.
    Changes from AND to OR (or vice versa) increment the count.
    Examples: "a && b && c" = +1, "a && b || c" = +2

    Args:
        stripped: Left-stripped line content
        language: Programming language

    Returns:
        Complexity increment from logical operators
    """
    # Get language-specific patterns
    and_pattern, or_pattern = _get_logical_operator_patterns(language)

    # Find all operators
    all_ops = _find_all_operators(stripped, and_pattern, or_pattern)

    # Count sequences
    return _count_operator_sequences(all_ops)


def _process_code_line(line: str, control_flow: List[str], language: str, base_indent: Optional[int]) -> Tuple[int, Optional[int]]:
    """Process a single line of code for cognitive complexity.

    Args:
        line: Source code line
        control_flow: List of control flow keywords
        language: Programming language
        base_indent: Base indentation level

    Returns:
        Tuple of (complexity_increment, updated_base_indent)
    """
    stripped = line.lstrip()

    # Skip empty lines and comments
    if not stripped or _is_comment_line(stripped):
        return 0, base_indent

    # Calculate nesting level from indentation
    current_nesting, base_indent = _calculate_line_indentation(line, base_indent)

    complexity = 0

    # Check for control flow keywords
    keyword = _match_control_flow_keyword(stripped, control_flow)
    if keyword:
        complexity += _calculate_keyword_complexity(keyword, stripped, current_nesting)
    # Handle 'else if' in C-style languages (not caught by keyword match)
    elif stripped.startswith("else if"):
        complexity += 1  # No nesting penalty for else if

    # Count logical operator sequences
    complexity += _count_logical_operator_sequences(stripped, language)

    return complexity, base_indent


def calculate_cognitive_complexity(code: str, language: str) -> int:
    """Calculate cognitive complexity with nesting penalties.

    Based on SonarSource cognitive complexity specification:
    - +1 for each control flow break (if, for, while, catch, switch, etc.)
    - +N nesting penalty when nested (N = current nesting level)
    - +1 for each sequence of logical operators (not each operator)
    - else doesn't increment, but else if does

    Args:
        code: Function source code
        language: Programming language

    Returns:
        Cognitive complexity score
    """
    patterns = get_complexity_patterns(language)
    complexity = 0
    base_indent: Optional[int] = None

    # Get language-specific control flow keywords
    control_flow = _get_control_flow_keywords(language, patterns)

    # Docstring prose reads as control flow to the per-line keyword and logical
    # operator passes below; blank it out first (line positions are preserved,
    # so nesting penalties are unaffected).
    code = strip_non_code(code)

    # Process each line
    for line in code.split("\n"):
        line_complexity, base_indent = _process_code_line(line, control_flow, language, base_indent)
        complexity += line_complexity

    return complexity


def calculate_nesting_depth(code: str, language: str) -> int:
    """Calculate maximum nesting depth.

    Args:
        code: Function source code
        language: Programming language

    Returns:
        Maximum nesting depth
    """
    # Indented docstring prose would otherwise register as nesting.
    lines = strip_non_code(code).split("\n")
    max_depth = 0
    base_indent = None

    for line in lines:
        stripped = line.lstrip()
        if not stripped:
            continue

        indent = len(line) - len(stripped)
        if base_indent is None:
            base_indent = indent
            continue

        # Calculate depth from indentation difference
        indent_diff = indent - base_indent
        depth = max(0, indent_diff // IndentationDefaults.SPACES_PER_LEVEL)
        max_depth = max(max_depth, depth)

    return max_depth
