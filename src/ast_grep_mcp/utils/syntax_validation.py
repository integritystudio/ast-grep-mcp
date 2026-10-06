"""Shared syntax validation utilities.

This module provides common syntax validation and error suggestion
functionality used across the deduplication system.
"""

import re
from typing import List, Optional, Tuple

from ast_grep_mcp.constants import SyntaxValidationDefaults

# Configuration-driven error patterns and suggestions
ERROR_SUGGESTIONS = {
    # Python-specific errors
    "indentation": "Fix indentation - ensure consistent use of spaces/tabs",
    "unexpected eof": "Check for missing closing brackets, parentheses, or quotes",
    "unexpected indent": "Check for missing closing brackets, parentheses, or quotes",
    "invalid syntax": "Review syntax near the error location - check for typos",
    # Bracket/brace errors - check these with additional context
    "mismatched brace": "Balance opening and closing braces {}",
    "unbalanced brace": "Balance opening and closing braces {}",
    "mismatched parenthes": "Balance opening and closing parentheses ()",
    "unbalanced parenthes": "Balance opening and closing parentheses ()",
    "mismatched bracket": "Balance opening and closing brackets []",
    "unbalanced bracket": "Balance opening and closing brackets []",
}


def _find_error_suggestion(error_lower: str) -> Optional[str]:
    """Find a suggestion for the given error message.

    Args:
        error_lower: Lowercase error message

    Returns:
        Suggestion string or None if no match found
    """
    # Direct pattern matching using configuration
    for pattern, suggestion in ERROR_SUGGESTIONS.items():
        if pattern in error_lower:
            return suggestion
    return None


def suggest_syntax_fix(error: Optional[str], language: str, context: str = "file") -> str:
    """Generate syntax fix suggestion based on error.

    This is the shared implementation that replaces duplicate
    _suggest_syntax_fix methods in validators.

    Args:
        error: Error message from syntax validation
        language: Programming language (e.g., 'python', 'javascript')
        context: Context of the error ('file' or 'code')

    Returns:
        Suggested fix message tailored to the error and language
    """
    if not error:
        return "Check code syntax and formatting" if context == "code" else "Review file for syntax errors"

    error_lower = error.lower()

    # Try to find a matching suggestion
    suggestion = _find_error_suggestion(error_lower)
    if suggestion:
        return suggestion

    # Default suggestion with truncated error message
    return f"Review {language} syntax and fix the error: {error[: SyntaxValidationDefaults.ERROR_SUGGESTION_PREVIEW_LENGTH]}"


# Comments and string/char/template literals (C-family syntax: JS/TS, Java) whose
# brackets are text, not structure.
_LITERAL_OR_COMMENT = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'|`(?:\\.|[^`\\])*`",
    re.DOTALL,
)


def validate_bracket_balance(code: str) -> List[Tuple[str, str]]:
    """Check for balanced brackets, braces, and parentheses.

    Args:
        code: Code to validate

    Returns:
        List of (bracket_type, error_message) tuples for unbalanced brackets
    """
    errors = []
    code = _LITERAL_OR_COMMENT.sub(" ", code)

    bracket_pairs = [("{", "}", "braces"), ("(", ")", "parentheses"), ("[", "]", "brackets")]

    for open_char, close_char, name in bracket_pairs:
        open_count = code.count(open_char)
        close_count = code.count(close_char)
        if open_count != close_count:
            errors.append((name, f"Mismatched {name}"))

    return errors


def validate_python_syntax(code: str) -> Tuple[bool, str]:
    """Validate Python code syntax.

    Args:
        code: Python code to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not code or not code.strip():
        return False, "Code is empty"

    try:
        import ast

        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"Python syntax error: {e}"
    except Exception as e:
        return False, f"Validation error: {e}"


def validate_javascript_syntax(code: str) -> Tuple[bool, str]:
    """Basic validation for JavaScript/TypeScript syntax.

    Args:
        code: JavaScript/TypeScript code to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    bracket_errors = validate_bracket_balance(code)
    if bracket_errors:
        # Return the first error found
        _, error_msg = bracket_errors[0]
        return False, error_msg

    return True, ""


def validate_java_syntax(code: str) -> Tuple[bool, str]:
    """Basic validation for Java syntax.

    Args:
        code: Java code to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    bracket_errors = validate_bracket_balance(code)
    if bracket_errors:
        # Only check braces and parentheses for Java
        for name, error_msg in bracket_errors:
            if name in ("braces", "parentheses"):
                return False, error_msg

    return True, ""


_JS_LIKE_LANGUAGES = frozenset(["javascript", "typescript", "jsx", "tsx"])

_LANGUAGE_VALIDATORS = {
    "python": validate_python_syntax,
    "java": validate_java_syntax,
}


def validate_code_for_language(code: str, language: str) -> Tuple[bool, str]:
    """Validate code syntax for specific language.

    This is a shared implementation that can replace duplicate
    validation logic across modules.

    Args:
        code: Code to validate
        language: Programming language

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not code or not code.strip():
        return False, "Code is empty"

    lang_lower = language.lower()

    if lang_lower in _JS_LIKE_LANGUAGES:
        return validate_javascript_syntax(code)

    validator = _LANGUAGE_VALIDATORS.get(lang_lower)
    if validator:
        return validator(code)

    return True, ""
