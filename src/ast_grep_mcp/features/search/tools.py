"""Search feature MCP tool definitions."""

from typing import Any, Dict, List, Literal, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from ast_grep_mcp.core.executor import get_supported_languages
from ast_grep_mcp.features.search.docs import PATTERN_CATEGORIES, PATTERN_LANGUAGES, get_docs, get_pattern_examples
from ast_grep_mcp.features.search.service import (
    build_rule_impl,
    debug_pattern_impl,
    develop_pattern_impl,
    dump_syntax_tree_impl,
    find_code_by_rule_impl,
    find_code_impl,
    test_match_code_rule_impl,
)
from ast_grep_mcp.models.base import DumpFormat

_DUMP_SYNTAX_TREE_DOC = """Dump code's syntax structure or dump a query's pattern structure.
This is useful to discover correct syntax kind and syntax tree structure. Call it when debugging a rule.
The tool requires three arguments: code, language and format. The first two are self-explanatory.
`format` is the output format of the syntax tree.
use `format=cst` to inspect the code's concrete syntax tree structure, useful to debug target code.
use `format=pattern` to inspect how ast-grep interprets a pattern, useful to debug pattern rule.

Internally calls: ast-grep run --pattern <code> --lang <language> --debug-query=<format>
"""

_TEST_MATCH_DOC = """Test a code against an ast-grep YAML rule.
This is useful to test a rule before using it in a project.

Internally calls: ast-grep scan --inline-rules <yaml> --json --stdin
"""

_FIND_CODE_DOC = """Find code in a project folder that matches the given ast-grep pattern.
Pattern is good for simple and single-AST node result.
For more complex usage, please use YAML by `find_code_by_rule`.

Internally calls: ast-grep run --pattern <pattern> [--json] <project_folder>

Output formats:
- text (default): Compact text format with file:line-range headers and complete match text
- json: Full match objects with metadata including ranges, meta-variables, etc.

The max_results parameter limits the number of complete matches returned (not individual lines).

Example usage:
  find_code(pattern="class $NAME", max_results=20)  # Returns text format
  find_code(pattern="class $NAME", output_format="json")  # Returns JSON with metadata
"""

_FIND_CODE_BY_RULE_DOC = """Find code in a project folder using a custom YAML rule.
This is more powerful than find_code as it supports complex rules like:
- any/all conditions
- regex/inside/precedes/follows
- multiple patterns in one rule
- constraints and relations

Internally calls: ast-grep scan --inline-rules <yaml> [--json] <project_folder>

CRITICAL: Relational rules (inside, has, follows, precedes) need `stopBy: end` to search
the entire tree. The default `neighbor` only checks immediate parent/sibling.

Minimal Rule Template:
```yaml
id: my-rule
language: python
rule:
  pattern: |
    def $NAME($$$PARAMS):
      $$$BODY
```

Output formats:
- text (default): Compact text format with file:line-range headers and complete match text
- json: Full match objects with metadata including ranges, rule ID, matched text etc.
"""

_DEBUG_PATTERN_DOC = """Debug why a pattern doesn't match code.

Checks metavariable validation, AST comparison, match attempt, and best practices.

Metavariable syntax: $NAME (single node), $$$ARGS (zero or more), $_NAME (non-capturing)
Valid names: $META, $META_VAR, $META_VAR1. Invalid: $name (lowercase), $123, $KEBAB-CASE

Output includes: pattern_valid, pattern_ast, code_ast, ast_comparison, metavariables,
issues, suggestions, match_attempt.
"""

_GET_DOCS_DOC = """Get ast-grep documentation for the specified topic.

Topics: pattern, rules, relational, metavariables, workflow, strictness, all

Use this when you need guidance on pattern syntax, YAML rule configuration,
relational rules (inside, has, follows, precedes), strictness modes, or best practices.
"""

_BUILD_RULE_DOC = """Build a properly structured YAML rule from components.

Automatically adds required fields (id, language, rule) and sets `stopBy: end`
on relational rules. Returns YAML ready for `find_code_by_rule` or `test_match_code_rule`.

Relational parameters:
- inside: pattern that must CONTAIN the match (parent)
- has: pattern that must be INSIDE the match (child)
- follows: pattern that must PRECEDE the match
- precedes: pattern that must FOLLOW the match
- inside_kind / has_kind: node-type matching instead of patterns
"""

_GET_PATTERN_EXAMPLES_DOC = """Get common ast-grep pattern examples for a language.

Returns ready-to-use patterns organized by category with descriptions.

Categories: function, class, import, variable, control_flow, error_handling, async
Languages: JavaScript, TypeScript, Python, Go, Rust, Java, Ruby, C, C++
"""

_DEVELOP_PATTERN_DOC = """Interactive pattern development assistant.

Analyzes sample code and suggests patterns that will match it. Tests patterns and
provides guidance. Recommended starting point when unsure how to write a pattern.

Output fields: code_analysis, suggested_patterns, best_pattern, pattern_matches,
yaml_rule_template, next_steps, refinement_steps.

Workflow: develop_pattern() → find_code() → build_rule() → debug_pattern() → test_match_code_rule()
"""


def _register_dump_syntax_tree(mcp: FastMCP) -> None:
    """Register dump_syntax_tree tool."""

    @mcp.tool(description=_DUMP_SYNTAX_TREE_DOC)
    def dump_syntax_tree(
        code: str = Field(description="The code you need"),
        language: str = Field(description=f"The language of the code. Supported: {', '.join(get_supported_languages())}"),
        format: DumpFormat = Field(description="Code dump format. Available values: pattern, ast, cst", default="cst"),
    ) -> str:
        return dump_syntax_tree_impl(code, language, format)


def _register_test_match_code_rule(mcp: FastMCP) -> None:
    """Register test_match_code_rule tool."""

    @mcp.tool(description=_TEST_MATCH_DOC)
    def test_match_code_rule(
        code: str = Field(description="The code to test against the rule"),
        yaml_rule: str = Field(description="The ast-grep YAML rule to search. It must have id, language, rule fields."),
    ) -> List[Dict[str, Any]]:
        return test_match_code_rule_impl(code, yaml_rule)


def _register_find_code(mcp: FastMCP) -> None:
    """Register find_code tool."""

    @mcp.tool(description=_FIND_CODE_DOC)
    def find_code(
        project_folder: str = Field(description="The absolute path to the project folder. It must be absolute path."),
        pattern: str = Field(description="The ast-grep pattern to search for. Note, the pattern must have valid AST structure."),
        language: str = Field(
            description=f"The language of the code. Supported: {', '.join(get_supported_languages())}. "
            "If not specified, will be auto-detected based on file extensions.",
            default="",
        ),
        max_results: int = Field(default=0, description="Maximum results to return"),
        output_format: str = Field(default="text", description="'text' or 'json'"),
        max_file_size_mb: int = Field(
            default=0,
            description="Skip files larger than this size in MB. 0 = unlimited (default).",
        ),
        workers: int = Field(
            default=0,
            description="Number of parallel worker threads. 0 = auto (default).",
        ),
    ) -> str | List[Dict[str, Any]]:
        return find_code_impl(
            project_folder,
            pattern,
            language,
            max_results,
            output_format,  # type: ignore[arg-type]
            max_file_size_mb,
            workers,
        )


def _register_find_code_by_rule(mcp: FastMCP) -> None:
    """Register find_code_by_rule tool."""

    @mcp.tool(description=_FIND_CODE_BY_RULE_DOC)
    def find_code_by_rule(
        project_folder: str = Field(description="The absolute path to the project folder. It must be absolute path."),
        yaml_rule: str = Field(description="The ast-grep YAML rule to search. It must have id, language, rule fields."),
        max_results: int = Field(default=0, description="Maximum results to return"),
        output_format: str = Field(default="text", description="'text' or 'json'"),
    ) -> str | List[Dict[str, Any]] | Dict[str, Any]:
        return find_code_by_rule_impl(
            project_folder,
            yaml_rule,
            max_results,
            output_format,  # type: ignore[arg-type]
        )


def _register_debug_pattern(mcp: FastMCP) -> None:
    """Register debug_pattern tool."""

    @mcp.tool(description=_DEBUG_PATTERN_DOC)
    def debug_pattern(
        pattern: str = Field(description="The ast-grep pattern to debug"),
        code: str = Field(description="The code to match against"),
        language: str = Field(description=f"The programming language. Supported: {', '.join(get_supported_languages())}"),
    ) -> Dict[str, Any]:
        result = debug_pattern_impl(pattern, code, language)
        return result.to_dict()


def _register_get_ast_grep_docs(mcp: FastMCP) -> None:
    """Register get_ast_grep_docs tool."""

    @mcp.tool(description=_GET_DOCS_DOC)
    def get_ast_grep_docs(
        topic: Literal["pattern", "rules", "relational", "metavariables", "workflow", "strictness", "all"] = Field(
            description="Documentation topic to retrieve"
        ),
    ) -> str:
        return get_docs(topic)


def _call_build_rule(
    pattern: str,
    language: str,
    rule_id: Optional[str],
    inside: Optional[str],
    has: Optional[str],
    follows: Optional[str],
    precedes: Optional[str],
    inside_kind: Optional[str],
    has_kind: Optional[str],
    stop_by: str,
    message: Optional[str],
    severity: Optional[str],
    fix: Optional[str],
) -> str:
    return build_rule_impl(
        pattern=pattern,
        language=language,
        rule_id=rule_id,
        inside=inside,
        has=has,
        follows=follows,
        precedes=precedes,
        inside_kind=inside_kind,
        has_kind=has_kind,
        stop_by=stop_by,
        message=message,
        severity=severity,
        fix=fix,
    )


def _register_build_rule(mcp: FastMCP) -> None:
    """Register build_rule tool."""

    @mcp.tool(description=_BUILD_RULE_DOC)
    def build_rule(
        pattern: str = Field(description="The main pattern to match"),
        language: str = Field(description=f"Target language. Supported: {', '.join(get_supported_languages())}"),
        rule_id: Optional[str] = Field(default=None, description="Unique rule ID (auto-generated if not provided)"),
        inside: Optional[str] = Field(default=None, description="Pattern that must CONTAIN the match (parent)"),
        has: Optional[str] = Field(default=None, description="Pattern that must be INSIDE the match (child)"),
        follows: Optional[str] = Field(default=None, description="Pattern that must PRECEDE the match"),
        precedes: Optional[str] = Field(default=None, description="Pattern that must FOLLOW the match"),
        inside_kind: Optional[str] = Field(
            default=None, description="Node kind that must contain the match (e.g., 'function_declaration')"
        ),
        has_kind: Optional[str] = Field(default=None, description="Node kind that must be inside the match"),
        stop_by: str = Field(
            default="end",
            description="stopBy for relational rules: 'end' (search entire tree, DEFAULT), 'neighbor' (immediate only), or custom pattern",
        ),
        message: Optional[str] = Field(default=None, description="Human-readable message for matches"),
        severity: Optional[str] = Field(default=None, description="Severity: error, warning, info, hint"),
        fix: Optional[str] = Field(default=None, description="Auto-fix template using captured metavariables"),
    ) -> str:
        return _call_build_rule(
            pattern,
            language,
            rule_id,
            inside,
            has,
            follows,
            precedes,
            inside_kind,
            has_kind,
            stop_by,
            message,
            severity,
            fix,
        )


def _register_get_pattern_examples(mcp: FastMCP) -> None:
    """Register get_pattern_examples tool."""

    @mcp.tool(description=_GET_PATTERN_EXAMPLES_DOC)
    def get_pattern_examples_tool(
        language: str = Field(description=f"Target language. Available: {', '.join(PATTERN_LANGUAGES)}"),
        category: Optional[str] = Field(
            default=None,
            description=f"Optional category filter. Available: {', '.join(PATTERN_CATEGORIES)}, all. "
            "If not specified, shows all categories.",
        ),
    ) -> str:
        return get_pattern_examples(language, category)


def _register_develop_pattern(mcp: FastMCP) -> None:
    """Register develop_pattern tool."""

    @mcp.tool(description=_DEVELOP_PATTERN_DOC)
    def develop_pattern(
        code: str = Field(description="Sample code you want to match"),
        language: str = Field(description=f"The programming language. Supported: {', '.join(get_supported_languages())}"),
        goal: Optional[str] = Field(
            default=None,
            description="Optional: Describe what you're trying to find (e.g., 'console.log calls', 'functions with no return')",
        ),
    ) -> Dict[str, Any]:
        result = develop_pattern_impl(code, language, goal)
        return result.to_dict()


def register_search_tools(mcp: FastMCP) -> None:
    """Register search-related MCP tools.

    Args:
        mcp: FastMCP instance to register tools with
    """
    _register_dump_syntax_tree(mcp)
    _register_test_match_code_rule(mcp)
    _register_find_code(mcp)
    _register_find_code_by_rule(mcp)
    _register_debug_pattern(mcp)
    _register_develop_pattern(mcp)
    _register_get_ast_grep_docs(mcp)
    _register_build_rule(mcp)
    _register_get_pattern_examples(mcp)
