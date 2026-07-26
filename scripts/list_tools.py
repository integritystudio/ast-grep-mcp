"""List all MCP tools registered by this server.

Usage:
    uv run python scripts/list_tools.py           # grouped by category
    uv run python scripts/list_tools.py --flat    # one name per line
    uv run python scripts/list_tools.py --json    # JSON output
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from mcp.server.fastmcp import FastMCP

from ast_grep_mcp.server.registry import register_all_tools

CATEGORY_ORDER = [
    "search",
    "rewrite",
    "refactoring",
    "deduplication",
    "complexity",
    "quality",
    "schema",
    "documentation",
    "cross_language",
    "condense",
]

TOOL_CATEGORIES: dict[str, str] = {
    # search
    "find_code": "search",
    "find_code_by_rule": "search",
    "dump_syntax_tree": "search",
    "debug_pattern": "search",
    "build_rule": "search",
    "develop_pattern": "search",
    "get_ast_grep_docs": "search",
    "get_pattern_examples_tool": "search",
    "test_match_code_rule": "search",
    # rewrite
    "rewrite_code": "rewrite",
    "rollback_rewrite": "rewrite",
    "list_backups": "rewrite",
    "get_zod_rewrite_rule": "rewrite",
    "list_zod_rewrite_rules": "rewrite",
    # refactoring
    "extract_function": "refactoring",
    "rename_symbol": "refactoring",
    # deduplication
    "find_duplication": "deduplication",
    "analyze_deduplication_candidates": "deduplication",
    "apply_deduplication": "deduplication",
    "benchmark_deduplication": "deduplication",
    "calculate_ast_similarity": "deduplication",
    "calculate_semantic_similarity": "deduplication",
    # complexity
    "analyze_complexity": "complexity",
    "detect_code_smells": "complexity",
    "test_sentry_integration": "complexity",
    # quality
    "create_linting_rule": "quality",
    "list_rule_templates": "quality",
    "enforce_standards": "quality",
    "apply_standards_fixes": "quality",
    "generate_quality_report": "quality",
    "detect_security_issues": "quality",
    "detect_orphans": "quality",
    # schema
    "get_schema_type": "schema",
    "search_schemas": "schema",
    "get_type_hierarchy": "schema",
    "get_type_properties": "schema",
    "generate_schema_example": "schema",
    "generate_entity_id": "schema",
    "validate_entity_id": "schema",
    "build_entity_graph": "schema",
    "enhance_entity_graph": "schema",
    "detect_structured_data": "schema",
    "validate_structured_data": "schema",
    # documentation
    "generate_docstrings": "documentation",
    "generate_readme_sections": "documentation",
    "generate_api_docs": "documentation",
    "generate_changelog": "documentation",
    "sync_documentation": "documentation",
    # cross_language
    "search_multi_language": "cross_language",
    "find_language_equivalents": "cross_language",
    "convert_code_language": "cross_language",
    "refactor_polyglot": "cross_language",
    "generate_language_bindings": "cross_language",
    # condense
    "condense_extract_surface": "condense",
    "condense_normalize": "condense",
    "condense_strip": "condense",
    "condense_pack": "condense",
    "condense_estimate": "condense",
    "condense_train_dictionary": "condense",
}


async def collect_tools() -> list[tuple[str, str]]:
    mcp = FastMCP("ast-grep-mcp")
    register_all_tools(mcp)
    tools = await mcp.list_tools()
    def first_line(text: str | None) -> str:
        lines = (text or "").splitlines()
        return lines[0] if lines else ""

    return [(t.name, first_line(t.description)) for t in tools]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flat", action="store_true", help="Print one tool name per line")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    tools = asyncio.run(collect_tools())

    if args.json:
        print(json.dumps([{"name": n, "description": d} for n, d in tools], indent=2))
        return 0

    if args.flat:
        for name, _ in sorted(tools):
            print(name)
        return 0

    grouped: dict[str, list[tuple[str, str]]] = {c: [] for c in CATEGORY_ORDER}
    uncategorized: list[tuple[str, str]] = []
    for name, desc in tools:
        category = TOOL_CATEGORIES.get(name)
        if category is None:
            uncategorized.append((name, desc))
        else:
            grouped[category].append((name, desc))

    total = 0
    for category in CATEGORY_ORDER:
        items = grouped[category]
        if not items:
            continue
        total += len(items)
        print(f"\n{category} ({len(items)})")
        for name, desc in sorted(items):
            print(f"  {name}" + (f" — {desc}" if desc else ""))

    if uncategorized:
        total += len(uncategorized)
        print(f"\nuncategorized ({len(uncategorized)})")
        for name, desc in sorted(uncategorized):
            print(f"  {name}" + (f" — {desc}" if desc else ""))

    print(f"\nTotal: {total} tools")
    return 0


if __name__ == "__main__":
    sys.exit(main())
