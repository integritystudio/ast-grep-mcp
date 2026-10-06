"""MCP tool definitions for cross-language operations features.

This module registers MCP tools for:
- search_multi_language: Search across multiple programming languages
- find_language_equivalents: Find equivalent patterns across languages
- convert_code_language: Convert code between languages
- refactor_polyglot: Refactor across language boundaries
- generate_language_bindings: Generate API client bindings
"""

from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from ast_grep_mcp.constants import CrossLanguageDefaults
from ast_grep_mcp.core.logging import get_logger
from ast_grep_mcp.features.cross_language.binding_generator import generate_language_bindings_impl
from ast_grep_mcp.features.cross_language.language_converter import convert_code_language_impl
from ast_grep_mcp.features.cross_language.multi_language_search import search_multi_language_impl
from ast_grep_mcp.features.cross_language.pattern_equivalence import find_language_equivalents_impl
from ast_grep_mcp.features.cross_language.polyglot_refactoring import refactor_polyglot_impl
from ast_grep_mcp.models.cross_language import (
    ConversionResult,
    ConversionWarning,
    ConvertedCode,
    PatternEquivalence,
    PatternEquivalenceResult,
    PatternExample,
    TypeMapping,
)
from ast_grep_mcp.utils.tool_context import tool_context

# =============================================================================
# Response Formatting Helpers
# =============================================================================


def _format_example(ex: PatternExample) -> Dict[str, Any]:
    """Format a single pattern example."""
    return {
        "language": ex.language,
        "code": ex.code,
        "description": ex.description,
        "notes": ex.notes,
    }


def _format_equivalence(e: PatternEquivalence) -> Dict[str, Any]:
    """Format a single pattern equivalence."""
    return {
        "pattern_id": e.pattern_id,
        "concept": e.concept,
        "category": e.category,
        "description": e.description,
        "examples": [_format_example(ex) for ex in e.examples],
        "related_patterns": e.related_patterns,
        "complexity_comparison": e.complexity_comparison,
    }


def _format_equivalents_result(result: PatternEquivalenceResult) -> Dict[str, Any]:
    """Format find_language_equivalents result."""
    return {
        "pattern_description": result.pattern_description,
        "source_language": result.source_language,
        "target_languages": result.target_languages,
        "equivalences": [_format_equivalence(e) for e in result.equivalences],
        "suggestions": result.suggestions,
        "execution_time_ms": result.execution_time_ms,
    }


def _format_type_mapping(t: TypeMapping) -> Dict[str, str]:
    """Format a single type mapping."""
    return {"source": t.source_type, "target": t.target_type}


def _format_warning(w: ConversionWarning) -> Dict[str, Any]:
    """Format a single conversion warning."""
    return {
        "severity": w.severity,
        "message": w.message,
        "line_number": w.line_number,
        "suggestion": w.suggestion,
    }


def _format_conversion(c: ConvertedCode) -> Dict[str, Any]:
    """Format a single code conversion."""
    return {
        "source_code": c.source_code,
        "converted_code": c.converted_code,
        "from_language": c.from_language,
        "to_language": c.to_language,
        "style": c.style.value,
        "type_mappings": [_format_type_mapping(t) for t in c.type_mappings],
        "warnings": [_format_warning(w) for w in c.warnings],
        "imports_needed": c.imports_needed,
        "success": c.success,
    }


def _format_conversion_result(result: ConversionResult) -> Dict[str, Any]:
    """Format convert_code_language result."""
    return {
        "conversions": [_format_conversion(c) for c in result.conversions],
        "total_functions": result.total_functions,
        "successful_conversions": result.successful_conversions,
        "failed_conversions": result.failed_conversions,
        "execution_time_ms": result.execution_time_ms,
    }


# =============================================================================
# Tool Implementations
# =============================================================================


def _format_search_result(result: Any) -> Dict[str, Any]:
    return {
        "query": result.query,
        "languages_searched": result.languages_searched,
        "matches": [
            {
                "language": m.language,
                "file_path": m.file_path,
                "line_number": m.line_number,
                "code_snippet": m.code_snippet,
                "semantic_group": m.semantic_group,
                "confidence": m.confidence,
            }
            for m in result.matches
        ],
        "total_matches": result.total_matches,
        "matches_by_language": result.matches_by_language,
        "semantic_groups": result.semantic_groups,
        "execution_time_ms": result.execution_time_ms,
    }


def search_multi_language_tool(
    project_folder: str,
    semantic_pattern: str,
    languages: Optional[List[str]] = None,
    group_by: str = "semantic",
    max_results_per_language: int = CrossLanguageDefaults.MAX_RESULTS_PER_LANGUAGE,
) -> Dict[str, Any]:
    if languages is None:
        languages = ["auto"]
    logger = get_logger("tool.search_multi_language")
    logger.info(
        "tool_invoked",
        tool="search_multi_language",
        project_folder=project_folder,
        semantic_pattern=semantic_pattern,
        languages=languages,
    )
    with tool_context("search_multi_language", project_folder=project_folder, semantic_pattern=semantic_pattern):
        return _format_search_result(
            search_multi_language_impl(
                project_folder=project_folder,
                semantic_pattern=semantic_pattern,
                languages=languages,
                group_by=group_by,
                max_results_per_language=max_results_per_language,
            )
        )


def find_language_equivalents_tool(
    pattern_description: str,
    source_language: Optional[str] = None,
    target_languages: Optional[List[str]] = None,
) -> Dict[str, Any]:
    logger = get_logger("tool.find_language_equivalents")
    logger.info(
        "tool_invoked",
        tool="find_language_equivalents",
        pattern_description=pattern_description,
        source_language=source_language,
    )
    with tool_context("find_language_equivalents", pattern_description=pattern_description, source_language=source_language):
        return _format_equivalents_result(
            find_language_equivalents_impl(
                pattern_description=pattern_description,
                source_language=source_language,
                target_languages=target_languages,
            )
        )


def convert_code_language_tool(
    code_snippet: str,
    from_language: str,
    to_language: str,
    conversion_style: str = "idiomatic",
    include_comments: bool = True,
) -> Dict[str, Any]:
    logger = get_logger("tool.convert_code_language")
    logger.info(
        "tool_invoked",
        tool="convert_code_language",
        from_language=from_language,
        to_language=to_language,
        conversion_style=conversion_style,
    )
    with tool_context("convert_code_language", from_language=from_language, to_language=to_language):
        return _format_conversion_result(
            convert_code_language_impl(
                code_snippet=code_snippet,
                from_language=from_language,
                to_language=to_language,
                conversion_style=conversion_style,
                include_comments=include_comments,
            )
        )


def _format_polyglot_result(result: Any) -> Dict[str, Any]:
    changes_source = result.changes_made if not result.dry_run else result.plan.changes
    return {
        "plan": {
            "refactoring_type": result.plan.refactoring_type.value,
            "symbol_name": result.plan.symbol_name,
            "new_name": result.plan.new_name,
            "affected_languages": result.plan.affected_languages,
            "changes_count": len(result.plan.changes),
            "risks": result.plan.risks,
            "requires_manual_review": result.plan.requires_manual_review,
        },
        "changes": [
            {
                "language": c.language,
                "file_path": c.file_path,
                "line_number": c.line_number,
                "original_code": c.original_code,
                "new_code": c.new_code,
                "change_type": c.change_type,
            }
            for c in changes_source
        ],
        "dry_run": result.dry_run,
        "files_modified": result.files_modified,
        "validation_passed": result.validation_passed,
        "validation_errors": result.validation_errors,
        "execution_time_ms": result.execution_time_ms,
        "backup_id": result.backup_id,
    }


def refactor_polyglot_tool(
    project_folder: str,
    refactoring_type: str,
    symbol_name: str,
    new_name: Optional[str] = None,
    affected_languages: Optional[List[str]] = None,
    dry_run: bool = True,
) -> Dict[str, Any]:
    if affected_languages is None:
        affected_languages = ["all"]
    logger = get_logger("tool.refactor_polyglot")
    logger.info(
        "tool_invoked",
        tool="refactor_polyglot",
        project_folder=project_folder,
        refactoring_type=refactoring_type,
        symbol_name=symbol_name,
        dry_run=dry_run,
    )
    with tool_context("refactor_polyglot", project_folder=project_folder, refactoring_type=refactoring_type):
        return _format_polyglot_result(
            refactor_polyglot_impl(
                project_folder=project_folder,
                refactoring_type=refactoring_type,
                symbol_name=symbol_name,
                new_name=new_name,
                affected_languages=affected_languages,
                dry_run=dry_run,
            )
        )


def _format_bindings_result(result: Any) -> Dict[str, Any]:
    return {
        "api_name": result.api_name,
        "api_version": result.api_version,
        "base_url": result.base_url,
        "endpoints_count": result.endpoints_count,
        "bindings": [
            {
                "language": b.language,
                "file_name": b.file_name,
                "code": b.code,
                "imports": b.imports,
                "dependencies": b.dependencies,
                "types_generated": b.types_generated,
            }
            for b in result.bindings
        ],
        "warnings": result.warnings,
        "execution_time_ms": result.execution_time_ms,
    }


def generate_language_bindings_tool(
    api_definition_file: str,
    target_languages: Optional[List[str]] = None,
    binding_style: str = "native",
    include_types: bool = True,
) -> Dict[str, Any]:
    if target_languages is None:
        target_languages = ["python", "typescript", "javascript"]
    logger = get_logger("tool.generate_language_bindings")
    logger.info(
        "tool_invoked",
        tool="generate_language_bindings",
        api_definition_file=api_definition_file,
        target_languages=target_languages,
        binding_style=binding_style,
    )
    with tool_context("generate_language_bindings", api_definition_file=api_definition_file, binding_style=binding_style):
        return _format_bindings_result(
            generate_language_bindings_impl(
                api_definition_file=api_definition_file,
                target_languages=target_languages,
                binding_style=binding_style,
                include_types=include_types,
            )
        )


# =============================================================================
# MCP Registration
# =============================================================================


def register_cross_language_tools(mcp: FastMCP) -> None:
    """Register all cross-language feature tools with MCP server."""

    @mcp.tool()
    def search_multi_language(
        project_folder: str = Field(description="Root folder of the project (absolute path)"),
        semantic_pattern: str = Field(description="Semantic pattern to search for (e.g., 'async function', 'try catch')"),
        languages: Optional[List[str]] = Field(default=None, description="Languages to search (['auto'] for auto-detection)"),
        group_by: str = Field(default="semantic", description="Grouping strategy ('semantic', 'language', 'file')"),
        max_results_per_language: int = Field(
            default=CrossLanguageDefaults.MAX_RESULTS_PER_LANGUAGE,
            description="Maximum results per language",
        ),
    ) -> Dict[str, Any]:
        """Search across multiple programming languages for semantically equivalent patterns."""
        return search_multi_language_tool(project_folder, semantic_pattern, languages, group_by, max_results_per_language)

    @mcp.tool()
    def find_language_equivalents(
        pattern_description: str = Field(description="Description of the pattern to find (e.g., 'list comprehension')"),
        source_language: Optional[str] = Field(default=None, description="Optional source language to highlight"),
        target_languages: Optional[List[str]] = Field(default=None, description="Languages to include in results"),
    ) -> Dict[str, Any]:
        """Find equivalent patterns across programming languages."""
        return find_language_equivalents_tool(pattern_description, source_language, target_languages)

    @mcp.tool()
    def convert_code_language(
        code_snippet: str = Field(description="Code to convert"),
        from_language: str = Field(description="Source language (python, typescript, javascript, java)"),
        to_language: str = Field(description="Target language (python, typescript, javascript, kotlin)"),
        conversion_style: str = Field(default="idiomatic", description="Conversion style ('literal', 'idiomatic', 'compatible')"),
        include_comments: bool = Field(default=True, description="Whether to include conversion comments"),
    ) -> Dict[str, Any]:
        """Convert code from one programming language to another."""
        return convert_code_language_tool(code_snippet, from_language, to_language, conversion_style, include_comments)

    @mcp.tool()
    def refactor_polyglot(
        project_folder: str = Field(description="Root folder of the project"),
        refactoring_type: str = Field(description="Type of refactoring ('rename_api', 'extract_constant', 'update_contract')"),
        symbol_name: str = Field(description="Symbol being refactored"),
        new_name: Optional[str] = Field(default=None, description="New name (required for rename operations)"),
        affected_languages: Optional[List[str]] = Field(default=None, description="Languages to include (['all'] for all)"),
        dry_run: bool = Field(default=True, description="If True, only preview changes"),
    ) -> Dict[str, Any]:
        """Refactor across multiple programming languages atomically."""
        return refactor_polyglot_tool(project_folder, refactoring_type, symbol_name, new_name, affected_languages, dry_run)

    @mcp.tool()
    def generate_language_bindings(
        api_definition_file: str = Field(description="Path to API spec file (OpenAPI/Swagger JSON or YAML)"),
        target_languages: Optional[List[str]] = Field(default=None, description="Languages to generate bindings for"),
        binding_style: str = Field(default="native", description="Binding style ('native', 'sdk', 'minimal')"),
        include_types: bool = Field(default=True, description="Whether to include type definitions"),
    ) -> Dict[str, Any]:
        """Generate API client bindings for multiple languages from specifications."""
        return generate_language_bindings_tool(api_definition_file, target_languages, binding_style, include_types)
