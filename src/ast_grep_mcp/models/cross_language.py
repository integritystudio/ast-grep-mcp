"""Data models for cross-language operations features.

This module defines data models for:
- Multi-language search results
- Pattern equivalence mapping
- Language conversion
- Polyglot refactoring
- API binding generation
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ConversionStyle(Enum):
    """Code conversion style options."""

    LITERAL = "literal"  # Direct translation preserving structure
    IDIOMATIC = "idiomatic"  # Use target language idioms
    COMPATIBLE = "compatible"  # Maximum cross-platform compatibility


class BindingStyle(Enum):
    """API binding generation style."""

    NATIVE = "native"  # Native language style (requests, fetch, etc.)
    SDK = "sdk"  # Full SDK with types and utilities
    MINIMAL = "minimal"  # Minimal implementation


class RefactoringType(Enum):
    """Cross-language refactoring types."""

    RENAME_API = "rename_api"  # Rename API endpoint across frontend/backend
    EXTRACT_CONSTANT = "extract_constant"  # Extract to shared config
    UPDATE_CONTRACT = "update_contract"  # Update API contract signature


# =============================================================================
# Multi-Language Search Models
# =============================================================================


@dataclass
class SemanticPattern:
    """A semantic pattern that can match across languages.

    Attributes:
        description: Human-readable pattern description
        concept: The abstract concept being matched
        language_patterns: Language-specific ast-grep patterns
    """

    description: str
    concept: str
    language_patterns: Dict[str, str] = field(default_factory=dict)


@dataclass
class MultiLanguageMatch:
    """A match found in multi-language search.

    Attributes:
        language: Programming language of the match
        file_path: Path to the file containing the match
        line_number: Line number (1-indexed)
        code_snippet: The matched code
        semantic_group: Semantic grouping category
        confidence: Confidence score for semantic matching
    """

    language: str
    file_path: str
    line_number: int
    code_snippet: str
    semantic_group: str = ""
    confidence: float = 1.0


@dataclass
class MultiLanguageSearchResult:
    """Result of multi-language search operation.

    Attributes:
        query: Original semantic query
        languages_searched: Languages that were searched
        matches: List of matches grouped by semantic similarity
        total_matches: Total number of matches
        matches_by_language: Count of matches per language
        semantic_groups: Distinct semantic groups found
        execution_time_ms: Execution time in milliseconds
    """

    query: str
    languages_searched: List[str]
    matches: List[MultiLanguageMatch]
    total_matches: int = 0
    matches_by_language: Dict[str, int] = field(default_factory=dict)
    semantic_groups: List[str] = field(default_factory=list)
    execution_time_ms: int = 0


# =============================================================================
# Pattern Equivalence Models
# =============================================================================


@dataclass
class PatternExample:
    """An example of a pattern in a specific language.

    Attributes:
        language: Programming language
        code: Example code
        description: Explanation of this implementation
        notes: Additional notes or caveats
    """

    language: str
    code: str
    description: str = ""
    notes: List[str] = field(default_factory=list)


@dataclass
class PatternEquivalence:
    """Equivalent patterns across multiple languages.

    Attributes:
        pattern_id: Unique identifier for this pattern
        concept: The abstract concept (e.g., "list comprehension")
        category: Pattern category (control_flow, data_structures, etc.)
        description: Detailed description
        examples: Examples in each supported language
        related_patterns: Related pattern IDs
        complexity_comparison: How complexity compares across languages
    """

    pattern_id: str
    concept: str
    category: str
    description: str
    examples: List[PatternExample] = field(default_factory=list)
    related_patterns: List[str] = field(default_factory=list)
    complexity_comparison: Dict[str, str] = field(default_factory=dict)


@dataclass
class PatternEquivalenceResult:
    """Result of pattern equivalence lookup.

    Attributes:
        pattern_description: The input pattern description
        source_language: Source language (if specified)
        target_languages: Target languages searched
        equivalences: Found pattern equivalences
        suggestions: Additional pattern suggestions
        execution_time_ms: Execution time
    """

    pattern_description: str
    source_language: Optional[str]
    target_languages: List[str]
    equivalences: List[PatternEquivalence]
    suggestions: List[str] = field(default_factory=list)
    execution_time_ms: int = 0


# =============================================================================
# Language Conversion Models
# =============================================================================


@dataclass
class TypeMapping:
    """Type mapping between languages.

    Attributes:
        source_type: Type in source language
        target_type: Type in target language
        needs_import: Import statement needed (if any)
        notes: Conversion notes
    """

    source_type: str
    target_type: str
    needs_import: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class ConversionWarning:
    """Warning generated during code conversion.

    Attributes:
        severity: Warning severity (info, warning, error)
        message: Warning message
        line_number: Line number in source code
        suggestion: Suggested resolution
    """

    severity: str
    message: str
    line_number: Optional[int] = None
    suggestion: Optional[str] = None


@dataclass
class ConvertedCode:
    """Result of code conversion.

    Attributes:
        source_code: Original source code
        converted_code: Converted code
        from_language: Source language
        to_language: Target language
        style: Conversion style used
        type_mappings: Type mappings applied
        warnings: Conversion warnings
        imports_needed: Import statements needed
        success: Whether conversion was successful
    """

    source_code: str
    converted_code: str
    from_language: str
    to_language: str
    style: ConversionStyle
    type_mappings: List[TypeMapping] = field(default_factory=list)
    warnings: List[ConversionWarning] = field(default_factory=list)
    imports_needed: List[str] = field(default_factory=list)
    success: bool = True


@dataclass
class ConversionResult:
    """Complete result of language conversion operation.

    Attributes:
        conversions: List of converted code blocks
        total_functions: Total functions converted
        successful_conversions: Number of successful conversions
        failed_conversions: Number of failed conversions
        execution_time_ms: Execution time
    """

    conversions: List[ConvertedCode]
    total_functions: int = 0
    successful_conversions: int = 0
    failed_conversions: int = 0
    execution_time_ms: int = 0


# =============================================================================
# Polyglot Refactoring Models
# =============================================================================


@dataclass
class PolyglotChange:
    """A change in a specific language during polyglot refactoring.

    Attributes:
        language: Programming language
        file_path: Path to file
        line_number: Line number of change
        original_code: Original code
        new_code: New code after refactoring
        change_type: Type of change made
    """

    language: str
    file_path: str
    line_number: int
    original_code: str
    new_code: str
    change_type: str = "modify"


@dataclass
class PolyglotRefactoringPlan:
    """Plan for polyglot refactoring.

    Attributes:
        refactoring_type: Type of refactoring
        symbol_name: Symbol being refactored
        new_name: New name (if rename)
        affected_languages: Languages that will be modified
        changes: Planned changes by language
        risks: Identified risks
        requires_manual_review: Files requiring manual review
    """

    refactoring_type: RefactoringType
    symbol_name: str
    new_name: Optional[str]
    affected_languages: List[str]
    changes: List[PolyglotChange] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    requires_manual_review: List[str] = field(default_factory=list)


@dataclass
class PolyglotRefactoringResult:
    """Result of polyglot refactoring operation.

    Attributes:
        plan: The refactoring plan
        dry_run: Whether this was a preview only
        changes_made: List of changes made
        files_modified: Files that were modified
        validation_passed: Whether validation passed
        validation_errors: Any validation errors
        execution_time_ms: Execution time
        backup_id: Backup of the modified files, for rollback (None on dry run)
    """

    plan: PolyglotRefactoringPlan
    dry_run: bool
    changes_made: List[PolyglotChange] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    validation_passed: bool = True
    validation_errors: List[str] = field(default_factory=list)
    execution_time_ms: int = 0
    backup_id: Optional[str] = None


# =============================================================================
# API Binding Generation Models
# =============================================================================


@dataclass
class ApiEndpoint:
    """An API endpoint parsed from specification.

    Attributes:
        path: Endpoint path
        method: HTTP method
        operation_id: Operation identifier
        summary: Short summary
        description: Full description
        parameters: Endpoint parameters
        request_body: Request body schema
        responses: Response schemas
        tags: API tags
    """

    path: str
    method: str
    operation_id: str
    summary: str = ""
    description: str = ""
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    request_body: Optional[Dict[str, Any]] = None
    responses: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class GeneratedBinding:
    """A generated API client binding.

    Attributes:
        language: Target language
        file_name: Suggested file name
        code: Generated code
        imports: Required imports
        dependencies: Package dependencies needed
        types_generated: Type definitions generated
    """

    language: str
    file_name: str
    code: str
    imports: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    types_generated: List[str] = field(default_factory=list)


@dataclass
class BindingGenerationResult:
    """Result of API binding generation.

    Attributes:
        api_name: Name of the API
        api_version: API version
        base_url: Base URL of the API
        endpoints_count: Number of endpoints processed
        bindings: Generated bindings per language
        warnings: Generation warnings
        execution_time_ms: Execution time
    """

    api_name: str
    api_version: str
    base_url: str
    endpoints_count: int
    bindings: List[GeneratedBinding]
    warnings: List[str] = field(default_factory=list)
    execution_time_ms: int = 0


# =============================================================================
# Supported Language Pairs
# =============================================================================

# Language pairs supported for conversion
SUPPORTED_CONVERSION_PAIRS = [
    ("python", "typescript"),
    ("python", "javascript"),
    ("typescript", "python"),
    ("javascript", "python"),
    ("javascript", "typescript"),
    ("java", "kotlin"),
]

# Languages supported for multi-language search
SUPPORTED_LANGUAGES = [
    "python",
    "typescript",
    "javascript",
    "java",
    "kotlin",
    "go",
    "rust",
    "c",
    "cpp",
    "csharp",
    "ruby",
    "php",
    "swift",
]

# Languages supported for binding generation
BINDING_LANGUAGES = [
    "python",
    "typescript",
    "javascript",
    "java",
    "go",
    "rust",
]
