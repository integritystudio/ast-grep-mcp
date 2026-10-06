"""Zod-specific rewrite rule templates for common patterns and best practices.

This module provides pre-built AST-grep rewrite rules for enforcing Zod best practices
and import patterns. Use with rewrite_code() tool.
"""

from typing import Dict, Literal

# Zod Schema Validation Rules
ZOD_SCHEMA_RULES: Dict[str, str] = {
    "no-any-schema": """id: no-any-schema
language: typescript
rule:
  pattern: z.any()
fix: z.unknown().catch({})
message: "Disallow z.any() - use z.unknown().catch({}) instead"
severity: error
""",
    "require-schema-suffix": """id: require-schema-suffix
language: typescript
rule:
  pattern: const $NAME = z.object($$ARGS)
fix: const $${NAME}Schema = z.object($$ARGS)
message: "Schema variables must end with 'Schema' suffix"
severity: warning
""",
    "require-error-message": """id: require-error-message
language: typescript
rule:
  pattern: $SCHEMA.refine($CB)
fix: $SCHEMA.refine($CB, { message: "Validation failed" })
message: "refine() must include error message in second argument"
severity: error
""",
    "prefer-enum-over-literal-union": """id: prefer-enum-over-literal-union
language: typescript
rule:
  pattern: z.union([z.literal($LIT1), z.literal($LIT2)])
fix: z.enum([$LIT1, $LIT2])
message: "Use z.enum() instead of union of literals"
severity: warning
""",
    "no-optional-and-default-together": """id: no-optional-and-default-together
language: typescript
rule:
  pattern: $SCHEMA.optional().default($VAL)
fix: $SCHEMA.default($VAL)
message: "Don't use both .optional() and .default() - just use .default()"
severity: error
""",
    "no-string-schema-with-uuid": """id: no-string-schema-with-uuid
language: typescript
rule:
  pattern: z.string().uuid()
fix: z.uuid()
message: "Use z.uuid() instead of z.string().uuid()"
severity: warning
""",
    "prefer-string-schema-with-trim": """id: prefer-string-schema-with-trim
language: typescript
rule:
  pattern: z.string()
  inside: z.object($$ARGS)
fix: z.string().trim()
message: "Consider adding .trim() to string schemas to prevent whitespace issues"
severity: info
""",
    "no-empty-custom-schema": """id: no-empty-custom-schema
language: typescript
rule:
  pattern: z.custom()
fix: z.custom((val) => true)
message: "z.custom() requires an implementation function"
severity: error
""",
}

# Zod Import Rules
ZOD_IMPORT_RULES: Dict[str, str] = {
    "import-zod-named-to-namespace": """id: import-zod-named-to-namespace
language: typescript
rule:
  pattern: import { z } from "zod"
fix: import * as z from "zod"
message: "Use namespace import for better tree-shaking and bundle optimization"
severity: warning
""",
    "import-zod-default-to-namespace": """id: import-zod-default-to-namespace
language: typescript
rule:
  pattern: import z from "zod"
fix: import * as z from "zod"
message: "Use namespace import for better tree-shaking and bundle optimization"
severity: warning
""",
    "import-zod-type-default-to-namespace": """id: import-zod-type-default-to-namespace
language: typescript
rule:
  pattern: import type z from "zod"
fix: import type * as z from "zod"
message: "Use namespace type import for consistency"
severity: warning
""",
    "import-zod-mixed-to-namespace": """id: import-zod-mixed-to-namespace
language: typescript
rule:
  pattern: import z, { $$EXPORTS } from "zod"
fix: import * as z from "zod"; import { $$EXPORTS } from "zod"
message: "Separate namespace and named imports"
severity: warning
""",
    "import-zod-subpath-to-namespace": r"""id: import-zod-subpath-to-namespace
language: typescript
rule:
  kind: import_statement
  regex: ^import\s*\{\s*z\s*\}
  has:
    field: source
    pattern: $SRC
    regex: ^["']zod/
fix: import * as z from $SRC;
message: "Use namespace import for zod subpath"
severity: warning
""",
}

# Combined all rules
ALL_ZOD_RULES: Dict[str, str] = {**ZOD_SCHEMA_RULES, **ZOD_IMPORT_RULES}


def get_zod_rule(rule_id: str) -> str:
    """Get a pre-built Zod rule template by ID.

    Args:
        rule_id: Rule identifier (e.g., 'no-any-schema', 'import-zod-namespace')

    Returns:
        YAML rule string ready to use with rewrite_code()

    Raises:
        KeyError: If rule_id not found

    Example:
        >>> rule = get_zod_rule('no-any-schema')
        >>> result = rewrite_code_impl('/path/to/project', rule, dry_run=True)
    """
    if rule_id not in ALL_ZOD_RULES:
        available = sorted(ALL_ZOD_RULES.keys())
        raise KeyError(f"Rule '{rule_id}' not found. Available rules: {available}")
    return ALL_ZOD_RULES[rule_id]


def list_schema_rules() -> Dict[str, str]:
    """List all available Zod schema validation rules.

    Returns:
        Dict mapping rule_id to rule description
    """
    return {
        "no-any-schema": "Disallow z.any() - use z.unknown().catch({}) instead",
        "require-schema-suffix": "Require 'Schema' suffix on schema variable names",
        "require-error-message": "Require error message in refine() callbacks",
        "prefer-enum-over-literal-union": "Use z.enum() instead of union of literals",
        "no-optional-and-default-together": "Don't use both .optional() and .default()",
        "no-string-schema-with-uuid": "Use z.uuid() instead of z.string().uuid()",
        "prefer-string-schema-with-trim": "Add .trim() to string schemas in objects",
        "no-empty-custom-schema": "Require implementation function in z.custom()",
    }


def list_import_rules() -> Dict[str, str]:
    """List all available Zod import rules.

    Returns:
        Dict mapping rule_id to rule description
    """
    return {
        "import-zod-named-to-namespace": "Convert import { z } to import * as z",
        "import-zod-default-to-namespace": "Convert import z to import * as z",
        "import-zod-type-default-to-namespace": "Convert import type z to import type * as z",
        "import-zod-mixed-to-namespace": "Separate namespace and named imports",
        "import-zod-subpath-to-namespace": "Convert subpath named imports to namespace",
    }


def get_rules_by_category(category: Literal["schema", "import", "all"]) -> Dict[str, str]:
    """Get Zod rules by category.

    Args:
        category: 'schema' for validation rules, 'import' for import rules, 'all' for both

    Returns:
        Dict mapping rule_id to YAML rule string
    """
    if category == "schema":
        return ZOD_SCHEMA_RULES
    elif category == "import":
        return ZOD_IMPORT_RULES
    elif category == "all":
        return ALL_ZOD_RULES
    else:
        raise ValueError(f"Invalid category: {category}. Use 'schema', 'import', or 'all'")
