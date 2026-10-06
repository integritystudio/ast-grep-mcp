"""
Code generation module for deduplication refactoring.

This module provides functionality for generating refactored code,
including function extraction, parameter inference, and import generation.
"""

import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

from ...constants import RegexCaptureGroups, SemanticVolumeDefaults
from ...core.logging import get_logger
from ...utils.parsing import detect_triple_quote, skip_blank_lines
from ...utils.text import indent_lines

# Language-specific configuration for function generation
FUNCTION_GENERATORS: Dict[str, Callable[..., str]] = {}

# Language-specific configuration for type inference
TYPE_INFERENCE_CONFIG: Dict[str, Dict[str, Dict[str, Union[str, bool]]]] = {
    "python": {
        "literal": {"string": "str", "number_int": "int", "number_float": "float", "boolean": "bool"},
        "identifier": {"default": "Any"},
        "type": {"use_value": True},
    },
    "typescript": {
        "literal": {"string": "string", "number": "number", "boolean": "boolean"},
        "identifier": {"default": "any"},
        "type": {"use_value": False},
    },
    "javascript": {
        "literal": {"string": "string", "number": "number", "boolean": "boolean"},
        "identifier": {"default": "any"},
        "type": {"use_value": False},
    },
    "java": {
        "literal": {"string": "String", "number_int": "int", "number_float": "double", "boolean": "boolean"},
        "identifier": {"default": "Object"},
        "type": {"use_value": True},
    },
}

# Identifier naming patterns for type inference
IDENTIFIER_TYPE_PATTERNS: Dict[str, List[Tuple[str, str]]] = {
    "python": [
        ("id", "int"),
        ("_id", "int"),
        ("count", "int"),
        ("num", "int"),
        ("index", "int"),
        ("idx", "int"),
        ("size", "int"),
        ("length", "int"),
        ("len", "int"),
        ("name", "str"),
        ("title", "str"),
        ("label", "str"),
        ("text", "str"),
        ("message", "str"),
        ("msg", "str"),
        ("path", "str"),
        ("url", "str"),
        ("uri", "str"),
        ("email", "str"),
        ("is_", "bool"),
        ("has_", "bool"),
        ("can_", "bool"),
        ("should_", "bool"),
        ("enabled", "bool"),
        ("disabled", "bool"),
        ("active", "bool"),
        ("valid", "bool"),
        ("flag", "bool"),
        ("items", "List"),
        ("list", "List"),
        ("data", "Dict"),
        ("dict", "Dict"),
        ("map", "Dict"),
        ("config", "Dict"),
    ],
    "typescript": [
        ("id", "number"),
        ("Id", "number"),
        ("count", "number"),
        ("num", "number"),
        ("index", "number"),
        ("idx", "number"),
        ("size", "number"),
        ("length", "number"),
        ("name", "string"),
        ("title", "string"),
        ("label", "string"),
        ("text", "string"),
        ("message", "string"),
        ("path", "string"),
        ("url", "string"),
        ("email", "string"),
        ("is", "boolean"),
        ("has", "boolean"),
        ("can", "boolean"),
        ("should", "boolean"),
        ("enabled", "boolean"),
        ("disabled", "boolean"),
        ("active", "boolean"),
        ("valid", "boolean"),
        ("items", "Array"),
        ("list", "Array"),
        ("data", "object"),
        ("config", "object"),
    ],
}


def _infer_from_identifier_name(identifier: str, language: str) -> str:
    """Infer type from identifier naming conventions.

    Analyzes the identifier name to guess its type based on common
    naming patterns (e.g., 'user_id' suggests int, 'is_valid' suggests bool).

    Args:
        identifier: The identifier name to analyze
        language: Programming language for type syntax

    Returns:
        Inferred type string for the language
    """
    identifier_lower = identifier.lower()

    # Get language-specific patterns, fall back to python
    patterns = IDENTIFIER_TYPE_PATTERNS.get(language, IDENTIFIER_TYPE_PATTERNS.get("python", []))

    # Check each pattern
    for pattern, inferred_type in patterns:
        if pattern in identifier_lower or identifier_lower.startswith(pattern) or identifier_lower.endswith(pattern):
            return inferred_type

    # Fall back to default type for the language
    config = TYPE_INFERENCE_CONFIG.get(language, TYPE_INFERENCE_CONFIG.get("python", {}))
    return str(config.get("identifier", {}).get("default", "Any"))


# Null/undefined type mappings by language
NULL_TYPE_MAP: Dict[str, Dict[str, str]] = {
    "python": {"None": "None", "null": "None", "nil": "None", "undefined": "None"},
    "typescript": {"None": "null", "null": "null", "nil": "null", "undefined": "undefined"},
    "javascript": {"None": "null", "null": "null", "nil": "null", "undefined": "undefined"},
}

# Collection type mappings by language
COLLECTION_TYPE_MAP: Dict[str, Dict[str, str]] = {
    "python": {"list": "List", "dict": "Dict"},
    "typescript": {"list": "Array", "dict": "object"},
    "javascript": {"list": "Array", "dict": "object"},
}


def _is_boolean_literal(value: str) -> bool:
    """Check if value is a boolean literal."""
    return value in ("True", "False", "true", "false")


def _is_null_literal(value: str) -> bool:
    """Check if value is a null/None/undefined literal."""
    return value in ("None", "null", "nil", "undefined")


def _is_quoted_string(value: str) -> bool:
    """Check if value is a quoted string literal."""
    return (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'"))


def _is_integer_literal(value: str) -> bool:
    """Check if value is an integer literal."""
    return value.lstrip("-").isdigit()


def _try_get_float_type(value: str, literal_types: Dict[str, Any]) -> Optional[str]:
    """Try to parse value as float and return appropriate type."""
    try:
        float(value)
        if "." in value or "e" in value.lower():
            return str(literal_types.get("number_float", literal_types.get("number", "float")))
        return str(literal_types.get("number_int", literal_types.get("number", "int")))
    except ValueError:
        return None


def _get_collection_type(value: str, language: str) -> Optional[str]:
    """Get collection type for list/dict literals."""
    lang_types = COLLECTION_TYPE_MAP.get(language, COLLECTION_TYPE_MAP["python"])
    if value.startswith("[") and value.endswith("]"):
        return lang_types["list"]
    if value.startswith("{") and value.endswith("}"):
        return lang_types["dict"]
    return None


def _infer_single_value_type(value: str, language: str) -> str:
    """Infer type from a single literal value.

    Analyzes the value to determine its type based on its format
    (e.g., '123' is int, '"hello"' is string, 'True' is bool).

    Args:
        value: The literal value as a string
        language: Programming language for type syntax

    Returns:
        Inferred type string for the language
    """
    value_stripped = value.strip()

    config = TYPE_INFERENCE_CONFIG.get(language, TYPE_INFERENCE_CONFIG.get("python", {}))
    literal_types = config.get("literal", {})

    if _is_boolean_literal(value_stripped):
        return str(literal_types.get("boolean", "bool"))

    if _is_null_literal(value_stripped):
        null_map = NULL_TYPE_MAP.get(language, NULL_TYPE_MAP.get("python", {}))
        return null_map.get(value_stripped, "null")

    if _is_quoted_string(value_stripped):
        return str(literal_types.get("string", "str"))

    if _is_integer_literal(value_stripped):
        return str(literal_types.get("number_int", literal_types.get("number", "int")))

    float_type = _try_get_float_type(value_stripped, literal_types)
    if float_type:
        return float_type

    collection_type = _get_collection_type(value_stripped, language)
    if collection_type:
        return collection_type

    return str(literal_types.get("string", "str"))


class CodeGenerator:
    """Generates refactored code for deduplication."""

    def __init__(self, language: str = "python") -> None:
        """Initialize the code generator.

        Args:
            language: Programming language for code generation
        """
        self.language = language
        self.logger = get_logger("deduplication.generator")

    def generate_function_call(self, function_name: str, arguments: List[str], assign_to: Optional[str] = None) -> str:
        """
        Generate a function call to replace duplicated code.

        Args:
            function_name: Name of the function to call
            arguments: List of argument values
            assign_to: Optional variable to assign result to

        Returns:
            Generated function call code
        """
        call = f"{function_name}({', '.join(arguments)})"

        if assign_to:
            assign_formats = {
                "javascript": f"const {assign_to} = {call};",
                "typescript": f"const {assign_to} = {call};",
                "python": f"{assign_to} = {call}",
            }
            return assign_formats.get(self.language, f"{assign_to} = {call};")

        return call if self.language == "python" else f"{call};"

    def _generate_python_import(self, module_path: str, import_names: List[str], is_relative: bool) -> str:
        """Generate Python import statement."""
        names = ", ".join(import_names)
        if is_relative:
            prefix = "." if module_path else ""
            return f"from {prefix}{module_path} import {names}"
        if module_path:
            return f"from {module_path} import {names}"
        return f"import {names}"

    def _generate_js_import(self, module_path: str, import_names: List[str], is_relative: bool) -> str:
        """Generate JavaScript/TypeScript import statement."""
        names = "{" + ", ".join(import_names) + "}"
        extension = ".js" if self.language == "javascript" else ""
        path = f"./{module_path}" if is_relative else module_path
        return f"import {names} from '{path}{extension}';"

    def generate_import_statement(self, module_path: str, import_names: List[str], is_relative: bool = False) -> str:
        """
        Generate an import statement for the extracted function.

        Args:
            module_path: Module or file path to import from
            import_names: Names to import
            is_relative: Whether to use relative import

        Returns:
            Generated import statement
        """
        if self.language == "python":
            return self._generate_python_import(module_path, import_names, is_relative)
        if self.language in ["javascript", "typescript"]:
            return self._generate_js_import(module_path, import_names, is_relative)
        if self.language == "java":
            return f"import {module_path}.{import_names[0]};"
        return f"// Import {', '.join(import_names)} from {module_path}"

    def _collect_variation_params(self, code_variations: List[Dict[str, Any]], seen_names: set[str]) -> List[Dict[str, str]]:
        """Collect parameters from code variations, deduplicating by name."""
        params: List[Dict[str, str]] = []
        for variation in code_variations:
            param_name = variation.get("suggested_param_name")
            if param_name and param_name not in seen_names:
                params.append({"name": param_name, "type": self._infer_parameter_type(variation) or "Any"})
                seen_names.add(param_name)
        return params

    def _collect_external_params(self, base_code: str, seen_names: set[str]) -> List[Dict[str, str]]:
        """Collect parameters from external dependencies not already seen."""
        params: List[Dict[str, str]] = []
        for dep in self._find_external_dependencies(base_code):
            if dep not in seen_names:
                params.append({"name": dep, "type": "Any"})
                seen_names.add(dep)
        return params

    def infer_function_parameters(self, code_variations: List[Dict[str, Any]], base_code: str) -> List[Dict[str, str]]:
        """
        Infer function parameters from code variations.

        Args:
            code_variations: List of code variations
            base_code: Base code to analyze

        Returns:
            List of inferred parameters
        """
        seen_names: set[str] = set()
        parameters = self._collect_variation_params(code_variations, seen_names)
        parameters.extend(self._collect_external_params(base_code, seen_names))
        return parameters

    def _get_python_number_type(self, value: str, lang_config: Dict[str, Any]) -> Optional[str]:
        """Get Python-specific number type (int vs float)."""
        if "." not in value:
            return cast(Optional[str], lang_config.get("number_int", "int"))
        return cast(Optional[str], lang_config.get("number_float", "float"))

    def _get_literal_type(self, literal_type: str, value: str, lang_config: Dict[str, Any]) -> Optional[str]:
        """Get type mapping for literal values."""
        if literal_type == "string":
            return cast(Optional[str], lang_config.get("string"))
        if literal_type == "boolean":
            return cast(Optional[str], lang_config.get("boolean"))
        if literal_type == "number":
            if self.language == "python":
                return self._get_python_number_type(value, lang_config)
            return cast(Optional[str], lang_config.get("number"))
        return None

    def _infer_literal_type(self, variation: Dict[str, Any], value: str, category_config: Dict[str, Any]) -> Optional[str]:
        """Infer type for a literal variation."""
        literal_type = variation.get("literal_type", "")
        return self._get_literal_type(literal_type, value, category_config)

    def _infer_type_category(self, value: str, category_config: Dict[str, Any]) -> Optional[str]:
        """Infer type for a 'type' category variation."""
        if category_config.get("use_value"):
            return str(value) if value else None
        return None

    def _infer_parameter_type(self, variation: Dict[str, Any]) -> Optional[str]:
        """Infer parameter type from variation."""
        category = variation.get("category", "")
        value = variation.get("old_value", "")

        lang_config = TYPE_INFERENCE_CONFIG.get(self.language, {})
        if not lang_config:
            return None

        category_config = lang_config.get(category, {})
        if not category_config:
            return None

        if category == "literal":
            return self._infer_literal_type(variation, value, category_config)
        if category == "identifier":
            return cast(Optional[str], category_config.get("default"))
        if category == "type":
            return self._infer_type_category(value, category_config)
        return None

    def _find_external_dependencies(self, code: str) -> List[str]:
        """Find external dependencies that should be parameters."""
        dependencies: List[str] = []

        # Simple heuristic: find identifiers that look like they come from outside
        # This is language-specific and simplified
        if self.language == "python":
            # Look for common patterns like self.x, module.function, etc.
            patterns = [
                r"\bself\.(\w+)",  # Class attributes
                r"(?<![\w.])(\w+)\(",  # Function calls (not methods or name tails)
            ]
            for pattern in patterns:
                matches = re.findall(pattern, code)
                dependencies.extend(m for m in matches if m not in ["print", "len", "range", "str", "int"])

        return list(set(dependencies))[: SemanticVolumeDefaults.TOP_RESULTS_LIMIT]  # Limit to top dependencies

    def _module_docstring_lines(self, module_name: str) -> List[str]:
        """Generate language-specific module docstring lines."""
        desc = f"{module_name} - Extracted common functionality"
        if self.language == "python":
            return ['"""', desc, '"""', ""]
        if self.language in ["javascript", "typescript"]:
            return ["/**", f" * {desc}", " */", ""]
        return []

    def generate_module_structure(self, module_name: str, functions: List[Dict[str, str]], imports: Optional[List[str]] = None) -> str:
        """
        Generate a complete module with extracted functions.

        Args:
            module_name: Name of the module
            functions: List of function definitions
            imports: Optional list of import statements

        Returns:
            Complete module code
        """
        module_code: List[str] = self._module_docstring_lines(module_name)

        if imports:
            module_code.extend(imports)
            module_code.append("")

        for func in functions:
            module_code.append(func.get("code", ""))
            module_code.append("")

        if self.language in ["javascript", "typescript"]:
            export_names = [f.get("name", "") for f in functions]
            if export_names:
                module_code.append(f"export {{ {', '.join(export_names)} }};")

        return "\n".join(module_code)

    def _validate_python_code(self, code: str) -> Tuple[bool, Optional[str]]:
        """Validate Python code syntax."""
        try:
            compile(code, "<string>", "exec")
            return True, None
        except SyntaxError as e:
            return False, str(e)

    def _validate_js_ts_code(self, code: str) -> Tuple[bool, Optional[str]]:
        """Validate JavaScript/TypeScript code via bracket matching."""
        if code.count("{") != code.count("}"):
            return False, "Mismatched braces"
        if code.count("(") != code.count(")"):
            return False, "Mismatched parentheses"
        return True, None

    def validate_generated_code(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate generated code for syntax errors.

        Args:
            code: Generated code to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if self.language == "python":
            return self._validate_python_code(code)
        if self.language in ["javascript", "typescript"]:
            return self._validate_js_ts_code(code)
        return True, None

    def _reindent_body(self, body: str) -> List[str]:
        """Re-indent body lines to 4-space base indent, preserving relative indentation."""
        body_lines = body.split("\n")
        base_indent = len(body_lines[0]) - len(body_lines[0].lstrip()) if body_lines else 0
        result = []
        for line in body_lines:
            stripped = line.lstrip()
            if stripped:
                original_indent = len(line) - len(stripped)
                relative_indent = original_indent - base_indent
                indent = "    " + " " * max(0, relative_indent)
                result.append(indent + stripped)
            else:
                result.append("")
        return result

    def render_python_function(
        self,
        name: str,
        params: str,
        body: str,
        return_type: Optional[str] = None,
        docstring: Optional[str] = None,
        decorators: Optional[List[str]] = None,
    ) -> str:
        """
        Render a Python function from components.

        Args:
            name: Function name
            params: Formatted parameter string
            body: Function body code
            return_type: Optional return type annotation
            docstring: Optional function docstring
            decorators: Optional list of decorator strings

        Returns:
            Complete Python function code
        """
        lines: List[str] = []

        if decorators:
            lines.extend(f"@{d}" for d in decorators)

        return_annotation = f" -> {return_type}" if return_type else ""
        lines.append(f"def {name}({params}){return_annotation}:")

        if docstring:
            lines.append(f'    """{docstring}"""')

        lines.extend(self._reindent_body(body))

        return "\n".join(lines)

    def _process_conditional(self, match: re.Match[str], variables: Dict[str, str]) -> str:
        """Process conditional template blocks."""
        condition_type = match.group(RegexCaptureGroups.FIRST)  # 'if' or 'unless'
        var_name = match.group(RegexCaptureGroups.SECOND)
        content = match.group(RegexCaptureGroups.THIRD)

        var_value = variables.get(var_name, "")
        is_truthy = var_value and var_value.lower() not in ("false", "0", "")

        if condition_type == "if":
            return content if is_truthy else ""
        return "" if is_truthy else content  # unless

    def _process_each_loop(self, match: re.Match[str], variables: Dict[str, str], strict: bool) -> str:
        """Process each loop template blocks."""
        var_name = match.group(RegexCaptureGroups.FIRST)
        loop_content = match.group(RegexCaptureGroups.SECOND)

        if var_name not in variables:
            if strict:
                raise ValueError(f"Missing list variable: {var_name}")
            return ""

        items = variables[var_name].split(",") if variables[var_name] else []
        result = "".join(loop_content.replace("{{.}}", item.strip()) for item in items)

        # Remove trailing space if the loop content ends with a space
        if loop_content.endswith(" ") and result.endswith(" "):
            result = result.rstrip()
        return result

    def _substitute_simple_variable(self, match: re.Match[str], variables: Dict[str, str], strict: bool) -> str:
        """Substitute simple template variable."""
        var_name = match.group(RegexCaptureGroups.FIRST)
        if var_name not in variables:
            if strict:
                raise ValueError(f"Missing required variable: {var_name}")
            return ""
        return variables[var_name]

    def substitute_template_variables(self, template: str, variables: Dict[str, str], strict: bool = False) -> str:
        """
        Substitute template variables in a string template.

        Supports:
        - Simple variables: {{name}}
        - Conditionals: {{#if var}}...{{/if}}, {{#unless var}}...{{/unless}}
        - Loops: {{#each items}}{{.}}{{/each}}

        Args:
            template: Template string with {{variable}} placeholders
            variables: Dictionary of variable values
            strict: If True, raise error on missing variables

        Returns:
            Template with variables substituted

        Raises:
            ValueError: If strict=True and required variable is missing
        """
        result = template

        # Process conditionals (if/unless)
        result = re.sub(
            r"\{\{#(if|unless)\s+(\w+)\}\}(.*?)\{\{/\1\}\}", lambda m: self._process_conditional(m, variables), result, flags=re.DOTALL
        )

        # Process loops {{#each items}}{{.}}{{/each}}
        result = re.sub(
            r"\{\{#each\s+(\w+)\}\}(.*?)\{\{/each\}\}", lambda m: self._process_each_loop(m, variables, strict), result, flags=re.DOTALL
        )

        # Process simple variables {{var}}
        result = re.sub(r"\{\{(\w+)\}\}", lambda m: self._substitute_simple_variable(m, variables, strict), result)

        return result

    def preserve_call_site_indentation(self, original_code: str, replacement: str) -> str:
        """
        Preserve the indentation from the original code when replacing it.

        Args:
            original_code: Original code with indentation
            replacement: Replacement code to indent

        Returns:
            Replacement code with original indentation preserved
        """
        if not original_code or not replacement:
            return replacement

        # Extract indentation from original
        indent = ""
        for char in original_code:
            if char in " \t":
                indent += char
            else:
                break

        # Apply indentation to replacement
        if "\n" in replacement:
            return "\n".join(indent_lines(replacement, prefix=indent))
        else:
            return indent + replacement

    def detect_import_insertion_point(self, content: str, language: str) -> int:
        """
        Detect where to insert import statements in a file.

        Args:
            content: File content
            language: Programming language

        Returns:
            Line number (1-indexed) where imports should be inserted
        """
        lines = content.split("\n")

        if language == "python":
            return self._detect_python_import_point(lines)
        elif language in ["javascript", "typescript"]:
            return self._detect_js_import_point(lines)
        elif language == "java":
            return self._detect_java_import_point(lines)
        else:
            # Generic: after first line
            return 1

    def _detect_python_import_point(self, lines: List[str]) -> int:
        """Find import insertion point for Python files."""
        # Handle empty file
        if not lines or (len(lines) == 1 and not lines[0].strip()):
            return 1

        # Skip module docstring (if present)
        position = self._skip_python_module_docstring(lines, start=0)

        # Skip top-level comments
        position = self._skip_python_comments(lines, start=position)

        # Find end of existing imports
        last_import_line = self._find_last_import_line(lines, start=position)

        return last_import_line + 1

    def _process_docstring_scan_line(self, line: str, i: int, docstring_quotes: Optional[str]) -> Tuple[Optional[int], Optional[str]]:
        """Process one line of docstring scanning. Returns (result_pos_or_None, updated_docstring_quotes)."""
        quotes = detect_triple_quote(line)
        if docstring_quotes is None:
            if quotes is None:
                return i, None
            if line.count(quotes) >= 2:
                return i + 1, None
            return None, quotes
        if quotes == docstring_quotes:
            return i + 1, None
        return None, docstring_quotes

    def _skip_python_module_docstring(self, lines: List[str], start: int) -> int:
        """Skip module-level docstring and return position after it."""
        docstring_quotes: Optional[str] = None

        for i, raw_line in enumerate(lines[start:], start=start):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            result, docstring_quotes = self._process_docstring_scan_line(line, i, docstring_quotes)
            if result is not None:
                return result

        return len(lines)

    def _skip_python_comments(self, lines: List[str], start: int) -> int:
        """Skip comment lines and return position after comments.

        Args:
            lines: List of file lines
            start: Starting position

        Returns:
            Position after comments
        """
        position = start
        while position < len(lines) and lines[position].strip().startswith("#"):
            position += 1
        return position

    def _find_last_import_line(self, lines: List[str], start: int) -> int:
        """Find the last import statement line.

        Args:
            lines: List of file lines
            start: Starting position

        Returns:
            Position of last import (or start if no imports found)
        """
        position = start
        last_import = start

        while position < len(lines):
            line = lines[position].strip()

            # Track import statements
            if self._is_import_line(line):
                last_import = position + 1
            # Stop at first non-import, non-comment, non-empty line
            elif line and not line.startswith("#"):
                break

            position += 1

        return last_import

    def _is_import_line(self, line: str) -> bool:
        """Check if line is an import statement.

        Args:
            line: Stripped line to check

        Returns:
            True if line is an import statement
        """
        return line.startswith("import ") or line.startswith("from ")

    _USE_STRICT_DIRECTIVES = ('"use strict";', "'use strict';")

    def _skip_use_strict(self, lines: List[str], start: int) -> int:
        """Skip 'use strict' directive and following blank line. Returns new position."""
        i = start
        while i < len(lines):
            line = lines[i].strip()
            if line in self._USE_STRICT_DIRECTIVES:
                i += 1
                return skip_blank_lines(lines, i)
            if line:
                return i
            i += 1
        return i

    def _is_js_import_line(self, line: str) -> bool:
        """Check if line is a JS/TS import or require statement."""
        return (
            line.startswith("import ")
            or (line.startswith("const ") and "require(" in line)
            or (line.startswith("var ") and "require(" in line)
        )

    def _detect_js_import_point(self, lines: List[str]) -> int:
        """Find import insertion point for JavaScript/TypeScript files."""
        i = self._skip_use_strict(lines, 0)

        last_import = i
        while i < len(lines):
            line = lines[i].strip()
            if self._is_js_import_line(line):
                last_import = i + 1
            elif line and not line.startswith("//"):
                break
            i += 1

        return last_import + 1

    def _skip_java_package(self, lines: List[str], start: int) -> int:
        """Skip Java package declaration and blank lines after it. Returns new position."""
        i = start
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("package "):
                return skip_blank_lines(lines, i + 1)
            if line and not line.startswith("//"):
                return i
            i += 1
        return i

    def _detect_java_import_point(self, lines: List[str]) -> int:
        """Find import insertion point for Java files."""
        i = self._skip_java_package(lines, 0)

        last_import = i
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("import "):
                last_import = i + 1
            elif line and not line.startswith("//"):
                break
            i += 1

        return last_import + 1


_PARAM_PREFIXES_TO_STRIP = ["get_", "set_", "is_", "has_", "_"]


def _strip_param_prefix(name: str) -> str:
    """Strip common prefixes from an identifier to form a base parameter name."""
    for prefix in _PARAM_PREFIXES_TO_STRIP:
        if name.startswith(prefix) and len(name) > len(prefix):
            return name[len(prefix) :]
    return name


def _build_param_candidates(base_name: str) -> List[str]:
    """Build ordered list of candidate parameter names from a base name."""
    candidates = [base_name, f"{base_name}_value", f"{base_name}_param", f"new_{base_name}"]
    candidates.extend(f"{base_name}_{i}" for i in range(1, 10))
    return candidates


def generate_parameter_name(identifier: str, all_identifiers: List[str]) -> str:
    """
    Generate a unique parameter name based on an identifier.

    Creates a meaningful parameter name from an identifier, ensuring
    it doesn't conflict with existing identifiers.

    Args:
        identifier: The original identifier to base the parameter name on
        all_identifiers: List of existing identifiers to avoid conflicts

    Returns:
        A unique parameter name
    """
    base_name = _strip_param_prefix(identifier.lower().strip()) or "param"
    existing_lower = {ident.lower() for ident in all_identifiers}
    for candidate in _build_param_candidates(base_name):
        if candidate not in existing_lower:
            return candidate

    import time

    return f"{base_name}_{int(time.time()) % 1000}"


def _find_type_annotation(identifier: str, context: str, language: str) -> Optional[str]:
    """Find explicit type annotation for identifier in context."""
    if language == "python":
        pattern = rf"{re.escape(identifier)}\s*:\s*([A-Za-z_][A-Za-z0-9_\[\],\s]*)"
    elif language in ("typescript", "javascript"):
        pattern = rf"{re.escape(identifier)}\s*:\s*([A-Za-z_][A-Za-z0-9_<>\[\],\s]*)"
    else:
        return None

    match = re.search(pattern, context)
    return match.group(RegexCaptureGroups.FIRST).strip() if match else None


def _get_usage_patterns(identifier: str) -> List[Tuple[str, str, str]]:
    """Get usage patterns for type inference."""
    esc_id = re.escape(identifier)
    return [
        (r"len\s*\(\s*" + esc_id, "Sequence", "Iterable"),
        (r"int\s*\(\s*" + esc_id, "str", "string"),
        (r"str\s*\(\s*" + esc_id, "Any", "any"),
        (r"float\s*\(\s*" + esc_id, "str", "string"),
        (r"\.append\s*\(", "List", "Array"),
        (r"\.items\s*\(\s*\)", "Dict", "object"),
        (r"\.keys\s*\(\s*\)", "Dict", "object"),
        (r"\.values\s*\(\s*\)", "Dict", "object"),
        (r"for\s+\w+\s+in\s+" + esc_id, "Iterable", "Iterable"),
        (r"if\s+" + esc_id + r"\s*:", "bool", "boolean"),
        (r"while\s+" + esc_id + r"\s*:", "bool", "boolean"),
    ]


def _infer_from_usage_patterns(identifier: str, context: str, language: str) -> Optional[str]:
    """Infer type from common usage patterns in context."""
    for pattern, py_type, js_type in _get_usage_patterns(identifier):
        if re.search(pattern, context):
            return py_type if language == "python" else js_type
    return None


def _infer_from_operations(identifier: str, context: str, language: str) -> Optional[str]:
    """Infer type from arithmetic/string operations."""
    esc_id = re.escape(identifier)

    # Check for numeric comparisons
    if re.search(rf"{esc_id}\s*[<>=]+\s*\d", context) or re.search(rf"\d\s*[<>=]+\s*{esc_id}", context):
        return "int" if language == "python" else "number"

    # Check for string concatenation
    if re.search(rf'{esc_id}\s*\+\s*["\']', context) or re.search(rf'["\'\s]\s*\+\s*{esc_id}', context):
        return "str" if language == "python" else "string"

    return None


def infer_parameter_type(identifier: str, context: str, language: str = "python") -> str:
    """
    Infer the type of a parameter from its identifier and usage context.

    Combines identifier name analysis with contextual clues to determine
    the most likely type for a parameter.

    Args:
        identifier: The identifier/parameter name
        context: Surrounding code context where the identifier is used
        language: Programming language (default: python)

    Returns:
        Inferred type string appropriate for the language
    """
    config = TYPE_INFERENCE_CONFIG.get(language, TYPE_INFERENCE_CONFIG.get("python", {}))
    default_type = str(config.get("identifier", {}).get("default", "Any"))

    # Try to infer from identifier name first
    type_from_name = _infer_from_identifier_name(identifier, language)
    if type_from_name != default_type:
        return type_from_name

    # Try explicit type annotation
    annotation = _find_type_annotation(identifier, context, language)
    if annotation:
        return annotation

    # Try usage patterns
    usage_type = _infer_from_usage_patterns(identifier, context, language)
    if usage_type:
        return usage_type

    # Try operations
    op_type = _infer_from_operations(identifier, context, language)
    if op_type:
        return op_type

    return default_type
