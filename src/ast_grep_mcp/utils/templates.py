"""Code generation templates and formatting utilities.

This module provides templates and formatting functions for generating
code in various languages (Python, Java, TypeScript, JavaScript).
Used primarily by the duplication detection system for refactoring.
"""

import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, List, Optional, Tuple

from ast_grep_mcp.constants import SubprocessDefaults
from ast_grep_mcp.utils.text import indent_lines

# =============================================================================
# Python Templates
# =============================================================================

PYTHON_CLASS_TEMPLATE: str = """{decorators}class {name}{bases}:
{docstring}{class_vars}{methods}"""


def _ensure_python_indent(methods: str) -> str:
    if not methods or methods.startswith("    "):
        return methods
    return "\n".join(f"    {line}" if line.strip() else line for line in methods.split("\n"))


def format_python_class(
    name: str,
    methods: str,
    decorators: Optional[str] = None,
    bases: Optional[List[str]] = None,
    docstring: Optional[str] = None,
    class_vars: Optional[str] = None,
) -> str:
    """Format a Python class using the template.

    Args:
        name: Class name
        methods: Formatted method definitions (already indented)
        decorators: Optional decorator strings (e.g., "@dataclass\\n")
        bases: Optional list of base classes for inheritance
        docstring: Optional class-level docstring
        class_vars: Optional class variable definitions

    Returns:
        Formatted Python class string
    """
    return PYTHON_CLASS_TEMPLATE.format(
        decorators=f"{decorators}\n" if decorators else "",
        name=name,
        bases=f"({', '.join(bases)})" if bases else "",
        docstring=f'    """{docstring}"""\n\n' if docstring else "",
        class_vars=f"{class_vars}\n\n" if class_vars else "",
        methods=_ensure_python_indent(methods),
    )


# =============================================================================
# Java Templates
# =============================================================================

JAVA_METHOD_TEMPLATE: str = """{javadoc}{annotations}{modifiers}{type_params}{return_type} {name}({params}){throws} {{
{body}
}}"""


def _try_google_java_format(code: str) -> Optional[str]:
    """Try to format Java code using google-java-format.

    Args:
        code: Java source code to format

    Returns:
        Formatted code if successful, None if tool unavailable or error
    """
    if not shutil.which("google-java-format"):
        return None

    with tempfile.NamedTemporaryFile(mode="w", suffix=".java", delete=False) as f:
        f.write(code)
        temp_path = f.name

    try:
        result = subprocess.run(
            ["google-java-format", temp_path],
            capture_output=True,
            text=True,
            timeout=SubprocessDefaults.AST_GREP_TIMEOUT_SECONDS,
        )
        if result.returncode == 0:
            return result.stdout
    except (subprocess.TimeoutExpired, subprocess.SubprocessError):
        pass
    finally:
        # Also on timeout, which previously skipped cleanup
        try:
            os.unlink(temp_path)
        except OSError:
            pass

    return None


def _java_import_sort_key(imp: str) -> tuple[int, str]:
    name = imp.replace("import ", "").replace("static ", "").strip().rstrip(";")
    if name.startswith("java."):
        return (0, name)
    if name.startswith("javax."):
        return (1, name)
    return (2, name)


def _classify_java_line(
    line: str,
    in_imports: bool,
    import_lines: list[str],
    non_import_lines: list[str],
) -> bool:
    stripped = line.strip()
    if stripped.startswith("import "):
        import_lines.append(stripped)
        return True
    if stripped.startswith("package "):
        non_import_lines.append(stripped)
        return in_imports
    if stripped:
        non_import_lines.append(stripped)
        return False
    if not in_imports:
        non_import_lines.append("")
    return in_imports


def _process_java_imports(lines: list[str]) -> tuple[list[str], list[str]]:
    """Separate and sort Java import statements.

    Args:
        lines: Raw code lines

    Returns:
        Tuple of (sorted_import_lines, non_import_lines)
    """
    import_lines: list[str] = []
    non_import_lines: list[str] = []
    in_imports = False

    for line in lines:
        in_imports = _classify_java_line(line, in_imports, import_lines, non_import_lines)

    import_lines.sort(key=_java_import_sort_key)
    return import_lines, non_import_lines


def _merge_package_imports_code(import_lines: list[str], non_import_lines: list[str]) -> list[str]:
    """Merge package declaration, imports, and code with proper spacing.

    Args:
        import_lines: Sorted import statements
        non_import_lines: Package declaration and code

    Returns:
        Merged lines with proper spacing
    """
    all_lines: list[str] = []
    found_package = False

    for line in non_import_lines:
        if line.startswith("package "):
            all_lines.append(line)
            found_package = True
            if import_lines:
                all_lines.append("")  # Blank line after package
                all_lines.extend(import_lines)
                all_lines.append("")  # Blank line after imports
        else:
            all_lines.append(line)

    if not found_package and import_lines:
        all_lines = import_lines + [""] + all_lines

    return all_lines


def _compute_java_indent(stripped: str, indent_level: int) -> tuple[int, int]:
    """Return (display_indent, new_indent_level) for a Java line."""
    closes = stripped.startswith("}") or stripped.startswith(")")
    if closes:
        indent_level = max(0, indent_level - 1)
    display = indent_level
    open_braces = stripped.count("{") - stripped.count("}")
    new_level = max(0, indent_level + open_braces)
    return display, new_level


def _apply_java_indentation(lines: list[str]) -> str:
    """Apply consistent indentation and brace formatting.

    Args:
        lines: Code lines to format

    Returns:
        Formatted code string with proper indentation
    """
    formatted_lines: list[str] = []
    indent_level = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            formatted_lines.append("")
            continue

        display, indent_level = _compute_java_indent(stripped, indent_level)
        formatted_line = re.sub(r"(\S)\{", r"\1 {", "    " * display + stripped)
        formatted_lines.append(formatted_line)

    return "\n".join(formatted_lines)


def format_java_code(code: str) -> str:
    """Format Java code using google-java-format or basic formatting fallback.

    Attempts to use google-java-format for professional formatting. Falls back
    to basic formatting rules if the tool is not available.

    Args:
        code: Java source code string to format

    Returns:
        Formatted Java code string

    Examples:
        >>> format_java_code("public class Foo{int x;}")
        'public class Foo {\\n    int x;\\n}'
    """
    # Try external formatter first
    formatted = _try_google_java_format(code)
    if formatted is not None:
        return formatted

    # Fall back to basic formatting
    lines = code.split("\n")
    import_lines, non_import_lines = _process_java_imports(lines)
    all_lines = _merge_package_imports_code(import_lines, non_import_lines)
    return _apply_java_indentation(all_lines)


def _format_javadoc(javadoc: Optional[str]) -> str:
    if not javadoc:
        return ""
    lines = javadoc.strip().split("\n")
    body = "".join(f" * {line}\n" for line in lines)
    return f"/**\n{body} */\n"


def _indent_java_body(body: str) -> str:
    return "\n".join(indent_lines(body.strip()))


def format_java_method(
    name: str,
    params: str,
    body: str,
    return_type: str = "void",
    modifiers: Optional[List[str]] = None,
    annotations: Optional[List[str]] = None,
    type_params: Optional[str] = None,
    throws: Optional[List[str]] = None,
    javadoc: Optional[str] = None,
) -> str:
    """Format a Java method using the template.

    Args:
        name: Method name
        params: Parameter list (e.g., "String name, int age")
        body: Method body (will be indented with 4 spaces)
        return_type: Return type (default: "void")
        modifiers: Optional list of modifiers ["public", "static", "final"]
        annotations: Optional list of annotations ["@Override", "@Deprecated"]
        type_params: Optional type parameters (e.g., "<T extends Comparable<T>>")
        throws: Optional list of exceptions ["IOException", "SQLException"]
        javadoc: Optional Javadoc comment (without /** */)

    Returns:
        Formatted Java method string
    """
    return JAVA_METHOD_TEMPLATE.format(
        javadoc=_format_javadoc(javadoc),
        annotations="\n".join(annotations) + "\n" if annotations else "",
        modifiers=" ".join(modifiers) + " " if modifiers else "",
        type_params=type_params + " " if type_params else "",
        return_type=return_type,
        name=name,
        params=params,
        throws=" throws " + ", ".join(throws) if throws else "",
        body=_indent_java_body(body),
    )


# =============================================================================
# TypeScript Templates
# =============================================================================

TYPESCRIPT_CLASS_TEMPLATE: str = """{jsdoc}{export}{abstract}class {name}{type_params}{extends}{implements} {{
{properties}{constructor}{methods}}}
"""

TYPESCRIPT_FUNCTION_TEMPLATE: str = """{jsdoc}{export}function {name}{type_params}({params}){return_type} {{
{body}
}}"""

TYPESCRIPT_ASYNC_FUNCTION_TEMPLATE: str = """{jsdoc}{export}async function {name}{type_params}({params}){return_type} {{
{body}
}}"""

TYPESCRIPT_ARROW_FUNCTION_TEMPLATE: str = """{jsdoc}{export}const {name} = {async}{type_params}({params}){return_type} => {{
{body}
}};"""


def _build_typescript_class_args(
    name: str,
    export: bool,
    abstract: bool,
    type_params: Optional[str],
    extends: Optional[str],
    implements: Optional[List[str]],
    jsdoc: Optional[str],
    properties: Optional[List[str]],
    constructor: Optional[str],
    methods: Optional[List[str]],
) -> dict[str, Any]:
    return {
        "jsdoc": f"{jsdoc}\n" if jsdoc else "",
        "export": "export " if export else "",
        "abstract": "abstract " if abstract else "",
        "name": name,
        "type_params": type_params or "",
        "extends": f" extends {extends}" if extends else "",
        "implements": f" implements {', '.join(implements)}" if implements else "",
        "properties": "\n".join(properties) + "\n\n" if properties else "",
        "constructor": constructor + "\n\n" if constructor else "",
        "methods": "\n\n".join(methods) + "\n" if methods else "",
    }


def format_typescript_class(
    name: str,
    *,
    export: bool = False,
    abstract: bool = False,
    type_params: Optional[str] = None,
    extends: Optional[str] = None,
    implements: Optional[List[str]] = None,
    jsdoc: Optional[str] = None,
    properties: Optional[List[str]] = None,
    constructor: Optional[str] = None,
    methods: Optional[List[str]] = None,
) -> str:
    """Format a TypeScript class using the template.

    Args:
        name: Class name
        export: Whether to export the class
        abstract: Whether the class is abstract
        type_params: Generic type parameters (e.g., "<T, U>")
        extends: Parent class name
        implements: List of interface names to implement
        jsdoc: JSDoc comment block (without leading newline)
        properties: List of property declarations with indentation
        constructor: Constructor code with indentation
        methods: List of method implementations with indentation

    Returns:
        Formatted TypeScript class code
    """
    return TYPESCRIPT_CLASS_TEMPLATE.format(
        **_build_typescript_class_args(name, export, abstract, type_params, extends, implements, jsdoc, properties, constructor, methods)
    )


def _select_typescript_template(use_arrow: bool, is_async: bool) -> str:
    """Select the appropriate TypeScript function template.

    Args:
        use_arrow: Use arrow function syntax
        is_async: Function is async

    Returns:
        Template string
    """
    if use_arrow:
        return TYPESCRIPT_ARROW_FUNCTION_TEMPLATE
    if is_async:
        return TYPESCRIPT_ASYNC_FUNCTION_TEMPLATE
    return TYPESCRIPT_FUNCTION_TEMPLATE


def _format_typescript_params(params: List[Tuple[str, str]]) -> str:
    """Format TypeScript function parameters with types.

    Args:
        params: List of (param_name, param_type) tuples

    Returns:
        Formatted parameters string
    """
    return ", ".join(f"{p[0]}: {p[1]}" for p in params)


def _format_typescript_return_type(return_type: str, is_async: bool) -> str:
    """Format TypeScript return type annotation.

    Args:
        return_type: Return type annotation
        is_async: Whether function is async

    Returns:
        Formatted return type string
    """
    if not return_type:
        return ""

    if is_async and not return_type.startswith("Promise"):
        return f": Promise<{return_type}>"

    return f": {return_type}"


def _indent_function_body(body: str) -> str:
    """Indent function body with 2 spaces (JS/TS convention).

    Args:
        body: Function body to indent

    Returns:
        Indented body string
    """
    return "\n".join(f"  {line}" if line.strip() else line for line in body.split("\n"))


def format_typescript_function(
    name: str,
    params: List[Tuple[str, str]],
    body: str,
    return_type: str = "void",
    export: bool = False,
    is_async: bool = False,
    type_params: Optional[List[str]] = None,
    jsdoc: Optional[str] = None,
    use_arrow: bool = False,
) -> str:
    """Generate a TypeScript function from template.

    Args:
        name: Function name
        params: List of (param_name, param_type) tuples
        body: Function body (will be indented)
        return_type: Return type annotation (default "void")
        export: Whether to export the function
        is_async: Whether function is async
        type_params: Optional generic type parameters (e.g., ["T", "U"])
        jsdoc: Optional JSDoc comment block
        use_arrow: Use arrow function syntax instead of declaration

    Returns:
        Formatted TypeScript function string
    """
    format_args = {
        "jsdoc": f"{jsdoc}\n" if jsdoc else "",
        "export": "export " if export else "",
        "name": name,
        "type_params": f"<{', '.join(type_params)}>" if type_params else "",
        "params": _format_typescript_params(params),
        "return_type": _format_typescript_return_type(return_type, is_async),
        "body": _indent_function_body(body),
    }
    if use_arrow:
        format_args["async"] = "async " if is_async else ""
    return _select_typescript_template(use_arrow, is_async).format(**format_args)


# =============================================================================
# JavaScript Templates
# =============================================================================

JAVASCRIPT_FUNCTION_TEMPLATE: str = """{jsdoc}{export}function {name}({params}) {{
{body}
}}"""

JAVASCRIPT_ASYNC_FUNCTION_TEMPLATE: str = """{jsdoc}{export}async function {name}({params}) {{
{body}
}}"""

JAVASCRIPT_ARROW_FUNCTION_TEMPLATE: str = """{jsdoc}{export}const {name} = {async}({params}) => {{
{body}
}};"""


def _select_javascript_template(use_arrow: bool, is_async: bool) -> str:
    """Select the appropriate JavaScript function template.

    Args:
        use_arrow: Use arrow function syntax
        is_async: Function is async

    Returns:
        Template string
    """
    if use_arrow:
        return JAVASCRIPT_ARROW_FUNCTION_TEMPLATE
    if is_async:
        return JAVASCRIPT_ASYNC_FUNCTION_TEMPLATE
    return JAVASCRIPT_FUNCTION_TEMPLATE


def _format_javascript_params(params: List[str]) -> str:
    """Format JavaScript function parameters.

    Args:
        params: List of parameter names

    Returns:
        Formatted parameters string
    """
    return ", ".join(params)


def format_javascript_function(
    name: str,
    params: List[str],
    body: str,
    export: bool = False,
    is_async: bool = False,
    jsdoc: Optional[str] = None,
    use_arrow: bool = False,
) -> str:
    """Generate a JavaScript function from template.

    Args:
        name: Function name
        params: List of parameter names
        body: Function body (will be indented)
        export: Whether to export the function
        is_async: Whether function is async
        jsdoc: Optional JSDoc comment block
        use_arrow: Use arrow function syntax instead of declaration

    Returns:
        Formatted JavaScript function string
    """
    # Format placeholders
    export_str = "export " if export else ""
    async_str = "async " if is_async else ""
    jsdoc_str = f"{jsdoc}\n" if jsdoc else ""

    # Format parameters and indent body
    params_str = _format_javascript_params(params)
    indented_body = _indent_function_body(body)

    # Select template
    template = _select_javascript_template(use_arrow, is_async)

    # Format with appropriate template
    format_args = {"jsdoc": jsdoc_str, "export": export_str, "name": name, "params": params_str, "body": indented_body}

    # Arrow functions need the async keyword differently
    if use_arrow:
        format_args["async"] = async_str

    return template.format(**format_args)
