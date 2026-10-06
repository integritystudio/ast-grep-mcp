"""Function extraction logic for refactoring operations.

This module handles:
- Generating function signatures from code analysis
- Creating function bodies with proper indentation
- Generating call site replacements
- Inserting extracted functions at proper locations
"""

import textwrap
from typing import Dict, List, Optional

from ast_grep_mcp.core.logging import get_logger

from ...features.rewrite.backup import create_backup, restore_backup
from ...models.refactoring import (
    CodeSelection,
    ExtractFunctionResult,
    FunctionSignature,
    VariableInfo,
)
from ...utils.text import indent_lines, read_file_lines, write_file_lines

logger = get_logger(__name__)


class FunctionExtractor:
    """Handles function extraction refactoring."""

    def __init__(self, language: str) -> None:
        """Initialize extractor for specific language.

        Args:
            language: Programming language (python, typescript, javascript, java)
        """
        self.language = language

    def _perform_extraction_steps(
        self,
        selection: CodeSelection,
        function_name: str,
        extract_location: str,
        dry_run: bool,
    ) -> tuple[FunctionSignature, str, str, int, str, Optional[str]]:
        """Run all extraction steps and return the resulting components."""
        signature = self._generate_signature(selection, function_name)
        function_body = self._generate_function_body(selection, signature)
        call_replacement = self._generate_call_site(selection, signature)
        insertion_line = self._determine_insertion_line(selection, extract_location)
        diff_preview = self._generate_diff_preview(selection, function_body, call_replacement, insertion_line)
        backup_id = self._apply_extraction(selection, function_body, call_replacement, insertion_line) if not dry_run else None
        return signature, function_body, call_replacement, insertion_line, diff_preview, backup_id

    def extract_function(
        self,
        selection: CodeSelection,
        function_name: Optional[str] = None,
        extract_location: str = "before",
        dry_run: bool = True,
    ) -> ExtractFunctionResult:
        """Extract selected code into a new function.

        Args:
            selection: Analyzed code selection
            function_name: Name for extracted function (auto-generated if None)
            extract_location: Where to place function ('before', 'after', 'top')
            dry_run: If True, only preview changes

        Returns:
            ExtractFunctionResult with success status and details
        """
        logger.info(
            "extracting_function",
            file_path=selection.file_path,
            start_line=selection.start_line,
            end_line=selection.end_line,
            function_name=function_name,
            dry_run=dry_run,
        )

        try:
            if not function_name:
                function_name = self._generate_function_name(selection)
            signature, function_body, call_replacement, insertion_line, diff_preview, backup_id = self._perform_extraction_steps(
                selection, function_name, extract_location, dry_run
            )
            return ExtractFunctionResult(
                success=True,
                function_signature=signature,
                function_body=function_body,
                call_site_replacement=call_replacement,
                insertion_line=insertion_line,
                diff_preview=diff_preview,
                backup_id=backup_id,
            )

        except Exception as e:
            logger.error("extract_function_failed", error=str(e))
            return ExtractFunctionResult(
                success=False,
                error=str(e),
            )

    def _generate_function_name(self, selection: CodeSelection) -> str:
        """Generate a function name based on code content.

        Args:
            selection: Code selection

        Returns:
            Generated function name
        """
        # Simple heuristic: look for common action verbs or nouns
        content_lower = selection.content.lower()

        action_verbs = [
            "validate",
            "process",
            "calculate",
            "format",
            "parse",
            "convert",
            "transform",
            "filter",
            "map",
            "reduce",
            "check",
            "verify",
            "create",
            "update",
            "delete",
            "get",
            "set",
            "fetch",
            "load",
            "save",
            "send",
        ]

        for verb in action_verbs:
            if verb in content_lower:
                return f"{verb}_extracted"

        # Default fallback
        return "extracted_function"

    def _build_parameters(self, selection: CodeSelection) -> List[Dict[str, str]]:
        """Build parameter list from selection's needed parameters."""
        parameters: List[Dict[str, str]] = []
        for param_name in selection.parameters_needed:
            var_info = self._find_variable(selection, param_name)
            param_type = var_info.inferred_type if var_info and var_info.inferred_type else ""
            parameters.append({"name": param_name, "type": param_type})
        return parameters

    def _find_variable(self, selection: CodeSelection, name: str) -> Optional[VariableInfo]:
        """Find a variable by name in the selection's variable list."""
        return next((v for v in selection.variables if v.name == name), None)

    def _determine_return_type(self, selection: CodeSelection) -> Optional[str]:
        """Determine return type from selection's return values."""
        if len(selection.return_values) == 0:
            return "None" if self.language == "python" else "void"
        if len(selection.return_values) == 1:
            var_info = self._find_variable(selection, selection.return_values[0])
            return var_info.inferred_type if var_info else None
        if self.language == "python":
            return f"tuple[{', '.join(['Any'] * len(selection.return_values))}]"
        if self.language in ("typescript", "javascript"):
            return f"[{', '.join(['any'] * len(selection.return_values))}]"
        return None

    def _generate_signature(
        self,
        selection: CodeSelection,
        function_name: str,
    ) -> FunctionSignature:
        """Generate function signature from analyzed selection."""
        parameters = self._build_parameters(selection)
        return_type = self._determine_return_type(selection)
        docstring = self._generate_docstring(function_name, parameters, return_type, selection)

        return FunctionSignature(
            name=function_name,
            parameters=parameters,
            return_type=return_type,
            docstring=docstring,
        )

    def _generate_docstring(
        self,
        function_name: str,
        parameters: List[Dict[str, str]],
        return_type: Optional[str],
        selection: CodeSelection,
    ) -> str:
        """Generate docstring for extracted function."""
        if self.language == "python":
            return self._generate_python_docstring(parameters, return_type, selection)
        if self.language in ("typescript", "javascript"):
            return self._generate_jsdoc(parameters, return_type, selection)
        return ""

    def _generate_python_docstring(
        self,
        parameters: List[Dict[str, str]],
        return_type: Optional[str],
        selection: CodeSelection,
    ) -> str:
        """Generate Python-style docstring."""
        lines = [f'"""Extracted function from {selection.file_path}.']
        if parameters:
            lines.append("")
            lines.append("Args:")
            for param in parameters:
                param_type = f" ({param.get('type')})" if param.get("type") else ""
                lines.append(f"    {param['name']}{param_type}: Parameter extracted from code")
        if return_type and return_type != "None":
            lines.append("")
            lines.append("Returns:")
            lines.append(f"    {return_type}: Extracted return value")
        lines.append('"""')
        return "\n".join(lines)

    def _generate_jsdoc(
        self,
        parameters: List[Dict[str, str]],
        return_type: Optional[str],
        selection: CodeSelection,
    ) -> str:
        """Generate JSDoc-style docstring."""
        lines = ["/**", f" * Extracted function from {selection.file_path}"]
        if parameters:
            lines.append(" *")
            for param in parameters:
                param_type = param.get("type", "any")
                lines.append(f" * @param {{{param_type}}} {param['name']} - Parameter extracted from code")
        if return_type:
            lines.append(f" * @returns {{{return_type}}} Extracted return value")
        lines.append(" */")
        return "\n".join(lines)

    def _get_signature_line(self, signature: FunctionSignature) -> str:
        """Get the language-appropriate signature line."""
        if self.language == "python":
            return signature.to_python_signature()
        return signature.to_typescript_signature()

    def _generate_function_body(
        self,
        selection: CodeSelection,
        signature: FunctionSignature,
    ) -> str:
        """Generate complete function body with signature, docstring, and code."""
        lines = [self._get_signature_line(signature)]

        if signature.docstring:
            lines.extend(indent_lines(signature.docstring))

        # The selection keeps its original nesting; re-indent from column 0
        lines.extend(indent_lines(textwrap.dedent(selection.content)))

        if selection.return_values:
            return_stmt = self._generate_return_statement(selection.return_values)
            lines.append(f"    {return_stmt}")

        if self.language in ("typescript", "javascript"):
            lines.append("}")

        return "\n".join(lines)

    def _generate_return_statement(self, return_values: List[str]) -> str:
        """Generate return statement for extracted function.

        Args:
            return_values: List of variable names to return

        Returns:
            Return statement
        """
        if len(return_values) == 0:
            return ""
        elif len(return_values) == 1:
            return f"return {return_values[0]}"
        else:
            # Multiple returns - tuple for Python, array for JS/TS
            if self.language == "python":
                return f"return {', '.join(return_values)}"
            else:
                return f"return [{', '.join(return_values)}]"

    def _generate_call_site(
        self,
        selection: CodeSelection,
        signature: FunctionSignature,
    ) -> str:
        """Generate code to replace selection with function call.

        Args:
            selection: Code selection
            signature: Function signature

        Returns:
            Call site replacement code
        """
        # Build argument list
        args = ", ".join(signature.parameters[i]["name"] for i in range(len(signature.parameters)))

        # Generate call
        function_call = f"{signature.name}({args})"

        # Handle return value assignment
        if len(selection.return_values) == 0:
            return f"{selection.indentation}{function_call}"
        elif len(selection.return_values) == 1:
            return f"{selection.indentation}{selection.return_values[0]} = {function_call}"
        else:
            # Multiple returns - tuple unpacking
            vars_str = ", ".join(selection.return_values)
            return f"{selection.indentation}{vars_str} = {function_call}"

    def _find_import_section_end(self, file_path: str) -> int:
        """Find the line number after the import section ends.

        Args:
            file_path: Path to the source file

        Returns:
            Line number after last import (1-indexed), or 1 if no imports
        """
        try:
            lines = read_file_lines(file_path)
        except Exception as e:
            logger.warning("failed_to_read_file_for_imports", error=str(e))
            return 1

        last_import_line, _ = self._scan_imports(lines)
        return last_import_line + 1 if last_import_line > 0 else 1

    def _process_scan_line(self, stripped: str, i: int, last_import_line: int, in_multiline: bool) -> tuple[int, bool, bool]:
        """Process one line during import scan.

        Returns:
            Tuple of (last_import_line, in_multiline, should_break)
        """
        if self._should_skip_line(stripped):
            return last_import_line, in_multiline, False
        if not in_multiline and self._is_import_start(stripped):
            return i, self._check_multiline_import(stripped), False
        if in_multiline:
            return i, ")" not in stripped, False
        if last_import_line > 0:
            return last_import_line, in_multiline, True
        return last_import_line, in_multiline, False

    def _scan_imports(self, lines: list[str]) -> tuple[int, bool]:
        """Scan lines for import statements.

        Args:
            lines: Source file lines

        Returns:
            Tuple of (last_import_line, in_multiline_import)
        """
        last_import_line = 0
        in_multiline = False

        for i, line in enumerate(lines, start=1):
            last_import_line, in_multiline, should_break = self._process_scan_line(line.strip(), i, last_import_line, in_multiline)
            if should_break:
                break

        return last_import_line, in_multiline

    def _should_skip_line(self, stripped: str) -> bool:
        """Check if line should be skipped (empty or comment)."""
        return not stripped or stripped.startswith("#")

    def _is_import_start(self, stripped: str) -> bool:
        """Check if line starts an import statement."""
        return stripped.startswith("import ") or stripped.startswith("from ")

    def _check_multiline_import(self, stripped: str) -> bool:
        """Check if import starts a multiline block."""
        return "(" in stripped and ")" not in stripped

    def _determine_insertion_line(
        self,
        selection: CodeSelection,
        extract_location: str,
    ) -> int:
        """Determine where to insert the extracted function.

        Args:
            selection: Code selection
            extract_location: 'before', 'after', or 'top'

        Returns:
            Line number for insertion (1-indexed)
        """
        if extract_location == "top":
            # Insert at top of file (after imports)
            return self._find_import_section_end(selection.file_path)

        # A module-level function must not land inside the selection's enclosing definition
        lines = read_file_lines(selection.file_path)
        if extract_location == "after":
            return self._line_after_enclosing_definition(lines, selection)
        return self._line_before_enclosing_definition(lines, selection)

    @staticmethod
    def _is_top_level_code(line: str) -> bool:
        stripped = line.strip()
        return bool(stripped) and not line[0].isspace() and not stripped.startswith(("#", "//"))

    def _line_before_enclosing_definition(self, lines: List[str], selection: CodeSelection) -> int:
        """1-indexed line of the enclosing top-level definition (above its decorators)."""
        if not selection.indentation:
            return selection.start_line
        index = selection.start_line - 1
        while index > 0 and not self._is_top_level_code(lines[index]):
            index -= 1
        while index > 0 and lines[index - 1].lstrip().startswith("@"):
            index -= 1
        return index + 1

    def _line_after_enclosing_definition(self, lines: List[str], selection: CodeSelection) -> int:
        """1-indexed line just past the enclosing top-level definition."""
        if not selection.indentation:
            return selection.end_line + 1
        for index in range(selection.end_line, len(lines)):
            if self._is_top_level_code(lines[index]):
                # A closing brace at column 0 belongs to the enclosing definition
                return index + 2 if lines[index].lstrip().startswith("}") else index + 1
        return len(lines) + 1

    def _generate_diff_preview(
        self,
        selection: CodeSelection,
        function_body: str,
        call_replacement: str,
        insertion_line: int,
    ) -> str:
        """Generate unified diff preview of changes.

        Args:
            selection: Code selection
            function_body: Generated function code
            call_replacement: Call site replacement
            insertion_line: Where function will be inserted

        Returns:
            Unified diff string
        """
        lines = []
        lines.append(f"--- {selection.file_path}")
        lines.append(f"+++ {selection.file_path}")
        lines.append(f"@@ -{selection.start_line},{selection.end_line - selection.start_line + 1} +{insertion_line},... @@")

        # Show function insertion
        lines.append(f"+ {function_body.split(chr(10))[0]}")  # First line of function
        lines.append("+   ...")
        lines.append("+")

        # Show call site replacement
        lines.append(f"- {selection.content.split(chr(10))[0]}")  # First line of original
        lines.append("-   ...")
        lines.append(f"+ {call_replacement}")

        return "\n".join(lines)

    def _needs_typing_any(self, function_body: str) -> bool:
        """Return True when the generated Python signature references ``Any``."""
        return "Any" in function_body

    @staticmethod
    def _has_typing_any(lines: List[str]) -> bool:
        """Return True when the file already imports ``Any`` from ``typing``."""
        import re

        _TYPING_ANY_RE = re.compile(r"^\s*from\s+typing\s+import\b.*\bAny\b")
        return any(_TYPING_ANY_RE.search(line) for line in lines)

    def _inject_typing_any(self, lines: List[str]) -> List[str]:
        """Insert ``from typing import Any`` after the last import, or at line 1."""
        last_import_line, _ = self._scan_imports(lines)
        insert_at = last_import_line  # 0-indexed: insert after this line
        new_line = "from typing import Any\n"

        # Extend an existing ``from typing import …`` line when possible so we
        # do not create a duplicate import statement.
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("from typing import") and "Any" not in stripped:
                names_part = stripped[len("from typing import") :].strip()
                lines[i] = f"from typing import Any, {names_part}\n"
                return lines

        lines.insert(insert_at, new_line)
        return lines

    def _apply_extraction(
        self,
        selection: CodeSelection,
        function_body: str,
        call_replacement: str,
        insertion_line: int,
    ) -> str:
        """Apply the function extraction to the file.

        Returns:
            Backup ID for rollback
        """
        import os

        project_folder = os.path.dirname(selection.file_path)
        backup_id = create_backup([selection.file_path], project_folder)

        try:
            lines = read_file_lines(selection.file_path)

            # BR-04: ensure `from typing import Any` is present when the
            # extracted Python signature references `Any` (e.g. multi-value
            # returns emit `-> tuple[Any, Any]`).
            if self.language == "python" and self._needs_typing_any(function_body) and not self._has_typing_any(lines):
                lines = self._inject_typing_any(lines)
                # Recalculate insertion_line — the injected import shifts every
                # line that follows it down by one.
                last_import_line, _ = self._scan_imports(lines)
                if insertion_line > last_import_line:
                    insertion_line += 1

            function_lines = [line + "\n" for line in function_body.split("\n")]
            function_lines.append("\n\n")
            if insertion_line > 1 and lines[insertion_line - 2].strip():
                function_lines.insert(0, "\n\n")
            call_lines = [call_replacement + "\n"]
            selection_slice = slice(selection.start_line - 1, selection.end_line)

            # Apply the lower edit first so the other's line numbers stay valid
            if insertion_line > selection.end_line:
                lines[insertion_line - 1 : insertion_line - 1] = function_lines
                lines[selection_slice] = call_lines
            else:
                lines[selection_slice] = call_lines
                lines[insertion_line - 1 : insertion_line - 1] = function_lines

            write_file_lines(selection.file_path, lines)
            logger.info("extraction_applied", file_path=selection.file_path, backup_id=backup_id)
            return backup_id

        except Exception as e:
            restore_backup(backup_id, project_folder)
            raise RuntimeError(f"Failed to apply extraction: {e}")
