"""Apply deduplication refactoring with validation and rollback."""

import os
from typing import Any, Dict, List, Optional, cast

from ...constants import DisplayDefaults
from ...core.logging import get_logger
from .applicator_backup import DeduplicationBackupManager
from .applicator_executor import RefactoringExecutor
from .applicator_post_validator import RefactoringPostValidator
from .applicator_validator import RefactoringPlanValidator
from .generator import CodeGenerator

__all__ = [
    "DeduplicationApplicator",
    "_plan_file_modification_order",
    "_add_import_to_content",
    "_generate_import_for_extracted_function",
]


class DeduplicationApplicator:
    """Applies deduplication refactoring with backup and validation."""

    def __init__(self) -> None:
        """Initialize the deduplication applicator."""
        self.logger = get_logger("deduplication.applicator")
        self.code_generator = CodeGenerator()
        self.validator = RefactoringPlanValidator()
        self.executor = RefactoringExecutor()
        self.post_validator = RefactoringPostValidator()

    def apply_deduplication(
        self,
        project_folder: str,
        group_id: int,
        refactoring_plan: Dict[str, Any],
        dry_run: bool = True,
        backup: bool = True,
        extract_to_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Apply automated deduplication refactoring with comprehensive syntax validation.

        Phase 3.5 VALIDATION PIPELINE:
        1. PRE-VALIDATION: Validate all generated code before applying
        2. APPLICATION: Create backup and apply changes
        3. POST-VALIDATION: Validate modified files
        4. AUTO-ROLLBACK: Restore from backup if validation fails

        Args:
            project_folder: The absolute path to the project folder
            group_id: The duplication group ID from find_duplication results
            refactoring_plan: The refactoring plan with generated_code, files_affected, strategy, language
            dry_run: Preview changes without applying (default: true for safety)
            backup: Create backup before applying changes (default: true)
            extract_to_file: Where to place extracted function (auto-detect if None)

        Returns:
            Dict with:
            - status: "preview" | "success" | "failed" | "rolled_back"
            - validation: Pre and post validation results with detailed errors
            - errors: Detailed error info with file, line, message, and suggested fix
        """
        self.logger.info("apply_deduplication_start", project_folder=project_folder, group_id=group_id, dry_run=dry_run, backup=backup)

        # Initialize validation results
        validation_result: Dict[str, Any] = {
            "pre_validation": {"passed": False, "errors": []},
            "post_validation": {"passed": False, "errors": []},
        }

        try:
            # Step 1: Validate and prepare plan
            plan_result = self._validate_and_prepare_plan(
                project_folder, refactoring_plan, validation_result, group_id, extract_to_file, dry_run, backup
            )
            if "early_return" in plan_result:
                return cast(Dict[str, Any], plan_result["early_return"])

            # Extract plan components
            _files_to_modify = plan_result["files_to_modify"]  # noqa: F841
            generated_code = plan_result["generated_code"]
            language = plan_result["language"]
            strategy = plan_result["strategy"]
            orchestration_plan = plan_result["orchestration_plan"]
            backup_id = plan_result["backup_id"]

            # Step 2: Apply changes with validation
            apply_result = self._apply_changes_with_validation(orchestration_plan, generated_code, language, backup_id, project_folder)

            modified_files = apply_result["modified_files"]

            # Step 3: Handle post-validation and potential rollback
            rollback_response = self._validate_and_rollback_if_needed(
                modified_files, language, validation_result, backup_id, project_folder, group_id
            )
            if rollback_response:
                return rollback_response

            # Step 4: Build and return success response
            return self._build_success_response(modified_files, validation_result, backup_id, project_folder, group_id, strategy)

        except Exception as e:
            self.logger.error("apply_deduplication_failed", error=str(e)[: DisplayDefaults.ERROR_OUTPUT_PREVIEW_LENGTH])
            raise

    def _validate_and_prepare_plan(
        self,
        project_folder: str,
        refactoring_plan: Dict[str, Any],
        validation_result: Dict[str, Any],
        group_id: int,
        extract_to_file: Optional[str],
        dry_run: bool,
        backup: bool = True,
    ) -> Dict[str, Any]:
        """Validate inputs, extract plan components, perform pre-validation and handle dry-run.

        This consolidated method combines validation, pre-validation, dry-run handling,
        and backup creation to reduce cyclomatic complexity in the main function.

        Args:
            project_folder: Project root folder
            refactoring_plan: The refactoring plan
            validation_result: Validation results dict
            group_id: Duplication group ID
            extract_to_file: Optional file to extract to
            dry_run: Whether in dry-run mode
            backup: Whether to create backup

        Returns:
            Dict with plan data or early_return response
        """
        # Extract and validate plan components
        plan_data = self._extract_plan_components(project_folder, refactoring_plan, validation_result, group_id, extract_to_file)
        if "early_return" in plan_data:
            return plan_data

        # Perform pre-validation
        pre_validation_response = self._perform_pre_validation(refactoring_plan, group_id, project_folder, validation_result, dry_run)
        if pre_validation_response:
            return {"early_return": pre_validation_response}

        # Handle dry-run mode
        if dry_run:
            return {"early_return": self._handle_dry_run(plan_data["files_to_modify"], validation_result, group_id, plan_data["strategy"])}

        # Create backup if needed
        backup_id = self._create_backup_if_needed(
            backup, project_folder, plan_data["files_to_modify"], group_id, plan_data["strategy"], plan_data["orchestration_plan"]
        )

        # Return complete plan with backup_id
        return {**plan_data, "backup_id": backup_id}

    def _extract_plan_components(
        self,
        project_folder: str,
        refactoring_plan: Dict[str, Any],
        validation_result: Dict[str, Any],
        group_id: int,
        extract_to_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate inputs and extract plan components.

        Args:
            project_folder: Project root folder
            refactoring_plan: The refactoring plan
            validation_result: Validation results dict
            group_id: Duplication group ID
            extract_to_file: Optional file to extract to

        Returns:
            Dict with extracted plan data or early_return response
        """
        # Validate inputs
        if not os.path.isdir(project_folder):
            raise ValueError(f"Project folder does not exist: {project_folder}")
        if not refactoring_plan:
            raise ValueError("refactoring_plan is required")

        # Extract plan components
        files_affected = refactoring_plan.get("files_affected", [])
        generated_code = refactoring_plan.get("generated_code", {})
        language = refactoring_plan.get("language", "python")
        strategy = refactoring_plan.get("strategy", "extract_function")

        if not files_affected:
            return {
                "early_return": self._build_response("no_changes", "No files affected", validation_result, dry_run=True, group_id=group_id)
            }

        # Resolve file paths
        files_to_modify = self._resolve_file_paths(files_affected, project_folder)
        if not files_to_modify:
            return {
                "early_return": self._build_response("no_files", "No valid files found", validation_result, dry_run=True, group_id=group_id)
            }

        # Create orchestration plan
        orchestration_plan = self._plan_file_modification_order(files_to_modify, generated_code, extract_to_file, project_folder, language)

        return {
            "files_to_modify": files_to_modify,
            "generated_code": generated_code,
            "language": language,
            "strategy": strategy,
            "orchestration_plan": orchestration_plan,
        }

    def _perform_pre_validation(
        self, refactoring_plan: Dict[str, Any], group_id: int, project_folder: str, validation_result: Dict[str, Any], dry_run: bool
    ) -> Optional[Dict[str, Any]]:
        """Perform pre-validation on the refactoring plan.

        Args:
            refactoring_plan: The refactoring plan
            group_id: Duplication group ID
            project_folder: Project root folder
            validation_result: Validation results dict
            dry_run: Whether in dry-run mode

        Returns:
            Error response if validation fails, None if successful
        """
        pre_validation_result = self.validator.validate_plan(refactoring_plan, group_id, project_folder)
        validation_result["pre_validation"] = pre_validation_result.to_dict()

        if not pre_validation_result.is_valid:
            return self._build_response(
                "failed",
                f"Pre-validation failed with {len(pre_validation_result.errors)} error(s)",
                validation_result,
                errors=pre_validation_result.errors,
                dry_run=dry_run,
                group_id=group_id,
            )
        return None

    def _handle_dry_run(
        self, files_to_modify: List[str], validation_result: Dict[str, Any], group_id: int, strategy: str
    ) -> Dict[str, Any]:
        """Handle dry-run mode by building preview response.

        Args:
            files_to_modify: List of files that would be modified
            validation_result: Validation results
            group_id: Duplication group ID
            strategy: Refactoring strategy

        Returns:
            Dry-run preview response
        """
        return self._build_dry_run_response(files_to_modify, validation_result, group_id, strategy)

    def _create_backup_if_needed(
        self,
        backup: bool,
        project_folder: str,
        files_to_modify: List[str],
        group_id: int,
        strategy: str,
        orchestration_plan: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Create backup if requested.

        Args:
            backup: Whether to create backup
            project_folder: Project root folder
            files_to_modify: Files to backup
            group_id: Duplication group ID
            strategy: Refactoring strategy
            orchestration_plan: File modification plan; used to include the
                target file in the backup set and to track newly created files
                for deletion on rollback.

        Returns:
            Backup ID if created, None otherwise
        """
        if not backup:
            return None

        # Separate target-file entries into those that already exist (need a
        # content backup) and those that will be created fresh (need deletion
        # on rollback, not restoration).
        existing_target_files: List[str] = []
        created_files: List[str] = []
        for entry in (orchestration_plan or {}).get("create_files", []):
            path = entry.get("path", "")
            if not path:
                continue
            if entry.get("append", False):
                # File exists and will be appended to — back up its current content.
                existing_target_files.append(path)
            else:
                # File does not yet exist — track it for deletion on rollback.
                created_files.append(path)

        files_to_backup = [fp for fp in files_to_modify + existing_target_files if os.path.exists(fp)]

        backup_manager = DeduplicationBackupManager(project_folder)
        backup_id = backup_manager.create_backup(
            files=files_to_backup,
            metadata={"duplicate_group_id": group_id, "strategy": strategy, "file_count": len(files_to_modify)},
            created_files=created_files,
        )
        return backup_id

    def _apply_changes_with_validation(
        self,
        orchestration_plan: Dict[str, Any],
        generated_code: Dict[str, Any],
        language: str,
        backup_id: Optional[str],
        project_folder: str,
    ) -> Dict[str, Any]:
        """Apply changes with error handling and potential rollback.

        Args:
            orchestration_plan: File modification plan
            generated_code: Generated code with replacements
            language: Programming language
            backup_id: Backup ID for rollback if needed
            project_folder: Project root folder

        Returns:
            Dict with modified_files
        """
        try:
            raw_replacements = generated_code.get("replacements", {})
            normalized_replacements = self._normalize_replacement_keys(raw_replacements, project_folder)
            apply_result = self.executor.apply_changes(orchestration_plan, normalized_replacements, language, dry_run=False)
            return {"modified_files": apply_result["modified_files"]}

        except Exception:
            # Rollback on application failure
            if backup_id:
                backup_manager = DeduplicationBackupManager(project_folder)
                backup_manager.rollback(backup_id)
            raise

    def _validate_and_rollback_if_needed(
        self,
        modified_files: List[str],
        language: str,
        validation_result: Dict[str, Any],
        backup_id: Optional[str],
        project_folder: str,
        group_id: int,
    ) -> Optional[Dict[str, Any]]:
        """Perform post-validation and rollback if needed.

        Args:
            modified_files: List of modified files
            language: Programming language
            validation_result: Validation results dict
            backup_id: Backup ID for rollback
            project_folder: Project root folder
            group_id: Duplication group ID

        Returns:
            Rollback response if validation fails, None if successful
        """
        post_validation_result = self.post_validator.validate_modified_files(modified_files, language)
        validation_result["post_validation"] = post_validation_result.to_dict()

        # POST-VALIDATION FAILURE: handle with or without a backup
        if not post_validation_result.is_valid:
            if backup_id:
                # AUTO-ROLLBACK when a backup exists
                backup_manager = DeduplicationBackupManager(project_folder)
                restored = backup_manager.rollback(backup_id)
                return self._build_response(
                    "rolled_back",
                    f"Rolled back due to {len(post_validation_result.errors)} validation error(s)",
                    validation_result,
                    files_restored=restored,
                    backup_id=backup_id,
                    errors=post_validation_result.errors,
                    group_id=group_id,
                )
            # No backup — files on disk are broken; surface the failure explicitly
            return self._build_response(
                "validation_failed",
                f"Post-validation failed with {len(post_validation_result.errors)} error(s); no backup available to restore",
                validation_result,
                errors=post_validation_result.errors,
                group_id=group_id,
            )
        return None

    def _build_success_response(
        self,
        modified_files: List[str],
        validation_result: Dict[str, Any],
        backup_id: Optional[str],
        project_folder: str,
        group_id: int,
        strategy: str,
    ) -> Dict[str, Any]:
        """Build success response.

        Args:
            modified_files: List of modified files
            validation_result: Validation results
            backup_id: Backup ID if created
            project_folder: Project root folder
            group_id: Duplication group ID
            strategy: Refactoring strategy

        Returns:
            Success response dictionary
        """
        response = self._build_response(
            "success",
            f"Applied deduplication to {len(modified_files)} file(s)",
            validation_result,
            files_modified=modified_files,
            backup_id=backup_id,
            group_id=group_id,
            strategy=strategy,
        )

        if backup_id:
            response["rollback_command"] = f"rollback_rewrite(project_folder='{project_folder}', backup_id='{backup_id}')"

        return response

    def _resolve_file_paths(self, files_affected: List[Any], project_folder: str) -> List[str]:
        """Resolve file paths from files_affected list.

        Args:
            files_affected: List of file paths or file info dicts
            project_folder: Project root folder

        Returns:
            List of resolved absolute file paths
        """
        files_to_modify = []
        for file_info in files_affected:
            file_path = file_info if isinstance(file_info, str) else file_info.get("file", "")
            if file_path and os.path.isfile(file_path):
                files_to_modify.append(file_path)
            elif file_path and os.path.isfile(os.path.join(project_folder, file_path)):
                files_to_modify.append(os.path.join(project_folder, file_path))
        return files_to_modify

    def _normalize_replacement_keys(self, replacements: Dict[str, Dict[str, Any]], project_folder: str) -> Dict[str, Dict[str, Any]]:
        """Resolve relative replacement keys to absolute paths.

        The pre-validator accepts relative replacement keys (joining with
        project_folder), but the executor looks up replacements by the absolute
        path used in the orchestration plan. Normalizing here ensures the
        lookup in _update_single_file always matches.

        Args:
            replacements: Dict keyed by file path (may be relative or absolute)
            project_folder: Project root used to resolve relative paths

        Returns:
            Dict with all keys resolved to absolute paths
        """
        normalized: Dict[str, Dict[str, Any]] = {}
        for key, value in replacements.items():
            abs_key = key if os.path.isabs(key) else os.path.join(project_folder, key)
            normalized[abs_key] = value
        return normalized

    def _build_response(self, status: str, message: str, validation_result: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """Build standardized response dictionary.

        Args:
            status: Response status
            message: Response message
            validation_result: Validation results
            **kwargs: Additional response fields

        Returns:
            Response dictionary
        """
        response = {"status": status, "message": message, "validation": validation_result, "dry_run": kwargs.get("dry_run", False)}
        response.update(kwargs)
        return response

    def _build_dry_run_response(
        self, files_to_modify: List[str], validation_result: Dict[str, Any], group_id: int, strategy: str
    ) -> Dict[str, Any]:
        """Build dry run preview response.

        Args:
            files_to_modify: List of files that would be modified
            validation_result: Validation results
            group_id: Duplication group ID
            strategy: Refactoring strategy

        Returns:
            Preview response dictionary
        """
        changes_preview = []
        for fp in files_to_modify:
            with open(fp, "r") as f:
                changes_preview.append({"file": fp, "lines": len(f.read().splitlines())})

        return {
            "status": "preview",
            "dry_run": True,
            "message": f"Preview of changes to {len(files_to_modify)} file(s)",
            "changes_preview": changes_preview,
            "validation": validation_result,
            "group_id": group_id,
            "strategy": strategy,
        }

    def _resolve_target_file(
        self,
        files_to_modify: List[str],
        extract_to_file: Optional[str],
        generated_code: Dict[str, Any],
        project_folder: str,
    ) -> Optional[str]:
        """Resolve the target file path for extracted function placement.

        Args:
            files_to_modify: Files being modified
            extract_to_file: Explicit target file override
            generated_code: Generated code dict with optional extract_to_file
            project_folder: Project root folder

        Returns:
            Absolute target file path, or None if files_to_modify is empty
            and no explicit target is specified. Note: auto-synthesizes a
            _extracted_utils path as fallback when files_to_modify is non-empty.
        """
        target_file = extract_to_file or generated_code.get("extract_to_file")
        if not target_file and files_to_modify:
            first_file = files_to_modify[0]
            file_dir = os.path.dirname(first_file)
            ext = os.path.splitext(first_file)[1]
            target_file = os.path.join(file_dir, f"_extracted_utils{ext}")

        if target_file and not os.path.isabs(target_file):
            target_file = os.path.join(project_folder, target_file)

        return target_file

    def _plan_file_updates(
        self,
        plan: Dict[str, Any],
        files_to_modify: List[str],
        target_file: Optional[str],
        extracted_function: str,
        function_name: str,
        project_folder: str,
        language: str,
    ) -> None:
        """Populate update_files and import_additions in the plan.

        Args:
            plan: Orchestration plan dict to populate
            files_to_modify: Files being modified
            target_file: Target file for extracted function
            extracted_function: Extracted function code
            function_name: Name of extracted function
            project_folder: Project root folder
            language: Programming language
        """
        for file_path in files_to_modify:
            plan["update_files"].append({"path": file_path, "operation": "replace_duplicate"})

            if extracted_function and target_file and file_path != target_file:
                import_stmt = self._generate_import_for_extracted_function(
                    source_file=file_path,
                    target_file=target_file,
                    function_name=function_name,
                    project_folder=project_folder,
                    language=language,
                )

                if import_stmt:
                    plan["import_additions"][file_path] = {
                        "import_statement": import_stmt,
                        "from_file": target_file,
                        "function_name": function_name,
                    }

    def _plan_file_modification_order(
        self, files_to_modify: List[str], generated_code: Dict[str, Any], extract_to_file: Optional[str], project_folder: str, language: str
    ) -> Dict[str, Any]:
        """Plan the order of file modifications for atomic deduplication."""
        plan: Dict[str, Any] = {"create_files": [], "update_files": [], "import_additions": {}}

        extracted_function = generated_code.get("extracted_function", "")
        function_name = generated_code.get("function_name", "extracted_function")

        target_file = self._resolve_target_file(files_to_modify, extract_to_file, generated_code, project_folder)

        # Plan file creation for extracted function
        if extracted_function and target_file:
            append_mode = os.path.exists(target_file)
            plan["create_files"].append(
                {
                    "path": target_file,
                    "content": extracted_function,
                    "append": append_mode,
                    "operation": "append" if append_mode else "create",
                }
            )

        # Plan updates for duplicate location files
        self._plan_file_updates(plan, files_to_modify, target_file, extracted_function, function_name, project_folder, language)

        return plan

    def _generate_import_for_extracted_function(
        self, source_file: str, target_file: str, function_name: str, project_folder: str, language: str
    ) -> str:
        """Generate import statement for an extracted function."""
        # Calculate relative path from source to target
        source_dir = os.path.dirname(source_file)
        target_rel = os.path.relpath(target_file, source_dir)

        # Count ".." components before joining with dots — the code generator adds
        # one leading dot for the current package, so N parent hops need N dots here.
        module_rel = os.path.splitext(target_rel)[0]
        parts = module_rel.replace(os.sep, "/").split("/")
        parent_count = sum(1 for p in parts if p == "..")
        module_parts = [p for p in parts if p not in ("..", ".")]
        module_path = "." * parent_count + ".".join(module_parts)

        # Generate import using code generator
        return self.code_generator.generate_import_statement(
            module_path=module_path, import_names=[function_name], is_relative=module_path.startswith(".")
        )

    def _add_import_to_content(self, content: str, import_statement: str, language: str) -> str:
        """Add an import statement to file content."""
        if not import_statement:
            return content

        # Check if import already exists
        if import_statement.strip() in content:
            return content

        lines = content.split("\n")
        lang = language.lower()

        inserters = {
            "python": self._insert_python_import,
            "javascript": self._insert_js_import,
            "typescript": self._insert_js_import,
            "jsx": self._insert_js_import,
            "tsx": self._insert_js_import,
            "java": self._insert_java_import,
        }
        inserter = inserters.get(lang, self._insert_default_import)
        inserter(lines, import_statement)

        return "\n".join(lines)

    @staticmethod
    def _find_last_module_import(lines: List[str]) -> int:
        """Index of the last module-level import, or -1 if there is none.

        Only column-0 imports count: an indented function-local import must not
        become an insertion anchor (BUG-10). Scanning stops at the first line of
        real code after an import, so imports further down the file (inside a
        function, or after a conditional) are not picked up.
        """
        last_import_idx = -1
        for i, line in enumerate(lines):
            if line.startswith("import ") or line.startswith("from "):
                last_import_idx = i
                continue
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and last_import_idx >= 0:
                break
        return last_import_idx

    @staticmethod
    def _skip_leading_comments(lines: List[str]) -> int:
        """Index of the first line that is neither blank nor a comment."""
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith("#"):
                return i
        return len(lines)

    @staticmethod
    def _skip_module_docstring(lines: List[str], start: int) -> int:
        """Index just past the module docstring starting at ``start``.

        Returns ``start`` unchanged when no docstring begins there. An unclosed
        multi-line docstring consumes the rest of the file, matching the
        behaviour of inserting after it rather than inside it.
        """
        if start >= len(lines):
            return start

        stripped = lines[start].strip()
        quote = stripped[:3]
        if quote not in ('"""', "'''"):
            return start

        # Closing quotes on the same line: a single-line docstring.
        if quote in stripped[3:]:
            return start + 1

        for i in range(start + 1, len(lines)):
            if quote in lines[i]:
                return i + 1
        return len(lines)

    @classmethod
    def _insert_python_import(cls, lines: List[str], import_statement: str) -> None:
        """Insert import into Python source lines."""
        last_import_idx = cls._find_last_module_import(lines)
        if last_import_idx >= 0:
            lines.insert(last_import_idx + 1, import_statement)
            return

        # No imports found — insert after shebang/encoding comments and any
        # module docstring, so the docstring is not demoted to a plain string
        # expression (BUG-10).
        insert_idx = cls._skip_module_docstring(lines, cls._skip_leading_comments(lines))

        lines.insert(insert_idx, import_statement)
        if insert_idx > 0:
            lines.insert(insert_idx, "")

    @staticmethod
    def _insert_js_import(lines: List[str], import_statement: str) -> None:
        """Insert import into JS/TS source lines."""
        last_import_idx = -1
        for i, line in enumerate(lines):
            if "import " in line or "require(" in line:
                last_import_idx = i

        if last_import_idx >= 0:
            lines.insert(last_import_idx + 1, import_statement)
        else:
            lines.insert(0, import_statement)
            lines.insert(1, "")

    @staticmethod
    def _insert_java_import(lines: List[str], import_statement: str) -> None:
        """Insert import into Java source lines."""
        package_idx = -1
        last_import_idx = -1
        for i, line in enumerate(lines):
            if line.strip().startswith("package "):
                package_idx = i
            elif line.strip().startswith("import "):
                last_import_idx = i

        if last_import_idx >= 0:
            lines.insert(last_import_idx + 1, import_statement)
        elif package_idx >= 0:
            lines.insert(package_idx + 1, "")
            lines.insert(package_idx + 2, import_statement)
        else:
            lines.insert(0, import_statement)
            lines.insert(1, "")

    @staticmethod
    def _insert_default_import(lines: List[str], import_statement: str) -> None:
        """Insert import at file top (fallback for unsupported languages)."""
        lines.insert(0, import_statement)
        lines.insert(1, "")


# Module-level functions for backward compatibility with tests
_applicator_instance = None


def _get_applicator() -> Any:
    """Get or create the global applicator instance."""
    global _applicator_instance
    if _applicator_instance is None:
        _applicator_instance = DeduplicationApplicator()
    return _applicator_instance


def _plan_file_modification_order(
    files_to_modify: List[str], generated_code: Dict[str, Any], extract_to_file: Optional[str], project_folder: str, language: str
) -> Dict[str, Any]:
    """Module-level wrapper for _plan_file_modification_order."""
    result = _get_applicator()._plan_file_modification_order(files_to_modify, generated_code, extract_to_file, project_folder, language)
    if not isinstance(result, dict):
        raise TypeError(f"Expected dict from _plan_file_modification_order, got {type(result).__name__}")
    return result


def _add_import_to_content(content: str, import_statement: str, language: str) -> str:
    """Module-level wrapper for _add_import_to_content."""
    result = _get_applicator()._add_import_to_content(content, import_statement, language)
    if not isinstance(result, str):
        raise TypeError(f"Expected str from _add_import_to_content, got {type(result).__name__}")
    return result


def _generate_import_for_extracted_function(
    source_file: str, target_file: str, function_name: str, project_folder: str, language: str
) -> str:
    """Module-level wrapper for _generate_import_for_extracted_function."""
    result = _get_applicator()._generate_import_for_extracted_function(source_file, target_file, function_name, project_folder, language)
    if not isinstance(result, str):
        raise TypeError(f"Expected str from _generate_import_for_extracted_function, got {type(result).__name__}")
    return result
