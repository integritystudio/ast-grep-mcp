"""Tests for documentation generation feature."""

from unittest.mock import patch

from ast_grep_mcp.models.documentation import (
    ApiRoute,
    ChangelogEntry,
    ChangelogVersion,
    ChangeType,
    DocstringStyle,
    DocSyncResult,
    FunctionSignature,
    GeneratedDocstring,
    ParameterInfo,
    ProjectInfo,
    RouteParameter,
)


class TestDataModels:
    """Tests for documentation data models."""

    def test_parameter_info_creation(self):
        """Test ParameterInfo model creation."""
        param = ParameterInfo(
            name="user_id",
            type_hint="int",
            default_value="0",
        )
        assert param.name == "user_id"
        assert param.type_hint == "int"
        assert param.default_value == "0"

    def test_function_signature_creation(self):
        """Test FunctionSignature model creation."""
        sig = FunctionSignature(
            name="process_user",
            parameters=[
                ParameterInfo(name="user", type_hint="dict"),
                ParameterInfo(name="validate", type_hint="bool", default_value="True"),
            ],
            return_type="dict",
            decorators=["@staticmethod"],
            is_async=False,
            is_method=False,
            file_path="/path/to/file.py",
            start_line=10,
            end_line=20,
            existing_docstring=None,
        )
        assert sig.name == "process_user"
        assert len(sig.parameters) == 2
        assert sig.return_type == "dict"
        assert sig.decorators == ["@staticmethod"]

    def test_generated_docstring_creation(self):
        """Test GeneratedDocstring model creation."""
        docstring = GeneratedDocstring(
            function_name="get_user",
            docstring='"""Get a user by ID.\n\nArgs:\n    user_id: The user ID.\n\nReturns:\n    User dictionary.\n"""',
            style=DocstringStyle.GOOGLE,
            file_path="/path/file.py",
            line_number=10,
        )
        assert docstring.function_name == "get_user"
        assert "Args:" in docstring.docstring
        assert docstring.style == DocstringStyle.GOOGLE

    def test_api_route_creation(self):
        """Test ApiRoute model creation."""
        route = ApiRoute(
            method="POST",
            path="/api/users",
            handler_name="create_user",
            file_path="/path/routes.py",
            line_number=50,
            parameters=[
                RouteParameter(name="name", location="body", type_hint="string", required=True),
            ],
            description="Create a new user",
            authentication="bearer",
        )
        assert route.method == "POST"
        assert route.path == "/api/users"
        assert len(route.parameters) == 1
        assert route.authentication == "bearer"

    def test_changelog_entry_creation(self):
        """Test ChangelogEntry model creation."""
        entry = ChangelogEntry(
            change_type=ChangeType.ADDED,
            description="New feature X",
            commit_hash="abc123",
            is_breaking=False,
        )
        assert entry.change_type == ChangeType.ADDED
        assert entry.description == "New feature X"
        assert entry.is_breaking is False

    def test_changelog_version_creation(self):
        """Test ChangelogVersion model creation."""
        version = ChangelogVersion(
            version="1.2.0",
            date="2024-01-15",
            entries={
                ChangeType.ADDED: [
                    ChangelogEntry(change_type=ChangeType.ADDED, description="New feature"),
                ],
                ChangeType.FIXED: [
                    ChangelogEntry(change_type=ChangeType.FIXED, description="Bug fix"),
                ],
            },
        )
        assert version.version == "1.2.0"
        assert len(version.entries[ChangeType.ADDED]) == 1
        assert len(version.entries[ChangeType.FIXED]) == 1


class TestDocstringGenerator:
    """Tests for docstring generation."""

    def test_infer_description_from_name(self):
        """Test description inference from function names."""
        from ast_grep_mcp.features.documentation.docstring_generator import (
            _infer_description_from_name,
        )

        assert "get" in _infer_description_from_name("get_user").lower()
        assert "create" in _infer_description_from_name("create_account").lower()
        assert "update" in _infer_description_from_name("update_record").lower()
        assert "delete" in _infer_description_from_name("delete_item").lower()
        assert "check" in _infer_description_from_name("is_valid").lower()
        assert "check" in _infer_description_from_name("has_permission").lower()
        assert "calculate" in _infer_description_from_name("calculate_total").lower()
        assert "validate" in _infer_description_from_name("validate_input").lower()
        assert "parse" in _infer_description_from_name("parse_data").lower()

    def test_infer_parameter_description(self):
        """Test parameter description inference."""
        from ast_grep_mcp.features.documentation.docstring_generator import (
            _infer_parameter_description,
        )

        param = ParameterInfo(name="user_id", type_hint="int")
        desc = _infer_parameter_description(param, "get_user")
        assert len(desc) > 0

        param = ParameterInfo(name="callback", type_hint="Callable")
        desc = _infer_parameter_description(param, "register_handler")
        assert len(desc) > 0

        param = ParameterInfo(name="timeout", type_hint="float")
        desc = _infer_parameter_description(param, "connect")
        assert len(desc) > 0

    def test_generate_google_docstring(self):
        """Test Google-style docstring generation."""
        from ast_grep_mcp.features.documentation.docstring_generator import (
            _generate_google_docstring,
        )

        func = FunctionSignature(
            name="get_user",
            parameters=[
                ParameterInfo(name="user_id", type_hint="int"),
                ParameterInfo(name="include_deleted", type_hint="bool", default_value="False"),
            ],
            return_type="dict",
            decorators=[],
            is_async=False,
            is_method=False,
            file_path="/path/file.py",
            start_line=1,
            end_line=10,
            existing_docstring=None,
        )

        docstring = _generate_google_docstring(func)
        assert '"""' in docstring
        assert "Args:" in docstring
        assert "user_id" in docstring
        assert "include_deleted" in docstring
        assert "Returns:" in docstring

    def test_generate_numpy_docstring(self):
        """Test NumPy-style docstring generation."""
        from ast_grep_mcp.features.documentation.docstring_generator import (
            _generate_numpy_docstring,
        )

        func = FunctionSignature(
            name="calculate_mean",
            parameters=[
                ParameterInfo(name="values", type_hint="List[float]"),
            ],
            return_type="float",
            decorators=[],
            is_async=False,
            is_method=False,
            file_path="/path/file.py",
            start_line=1,
            end_line=10,
            existing_docstring=None,
        )

        docstring = _generate_numpy_docstring(func)
        assert '"""' in docstring
        assert "Parameters" in docstring
        assert "----------" in docstring
        assert "values" in docstring
        assert "Returns" in docstring

    def test_generate_sphinx_docstring(self):
        """Test Sphinx-style docstring generation."""
        from ast_grep_mcp.features.documentation.docstring_generator import (
            _generate_sphinx_docstring,
        )

        func = FunctionSignature(
            name="save_file",
            parameters=[
                ParameterInfo(name="path", type_hint="str"),
                ParameterInfo(name="content", type_hint="bytes"),
            ],
            return_type="bool",
            decorators=[],
            is_async=False,
            is_method=False,
            file_path="/path/file.py",
            start_line=1,
            end_line=10,
            existing_docstring=None,
        )

        docstring = _generate_sphinx_docstring(func)
        assert '"""' in docstring
        assert ":param path:" in docstring
        assert ":param content:" in docstring
        assert ":return:" in docstring or ":returns:" in docstring

    def test_generate_jsdoc(self):
        """Test JSDoc generation for JavaScript/TypeScript."""
        from ast_grep_mcp.features.documentation.docstring_generator import (
            _generate_jsdoc,
        )

        func = FunctionSignature(
            name="fetchUser",
            parameters=[
                ParameterInfo(name="userId", type_hint="string"),
            ],
            return_type="Promise<User>",
            decorators=[],
            is_async=True,
            is_method=False,
            file_path="/path/file.ts",
            start_line=1,
            end_line=10,
            existing_docstring=None,
        )

        docstring = _generate_jsdoc(func)
        assert "/**" in docstring
        assert "@param" in docstring
        assert "userId" in docstring
        assert "@returns" in docstring
        assert "@async" in docstring

    def test_function_signature_parser_python(self, tmp_path):
        """Test Python function parsing."""
        from ast_grep_mcp.features.documentation.docstring_generator import (
            FunctionSignatureParser,
        )

        test_file = tmp_path / "test.py"
        test_file.write_text('''
def simple_function(x: int, y: str = "default") -> bool:
    """Existing docstring."""
    return True

async def async_function(data: dict) -> list:
    return []

class MyClass:
    def method(self, value: float) -> None:
        pass
''')

        parser = FunctionSignatureParser("python")
        functions = parser.parse_file(str(test_file))

        assert len(functions) >= 3

        # Check simple_function
        simple = next(f for f in functions if f.name == "simple_function")
        assert simple.return_type == "bool"
        assert len(simple.parameters) == 2
        assert simple.parameters[0].name == "x"
        assert simple.parameters[0].type_hint == "int"
        assert simple.parameters[1].name == "y"
        assert simple.parameters[1].default_value == '"default"'
        assert simple.existing_docstring is not None

        # Check async_function
        async_func = next(f for f in functions if f.name == "async_function")
        assert async_func.is_async is True

        # Check method
        method = next(f for f in functions if f.name == "method")
        assert method.is_method is True


class TestReadmeGenerator:
    """Tests for README generation."""

    def test_detect_package_manager(self, tmp_path):
        """Test package manager detection."""
        from ast_grep_mcp.features.documentation.readme_generator import (
            _detect_package_manager,
        )

        # npm
        (tmp_path / "package.json").write_text("{}")
        result = _detect_package_manager(str(tmp_path))
        # Result is a tuple (package_manager, version, install_cmd)
        assert result[0] == "npm" or "npm" in str(result)

        # pip
        tmp_path2 = tmp_path / "pip_project"
        tmp_path2.mkdir()
        (tmp_path2 / "requirements.txt").write_text("")
        result = _detect_package_manager(str(tmp_path2))
        assert result[0] == "pip" or "pip" in str(result)

    def test_detect_language(self, tmp_path):
        """Test primary language detection."""
        from ast_grep_mcp.features.documentation.readme_generator import (
            _detect_language,
        )

        # Python project
        (tmp_path / "main.py").write_text("print('hello')")
        (tmp_path / "utils.py").write_text("x = 1")
        assert _detect_language(str(tmp_path)) == "python"

        # TypeScript project
        tmp_path2 = tmp_path / "ts_project"
        tmp_path2.mkdir()
        (tmp_path2 / "index.ts").write_text("const x = 1")
        (tmp_path2 / "utils.ts").write_text("const y = 2")
        assert _detect_language(str(tmp_path2)) == "typescript"

    def test_generate_installation_section(self):
        """Test installation section generation."""
        from ast_grep_mcp.features.documentation.readme_generator import (
            _generate_installation_section,
        )

        info = ProjectInfo(
            name="my-project",
            description="A test project",
            language="python",
            frameworks=["FastAPI"],
            package_manager="pip",
            entry_points=["main.py"],
            dependencies=["fastapi", "uvicorn"],
        )

        section = _generate_installation_section(info)
        assert section.title == "Installation"
        assert len(section.content) > 0

    def test_generate_usage_section(self):
        """Test usage section generation."""
        from ast_grep_mcp.features.documentation.readme_generator import (
            _generate_usage_section,
        )

        info = ProjectInfo(
            name="my-api",
            description="An API project",
            language="python",
            frameworks=["FastAPI"],
            package_manager="pip",
            entry_points=["main.py"],
            dependencies=[],
        )

        section = _generate_usage_section(info)
        assert section.title == "Usage"
        assert len(section.content) > 0


class TestApiDocsGenerator:
    """Tests for API documentation generation."""

    def test_express_route_parser(self, tmp_path):
        """Test Express.js route parsing."""
        from ast_grep_mcp.features.documentation.api_docs_generator import (
            ExpressRouteParser,
        )

        test_file = tmp_path / "routes.js"
        test_file.write_text("""
const router = express.Router();

router.get('/users', getUsers);
router.post('/users', createUser);
router.get('/users/:id', getUserById);
router.put('/users/:id', updateUser);
router.delete('/users/:id', deleteUser);
""")

        parser = ExpressRouteParser()
        routes = parser.parse_file(str(test_file))

        assert len(routes) == 5

        get_users = next(r for r in routes if r.path == "/users" and r.method == "GET")
        assert get_users.handler_name == "getUsers"

        get_by_id = next(r for r in routes if ":id" in r.path and r.method == "GET")
        assert len(get_by_id.parameters) == 1
        assert get_by_id.parameters[0].name == "id"
        assert get_by_id.parameters[0].location == "path"

    def test_fastapi_route_parser(self, tmp_path):
        """Test FastAPI route parsing."""
        from ast_grep_mcp.features.documentation.api_docs_generator import (
            FastAPIRouteParser,
        )

        test_file = tmp_path / "routes.py"
        test_file.write_text("""
from fastapi import FastAPI

app = FastAPI()

@app.get("/items")
async def get_items():
    return []

@app.post("/items")
async def create_item(item: Item):
    return item

@app.get("/items/{item_id}")
async def get_item(item_id: int):
    return {"id": item_id}
""")

        parser = FastAPIRouteParser()
        routes = parser.parse_file(str(test_file))

        assert len(routes) == 3

        get_items = next(r for r in routes if r.path == "/items" and r.method == "GET")
        assert get_items.handler_name == "get_items"

        get_item = next(r for r in routes if "{item_id}" in r.path)
        assert len(get_item.parameters) == 1
        assert get_item.parameters[0].name == "item_id"

    def test_generate_markdown_docs(self):
        """Test Markdown API documentation generation."""
        from ast_grep_mcp.features.documentation.api_docs_generator import (
            _generate_markdown_docs,
        )

        routes = [
            ApiRoute(
                method="GET",
                path="/api/users",
                handler_name="get_users",
                file_path="/path/routes.py",
                line_number=10,
                parameters=[],
                description="Get all users",
            ),
            ApiRoute(
                method="POST",
                path="/api/users",
                handler_name="create_user",
                file_path="/path/routes.py",
                line_number=20,
                parameters=[
                    RouteParameter(name="name", location="body", type_hint="string", required=True),
                ],
                description="Create a user",
                authentication="bearer",
            ),
        ]

        markdown = _generate_markdown_docs(routes, "Test API")
        assert "API" in markdown
        assert "GET" in markdown
        assert "POST" in markdown
        assert "/api/users" in markdown

    def test_generate_openapi_spec(self):
        """Test OpenAPI specification generation."""
        from ast_grep_mcp.features.documentation.api_docs_generator import (
            _generate_openapi_spec,
        )

        routes = [
            ApiRoute(
                method="GET",
                path="/api/items/{id}",
                handler_name="get_item",
                file_path="/path/routes.py",
                line_number=10,
                parameters=[
                    RouteParameter(name="id", location="path", type_hint="integer", required=True),
                ],
                description="Get an item by ID",
            ),
        ]

        spec = _generate_openapi_spec(routes, "Test API")
        assert spec["openapi"] == "3.0.0"
        assert spec["info"]["title"] == "Test API"
        assert "/api/items/{id}" in spec["paths"]
        assert "get" in spec["paths"]["/api/items/{id}"]


class TestChangelogHelpers:
    """Tests for changelog generator helper functions."""

    _MODULE = "ast_grep_mcp.features.documentation.changelog_generator"
    _MOCK_TARGET = f"{_MODULE}._run_git_command"

    def test_resolve_version_ref_head(self):
        """Test HEAD passthrough."""
        from ast_grep_mcp.features.documentation.changelog_generator import (
            _resolve_version_ref,
        )

        # HEAD should return immediately without git calls
        with patch(self._MOCK_TARGET) as mock_git:
            assert _resolve_version_ref("/fake", "HEAD") == "HEAD"
            assert _resolve_version_ref("/fake", "head") == "HEAD"
            mock_git.assert_not_called()

    def test_resolve_version_ref_vprefixed_tag(self):
        """Test resolution to v-prefixed tag."""
        from ast_grep_mcp.features.documentation.changelog_generator import (
            _resolve_version_ref,
        )

        with patch(self._MOCK_TARGET) as mock_git:
            mock_git.return_value = (True, "abc123")
            result = _resolve_version_ref("/fake", "1.2.0")
            assert result == "v1.2.0"
            mock_git.assert_called_once_with("/fake", ["rev-parse", "v1.2.0"])

    def test_resolve_version_ref_bare_ref(self):
        """Test fallback to bare ref when v-prefix fails."""
        from ast_grep_mcp.features.documentation.changelog_generator import (
            _resolve_version_ref,
        )

        with patch(self._MOCK_TARGET) as mock_git:
            mock_git.side_effect = [
                (False, "not found"),  # v-prefix fails
                (True, "abc123"),  # bare ref succeeds
            ]
            result = _resolve_version_ref("/fake", "some-branch")
            assert result == "some-branch"

    def test_resolve_version_ref_fallback_head(self):
        """Test fallback to HEAD when both ref attempts fail."""
        from ast_grep_mcp.features.documentation.changelog_generator import (
            _resolve_version_ref,
        )

        with patch(self._MOCK_TARGET) as mock_git:
            mock_git.return_value = (False, "not found")
            result = _resolve_version_ref("/fake", "nonexistent")
            assert result == "HEAD"

    def test_get_first_commit_success(self):
        """Test successful first commit retrieval."""
        from ast_grep_mcp.features.documentation.changelog_generator import (
            _get_first_commit,
        )

        with patch(self._MOCK_TARGET) as mock_git:
            mock_git.return_value = (True, "aaa111")
            assert _get_first_commit("/fake") == "aaa111"
            mock_git.assert_called_once_with("/fake", ["rev-list", "--max-parents=0", "HEAD"])

    def test_get_first_commit_failure(self):
        """Test None return on failure."""
        from ast_grep_mcp.features.documentation.changelog_generator import (
            _get_first_commit,
        )

        with patch(self._MOCK_TARGET) as mock_git:
            mock_git.return_value = (False, "error")
            assert _get_first_commit("/fake") is None

    def test_find_previous_tag_skips_excluded(self):
        """Test that excluded ref is skipped."""
        from ast_grep_mcp.features.documentation.changelog_generator import (
            _find_previous_tag,
        )

        with patch(self._MOCK_TARGET) as mock_git:
            mock_git.side_effect = [
                (True, "v2.0.0\nv1.0.0\nv0.9.0"),  # tag list
            ]
            result = _find_previous_tag("/fake", "v2.0.0")
            assert result == "v1.0.0"
            mock_git.assert_called_once()  # should not reach _get_first_commit

    def test_find_previous_tag_returns_first_when_all_excluded(self):
        """Test fallback to first commit when only excluded tag exists."""
        from ast_grep_mcp.features.documentation.changelog_generator import (
            _find_previous_tag,
        )

        with patch(self._MOCK_TARGET) as mock_git:
            mock_git.side_effect = [
                (True, "v1.0.0"),  # tag list (only the excluded one)
                (True, "first-commit"),  # rev-list for first commit
            ]
            result = _find_previous_tag("/fake", "v1.0.0")
            assert result == "first-commit"

    def test_find_previous_tag_empty_tag_list(self):
        """Test fallback when tag list is empty (success=True, tags="")."""
        from ast_grep_mcp.features.documentation.changelog_generator import (
            _find_previous_tag,
        )

        with patch(self._MOCK_TARGET) as mock_git:
            mock_git.side_effect = [
                (True, ""),  # empty tag list
                (True, "first-commit"),  # rev-list for first commit
            ]
            result = _find_previous_tag("/fake", "v2.0.0")
            assert result == "first-commit"

    def test_find_previous_tag_git_failure(self):
        """Test fallback when git tag command fails."""
        from ast_grep_mcp.features.documentation.changelog_generator import (
            _find_previous_tag,
        )

        with patch(self._MOCK_TARGET) as mock_git:
            mock_git.side_effect = [
                (False, "error"),  # tag command fails
                (True, "first-commit"),  # rev-list for first commit
            ]
            result = _find_previous_tag("/fake", "v1.0.0")
            assert result == "first-commit"

    def test_resolve_version_ref_already_vprefixed(self):
        """Test input already starting with v skips double-prefix."""
        from ast_grep_mcp.features.documentation.changelog_generator import (
            _resolve_version_ref,
        )

        with patch(self._MOCK_TARGET) as mock_git:
            mock_git.side_effect = [
                (False, "not found"),  # vv1.2.0 fails
                (True, "abc123"),  # v1.2.0 succeeds as bare ref
            ]
            result = _resolve_version_ref("/fake", "v1.2.0")
            assert result == "v1.2.0"  # returned via bare-ref path

    def test_get_commit_range_to_version_uses_resolve(self):
        """Test to_version is resolved via _resolve_version_ref."""
        from ast_grep_mcp.features.documentation.changelog_generator import (
            _get_commit_range,
        )

        with patch(self._MOCK_TARGET) as mock_git:
            mock_git.side_effect = [
                (True, "abc123"),  # _resolve_version_ref: rev-parse v2.0.0
                (True, "v2.0.0\nv1.0.0"),  # _find_previous_tag: tag list
            ]
            from_ref, to_ref = _get_commit_range("/fake", None, "2.0.0")
            assert to_ref == "v2.0.0"

    def test_get_commit_range_from_version_uses_resolve(self):
        """Test from_version uses _resolve_version_ref (bare-ref fallback)."""
        from ast_grep_mcp.features.documentation.changelog_generator import (
            _get_commit_range,
        )

        with patch(self._MOCK_TARGET) as mock_git:
            mock_git.side_effect = [
                (True, "abc123"),  # _resolve_version_ref for to_version: v2.0.0
                (False, "not found"),  # _resolve_version_ref for from_version: vbad-ref fails
                (True, "def789"),  # _resolve_version_ref for from_version: bad-ref succeeds
            ]
            from_ref, to_ref = _get_commit_range("/fake", "bad-ref", "2.0.0")
            assert from_ref == "bad-ref"  # resolved via bare-ref fallback
            assert to_ref == "v2.0.0"

    def test_get_commit_range_from_version_vprefixed(self):
        """Test from_version resolves to v-prefixed tag."""
        from ast_grep_mcp.features.documentation.changelog_generator import (
            _get_commit_range,
        )

        with patch(self._MOCK_TARGET) as mock_git:
            mock_git.side_effect = [
                (True, "abc123"),  # _resolve_version_ref for to_version
                (True, "def456"),  # from_version v-prefix succeeds
            ]
            from_ref, to_ref = _get_commit_range("/fake", "1.0.0", "2.0.0")
            assert from_ref == "v1.0.0"
            assert to_ref == "v2.0.0"

    def test_get_commit_range_from_version_falls_back_to_head(self):
        """Test from_version falls back to HEAD when both v-prefix and bare ref fail."""
        from ast_grep_mcp.features.documentation.changelog_generator import (
            _get_commit_range,
        )

        with patch(self._MOCK_TARGET) as mock_git:
            mock_git.side_effect = [
                (True, "abc123"),  # _resolve_version_ref for to_version: v2.0.0
                (False, "not found"),  # _resolve_version_ref for from_version: vbogus fails
                (False, "not found"),  # _resolve_version_ref for from_version: bogus fails
            ]
            from_ref, to_ref = _get_commit_range("/fake", "bogus", "2.0.0")
            assert from_ref == "HEAD"  # falls back to HEAD
            assert to_ref == "v2.0.0"

    def test_get_commit_range_no_from_uses_previous_tag(self):
        """Test None from_version falls back to _find_previous_tag."""
        from ast_grep_mcp.features.documentation.changelog_generator import (
            _get_commit_range,
        )

        with patch(self._MOCK_TARGET) as mock_git:
            mock_git.side_effect = [
                (True, "abc123"),  # _resolve_version_ref for to_version
                (True, "v2.0.0\nv1.0.0"),  # _find_previous_tag: tag list
            ]
            from_ref, to_ref = _get_commit_range("/fake", None, "2.0.0")
            assert from_ref == "v1.0.0"  # previous tag, skipping v2.0.0
            assert to_ref == "v2.0.0"

    def test_run_git_command_invalid_cwd(self, tmp_path):
        """Test _run_git_command returns failure for non-existent directory."""
        from ast_grep_mcp.features.documentation.changelog_generator import (
            _run_git_command,
        )

        nonexistent = str(tmp_path / "does_not_exist")
        success, message = _run_git_command(nonexistent, ["status"])
        assert success is False
        assert "Directory not found" in message


class TestGetCommitsParsing:
    """Commit fields must survive '|' in subjects and author names."""

    def test_pipe_in_subject_and_author(self, tmp_path):
        import subprocess

        from ast_grep_mcp.features.documentation.changelog_generator import _get_commits

        def git(*args: str) -> None:
            subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

        git("init", "-q")
        (tmp_path / "f.txt").write_text("x\n")
        git("add", "f.txt")
        git("-c", "user.name=A|B", "-c", "user.email=a@b.c", "commit", "-q", "-m", "fix: handle a|b", "-m", "body ---COMMIT--- text")

        [commit] = _get_commits(str(tmp_path), "", "HEAD")

        assert commit.message == "fix: handle a|b"
        assert commit.author == "A|B"
        assert commit.author_email == "a@b.c"
        assert commit.body == "body ---COMMIT--- text"


class TestChangelogGenerator:
    """Tests for changelog generation."""

    def test_parse_conventional_commit(self):
        """Test conventional commit parsing."""
        from ast_grep_mcp.features.documentation.changelog_generator import (
            _parse_conventional_commit,
        )

        # Standard commit - function takes (subject, body)
        result = _parse_conventional_commit("feat: add user authentication", "")
        assert result is not None
        assert result["type"] == "feat"
        assert result["is_breaking"] is False

        # Breaking change
        result = _parse_conventional_commit("feat!: redesign API", "")
        assert result is not None
        assert result["is_breaking"] is True

        # With scope
        result = _parse_conventional_commit("fix(auth): resolve token refresh issue", "")
        assert result is not None
        assert result["type"] == "fix"
        assert result["scope"] == "auth"

        # Breaking change in body
        result = _parse_conventional_commit("feat: new feature", "BREAKING CHANGE: old API removed")
        assert result["is_breaking"] is True

        # Issue references
        result = _parse_conventional_commit("fix: resolve issue #123", "Fixes #456")
        assert "123" in result["issues"]
        assert "456" in result["issues"]


class TestSyncChecker:
    """Tests for documentation sync checking."""

    def test_extract_docstring_params_google(self):
        """Test extracting params from Google-style docstrings."""
        from ast_grep_mcp.features.documentation.sync_checker import (
            _extract_docstring_params,
        )

        docstring = '''"""Do something.

Args:
    user_id: The user ID.
    name: The user name.
    active (bool): Whether user is active.

Returns:
    User dictionary.
"""'''

        params = _extract_docstring_params(docstring, "python")
        assert "user_id" in params
        assert "name" in params
        assert "active" in params

    def test_extract_docstring_params_google_stops_at_next_section(self):
        """Indented docstrings must not leak Returns/Raises or wrapped lines into params."""
        from ast_grep_mcp.features.documentation.sync_checker import (
            _extract_docstring_params,
        )

        docstring = '''"""Do something.

        Args:
            a: First value, described at length so that it
                wraps onto: a continuation line.

            b (int): Second value.

        Returns:
            Sum of a and b.

        Raises:
            ValueError: If a is negative.
        """'''

        assert _extract_docstring_params(docstring, "python") == ["a", "b"]

    def test_extract_docstring_params_sphinx(self):
        """Test extracting params from Sphinx-style docstrings."""
        from ast_grep_mcp.features.documentation.sync_checker import (
            _extract_docstring_params,
        )

        docstring = '''"""Do something.

:param user_id: The user ID.
:param name: The user name.
:returns: User dictionary.
"""'''

        params = _extract_docstring_params(docstring, "python")
        assert "user_id" in params
        assert "name" in params

    def test_extract_docstring_params_jsdoc(self):
        """Test extracting params from JSDoc comments."""
        from ast_grep_mcp.features.documentation.sync_checker import (
            _extract_docstring_params,
        )

        docstring = """/**
 * Do something.
 * @param {string} userId - The user ID.
 * @param {string} name - The user name.
 * @returns {Object} User object.
 */"""

        params = _extract_docstring_params(docstring, "typescript")
        assert "userId" in params
        assert "name" in params

    def test_extract_docstring_return(self):
        """Test return documentation detection."""
        from ast_grep_mcp.features.documentation.sync_checker import (
            _extract_docstring_return,
        )

        # Google style
        assert _extract_docstring_return("Returns:\n    Value", "python") is True
        assert _extract_docstring_return("No return here", "python") is False

        # Sphinx style
        assert _extract_docstring_return(":return: Value", "python") is True
        assert _extract_docstring_return(":returns: Value", "python") is True

        # JSDoc
        assert _extract_docstring_return("@returns {string}", "typescript") is True
        assert _extract_docstring_return("@return {string}", "javascript") is True

    def test_check_docstring_sync_missing(self):
        """Test sync check for missing docstrings."""
        from ast_grep_mcp.features.documentation.sync_checker import (
            _check_docstring_sync,
        )

        func = FunctionSignature(
            name="get_user",
            parameters=[ParameterInfo(name="user_id", type_hint="int")],
            return_type="dict",
            decorators=[],
            is_async=False,
            is_method=False,
            file_path="/path/file.py",
            start_line=10,
            end_line=20,
            existing_docstring=None,
        )

        issues = _check_docstring_sync(func, "python")
        assert len(issues) == 1
        assert issues[0].issue_type == "undocumented"

    def test_check_docstring_sync_stale_param(self):
        """Test sync check for stale documented parameters."""
        from ast_grep_mcp.features.documentation.sync_checker import (
            _check_docstring_sync,
        )

        func = FunctionSignature(
            name="get_user",
            parameters=[ParameterInfo(name="user_id", type_hint="int")],
            return_type="dict",
            decorators=[],
            is_async=False,
            is_method=False,
            file_path="/path/file.py",
            start_line=10,
            end_line=20,
            existing_docstring='''"""Get user.

Args:
    user_id: The ID.
    old_param: This param no longer exists.

Returns:
    User dict.
"""''',
        )

        issues = _check_docstring_sync(func, "python")
        stale_issues = [i for i in issues if i.issue_type == "stale"]
        assert len(stale_issues) == 1
        assert "old_param" in stale_issues[0].description

    def test_check_markdown_links(self, tmp_path):
        """Test broken link detection in markdown files."""
        from ast_grep_mcp.features.documentation.sync_checker import (
            _check_markdown_links,
        )

        # Create existing file
        (tmp_path / "existing.md").write_text("# Existing")

        # Create markdown with links
        md_file = tmp_path / "test.md"
        md_file.write_text("""# Test

[Good link](existing.md)
[Broken link](nonexistent.md)
[External](https://example.com)
[Anchor](#section)
""")

        issues = _check_markdown_links(str(md_file), str(tmp_path))
        assert len(issues) == 1
        assert issues[0].issue_type == "broken_link"
        assert "nonexistent.md" in issues[0].description


class TestToolsIntegration:
    """Integration tests for documentation tools."""

    def test_generate_docstrings_impl(self, tmp_path):
        """Test docstring generation implementation."""
        from ast_grep_mcp.features.documentation.docstring_generator import (
            generate_docstrings_impl,
        )

        test_file = tmp_path / "test.py"
        test_file.write_text("""
def get_user(user_id: int) -> dict:
    return {"id": user_id}

def create_user(name: str, email: str) -> dict:
    return {"name": name, "email": email}
""")

        result = generate_docstrings_impl(
            project_folder=str(tmp_path),
            file_pattern="**/*.py",
            language="python",
            style="google",
            overwrite_existing=False,
        )

        assert result.total_functions >= 2
        assert result.functions_generated >= 2
        assert len(result.docstrings) >= 2

    def test_generate_readme_sections_impl(self, tmp_path):
        """Test README section generation implementation."""
        from ast_grep_mcp.features.documentation.readme_generator import (
            generate_readme_sections_impl,
        )

        # Create a simple project structure
        (tmp_path / "main.py").write_text("print('hello')")
        (tmp_path / "requirements.txt").write_text("requests\nflask")

        result = generate_readme_sections_impl(
            project_folder=str(tmp_path),
            sections=["installation", "usage"],
        )

        assert len(result.sections) >= 1
        assert any(s.title == "Installation" for s in result.sections)

    def test_sync_documentation_impl(self, tmp_path):
        """Test documentation sync checking implementation."""
        from ast_grep_mcp.features.documentation.sync_checker import (
            sync_documentation_impl,
        )

        # Create Python file with undocumented function
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def public_function(x: int) -> int:
    return x * 2
""")

        result = sync_documentation_impl(
            project_folder=str(tmp_path),
            language="python",
            doc_types=["docstrings"],
            check_only=True,
        )

        assert isinstance(result, DocSyncResult)
        assert result.total_functions >= 1
