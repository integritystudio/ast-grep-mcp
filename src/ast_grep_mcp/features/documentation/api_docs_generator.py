"""API documentation generation service.

This module provides functionality for generating API documentation
from route definitions across different web frameworks.
"""

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Protocol, Tuple

import sentry_sdk

from ast_grep_mcp.constants import ConversionFactors, FilePatterns, RegexCaptureGroups
from ast_grep_mcp.core.logging import get_logger
from ast_grep_mcp.models.documentation import (
    ApiDocsResult,
    ApiRoute,
    RouteParameter,
)
from ast_grep_mcp.utils.file_discovery import find_source_files, language_extensions

logger = get_logger(__name__)


# =============================================================================
# Framework Detection
# =============================================================================

# JS API frameworks: (dep_key, framework_name)
_JS_API_FRAMEWORKS: List[Tuple[str, str]] = [
    ("express", "express"),
    ("fastify", "fastify"),
    ("@nestjs/core", "nestjs"),
    ("koa", "koa"),
    ("hono", "hono"),
]

# Python API frameworks: (pattern, framework_name) - order matters (first match wins)
_PYTHON_API_FRAMEWORKS: List[Tuple[str, str]] = [
    ("fastapi", "fastapi"),
    ("flask", "flask"),
    ("django", "django"),
    ("starlette", "starlette"),
]


def _detect_js_api_framework(project_folder: str) -> Optional[str]:
    """Detect JavaScript API framework from package.json.

    Args:
        project_folder: Project root

    Returns:
        Framework name or None
    """
    package_json = os.path.join(project_folder, "package.json")
    if not os.path.exists(package_json):
        return None

    try:
        with open(package_json, "r") as f:
            data = json.load(f)
        deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
    except (json.JSONDecodeError, OSError):
        return None

    for dep_key, framework_name in _JS_API_FRAMEWORKS:
        if dep_key in deps:
            return framework_name
    return None


def _read_file_lower(filepath: str) -> str:
    """Return lowercased file contents, or empty string on error."""
    try:
        with open(filepath, "r") as f:
            return f.read().lower()
    except OSError:
        return ""


def _detect_python_api_framework(project_folder: str) -> Optional[str]:
    """Detect Python API framework from dependency files.

    Args:
        project_folder: Project root

    Returns:
        Framework name or None
    """
    deps_content = "".join(_read_file_lower(os.path.join(project_folder, fn)) for fn in ("requirements.txt", "pyproject.toml"))
    if not deps_content:
        return None
    for pattern, framework_name in _PYTHON_API_FRAMEWORKS:
        if pattern in deps_content:
            return framework_name
    return None


def _detect_framework(project_folder: str, language: str) -> Optional[str]:
    """Detect web framework used in the project.

    Args:
        project_folder: Project root
        language: Primary language

    Returns:
        Framework name or None
    """
    # Check JS frameworks first
    js_framework = _detect_js_api_framework(project_folder)
    if js_framework:
        return js_framework

    # Then check Python frameworks
    return _detect_python_api_framework(project_folder)


# =============================================================================
# Route Parsers
# =============================================================================


class RouteParser(Protocol):
    """Protocol for route parsers."""

    def parse_file(self, file_path: str) -> List[ApiRoute]:
        """Parse routes from a file."""
        ...


class ExpressRouteParser:
    """Parse Express.js routes."""

    def parse_file(self, file_path: str) -> List[ApiRoute]:
        """Parse routes from an Express file."""
        routes = []

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            lines = content.split("\n")

        # Pattern for app.get/post/etc. or router.get/post/etc.
        # Matches: app.get('/path', handler), router.post('/path', ...)
        route_pattern = re.compile(r'(?:app|router)\.(get|post|put|delete|patch|options|head)\s*\(\s*[\'"`]([^\'"`]+)[\'"`]')

        for i, line in enumerate(lines):
            match = route_pattern.search(line)
            if not match:
                continue
            method = match.group(RegexCaptureGroups.FIRST).upper()
            path = match.group(RegexCaptureGroups.SECOND)
            handler_match = re.search(r",\s*(\w+)\s*\)", line)
            handler_name = handler_match.group(RegexCaptureGroups.FIRST) if handler_match else "anonymous"
            routes.append(
                ApiRoute(
                    path=path,
                    method=method,
                    handler_name=handler_name,
                    file_path=file_path,
                    line_number=i + 1,
                    parameters=self._extract_path_params(path),
                )
            )

        return routes

    def _extract_path_params(self, path: str) -> List[RouteParameter]:
        """Extract path parameters from route path."""
        return [
            RouteParameter(name=m.group(RegexCaptureGroups.FIRST), location="path", required=True) for m in re.finditer(r":(\w+)", path)
        ]


class FastAPIRouteParser:
    """Parse FastAPI routes."""

    _DECORATOR_RE = re.compile(r'@(?:app|router)\.(get|post|put|delete|patch|options|head)\s*\(\s*[\'"`]([^\'"`]+)[\'"`]')
    _DEF_RE = re.compile(r"(?:async\s+)?def\s+(\w+)")

    def _find_func_line(self, lines: List[str], start: int) -> Optional[str]:
        """Return the first def/async def line after start, or None."""
        for line in lines[start:]:
            stripped = line.strip()
            if stripped.startswith("def ") or stripped.startswith("async def "):
                return line
        return None

    def _parse_handler(self, lines: List[str], decorator_idx: int) -> "tuple[str, List[RouteParameter]]":
        """Return (handler_name, params) from the function following decorator_idx."""
        func_line = self._find_func_line(lines, decorator_idx + 1)
        if not func_line:
            return "unknown", []
        name_match = self._DEF_RE.search(func_line)
        handler_name = name_match.group(RegexCaptureGroups.FIRST) if name_match else "unknown"
        return handler_name, self._extract_params(func_line)

    def parse_file(self, file_path: str) -> List[ApiRoute]:
        """Parse routes from a FastAPI file."""
        routes = []
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.read().split("\n")

        for i, line in enumerate(lines):
            match = self._DECORATOR_RE.search(line)
            if not match:
                continue
            method = match.group(RegexCaptureGroups.FIRST).upper()
            path = match.group(RegexCaptureGroups.SECOND)
            handler_name, params = self._parse_handler(lines, i)
            routes.append(
                ApiRoute(
                    path=path,
                    method=method,
                    handler_name=handler_name,
                    file_path=file_path,
                    line_number=i + 1,
                    parameters=self._extract_path_params(path) + params,
                )
            )

        return routes

    def _extract_path_params(self, path: str) -> List[RouteParameter]:
        """Extract path parameters from route path."""
        return [
            RouteParameter(name=m.group(RegexCaptureGroups.FIRST), location="path", required=True) for m in re.finditer(r"\{(\w+)\}", path)
        ]

    def _param_from_part(self, part: str) -> Optional[RouteParameter]:
        """Return a RouteParameter for a Query/Body annotated param part, or None."""
        if "Query" in part:
            location = "query"
        elif "Body" in part:
            location = "body"
        else:
            return None
        name_match = re.match(r"(\w+)\s*:", part)
        if not name_match:
            return None
        return RouteParameter(
            name=name_match.group(RegexCaptureGroups.FIRST),
            location=location,
            required="..." not in part,
        )

    def _extract_params(self, func_line: str) -> List[RouteParameter]:
        """Extract query/body parameters from function signature."""
        param_match = re.search(r"\((.*)\)", func_line)
        if not param_match:
            return []
        parts = [p.strip() for p in param_match.group(RegexCaptureGroups.FIRST).split(",")]
        params = []
        for part in parts:
            if not part or part in ("self", "cls"):
                continue
            p = self._param_from_part(part)
            if p:
                params.append(p)
        return params


class FlaskRouteParser:
    """Parse Flask routes."""

    _DECORATOR_RE = re.compile(r'@(?:\w+)\.route\s*\(\s*[\'"`]([^\'"`]+)[\'"`](?:.*methods\s*=\s*\[([^\]]+)\])?')

    def parse_file(self, file_path: str) -> List[ApiRoute]:
        """Parse routes from a Flask file."""
        routes: List[ApiRoute] = []
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.read().split("\n")

        for i, line in enumerate(lines):
            match = self._DECORATOR_RE.search(line)
            if not match:
                continue
            path, methods = self._parse_decorator_match(match)
            handler_name = self._find_next_handler_name(lines, i)
            routes.extend(self._build_routes_for_methods(path, methods, handler_name, file_path, i))

        return routes

    def _parse_decorator_match(self, match: "re.Match[str]") -> "tuple[str, list[str]]":
        """Return (path, methods) from a @*.route decorator match."""
        path = match.group(RegexCaptureGroups.FIRST)
        methods_str = match.group(RegexCaptureGroups.SECOND)
        methods = [m.strip().strip("'\"") for m in methods_str.split(",")] if methods_str else ["GET"]
        return path, methods

    def _find_next_handler_name(self, lines: List[str], decorator_idx: int) -> str:
        """Return the function name on the next `def` line after decorator_idx."""
        for line in lines[decorator_idx + 1 :]:
            stripped = line.strip()
            if stripped.startswith("def "):
                name_match = re.search(r"def\s+(\w+)", line)
                return name_match.group(RegexCaptureGroups.FIRST) if name_match else "unknown"
            if stripped and not stripped.startswith(("@", "#")):
                break
        return "unknown"

    def _build_routes_for_methods(self, path: str, methods: List[str], handler_name: str, file_path: str, line_idx: int) -> List[ApiRoute]:
        """Build one ApiRoute per HTTP method."""
        path_params = self._extract_path_params(path)
        return [
            ApiRoute(
                path=path,
                method=method.upper(),
                handler_name=handler_name,
                file_path=file_path,
                line_number=line_idx + 1,
                parameters=path_params,
            )
            for method in methods
        ]

    def _extract_path_params(self, path: str) -> List[RouteParameter]:
        """Extract path parameters from route path."""
        return [
            RouteParameter(
                name=m.group(RegexCaptureGroups.SECOND),
                location="path",
                type_hint=m.group(RegexCaptureGroups.FIRST),
                required=True,
            )
            for m in re.finditer(r"<(?:(\w+):)?(\w+)>", path)
        ]


# =============================================================================
# Documentation Generators
# =============================================================================


def _group_routes_by_prefix(routes: List[ApiRoute]) -> Dict[str, List[ApiRoute]]:
    groups: Dict[str, List[ApiRoute]] = {}
    for route in routes:
        parts = route.path.strip("/").split("/")
        group = parts[0] if parts else "root"
        groups.setdefault(group, []).append(route)
    return groups


def _markdown_param_table(route: ApiRoute) -> List[str]:
    if not route.parameters:
        return []
    lines = [
        "**Parameters:**",
        "",
        "| Name | Location | Type | Required | Description |",
        "|------|----------|------|----------|-------------|",
    ]
    for param in route.parameters:
        required = "Yes" if param.required else "No"
        lines.append(f"| `{param.name}` | {param.location} | {param.type_hint or 'string'} | {required} | {param.description or '-'} |")
    lines.append("")
    return lines


def _markdown_default_responses() -> List[str]:
    return [
        "**Responses:**",
        "",
        "| Status | Description |",
        "|--------|-------------|",
        "| 200 | Success |",
        "| 400 | Bad Request |",
        "| 404 | Not Found |",
        "| 500 | Internal Server Error |",
        "",
    ]


def _markdown_route_section(route: ApiRoute) -> List[str]:
    lines: List[str] = [f"### `{route.method}` {route.path}", ""]
    if route.description:
        lines += [route.description, ""]
    lines.append(f"**Handler:** `{route.handler_name}` ({route.file_path}:{route.line_number})")
    lines.append("")
    lines += _markdown_param_table(route)
    if not route.responses:
        lines += _markdown_default_responses()
    lines += ["---", ""]
    return lines


def _generate_markdown_docs(routes: List[ApiRoute], framework: str) -> str:
    """Generate Markdown API documentation.

    Args:
        routes: List of parsed routes
        framework: Framework name

    Returns:
        Markdown string
    """
    lines: List[str] = ["# API Documentation", "", f"Generated from {framework} routes.", ""]
    for group, group_routes in sorted(_group_routes_by_prefix(routes).items()):
        lines += [f"## {group.title()} Endpoints", ""]
        for route in sorted(group_routes, key=lambda r: (r.path, r.method)):
            lines += _markdown_route_section(route)
    return "\n".join(lines)


def _build_openapi_param(param: RouteParameter) -> Dict[str, Any]:
    """Build OpenAPI parameter object.

    Args:
        param: Route parameter

    Returns:
        OpenAPI parameter object
    """
    param_obj: Dict[str, Any] = {
        "name": param.name,
        "in": param.location if param.location != "body" else "query",
        "required": param.required,
        "schema": {"type": param.type_hint or "string"},
    }
    if param.description:
        param_obj["description"] = param.description
    return param_obj


def _build_openapi_request_body(body_params: List[RouteParameter]) -> Dict[str, Any]:
    """Build OpenAPI request body object.

    Args:
        body_params: Body parameters

    Returns:
        OpenAPI requestBody object
    """
    properties = {p.name: {"type": p.type_hint or "string"} for p in body_params}
    schema: Dict[str, Any] = {"type": "object", "properties": properties}
    json_content: Dict[str, Any] = {"application/json": {"schema": schema}}
    return {"content": json_content}


def _build_openapi_operation(route: ApiRoute) -> Dict[str, Any]:
    """Build OpenAPI operation object for a route.

    Args:
        route: API route

    Returns:
        OpenAPI operation object
    """
    method = route.method.lower()
    operation: Dict[str, Any] = {
        "operationId": route.handler_name,
        "summary": route.description or f"{route.method} {route.path}",
        "responses": {"200": {"description": "Successful response"}},
    }

    # Add parameters
    parameters = [_build_openapi_param(p) for p in route.parameters]
    if parameters:
        operation["parameters"] = parameters

    # Add request body for POST/PUT/PATCH
    if method in ("post", "put", "patch"):
        body_params = [p for p in route.parameters if p.location == "body"]
        if body_params:
            operation["requestBody"] = _build_openapi_request_body(body_params)

    return operation


def _generate_openapi_spec(routes: List[ApiRoute], project_name: str = "API") -> Dict[str, Any]:
    """Generate OpenAPI 3.0 specification.

    Args:
        routes: List of parsed routes
        project_name: Project name for the spec

    Returns:
        OpenAPI spec as dictionary
    """
    spec: Dict[str, Any] = {
        "openapi": "3.0.0",
        "info": {"title": project_name, "version": "1.0.0"},
        "paths": {},
    }

    for route in routes:
        if route.path not in spec["paths"]:
            spec["paths"][route.path] = {}
        spec["paths"][route.path][route.method.lower()] = _build_openapi_operation(route)

    return spec


# =============================================================================
# Main Generator
# =============================================================================


_ROUTE_PATTERNS: Dict[str, List[str]] = {
    "express": ["routes", "router", "api", "controllers"],
    "fastapi": ["routes", "router", "api", "endpoints", "views"],
    "flask": ["routes", "views", "api", "blueprints"],
    "fastify": ["routes", "router", "api"],
    "nestjs": ["controller", "routes"],
}

def _is_route_file(full_path: str, file: str, project_folder: str, patterns: List[str], exts: List[str]) -> bool:
    if not any(file.endswith(ext) for ext in exts):
        return False
    rel_path = os.path.relpath(full_path, project_folder).lower()
    return any(p in rel_path for p in patterns) or any(p in file.lower() for p in patterns)


def _find_route_files(project_folder: str, language: str, framework: str) -> List[str]:
    """Find files likely to contain route definitions.

    Args:
        project_folder: Project root
        language: Primary language
        framework: Framework name

    Returns:
        List of file paths
    """
    patterns = _ROUTE_PATTERNS.get(framework, ["routes", "api", "controllers"])
    exts = language_extensions(language, fallback=[".py", ".js", ".ts"])
    candidates = find_source_files(
        project_folder, language, extensions=exts, skip_dir_names=FilePatterns.SKIP_DIR_NAMES
    )
    return [str(f) for f in candidates if _is_route_file(str(f), f.name, project_folder, patterns, exts)]


_PARSERS: Dict[str, RouteParser] = {
    "express": ExpressRouteParser(),
    "fastapi": FastAPIRouteParser(),
    "flask": FlaskRouteParser(),
    "fastify": ExpressRouteParser(),
    "starlette": FastAPIRouteParser(),
}


def _parse_all_routes(parser: RouteParser, route_files: List[str]) -> List[ApiRoute]:
    all_routes: List[ApiRoute] = []
    for file_path in route_files:
        try:
            all_routes.extend(parser.parse_file(file_path))
        except Exception as e:
            logger.warning("file_parse_error", file=file_path, error=str(e))
            sentry_sdk.capture_exception(e)
    return all_routes


def _elapsed_ms(start_time: float) -> int:
    return int((time.time() - start_time) * ConversionFactors.MILLISECONDS_PER_SECOND)


def _build_docs_result(
    project_folder: str,
    framework: str,
    parser: RouteParser,
    language: str,
    output_format: str,
    start_time: float,
) -> ApiDocsResult:
    route_files = _find_route_files(project_folder, language, framework)
    all_routes = _parse_all_routes(parser, route_files)
    markdown = _generate_markdown_docs(all_routes, framework)
    openapi_spec = None
    if output_format in ("openapi", "both"):
        openapi_spec = _generate_openapi_spec(all_routes, os.path.basename(project_folder))
    execution_time = _elapsed_ms(start_time)
    logger.info("generate_api_docs_completed", routes_found=len(all_routes), framework=framework, execution_time_ms=execution_time)
    return ApiDocsResult(
        routes=all_routes,
        markdown=markdown,
        openapi_spec=openapi_spec,
        framework=framework,
        execution_time_ms=execution_time,
    )


def generate_api_docs_impl(
    project_folder: str,
    language: str,
    framework: Optional[str] = None,
    output_format: str = "markdown",
    include_examples: bool = True,
) -> ApiDocsResult:
    """Generate API documentation from route definitions.

    Args:
        project_folder: Root folder of the project
        language: Programming language
        framework: Framework name (or None for auto-detect)
        output_format: Output format ('markdown', 'openapi', 'both')
        include_examples: Whether to include request/response examples

    Returns:
        ApiDocsResult with generated documentation
    """
    start_time = time.time()
    logger.info(
        "generate_api_docs_started",
        project_folder=project_folder,
        language=language,
        framework=framework,
        output_format=output_format,
    )

    if not framework:
        framework = _detect_framework(project_folder, language)

    if not framework:
        logger.warning("no_framework_detected")
        return ApiDocsResult(
            routes=[],
            markdown="# API Documentation\n\nNo web framework detected.",
            framework=None,
            execution_time_ms=_elapsed_ms(start_time),
        )

    parser = _PARSERS.get(framework)
    if not parser:
        logger.warning("unsupported_framework", framework=framework)
        return ApiDocsResult(
            routes=[],
            markdown=f"# API Documentation\n\nFramework '{framework}' is not yet supported.",
            framework=framework,
            execution_time_ms=_elapsed_ms(start_time),
        )

    return _build_docs_result(project_folder, framework, parser, language, output_format, start_time)
