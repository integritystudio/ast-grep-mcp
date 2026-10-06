"""Tests for OrphanDetector JS/TS import resolution and two-pass graph build."""

from pathlib import Path

import pytest

from ast_grep_mcp.features.quality.orphan_detector import OrphanDetector
from ast_grep_mcp.models.orphan import OrphanAnalysisConfig


@pytest.fixture
def detector() -> OrphanDetector:
    return OrphanDetector(
        OrphanAnalysisConfig(
            include_patterns=["**/*.ts", "**/*.tsx", "**/*.js", "**/*.jsx", "**/*.py"],
        )
    )


@pytest.fixture
def ts_project(tmp_path: Path) -> Path:
    """TS project with various import styles."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "utils").mkdir()
    return tmp_path


class TestResolveJsRelativeImport:
    """Tests for _resolve_js_relative_import edge cases."""

    def test_extensionless_resolves_to_ts(self, detector: OrphanDetector, ts_project: Path) -> None:
        """import './foo' should resolve to foo.ts on disk."""
        src = ts_project / "src"
        (src / "foo.ts").write_text("export const x = 1;")
        index = src / "index.ts"
        index.write_text("import { x } from './foo';")

        result = detector._resolve_js_relative_import(index, "./foo", ts_project)
        assert result == "src/foo.ts"

    def test_js_extension_remaps_to_ts(self, detector: OrphanDetector, ts_project: Path) -> None:
        """import './helper.js' should resolve to helper.ts (TS ESM convention)."""
        src = ts_project / "src"
        (src / "helper.ts").write_text("export function help() {}")
        index = src / "index.ts"
        index.write_text("import { help } from './helper.js';")

        result = detector._resolve_js_relative_import(index, "./helper.js", ts_project)
        assert result == "src/helper.ts"

    def test_js_extension_remaps_to_tsx(self, detector: OrphanDetector, ts_project: Path) -> None:
        """import './Component.js' should resolve to Component.tsx when .ts doesn't exist."""
        src = ts_project / "src"
        (src / "Component.tsx").write_text("export default function C() {}")
        index = src / "index.ts"
        index.write_text("import C from './Component.js';")

        result = detector._resolve_js_relative_import(index, "./Component.js", ts_project)
        assert result == "src/Component.tsx"

    def test_jsx_extension_remaps_to_tsx(self, detector: OrphanDetector, ts_project: Path) -> None:
        """import './Widget.jsx' should resolve to Widget.tsx."""
        src = ts_project / "src"
        (src / "Widget.tsx").write_text("export default function W() {}")
        index = src / "index.ts"
        index.write_text("import W from './Widget.jsx';")

        result = detector._resolve_js_relative_import(index, "./Widget.jsx", ts_project)
        assert result == "src/Widget.tsx"

    def test_extensionless_resolves_to_index(self, detector: OrphanDetector, ts_project: Path) -> None:
        """import './utils' should resolve to utils/index.ts."""
        src = ts_project / "src"
        (src / "utils" / "index.ts").write_text("export const util = 1;")
        index = src / "index.ts"
        index.write_text("import { util } from './utils';")

        result = detector._resolve_js_relative_import(index, "./utils", ts_project)
        assert result == "src/utils/index.ts"

    def test_actual_js_file_resolves_directly(self, detector: OrphanDetector, ts_project: Path) -> None:
        """import './legacy.js' resolves to legacy.js when file exists (no .ts on disk)."""
        src = ts_project / "src"
        (src / "legacy.js").write_text("module.exports = {};")
        index = src / "index.ts"
        index.write_text("import legacy from './legacy.js';")

        result = detector._resolve_js_relative_import(index, "./legacy.js", ts_project)
        assert result == "src/legacy.js"

    def test_nonexistent_import_returns_none(self, detector: OrphanDetector, ts_project: Path) -> None:
        """Unresolvable import returns None."""
        index = ts_project / "src" / "index.ts"
        index.write_text("")

        result = detector._resolve_js_relative_import(index, "./missing", ts_project)
        assert result is None

    def test_js_remap_prefers_ts_over_tsx(self, detector: OrphanDetector, ts_project: Path) -> None:
        """When both .ts and .tsx exist, .ts should be preferred for .js imports."""
        src = ts_project / "src"
        (src / "dual.ts").write_text("export const a = 1;")
        (src / "dual.tsx").write_text("export default function D() {}")
        index = src / "index.ts"
        index.write_text("import { a } from './dual.js';")

        result = detector._resolve_js_relative_import(index, "./dual.js", ts_project)
        assert result == "src/dual.ts"


class TestBuildDependencyGraphTwoPass:
    """Verify two-pass build resolves edges regardless of file discovery order."""

    def test_edge_resolved_when_target_discovered_after_source(self, detector: OrphanDetector, tmp_path: Path) -> None:
        """Target file discovered in pass 1 should be resolvable in pass 2 edge resolution."""
        src = tmp_path / "src"
        src.mkdir()
        # a.ts imports b.ts; if files were single-pass and a.ts is visited first,
        # b.ts wouldn't be in graph.files yet
        (src / "a.ts").write_text("import { b } from './b';")
        (src / "b.ts").write_text("export const b = 1;")

        graph = detector._build_dependency_graph(tmp_path)

        assert "src/a.ts" in graph.files
        assert "src/b.ts" in graph.files
        edges = [(e.source, e.target) for e in graph.edges]
        assert ("src/a.ts", "src/b.ts") in edges

    def test_python_cross_imports_resolved(self, detector: OrphanDetector, tmp_path: Path) -> None:
        """Python files importing each other should both appear as edges."""
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "alpha.py").write_text("from pkg.beta import thing")
        (pkg / "beta.py").write_text("from pkg.alpha import other")

        graph = detector._build_dependency_graph(tmp_path)

        assert "pkg/alpha.py" in graph.files
        assert "pkg/beta.py" in graph.files
        sources = {e.source for e in graph.edges}
        assert "pkg/alpha.py" in sources
        assert "pkg/beta.py" in sources

    def test_all_files_collected_before_edge_extraction(self, detector: OrphanDetector, tmp_path: Path) -> None:
        """All files should be in graph.files before any import extraction occurs."""
        src = tmp_path / "src"
        src.mkdir()
        # Create a chain: c imports b, b imports a
        (src / "a.ts").write_text("export const a = 1;")
        (src / "b.ts").write_text("import { a } from './a';\nexport const b = a;")
        (src / "c.ts").write_text("import { b } from './b';")

        graph = detector._build_dependency_graph(tmp_path)

        assert len(graph.files) == 3
        targets = {e.target for e in graph.edges}
        assert "src/a.ts" in targets
        assert "src/b.ts" in targets


class TestImportFormsCreateEdges:
    """Import forms that reference a file must create an edge, or the file looks orphaned."""

    def test_bare_relative_import_of_sibling(self, detector: OrphanDetector, tmp_path: Path) -> None:
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "helpers.py").write_text("X = 1\n")
        (pkg / "main.py").write_text("from . import helpers\n")

        graph = detector._build_dependency_graph(tmp_path)

        assert graph.get_importers("pkg/helpers.py")

    def test_barrel_reexport_and_dynamic_import(self, detector: OrphanDetector, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.ts").write_text("export const a = 1;")
        (src / "b.ts").write_text("export const b = 2;")
        (src / "index.ts").write_text("export { a } from './a';\nconst lazy = () => import('./b');\n")

        graph = detector._build_dependency_graph(tmp_path)

        assert graph.get_importers("src/a.ts")
        assert graph.get_importers("src/b.ts")

    def test_non_utf8_orphan_does_not_abort(self, detector: OrphanDetector, tmp_path: Path) -> None:
        (tmp_path / "latin1.py").write_bytes(b"# caf\xe9\nx = 1\n")

        result = detector.analyze(str(tmp_path))

        assert [o.file_path for o in result.orphan_files] == ["latin1.py"]
