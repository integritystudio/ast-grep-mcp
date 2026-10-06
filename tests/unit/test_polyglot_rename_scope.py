"""Tests that polyglot renames only touch project sources and are backed up."""

from ast_grep_mcp.features.cross_language.tools import refactor_polyglot_tool

OLD, NEW = "fetchUser", "loadUser"


def test_apply_skips_vendored_dirs_and_returns_backup(tmp_path) -> None:
    src = tmp_path / "src"
    vendored = tmp_path / "node_modules" / "lib"
    src.mkdir()
    vendored.mkdir(parents=True)
    (src / "app.js").write_text(f"export function {OLD}() {{}}\n")
    (vendored / "index.js").write_text(f"function {OLD}() {{}}\n")

    result = refactor_polyglot_tool(str(tmp_path), "rename_api", OLD, NEW, ["javascript"], dry_run=False)

    assert NEW in (src / "app.js").read_text()
    assert OLD in (vendored / "index.js").read_text()
    assert result["backup_id"]


def test_project_inside_skip_named_dir_is_still_scanned(tmp_path) -> None:
    project = tmp_path / "venv" / "proj"
    project.mkdir(parents=True)
    (project / "app.js").write_text(f"export function {OLD}() {{}}\n")

    result = refactor_polyglot_tool(str(project), "rename_api", OLD, NEW, ["javascript"], dry_run=False)

    assert NEW in (project / "app.js").read_text()
    assert result["files_modified"]
