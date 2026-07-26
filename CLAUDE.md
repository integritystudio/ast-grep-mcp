# CLAUDE.md

## Quick Start

```bash
uv sync                          # Install dependencies
uv run pytest                    # Run all tests (1,849 collected)
uv run pytest tests/unit/        # Unit tests only
uv run pytest tests/integration/ # Integration tests (slower)
uv run pytest tests/quality/     # Quality regression tests
uv run pytest tests/performance/ # Performance benchmarks (slow)
uv run ruff check . && uv run mypy src/ # Lint and type check
uv run main.py                   # Run MCP server locally
doppler run -- uv run main.py    # Run with Doppler secrets
```

**Tip:** Use `make help` to see all available commands (or `make <target>` to run tests directly).

## Testing Tips

**Run a single test:** `uv run pytest tests/unit/test_foo.py::TestClass::test_method -v`

**Filter by name:** `uv run pytest -k "cache" -v` (runs tests matching "cache")

**Stop on first failure:** `uv run pytest -x`

**With coverage:** `uv run pytest --cov=src/ast_grep_mcp --cov-report=term-missing`

## Environment Setup

**uv vs bare python:** Always prefix commands with `uv run`. Bare `python` fails with ModuleNotFoundError (see project memory for context).

**Secrets:** Use `doppler run` when working with sensitive APIs (GitHub, Sentry, etc.). For local dev without secrets, set env vars directly: `LOG_LEVEL=debug uv run main.py`.

**Tool language argument:** `analyze_complexity`, `detect_code_smells`, `detect_security_issues` require explicit `language` parameter (not auto-detected).

## Project Memory

See project memory (persisted auto-context in Claude Code) for session-persisted patterns: Python execution quirks, tool parameter requirements, non-obvious gotchas. Updated automatically across sessions; check it when encountering familiar-looking errors.

## Overview

Modular MCP server (126 modules) with ast-grep structural code search, Schema.org tools, refactoring, deduplication, quality, documentation generation, and semantic code condensation.

**59 Tools:** Search (9), Rewrite (5), Refactoring (2), Deduplication (6), Schema.org (11), Complexity (3), Quality (7), Documentation (5), Cross-Language (5), Condense (6)

**Deps:** ast-grep CLI (required), Doppler CLI (optional), Python 3.13+, uv

## Architecture

### Efficiency
- **file tree** load docs/repomix/token-tree.txt into memory before multi-file read/explore or
refactoring tasks for context.
- **compression** use token-tree to identify the correct section in the lossless
  compression file at docs/repomix/repomix.xml during complex debugging or read/explore operations


### Structure
```
src/ast_grep_mcp/
├── core/           # Config, cache, executor, logging, sentry, usage tracking
├── models/         # Data models
├── utils/          # Templates, formatters, validation
├── features/       # search, rewrite, refactoring, schema, deduplication, complexity, quality, documentation, cross_language, condense
└── server/         # MCP server registry
```

**Import:** `from ast_grep_mcp.features.search.service import find_code_impl`

## Code Quality

Quality gates: Ruff + mypy + pytest + analyze_codebase.py

```bash
uv run pytest tests/quality/test_complexity_regression.py -v
```

## Config

**Environment:** `AST_GREP_CONFIG`, `LOG_LEVEL`, `SENTRY_DSN`, `CACHE_DISABLED`/`CACHE_SIZE`/`CACHE_TTL`

Config loads via **pydantic-settings** — env vars are auto-read and type-coerced on `AstGrepConfig` (`core/config.py`); no manual `os.getenv` needed.

## Caching & Timeouts

**MinHash & CodeBERT similarity caches** (BUG-12, 2026-07-26): Keyed on exact code string (not `hash()`), bounded at 1024 entries via LRU, with move-to-end on hit. Located in `similarity.py` and `ranker.py`.

**Timeout helpers** (BUG-06, 2026-07-25): Shared `utils/futures.py:map_with_per_item_timeout()` used by dedup enrichment, batch coverage, ranker, and multi-language search. Enforces per-item timeouts plus an optional shared deadline (`total_timeout_seconds`; enrichment caps it at `MAX_TIMEOUT_SECONDS` = 300s). Callers use `WaitTimeoutError` to distinguish wait timeouts from worker-raised `TimeoutError`.

## Tool Response Field Names

When calling tools programmatically, use these field names (NOT `line`/`file_path`):

- **analyze_complexity** (`complexity.tools`): top-level keys: `summary`, `thresholds`, `functions`, `message`, `storage`. `summary` keys: `total_functions`, `total_files`, `exceeding_threshold` (no trailing `s`), `avg_cyclomatic`, `avg_cognitive`, `max_cyclomatic`, `max_cognitive`, `max_nesting`, `analysis_time_seconds`. `functions[]` with `name`, `file`, `lines`, `cyclomatic`, `cognitive`, `nesting_depth`, `length`, `exceeds`
- **detect_code_smells** (`complexity.tools`): top-level keys: `project_folder`, `language`, `files_analyzed`, `total_smells`, `summary`, `smells`, `thresholds`, `execution_time_ms`. `summary` has `by_type` and `by_severity` (`high`/`medium`/`low`). `smells[]` with `file`, `line`, `severity`, `smell_type`, `message`
- **find_duplication** (`deduplication.tools`): `summary`, `duplication_groups[]`, `refactoring_suggestions[]`; group keys: `group_id`, `similarity_score`, `instances[]` (with `file`, `lines`, `code_preview`)
- **benchmark_deduplication** (`deduplication.tools`): `results[]` with `name`, `mean_ms`, `median_ms`, `p95_ms`
- **enforce_standards** (`quality.tools`): `summary`, `violations[]` with `file`, `line`, `column`, `severity`, `rule_id`, `message`, `code_snippet`
- **detect_security_issues** (`quality.tools`): `summary`, `issues[]` with `file`, `line`, `severity`, `issue_type`, `title`, `cwe_id`
- **detect_orphans** (`quality.tools`): `summary`, `orphan_files[]` (with `file_path`, `lines`, `status`), `orphan_functions[]` (with `file`, `line`, `name`)

Import pattern: `from ast_grep_mcp.features.<module>.tools import <tool_name>_tool`
Exception: **search** tools use `_impl` functions — `from ast_grep_mcp.features.search.service import find_code_impl` (there is no `find_code_tool`).

## Public API vs Internals

- **search tools** — use `find_code_impl`, `find_code_by_rule_impl`, `dump_syntax_tree_impl`, `debug_pattern_impl`, `build_rule_impl`, `develop_pattern_impl` from `search.service`. The `tools.py` registers inner functions via `@mcp.tool()` that are not importable.
- **search doc helpers** — `get_docs(topic)` and `get_pattern_examples(language, category=None)` live in `search.docs`, not `search.service`. There is no `get_ast_grep_docs_impl` or `get_pattern_examples_tool`.
- **get_zod_rewrite_rule / list_zod_rewrite_rules** — from `rewrite.tools`; `get_zod_rewrite_rule(rule_id)` returns `{rule_id, yaml_rule}` ready for `rewrite_code()`. `list_zod_rewrite_rules(category)` browses rules by `'schema'`, `'import'`, or `'all'`. Rule IDs defined in `rewrite.zod_templates`.
- **rewrite_code** — `from ast_grep_mcp.features.rewrite.service import rewrite_code_impl`; signature is `rewrite_code_impl(project_folder, yaml_rule, dry_run=True, backup=True, ...)`. The `yaml_rule` must be a complete YAML rule string with `fix` field — do NOT pass separate `pattern`/`replacement`/`language` positional args.
- **extract_function** — always call via `extract_function_tool(project_folder, file_path, start_line, end_line, language)` from `refactoring.tools`. Do NOT instantiate `FunctionExtractor` directly; it is an internal class that requires `language` and does not accept `project_folder`.
- **refactor_polyglot** — `refactoring_type` accepts `rename_api`, `extract_constant`, `update_contract`. `rename` is also accepted as an alias for `rename_api`.
- **deduplication** — use `analyze_deduplication_candidates_tool()` or `find_duplication_tool()` from `deduplication.tools`. There is no `find_dedup_candidates`.
- **generate_language_bindings** — expects OpenAPI/Swagger spec, NOT `package.json`.
- **build_entity_graph** — expects entity definition dicts, NOT raw JSON-LD. **enhance_entity_graph** → use `analyze_entity_graph()` from `schema.enhancement_service`; expects existing JSON-LD files.
- **YAML `$VAR` in patterns** — use raw strings or single-quoted YAML to prevent shell expansion of `$MSG`, `$NAME`, etc.
- **deduplication reporting** — `create_enhanced_duplication_response(candidates, include_diffs, include_colors)` is a method on `DuplicationReporter` in `deduplication.reporting`, not a module-level function.
- **deduplication recommendations** — `generate_deduplication_recommendation(score, complexity, lines, has_tests, files)` is a method on `RecommendationEngine` in `deduplication.recommendations`, not a module-level function.
- **DuplicationRanker parallelization** — pass `max_workers=N` to enable parallel candidate scoring (ThreadPoolExecutor); `max_workers=0` disables it. Default (`None`) uses ThreadPoolExecutor's default pool.
- **calculate_ast_similarity** / **calculate_semantic_similarity** — import as `calculate_ast_similarity_tool` / `calculate_semantic_similarity_tool` from `deduplication.tools`. Accept two code strings + language; return `{"similarity_score": float, "method": str, ...}`. Semantic variant uses CodeBERT; AST variant is structural.

See [docs/BACKLOG.md](docs/BACKLOG.md) for open items and deferred work.

## Analysis Scripts

- Full analysis suite: `uv run python scripts/run_all_analysis.py [src_path]`

## Notes

- YAML rules support `kind`-based matching (e.g., `kind: catch_clause` with `has`); add `stopBy: end` to relational rules
- Windows: use `shell=True` for npm-installed ast-grep
- **MCP tool handlers are synchronous** — call directly, do NOT wrap in `asyncio.run()`. Exception: `async_stream_ast_grep_results()` in `core/executor.py` is async; schema tools in `schema/tools.py` are async.
- CLI invocation: `uv run python -c "from ast_grep_mcp.features.X.tools import Y; print(Y(...))"`
- Codebase analyzer: `uv run python analyze_codebase.py <path> -l <language> [--fix]`
- ast-grep supported languages: python, javascript, typescript, tsx, html, css, json, yaml, rust, go, java, kotlin, c, cpp, csharp, swift, ruby, lua, scala — **not** dart

## Docs

- [docs/CHANGELOG.md](docs/CHANGELOG.md) - Version history (index to `docs/changelog/` entries)
- [docs/PATTERNS.md](docs/PATTERNS.md) - Refactoring patterns
- [docs/DEDUPLICATION-GUIDE.md](docs/DEDUPLICATION-GUIDE.md) - Deduplication workflow
- [docs/CONFIGURATION.md](docs/CONFIGURATION.md) - Configuration options
- [docs/SENTRY-INTEGRATION.md](docs/SENTRY-INTEGRATION.md) - Error tracking
- [docs/BENCHMARKING.md](docs/BENCHMARKING.md) - Performance benchmarking
- [docs/CODE-CONDENSE-PHASE-2.md](docs/CODE-CONDENSE-PHASE-2.md) - Condense phase 2 design
- [docs/BACKLOG.md](docs/BACKLOG.md) - Open backlog items
- [docs/BACKFILLING.md](docs/BACKFILLING.md) - OTEL telemetry backfilling for skills
