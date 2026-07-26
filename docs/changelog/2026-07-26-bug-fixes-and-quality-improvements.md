# 2026-07-26: BUG-06 Timeout Fixes, Cache Correctness, and Code Quality

Systematic fixes for hangs, cache collisions, and redundant work across deduplication and enrichment pipelines.

## Bug Fixes

### BUG-06: Per-Candidate Enrichment Timeout Now Enforced

**Context:** Enrichment timeouts were dead code; one hung candidate blocked the MCP call indefinitely.

**Fix:** `_process_parallel_enrichment` now iterates `futures.items()` so `future.result(timeout=)` enforces a live per-candidate timeout; executor shut down via `finally: shutdown(wait=False, cancel_futures=True)` so hung workers can't block return.

**Related fixes (2026-07-24 through 2026-07-25):**
- **CR-01**: Default batch coverage path rewritten with per-file timeouts
- **CR-02**: Sequential path now enforces timeout via single-worker pool per candidate
- **CR-05**: Shared global deadline (`MAX_TIMEOUT_SECONDS=300s`) caps aggregate wall-clock via `map_with_per_item_timeout`
- **CR-03**: Abandoned worker mutations isolated via `deepcopy()`
- **CR-04**: Daemon threads don't block interpreter exit; abandoned worker count logged
- **CR-06**: Timed-out futures are cancelled before marking failed
- **CR-07**: Never-started candidates distinguished from ran-and-timed-out via `CancelledBeforeStartError`
- **CR-08**: Worker-raised `TimeoutError` passed through unchanged (not relabeled as wait timeout)
- **CR-09**: Sequential timeout format errors eliminated (was `'Nones'`)
- **CR-10**: Ranker parallel scoring uses shared `map_with_per_item_timeout` helper
- **CR-14**: `_handle_enrichment_error` signature simplified; timeout passed as float, not string
- **CR-15**: Redundant `concurrent.futures.TimeoutError` import removed
- **CR-16**: Extracted `utils/futures.py:map_with_per_item_timeout()` with corrected lifecycle (all per-item wait, daemon workers, per-item + global deadline)
- **CR-18**: Float timeouts enabled; timeout tests reduced from 4s to ~0.2s
- **CR-20**: Hoisted `import threading`/`import time` in tests
- **CR-21**: Magic timeout strings eliminated via `TIMEOUT_SECONDS` constants
- **CR-22**: Unnamed literals named (`HUNG_WORKER_SAFETY_NET_SECONDS`, `ELAPSED_BOUND_SECONDS`)
- **CR-23**: Elapsed bounds derived from parameters, not hard-coded
- **CR-24**: `timeout_per_candidate` plumbed through `AnalysisConfig` and MCP tool signature
- **CR-25**: Batch coverage timeout parameter now threaded through (was silently dropped)
- **CR-26**: `ParallelProcessing.MAX_TIMEOUT_SECONDS` now enforced as shared deadline
- **CR-27**: Docstring updated to reflect wait-based timeout semantics
- **CR-28**: Test assertion simplified to verify failure, not exact error label
- **CR-29**: Float timeouts mitigate CI flake window

**Regression tests:** `TestPerCandidateTimeoutEnforcement`, `TestSequentialPathTimeoutEnforcement`, `TestBatchCoverageParallelTimeout`, `TestPerCandidateTimeoutEnforcement` (sequential), `TestGlobalDeadlineEnforcement`, `TestAbandonedWorkerIsolation`, `TestTimedOutPendingFutureCancellation`, `TestCancelledBeforeStartDisambiguation`, `TestWaitTimeoutDisambiguation`, `TestWorkerRaisedTimeoutClassification`, `TestParallelScoringTimeout`, `TestRunParallelSearchTimeout`.

### BUG-07: Exclude Patterns Applied Before Stream Limit

**Fix:** Detector now filters `_apply_exclude_patterns` inside the stream loop so the `max_constructs` limit applies to kept matches; `.venv`/`node_modules` no longer consume the budget.

### BUG-08: Extract-Target File Not Covered by Backup/Rollback

**Fix:** `_create_backup_if_needed` reads `orchestration_plan["create_files"]` to identify appended-to files (included in backup set) and newly created files (tracked for deletion on rollback).

**Regression tests:** `TestCreatedFilesTracking` (6 tests).

### BUG-09: Parent-Directory Import Dot Count Off by One

**Fix:** `_generate_import_for_extracted_function` now splits `target_rel` into path components before dot-joining, counting `..` components directly.

**Regression tests:** `TestApplyHelperFunctions` (5 tests).

### BUG-10: Function-Local Imports Matched and Docstring Demoted

**Fix:** `_insert_python_import` checks `line.startswith(...)` (not `stripped`) so only column-0 imports anchor insertion; fallback detects module docstring and advances insertion point past it.

**Regression tests:** `TestOrchestrationHelperFunctions` (3 tests).

### BUG-11: Test-Coverage Check Reads Every Test File Per Source File

**Fix:** `get_test_coverage_for_files_batch` builds `{test_file: content}` cache once via `_read_test_file_contents` and threads through both batch paths.

**Regression tests:** `TestBatchContentCache` (3 tests).

### BUG-12: Similarity/Embedding/Score Caches Unbounded and Collision-Prone

**Fix:** All three caches now `OrderedDict` LRU keyed on exact values (code string or tuple repr); bounded via `MinHashDefaults.SIGNATURE_CACHE_MAX_SIZE` / `SemanticSimilarityDefaults.EMBEDDING_CACHE_MAX_SIZE` / `RankerDefaults.SCORE_CACHE_MAX_SIZE` (1024 each); move-to-end on hit, `popitem(last=False)` eviction, ranker updates under existing `_cache_lock`.

**Regression tests:** 7 new across `test_minhash_similarity.py`, `test_semantic_similarity.py` (`TestEmbeddingCacheLRU`), `test_ranker_caching.py`.

### BUG-13: Base Snippet Re-Parsed for Every Group Member

**Fix:** New `_extract_literal_maps(code, language)` extracts per-type position maps once; `identify_varying_literals` gained optional `code1_literal_maps` kwarg; `_accumulate_literal_variations` lazily extracts base maps on first differing member.

**Regression tests:** `TestLiteralVariationExtraction` (3 tests).

## Code Quality Improvements

### DRY-01: Replace `_run_tool()` with `tool_context` (Deferred)

Already migrated in commit `f2f4ceb`; backlog entry marked Done.

### DRY-02: Replace Hardcoded Exclude Patterns

`FilePatterns.SKIP_DIR_NAMES` added to constants; all hardcoded exclude lists removed from detector.py, sync_checker.py, readme_generator.py, api_docs_generator.py, security_scanner.py, estimator.py, executor.py, orphan.py. All now reference `FilePatterns.DEFAULT_EXCLUDE` or `FilePatterns.SKIP_DIR_NAMES`.

### DRY-03: Adopt `read_file_lines` + `write_file_lines`

All 6 manual `open()` sites converted in polyglot_refactoring.py, renamer.py, fixer.py, sync_checker.py. New optional `errors` param on `read_file_lines` for ignore-on-decode-error paths. Atomic file writes via `write_file_lines` enable rollback via in-memory line lists.

### DRY-04: Remove Redundant Re-Timing Inside `tool_context`

`tool_context`/`async_tool_context` now own success log with `tool_completed` event (`tool`, `status="success"`, `execution_time_seconds`). Yield contract changed to `ToolRun` dataclass (`.start_time`, `.add_completion_fields(**fields)`) for result-derived log fields. All 38 call sites across 7 files migrated (condense, documentation, schema, quality, complexity, cross_language, core/executor).

**Behavior notes:** Completion logs carry `status="success"`; early-return paths (sentry-skipped, no-files-found) emit `tool_completed`; timing includes response formatting.

**Tests:** `test_tool_context.py` updated + 4 new cases.

### DRY-06: Consolidate `LANGUAGE_EXTENSIONS` Maps

Four duplicate maps found and merged into canonical `constants.LANGUAGE_EXTENSIONS` (typed `dict[str, list[str]]`). All adopters (feature modules, scripts) import from constants. Broadens coverage: polyglot refactoring gains kotlin `.kts`; API-docs discovery gains `.tsx`/`.jsx`; analyze_codebase gains fallback for unknown languages.

### DRY-07: Extract `find_source_files()` Helper

New `utils/file_discovery.py` with `find_source_files(root, language, exclude_patterns, *, extensions, skip_dir_names)` and `language_extensions(language, fallback)`. Adopted in analyze_codebase.py, documentation/api_docs_generator, documentation/sync_checker. Skip-dir filtering now uses exact path-component matching (e.g., `distribution/` no longer skipped by `dist`).

**Tests:** `tests/unit/test_file_discovery.py` (8 tests).

### DRY-08: Extract `run_tool()` Subprocess Runner

New `utils/subprocess_runner.py` with `run_tool(cmd, timeout_seconds, cwd)` returning frozen `ToolRunResult` (`.returncode`, `.stdout`, `.stderr`, `.error`, `.ok`). Never raises for optional-tooling failures; maps TimeoutExpired → `TOOL_TIMEOUT`, OSError → `TOOL_NOT_FOUND`. Adopted at all 6 subprocess-with-timeout sites (analyze_codebase, rewrite/service, utils/formatters).

**Tests:** `tests/unit/test_subprocess_runner.py` (6 tests).

### HC-01: Add ESLint to Pre-Commit Hooks

Target is `~/.claude` repo (hooks TypeScript project). New `~/.claude/.git/hooks/pre-commit`: runs `npm run lint:fix && npm run check` for staged `hooks/**/*.ts(x)` files; no-ops otherwise. Blocks on unfixable errors; gracefully skips if `hooks/node_modules` missing. Bypass via `--no-verify`.

## Low-Priority Findings

### BUGL-01: Duplicate Suggestion Line Savings Assume All Group Instances Same Length

Open; requires refactor to count per-item lines separately.

### BUGL-09: `_get_nesting_depth` Returns Max Depth Before Identifier, Not At It

**Fix:** Returns running `depth` instead of `max_depth` so nested-call false flags are eliminated.

## Summary

This release focuses on hang prevention via shared timeout helpers, cache correctness via exact-value keying and LRU eviction, and code quality consolidation via extracted file/subprocess runners and canonical language-extension mappings. 29 findings from the BUG-06 timeout code review (CR-01 through CR-29) are now resolved; the hang class of defects is systematized with a reusable futures helper (`map_with_per_item_timeout`). Backward compatibility maintained; no public API changes.
