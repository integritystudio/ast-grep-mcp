# Backlog

**See [docs/changelog/2026-03-08-complexity-refactoring-queue.md](changelog/2026-03-08-complexity-refactoring-queue.md) for historical complexity refactoring metrics (434 → 0 offenders, 2026-03-08 refresh completed 2026-03-09).**








## Low-Priority Findings (dedup module review, 2026-07-21)

- [ ] **BUGL-01** (P3) Suggestion line savings assume all group instances have `group[0]`'s length — `detector.py:627-634` computes `lines * len(group)`; groups hold *similar* code of differing lengths, so `total_duplicated_lines`/`potential_line_savings` are wrong whenever lengths differ. Nearby `_meets_min_savings` (`detector.py:536-537`) counts per-item correctly — a group can pass the filter yet report a different savings number.
- [ ] **BUGL-03** (P3) Diff summary miscounts multi-line replacements — `diff.py:42-53` pairs a `-` line with a `+` only when directly adjacent, but unified diffs emit all `-` then all `+` per hunk, so a 2-line replacement counts as deletion+modification+addition and `old`/`new` lines are mis-paired in the `changes` list.
- [ ] **BUGL-08** (P4) Impact analyzer path/location parsing latent bugs — `impact.py:259-263` filters exclusions on raw relative paths before absolutizing (`:267-269`), so absolute `exclude_files` never match and duplicates' own files inflate `breaking_change_risk`; `impact.py:165-171` `_parse_files_from_locations` only parses `"file:line"` strings while detector suggestions emit dicts (`detector.py:623`), so `files_in_group` is always empty. No production caller of `analyze_deduplication_impact` found in `src/` — latent at the module boundary.

## Deferred

- [ ] **DRY-09** (P4) Delete `out()` wrapper in `analyze_codebase.py:53-55` — thin passthrough to `console.log(str(...))`; ~40 callsites could call `console.log` directly. Low priority (churn vs. clarity). -- `analyze_codebase.py:53` (2026-04-23 analyze_codebase reuse review)
- [ ] **DF-01** (Low) Strategy pattern filter for deduplication — per `docs/duplicate-detector-misses.md` investigation. Only candidate (Group 5) would save ~18 lines with minor signature mismatch; over-engineering for marginal benefit. (deferred 2026-03-08)
- [ ] **CF-04** (P3) Config-aware search mode — complex feature for PM2/Zod/JSON-LD config patterns. Deferred from 2026-03-11 session as out of scope. -- `src/ast_grep_mcp/features/search/`
- [ ] **SR-01** (P2) Unit tests for security scanner — add pytest test file for `detect_security_issues_impl()` covering SQL injection, XSS, command injection, secrets, and crypto patterns. Currently no test coverage. -- `tests/unit/test_detect_security_issues.py` (new)
- [ ] **SR-02** (P3) Integration tests for Python security YAML rules — validate that 12 rules in `rules/python-security-high-priority.yaml` fire correctly on sample vulnerable code. Add test fixtures for each CWE. -- `tests/integration/test_python_security_rules.py` (new)
- [ ] **SR-03** (P3) Document Python security rule coverage — create guide explaining which CWEs are covered (798, 327, 489, 521, 532, 377, 295), known limitations (e.g., follow/precedes matching edge cases), and migration path for integrating rules into quality/rules.py template system. -- `docs/PYTHON-SECURITY-RULES-GUIDE.md` (new)
- [ ] **SR-04** (P4) Expand security pattern examples — add coverage for additional CWE types (330: Use of Insufficiently Random Values, 434: Unrestricted Upload, 611: XXE). Currently 30+ patterns focused on CWE-798/327/489/521. -- `src/ast_grep_mcp/features/search/pattern_examples.json`

## Completed Items (Migrated to Changelog)

- [x] **LM-01–LM-05** Library Migration Phase 1–2 → [docs/changelog/2026-04-19-library-migration-phase1-phase2.md](changelog/2026-04-19-library-migration-phase1-phase2.md)
- [x] **FG-01** Schema detection Liquid/Jekyll template fallback → [docs/changelog/2026-04-20-schema-liquid-jekyll-fallback.md](changelog/2026-04-20-schema-liquid-jekyll-fallback.md)
- [x] **MQ-01–MQ-02** Minor Code Quality Items → [docs/changelog/2026-03-12-minor-code-quality-items.md](changelog/2026-03-12-minor-code-quality-items.md)
- [x] **PA-01–PA-05** Pattern Analysis performance optimization → [docs/changelog/2026-04-20-pattern-analysis-performance-optimization.md](changelog/2026-04-20-pattern-analysis-performance-optimization.md)
- [x] **PA-06** Parallel candidate scoring → [docs/changelog/2026-04-20-pa06-parallel-scoring.md](changelog/2026-04-20-pa06-parallel-scoring.md)
- [x] **BUG-01, BUG-03–BUG-05, BUGL-02, BUGL-04–BUGL-07, BUG-14, DRY-05, FLAKY-01** Deduplication quality fixes → [docs/changelog/2026-07-21-deduplication-quality-fixes.md](changelog/2026-07-21-deduplication-quality-fixes.md)
- [x] **BUG-02, BUG-06–BUG-13, BUG-06-CR-01–CR-16, CR-18, CR-20–CR-29, BUGL-09, DRY-01–DRY-04, DRY-06–DRY-08, HC-01** Bug fixes and quality improvements → [docs/changelog/2026-07-26-bug-fixes-and-quality-improvements.md](changelog/2026-07-26-bug-fixes-and-quality-improvements.md)

