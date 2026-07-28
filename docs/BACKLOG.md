# Backlog

**See [docs/changelog/2026-03-08-complexity-refactoring-queue.md](changelog/2026-03-08-complexity-refactoring-queue.md) for historical complexity refactoring metrics (434 → 0 offenders, 2026-03-08 refresh completed 2026-03-09).**








## TypeScript Deduplication Gaps (2026-07-28)

Full evidence and reproduction steps: [docs/typescript-dedup-gaps.md](typescript-dedup-gaps.md). Verified against `~/.claude/hooks` (60 `.ts` files, 313 function declarations) — the TypeScript path reports 0 duplicate groups for tooling reasons, not corpus reasons.

- [ ] **TSD-01** (P1) Structure-hash bucketing destroys recall — `detector.py:390-410` buckets constructs before scoring; on a 245-function corpus it produced 244 buckets / 243 singletons, so only **2** functions were ever compared to anything and at most 1 group could ever be returned. Replace with an LSH-banded key over the existing MinHash signatures (`similarity.py`) or make bucketing opt-in. Needs a recall regression test — current behavior passes any test that only checks for false positives.
- [ ] **TSD-02** (P1) TypeScript function pattern matches ~1% of functions — `detector.py:221` `function $NAME($$$) { $$$ }` matched 3 functions vs 312 for `kind: function_declaration` on the same corpus; misses typed params and return-type annotations. Drive TS construct discovery from `kind:`-based rules.
- [ ] **TSD-03** (P1) `function_definition` resolves to `const $NAME = $$$` for JS/TS — `detector.py:219`, hardcoded by `tools.py:53` and `analysis_orchestrator.py:142`. Matches all 1431 consts, hits the 1000 cap (`detector.py:283-285`) on one-liners, then `min_lines` discards them; real function declarations and methods are never scanned via the public tools. Reported `total_constructs: 1000` is the cap, not a count.
- [ ] **TSD-04** (P2) `method_definition` pattern is unparseable and raises — `detector.py:222` `$NAME($$$) { $$$ }` fails ast-grep with exit 8 ("Multiple AST nodes are detected"), surfacing as `AstGrepExecutionError` rather than a degraded result. Class methods are unreachable for TypeScript. Needs `kind: method_definition`.
- [ ] **TSD-05** (P3) Detector logs `language=python` when constructed for TypeScript — `analyze_deduplication_candidates(language='typescript')` emits `detector_initialized language=python`. Confirm no Python detector is being used for TS scoring; misleading during language-specific debugging.
- [ ] **TSD-06** (P2) Add a TypeScript recall fixture and per-`(language, construct_type)` pattern validation test — every defect above survives the current suite because nothing asserts non-zero TS recall, and TSD-02/TSD-04 are both patterns that a trivial parse-check would have caught.

### Follow-ups from the TSD-01 fix (verified 2026-07-28, post-merge)

TSD-01/02/03/04 are fixed on `main` (construct discovery now finds 315 real functions instead of 1000 truncated consts). These two are residual, found while investigating a known-duplicate pair the fixed pipeline still misses. Evidence: [docs/typescript-dedup-gaps.md](typescript-dedup-gaps.md#follow-ups-verified-post-merge).

- [x] **TSD-07** (P1) **Fixed 2026-07-28** — `find_all_similar_pairs` / `_verify_candidates` take an optional `scorer` (defaulting to the MinHash estimate, so the standalone API is unchanged); `group_duplicates` injects `self.calculate_similarity`, restoring `similarity_mode` to the grouping path. Verified: both `trace-context.ts` pairs now group in hybrid mode, `minhash` mode is unchanged, and the regression test fails when the injection is reverted. **Perf impact is negligible** — LSH narrows 235 constructs (27,495 possible pairs) to 5 candidates, so hybrid scores 5 pairs, not O(n²); wall time unchanged at ~0.4s. Original report: MinHash-only verification drops true duplicates, bypassing the hybrid scorer — `similarity.py:426` `_verify_candidates` makes the final keep/drop decision with `estimate_similarity()` (raw MinHash) against the *full* `min_similarity`, while candidate generation correctly widens the threshold via `lsh_recall_margin` (0.80 → 0.60). The margin is self-defeating: it widens the net, then re-applies the strict threshold using the very metric whose imprecision the margin exists to absorb. Measured on `trace-context.ts` `saveSessionContext`/`savePromptContext`: `minhash=0.7734` (dropped) vs `hybrid=0.8715` (would be kept); the sibling load pair scored `minhash=0.8125` / `hybrid=0.8808` and survived only by luck of the estimate. **Regression scope:** before TSD-01, grouping decided via `_collect_similar_items` → `calculate_similarity` (`detector.py:462`, the hybrid AST/CodeBERT scorer); `group_duplicates` now delegates wholly to `find_all_similar_pairs`, so `similarity_mode="hybrid"` is bypassed for grouping and the 3-stage pipeline is dead code on that path. Fix: have verification use the caller's real scorer (inject it, or verify in the detector after candidate retrieval). Confirmed by patching verification to the hybrid scorer — both pairs then survive.
- [ ] **TSD-08** (P3) Partial LSH recall permanently suppresses the brute-force fallback — `similarity.py:_should_use_fallback` returns `False` as soon as `len(candidates) > 0`, so the all-pairs fallback only ever fires on a *completely* empty candidate set. A partially populated candidate set never self-corrects, even on corpora well under `max_fallback_items` (100) where brute force is cheap. Not the cause of the TSD-07 miss (candidates were correct there), but it removes the safety net that would otherwise mask LSH recall gaps on small inputs.

**Not a defect (documented so it is not re-investigated):** after TSD-07 is fixed, `~/.claude/hooks` still reports 0 groups — `_meets_min_savings` drops the surviving pair because 11 duplicated lines is under `DetectorDefaults.MIN_LINE_SAVINGS = 20`. That threshold is deliberate. Any recall fixture for TSD-07 therefore needs **≥20 duplicated lines**, or it cannot distinguish "found nothing" from "found it and the savings filter ate it."

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

