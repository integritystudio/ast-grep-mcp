# TypeScript Deduplication Gaps

**Investigated:** 2026-07-28
**Corpus:** `~/.claude/hooks` — 60 `.ts` files (excluding `dist/`, `node_modules/`), 313 `function` declarations, 35 arrow-function consts, 12 classes.

## Summary

Running `find_duplication` and `analyze_deduplication_candidates` against a real
60-file TypeScript codebase reports **0 duplicate groups**. That result is an
artifact of four independent defects, not a property of the corpus. Each defect
alone is sufficient to force a near-empty result; stacked, the TypeScript path
cannot report duplication at all.

Reproduction of the reported numbers:

```bash
LOG_LEVEL=error uv run python -c "
from ast_grep_mcp.features.deduplication.tools import find_duplication_tool
print(find_duplication_tool('/Users/alyshialedlie/.claude/hooks', 'typescript',
      min_similarity=0.8, min_lines=5,
      exclude_patterns=['**/dist/**','**/node_modules/**'])['summary'])"
# {'total_constructs': 1000, 'duplicate_groups': 0, ...}
```

## Findings

### TSD-01 (P1) — Structure-hash bucketing destroys recall

`detector.py:390-410` (`_create_hash_buckets` / `_calculate_structure_hash`)
buckets constructs before comparison so that only same-bucket pairs are scored.
On this corpus the hash is so discriminative it isolates nearly every function:

| metric | value |
| --- | --- |
| functions surviving `min_lines=5` | 245 |
| buckets produced | 244 |
| singleton buckets | 243 |
| functions ever compared to anything | **2** |

So 243 of 245 functions are compared against *nothing*, and the maximum number
of groups the pipeline can return is 1. The bucketing is described in-code as an
O(n²) optimization, but it is currently a correctness bug: it trades essentially
all recall for speed. This is the dominant defect — fixing the pattern bugs
below still yields 0 groups while this stands.

Suggested direction: bucket on a coarse, similarity-preserving key (LSH bands
over the existing MinHash signatures — the MinHash machinery is already present
in `similarity.py`), or make bucketing opt-in with a documented recall cost.
Whatever replaces it needs a recall regression test with a known-duplicate
fixture; the current behavior would pass any test that only asserts
"no false positives".

### TSD-02 (P1) — TypeScript function pattern matches ~1% of functions

`detector.py:221` maps `traditional_function` to `function $NAME($$$) { $$$ }`.
Verified directly against the ast-grep CLI on the same corpus:

```
function $NAME($$$) { $$$ }   ->   3 matches
kind: function_declaration    -> 312 matches
```

The pattern misses typed parameters and return-type annotations — i.e. nearly
all real TypeScript. `class $NAME` by contrast matched 12/12 classes correctly,
so the breakage is specific to the function patterns.

Fix: drive TypeScript construct discovery from `kind:`-based YAML rules
(`function_declaration`, `method_definition`, `arrow_function`,
`class_declaration`) rather than source patterns. The project already documents
`kind`-based matching as the approach for this class of problem (see CLAUDE.md
Notes).

### TSD-03 (P1) — `function_definition` for JS/TS resolves to `const $NAME = $$$`

`detector.py:219` maps the *default* construct type to `const $NAME = $$$`, and
both `find_duplication_tool` (`tools.py:53`) and the orchestrator
(`analysis_orchestrator.py:142`) hardcode `construct_type="function_definition"`.
Consequences on this corpus:

- The pattern matches every `const` declaration — 1431 hits, the vast majority
  one-line non-functions.
- The 1000-construct cap (`detector.py:283-285`) truncates the list, so the
  budget is consumed by trivial consts.
- `min_lines` then discards nearly all survivors, leaving effectively nothing.
- Actual `function` declarations and class methods are **never scanned** through
  the public tool entry points at all.

The misleading `total_constructs: 1000` in the summary is this cap, not a real
construct count — it reads as "plenty analyzed" when almost nothing relevant was.

### TSD-04 (P2) — `method_definition` pattern is invalid and raises

`detector.py:222` maps `method_definition` to `$NAME($$$) { $$$ }`, which is not
parseable as standalone TypeScript. ast-grep exits 8 and the call raises
`AstGrepExecutionError`:

```
Error: Cannot parse query as a valid pattern.
╰▻ Multiple AST nodes are detected. Please check the pattern source `$NAME($$$) { $$$ }`.
```

Class methods are therefore unreachable for TypeScript by any construct type.
This needs `kind: method_definition` (same fix as TSD-02). Note the tool
surfaces this as a hard exception rather than a degraded result, so any caller
iterating construct types crashes.

### TSD-05 (P3) — Detector logs `language=python` when constructed for TypeScript

`analyze_deduplication_candidates` with `language='typescript'` emits
`detector_initialized language=python similarity_mode=hybrid` before the
TypeScript search runs. Cosmetic if it is a second detector instance for an
unrelated purpose, but it makes the logs actively misleading while debugging
language-specific behavior — worth confirming the orchestrator is not
constructing a Python detector and using it for scoring.

## Cross-cutting recommendations

1. **Add a TypeScript recall fixture.** A small fixture with N known-duplicate
   functions, asserting the tool finds them. Every defect above survives the
   current suite because nothing asserts non-zero recall on TypeScript.
2. **Do not report cap values as counts.** `total_constructs` should distinguish
   "found" from "analyzed after truncation", and truncation should be visible in
   the response, not just the logs (compare the `files_scanned` fix in 4346b26).
3. **Validate construct patterns at startup or in tests.** TSD-04 is a pattern
   that cannot ever parse; a test that runs each `(language, construct_type)`
   pair against a trivial fixture would have caught TSD-02 and TSD-04 both.

## Follow-ups (verified post-merge)

TSD-01 through TSD-04 are fixed on `main`. Construct discovery now returns 315
real functions instead of 1000 truncated one-line consts. Two residual defects
remain, found by brute-forcing all 29,403 pairs to establish ground truth.

### Ground truth for `~/.claude/hooks`

The corpus genuinely has very little duplication — **exactly 2 pairs score
≥0.80**, both in `lib/trace-context.ts`:

| pair | hybrid | minhash |
| --- | --- | --- |
| `saveSessionContext` ↔ `savePromptContext` (9 lines each) | 0.8715 | 0.7734 |
| `loadSessionContext` ↔ `loadPromptContext` (11 lines each) | 0.8808 | 0.8125 |

Both were real duplication. The save pair differed only in the type name and the
path helper — and `PromptTraceContext` was a `@deprecated` alias
(`type PromptTraceContext = TraceContext`), i.e. the *same type*, so the two
functions differed by exactly one token: `getSessionTracePath` vs
`getContextPath`. The load pair added a differing TTL constant.

Both have since been refactored away by hand in `~/.claude`
(commit `2950f7a2`, "consolidate duplicated helpers, constants, and metric
instruments"), which routed all four through `writeJson`/`loadJson` helpers in
`./file-utils.js` — the same fix the tool should have recommended. The finding
was correct; the tool simply did not report it.

### TSD-07 (P1) — MinHash-only verification drops true duplicates

`find_all_similar_pairs` (`similarity.py:322`) uses two different metrics for
two stages:

1. **Candidate generation is correct.** `lsh_recall_margin` widens the LSH
   threshold 0.80 → 0.60, and the save pair *is* in the candidate set (verified
   directly against `_find_lsh_candidates`).
2. **Verification discards it.** `_verify_candidates` (`similarity.py:426`)
   makes the final keep/drop call with `estimate_similarity()` — the raw MinHash
   estimate — against the full `min_similarity`.

So the margin is self-defeating: it widens the net, then re-applies the strict
threshold using the very metric whose imprecision the margin exists to absorb.
MinHash under-estimates relative to the hybrid scorer (0.7734 vs 0.8715), and
the pair is lost. Patching verification to use the hybrid scorer recovers both
pairs.

**Regression scope.** Before TSD-01, grouping decided via
`_collect_similar_items` → `calculate_similarity` (`detector.py:462`) — the
hybrid AST/CodeBERT scorer. `group_duplicates` now delegates wholly to
`find_all_similar_pairs`, which is MinHash-only, so `similarity_mode="hybrid"`
is bypassed for grouping and the 3-stage pipeline is dead code on that path.
TSD-01 traded a catastrophic recall bug for a subtler one.

### TSD-08 (P3) — Partial recall suppresses the brute-force fallback

`_should_use_fallback` returns `False` as soon as `len(candidates) > 0`, so the
all-pairs fallback fires only on a *completely* empty candidate set. A partially
populated set never self-corrects, even on corpora far below
`max_fallback_items` (100) where brute force costs ~5s. Not the cause of the
TSD-07 miss, but it removes the safety net that would otherwise mask LSH recall
gaps on small inputs.

### Not a defect

Even with TSD-07 fixed, this corpus reports 0 groups: `_meets_min_savings`
drops the surviving pair because 11 duplicated lines is under
`DetectorDefaults.MIN_LINE_SAVINGS = 20`. That threshold is deliberate. A recall
fixture for TSD-07 therefore needs **≥20 duplicated lines**, or it cannot
distinguish "found nothing" from "found it and the savings filter ate it."

### Method note

An earlier revision of this investigation reported the pipeline finding 1 of the
2 pairs partly for the wrong reason: the scratch harness omitted `range` from
its match dicts, so `_get_item_key` returned `":0"` for every function in the
file and `_merge_overlapping_groups` collapsed them. Re-run with complete match
dicts, the shipped code finds 1 group for the reason given in TSD-07. Any
harness that constructs match dicts by hand must include `file` **and** `range`.
