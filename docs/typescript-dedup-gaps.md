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

## Not established

The investigation was stopped before a brute-force pairwise comparison of the
312 real functions completed, so **the actual duplication level in
`~/.claude/hooks` is still unknown.** Nothing here should be read as "the hooks
codebase has no duplication" — only that the current tooling cannot tell us
either way.
