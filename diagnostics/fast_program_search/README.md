# Fast program-search development evidence

No new task-oracle evaluations or model training. See
[`docs/FAST_PROGRAM_SEARCH.md`](../../docs/FAST_PROGRAM_SEARCH.md) for the full
implementation, constraints, observations, negative results and next decision.

- `fixed_1`: initial explicit program-only mode, four real warm-state batches.
- `instrumented_1`: cProfile diagnostic, excluded from timing comparisons.
- `reuse_1`: bounded deterministic-work reuse, same proposal randomizations.
- `bindings_1`: reuse plus exact bit-mask attachment matching.
- `untried_1`: separate conditional-mutation-without-replacement proposal assay.
- `throughput_comparison.json`: exact candidate/attempt equivalence for all four
  `fixed_1`/`bindings_1` pools; 68.705 to 55.215 summed proposal seconds.
- `neighborhood_comparison.json`: untried changes the proposal; 245 versus 258
  attempts for four twelve-candidate pools, but 62 versus 58 duplicate attempts.

Each run retains a recipe, source/input hashes, result, exact batches and snapshots.
Source is the recorded local dirty-worktree development version, not a clean
benchmark release. No comparison with remote historical timing is treated as a
matched speedup. Endpoints are unscored and new relative to each specified warm
archive, not necessarily globally novel or distinct across randomizations.

These artifacts do not replace previous paid T4/PMO results. In particular, no
unresolved PMO query was retried or credited as free.
