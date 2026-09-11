# Complete-edit chooser: rejected development recipe

The cut-context outcome model failed its predeclared advancement rule. Do not
deploy or scale this fixed recipe. This result does not change the best observed
Perindopril endpoint (0.6835298931) or establish competitive PMO performance.

## What was implemented

The proposal scorer learns complete-edit score differences from paid beneficial
and damaging edits. It uses source attachment context, removed and added material,
their interaction, and fixed molecular counts. Compilation failure has its own
diagnostic target, never a fabricated task reward. A 20% positive base mixture
preserves supplied proposal support. The frozen reference, executor, generic
channel, other macros, and benchmark gates are unchanged. No endpoint is committed
without compilation; this module is not yet wired into a live optimizer.

## Evidence

396 input records reconcile to 369 outcomes/attempts with all origins retained.
Split assignment preceded feature preparation: 240 training attempts (188 scored,
38 improving, 97 parent identities), 64 calibration attempts, 50 chronological
scored outcomes, and 15 explicit overlap exclusions. Failed attempts remain in the
completion dataset. Latest-run failures lacking operands in the extracted summary
are not silently assigned reward labels or included in its completion denominator.

Mean score change among one selected logged candidate per eligible parent:

| Selector | Calibration, 11 pools | Later audit, 12 pools |
|---|---:|---:|
| Uniform expectation | -0.287552 | -0.486545 |
| Positive-context heuristic | -0.289171 | -0.528470 |
| New complete-edit context model | -0.313547 | -0.347455 |
| Same-data endpoint-only model | -0.230141 | -0.350749 |
| Best available paid choice (diagnostic ceiling) | -0.155474 | -0.340079 |

These negative changes describe forced choices from logged pools, not a decrease
of the optimizer's retained archive champion. Only one pool in each panel contains
any improving candidate. The new model misses that calibration improvement; the
endpoint-only model selects it. Both select the later improvement. Later precision
is therefore 1/12, with coverage 1/1, not a robust generalization estimate. Seventeen
calibration and sixteen chronological singleton pools are reported separately.

The fixed 20%-base stochastic policy has calibration mean gain -0.284012 and later
mean gain -0.415334. Its calibration improvement probability (0.012863) is below
uniform (0.045455); the corresponding endpoint-only values are -0.247208 and
0.065238. The negative decision is not just an artifact of greedy selection.

## Interpretation and next decision

Measured: context fitting does not reliably beat a simpler endpoint model. On the
later logged pool, even perfect ranking could improve only one of twelve eligible
parents. Better allocation cannot manufacture missing candidate chemistry.

Decision: reject this context-kernel recipe and do not repeat the unchanged long
mixture. Retain the paid-data preparation and endpoint-only comparator. The next
candidate should use planned molecular endpoints to choose complete edits before
expensive compilation, retaining broad program and primitive exploration. That
is an untested next implementation, not a result claimed by this audit. Fresh
generation with new task labels requires its own bounded recipe.

## Cost, provenance and checks

Zero new oracle calls or remote jobs. Final preparation: 2.714706 seconds;
fit/evaluation: 1.941531 seconds (fit 1.517974; scoring/reporting 0.402696).
The reporting extension added the logged-pool ceiling and provenance fields;
the recipe, rows, split and ranking results were unchanged. Historical regime:
249455 prescreen plus 2217 development physical calls, not no-prescreen PMO.

`report.json` binds raw input hashes, implementation hashes, the exact split,
software, CPU/float64 configuration, timestamps and per-parent selections.
Prepared SHA-256: `be1cee7bc6ed8570ea1bad7cfc29ea27cc93a1e0dc8ae9a964df95c86a3a5f3c`.
Model content identity: `8d29fa20411e8393e79b48308cde3599e5368dc899237879c9ef0d08c8a9a5ea`.
Inherited bank provenance limitations remain. This is exposed retrospective
development, not unbiased off-policy evaluation, a sealed test, or a PMO AUC.

Twelve focused tests passed in 2.81 seconds, covering beneficial/damaging labels,
failure-label separation, source/product leakage exclusions, deduplication,
slot-permutation invariance, batch parity, sampling support and existing compiler
behavior. Ruff checks passed. The repository-wide suite was not run; the overall
competitive-controller milestone is not complete. No branch push or deployment.

Commands, from the research worktree with its qualified chemistry import path:

```sh
python tools/pmo_edit_chooser.py prepare --output diagnostics/pmo_edit_chooser
python tools/pmo_edit_chooser.py evaluate --output diagnostics/pmo_edit_chooser
python -m pytest -q tests/test_edit_chooser.py tests/test_edit_replay.py tests/test_donor_program.py
```
