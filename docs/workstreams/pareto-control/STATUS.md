# STATUS — Workstream E, target-free Pareto / preference control

**Status:** `DESIGN_ONLY`. The Stage 0 census is **RUNNING LOCALLY and has not
yet written its artifact**; `diagnostics/pareto_tradeoff_census.json` does not
exist at this commit, and the manifest records it as `null` rather than
fabricating a hash.

**Branch / commit:** `codex/compose-pareto-control`, based on `a0e680d`.

**Held-out opened:** **NO.** `reserve_source_keys` was never read. Every
measurement uses `training_source_keys`. **No Modal run has been launched.**

## What is running

`scripts/pareto_tradeoff_census.py --sources 60 --reach-sources 20`, locally,
CPU only. Completed phases, from the run log:

| phase | result |
|---|---|
| held-in pool | 96,094 sources; 17,907 eligible in the [18, 38] heavy-atom band |
| local kernel built | 94.5 s |
| **I-A** real successor fiber | **120 decision states, mean fiber width 586**, depths 0 and 1 |
| **I-B** one-cut MMP pairs | **81,500 pairs, 5,592 eligible decision states** |
| properties scored | ~70k candidate molecules |
| frozen similarity normalizer | centre 0.6000, IQR 0.2503 (held-in recipe) |
| frozen utopia `z*` (p99) | P +2.7315 · D +0.5554 · S +1.0653 |
| frozen reference `r` (p5) | P −1.2689 · D −1.3547 · S −1.2144 |
| **C5 reach rollouts** | **IN PROGRESS** — 20 sources x 3 objectives x 6 edits |

The reach rollouts are the remaining phase and are running well over the
original estimate: a potency-greedy trajectory climbs toward larger molecules
whose fibers are more expensive to enumerate than the depth-0/1 states used to
project the cost.

**When it finishes**, two commands complete the lane — neither hand-types a
number:

```bash
python3 scripts/pareto_render_status.py        # regenerates THIS file
python3 scripts/pareto_write_handoff_manifest.py
```

## Pilot verdict — `SMOKE_HELD_IN`, n = 3 sources. NOT the deliverable census.

An identical code path on 3 sources / 6 decision states, kept only because it is
what validated the pipeline. **Treat the direction as indicative and the numbers
as provisional; the 60-source run supersedes it.**

| pair | Spearman rho | verdict | failed gates |
|---|---:|---|---|
| `potency_vs_developability` **(adopted)** | −0.252 | **PASS** | — |
| `potency_vs_source_similarity` | −0.119 | FAIL | G2, G3, G4 |
| `developability_vs_source_similarity` | −0.090 | FAIL | G2, G4 |

Pair 1 on the pilot: tradeoff-move fraction **0.503**; every state offered both
a potency-up/developability-down candidate and its converse (**1.000 / 1.000**);
mean **2.83 of 5** distinct Chebyshev selections with **0.000** unanimous
states; mean front size 5.8, and 5.0 distinct front points selected across a
101-point weight grid.

### The one finding unlikely to move with more sources

**Source similarity is INERT under this executor.** A similarity-maximizing
6-edit greedy rollout moved `z_S` by a median of **exactly 0.000**: at every
step there exists a legal successor whose ECFP4 fingerprint is *identical* to
the source. "Stay similar to `x_0`" therefore costs nothing and exerts no
tradeoff pressure, so it cannot serve as a Pareto axis. That is a structural
property of the operator inventory, not a small-sample artifact, and it is why
both fallback pairs fail G4.

Note that the *withdrawn* G4 statistic would have called similarity saturated
for a completely different and bogus reason — see `DECISION_LOG.md` D-007.

## Next action

Let the census finish, regenerate this file and the manifest, then **authorize
the 12-source held-in smoke** in `modal_apps/pareto_control_app.py`. The cohort
is already frozen at `diagnostics/pareto_control_cohort.json`
(sha256 `adea8e5510852d69`), disjoint from the census sources. Costed in
`HANDOFF.md`: ~700 kernel calls per source, ~1.4 h wall and ~$18 at `cpu=8.0`.

## Tests

76 lane tests pass: 36 on the arms, 23 on the instrument gate, 8 on the
analysis, 9 pinning the config to the code. `pytest tests/ --collect-only`
reports 4,030 tests with no import errors.
