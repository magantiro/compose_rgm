# Workstream D — status

**Status:** `DESIGN_ONLY`, plus one `SMOKE_HELD_IN` instrument check. **HOLDING**
by instruction.
**Branch:** `codex/compose-baseline-qualification` (base `04f1c46`).
**Held-out data opened:** **no.** No sealed panel, no confirmatory reserve, no
matched reserve was touched.
**Running now:** nothing. **No Modal job was launched; nothing external was
installed.** The accounting smoke ran locally in 0.22 s of CPU.
**Last completed gate:** dual oracle accounting frozen and its instrument check
**PASS** (`diagnostics/baselines/graph_ga_accounting_smoke.json`).

## HOLD

Per the main workstream: the accounting smoke passed, so **stop**. Do not
proceed to MARS, REINVENT, GraphXForm, DDSBM or HN-GFN. Lane 1 is investigating
whether `R_theta`'s iterated dynamics are pathological; if that forces a retrain,
every downstream comparison would have to be repeated, so no expensive adapter
should be built against a model that might be replaced.

## Frozen this round: dual oracle accounting

Both counters, always logged, never substituted:

| counter | definition | used for |
|---|---|---|
| `benchmark_native` | unique valid canonical molecules scored | **only** comparison against published PMO numbers |
| `raw_compute` | every invocation incl. duplicates, rejects, invalids, rescores | **all** efficiency claims |

The instrument check shows why this is not a formality: **the same candidate
stream under a budget of 120 bought 120 distinct molecules under
`benchmark_native` (288 invocations) and 59 under `raw_compute` (120
invocations)** — a 2× misstatement on a benign stream, and far more on MARS,
which has no cache anywhere. Implemented in
`src/compose_v4/experiments/oracle_accounting.py`.

## Naming rule that must not slip

The MARS arm is **"restart at `x_τ` under a new objective"** — a new optimization
launched from the current molecule. **Never** call it same-prefix retargeting or
continuation. The molecule is preserved; the proposal, imitation dataset and
temperature are not.

## The finding to read first

Three baselines natively do something the COMPOSE Claim-4 story treats as
distinctive. None of them does the full claim, but the paper must state the
**conjunction**, never the part.

1. **REINVENT 4 changes the objective mid-run, natively.** Staged/curriculum
   learning varies the scoring function in stages and carries the agent
   checkpoint across the switch. It re-targets a *policy*, not a realized
   molecule, and it re-adapts by further RL — but "no existing method changes
   objective mid-run" is now a sentence the paper cannot write.
2. **HN-GFN changes the goal without retraining**, for scalarization weights over
   a fixed objective set, via a hypernetwork over the prediction heads.
3. **GraphXForm applies structural action masking at every step**, pre-softmax.
   Real pathwise enforcement — but only over valence, atom type, atom count and
   bonding legality, never labeled subgraphs.

And one correction that goes **against** COMPOSE: **DDSBM does support effective
atom addition and deletion** through a dummy atom type inside a padded node
array. `docs/RELATED_WORK_MATRIX.md` marks it `✗` on `var-card` and
`birth/death`; on this evidence both should be `~`.

## Scoping that changes how the tables are read

"Future-aware control helps" is **not** a universal COMPOSE claim. Established on
hard exact-target recovery (greedy 26/65 → verified rollout 40/65, sealed);
**measured absent** on an easy target-free property goal (greedy 28/30 = verified
28/30, headroom 0, gate CLOSED). Therefore:

- the **C4a static-optimization table is a competitiveness sanity check**, not a
  claim-bearing test — PMO's DRD2/GSK3β/JNK3 tasks are exactly that easy regime;
- the regime a baseline would actually have to win is **C4b**, where **none of
  the six methods is applicable** — report `N/A`, never as a COMPOSE win;
- **C4c is judged on post-switch oracle cost and no-restart**, not on endpoint
  score.

Full scoping with artifacts: `FAIRNESS_CONTRACT.md` §0a.

## Verdicts

| method | verdict | why |
|---|---|---|
| MARS | `MUST_RUN` | closest iterative editor; source-conditionable with stock code; CC BY-NC license and a 2021 dependency stack are the risks |
| GraphXForm | `MUST_RUN` | exact source conditioning via `start_from_smiles`; constructive-only action space is a hard scope limit |
| GraphGA | `MUST_RUN` | PMO rank 2, MIT, RDKit-only, CPU-native, no training — the cheapest and cleanest comparator |
| REINVENT | `MUST_RUN` | PMO rank 1, and the only external arm that natively switches objective mid-run |
| HN-GFN | `CONDITIONAL` | gate: GPU authorisation **and** a working BoTorch pin (upstream issue #1 is unresolved) |
| DDSBM | `CONTEXT_ONLY` | **no LICENSE file at all**, no checkpoints, intermediates are not molecules, zero oracle calls at sampling |

## Costed smoke plan — nothing executed

| method | plan | projected CPU-core-hours | confidence |
|---|---|---:|---|
| GraphGA | 5 sources, pop 120, 10 generations | 0.1 | HIGH |
| REINVENT (A, PMO-wrapped) | 5 sources, ~1000 calls each | 0.3 | MEDIUM-HIGH |
| REINVENT (B, v4 staged learning) | 2–3 sources, 2 stages × 25 steps | 0.5–1.0 | MEDIUM |
| MARS | 5 sources, 1 chain each, 50 steps | 1.5 (incl. one-time ChEMBL vocab build) | LOW on env |
| GraphXForm | 5 sources, 3 epochs, beam 32 | 2–4 | MEDIUM |
| HN-GFN, DDSBM | not runnable CPU-only | — | — |
| **total** | | **~4.4–6.9** | |

Under USD 1 of compute at a nominal CPU rate. **The cost of this lane is
engineering time on dependency rot, not credits.**

## Next action

**None. HOLDING.** The lane's work is complete and the instrument gate passed.
Resume only on an explicit instruction from the main workstream, and only after
Lane 1 resolves whether `R_theta` is being retrained.

When it resumes, the first bounded action is: **vendor upstream GB-GA (MIT) and
run the real GraphGA adapter on 3–5 held-in sources**, 0.1 CPU-core-hours. The
accountant it plugs into is already built and tested.

## Manuscript-lane merge request

`docs/RELATED_WORK_MATRIX.md` was amended on this branch with lead approval:
DDSBM `var-card`/`birth/death` corrected `✗ → ~`; new rows for GraphXForm and
REINVENT 4; per-cell sentences for every new `~`; `loeffler2024reinvent4` added
to `paper_iclr_stochastic_rewriting/references.bib`. **The MARS `pathwise` cell
was deliberately left unchanged** — see verification-debt item 4 in that file.

## Open decisions this lane deliberately did not make

- **How to report HN-GFN's surrogate.** Its 1000-call budget is spent against a
  learned proxy, not the true oracle.
- **The intended reading of the matrix's `pathwise` column** (affordance vs
  released code). It decides MARS, MIMOSA, Graph GA and Kappa-style rewriting
  together, and a change there would flatter COMPOSE, so it is not ours to make.

## Resolved since the last handoff

- **The oracle-counting convention** — now frozen as dual accounting, above.
