# Workstream D — status

**Status:** `DESIGN_ONLY`, plus one `SMOKE_HELD_IN` instrument check. **HOLDING**
by instruction.
**Branch:** `codex/compose-baseline-qualification` (base `04f1c46`).
**Held-out data opened:** **no.** No sealed panel, no confirmatory reserve, no
matched reserve was touched.
**Running now:** nothing. **No Modal job was launched; nothing external was
installed.** The harness stress test ran locally in under a CPU-second.
**Last completed gate:** three-counter oracle accounting frozen, and the
**ORACLE-ACCOUNTING HARNESS STRESS TEST — PASS**
(`diagnostics/baselines/oracle_accounting_harness_stress_test.json`).

## That stress test is NOT a GraphGA result

**No GraphGA run has happened.** Upstream GB-GA was never vendored and nothing
external was installed. The stress test drives the shared accountant with a
GA-*shaped*, adversarial-by-construction candidate stream (BRICS recombination,
survivor rescores, alternate SMILES spellings, malformed strings). It says
nothing whatever about GraphGA's behaviour.

Before any claim-bearing GraphGA comparison we still need: the actual upstream
implementation vendored; the production RDKit pin **2024.3.5**, or an explicitly
isolated environment whose canonicalization is reconciled against it; and the
real algorithm terminating against every counter. The stress test ran on rdkit
2025.09.6 and is stamped `pin_matches_production: false`, which keeps its
canonical keys out of any scientific result.

## Where the real implementations stand

| method | environment | adapter | evidence |
|---|---|---|---|
| **GraphGA** | native, RDKit-only | **BUILT AND SMOKED** — real upstream vendored byte-identical | `diagnostics/baselines/graph_ga_held_in_smoke.json`, PASS 9/9 |
| **MARS** | **BUILT AND RUN**, python 3.11 CPU venv | **BUILT AND SMOKED** | `diagnostics/baselines/mars_held_in_smoke.json`, PASS 4/4 |
| **REINVENT 4** | **FEASIBLE**, installs on python 3.14 CPU | designed | undeclared `scipy` is the only packaging bug |
| **GraphXForm** | **FEASIBLE** — `torch_scatter` compiled, transformer instantiates (31.5 M params), checkpoint 331 MiB verified | designed | expected blocker did not occur |
| **DDSBM** | **BLOCKED** — no LICENSE (legal, not technical), python 3.9 pin, `graph-tool` conda-only, `orca/` missing from the clone | none | days, needs Linux |
| **HN-GFN** | **BLOCKED** — zero dependency pins, botorch API drift | none | ~1 day of version archaeology |

No sweep was run. No Modal job was launched. Nothing was installed into the
project environment — probes used throwaway venvs under `/tmp`.

## Frozen this round: three-counter oracle accounting

All three always logged, never substituted:

| counter | definition | used for |
|---|---|---|
| `unique_valid_canonical_evaluations` | distinct valid canonical molecules evaluated | **only** comparison against published PMO numbers |
| `oracle_requests` | every scoring request incl. duplicates, rejects, invalids — **algorithmic demand, not CPU** | efficiency claims about search behaviour |
| `evaluator_calls` | expensive oracle executions after caching — **real work** | what the objective genuinely cost |

> **Conceptual invariant: caching may reduce evaluator work, but it cannot erase
> wasteful algorithmic requests.**

That is why a cache hit still increments `oracle_requests`. For a literal-compute
number, report wall and core time separately.

The stress test shows why this is not a formality. Same candidate stream, budget
120: **120 distinct molecules under `unique_valid_canonical_evaluations` (288
requests) versus 59 under `oracle_requests` (120 requests)** — a 2× misstatement
on a benign stream, and far more on MARS, which has no cache anywhere. Disabling
the cache on that identical stream left requests at 288 and unique at 120 while
moving `evaluator_calls` from 120 to 262.

Required per-method ratio: **`oracle_requests / unique_valid_canonical_evaluations`**.
Implemented in `src/compose_v4/experiments/oracle_accounting.py`.

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

## The number that vindicates the three-counter freeze

Measured on real upstream code, same objective, same accounting:

| method | `oracle_requests` | `unique_valid_canonical_evaluations` | demand ratio |
|---|---:|---:|---:|
| GraphGA | 120 | 92 | **1.30** |
| MARS | 280 | 75 | **3.73** |

MARS has no cache anywhere and rescores its current molecule whenever a proposal
is rejected. Reporting its 75 beside GraphGA's 92 as if they were the same
quantity would understate its actual demand by 3.7×. This is exactly the
substitution the freeze exists to prevent, now measured rather than argued.

## Two findings that change what the paper may claim

1. **REINVENT 4's stage boundary is far softer than we recorded.** Verified in
   source: the Agent network is created once *outside* the stage loop; the Adam
   optimizer is constructed **once** and the same object is placed in every work
   package, so `exp_avg`, `exp_avg_sq` and the step counter **carry across the
   objective switch unbroken**; there is **no LR scheduler in the RL path at
   all**; and the inception replay buffer is never cleared, carrying stage-N
   molecules with their **stale stage-N scores** into stage N+1. Only the
   diversity-filter memories purge. So a stage boundary changes the scoring
   function and the termination criterion and essentially nothing about the
   optimizer state. **"The baseline must retrain" is too weak a distinction.**
   What survives: COMPOSE changes goal at an arbitrary step of a realized
   *molecular* history and recomputes only the control law.
2. **HN-GFN is a LINEAR weighted sum, and Lane 4 froze Chebyshev.** Its opt-in
   `Tchebycheff` branch is not classical Chebyshev either — an augmented max-min
   with no ideal point and a hardcoded 0.1. Linear scalarization cannot recover
   concave front regions. This must be reported as a difference, not described
   as matching. See `FAIRNESS_MATRIX.md`.

## The one remaining bounded job

Implement **exactly one** native GraphXForm adapter on the frozen route —
official pretrained checkpoint → native objective-specific fine-tuning →
native beam/TASAR search → thin common evaluation adapter — run **one** held-in
smoke under all three counters, document applicability, and **stop before any
benchmark sweep**. The set is closed; do not add a baseline because this one is
involved.

The adapter canonicalises, evaluates, counts and connects the frozen oracle. It
does **not** choose actions, rewrite masks, disable TERMINATE, or alter the
proposal distribution.

## Next action

**None. Deliverables A–C are complete; stopping before expensive sweeps.** The lane's work is complete and the harness gate passed.
Resume only on an explicit instruction from the main workstream, and only after
Lane 1 resolves whether `R_theta` is being retrained.

The next bounded action, when authorised: **apply the one-line `rdkit.six` fix
and bring MARS up on python 3.11**, then smoke it exactly as GraphGA was smoked.
It is the last `MUST_RUN` method whose environment is not yet proven end to end.

Still outstanding before any claim-bearing comparison: **the production RDKit
pin 2024.3.5**. Both smokes ran on 2025.09.6 and are stamped
`pin_matches_production: false`, which keeps their canonical keys out of any
scientific result.

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
