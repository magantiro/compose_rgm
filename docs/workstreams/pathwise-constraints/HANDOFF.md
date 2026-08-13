# Workstream

- **Name:** Workstream C — Pathwise constraints
- **Claim ID:** Claim 4 (trajectory-level design / intervention)
- **Branch:** `codex/compose-pathwise-constraints`
- **Base commit:** `04f1c4661c0f4977d00849893550e02d52dc7c64`
- **HEAD commit:** see `handoff.json` (`head_commit`)
- **Working tree clean:** yes
- **Status:** `SMOKE_HELD_IN` — stage A complete, **G1 FAILED**, lane stopped
- **Held-out data opened:** **no**

# One-sentence scientific question

> Can COMPOSE require an exact labeled structural motif to be present at
> *every* committed molecular state, and does that constraint actually bind —
> i.e. does endpoint-only filtering return molecules that reached validity by
> passing through forbidden intermediates?

# Salvage lane — reversibility census: NO FAMILY PASSES

Families, thresholds, rank order and pass criteria were sealed in commit
`bf14d53` **before** the measuring script existed. Computed from the committed
stage-A shards: **42 unconstrained trajectories, 294 states, no new compute.**

| Family | violators | returned | **return rate** | events | V1 | V2 | V3 | Verdict |
|---|---:|---:|---:|---:|:--:|:--:|:--:|---|
| **A** undesired reactive group | 1/35 | 1 | 1.000 | **1** | ✓ | ✓ | ✗ | **FAIL** |
| **B** cLogP corridor `[2.369, 4.452]` | 7/21 | 6 | **0.857** | **6** | ✓ | ✓ | ✗ | **FAIL** |
| **C** heavy atoms `[18, 38]` | 0/42 | 0 | 0.000 | 0 | ✗ | ✗ | ✗ | **FAIL** |

V1 non-vacuous · V2 return rate ≥ 0.10 · V3 ≥ 20 endpoint-valid/path-invalid
events · V4 mask leaves room — **V4 was not evaluable**, since the shards store
trajectories but not enumerated successor sets.

**Verdict: no family passes. Pathwise constraints leave the main paper. The
ring-system negative goes to the appendix. No fourth predicate was searched.**

## The failure modes differ, and the difference is material

- **C is vacuous** — heavy-atom count never left the frozen band in 42
  trajectories. Dead for a scientific reason, like the ring system.
- **A and B are genuinely REVERSIBLE** — return rates 1.000 and 0.857, both far
  above the 0.10 floor. They fail on **event count alone**, which is a property
  of the 42-trajectory pool inherited from the 6-source smoke panel. At the
  observed rates, V3 needs ~**701** trajectories for A and ~**71** for B.

So the reversibility question the salvage lane was asked to settle got a
**positive answer for cLogP** — and an underpowered one. That is different from
the ring system, which failed on reversibility itself (0 of 19).

## Family B's excursions are not boundary noise

| Metric | Value |
|---|---:|
| excursions measured | 7 |
| min depth | 0.120 logP |
| **median depth** | **0.700 logP** |
| max depth | 0.954 logP |
| median as fraction of corridor width | **0.336** |
| excursions below the 0.1-unit noise threshold | **0 / 7** |

Example (source 1, unconstrained greedy): cLogP
3.01 → 2.40 → 2.57 → **1.67** → 2.27 → 2.62 → 2.47 — a real departure below the
corridor, and a return.

*Disclosure: this diagnostic was added after seeing B's return rate, as a guard
against an obvious artifact. It changed no threshold, no criterion and no
verdict.*

## What this lane will not do

B's near-miss is exactly the kind of result that invites relitigating a frozen
bar. The criteria were fixed before measurement so that the person who measured
cannot do that. **Recorded as FAIL.** Whether a larger pool is worth buying is
escalated below, not decided here.

---

# Ring-system result (CLOSED) — G1 FAILED

> **Motif destruction is absorbing under the frozen kernel at H = 6.** Of 42
> rollouts on the unconstrained support, **19 broke** the protected motif and
> **0 recovered** by the end. Endpoint validity therefore *implies* path
> validity on this panel, so endpoint-only filtering is sufficient **by
> dynamics, not by luck** — and the premise this lane was built on is
> empirically false in this regime.

Per the pre-committed anti-tuning rule, the motif rule, horizon and panel were
**not** changed. Stage B was not run.

# Claim this work can support

> The machinery works: an exact atom/bond-labeled subgraph requirement can be
> imposed on every committed state of a COMPOSE trajectory, at zero additional
> kernel cost, and it is a genuine restriction on the support (the mask removes
> a mean 22.6% of legal successors). **But on this held-in panel that
> restriction never changes what endpoint-only filtering returns**, because no
> trajectory that leaves the feasible set ever comes back.

A secondary and weaker claim does survive, reported separately below:
endpoint-only handling **failed to return anything** on 2/6 sources where the
masked arms succeeded 6/6.

# Claims this work cannot support

- **The lane's designed claim — that endpoint-only filtering returns molecules
  which traversed forbidden intermediates. This was measured and is FALSE
  here (0/19).**
- That constrained control beats unconstrained control on the objective.
- That the protected motif is a medicinally meaningful pharmacophore.
- That future-aware control beats greedy under the mask — guaranteed sign, and
  stage B was not run in any case.
- Anything about held-out generalisation.
- Any rate claim from the secondary finding: 6 sources, 2 events.

# Frozen inputs

| Object | Path / ID | SHA-256 / identity | Verified? |
|---|---|---|---|
| Process-V2 chemistry | `compose_v4.experiments.production_successor_kernel.canonical_successor_result`, `TIME_POINT=0.5`, `slots=48` | unchanged from `retarget_intervention_app.py` | yes, by reuse |
| `R_theta` checkpoint | volume `compose-v4-artifacts`, `/artifacts/editing_v2/r_theta_run/runs/run_v2_01` | loaded identically to `retarget_intervention_app.py`; **not retrained** | not re-hashed (volume not mounted locally) |
| split / panel | `diagnostics/pathwise_constraints_smoke_panel.json` | `panel_sha256 8b47a0e417dc1403863e2ec3b592cde50f942ebb1e5de81b560278e57e72fabd` | yes |
| held-in pool | `diagnostics/editing_v2_matched_validation_reserve_ids.json.gz` | `ba9270faf8bea1a67ad103c6da887efeef257a66fcb9ca1c3faf088f770994a1` | yes |
| sampling law | `R_theta` reference probabilities from the kernel; shortlist 3, rollouts 6 | in `configs/pathwise_constraints_protocol_v1.json` | yes |
| goal/oracle | `B = P AND D`; `artifacts/oracles/drd2_svm_v1` | npz `7c9224c12c423ba24cce7d15715409eb4dee26af40ab64fb621b3336079f0815` | yes |
| normalizers | `diagnostics/retarget_goal_language_normalizers.json` | `187d1ccc60b00c858f0f54f729a99ea91e06d33a913287886c80080889a76cc8` | yes |

# Protocol

- **Panel construction:** held-in `training_source_keys` only, shuffled under
  seed 20260813, first N eligible in scan order. Eligibility reads the source
  molecule only. The 30 retargeting-cohort sources are excluded by SMILES.
- **Arms:** stage A — `unconstrained_greedy`, `endpoint_only`,
  `pathwise_greedy`, `pathwise_stochastic`, `mask_only_sampling`; stage B —
  `unconstrained_verified`, `pathwise_verified`.
- **Primary metric:** `endpoint_valid_path_invalid` rate, on two denominators
  (per-source unconstrained trajectories, and every endpoint-valid rollout
  `endpoint_only` would have returned).
- **Secondary metrics:** mask `removed_fraction` along unconstrained states;
  paired terminal utility `pathwise_stochastic − endpoint_only`; completion
  rate; endpoint motif validity; `B`/`P`/`D` success; marginal kernel calls per
  arm in run order.
- **Independent statistical unit:** the source molecule. Rollouts from one
  source are repeated measures.
- **Allowed calibration:** eligibility constants from the source-only census
  yield (none were in fact adjusted); one horizon check if `H=6` proves too
  short.
- **Stop rules:** G0–G5 in `PROTOCOL.md`. G1 (vacuity) is the one that decides
  whether the workstream has a claim.
- **Forbidden adaptations:** changing the motif rule until an arm wins;
  retraining anything; adding kernel-derived criteria to eligibility; opening
  any held-out panel; reporting either definitional quantity as a finding.

# What was implemented

- `src/compose_v4/experiments/pathwise_constraints.py` — exact labeled-subgraph
  predicate, hand-written SMARTS writer, motif derivation rule
  (`largest_ring_system_v1`), trajectory audit, successor mask, eligibility.
- `src/compose_v4/experiments/pathwise_arms.py` — all seven arms as pure
  policies over an injected kernel, so they are locally testable.
- `modal_apps/pathwise_constraints_app.py` — CPU-only Modal app: runtime
  loading, shared enumeration cache, per-source durable shards, volume commit,
  server-side `drive()` fan-out, two instrument preconditions, a cost circuit
  breaker.
- `scripts/pathwise_constraints_census.py` — local held-in viability census.
- `scripts/pathwise_select_smoke_panel.py` — frozen outcome-independent panel.
- `scripts/analyse_pathwise_constraints.py` — shard aggregation and gate
  verdicts; refuses to headline the definitional quantities.
- `configs/pathwise_constraints_protocol_v1.json` — the frozen protocol.
- `tests/test_pathwise_constraint.py`, `tests/test_pathwise_arms.py`,
  `tests/test_pathwise_analysis.py` — 64 tests.

# Tests and smoke checks

| Test | Result | Artifact |
|---|---|---|
| `tests/test_pathwise_constraint.py` (30) | PASS | predicate: ring opening, dearomatisation, element swap, charge change, bond order, ring contraction all rejected; outside-motif substitution accepted |
| `tests/test_pathwise_arms.py` (27) | PASS | all seven arms; masked arms never violate across 12 seeds; endpoint-only returns a path-invalid trajectory |
| `tests/test_pathwise_analysis.py` (7) | PASS | analyser driven end-to-end on synthetic shards; **gate returns FAIL when the motif is never violated** |
| launcher has no chemistry imports (2 AST tests) | PASS | app imports under the modal CLI interpreter with RDKit absent |
| SMARTS writer fuzz, 4 000 held-in sources | 0 failures | scratch probe; 1 500-source version is in the suite |
| Held-in census, 20 000 sources | complete | `diagnostics/pathwise_constraints_census.json` |
| **Modal stage-A smoke, 6 sources** | **COMPLETE, 6/6 shards** | `diagnostics/pathwise_constraints_smoke.json`, app `ap-QJiLH8rlpGNGySTMLjOMPH`, verified `ephemeral (detached)` |
| motif re-derivation under image RDKit 2024.3.5 | PASS on 6/6 | no aromaticity drift; 0 voided shards |

# Results

**`SMOKE_HELD_IN`. Six held-in sources. Not claim-bearing for the paper; this
is a premise check and it came back negative.**

## G1 — the premise gate (preregistered threshold 0.10)

| Metric | Value | Denominator |
|---|---:|---|
| unconstrained trajectories that ever violate the motif | **2** | 6 sources |
| unconstrained trajectories **endpoint-valid AND path-invalid** | **0** | 6 sources |
| `endpoint_only` **selected** trajectory endpoint-valid and path-invalid | **0** | 6 sources |
| endpoint-valid rollouts `endpoint_only` would return that are path-invalid | **0** | **19 rollouts** |

**VERDICT: FAIL.** Both denominators give 0.00 against a 0.10 threshold.

### Mechanism — why it failed

| Metric | Value |
|---|---:|
| rollouts on the unconstrained support | 42 |
| broke the motif at some point | 19 |
| ended motif-valid | 23 |
| **broke AND recovered by the end** | **0** |
| recovery rate | **0.0** |

The two sets are disjoint: every rollout that broke the motif stayed broken.
The frozen kernel's operators can open a labeled ring system but effectively
cannot reconstruct one within 6 edits.

## G0, G2, G3, G4

| Gate | Verdict | Value | Denominator |
|---|---|---:|---|
| G0 mask integrity | **PASS** | 0 leaks — *bug detector, not a finding* | 6 shards |
| G2 room to act — `removed_fraction` | **PASS** | mean **0.226**, median 0.055, max 0.999 | 42 states on unconstrained paths |
| G2 — removed reference mass | **PASS** | mean 0.344, median 0.191 | 42 states |
| G2 — states with empty masked support | **PASS** | **0** | 42 |
| G3 feasible improvement (`pathwise_greedy`) | **PASS** | mean **+3.10** worst-margin over source | 6 |
| G4 distinguishable | **PASS** | landings differ **6/6**; 4.33 distinct landings/source | 6 |

G2 passing while G1 fails is the precise finding: the mask **is** a real
restriction on the support; it just never changes the returned molecule.

## Terminal objective by arm

| Arm | `b_success` | endpoint motif-valid | completed | mean improvement |
|---|---:|---:|---:|---:|
| `unconstrained_greedy` | 6/6 | 4/6 | 6/6 | +3.13 |
| `endpoint_only` | **4/6** | 6/6 | 4/6 (2 selection failures) | +2.35 |
| `pathwise_greedy` | 6/6 | 6/6 | 6/6 | +3.10 |
| `pathwise_stochastic` | 6/6 | 6/6 | 6/6 | +3.09 |
| `mask_only_sampling` | 0/6 | 6/6 | 6/6 | +0.64 |

`mask_only_sampling` moving (+0.64) but never succeeding (0/6) confirms the
mask alone does no optimisation work — feasibility and control are separated
as designed.

## Secondary finding — free sign, tiny n

`endpoint_only` returned **nothing** on 2/6 sources: all six of its rollouts
ended motif-invalid, so the filter had nothing to select. The masked arms
succeeded 6/6. Nothing in the construction forced this, so the sign is free.

But it is a **different claim** from the lane's: *absorption plus wasted
budget*, not *excursion and return*. With 6 sources and 2 events it is an
existence proof, not a rate.

## Price of the guarantee

Paired `pathwise_stochastic` − `endpoint_only` terminal worst margin, matched
budget and matched policy: mean **+0.012**, median **−0.002**, range
[−0.037, +0.088], **n = 4** (restricted to sources where `endpoint_only`
returned anything). **The guarantee is essentially free where endpoint-only
works at all.**

## Local source-only census (unchanged, context)

| Metric | Value | Denominator |
|---|---:|---|
| sources with a ring system | 98.32% | 19 665 / 20 000 |
| motif atom count, median | 9 | 19 665 |
| motif fraction of molecule, median | 0.310 | 19 665 |
| eligibility yield | 74.91% | 14 983 / 20 000 |
| already satisfies `B` | 0.94% | 187 / 20 000 |
| SMARTS self-match failures | 0 | 3 932 with motif |

# Gate verdicts

| Gate | PASS / FAIL / INCONCLUSIVE | Evidence |
|---|---|---|
| Motif exists and is non-trivial | **PASS** | 98.3% have a ring system; median 31% of the molecule with 18 free atoms |
| Objective has headroom | **PASS** | 0.9% already satisfy `B`; potency binding on all 6 panel sources |
| Eligibility is workable | **PASS** | 74.9% yield with constants fixed before the census |
| Predicate is non-vacuous *as a predicate* | **PASS** | 6 mutation classes rejected in unit tests |
| G0 mask integrity | **PASS** | 0 leaks across 6 shards — bug detector only |
| **G1 constraint non-vacuous in the kernel** | **FAIL** | 0/6 sources and 0/19 endpoint-valid rollouts, threshold 0.10; 19 broke the motif, 0 recovered |
| G2 mask leaves room to act | **PASS** | mean `removed_fraction` 0.226; 0/42 states with empty support |
| G3 feasible paths can improve | **PASS** | `pathwise_greedy` +3.10 mean worst-margin over source |
| G4 distinguishable | **PASS** | landings differ 6/6 |
| G5 planning subclaim | **NOT ASSESSED** | stage B not authorised and not run |

## Binary headroom — a CEILING, not a null

`pathwise_greedy` reached `b_success` on **6/6** sources, so the denominator of
greedy failures against which any planning advantage could have shown is
**zero**. Had stage B run, its binary headroom would have been 0 **by
definition**. This is a ceiling, not a null result, and is a further reason
stage B would not have been informative on this panel.

## Guaranteed-sign warning, recorded before any data exists

`pathwise_verified ≥ pathwise_greedy` holds by policy improvement on **both**
the lexicographic utility and binary success (success is a threshold of the
worst margin, which cannot decrease). Only the **magnitude** of the gap, the
**top-1 disagreement rate**, and **binary headroom over the denominator of
sources where `pathwise_greedy` actually failed** are admissible. If that
denominator is 0 the result is a **ceiling, not a null**. The analysis script
emits this warning inline with the numbers.

Likewise "the pathwise arms had zero violations" is **definitional** and is
recorded only as a bug detector.

# Bugs, invalid instruments, and superseded runs

**No invalid runs. All 6 shards are `SMOKE_HELD_IN`; G0 found 0 mask leaks and
0 motif-drift voids.** One container was preempted mid-run and Modal restarted
it with the same input; the shard landed normally and is not affected.

Three defects were caught and fixed **before** any Modal spend:

- **Launch-time `ModuleNotFoundError: rdkit`.** `modal run` imports the app in
  the launcher's interpreter, which has no chemistry stack; the app's
  module-scope import of `pathwise_arms` pulled RDKit in transitively just to
  read five strings. Detected on the first launch attempt. **No Modal
  resources were consumed** — the failure was local, before dispatch. Fixed by
  a dependency-free `pathwise_arm_names` module; two AST tests now pin it.

- **`Chem.MolFragmentToSmarts` drops formal charge and aromaticity.** Detected
  by direct probe before any dependent code existed. No conclusion depended on
  it. Replaced with an explicit writer plus a self-match assertion on every
  derivation. Not an artifact status — it never produced one.
- **Arm policies were initially unreachable from tests** (they lived inside the
  Modal entrypoint). This is the arrangement that hid the retargeting lane's
  "no verified arm" defect for two full runs. Extracted to
  `src/compose_v4/experiments/pathwise_arms.py`; 27 tests now cover them.

# Known limitations

- **n = 6.** Every count in the results section sits on six sources. The G1
  rollout denominator (19) is larger but the sources are not independent of
  each other in the way a confirmatory panel would require. G1's failure is
  decisive in *direction* — 0 recoveries out of 19 breaks — but the panel is a
  smoke panel.
- **The result is horizon-specific.** H = 6 was frozen from the retargeting
  lane. Absorption is a statement about what the kernel can undo in six edits,
  not a claim that labeled ring systems are never restorable. Extending the
  horizon was deliberately **not** attempted, because doing so after seeing a
  failed gate is the tuning the protocol forbids.
- Two of the six panel sources carry unusual valences (`[PH]`, `[SH4]`). They
  entered under an outcome-independent rule and were **not** removed. Notably
  source 5 (`[SH4]`) is one of the two that violate, so the absorption finding
  does not rest solely on ordinary chemistry.
- Modal image pins RDKit 2024.3.5 while the panel was built under 2025.09.6.
  The motif re-derivation check **passed on all 6 shards**, so no aromaticity
  drift occurred on this panel.
- The mask's removal rate is extremely heterogeneous (median 0.055, max 0.999).
  A mean of 0.226 over 42 states is a poor summary of that spread; the
  per-state census is in the shards.
- Per-arm kernel-call attribution is order-dependent (shared cache). The run
  order is recorded so it stays interpretable, but marginal costs are not
  independent.
- Cost model is extrapolated from the retargeting cohort (≈123 s startup,
  ≈15.9 s/kernel call). Molecules with larger successor fibers will cost more.
- The verified controller's candidate strata (4/2/2) are inherited from the
  retargeting lane without re-tuning. That is deliberate — but it means the
  masked verified arm may shortlist fewer distinct candidates than intended
  when the mask removes much of the support.

# Exact reproduction commands

```bash
# environment/setup
cd <repo root>            # a worktree on branch codex/compose-pathwise-constraints
python3 -m pytest tests/test_pathwise_constraint.py \
                 tests/test_pathwise_arms.py \
                 tests/test_pathwise_analysis.py -q     # 64 passed

# local census and panel (both regenerate byte-identical artifacts)
python3 scripts/pathwise_constraints_census.py --scan 20000
python3 scripts/pathwise_select_smoke_panel.py --sources 6

# smoke -- EXECUTED 2026-08-12, app ap-QJiLH8rlpGNGySTMLjOMPH.
# CPU only, profile rahul-94866. --detach is MANDATORY; verify
# `modal app list` shows `ephemeral (detached)` before walking away.
# Re-running is a no-op: drive() skips indices whose shard is committed.
PYTHONPATH=src:. MODAL_PROFILE=rahul-94866 \
  modal run --detach modal_apps/pathwise_constraints_app.py --stage A --sources 6

# stage B was NOT run and is NOT recommended (G1 failed; greedy is at ceiling).

# pull shards off the volume
modal volume get compose-v4-artifacts \
  editing_v2/r_theta_run/pathwise_constraints_smoke_stageA/ \
  diagnostics/pathwise_constraints_smoke_stageA_shards/ --force

# analysis -- reproduces diagnostics/pathwise_constraints_smoke.json exactly
python3 scripts/analyse_pathwise_constraints.py \
    --shards diagnostics/pathwise_constraints_smoke_stageA_shards \
    --out diagnostics/pathwise_constraints_smoke.json
```

The six raw shards are committed to the repository, so the aggregate is
reconstructible without touching Modal.

# Durable artifacts

| Artifact | Path | SHA-256 | Purpose |
|---|---|---|---|
| **stage A analysis** | `diagnostics/pathwise_constraints_smoke.json` | `435e45e78a7b2a17be3b9462135d62f13433fa05005abb5c5dab896baf4a07cb` | **the gate verdicts — `SMOKE_HELD_IN`** |
| **stage A raw shards** | `diagnostics/pathwise_constraints_smoke_stageA_shards/00{0..5}.json` | 6 files, in-repo | per-source trajectories; aggregate reconstructible offline |
| held-in census | `diagnostics/pathwise_constraints_census.json` | `cf44a31f7e65d0c11bbd594359913d056a0c8f94eb34035cbbb2be93295886d0` | viability census, source-only |
| frozen panel | `diagnostics/pathwise_constraints_smoke_panel.json` | `6b3f93722d80c59752aae33060ba6f79ee4cc4c9177458ccc1143456ff52f8de` | the 6 smoke sources (`panel_sha256 8b47a0e4…`) |
| protocol | `configs/pathwise_constraints_protocol_v1.json` | see `handoff.json` | frozen scientific contract |
| predicate | `src/compose_v4/experiments/pathwise_constraints.py` | `d6caa8645eb9c65859e75c23eabde80f6ed37970dca008a498e9ceb6017dac94` | constraint + motif rule |
| arms | `src/compose_v4/experiments/pathwise_arms.py` | `3640787bb3f4fc33ed8ff27e1a72bf81ac4825d0da5382eddc84b9b758edfa6f` | all seven arm policies |
| arm names | `src/compose_v4/experiments/pathwise_arm_names.py` | `4d09aeab948a9af6c32e721cea3e3bce70a8b6f0fda810c045cc2b48a2561629` | dependency-free; keeps the launcher RDKit-free |
| modal app | `modal_apps/pathwise_constraints_app.py` | `5869f8b063a327553668d0d35ed1994ce340040f5d336b321285c3a173a2a39d` | the runner |
| analyser | `scripts/analyse_pathwise_constraints.py` | `99bb370b5664138d5bba52d3d4b0569aa58508cbeada0051855854b8c142e08a` | gate verdicts |

Nothing load-bearing lives in `/private/tmp` or a scratchpad. The raw shards
are in the repository, so `diagnostics/pathwise_constraints_smoke.json` can be
regenerated with no Modal access.

# Cost actually incurred

| | Estimated | Actual |
|---|---:|---:|
| kernel calls / source | 85 (65–110) | **51** (43–58) |
| seconds / source | ~1 475 | **853** (620–1 163) |
| container-hours, stage A | 2.5 (1.9–3.1) | **≈ 1.4** |

Stage A came in at ~56% of the estimate. The 360-call circuit breaker was never
approached (max observed 58). Stage B was not run, so its ≈4.3 h was not spent.

# Files changed

```text
src/compose_v4/experiments/pathwise_constraints.py
src/compose_v4/experiments/pathwise_arms.py
src/compose_v4/experiments/pathwise_arm_names.py
diagnostics/pathwise_constraints_smoke.json
diagnostics/pathwise_constraints_smoke_stageA_shards/000.json .. 005.json
modal_apps/pathwise_constraints_app.py
scripts/pathwise_constraints_census.py
scripts/pathwise_select_smoke_panel.py
scripts/analyse_pathwise_constraints.py
configs/pathwise_constraints_protocol_v1.json
diagnostics/pathwise_constraints_census.json
diagnostics/pathwise_constraints_smoke_panel.json
tests/test_pathwise_constraint.py
tests/test_pathwise_arms.py
tests/test_pathwise_analysis.py
docs/workstreams/pathwise-constraints/STATUS.md
docs/workstreams/pathwise-constraints/PROTOCOL.md
docs/workstreams/pathwise-constraints/DECISION_LOG.md
docs/workstreams/pathwise-constraints/HANDOFF.md
docs/workstreams/pathwise-constraints/handoff.json
```

# External evidence — GraphXForm is an endpoint-only comparator

From the baseline-qualification lane: **GraphXForm's action masking is
genuinely pathwise but covers only valence, atom type, atom count and bonding
legality. Its ring-size and bonding-pattern constraints are a TERMINAL filter
(`molecule_evaluator.py::infeasible_by_special_constraints`, asserted on
`mol.synthesis_done`), and there is no SMARTS or substructure matching anywhere
in that repository.**

So the strongest published graph-editing comparator instantiates this lane's
`endpoint_only` arm, not its pathwise arm. That makes the *design* question
well-posed: COMPOSE really can do something GraphXForm cannot.

**But it raises the stakes on G1 rather than lowering them.** With G1 failing,
the distinction is **academic in this regime**: if trajectories that break the
motif never come back, neither COMPOSE's pathwise mask nor GraphXForm's
terminal filter changes the returned molecule. The capability gap is real; the
*consequence* of that gap is what this run failed to demonstrate.

# Recommended next action

> **Accept both negatives and close the lane.** Under frozen criteria applied
> as written, the ring-system premise failed on reversibility and no
> replacement family cleared the bar. Move pathwise constraints out of the main
> paper, keep the ring-system absorption finding as an appendix result about
> the frozen kernel's legal support, and spend the remaining budget on lanes
> whose premise held.

## The one decision that is the lead's, not this lane's

Family B (cLogP corridor) is **reversible** — 6 of 7 violators returned, with
excursions a third of the corridor wide — and failed **only** the 20-event bar,
on a 42-trajectory pool inherited from a 6-source smoke panel.

If the lead judges that the reversibility question deserves a powered answer
rather than an underpowered one, the bounded run that would settle it:

- **~12 held-in sources × 6 rollouts ≈ 72 unconstrained trajectories**, which
  clears V3 at the observed rate (~71 needed).
- Stage-A cost basis: 51 kernel calls and 853 s per source ⇒ **≈ 2.8
  container-hours**, CPU only, `--detach`, resumable driver.
- It would also produce the successor sets needed for **V4**, which the current
  shards cannot answer.

This lane does **not** recommend it and does not consider it authorised. It is
recorded because the alternative — reporting "all three failed" without noting
that two failed on power and one on science — would misinform the decision.

# Actions explicitly not recommended

- Do **not** run stage B. Beyond being unauthorised, `pathwise_greedy` already
  succeeds 6/6, so binary headroom sits over a **zero denominator** — a ceiling
  by definition. It would buy nothing.
- Do **not** change the motif rule, the horizon, or the panel and re-run. That
  is the exact tuning the protocol forbids and main lane reconfirmed.
- Do **not** open a held-out panel. The premise gate failed; there is nothing
  to confirm.
- Do **not** report "the masked arms had zero violations" as a finding. It is
  definitional and is barred from the results table.
- Do **not** promote the secondary finding (`endpoint_only` returning nothing
  on 2/6) to the lane's headline. It is a different, weaker claim on n = 6.
- Do **not** merge the two extra arms into the paper's arm table without
  noting they were added beyond the brief's five (see `DECISION_LOG.md`).
- Do **not** search a fourth or fifth constraint predicate. Three were
  predeclared and ranked; searching further is the failure mode the salvage
  lane's bounds exist to prevent.
- Do **not** lower the 20-event bar to convert family B into a pass. If the
  lead wants B answered, buy the trajectories; do not move the line.

# Deviations from the brief, flagged for the lead

1. **Eligibility drops "the kernel offers motif-destroying successors"**
   (brief line 207). It is the numerator of the vacuity gate; filtering on it
   would make a non-vacuous mask true by construction. Measured and reported
   per source instead.
2. **Two arms added** beyond the brief's five, both to remove confounds
   (rollout budget, controller identity).
3. **Directory name** is `docs/workstreams/pathwise-constraints/` per the
   lead's mission; the workstream plan's deliverables list says
   `pathwise_constraints`. All non-doc paths use the underscored form.

# Main-session pickup checklist

- [ ] Read protocol before results.
- [ ] Verify all frozen-input hashes.
- [ ] Confirm held-out-open status. (It is: **not opened**.)
- [ ] Reproduce one smoke. (Local: the 64-test suite. Modal: nothing has run.)
- [ ] Inspect known-invalid runs. (There are none — nothing has run.)
- [ ] Decide explicitly whether to merge, authorize held-out evaluation, or stop.
