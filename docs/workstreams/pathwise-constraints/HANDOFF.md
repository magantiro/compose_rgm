# Workstream

- **Name:** Workstream C — Pathwise constraints
- **Claim ID:** Claim 4 (trajectory-level design / intervention)
- **Branch:** `codex/compose-pathwise-constraints`
- **Base commit:** `04f1c4661c0f4977d00849893550e02d52dc7c64`
- **HEAD commit:** see `handoff.json` (`head_commit`)
- **Working tree clean:** yes
- **Status:** `DESIGN_ONLY`
- **Held-out data opened:** **no**

# One-sentence scientific question

> Can COMPOSE require an exact labeled structural motif to be present at
> *every* committed molecular state, and does that constraint actually bind —
> i.e. does endpoint-only filtering return molecules that reached validity by
> passing through forbidden intermediates?

# Claim this work can support

> Because every COMPOSE state is a complete molecule and every transition is
> executable, an exact atom/bond-labeled subgraph requirement can be imposed on
> the support of the entire controlled process; forbidden intermediate
> molecules become unreachable by construction rather than filtered after the
> fact.

# Claims this work cannot support

- That constrained control beats unconstrained control on the objective.
- That the protected motif is a medicinally meaningful pharmacophore.
- That future-aware control beats greedy under the mask — that comparison has
  a guaranteed sign (see "Gate verdicts").
- Anything about held-out generalisation.
- **Any empirical claim whatsoever.** No Modal run has been executed. Every
  number below is either a source-only local census or a cost estimate.

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
| SMARTS writer fuzz, 4 000 held-in sources | 0 failures | scratch probe; 1 500-source version is in the suite |
| Held-in census, 20 000 sources | complete | `diagnostics/pathwise_constraints_census.json` |
| Modal smoke | **NOT RUN** | — |

# Results

**These are LOCAL SOURCE-ONLY CENSUS results (`SMOKE_HELD_IN`, census
component). No trajectory has been generated. Nothing here is claim-bearing.**

| Metric | Arm / condition | Value | Uncertainty / denominator |
|---|---|---:|---|
| sources with a ring system | held-in scan | 98.32% | 19 665 / 20 000 |
| motif atom count | median | 9 | 19 665 (p05 6, p95 16) |
| motif fraction of molecule | median | 0.310 | 19 665 (p05 0.177, p95 0.647) |
| heavy atoms outside motif | median | 18 | 19 665 (p05 7, p95 29) |
| eligibility yield | held-in scan | 74.91% | 14 983 / 20 000 |
| already satisfies `B` | held-in scan | 0.94% | 187 / 20 000 |
| already satisfies `D` | held-in scan | 32.63% | 6 526 / 20 000 |
| already satisfies `P` | held-in scan | 2.33% | 466 / 20 000 |
| SMARTS self-match failures | 4 000-source fuzz | 0 | 0 / 3 932 with motif |

# Gate verdicts

| Gate | PASS / FAIL / INCONCLUSIVE | Evidence |
|---|---|---|
| Motif exists and is non-trivial | **PASS** | 98.3% have a ring system; median 31% of the molecule with 18 free atoms — neither the whole molecule nor a one-atom pattern |
| Objective has headroom | **PASS** | 0.9% of held-in sources already satisfy `B`; potency is the binding term on all 6 panel sources |
| Eligibility is workable | **PASS** | 74.9% yield with constants fixed before the census |
| Predicate is non-vacuous *as a predicate* | **PASS** | 6 distinct mutation classes rejected in unit tests |
| **G1 constraint non-vacuous *in the kernel*** | **INCONCLUSIVE — requires stage A** | needs successor enumeration; not computable locally |
| **G2 mask leaves room to act** | **INCONCLUSIVE — requires stage A** | same |
| **G3 feasible paths can improve** | **INCONCLUSIVE — requires stage A** | same |
| **G4 distinguishable** | **INCONCLUSIVE — requires stage A** | same |
| G5 planning subclaim | **NOT ASSESSED** | stage B; do not run before G1 passes |

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

No invalid runs — nothing has been run. Two defects were caught and fixed
during construction:

- **`Chem.MolFragmentToSmarts` drops formal charge and aromaticity.** Detected
  by direct probe before any dependent code existed. No conclusion depended on
  it. Replaced with an explicit writer plus a self-match assertion on every
  derivation. Not an artifact status — it never produced one.
- **Arm policies were initially unreachable from tests** (they lived inside the
  Modal entrypoint). This is the arrangement that hid the retargeting lane's
  "no verified arm" defect for two full runs. Extracted to
  `src/compose_v4/experiments/pathwise_arms.py`; 27 tests now cover them.

# Known limitations

- The vacuity gate is **unresolved**. If the frozen kernel's legal support
  largely preserves ring systems, G1 fails and this workstream has no
  trajectory-level claim. That is a live possibility, not a formality.
- Two of the six panel sources carry unusual valences (`[PH]`, `[SH4]`). They
  entered under an outcome-independent rule and were **not** removed; they come
  from the training corpus and are representable, but they are not typical
  medicinal chemistry.
- Modal image pins RDKit 2024.3.5 while the panel was built under 2025.09.6.
  Aromaticity perception could differ. The app re-derives the motif on Modal
  and voids the shard on disagreement, but this has not been observed to pass
  or fail because nothing has run.
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

# smoke -- NOT YET AUTHORISED. CPU only, profile rahul-94866.
modal run modal_apps/pathwise_constraints_app.py --stage A --sources 6
# stage B ONLY if G1 passes:
# modal run modal_apps/pathwise_constraints_app.py --stage B --sources 6

# analysis (after copying shards off the volume)
python3 scripts/analyse_pathwise_constraints.py \
    --shards <dir with pathwise_constraints_smoke_stageA/*.json> \
    --out diagnostics/pathwise_constraints_smoke.json
```

# Durable artifacts

| Artifact | Path | SHA-256 | Purpose |
|---|---|---|---|
| held-in census | `diagnostics/pathwise_constraints_census.json` | `cf44a31f7e65d0c11bbd594359913d056a0c8f94eb34035cbbb2be93295886d0` | viability census, source-only |
| frozen panel | `diagnostics/pathwise_constraints_smoke_panel.json` | `6b3f93722d80c59752aae33060ba6f79ee4cc4c9177458ccc1143456ff52f8de` | the 6 smoke sources (`panel_sha256 8b47a0e4…`) |
| protocol | `configs/pathwise_constraints_protocol_v1.json` | see `handoff.json` | frozen scientific contract |
| predicate | `src/compose_v4/experiments/pathwise_constraints.py` | `d6caa8645eb9c65859e75c23eabde80f6ed37970dca008a498e9ceb6017dac94` | constraint + motif rule |
| arms | `src/compose_v4/experiments/pathwise_arms.py` | `d6eb1a8b454ad8ac11b4a656fdc719d1666008126671c74ae8c46c77fe278faa` | all seven arm policies |
| modal app | `modal_apps/pathwise_constraints_app.py` | see `handoff.json` | the runner |

Nothing load-bearing lives in `/private/tmp` or a scratchpad.

# Files changed

```text
src/compose_v4/experiments/pathwise_constraints.py
src/compose_v4/experiments/pathwise_arms.py
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

# Recommended next action

> Authorise **stage A only**: `modal run modal_apps/pathwise_constraints_app.py
> --stage A --sources 6`. Six held-in sources, CPU only, **≈ 2.5
> container-hours** (range 1.9 – 3.1), ≈ 30 min wall. It resolves the vacuity
> gate G1 — whether unconstrained trajectories ever violate the motif mid-path
> — which decides whether this workstream has a claim at all. Report the gate
> before anything else is spent.

# Actions explicitly not recommended

- Do **not** authorise stage B (the two verified arms, a further ≈ 4.3
  container-hours) before G1 passes. If the constraint is vacuous, planning
  inside the mask is a question about nothing.
- Do **not** open a held-out panel. This lane has not selected one and should
  not.
- Do **not** change the motif rule if G1 fails. A failed G1 is a reportable
  result about the frozen kernel's legal support, not a prompt to re-roll.
- Do **not** cite any number in this handoff as claim-bearing. All of it is
  either a source-only census or a cost estimate.
- Do **not** merge the two extra arms into the paper's arm table without
  noting they were added beyond the brief's five (see `DECISION_LOG.md`).

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
