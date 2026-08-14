# Protocol — multiobjective evidence package

**Artifact status: `DESIGN_ONLY`.** No arm has been run by this lane. This is the
scientific contract for the external-baseline and fairness half of the Pareto
block. Lane 4 owns the internal COMPOSE Pareto arms and this document does not
alter them.

---

## 1. Claim

> COMPOSE reuses one frozen executable molecular graph process across
> target-free preferences. Preference-specific purpose is imposed through
> inference-time control rather than objective-specific updates to the learned
> reference process.

## 2. Non-claims — stated first, because these are what the evidence must not drift into

- We do **not** claim to invent Pareto optimization, preference-conditioned
  generation, or Doob control.
- We do **not** claim COMPOSE beats a purpose-built preference-conditioned
  molecular Pareto generator at that generator's own global task. That is what
  Panel B measures, and the result is reported whichever way it comes out.
- We do **not** claim "planning beats greedy" as a Pareto headline.
  Exact-target recovery already carries that claim, and PepTune (ICML 2025,
  same lab) already establishes inference-time tree-search multiobjective
  guidance in the literature. See `SAME_LAB_LINEAGE.md`.
- **Barred phrasing, verbatim, from the development preregistration:**
  ~~"COMPOSE precisely sweeps the Pareto frontier according to user
  preference."~~ The smoke does not support it. Also barred:
  ~~"five weights → five perfectly monotonically ordered points"~~, and
  "five cleanly ordered Pareto regions".
- "Without retraining" is barred project-wide. The operational, checkable
  replacement is **zero parameter updates**.
- We do **not** claim any efficiency result without naming its resource axis.

## 3. Frozen inputs — read and cited, never reconstructed

| object | value | where frozen |
|---|---|---|
| objective pair | `potency_vs_developability`, `(P, D)`, both maximized | `diagnostics/pareto_tradeoff_census.json :: adopted_pair` |
| `P` | DRD2 SVM log-odds margin, centred and scaled by held-in median/IQR | `pareto_control_app.py:212` |
| `D` | soft-min at temperature 0.25 over clipped QED and cLogP-box z-scores | `pareto_control_app.py:213-219` |
| utopia `z*` (held-in p99) | `P = 2.7315048451066857`, `D = 0.555419816046901` | `pareto_tradeoff_census.json :: frozen_scales.utopia_p99` |
| nadir `r` (held-in p5) | `P = -1.2689119285385548`, `D = -1.3547257562637012` | `pareto_tradeoff_census.json :: frozen_scales.reference_p5` |
| preferences | `(0.1, 0.3, 0.5, 0.7, 0.9)`, applied as `w = [weight, 1 - weight]` | `pareto_control_app.py:103` |
| K (preference trajectories per arm per source) | 5 | `len(PREFERENCES)` |
| horizon / edit budget | `BUDGET = 6` | `pareto_control_app.py:102`, "Frozen by PROTOCOL.md section 7.1. Not tunable here." |
| shortlist | `top_immediate 4, top_reference 2, n_random 2` | `pareto_control_app.py` `SHORTLIST` |
| scalarization | augmented Chebyshev, `CHEBYSHEV_RHO = 1e-3`, minimized | `pareto_control.py` |
| panel | 12 held-in sources | `diagnostics/pareto_control_cohort.json` |
| DRD2 oracle | `artifacts/oracles/drd2_svm_v1`, pickle sha256 `dbc473fc…111100f` | `drd2_oracle_manifest.json` |

**These are read-only for this lane.** Objectives, weights, K, horizon, the
controller and the decision rules are not altered here, and the 12-source smoke
is neither rerun nor reinterpreted.

### The normalizer is not a bound

`z*` is the held-in p99, not an attainable utopia. Held-in-scaled hypervolume
**may exceed 1** and 12.7% of committed smoke endpoints do exceed it (max excess
0.909). The response is to report exceedance descriptively. **Never clip, and
never introduce a clipped or robust variant after seeing a result.** Enforced by
`multiobjective_qualification.load_frozen_scales`, which returns
`is_a_cap: False`, and by `assert_reference_is_frozen`, which refuses any
reference provenance other than the freeze.

---

## 4. Panel construction

### Panel A — source-conditioned control (COMPOSE-native)

Every arm starts from the **same supplied molecule** under the **same edit
budget**. Members: COMPOSE fixed-preference control; generate-and-rank P3/P4;
the empirical-family reference ablation; the preference-blind floor; and any
published method natively supporting supplied-source semantics.

**As of this audit, no external method qualifies for Panel A.** Fronts are
computed **per source** and aggregated with source-level paired bootstrap
intervals. Sources are never pooled into one global front.

### Panel B — global multiobjective competence

A separately frozen common benchmark on which every method has the same global
task freedom and the same objective budget. Members: HN-GFN, InversionGNN
conditionally, and COMPOSE if a global arm is admissible at all.

Labelled in the paper as a **conventional Pareto-competence comparison**, not as
the same-state capability experiment.

### The rule that must never be broken

> Never silently grant a global method a larger reachable set and call the
> comparison source-matched.

Mechanized: `assert_not_cross_panel` raises on any Panel A versus Panel B pair,
and `test_cross_panel_comparison_raises` holds it.

---

## 5. Metrics

### Preference responsiveness comes first (Q1)

Monotonic/rank response between requested preference and achieved property
balance; extreme-preference separation; distinct objective regions; distance
above the preference-blind floor.

> **Distinct SMILES is not the primary response metric.** Five distinct
> structures in the same property region do not establish preference control.
> Ordering was the test in the smoke, and ordering came back partial.

### Set-level front quality (Q2)

Final HV at the frozen number of preference trajectories; HV-AUC; per-`k` HV
curves; nondominated count; front coverage and spread; molecular diversity;
chemical-envelope fidelity. Nadir and property scaling are the frozen held-in
values. The reference is **never** defined from COMPOSE's own best front.

**HV-AUC is a mixture and carries a standing caveat**, quoted verbatim from
`diagnostics/pareto_smoke_hv_auc.json`:

> "HV-AUC's win count MUST NOT be quoted as a set-level result. The curve's
> early points approximate per-trajectory quality, where verified control's
> advantage is near-definitional; only k=K is the set-level object P2 tests.
> HV-AUC blends them."

Concretely: HV-AUC is 12W/0L, and at `k = 5`, the only set-level point, the same
committed trace is **10W/2L** with mean `+0.08896161819632241`. The 12W/0L must
never be presented as a cleaner version of the 10W/2L.

### Two questions that stay separate

Per-preference scalarized improvement carries a structural sign guarantee
(policy improvement); set-level hypervolume does **not** and moves either way.
The permanent regression case is smoke source 000, where verified improved every
preference's scalarized value while `HV_verified` (1.0060) fell **below**
`HV_greedy` (1.0074). Reporting these as one number destroys the finding.

### Resource axes (Q3), never merged

Canonical vocabulary in
`src/compose_v4/experiments/multiobjective_qualification.py::RESOURCE_AXES`:

| axis | scope |
|---|---|
| `algorithmic_oracle_requests` | cross-method |
| `unique_oracle_evaluations` | cross-method; the PMO-comparable convention |
| `evaluator_calls` | cross-method |
| `surrogate_calls` | cross-method; **zero for COMPOSE, large for any BO method** |
| `benchmark_eval_requests` | ours, never added back into the algorithmic counter |
| `kernel_calls` | COMPOSE-internal only |
| `completed_trajectories` | COMPOSE-internal only |
| `wall_core_seconds` | secondary only, never primary, never across hardware |

`reconcile_ledger` maps Lane 4's `CostLedger`, Lane 4's semantic correction, and
Workstream D's `OracleCounts` onto this vocabulary, and **raises on any counter
it does not recognise** — a counter dropped in translation is an efficiency
claim without an axis.

`raw_instrument_oracle_requests` is carried through as a passthrough record, not
an axis: it is the object a serial-versus-parallel parity replay must match
exactly, and mapping it onto an axis would attribute our harness overhead to the
method.

**The surrogate rule.** `oracle_efficiency_verdict` returns
`INCOMPARABLE_SURROGATE_ASYMMETRY`, not a number, when one method consumes a
learned surrogate and the other does not. HN-GFN's 1,000-call budget is spent
while its GFlowNet trains against a proxy; COMPOSE pays the frozen evaluator at
every decision. On an oracle axis alone that difference reports amortization
wearing efficiency's clothes.

### Q3 reporting shape

Do not reduce P3 to one arbitrary matched-budget number. Report resource
frontiers separately against completed trajectories, kernel/generative calls,
algorithmic objective requests, benchmark-only evaluations, actual evaluator
calls and batches, with wall/core time secondary. Where a kernel-matched point
is unreachable for a source, report `MATCHING_UNREACHABLE` for **that source**
and do not alter generate-and-rank until it matches.

Barred generalization, verbatim from the top-up preregistration:
~~"generate-and-rank cannot be kernel matched."~~ What is licensed is only:
"kernel matching is unreachable for source 000 under the frozen generate-and-rank
process."

### Q4 verdicts

`R_θ` versus the empirical family under identical control, same legal support,
source panel, preferences, controller, horizon, candidate limits and budgets,
with the corresponding reference law also governing any rollout or value
computation. Pareto quality, chemical-envelope fidelity, feasibility, source
preservation and resource use are evaluated **jointly and never collapsed into a
post-hoc weighted score**. Allowed verdicts: `R_θ dominates`;
`empirical-family dominates`; `incomparable`; `unresolved`.

---

## 6. Allowed calibration

- Environment and compatibility work that **restores** originally intended
  behaviour. The model case is the MARS sklearn fix. Repairing InversionGNN's
  call/definition arity mismatch so the authors' own function is callable is in
  this family.
- Building our evaluation, canonicalization and accounting adapter.
- Reporting applicability: which panel members a method can and cannot handle,
  always reported and never used to drop molecules quietly.

## 7. Forbidden adaptations

- Changing a published baseline's architecture, scalarization, search or
  training procedure. Specifically: **do not supply HN-GFN a Chebyshev.**
- Choosing actions for a method, rewriting its masks, disabling `TERMINATE`,
  altering its proposal distribution, or substituting our scalarization.
- Supplying a learned component the authors never released, or choosing its
  training data. InversionGNN ships no checkpoint; training one for it and
  calling the result InversionGNN would be inventing its learned component.
- Porting sequence-edit methods into homemade molecular graph methods.
- Comparing global and source-conditioned fronts as though they were one task.
- Adding a method because COMPOSE lost, or dropping one because COMPOSE won.
- Creating a new objective pair after outcomes are visible.
- Sweeping a knob and reporting the attractive value.

---

## 8. Panel B — RESOLVED 2026-08-13: no head-to-head row

**Superseding the earlier recommendation of `{GSK3β, JNK3}` in this section.**
That recommendation was made on the reasonable assumption that a pair both
external papers report is a pair they report *comparably*. The alignment audit
tested that assumption and it failed.

**Verdict: `{GSK3β, JNK3}` is not frozen.** Full evidence in
`BENCHMARK_ALIGNMENT_AUDIT.md`; the two decisive dimensions:

- **Oracle identity.** HN-GFN uses a 1024-bit ECFP4 with its own RandomForests;
  TDC — which InversionGNN calls — uses a 2048-bit ECFP4. Different feature
  dimension, therefore different models. The two papers' published numbers were
  not produced by the same oracle.
- **Budget.** The published budget is 1,000 true-oracle evaluations. COMPOSE
  spends 2,187 per preference trajectory on its cheapest guided arm and 53,977
  on its primary arm. It cannot complete one trajectory inside the whole budget.

The budget finding **generalizes to oracle-budget benchmarking as a class**
(10³–10⁴ conventional; COMPOSE's primary arm ~5.4 × 10⁴ per trajectory), so no
substitute benchmark repairs it.

**Consequently, Panel B carries no numeric COMPOSE-versus-external row.**
Published external values appear as **contextual, non-head-to-head**, each
carrying its oracle provenance and its budget, never in a shared-header column
with a COMPOSE number, with a caption naming the failed dimension.

**The freeze criterion actually applied** was external protocol overlap alone. No
COMPOSE outcome entered the decision, and the in-flight P3/P4 repair was neither
consulted nor waited for — a P3/P4 verdict may change how much emphasis Panel B
needs, and may never change which benchmark Panel B uses.

**What this costs and what it does not.** It removes the option of leaning on an
external row. It does not weaken the multiobjective claim, which was always
carried by Panel A and the Ring 3 causal controls — a global competence row was
context, not evidence for the claim in §1.

---

## 9. Stop rules

1. **No compute without explicit per-run approval from the lead.** Stage 2
   requires it; an HN-GFN smoke additionally requires GPU authorization, since
   the authors report 10 hours on one V100 and no checkpoint or CPU path exists.
2. **Baseline engineering creep.** If making an external method competitive
   starts to require inventing search rules, objectives, action masks or
   architectures for it, stop. At that point it is no longer a published
   baseline.
3. **Licence stop.** OP-GFN is CC BY-NC-ND; a derivative adapter is outside the
   grant. InversionGNN has no licence at all, so its code is not vendored.
   Neither is resolvable by engineering.
4. **Three integrity guards deep is a signal to stop, not to patch the fourth.**
5. If Panel B cannot be frozen without giving one method a task shaped to suit
   it, run no Panel B and say why. `N/A` is preferable to a distorted adaptation.

---

## 10. The five-question block, per recommended experiment

Required by `docs/COMPARATOR_ROLES_CANONICAL.md`. Every experiment this lane
still recommends answers all five, and an experiment that cannot name a
**falsifier** is not an experiment.

**This lane recommends two experiments and declines one.** None of the three is
designed here — E1 exists and is Lane 4's; E2 is conditional on a qualification
that is not finished; E3 is declined outright.

### E1 — source-conditioned fixed-preference Pareto control (Panel A)

**Status: exists, frozen, Lane 4 owns the runs. This lane owns its framing.**

| question | answer |
|---|---|
| **methodological axis** | can one frozen executable process be re-purposed across target-free preferences by inference-time control alone, with **zero parameter updates**, from an exact supplied source state? |
| **framework counterfactual** | `PENDING` — DDSBM, and GrIDDD if it qualifies. **Currently unfilled**, and that is the single largest gap in the multiobjective package. |
| **matched causal control** | generate-and-rank P3/P4 (when purpose is applied); empirical-family ablation (what `R_θ` buys); `unguided` floor (that response is not chance); greedy vs verified (lookahead) |
| **competence comparator** | HN-GFN and InversionGNN as **contextual, non-head-to-head** published values only. **No numeric row** — all six alignment dimensions failed. |
| **falsifier** | preference response indistinguishable from the preference-blind floor; or set-level HV at `k = K` failing to separate arms; or per-source fronts collapsing to one objective region. **Partly fired already:** the smoke returned monotone ordering on only 1 source in 6, guided ρ = +0.636 against a floor of +0.432 — "five cleanly ordered Pareto regions" is barred as unsupported. |

The falsifier has already bitten once and the claim was narrowed rather than the
test loosened. That is the record this block exists to preserve.

### E2 — framework-neighbor process comparison

**Status: conditional. Not designed, not scoped, not authorized.**

| question | answer |
|---|---|
| **methodological axis** | why an executable graph CTMC over **canonical legal-rewrite fibers**, rather than another graph-CTMC, bridge, or insert/delete diffusion abstraction? |
| **framework counterfactual** | this *is* the framework counterfactual — DDSBM, and GrIDDD if it qualifies |
| **matched causal control** | the same Panel A controls, since the executor is held fixed on the COMPOSE side |
| **competence comparator** | not applicable; this experiment is not about task performance |
| **falsifier** | an alternative graph process reaching comparable frontier quality **from the same supplied source at the same edit budget** would show the exact-fiber substrate is not what carries the result. That is a real possible outcome and it is reported if it occurs. |

**Precondition.** It runs only if a framework neighbor qualifies **without
substantial adaptation**. If our adapter would have to supply mechanism —
source conditioning, an edit budget, chemical support — the method drops to
`CONCEPTUAL_LINEAGE_ONLY` and this experiment does not exist. That is the same
rule that bars a homemade Edit Flows port, and it applies to GrIDDD and DDSBM
equally.

### E3 — Panel B head-to-head global multiobjective row

**Status: DECLINED. Recorded so the decline is auditable, not silent.**

| question | answer |
|---|---|
| **methodological axis** | none — it tests competence, not novelty |
| **framework counterfactual** | none; HN-GFN and InversionGNN are `TASK_COMPETENCE` |
| **matched causal control** | none available; neither method is source-conditioned |
| **competence comparator** | would have been HN-GFN and InversionGNN |
| **falsifier** | **already fired, before any COMPOSE number existed.** All six protocol dimensions fail, and the two papers disagree by 70× on HN-GFN's own oracle budget. |

Two independent lines of reasoning reached this decline — protocol forensics
(the alignment audit) and framework-first selection (the role amendment) — from
different directions. Recording both matters: agreement between independent
arguments is evidence; a single argument reused twice is not.
