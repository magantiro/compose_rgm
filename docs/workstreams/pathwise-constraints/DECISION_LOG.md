# Workstream C — Pathwise Constraints: Decision Log

Every material decision, the evidence available *before* it, the alternatives
rejected, and whether it touches a frozen object.

---

## 2026-08-12 — Objective is `B = P AND D`, not `D` alone

**Decision.** Reuse the frozen retargeting goal language with `B = P AND D` as
the terminal objective.

**Evidence before the decision.**
`diagnostics/retarget_calibration_result_3plus3_fixed.json` records
developability as a ceiling: greedy 28/30, verified 28/30, binary headroom 0,
gate CLOSED. The local held-in census (20 000 sources) independently shows
32.6% already satisfy `D` at step zero versus 2.3% for `P` and 0.9% for `B`.

**Alternatives rejected.** (a) `D` alone — no headroom, so "the mask costs
nothing" would be uninformative. (b) A bespoke motif-aware oracle — the brief
forbids inventing another oracle.

**Frozen object touched.** None. The goal language is imported verbatim.

---

## 2026-08-12 — Motif rule is the largest fused ring system, not Murcko

**Decision.** `largest_ring_system_v1`: connected components of the ring-bond
graph, largest by atom count, ties broken by ring count then sorted atom-index
tuple.

**Evidence before the decision.** Census of held-in molecules: 98.3% have at
least one ring system; median motif is 9 atoms and 31% of the molecule, with 18
free atoms outside it. The Murcko scaffold on these molecules is frequently
almost the whole molecule, which fails the brief's "neither nearly the whole
molecule nor a trivial one-atom pattern" condition.

**Alternatives rejected.** (a) Murcko scaffold — too large too often.
(b) Murcko *generic* scaffold — discards atom labels, and the constraint is
defined as exact-labeled. (c) A curated pharmacophore list — not derivable per
source and not outcome-independent.

**Frozen object touched.** None; new module.

---

## 2026-08-12 — `Chem.MolFragmentToSmarts` rejected as the query builder

**Decision.** Write the labeled SMARTS explicitly (`fragment_smarts`) rather
than using RDKit's fragment serialiser.

**Evidence before the decision.** Direct probe: on all five test molecules
`MolFragmentToSmarts` emitted bare `[#6]` / `[#7]` primitives — atomic number
only. It drops the aromaticity flag and the formal charge, so `[#7]` matches a
neutral amine and a quaternary ammonium alike. Under that query a
charge-changing edit would pass the pathwise constraint silently.

**Alternatives rejected.** (a) Post-processing the emitted token stream — the
token order does not reliably map to `atomsToUse`. (b) Building an `RWMol`
query with default comparators — the atom/bond matching semantics are implicit
and version-sensitive.

**Risk accepted, and its control.** A hand-written graph serialiser can be
wrong. Every derivation therefore self-checks that the query it built matches
the molecule it came from, and
`test_every_held_in_source_matches_its_own_derived_motif` runs that check over
1 500 held-in sources. A separate 4 000-source probe found 0 failures.

**Frozen object touched.** None.

---

## 2026-08-12 — "kernel offers motif-destroying successors" is NOT an eligibility criterion

**Decision.** Drop this criterion from source eligibility, against the letter
of the workstream brief, and report it as a per-source measurement instead.

**Evidence before the decision.** It is the numerator of the vacuity gate
(G1). Selecting sources on it would guarantee a non-vacuous mask by
construction — the same shape as the three instrument defects recorded in
`docs/RETARGETING_SAME_PREFIX_DESIGN.md`, where a statistic's sign was fixed
before any data existed.

**Alternatives rejected.** Applying it and reporting the gate anyway — the gate
would then be uninterpretable. Applying it and reporting a *different* gate —
no other statistic in this lane carries the claim.

**Enforcement.** `test_eligibility_never_reads_a_successor_set` pins the
`eligibility` signature, so reintroducing it fails the suite and forces the
change through this log.

**Frozen object touched.** Deviates from
`docs/workstreams/PARALLEL_WORKSTREAMS_AND_HANDOFF.md` line 207. Flagged for
the lead in `HANDOFF.md`.

---

## 2026-08-12 — Panel seed 20260813, retargeting cohort excluded

**Decision.** Shuffle the held-in pool under seed 20260813 and exclude the 30
committed retargeting sources by SMILES.

**Evidence before the decision.** `diagnostics/retarget_calibration_cohort.json`
records `seed: 20260812` and `scanned: 57`, so reusing that seed would re-draw
the same prefix of the shuffled pool. The workstream plan also requires the
Claim-2 confirmatory panel to be disjoint from "pathwise-constraint development
sources", which presupposes this lane owns its own sources.

**Alternatives rejected.** Reusing seed 20260812 with an offset — more fragile
and harder to state than a fresh seed plus an explicit exclusion list.

**Frozen object touched.** None.

---

## 2026-08-12 — Eligibility constants fixed before the census, not after

**Decision.** `motif_fraction ∈ [0.20, 0.70]`, `min_motif_atoms = 6`,
`min_free_atoms = 8`, heavy atoms 18–38.

**Evidence before the decision.** The heavy-atom band is reused verbatim from
the frozen retargeting cohort. The other three were written into the module
*before* the census ran, then checked: yield 74.9% of 20 000 held-in sources
(rejections: motif_fraction 2 626, size_band 2 093, too_few_free_atoms 1 207,
motif_too_small 458, no_ring_system 335, already_satisfies_goal 187). No
constant was adjusted afterwards.

**Why this is allowed calibration.** The census involves no successor
enumeration, no trajectory and no arm outcome, so it cannot bias any gate.

**Frozen object touched.** None.

---

## 2026-08-12 — Two extra arms beyond the brief's five

**Decision.** Add `pathwise_stochastic` (masked twin of `endpoint_only`) and
`unconstrained_verified` (like-for-like partner for `pathwise_verified`).

**Evidence before the decision.** The brief's arm list compares a best-of-N
selection arm (`endpoint_only`) against single-trajectory greedy arms. Any
utility gap between them would confound the mask with the rollout budget.
`pathwise_stochastic` differs from `endpoint_only` in exactly one thing — the
mask — so the paired delta is attributable. `unconstrained_verified` gives the
cost of the guarantee a matched-controller form.

**Cost.** Both are cheap relative to the verified arms; `pathwise_stochastic`
is stage A.

**Frozen object touched.** None; additive.

---

## 2026-08-12 — Smoke split into stage A and stage B

**Decision.** Ship the five non-lookahead arms as stage A and the two verified
arms as stage B, as separate invocations.

**Evidence before the decision.** Cost model fitted to committed artifacts:
`retarget_prefixes_committed.json` (3 kernel calls, median 170.7 s) and
`retarget_calibration_result_3plus3_fixed.json` (26 calls, median 535.65 s)
give ≈123 s container startup and ≈15.9 s per kernel call at `cpu=2.0`. The
verified arms dominate that cost, and stage A alone resolves G1, G2, G3, G4 and
the budget-matched price of the mask.

**Consequence.** If G1 fails, stage B is never paid for.

**Frozen object touched.** None.

---

## 2026-08-12 — Arm policies extracted from the Modal function

**Decision.** Move all arm policies into
`src/compose_v4/experiments/pathwise_arms.py` as pure functions over an
injected `successors`/`utility` pair; the Modal app wires the frozen runtime
into them.

**Evidence before the decision.** As first written, the policies lived inside
the Modal entrypoint and could only have been exercised by the run they were
meant to produce. The retargeting lane's "no verified arm" defect — both arms
were secretly greedy for two full runs — is exactly what that arrangement
hides.

**Result.** 27 local tests now cover masking, dead ends, endpoint-only
selection, goal-free sampling and the verified controller against a synthetic
successor graph, plus 7 that drive the analysis script end to end.

**Frozen object touched.** None.

---

## 2026-08-12 — Deliverable directory uses the hyphenated name

**Decision.** `docs/workstreams/pathwise-constraints/`.

**Evidence.** The lead's mission specifies the hyphenated path; the workstream
plan's deliverables list (line 316) uses `pathwise_constraints`. The direct
instruction wins. Config, app, tests and diagnostics all use the underscored
`pathwise_constraints` form as the plan specifies. Flagged in `HANDOFF.md` in
case the lead wants them unified.

**Frozen object touched.** None.

---

## 2026-08-12 — NO MODAL RUN LAUNCHED (superseded by the authorisation below)

**Decision.** Stop at a costed plan and hand back for authorisation.

**Evidence.** Explicit standing instruction from the lead: build everything,
produce a costed smoke plan, report estimated container-hours, and wait.

**State at the time.** Zero Modal invocations.

---

## 2026-08-12 — Launch-time RDKit dependency removed from the app module

**Decision.** Move the arm names and stage partition into a dependency-free
`src/compose_v4/experiments/pathwise_arm_names.py`, imported by both the app
and `pathwise_arms.py`.

**Evidence before the decision.** The first launch attempt died locally with
`ModuleNotFoundError: No module named 'rdkit'`. `modal run` imports the app in
the LAUNCHER's interpreter (`/Users/rmaganti/.local/pipx/venvs/modal/bin/python`),
which has no chemistry stack; the app's module-scope
`from compose_v4.experiments import pathwise_arms` pulled RDKit in transitively
just to read five strings. **No Modal resources were consumed** — the failure
was local, before dispatch.

**Alternatives rejected.** Installing RDKit into the modal CLI venv — fixes one
machine, not the pattern. Restating the tuples in the app — silent drift
between the app and the tests.

**Verification.** The app now imports cleanly under the exact interpreter that
failed, with RDKit confirmed absent. Two new tests pin it:
`test_arm_names_module_is_dependency_free` (AST: the names module imports
nothing) and `test_app_module_scope_touches_no_chemistry` (AST: no rdkit /
torch / numpy / scipy at app module scope).

**Note.** The entrypoint already read a committed JSON panel rather than
computing molecules at launch, which is the pattern the main lane recommends;
the defect was narrower than that and is now closed.

**Frozen object touched.** None.

---

## 2026-08-12 — Resume-by-shard filtering placed in the on-Modal driver

**Decision.** `drive()` reloads the volume, skips indices whose shard already
exists, and reports what it skipped.

**Evidence before the decision.** Main lane lost work to a client-side DNS
failure at launch and warned that `--detach` is necessary but not sufficient.
Filtering inside `run_source` would make a resume pay a container start plus a
full checkpoint load per already-finished task — the expensive half — just to
discover it had nothing to do.

**Frozen object touched.** None.

---

## 2026-08-12 — STAGE A RUN, AUTHORISED AND EXECUTED

**Decision.** Launch stage A only: 6 held-in sources, 5 cheap arms, CPU only,
`modal run --detach`.

**Evidence before the decision.** Explicit authorisation from main lane after
review of PROTOCOL → DECISION_LOG → handoff.json → results, with stage B
withheld pending the G1 result.

**Execution.** App `ap-QJiLH8rlpGNGySTMLjOMPH`, verified `ephemeral (detached)`
in `modal app list` before proceeding. All 6 shards committed to the volume.
One container was preempted mid-run and Modal restarted it with the same
input; the shard landed normally. Actual cost **51 kernel calls / source**
(estimate 85) and **853 s / source** (estimate ~1 475), ≈ **1.4
container-hours** against a 2.5 h estimate. The 360-call circuit breaker was
never approached.

**Frozen object touched.** None. `R_theta`, kernel, operators, goal language
all unmodified.

---

## 2026-08-12 — G1 FAILED. Motif rule NOT changed. Lane stopped.

**Decision.** Record the failure, stop the lane, and change nothing.

**Evidence.** `diagnostics/pathwise_constraints_smoke.json`:

- `endpoint_valid_path_invalid` = **0/6** sources, **0/19** endpoint-valid
  rollouts. Preregistered threshold 0.10. **FAIL.**
- Absorption: of **42** rollouts on the unconstrained support, **19 broke** the
  protected motif and **0 recovered** by the end. Recovery rate **0.0**.

**Interpretation.** Motif destruction is *absorbing* under the frozen kernel at
H = 6. The operators can open a labeled ring system but effectively cannot
reconstruct one within the horizon. Endpoint validity therefore **implies**
path validity, which makes endpoint-only filtering sufficient as a matter of
dynamics rather than luck. The premise the lane rests on — that endpoint-only
handling returns molecules which traversed forbidden intermediates — is
empirically false in this regime.

**Alternatives rejected, explicitly and per the pre-committed anti-tuning
rule.** Enlarging the horizon; loosening the motif to a smaller or more
permissive pattern; switching to a different ring system; relaxing the label
semantics; re-drawing the panel. Every one of these would be searching for a
motif that makes the desired arm win. The protocol names this and forbids it,
and main lane reconfirmed it at authorisation.

**What this does NOT show.** That pathwise masking is useless in general — only
that at H = 6, on this panel, under this kernel, it changes nothing that
endpoint filtering would have caught. A kernel with reversible ring
open/close, or a longer horizon, could give a different answer. That is a
future question, not a repair to this run.

**Frozen object touched.** None.

---

## 2026-08-12 — The one positive finding, and why it is reported separately

**Decision.** Report "`endpoint_only` failed to return anything on 2/6 sources
while the masked arms succeeded 6/6" as a *distinct and weaker* claim, not as a
rescue of G1.

**Evidence.** `endpoint_only` `selection_failed` on sources 1 and 3: all 6 of
its rollouts ended motif-invalid, so the filter had nothing to select.
`pathwise_greedy` and `pathwise_stochastic` both reached `b_success` 6/6.

**Why it is admissible.** Free sign — nothing in the construction forced those
failures; `endpoint_only` could have matched the masked arms everywhere.

**Why it is not the lane's claim.** The designed claim was *excursion and
return*. This is *absorption plus wasted budget*: endpoint-only handling spends
its whole allowance outside the feasible set and returns nothing. Real, but
different and weaker. With 6 sources and 2 events it is an existence proof, not
a rate.

**Frozen object touched.** None.

---

## 2026-08-13 — PREDECLARATION of three reversible-constraint families (sealed before measurement)

**Decision.** Declare families A, B, C, their thresholds, their rank order and
the pass criteria, and **commit them before running any measurement**.

**Evidence available before the decision.** Only the ring-system result: that
violation of a protected fused ring system is absorbing (19 broke, 0
recovered). Nothing about how any of A, B or C behaves on any trajectory — no
family had been evaluated when this was written.

**Why the ordering matters.** The lane's whole credibility rests on not
choosing a constraint because it produces the wanted answer. Committing the
declaration first makes the ordering checkable in git history rather than
asserted in prose. The precedence rule (take the first passing family in
A, B, C — never the most favourable) is fixed now for the same reason.

**Threshold provenance, deliberately not my choice.**
- A: 14 standard literature reactive/structural alerts, chemically motivated,
  never checked against a COMPOSE trajectory.
- B: cLogP `[p25, p75]` **read at runtime** from the frozen
  `retarget_goal_language_normalizers.json`. Loading rather than transcribing
  removes any opportunity to nudge the bound.
- C: `HEAVY_ATOM_BAND = (18, 38)`, reused verbatim from the already-frozen
  eligibility band. No new number.

**Alternatives rejected.** Picking a cLogP corridor width by trying several —
that is precisely the failure mode named in the authorisation. Adding a fourth
family as insurance — the bound is the point of the lane.

**Instrument check performed before sealing (not a measurement of the
families).** All 14 alert SMARTS parse and fire on hand-written positive
controls; three clean molecules produce no false positives; both corridors load
from their frozen files.

**Frozen object touched.** None. The ring-system protocol, threshold, horizon
and source set are untouched and remain CLOSED.

---

## 2026-08-13 — Census computed from committed shards; no new compute for V1–V3

**Decision.** Compute metrics 1–3 from the stage-A shards already in the
repository rather than launching a run.

**Evidence before the decision.** Availability check on the committed shards:
42 unconstrained-support trajectories, 294 committed states, all complete at
H = 6. Enumerated successor sets are **not** stored, so mask support-retention
and mask-empty frequency (V4) are not derivable and would need new compute.

**Consequence.** V1–V3 cost nothing. V4 is deferred, and any family clearing
V1–V3 is reported as CONDITIONAL PASS pending V4 rather than as a full pass.

**Frozen object touched.** None.

---

## 2026-08-13 — CENSUS RESULT: no family passes. Lane stops.

**Decision.** Apply the frozen criteria as written, record FAIL for all three
families, and stop. No fourth predicate searched.

**Evidence.** `diagnostics/pathwise_reversibility_census.json`:

| Family | violators | returned | return rate | events | verdict |
|---|---:|---:|---:|---:|---|
| A undesired motif | 1/35 | 1 | 1.000 | 1 | FAIL (V3) |
| B cLogP corridor | 7/21 | 6 | 0.857 | 6 | FAIL (V3) |
| C size corridor | 0/42 | 0 | 0.000 | 0 | FAIL (V1) |

**What I am NOT doing.** B's return rate of 0.857 is far above the 0.10 floor
and its excursions are real, so it is tempting to argue the 20-event bar was
too high for a 42-trajectory pool. I am not making that argument as a decision.
The criteria were frozen before measurement precisely so that a near-miss
cannot be relitigated by the person who measured it. Recorded as FAIL; the
question of a larger pool is escalated to the lead, not resolved here.

**The failure modes differ, and the difference is reported.** C is vacuous —
heavy-atom count never left the frozen band. A and B are reversible and fail on
event count alone, which is a property of the inherited 6-source pool. Calling
all three "FAIL" without that distinction would misinform the lead.

**Frozen object touched.** None.

---

## 2026-08-13 — Excursion-depth diagnostic added after seeing B's return rate

**Decision.** Add `excursion_depth` to the census and rerun.

**Evidence before the decision.** Family B returned 6 of 7 violators. The
obvious artifact for a corridor constraint is that "violations" are tiny
excursions a hair past the bound, in which case "reversible" would be noise
about where the quartile happens to fall rather than chemistry.

**Disclosure.** This diagnostic was added AFTER seeing the return rate. It
changes **no threshold, no criterion and no verdict** — it only qualifies a
number that already existed. B remains FAIL.

**Result.** Excursions are real: median depth 0.700 logP units, 33.6% of the
corridor width; 0 of 7 below the 0.1-unit boundary-noise threshold. Example
trajectory (source 1, unconstrained greedy): cLogP 3.01 → 2.40 → 2.57 → **1.67**
→ 2.27 → 2.62 → 2.47 — a genuine departure below the corridor and a return.

**Frozen object touched.** None.

---

## 2026-08-13 — Stage A2 authorised, drafted, committed, and NOT launched

**Decision.** Build the A2 protocol, panel, runner, analyser and tests; commit
them; launch nothing.

**Evidence before the decision.** Main lane's distinction between the three
failures: the ring motif and family C failed on the *scientific property*
(absorbing or never occurring), while family B failed an *absolute event-count*
requirement with the causal mechanism plainly present (7/21 violated, 6/7
returned, excursions 33.6% of corridor width, none boundary jitter). With 21
relevant trajectories a 20-event bar demanded the phenomenon in nearly every
one, which is not the question the bar was standing in for.

**Deferral rationale, from main lane.** Every A2 trajectory would be generated
under the frozen `R_theta` that lane 1 may discard. Spending ~2.1
container-hours now risks measuring a model about to be replaced.

**Stage A verdict.** Unchanged: FAIL. Not retroactively revised.

**Frozen object touched.** None. Corridor, horizon, `R_theta`, violation and
recovery definitions, and rollout law are all carried over unchanged.

---

## 2026-08-13 — STAGE B designed and committed. NOT launched.

**Decision.** Build the stage-B protocol, panel, runner, analyser and tests;
commit them; launch nothing.

**Authorisation.** Lead authorised stage-B *design* after A2 passed 5/5. The
situation is materially different from rings: the frozen corridor produced 25
events across 8 of 12 sources with the criterion holding at the CI lower bound,
excursions of median 0.693 logP lasting a median 4 of 6 steps, and a mask that
leaves real room to act.

**Objective.** DRD2 potency alone, frozen in the retargeting lane. Chosen
because it is independently motivated and already frozen. **No objective search
was performed** — searching objectives is how a pathwise effect gets
manufactured.

**Arms.** Exactly four causal arms in a 2×2 of {where enforced} × {how
navigated}, dispatched from a declared `(mask, endpoint_only, controller)`
triple through one code path. `unconstrained_potency` is descriptive only. No
fifth causal arm: each extra arm is another comparison a reader must be stopped
from making.

**Frozen object touched.** None. Corridor, horizon, `R_theta`, kernel, goal
language and verified-controller strata all carried over unchanged.

---

## 2026-08-13 — PRIMARY estimand switched to the UNCONDITIONAL hidden-path rate

**Decision.** Preregister both forms, with the **unconditional** rate primary:

- PRIMARY `P(x_H ∈ C AND ∃t<H: x_t ∉ C)` — denominator is every eligible source;
- SECONDARY `P(∃t<H: x_t ∉ C | x_H ∈ C)` — reported **with its denominator,
  every time**.

**Evidence before the decision.** The conditional form's denominator is
**controller-dependent**: an arm that rarely delivers an acceptable endpoint
can post a dramatic rate on a handful of trajectories. That is the same failure
family this project has caught repeatedly — a quantity whose value is driven by
something other than the effect it appears to measure.

**Also fixed.** Both estimands are reported **separately for greedy and
verified** endpoint-only control and never pooled, because the controller
changes which endpoints become acceptable and pooling mixes two denominators.

**Verification that the distinction is real.**
`test_conditional_fraction_can_be_dramatic_on_a_tiny_denominator` builds a case
where 10 of 12 sources fail to deliver at all: the conditional fraction reads
**1.0** while the honest unconditional rate is **2/12 = 0.167**. The test
asserts the report exposes both and that the primary is the smaller.

**Frozen object touched.** Supersedes the conditional-only estimand named in
the first stage-B authorisation. Recorded rather than silently swapped.

---

## 2026-08-13 — Terminal cost: three outcomes declared informative in advance

**Decision.** Record, before the run, that little/no cost, moderate cost, and
large cost or frequent support collapse are **all informative**, and that none
is a failure.

**Why now.** Once numbers exist, whichever outcome appears will be tempting to
narrate as the intended one. Writing all three down first removes that freedom.
The protocol also states explicitly that **there is no expectation that
pathwise beats endpoint-only on potency** — the mask can only shrink the
reachable set, so framing a potency win as the goal would be suspicious.

**Frozen object touched.** None.

---

## 2026-08-13 — Support-tight handling predeclared from the A2 spread

**Decision.** SUPPORT_TIGHT = median retained legal-successor fraction < 0.10,
measured along the **descriptive** arm's states. All 24 sources stay in the
primary ITT; a sensitivity analysis excluding them is secondary; the threshold
is frozen and is not redefined after seeing which arm suffers.

**Evidence before the decision.** A2's per-source median retention spanned
0.048 to 0.917 — a spread the pooled 0.573 conceals, on a design whose unit is
the source.

**Why 0.10 is not a new number.** It is the V4a viability threshold already in
use since the A2 protocol.

**Why the descriptive arm.** Measuring retention along a constrained arm's own
path would only ever visit states the mask had already approved, making the
classification depend on the outcome it is meant to condition.

**Frozen object touched.** None.

---

## 2026-08-13 — STAGE A2 EXECUTED. PASSES 5/5. Lane stopped as instructed.

**Decision.** Launch A2 exactly as committed at `39f0ef1c`; change nothing;
report; stop.

**Authorisation.** Lane 1 veto cleared — `R_theta` is KEPT. Its cycling is
inherited from a locally reversible training reference process (mutual-edge
fraction 0.733 against a 0.33 threshold frozen before that census) and the
preregistered REOPEN conditions all failed. A2 was therefore no longer void.

**Execution.** App `ap-CH7z7NUbkOxv7mvU74HPRD`, verified `ephemeral (detached)`
before proceeding. **No client-side `timeout` wrapper** — lane 1 lost a run at
22/36 today because killing a wrapped client cancelled the detached job.
Progress was tracked by polling `modal app list` and the volume instead. 12/12
shards committed, 0 void, 0 failures.

**Result.** All five pre-committed criteria pass:

| Criterion | Threshold | Observed |
|---|---:|---:|
| V3 event yield | ≥ 20 | **25** |
| V4a median retention | ≥ 0.10 | **0.573** |
| V4b mask-empty | ≤ 0.05 | **0.0038** |
| V5a source spread | ≥ 0.333 | **0.667** (8/12) |
| V5b largest source share | ≤ 0.50 | **0.24** |

**Nothing was adjusted.** Corridor, horizon, `R_theta`, violation and recovery
definitions, rollout law, panel and all five thresholds are exactly as
committed before the run. Stage A remains permanently FAIL and A2 is not a
reinterpretation of it.

**What this does and does not establish.** A2 shows the reversible-excursion
mechanism is present on **new** sources, spread across **distinct molecules**,
and that the mask leaves a controller room to act — the last of which had never
been measurable. It does **not** independently establish that cLogP corridors
are special: family B was selected because it showed the effect, so A2 is
developmental follow-up by construction.

**Frozen object touched.** None.

---

## 2026-08-13 — Two caveats recorded against a PASS, because they bear on the next design

**Decision.** Report retention heterogeneity and residual absorption alongside
the PASS rather than only the headline.

**Evidence.** Per-source median retention spans **0.048 to 0.917**; the pooled
median of 0.573 passes V4a comfortably but conceals two sources (1 and 5) where
a masked controller would face a very tight choice set. Separately, 4 of 12
sources produced no event at all, and 3 of those had violating rollouts that
never returned — absorption still occurs for some molecules.

**Why record it.** A PASS on pooled statistics is the easiest place for a
per-source failure mode to hide, and the downstream experiment is source-level.
Reporting only the pooled median would set up exactly the surprise this lane
has spent three stages learning to avoid.

**Frozen object touched.** None.

---

## 2026-08-13 — V5a source-spread threshold set at 1/3, from downstream sizing

**Decision.** Require ≥1/3 of A2 sources to show at least one endpoint-valid /
path-invalid event, plus a companion cap (V5b) of ≤50% of events from any
single source.

**Derivation, deliberately not from the current numbers.** The eventual causal
experiment is source-level; a source with no possible excursion is
non-informative because both arms return the same molecule. Using this
project's own panel sizing (`RETARGETING_SAME_PREFIX_DESIGN.md`: 20–24
development, 60–80 confirmatory) and requiring ~20 informative sources for a
stable paired estimate: 60-source panel → `p ≥ 20/60 = 1/3`; 24-source panel →
`p ≥ 8/24 = 1/3`. Both routes land on the same number.

**Disclosure, because it would otherwise look fitted.** Stage A's family-B
events came from 2 of 6 panel sources = 0.333, numerically equal to this
threshold. The denominators are not comparable: stage A did not require `x_0`
inside the corridor and only 3 of its 6 sources were source-feasible. Under
A2's own definition — all sources source-feasible by construction — the
stage-A analogue is 2/3 = 0.667. **V5a is set at half the previously observed
value, not at it.**

**V5b rationale.** With ≥4 event-sources, one molecule supplying more than half
the events would leave the clustered bootstrap effectively at `n ≈ 1`.

**Frozen object touched.** None; new criteria for a new stage.

---

## 2026-08-13 — V4 thresholds set with no corresponding data in hand

**Decision.** Median support retention ≥0.10; mask-empty states ≤5%.

**Why these cannot have been fitted.** No cLogP-mask measurement exists
anywhere in this lane — stage A masked on the ring motif, not the corridor. The
thresholds come from what a downstream controller needs: a median state that
still offers a real choice set, and a dead-end rate low enough that the
experiment measures control rather than dead-end accounting.

**Frozen object touched.** None.

---

## 2026-08-13 — A2 requires the source to start inside the corridor

**Decision.** Add `cLogP(x_0)` inside the corridor to A2 eligibility.

**Evidence before the decision.** Stage A did not require it and lost half its
sample to it: 21 of 42 trajectories, and 3 of 6 sources, started outside the
corridor and dropped out of the denominator after the fact.

**Why this is applicability, not tuning.** A source that starts outside the
corridor can never leave and return, so it cannot produce the event being
counted. The corridor itself is unchanged. The criterion reads `x_0` only, so
it is outcome-independent. It makes every A2 source contribute.

**Direction of any residual bias.** Requiring `x_0` inside does not select for
proximity to the boundary, so it does not bias toward violation; if anything a
mid-corridor source is less likely to exit. Panel headroom is reported
(0.01–1.86 below, 0.23–2.07 above, corridor width 2.083) so the panel can be
checked for edge-stacking.

**Frozen object touched.** None.

---

## 2026-08-13 — `ArmContext` generalised to an injected feasibility predicate

**Decision.** Add an optional `feasible` callable to `ArmContext`, defaulting to
the existing SMARTS motif check.

**Evidence before the decision.** A2 needs the same rollout law under a
corridor predicate. The alternatives were to pass a dummy SMARTS (dishonest) or
to fork the policy code (which is how the retargeting lane's "no verified arm"
defect survived two runs).

**Verification.** All 30 pre-existing arm tests pass unchanged, and the
reversibility census output is byte-identical after the edit, so no earlier
verdict moved.

**Frozen object touched.** None; additive and backwards compatible.

---

## 2026-08-12 — External evidence folded in: GraphXForm is an endpoint-only comparator

**Decision.** Record the baseline lane's finding in `PROTOCOL.md` and
`HANDOFF.md` as context that raises, not lowers, the importance of G1.

**Evidence, from the baseline-qualification lane.** GraphXForm's action masking
is genuinely pathwise but covers only valence, atom type, atom count and
bonding legality. Its ring-size and bonding-pattern constraints are a
**terminal** filter (`molecule_evaluator.py::infeasible_by_special_constraints`,
asserted on `mol.synthesis_done`), and there is no SMARTS or substructure
matching anywhere in that repository.

**Consequence.** The strongest published graph-editing comparator instantiates
this lane's `endpoint_only` arm, not its pathwise arm. That makes the *design*
question well-posed and well-motivated. But it also means that with G1 failing,
the distinction between the two approaches is **academic in this regime**: if
unconstrained trajectories that break the motif never come back, then neither
COMPOSE's mask nor GraphXForm's terminal filter changes the returned molecule.
The comparator finding strengthens the framing and does not soften the gate.

**Frozen object touched.** None.
