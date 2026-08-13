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
