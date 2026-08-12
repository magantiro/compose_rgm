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

## 2026-08-12 — NO MODAL RUN LAUNCHED

**Decision.** Stop at a costed plan and hand back for authorisation.

**Evidence.** Explicit standing instruction from the lead: build everything,
produce a costed smoke plan, report estimated container-hours, and wait.

**State.** Zero Modal invocations. Every artifact in this lane is
`DESIGN_ONLY` or a local source-only census.
