# Decision log — Workstream E, target-free Pareto / preference control

Every material decision, the evidence available *before* it, the alternatives
rejected, whether it changes a frozen object, and the commit that contains it.

---

## D-001 · 2026-08-13 · Lane opened on `codex/compose-pareto-control` from `a0e680d`

**Decision.** Base the lane on the main-lane HEAD `a0e680d` ("Add greedy-restart
completion arm; audit the recovery-fraction denominator") rather than on any
parallel lane branch, so the frozen goal language, the DRD2 oracle artifacts and
the committed retargeting calibration are all present and hash-identical to
main.

**Evidence before the decision.** `git log` on
`codex/editing-v2-successor-fiber-fastpath`; the four frozen-input hashes in
`PROTOCOL.md` §3 recomputed on this tree.

**Alternatives rejected.** Branching from `codex/compose-pathwise-constraints`
(would inherit an unrelated lane's uncommitted design surface).

**Changes a frozen object:** no.

---

## D-002 · 2026-08-13 · The census is run LOCALLY, not on Modal

**Decision.** Stage 0 runs entirely on this machine: RDKit + the frozen DRD2 SVM
+ one-cut MMP mining. No Modal run is launched.

**Evidence before the decision.**
- `artifacts/oracles/drd2_svm_v1/` is committed and loads in 0.5 s on CPU;
  `oracle.margin_many([smiles])` returns log-odds directly.
- `mine_one_cut_pairs` over the full held-in pool of 96,094 molecules costs
  **51.6 s** and yields **81,500** pairs, 30,976 molecules with >= 2 neighbours.
- The `R_theta` checkpoint is **not** in the repo — it lives on the Modal
  artifact volume at `/artifacts/editing_v2/r_theta_run/runs/run_v2_01/`, and
  `enumerate_factorized_marked_law(model, state, t)` needs the model even to
  produce the legal mask. So true successor-fiber enumeration is the *only*
  part of the census that cannot run locally.

**Consequence, recorded before results.** C3 and C4 are measured on the MMP
proxy, whose bias direction is known and one-sided (median neighbour degree 1
versus a ~500-wide real fiber, so it *understates* tradeoff availability and
distinguishability). `PROTOCOL.md` §6.2 therefore licenses a PASS on G2/G5 from
the proxy alone but only a `PROVISIONAL_REJECT_PENDING_FIBER` on a FAIL.

**Alternatives rejected.** Launching a Modal fiber census before the cheap local
one — it would cost hours to learn something the local instrument can rule in
for free, and the lane contract requires a costed plan to main first.

**Changes a frozen object:** no.

---

## D-003 · 2026-08-13 · Objective-pair order predeclared and committed BEFORE the census ran

**Decision.** The order is (1) potency vs developability, (2) potency vs source
similarity, (3) developability vs source similarity. The first pair passing all
five gates is adopted. No reordering after results.

**Evidence before the decision.** Only the mission contract and the retargeting
design document. Explicitly **not** consulted: any controller performance, any
census output.

**Alternatives rejected.** Choosing the pair after seeing which one makes the
controller look best — the failure mode `RETARGETING_SAME_PREFIX_DESIGN.md`
names as "engineered".

**Changes a frozen object:** no. Commit: this one, before the census commit.

---

## D-004 · 2026-08-13 · The unclipped-developability escape hatch is closed in advance

**Decision.** `z_D` uses the frozen clipped soft-min (`MARGIN_CLIP = 1.5`). If
O-D fails the saturation gate G4, this lane may **not** substitute an unclipped
soft-min or QED alone.

**Evidence before the decision.** The committed calibration
`retarget_calibration_result_3plus3_fixed.json` reports developability
`{at_switch: 5, greedy: 28, verified: 28}` of 30 sources in **three** edits.
That number was visible before the gate threshold was written, so G4's 0.85
ceiling is *not* a post-hoc threshold — it is a threshold chosen knowing the
committed evidence and stated anyway, which is the honest ordering: the reader
can see that O-D was expected to be at risk on G4 and that the threshold was
not moved to rescue it.

**Alternatives rejected.** Silently widening the objective until pair 1 passes.
That is exactly how "29/30 vs 28/30" became a reported result instead of a
recognised ceiling.

**Changes a frozen object:** no — it forbids a change.

---

## D-005 · 2026-08-13 · `gen_rank` is instantiated at two budget levels

**Decision.** `gen_rank@greedy` and `gen_rank@verified` are separate arm
instances.

**Evidence before the decision.** Observed cost from the committed calibration:
26 kernel calls per source for a greedy-style 3+3 protocol with an 8-candidate
lookahead, at ~22 s per kernel call. A verified 6-edit, 5-preference protocol
projects to ~600 kernel calls per source versus ~26 for greedy. One `gen_rank`
instance cannot hold native-oracle-call parity against two arms that differ ~23x
in cost.

**Alternatives rejected.** A single `gen_rank` at one budget — it would make
either P3 or P4 vary controller *and* budget, the exact parity defect the main
lane's audit caught.

**Changes a frozen object:** no.

---

## D-006 · 2026-08-13 · Census verdict

*(Recorded after the census ran; see `diagnostics/pareto_tradeoff_census.json`.)*

Entry appended below once the census completed — deliberately left as a separate
commit from D-001..D-005 so the git history shows the predeclaration preceded
the measurement.
