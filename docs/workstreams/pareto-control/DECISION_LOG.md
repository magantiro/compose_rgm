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

**Superseded in part by D-002b**, below, before any census ran.

---

## D-002b · 2026-08-13 · The REAL successor fiber is enumerable locally; it becomes the primary instrument

**Decision.** Census C2–C4 on the real model-gated canonical successor support
(instrument I-A), with the MMP proxy retained as a cross-check (I-B). Taken
**before** any census statistic was computed.

**Evidence before the decision.**
- `canonical_successor_result(model, state, 0.5)` returns **334 and 339**
  canonical successors for two drug-sized held-in molecules at **5.2–5.7 s per
  state** on this machine, after a 63 s runtime build. That is the real
  ~250–600-wide fiber, not a proxy.
- The legal mask is structural — it comes from the model's action tables, not
  its weights — so the frozen `R_theta` checkpoint (which is only on the Modal
  volume) is **not needed to enumerate support**. The census reads support and
  properties only and never reads a probability.
- The local Active8 pairs with gate-zero **v7** while the volume's `RUN_PATHS`
  pins **v6**, which `scripts/verification/kernel_cost_profile.py` warns about.
  This was **checked, not assumed**: `process_identity_sha256`,
  `gate_zero_structural_contract_sha256`, `contracts_binding_sha256`,
  `active8_completion_sha256`, `enforced_structural_clauses`,
  `model_family_counts` and `executor_rule_counts` are **byte-identical**
  between the two decisions. They differ in corpus accounting, not in the
  executable support.

**Why this matters for the verdict, not just for precision.** Under D-002 alone,
a G2/G5 failure could only ever have been `PROVISIONAL_REJECT_PENDING_FIBER`,
because the MMP proxy's bias runs one way. With I-A the failure direction
becomes conclusive too, so the gate can actually reject a pair rather than defer
it. A gate that can only pass is not a gate.

**Alternatives rejected.** Keeping the MMP proxy alone (cheaper, but leaves
every rejection provisional); using the purely combinatorial
`enumerate_action_fiber` (78 successors under `editing_v2_semantic` — a
different and much narrower support than the model-gated one, so it would have
censused a set the controller does not actually see).

**Changes a frozen object:** no. Changes this lane's census instrument, before
the census ran.

---

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

## D-007 · 2026-08-13 · A G4 statistic of MY OWN had no falsifying range; withdrawn before any pair verdict

**Decision.** Withdraw `best candidate at a state reaches the pooled p99` as the
G4 saturation statistic and replace it with **single-objective greedy rollouts
at the full 6-edit budget**.

**Evidence before the decision.** The first census smoke printed
`G4 reach fractions {'P': 0.133, 'D': 0.933, 'S': 1.0}`. The `S: 1.0` prompted
the project-wide question — *what value could this take if the hypothesis were
false?* — and the answer is: essentially none. With a fiber of width `n`,
`P(max of n draws >= pooled p99) = 1 - 0.99^n`, which is **0.9973 at n = 589**.
The measured fiber width was 589–613. The statistic was going to read ~1.0
whatever the chemistry did.

This is defect shape number six of the same family, and the first one this
project caught in its own instrument *before* the run rather than after.

**The replacement, and why it is two-sided.** Reach fraction is now the fraction
of held-in sources whose 6-edit single-objective greedy rollout attains the
objective's ceiling, where the ceiling is analytic for O-D (the clipped soft-min
cannot exceed `1.32669`) and the held-in pool p99 for O-P. Both can plainly come
out anywhere in [0, 1]. O-S has no reachable ceiling at all — `T = 1` requires
zero edits — so it is tested for **inertness** instead: an objective that six
real edits cannot move is not controllable and degenerates the pair. Inertness
can equally well come out false.

**Alternatives rejected.** Keeping the statistic and caveating it in prose. The
project has five recorded instances of exactly that not working.

**Changes a frozen object:** yes — it changes the G4 operationalization declared
in `PROTOCOL.md` §6.3/§6.4. Timing matters and is verifiable in the history: the
change was made when only the *reach fractions* had been seen, before any
per-pair gate verdict was computed or printed. The G4 **threshold** (0.85) was
not touched.

**Consequence worth recording.** Under the corrected statistic, developability
does **not** saturate as a continuous Pareto axis, even though the committed
retargeting calibration reports greedy reaching the binary developability region
on 28/30 sources in three edits. Reaching the region is not the same as
exhausting the axis: the region boundary sits far below the clip ceiling. The
old binary reading would have rejected pair 1 for the wrong reason.

---

## D-008 · 2026-08-13 · The executable gate rejected MY OWN parity table

**Decision.** Demote contrast P1 (`greedy_pref` vs `unguided`) from PRIMARY to
`CONTEXT_ONLY`, and record in `PROTOCOL.md` §8 that it varies **controller and
objective**.

**Evidence before the decision.** `scripts/pareto_instrument_gate.py`, run as a
design-stage self-test with no results in existence, failed check
`D5_contrast_parity` and printed
`P1: varies ["controller", "objective"], intended "objective"`. The protocol
table had claimed P1 varied only the controller. It was wrong: an arm with no
controller cannot have an objective either, so the two are inseparable in that
contrast.

This is the same shape as the confound the main lane's audit found in
`verified_retarget` vs `continue_A`, which inflated its effect by roughly 11%.

**Alternatives rejected.** Deleting P1 (the unguided floor is genuinely
informative); redefining `unguided` to carry an objective (it would stop being
unguided). The parity-clean forms of the same question already exist as P3
(closed vs open loop, same objective) and P5 (objective only), so nothing is
lost by demoting P1.

**Changes a frozen object:** yes — `PROTOCOL.md` §8. Before any run.

---

## D-009 · 2026-08-13 · The two budget axes differ ~600x, so P3/P4 are reported as a bracket

**Decision.** Run `gen_rank` at **both** matchings — kernel-matched and
native-oracle-matched — and report P3/P4 as a bracket rather than a number.

**Evidence before the decision.** One kernel call yields ~600 candidate
molecules (measured: fiber width 589–613). So matching `gen_rank` on native
oracle calls hands it ~600x the closed-loop arms' kernel budget, which is the
hypervolume inflation channel wearing a benchmark convention's clothes; matching
on kernel calls gives it far fewer distinct molecules than the closed-loop arms
see, which understates it on the axis the multi-objective literature budgets.

**Alternatives rejected.** Picking one axis and calling it fair. Whichever is
chosen decides the winner, which makes the choice a result rather than a method.

**Changes a frozen object:** extends `PROTOCOL.md` §7.2. Before any run.

---

## D-006 · 2026-08-13 · Census verdict

*(Recorded after the census ran; see `diagnostics/pareto_tradeoff_census.json`.)*

Entry appended below once the census completed — deliberately left as a separate
commit from D-001..D-005 so the git history shows the predeclaration preceded
the measurement.
