# Workstream D — decision log

Every material decision, the evidence available *before* it, the alternatives
rejected, and whether it changes a frozen object.

---

## 2026-08-12 — Branch created from `04f1c46`

- **Decision.** `codex/compose-baseline-qualification` branched from `04f1c46`
  ("Add parallel workstream plan and agent handoff template") on
  `codex/editing-v2-successor-fiber-fastpath`.
- **Evidence before.** That commit is the tip carrying
  `docs/workstreams/PARALLEL_WORKSTREAMS_AND_HANDOFF.md`; the worktree's previous
  checkout predated it and had no workstream directory.
- **Alternatives rejected.** Branching from the older worktree HEAD — would have
  produced a handoff that could not be checked against the template it must follow.
- **Changes a frozen object.** No.

## 2026-08-12 — No Modal run, no smoke executed

- **Decision.** This lane produces environment specifications, adapter designs
  and costed smoke plans only. Zero jobs launched, zero heavy dependencies
  installed.
- **Evidence before.** Explicit instruction from the lead: CPU-only budget, and
  "DO NOT LAUNCH ANY MODAL RUN … produce costed plans for the 3–5 source held-in
  smokes and STOP."
- **Alternatives rejected.** Running the GraphGA smoke, which is cheap enough
  (~0.1 CPU-core-hours) that it was tempting. Rejected: the instruction is
  categorical, and a partial smoke set is worse than none because it invites the
  reader to compare a measured cost against four projected ones.
- **Changes a frozen object.** No.
- **Consequence to carry into the handoff.** Every cost figure in this lane is a
  **projection, not a measurement.** Labelled as such everywhere.

## 2026-08-12 — One source of truth for capability facts

- **Decision.** The full per-method qualification record lives only in
  `docs/workstreams/baseline-qualification/comparator_registry_v3.json`. The
  pre-existing `configs/comparator_registry_v3.json` is updated to carry the
  aggregate verdict plus a `qualification_record` pointer, and nothing else.
  `tests/test_baseline_qualification.py` asserts the two agree.
- **Evidence before.** Two registries already exist:
  `configs/comparator_registry_v1.json` holds a *schema-version-2* registry
  validated by `tests/test_comparator_registry.py`, and
  `configs/comparator_registry_v3.json` holds a *schema-version-3* claim-scoped
  plan that no code validates. Adding a third unvalidated copy of the same facts
  was the obvious drift risk.
- **Alternatives rejected.** (a) Replacing `configs/comparator_registry_v3.json`
  wholesale — it also covers internal arms this lane has no authority over.
  (b) Duplicating capability fields into both files — guaranteed drift, and the
  drift would be invisible.
- **Changes a frozen object.** No. `configs/comparator_registry_v1.json` and its
  validator are untouched.

## 2026-08-12 — `restart_at_supplied_state_under_new_objective` split out as its own axis

- **Decision.** The registry records "can the objective change mid-trajectory
  with the realized history preserved" and "can the method be relaunched from a
  supplied molecule under a different objective" as **two** capability axes.
- **Evidence before.** MARS verification showed the two answers differ:
  mid-run objective change has zero support in paper or code, while a relaunch
  from a supplied molecule under a new `Estimator` works with stock code. A
  single axis would have recorded one of those and hidden the other.
- **Alternatives rejected.** A single `dynamic_goal_switching` cell with a prose
  footnote. Rejected: footnotes do not survive into a rendered table, and this is
  precisely the distinction a reviewer will attack.
- **Changes a frozen object.** No.

## 2026-08-12 — `docs/RELATED_WORK_MATRIX.md` deliberately NOT edited

- **Decision.** Two cells in the related-work matrix are in tension with what was
  verified here. Neither was changed. Both are reported to the paper lane with
  the evidence and a recommendation.
- **Evidence before.**
  1. **MARS `pathwise = ✓`.** Verified: the official MARS code contains no
     masking, no SMARTS, no substructure matching and no atom freezing; the only
     `mask` symbols are training-loss masks. `break_bond` can delete a fragment
     constituting a motif we wanted preserved. Under the matrix's stated column
     definition ("constraints enforced at every committed state") that reads ✗.
  2. **DDSBM `var-card = ✗`, `birth/death = ✗`.** Verified: every molecule is
     padded to the dataset-wide maximum with a first-class dummy atom type `X`,
     and the uniform CTMC transition flips slots C→X and X→C, so effective
     molecule size varies inside a fixed tensor. That reads `~`, not `✗`.
- **Alternatives rejected.** Editing both cells directly. Rejected for two
  different reasons, and the asymmetry is the point:
  - The MARS correction would make **COMPOSE look more unique**, and it rests on
    a reading of an ambiguous column. The matrix may legitimately have meant
    "this method family affords pathwise enforcement because all its states are
    complete molecules", which is true of MARS. A change that favours us, on an
    ambiguous definition, is exactly the change we should not make unilaterally.
  - The DDSBM correction makes **COMPOSE look less unique** and should be made —
    but `docs/RELATED_WORK_MATRIX.md` is owned by the manuscript workstream and
    editing it from a parallel branch invites a silent merge conflict on a
    paper-bearing file.
- **Changes a frozen object.** No — and that is the reason.

## 2026-08-12 — DDSBM classified `CONTEXT_ONLY`, not `CONDITIONAL`

- **Decision.** DDSBM is related work and, at most, a native-protocol
  reproduction. It is not a matched comparator for any COMPOSE experiment.
- **Evidence before.** (a) The official repository has **no LICENSE file** and
  the GitHub API reports `license: null`, so it is all-rights-reserved by
  default. (b) No checkpoints are released; the README TODO is unchecked and
  open issue #1 asked for them without result. (c) Its intermediate states are
  noisy categorical graphs with no validity, connectivity or sanitisation check,
  so every pathwise metric is `N/A` by construction. (d) It makes zero oracle
  calls at sampling time, so an oracle-budget comparison is not defined.
- **Alternatives rejected.** `CONDITIONAL` gated on GPU authorisation. Rejected:
  the license gap is not a compute gate and cannot be resolved by us.
- **Changes a frozen object.** No.

## 2026-08-12 — The oracle-accounting asymmetry is recorded as a fairness blocker, not resolved

- **Decision.** The registry records, but does not settle, the fact that HN-GFN's
  1000-call budget is spent against a *surrogate* inside a Bayesian-optimization
  loop while COMPOSE queries the true oracle. The decision on how to report it
  belongs to the main workstream.
- **Evidence before.** HN-GFN's GFlowNet reward is the acquisition function over
  the surrogate (`main_mobo.py::_get_reward` → `self.proxy`), and true-oracle
  calls occur only at the 8 batch-evaluation points. The reverse holds for MARS,
  which scores every proposal including rejections with no cache — ~10^6
  molecule scorings per run at code defaults.
- **Alternatives rejected.** Picking a convention here. Rejected: the choice
  changes which method looks efficient, so it is a claim-level decision and this
  lane does not own claim-level decisions.
- **Changes a frozen object.** No.
