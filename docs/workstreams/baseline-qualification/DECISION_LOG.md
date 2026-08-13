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

## 2026-08-12 — The "DO NOT implement during the C1 work" guard was removed from the external rows

- **Decision.** Each external comparator row in `configs/comparator_registry_v3.json`
  carried the note "Relevant only for optimization. DO NOT implement during the C1
  work." That sentence was dropped when the rows were rewritten.
- **Evidence before.** The guard existed to stop the Claim-1 workstream from being
  pulled into external-baseline plumbing before the reference-law result existed.
  `docs/EXPERIMENT_PLAN.md` records Experiment 1 as **done developmentally**
  (1.46 nats over uniform legal rewriting on the frozen matched reserve), and
  `docs/workstreams/PARALLEL_WORKSTREAMS_AND_HANDOFF.md` creates this lane
  specifically to qualify those adapters. The condition the guard protected has
  therefore expired.
- **Alternatives rejected.** Keeping the sentence. Rejected: it now reads as a
  standing prohibition on the very work this lane was created to do, and a stale
  prohibition is worse than none. The substantive half of the old note — that an
  external method says nothing about whether `R_theta` learned the declared
  successor law — was **kept verbatim**, because that half is still true and is
  the reason externals are `N/A` for C1, C2 and C3.
- **Changes a frozen object.** No.

## 2026-08-13 (main-lane decisions) — HN-GFN runs native; the retargeting claim becomes operational

- **Decision 1, adopted from this lane's recommendation.** HN-GFN runs under its
  **native published linear scalarization**. We do not supply a Chebyshev to
  match Lane 4's. Its official augmented-Tchebycheff mode may appear as a clearly
  labelled *secondary* configuration, never as a replacement for its default.
- **Evidence before.** Verified in source: `raw_reward = (weights*score).sum()`
  under `--scalar WeightedSum`, the argparse default; the MOBO entrypoint ignores
  `--scalar` and defaults to a linear UCB acquisition; the opt-in `Tchebycheff`
  branch is an augmented max-min with no ideal point and a hardcoded 0.1, so it
  is not classical Chebyshev either.
- **Alternatives rejected.** Supplying a Chebyshev to HN-GFN. Rejected on the
  principle that modifying a competitor's internal formulation to match ours is
  not fairness — it is changing the competitor, and a reviewer reads it that way.
- **What replaces it.** Comparison in a **common outcome space**: hypervolume,
  Pareto coverage, nondominated-set quality, preference coverage, and oracle
  usage under all three counters. Objectives and budget are matched; the
  scalarization is not, and is reported. **Binding reporting rule:** linear
  scalarization cannot recover concave front regions, so any coverage difference
  there is a **method property**, never a COMPOSE win.

- **Decision 2.** The phrase **"without retraining" is barred project-wide**, on
  the strength of this lane's REINVENT 4 source reading. The claim is now
  operational: *COMPOSE performs inference-time intervention on an explicit
  realized molecular STATE while all learned parameters remain fixed; changing
  the goal changes only the control computation. REINVENT carries forward a
  learned policy and its optimizer state and keeps updating it. COMPOSE carries
  forward the actual molecule `x_3` and performs zero parameter updates.*
- **Why the new wording is better.** It is checkable rather than definitional.
  "Zero parameter updates" is auditable from a code path; "without retraining"
  is a claim about what counts as training, which REINVENT's warm optimizer
  makes arguable.
- **How drift is prevented.** The wording lives in the registry JSON, renders
  into `COMPARATOR_MATRIX.md`, and `validate_qualification_registry` **rejects a
  registry whose operational claim contains the barred phrase**
  (`tests/test_baseline_qualification.py`). The Mol2Mol point — similarity
  anchor, not a continued state — sits beside it, as does the MARS naming rule.
- **Changes a frozen object.** Yes, deliberately: it replaces the wording of a
  paper-bearing claim, and the guard makes the replacement enforceable.

## 2026-08-13 (later still) — Hold lifted; real implementations, frozen suite, fairness matrix

- **Decision.** Vendored the **real** upstream GB-GA byte-identical, built an
  adapter that drives it under three-counter accounting, smoked it on 5 held-in
  sources, froze a five-task conventional suite, and produced the method x task
  fairness matrix. Environments for the other five methods were probed in
  throwaway `/tmp` venvs and reported honestly.
- **Evidence before.** The main lane lifted the hold: Lane 1 step 1 found the
  training reference process is itself locally reversible (mutual-edge fraction
  0.733 across 151,059 teacher transitions against an inherited threshold of
  0.33), so `R_theta`'s cycling is faithful modelling, and most of this lane's
  work is model-independent anyway.
- **What running the real code changed, that reading it could not.** Three
  findings, none visible from the paper or README:
  1. `crossover.average_size` / `crossover.size_stdev` are **undocumented
     required globals**. Unset, `mol_OK` raises `NameError` into a bare
     `except`, every candidate is rejected, `crossover` always returns `None`,
     and `reproduce()`'s unbounded `while` loop spins forever **with no error**.
     They are also a soft size prior, hence a fairness parameter: panel-derived
     versus upstream ZINC moved mean endpoint heavy atoms 14.65 to 16.25.
  2. **An oracle budget does not bound GB-GA's runtime.** Failed crossovers cost
     no oracle calls. A wall-clock guard is mandatory.
  3. **GB-GA cannot run task T2 at all.** Its roulette selection needs a
     non-negative objective (upstream's own `logP_max` clamps for this reason);
     the COMPOSE developability margin is negative for the entire initial
     population, so the clamp zeroes everything and
     `calculate_normalized_fitness` raises `ZeroDivisionError`. Measured, and
     recorded in the artifact as `upstream_clamp_probe` rather than dodged.
- **Alternatives rejected.** (a) Making T2 run for GraphGA with a constant shift
  and reporting the number. Rejected: the shift changes the fitness ratios
  roulette selection acts on, on a task the scoping already says is not
  claim-bearing. `inappropriate` is the honest cell. (b) Vendoring
  `scoring_functions.py` and its 4.5 MB of ZINC data. Rejected: it is the paper's
  own objective, not the algorithm; a faithful `calculate_scores` shim is
  reproduced verbatim instead, and `tests/test_graph_ga_adapter.py` asserts the
  three vendored algorithm files still hash to upstream.
- **Two verifications that weakened our own position, recorded because they do.**
  REINVENT 4 carries the **Adam optimizer state and the inception replay buffer**
  across a stage boundary and has **no LR scheduler at all**, so "the baseline
  must retrain" is too weak a distinction; and HN-GFN is a **linear** weighted
  sum while Lane 4 froze **Chebyshev**, so the two do not match.
- **Changes a frozen object.** Yes: it creates the frozen conventional suite and
  the fairness matrix, both frozen before any COMPOSE-versus-baseline outcome.

## 2026-08-13 (later) — Counters renamed and split three ways; the naming was misleading

- **Decision.** The two-counter scheme below is **SUPERSEDED**. `raw_compute` was
  a misleading name, because a duplicated request increments it even when the
  cache serves it — that is *demand*, not compute. The frozen scheme is now:
  - `unique_valid_canonical_evaluations` — the PMO-compatible, benchmark-native
    number (was `benchmark_native`);
  - `oracle_requests` — every scoring request the algorithm makes, including
    duplicates and rejects. **Algorithmic demand, not CPU** (was `raw_compute`);
  - `evaluator_calls` — expensive oracle executions actually performed after
    caching. **Real work.**
- **Evidence before.** The lead's reading of the shipped harness: the counter
  named "compute" was not measuring compute, and a reader would take it as CPU.
- **What was kept.** The design call that a cache hit still increments the demand
  counter, which the lead confirmed was correct and is precisely why the counter
  needed an honest name. It is now stated as a conceptual invariant in the
  fairness contract: *caching may reduce evaluator work, but it cannot erase
  wasteful algorithmic requests.* The required per-method ratio is kept and
  renamed to match what it divides:
  `oracle_requests / unique_valid_canonical_evaluations`.
- **What the rename forced beyond a find-and-replace.** With the old code,
  disabling the cache made `benchmark_native` count every *valid* request rather
  than every *distinct* one, so under the new name it would no longer have been
  counting unique molecules. Uniqueness is now tracked in a `_seen` set
  independent of the score cache, and a cache-disabled control run is part of the
  stress test: same stream, `oracle_requests` 288 and
  `unique_valid_canonical_evaluations` 120 unchanged, `evaluator_calls` 120 → 262.
- **Changes a frozen object.** Yes — it replaces a scheme frozen the same day,
  before any comparator consumed it.

## 2026-08-13 — Dual oracle accounting frozen; the blocker is resolved

> **SUPERSEDED** by the entry above. The decision to freeze and dual-log stands;
> only the counter names and the split into three changed. Names below are
> historical.

- **Decision.** The main workstream froze **both** counters:
  `benchmark_native` (unique valid canonical molecules scored, used **only**
  against published PMO numbers) and `raw_compute` (every invocation including
  duplicates, rejects, invalids and rescores, used for **all** efficiency
  claims). Neither may be substituted for the other. Implemented and enforced in
  `src/compose_v4/experiments/oracle_accounting.py`.
- **Evidence before.** This lane had recorded the two conventions as
  incompatible and declined to choose, because the choice changes the ranking and
  is claim-level. The instrument check then quantified it: on one benign
  candidate stream a nominal budget of 120 bought 120 distinct molecules under
  `benchmark_native` (288 invocations) and 59 under `raw_compute` (120
  invocations).
- **Alternatives rejected.** Picking one convention. Rejected by the main lane
  for the reason this lane flagged — either single number is misleading in a
  predictable direction, and logging both removes the opportunity to pick the
  flattering one after seeing results.
- **Design consequence.** A duplicate *request* increments `raw_compute` even
  though the cached evaluator is not called. That is deliberate: `raw_compute`
  measures what a method *demands*, so adding a cache to a wasteful method does
  not silently improve its efficiency number. `evaluator_calls` records actual
  CPU separately.
- **Changes a frozen object.** No. It creates one.

## 2026-08-13 — Matrix corrections merged after lead approval; MARS still untouched

- **Decision.** `docs/RELATED_WORK_MATRIX.md` amended on this branch: DDSBM
  `var-card` and `birth/death` `✗ → ~`; new rows for GraphXForm (`pathwise ~`,
  `complete ~`, `birth/death ✗`) and REINVENT 4 (`retarget ~`); a sentence for
  every new `~`; `loeffler2024reinvent4` appended to
  `paper_iclr_stochastic_rewriting/references.bib`. **The MARS `pathwise ✓` cell
  was not changed.**
- **Evidence before.** The per-cell evidence recorded in the qualification
  registry, plus explicit lead approval to merge, which removes the
  parallel-branch-ownership objection that blocked this on 2026-08-12.
- **Why MARS still stands.** The earlier reasoning is unchanged and was
  re-confirmed by the lead: the released MARS code has no masking, no SMARTS and
  no atom freezing, so under a *released-code* reading the cell is `✗`; but every
  MARS state is a complete molecule, so under an *affordance* reading it is `✓`.
  The change would flatter COMPOSE on an ambiguous column definition. Verification
  debt item 4 now records the ambiguity and notes that MARS, MIMOSA, Graph GA and
  Kappa-style rewriting all sit on it and must be decided together.
- **Changes a frozen object.** Yes — a manuscript-bearing file, with approval.
  Flagged in `STATUS.md` and `HANDOFF.md` for the manuscript lane to merge.

## 2026-08-13 (later) — The artifact is renamed so its headline cannot mislead

- **Decision.** The run is titled **ORACLE-ACCOUNTING HARNESS STRESS TEST**, not
  "GraphGA smoke", and the rename was applied at the **file** level:
  `scripts/baseline_accounting_smoke.py` → `scripts/oracle_accounting_stress_test.py`,
  and `diagnostics/baselines/graph_ga_accounting_smoke.json` →
  `diagnostics/baselines/oracle_accounting_harness_stress_test.json`.
- **Evidence before.** The lead's point, which the earlier version got wrong: the
  artifact disclaimed the GraphGA reading internally, but a reader who sees
  "GraphGA smoke — PASS" in a status file carries the wrong belief regardless of
  what the JSON says three levels down. A filename is a headline.
- **Alternatives rejected.** Keeping the filename and strengthening the internal
  disclaimer. Rejected for exactly the reason above.
- **Also recorded.** The three preconditions for any claim-bearing GraphGA
  comparison now appear in the artifact, the registry, `STATUS.md` and
  `HANDOFF.md`: upstream implementation vendored; production RDKit pin 2024.3.5
  or an isolated environment reconciled against it; the real algorithm
  terminating against every counter.
- **Changes a frozen object.** No.

## 2026-08-13 — The accounting smoke is an instrument check, not a GraphGA measurement

> Superseded in **naming only** by the entry above; the reasoning stands.

- **Decision.** The authorised "GraphGA accounting smoke" was run as an
  instrument check on the shared accountant, driven by a GA-*shaped* candidate
  stream (BRICS recombination, survivor rescores, alternate SMILES spellings,
  malformed strings) over 5 held-in sources. **Upstream GB-GA was not vendored
  and nothing external was installed.** Verdict PASS, well under a CPU-second.
- **Evidence before.** The instruction was explicit that the smoke's job is the
  plumbing — oracle wrapper, canonicalization, duplicate treatment, both
  conventions, budget termination — and *not* to measure GraphGA; combined with
  the standing prohibition on installing external dependencies.
- **Alternatives rejected.** Downloading GB-GA to make it a real GraphGA run.
  Rejected: it would install external code this lane is not authorised to
  install, and it would not test anything more about the accountant. The cost of
  the omission is that the GB-GA *algorithm* remains unexercised, which is
  recorded rather than glossed.
- **Honest limitation recorded in the artifact.** The run used rdkit 2025.09.6,
  **not** the production pin 2024.3.5, so its canonical keys must not be reused
  for a claim-bearing comparison. The artifact carries
  `pin_matches_production: false` and a warning.
- **Changes a frozen object.** No.

## 2026-08-12 — C4a demoted to a competitiveness sanity check after the main lane's calibration result

- **Decision.** The static-optimization comparison (C4a) against MARS,
  GraphXForm, GraphGA and REINVENT is classified as a **competitiveness sanity
  check, not a claim-bearing test.** The classification of those methods as
  `MUST_RUN` is unchanged — what changed is what a result there may be read as.
- **Evidence before.** Supplied by the main lane and cross-checked against
  `docs/RETARGETING_SAME_PREFIX_DESIGN.md`: on a target-free bounded
  developability goal, greedy reaches the region 28/30 and verified
  remaining-budget control also 28/30 — binary headroom **0**, gate CLOSED,
  subclaim B dropped. The continuous-utility advantage (+0.0361 mean, higher on
  23/30, lower on 0) has a sign guaranteed by the policy-improvement theorem, so
  only its magnitude is admissible: +3.2% of typical post-switch movement, never
  enough to flip a success. The earlier C0 probe on DRD2 was negative for the
  same reason. Meanwhile the sealed exact-target result stands at greedy 26/65
  vs verified rollout 40/65.
- **Why this matters here specifically.** PMO's DRD2/GSK3β/JNK3 tasks *are* easy
  target-free property goals. Without this scoping, a strong MARS or REINVENT
  result on them would be read — by us or by a reviewer — as refuting
  "future-aware control helps", a claim COMPOSE does not make for that regime;
  and a COMPOSE win would be oversold as supporting it.
- **Alternatives rejected.** (a) Demoting the four methods to `CONTEXT_ONLY`.
  Rejected: they are still the right comparators for competitiveness, and PMO's
  finding that simple methods beat newer ones under controlled budgets is exactly
  why the check is worth running. (b) Selecting a harder comparison task on which
  greedy fails, so that the C4a table becomes claim-bearing. Rejected as a direct
  violation of the anti-tuning rule, which this lane now inherits explicitly.
- **Changes a frozen object.** No. Verdicts and adapter statuses are unchanged;
  only the interpretation contract is added.
- **Recorded in.** `compose_claim_scoping` in the qualification registry,
  `FAIRNESS_CONTRACT.md` §0a, `PROTOCOL.md`, and enforced by
  `validate_qualification_registry`, which now refuses a registry that carries
  only the flattering half of the scoping.

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
