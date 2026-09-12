# Parent-edit controller development

Status: local implementation checkpoint, 2026-09-12. Zero new oracle calls authorized
in this unit. The preceding second-generation results are retained unchanged in
`diagnostics/t4_second_generation/scoring_1/`.

## Identity and scope

The problem is choosing productive complete molecular transformations under a
query and proposal-work budget. The new output is a selection over complete
parent/program/attachment candidates, not a molecular reference law or a
long-horizon future value. COMPOSE's exact executor and frozen reference remain
the substrate. T4 support remains 40 heavy atoms, with endpoint-only task gates.
PMO must not inherit T4's QED, SA or original-seed similarity constraints.

The claim to test later is that learned edit allocation improves useful endpoint
density and best-one/top-ten utility at equal candidate support and work. The
score-blind complete-program sampler is the primary control. Parent-only
allocation and endpoint-only prediction are separate ablations. This local unit
does not establish that claim or authorize a benchmark run.

## Acceptance for the local implementation

1. Audit imported whole-route blocks. Extract only independently replay-verified
   branches with typed interfaces; keep full routes and report abstentions. A
   branch-donation check must preserve another branch and change the endpoint.
2. Add an explicit fresh-current-state proposal mode. Its primitive horizon is
   independent of ancestry; its original seed, ancestry, prior scores and work
   remain recorded. Old snapshots retain their old recipe.
3. Expose supported edits to existing atoms through an explicit proposal path,
   using production enumeration and execution, without changing chemical support.
4. Add task-specific utility/eligibility and best-k archive accounting. PMO
   initialization is task-independent and charged, not a prerequisite paid archive.
5. Implement modest completed-program prediction over parent, edit, attachment
   and size features, fit only actual labels, and retain a score-blind allocation.
   Ensemble disagreement is not calibrated uncertainty or a Doob value.
6. Focused tests protect branch execution, ancestry, input-atom access, task
   separation, score direction, model provenance and train/evaluation separation.

## Fixed first diagnostic

Reuse the first-generation four-target archives as fitting data and the already
inspected second-generation locked pools as chronological retrospective checks.
Never fit on second-generation repeats or labels before measuring that check.
Fit each target/oracle domain separately. Balance duplicate endpoint/program
representations. Use ridge regularization 1.0, five deterministic endpoint-level
bootstrap members, seed 20260912, no hyperparameter sweep. Predict completed
oriented utility, not parent improvement alone. Reserve 25% score-blind queries
in the proposed selector; compare selections on the identical fully scored pool.
This is development evidence, not a fresh holdout or prospective qualification.

Positive selection enrichment justifies a separately authorized prospective
paired run. A null or harmful result leaves score-blind selection in production
and directs the next change toward representation or observation handling.
Better branch completion without task improvement is not controller success.

## Pending experimental authority

The suggested four-task, three-seed, two-arm PMO comparison needs 24 runs and up
to 24,000 queries including initialization. No part is launched here. A new T4
assay also requires its own candidate lock, query/cost ceiling and clean source.
No reference training, hidden reward-component screening, or oracle alteration.

## Implemented, not yet performance-qualified

- `program_decomposition.py`: primitive dependency components are merged by
  actual replay writes, including neighboring atoms changed by deletion. Every
  admitted branch executes independently and the recombined endpoint matches.
  The bounded branch-donation check uses real archived programs.
- `ProgramSearchConfig.parent_edit_recipe()`: explicitly enables exact-current-
  state continuation, verified decomposition and a 0.25 current-state-edit
  probability within the mutation channel. The matched parent law is score-blind.
  Old defaults and saved artifacts remain unchanged. `niche_score` is a separate
  bounded exploration-parent ablation, not the reported incumbent objective.
- `edit_learning_data.py`: locked parent/A/B/A+B panels, missing-label abstention,
  repeat-mean descriptive interactions and deterministic small structural niches.
- `parent_edit_model.py`: v1 reproduces the original chronological diagnostic;
  v2 adds selected-parent size/graph differences, mutation events, branch and
  attachment changes and an explicit direct-execution mask. A rebuilt constructor
  is not mislabeled as an executed parent-to-child trace.
- `parent_edit_search.py`: generates a common candidate pool and locks its query
  subset before scoring, retaining the entire pool and work ledger. Learned
  selection remains diagnostic-only, with 25% uniform audit allocation.
- `program_campaign.py`: one reusable initialization/score/propose/select/update
  runner, with hard query reservation, receipt deduplication, identified update
  rules, completed-round resume and a declared stagnation stop. A failure remains
  charged and stops further queries. Interrupted *pending* rounds currently
  require explicit receipt-based recovery; automatic pending-round recovery is
  not implemented and must not silently re-run the oracle.
- `program_task.py`: task direction/gates, charged task-independent initialization
  locks, and PMO logging-interval trapezoidal top-ten AUC with separately labeled
  flat-tail completion. The metric follows the inspected primary
  [PMO evaluator](https://github.com/wenhao-gao/mol_opt/blob/main/main/optimizer.py#L27).
  A real benchmark launch still needs frozen oracle/version and initialization
  manifests; a synthetic runner test is not benchmark evidence.

## Decisions and current evidence

1. The initial fixed v1 fit used eight first-generation endpoints per target.
   It improved archive gain versus all equally sized uniform subsets in three
   pools, harmed two and tied three. The pools are correlated and previously
   inspected. Do not promote the selector or treat them as eight replications.
2. An operand-only branch splitter admitted 0/32 real constructions. The
   observed-write repair admits 20/32, with all admitted branches and joint
   reconstructions replay-verified. Keep all twelve abstentions. A bounded
   exchange between real BRAF programs preserved one other branch and produced
   a different executable endpoint; it has not been docked.
3. The reported BRAF attachment/extension example is a degenerate interaction
   panel: attachment-only equals the parent canonically, and extension-only
   equals the joint candidate. Existing first scores give interaction zero.
   It supports the extension, not attachment synergy. No new calls are needed
   to resolve this particular panel. Future contrast locks should prefer
   genuinely distinct canonical endpoints.
4. The next v2 models use 19/23/21/18 unique endpoints for JAK2/FA7/BRAF/5HT1B,
   respectively, with 23/29/27/22 genuine observations including repeats. They
   are training-only snapshots for future evaluation. Never reuse these pools
   as a held-out success test for v2. No best-repeat labels or cross-target
   docking-score transfer.
5. The focused controller/runner/dependency suite passed 44 tests. Lint and
   formatting passed for touched files. No full repository suite or milestone
   release sign-off is claimed. No new oracle calls or remote jobs.

Artifacts before final source binding live under
`diagnostics/parent_edit_controller/`: `attempt_1` retains the initial mixed
audit and failed operand-only decomposition; `branch_repair_1` retains the
first observed-write result; `branch_repair_2` includes real donation and the
degenerate BRAF contrast; `next_fit_1` retains the v2 training snapshots.
Each records physical inputs and implementation identities. The final committed-
source reruns will be named `committed_audit`, `committed_branches`, and
`committed_next_fit`; they are reproductions, not independent experiments.

```sh
PYTHONPATH=src:.:tests OMP_NUM_THREADS=1 python tools/parent_edit_diagnostic.py --output <new-audit-directory>
PYTHONPATH=src:.:tests OMP_NUM_THREADS=1 python tools/parent_edit_diagnostic.py --branch-audit-only --output <new-branch-directory>
PYTHONPATH=src:.:tests OMP_NUM_THREADS=1 python tools/parent_edit_diagnostic.py --fit-next --output <new-training-directory>
```

Use RDKit 2024.03.5 for these saved exact traces. These commands are local,
zero-oracle computations and refuse to overwrite an existing output directory.
The next performance decision requires new, budget-authorized paired learning
rounds, with both arms using the same repaired proposal support.
