# COMPOSE active handoff

Prepared 2026-09-16, first in the 06:44:57–07:44:57 UTC window and continued in a
second session that completed the repository-wide verification and closed both
learned-prior prerequisites. Read the final status section before launching anything.

## Start here

1. Read `AGENTS.md` completely, then `docs/T4_AUTONOMOUS_RESET.md`.
2. Read `diagnostics/t4_strategy_reset/20260916/REPORT.md` for the evidence-based
   strategy reset, historical comparisons, support probes and missing-data limits.
3. Inspect `diagnostics/t4_objective_reset/` for actual tests/launches/results.
   Do not infer a launch or successful docking from implementation alone.

Active branch: `t4-objective-dynamic-reset-20260916`.
Current worktree:
`/Users/rmaganti/compose_rgm_git/.worktrees/t4-objective-dynamic-reset-20260916`.
This branch is in the same Git repository as `/Users/rmaganti/compose_rgm_git`.
Do not reset the original tree: it contains unrelated, user-owned PMO changes.
The active checkout was moved here from `/private/tmp` with `git worktree move`
on 2026-09-16. This is a location-only change, not a scientific revision or merge.
Historical paths in frozen receipts retain their original provenance.

## Scientific direction

One objective-driven Dynamic controller, with a later small route-informed joint
proposal prior. Keep successful v0 mutation/recombination/composition and shared
archive refinement. Add large protected moves as an eligible-pool proposal lane.
Use actual measured feedback for search. Do not restart compiler/primitive beam
work or optimize structural recovery in isolation.

Revision 1 implements support repairs and parent-allocation comparison only.
Revision 1a freezes the split-clean corpus, 1b measures actual-sampler support.
**No learned route prior has been trained. No IVG win is claimed. Nothing has
been docked.**

## Durable data

Compact audit scripts, manifests and verified summaries are tracked here.
The lossless normalized table is tracked as `scored_rows.jsonl.gz` with a
round-trip hash manifest. The larger raw receipt mirror and uncompressed table
also remain at `/Users/rmaganti/compose_rgm_git/diagnostics/t4_strategy_reset/20260916/`.
Modal recovery paths and hashes are in the manifests. Do not synthesize missing
call indices or conflate checkpoints with completed runs. Full-146, Dynamic and
IVG endpoints are training/diagnostic evidence, not runtime initial routes.

## Fast commands

From this worktree, use `/Users/rmaganti/compose_rgm_git/.venv/bin/python` and
`PYTHONPATH=src`. Tests are zero-oracle. Modal profile must be `nitya`.

```sh
# every branch test file
/Users/rmaganti/compose_rgm_git/.venv/bin/pytest -q \
  tests/test_objective_program_search.py tests/test_t4_objective_reset.py \
  tests/test_t4_reset_handoff.py tests/test_t4_proposal_prior_dataset.py \
  tests/test_t4_proposal_access_probe.py

# rebuild the split-clean corpus (seconds, byte-stable)
PYTHONPATH=src:. python tools/t4_proposal_prior_dataset.py

# rerun the support probe (about 28 minutes; completed contexts are reused)
PYTHONPATH=src:. python tools/t4_proposal_access_probe.py

# local zero-oracle yield probe on one cell
PYTHONPATH=src python tools/t4_objective_reset.py probe --cell fa7_0 --rounds 4 \
  --output diagnostics/t4_objective_reset/new_probe/fa7_0
```

The support probe needs the raw checkpoint mirror at
`/Users/rmaganti/compose_rgm_git/diagnostics/t4_strategy_reset/20260916/raw`, which
is not tracked in git. It is bound by
`configs/t4_proposal_prior_checkpoint_manifest_v1.json` and recoverable with
`fetch_history.py` against the recovery manifests; fetch into a new directory and
verify, never overwrite the frozen mirror.

Do not rerun the scored pilot under a fresh run identity if it has already
launched. Read its launch receipt and durable query records first.

## Final status

- `20cdcc38`: historical forensic audit and compact evidence.
- `80dc1aa3`: immutable runtime implementation and experiment contract.
- `5b3e228d`: lossless 35,895-record scored evidence pack and read-only diagnostics.
- `5a650544`: split-clean proposal-prior corpus, frozen contract and coverage gate.
- Later handoff commits change documentation/tools/tests only. The Modal run's
  runtime files remain bound to `80dc1aa3`; do not relabel it with a later revision.
- **Scored pilot has NOT launched.** No docking call, no Modal launch, no model fit
  has happened in this branch. No IVG comparison is claimed.

### Repository-wide verification: completed, failing, and attributed

The earlier receipt was terminated at 54% and is preserved. The run has since
completed once: **5,396 tests, 5,225 passed, 110 failed, 58 errors, 3 skipped,
38m30s**. It is recorded as `completed_and_failed` with `launch_authority: none`.
A failing receipt is not authorization to bypass the boundary.

Read its `source_revision_disclosure` before citing it: the run started from HEAD
`52dcfcd9` with the corpus work still uncommitted, and those files were committed
as `5a650544` while it was in flight. It is a development-time verification
receipt, not a clean-committed-source launch gate, and must not be presented as
one.

Provenance is settled by measurement, in
`verification/full_suite_classification.json`:

- `additions_only: true`. Every executable change on this branch versus `2114c405`
  is a new file; the only modified file is `AGENTS.md`. No pre-existing module
  changed, so no pre-existing test's behaviour or import closure moved.
- `failing_nodes_in_branch_test_files: []`.
- `failures_whose_text_names_a_branch_added_file: {}`. Not one of the 168 failing
  nodes mentions any file this branch adds. The thirteen repository-scanning tests,
  the only class an addition could break, are listed for individual reading;
  `test_slot_safety` names nine pre-existing files and no branch file.

Causes, from each node's own error text:

| Bucket | Full run | Alone, on cleared bytecode |
| --- | ---: | ---: |
| RingCore catalog drift `82fd910c` vs frozen `639ff607` | 82 | 84 |
| torch grad-mode leak | 46 | **0** |
| Missing vendored artifact or module | 11 | 11 |
| Stale-bytecode source lookup | 1 | 0 |
| Unclassified | 28 | 25 |

48 of the 168 pass when run alone: cross-test state pollution in full-suite order,
a real repository problem but not a defect introduced here. The catalog drift is the
documented environment fact: this checkout is Python 3.12.9 with a newer RDKit,
while production is pinned to Python 3.11 / RDKit 2024.3.5. **The boundary as
written cannot be met in this environment**; meeting it needs the pinned image.
That is a decision for the owner, not something to waive.

One confound was found and removed: `tests/__pycache__` still carried the
pre-`worktree move` `/private/tmp/...` path as `co_filename`, so `linecache` could
not open it and `inspect.getsource` failed. The executed bytes were current. All
bytecode was cleared before the isolated recheck.

### Branch tests, lint

92 tests pass across the five branch test files (`focused.xml` 14, `handoff.xml` 5,
`milestone.xml` 73). Four production mutations were killed on each of the two new
modules. Lint: `src/` holds 513 legacy violations, unchanged from the baseline, with
**zero in any file this branch adds**; the whole worktree holds 4,274, disclosed so
the narrower number is not mistaken for it.

Note a coverage limitation rather than folding it into the counts: collection for
the full run began before `tests/test_t4_proposal_prior_dataset.py` and
`tests/test_t4_proposal_access_probe.py` existed, so those two are verified in
`milestone.xml` and are not part of the 5,396.

### The two prerequisites for a learned prior are now closed

`AGENTS.md` requires split-clean data and actual-sampler support to be verified
before any proposal-prior training. Both are done, zero-oracle, nothing fitted.

**Split-clean corpus — GO.** `configs/t4_proposal_prior_dataset_v1.json` frozen
before extraction; `diagnostics/t4_proposal_prior/dataset_v1/`. 35,895 rows admit
into 34,073 labelled constructions. No endpoint, source state, entry id or protocol
crosses a cell, so no lineage component crosses a leave-one-target-out fold. 18
construction families are dense in every fold, including all six the strategy report
names; `pendant_benzene` and `remodel_linker` live in one target only and are absent
from some fold's training side. Two limits that matter more than the row count: the
corpus holds only **133 lineage components**, so a held-out fold carries 16 to 44
independent genealogies and a weighted effective n of 65 to 676; and the frozen
equal-mass-per-lineage rule gives the heaviest record **454x uniform** mass. Both
were computed after the hierarchy was frozen and neither changed it.

**Actual-sampler support — PASS.** `configs/t4_proposal_access_probe_v1.json`;
`diagnostics/t4_proposal_prior/access_probe_v1/`. All fifteen declared families are
realized through the production sampler; `construct_substituted_ring`, the family
the strategy report measured as absent from all four frozen JAK2 panels, is realized
359 times. 198 to 255 distinct proposals per 256 attempts, every round drawing new
ones. All five supplied champions still realize exactly.

**The new negative, and it is the important one.** Eligible yield is ordered by each
cell's own root QED: parp1_0 0.888 gives 84 eligible, jak2_1 0.712 gives 55,
5ht1b_0 0.438 gives 16, braf_1 0.346 gives 5, fa7_0 0.284 gives **0**. FA7 produced
255 distinct proposals and 240 unique endpoints with zero executor rejections, then
failed the endpoint filter 239 times. On the pinned image all 34 recorded FA7
attempts fail for one reason, QED not strictly above 0.6, with similarity at 0.51 to
0.72. FA7's zero yield is the endpoint gate, not proposal support, and a wider
sampler will not fix it. FA7-0 is one of the three declared pilot cells, so on this
evidence it will likely spend its allowance producing nothing eligible. That is a
budget decision, not a reason to relax the gate. 256 draws is not an impossibility
proof.

## Modal identity and status command

- App: `ap-55DoTOih9uphyoMZQ2RI9v`
- Coordinator call: `fc-01M2MHDTF5S48DE527J8RERZCJ`
- Volume: `compose-t4-objective-reset-20260916`
- Output prefix: `bf58ef0f797d8bfa898052be336821c50105b2472ae970d1ba41277eed15af13`
- Durable local results: `diagnostics/t4_objective_reset/remote_probe/`.
- Launch receipt: `diagnostics/t4_objective_reset/launches/probe_bf58ef0f797d8bfa898052be336821c50105b2472ae970d1ba41277eed15af13.json`.

```sh
MODAL_PROFILE=nitya PYTHONPATH=src:. /Users/rmaganti/compose_rgm_git/.venv/bin/python tools/t4_objective_reset_status.py diagnostics/t4_objective_reset/launches/probe_bf58ef0f797d8bfa898052be336821c50105b2472ae970d1ba41277eed15af13.json --output diagnostics/t4_objective_reset/remote_probe/new_status.json
```

That command is read-only on Modal. For a future scored launch it also reports
v0/v1/v2.1/Full-146 scores at the exact same charged-call count, only when complete
verified historical indexing exists. It never substitutes a final score or a
delta=.6 result for a delta=.4 matched-call reference.

## Exactly where the next agent should pick up

1. **The launch boundary is resolved as far as evidence can resolve it, and it is
   red.** The suite completed, failed, and none of its failures belong to this
   branch. Do not rerun it here expecting a different answer and do not rewrite
   frozen files to clean up legacy lint. The open question is an owner decision:
   either accept the attributed receipt for this bounded development pilot under the
   2026-09-08 T4 development policy, or reproduce the pinned environment
   (python 3.11, rdkit 2024.3.5) and run it there. Exact command for the latter:

   ```sh
   KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src:. \
     .venv/bin/python -m pytest -q -rf -p no:cacheprovider \
     --junitxml=diagnostics/t4_objective_reset/verification/full_suite.xml
   PYTHONPATH=src:. .venv/bin/python tools/t4_reset_suite_classification.py \
     --full-suite diagnostics/t4_objective_reset/verification/full_suite.xml \
     --output diagnostics/t4_objective_reset/verification/full_suite_classification.json
   ```

2. **Fit the prior now, under a new frozen contract.** Both prerequisites are
   closed. The corpus binds program payloads by checkpoint hash and entry id rather
   than copying them, so the fitting step resolves them inside its own training
   fold; derive every vocabulary, feature moment and density after the split. Decide
   the weighting explicitly with the measured numbers above: equal mass per lineage
   component over-corrects on 133 components, so declare a capped or size-damped
   variant in that contract rather than inheriting this one. Scope the prior to the
   18 dense families; `pendant_benzene` and `remodel_linker` are not supported by a
   fold-clean split and a prior over them would be fitting one target.

3. **Judge it against the probe, not against route identity.** The generic sampler
   already realizes every declared family, so the prior's job is not coverage, it is
   putting probability on productive bindings and parameters. Rerun
   `tools/t4_proposal_access_probe.py` with the prior as a second arm on the same
   contexts, seeds and attempt budget, and compare conditional rank and yield.

4. **Decide FA7 before spending on it.** The 120-call two-arm pilot tests parent
   allocation on common repaired support; it does not test the unimplemented prior
   and must not be described as if it did. FA7-0 is measured to be gated by QED, not
   support. Reallocating its 40 calls, or keeping it as a declared negative control,
   is an owner decision that should be recorded before launch, not discovered after.

5. Follow the promotion/kill ladder in the contract and preserve every failure. No
   full benchmark is part of this implementation milestone.

## Original worktree and portability

The original worktree retains user-owned uncommitted PMO files and its AGENTS.md
edits. It is intentionally not globally clean. `NEXT_AGENT.md` there points here.
Do not overwrite it, reset it or merge blindly. This implementation branch is
local and unpushed; Git push was not part of the user authorization.

This checkout is now in the persistent project folder, not temporary storage.
Its Git history is stored in the parent repository's `.git` directory. The
parent ignores `.worktrees/`; the nested checkout has its own branch and index
and tracks its files normally. Do not copy its files over the PMO checkout or
merge the branches without reviewing overlapping work.

If another checkout is needed, use Git to create or move it into a persistent
directory. All implementation, compact evidence and compressed scored data are
committed in the same repository. A local worktree is not an off-machine backup;
the branch remains unpushed.
