# COMPOSE active handoff

Prepared during the 2026-09-16 06:44:57–07:44:57 UTC work window.
Read the final status section before launching anything.

## Start here

1. Read `AGENTS.md` completely, then `docs/T4_AUTONOMOUS_RESET.md`.
2. Read `diagnostics/t4_strategy_reset/20260916/REPORT.md` for the evidence-based
   strategy reset, historical comparisons, support probes and missing-data limits.
3. Inspect `diagnostics/t4_objective_reset/` for actual tests/launches/results.
   Do not infer a launch or successful docking from implementation alone.

Active branch: `t4-objective-dynamic-reset-20260916`.
Current worktree: `/private/tmp/compose-t4-objective-dynamic-reset-20260916`.
This branch is in the same Git repository as `/Users/rmaganti/compose_rgm_git`.
Do not reset the original tree: it contains unrelated, user-owned PMO changes.

## Scientific direction

One objective-driven Dynamic controller, with a later small route-informed joint
proposal prior. Keep successful v0 mutation/recombination/composition and shared
archive refinement. Add large protected moves as an eligible-pool proposal lane.
Use actual measured feedback for search. Do not restart compiler/primitive beam
work or optimize structural recovery in isolation.

Revision 1 implements support repairs and parent-allocation comparison only.
**No learned route prior has been trained in this revision. No IVG win is claimed.**

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
/Users/rmaganti/compose_rgm_git/.venv/bin/pytest -q tests/test_objective_program_search.py tests/test_t4_objective_reset.py
PYTHONPATH=src /Users/rmaganti/compose_rgm_git/.venv/bin/python tools/t4_objective_reset.py probe --cell fa7_0 --rounds 4 --output diagnostics/t4_objective_reset/new_probe/fa7_0
```

Do not rerun the scored pilot under a fresh run identity if it has already
launched. Read its launch receipt and durable query records first.

## Final status

- `20cdcc38`: historical forensic audit and compact evidence.
- `80dc1aa3`: immutable runtime implementation and experiment contract.
- `5b3e228d`: lossless 35,895-record scored evidence pack and read-only diagnostics.
- Later handoff commits change documentation/tools/tests only. The Modal run's
  runtime files remain bound to `80dc1aa3`; do not relabel it with a later revision.
- 14 focused runtime tests passed, plus five handoff/cross-lane/resume/comparison
  tests. Touched-code lint passes. Tests use clearly synthetic fixture labels,
  not fabricated benchmark scores.
- Five pinned Modal probes completed with **zero oracle calls**. Eligible endpoints:
  5HT1B 5, BRAF 1, JAK2 3, PARP1 8, FA7 0. FA7 used 34 fresh attempts over four
  bounded rounds, 92.6 seconds. The old repetition defect is repaired; FA7 yield
  is not solved. Candidate counts are not docking results.
- Conditional known-answer JAK2 ring probe: 64 distinct fresh panels, 803 unique
  endpoints, zero exact teacher-ring hits. This diagnoses weak parameter mass;
  it is not an impossibility proof or an autonomous full-route test. Do not widen
  the random sweep as the default response.
- **Scored pilot has NOT launched.** Repository-wide verification is not green.
  The recorded lint check has 513 violations, none in changed runtime source
  files. Broad-test outcome is recorded under `verification/`; a failed or
  incomplete receipt may not be passed to the scored launcher.
  The broad run was stopped after failures/errors and 54% reported progress;
  its exit code is 143, not a complete test result. See
  `verification/full_suite_execution.json`. It is no longer running. The initial
  collection-only PYTHONPATH error is preserved separately. Do not claim all
  broad-suite failures have been diagnosed or are proven irrelevant.
- No learned route prior was fitted. No new IVG comparison or docking improvement
  was obtained. Existing PMO work and old experiments were untouched.

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

1. Inspect the verification receipt and failure report. Do not repeatedly rerun
   the whole repository or rewrite frozen files to clean up unrelated legacy
   lint. Resolve the scoped launch boundary explicitly; a failed receipt is not
   authorization to bypass it.
2. Continue **this one controller**, not another large macro-model branch. The
   immediate scientific work is a small joint, context-conditioned program/
   binding/parameter prior from the recovered measured data, with source/lineage
   splits frozen first. Use known-good transformations as diagnostic probes.
3. Require actual-sampler support and probability/rank checks before training or
   scoring. The JAK2 fresh-panel result demonstrates that changing RNG alone is
   not adequate distillation. The exact complete-region compiler stays unchanged.
4. The existing bounded two-arm 120-call proposal tests search allocation, not
   the unimplemented learned prior. Do not silently add a third arm or claim
   these support repairs match Full-146. Follow the promotion/kill ladder in the
   contract and preserve every failure.
5. Once a learned prior has demonstrated useful support, add it to this same
   shared archive/v0 refinement path under a new frozen revision. Judge it by
   matched-call docking utility, not route identity. User has authorized bounded
   Modal compute; no full benchmark is part of this implementation milestone.

## Original worktree and portability

The original worktree retains user-owned uncommitted PMO files and its AGENTS.md
edits. It is intentionally not globally clean. `NEXT_AGENT.md` there points here.
Do not overwrite it, reset it or merge blindly. This implementation branch is
local and unpushed; Git push was not part of the user authorization.

If this temporary worktree disappears, recreate the branch in a new directory
with `git worktree add <new-directory> t4-objective-dynamic-reset-20260916` after
resolving any stale worktree registration. All implementation, compact evidence
and compressed scored data are committed in the same repository.
