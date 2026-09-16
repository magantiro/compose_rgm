# COMPOSE active handoff

Updated during the 2026-09-16 06:44:57–07:44:57 UTC work window.
The final status section below will be filled with exact commits, checks and
Modal call IDs before handoff. Until then, work is in progress.

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
The original recovered raw receipt mirror and 47-MB normalized scored table live
at `/Users/rmaganti/compose_rgm_git/diagnostics/t4_strategy_reset/20260916/`.
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

Pending final update at the agreed one-hour handoff.
