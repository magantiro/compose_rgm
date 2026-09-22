# Canonical T4 shared-controller run — how to resume, reconcile and read it

Run id `3960cba7d11e2d9dddcbe0a56d8fd886373d59a6a654a48be18ecf4791db3721`
App / volume `compose-t4-canonical-shared-controller` (Modal profile `rahul-94866`)
Mount `/t4_canonical`, layout `<run_id>/<arm>/<cell>/{round_NNN_lock.json, checkpoint.json, result.json}`

The run does not depend on any local process. Cells were spawned as independent
calls against the DEPLOYED app; each carries `retries=8` and resumes from its own
checkpoint, so preemption costs at most one round and never the panel.

## Resume

    python3 tools/launch_t4_canonical.py --mode resume \
        --run-id 3960cba7d11e2d9dddcbe0a56d8fd886373d59a6a654a48be18ecf4791db3721

Re-spawns exactly the cells with no `result.json` — including the ten arm-C
delta=0.4 cells deliberately deferred (see `QUEUE_ORDER_NOTE.md`). It is safe to run
repeatedly: a cell that already has a terminal result returns it without charging
anything, and a cell mid-flight is resumed from its last completed round.

`--mode resume` re-runs the full `verify` gate first and REFUSES if any pinned
runtime input has moved, so a resume cannot silently run different code than the
launch did. Use `--dry-run` to see what it would re-spawn without spawning it.

**Why it is safe to run while cells are still going.** A running cell has no
`result.json`, so "no result" alone is NOT a licence to re-spawn: two containers on
one cell would write the same round locks and the same checkpoint from two different
search states. Resume therefore skips a cell whose most recent spawned call has not
terminated, established by GETTING the call rather than by reading a task count.
MEASURED caveat that the implementation has to handle: `get(timeout=0)` raises
`TimeoutError` for a CANCELLED call exactly as it does for a running one, so a
deliberate cancellation is recorded as a sealed `*_cancel_*.json` receipt and
subtracted. Any other terminal state — returned, raised, retries exhausted — does
surface through `get`, because the output or the exception is there to be fetched.

## Status

    python3 tools/launch_t4_canonical.py --mode status --run-id <id>

## Reconcile and render

    python3 scripts/t4_canonical_reconcile.py --run-id <id>
    python3 scripts/t4_canonical_table.py

`reconcile` mirrors the volume with the Python API (never the CLI — `modal volume get`
silently collapses a directory onto one path and returns 0, and `modal volume ls` can
return the parent listing), never re-downloads an existing round lock, and derives
every number from LOCKS rather than checkpoints. It also re-gates every docked
molecule through the unmodified production `Fiber.check`, which is where
`invalid_chemistry_oracle_calls` comes from; that number must be zero.

Both are safe to run while the campaign is still going and report the partial state.

## Reading the numbers

- **Reconciled charged calls come from round locks**, which are immutable and are
  published BEFORE any of that round's docking. `checkpoint.json` is overwritable and
  a restarted container can move it backwards; never resolve a disagreement by taking
  the larger.
- A lock above the last checkpointed round is an INTERRUPTED round. Its queries are
  debited in full and never re-docked, so the reconciled count is a conservative upper
  bound on what the run got value from and a lower bound on lifetime spend if any cell
  rolled back and advanced past it. Check per-cell ladder monotonicity
  (`lock_ladder_monotone`) before quoting a total.
- **Within-run arm comparisons are sound; cross-run per-cell margins are not.**
  `qvina02` is seeded, `obabel --gen3D` is not, and one T4 seed molecule has been
  measured at -7.5 / -8.30 / -8.8 across three runs.
- A cell whose best molecule is its own docked seed produced NOTHING. Seed-only rows
  are not successes.
- Arms A and B carry no expansion at all, so an empty candidate pool ends those cells
  with `candidate_exhaustion` and unspent budget. That is the ablation working, not a
  defect — it is the condition arm C exists to answer.

## What is NOT in here

`diagnostics/T4_FROZEN_RESULT_v1.*` is the earlier mixed-provenance panel, preserved
unmodified as the development evidence that motivated adaptive support expansion. No
score from it is spliced into this table, and nothing here modifies it.
