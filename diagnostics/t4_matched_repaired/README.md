# Repaired matched T4 round

One repaired paired development round was authorized and launched after the
exact-slot append-contract repair. The scientific result is pending; this
launch record does not claim improved discovery or docking performance.

Source: `cb7e94cfd611c632d530fccbed8ed2a00306b637`.
Prospective scope: `docs/T4_REPAIRED_MATCHED_ROUND.md` (commit `c3dda37`).
Call: `fc-01M1ZTTYJEDDG1BDNVW91CXDBK`.
Volume: `compose-v4-artifacts`.
Namespace: `/t4_matched_pilot/f1464df61efaf23e04574fcebcb037116507b8c5d02f9c9bdc166facfe019889`.

The run uses the unchanged self-hashed matched-pilot contract, one round per
arm, at most 40 docking calls total, a common 20,000 public-executor ceiling
per arm, one CPU, no GPU, one-hour timeout, and zero retries. Both candidate
locks precede every docking call. Old outcomes are not used to select weights,
templates, new candidates, or retries. The old run and repair evidence remain
unchanged.

`launch_plan.json` records authorization, exact identities, artifact-reuse
assessment, resource census, price estimate, and the unchanged configuration.
`attempt_1/spawn.json` preserves the full launcher receipt and serialized-source
hash manifest. Durable heartbeats confirm that the remote call started and
entered initialization; it does not depend on a live local launch client.

## Launch verification

The frozen implementation's full suite completed in 2,177.89 seconds:
4,490 passed, 49 failed, 58 errors, two skipped, one expected failure. The
repository is not green, and no broad milestone completion is claimed.

Of the 107 nonpassing cases, 105 match the previous full-suite identities and
failure/error messages (ignoring one process-memory address). The eight earlier
gradient-mode failures now pass. The two additional failures are multiprocessing
environment failures: one explicitly reports a denied torch shared-memory
manager; one worker test hit its 300-second timeout. Each unchanged exact node
passed once with the necessary subprocess/shared-memory permissions. No suite
restart, test relaxation, source repair, or gate change was used.

`launch_review.json` binds both full-suite XML hashes, both exact-node rechecks,
the launch plan, and the repair result. It records the per-case comparison and
the decision to run the single bounded development comparison, still subject to
all unchanged remote runtime/input gates. The known local RDKit catalog drift
is not treated as a passing production runtime; the remote pinned environment
must pass its own frozen checks.

Preflight confirmed zero mounted-tree drift and a fully clean exact launch
worktree. Deployment completed before spawning through `tools/t4_launch.py`.
All scientific source, configs, launch surface, and tests remained unchanged
between the verified revision and the launched revision. The main checkout's
later changes are documentation and recorded artifacts only.

Raw pytest XML is retained byte-for-byte, including traceback whitespace.
Do not rewrite it to make whitespace checks green. No branch has been pushed.

## Reading progress

```sh
modal volume get compose-v4-artifacts \
  /t4_matched_pilot/f1464df61efaf23e04574fcebcb037116507b8c5d02f9c9bdc166facfe019889/heartbeat.json -
```

Completed parent units are under each arm's `parents/`; candidate locks,
docking-start receipts, docking results, and the final result are separate
durable files. A phase heartbeat is progress, not a completed result. Do not
relaunch if the call is merely quiet; inspect its volume receipts first.
