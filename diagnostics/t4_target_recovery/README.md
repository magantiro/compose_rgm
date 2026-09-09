# Target-informed controller recovery: launched, outcome pending

This is one answer-known development attempt, not replay of a witness and not
blind docking optimization. No recovery result or docking improvement is yet
claimed. The prior 51-call docking archive remains unchanged.

## Frozen run

- Scientific source: `33abd643ee1b8db4ff9931f8f1998ecd78b61a70`.
- Implementation: `767f3be`, with post-edit option-state and scoring-asset
  provenance added by `33abd64`. Both are committed and unpushed.
- Contract: `configs/t4_target_recovery.json`, physical SHA-256
  `0aab60dff411196551a5f46abd56b0db67bfe06e12aa6e7d7be3a77ba42c9ecf`.
- Scope: `docs/T4_TARGET_RECOVERY.md` at the scientific revision.
- Run ID: `70611576f82d337385717acac425591ed6dd76fc21b017ab8ad59a6ac4949936`.
- Modal call: `fc-01M23MZG5XMSVHNJWJKDEVT86B`.
- App: `genmol-t4-opt`; session: `compose_iclr`.
- Volume: `compose-v4-artifacts`.
- Namespace: `t4_target_recovery/<run_id>/`.
- Complete launch/source manifest: `spawn.json`, physical SHA-256
  `6db8667093c7d72c1915fac868cc881ee8c457311865a5da9820b85cc18d3614`.

Source is the exact original PARP1 seed0 in the complete 51-call archive. The
destination is the winner identified in the frozen contract; its saved 23-step
witness is not supplied to the controller. Source/target selection is informed
by inspected development evidence and cannot become a held-out evaluation.

The fixed graph-feature similarity to that destination replaces docking value
only in this diagnostic. Exact canonical 2D equality defines recovery, not a
high fingerprint similarity. The existing hierarchy, untrained balanced option
prior, generic, full primitive support, executor, kappa=1 and exploration floors
remain. The separate 32-edit horizon does not modify the frozen 16-edit T4
recipe. Eight full reference planning rollouts at most; no model fitting,
training or new oracle calls. See the contract for the complete allocation.

## Verification and execution

Strict preflight passed on a clean detached worktree:
`/private/tmp/compose-winner-paths.W1n7sh`. Unrelated dirty scaffold/model work
was excluded. No active T4 task or previous target-recovery volume namespace
existed at the prelaunch inventory. The first sandboxed volume query could not
connect; the network-enabled inventory succeeded. No duplicate run was started.

Focused dependency suite at `767f3be`: **64 passed**, zero failures/errors/skips,
5.52 seconds. Receipt `focused_tests.xml`, SHA-256
`1e992fe95137c62dc47c4c651553124b364cead1eaf4459833cfc0f5f8307f42`.
This covers target recovery, task search, anytime credit, T4 preparation,
option continuation, pathwise gates and lazy reference parity.

After the provenance-only addition, the eight target-recovery tests passed
again on clean `33abd64` using the isolated RDKit 2024.03.5 / NumPy 1.26.4
chemistry overlay, in 1.56 seconds. Receipt
`pinned_chemistry_tests_33abd64.xml`, SHA-256
`559fd9d20030b5c25cbdcd3aa88591feaf321041b2a85b69ad807fc0f53ea9d0`.
The earlier eight-test pinned run at `767f3be` also passed; its separate receipt
is retained. Local Python is 3.12.9. This is chemistry-runtime checking, not a
claim of full Modal environment equivalence.

Touched standalone Python files passed Ruff lint/format checks; the Modal app
compiled; staged and final diff whitespace checks passed. The unrelated full
repository suite was not run under the scoped T4 development policy. This is
not a release/milestone qualification.

Deployment completed in 55.697 seconds, followed by strict preflight and
`python3 tools/t4_launch.py --target-recovery`, which durably spawned the call.
No `modal run --detach` or ephemeral driver was used. The first remote listing
confirmed `launch.json` was published at 2026-09-09 13:55 EDT.

## Monitoring and next action

Read `heartbeat.json`, `runtime_gate.json`, `decisions/`, `search.json`,
`result.json` and `failure.json` under the namespace above. Every completed
learned-law enumeration is persisted under `laws/`; every committed decision
has an exact-state receipt. Heartbeats are emitted every 30 seconds. One CPU,
8 GiB, no accelerator; estimated 10-45 minutes with a 70-minute platform cap.
These are an estimate and resource ceiling, not measured completed-run cost.

Review the actual recovery outcome before choosing another experiment. A
planning-only target encounter is not a committed recovery. If the run stalls
or fails, compare saved decisions with the known path only in a separate
post-run diagnostic. Do not automatically retry, dock, change the guide,
extend the horizon or launch another seed. No solution is promised in advance.
