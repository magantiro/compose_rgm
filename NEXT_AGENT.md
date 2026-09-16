# Active COMPOSE handoff (2026-09-16)

The active T4 strategy-reset implementation is isolated on branch
`t4-objective-dynamic-reset-20260916`, currently checked out at
`/private/tmp/compose-t4-objective-dynamic-reset-20260916`.

Read that branch's `HANDOFF.md` first, then `docs/T4_AUTONOMOUS_RESET.md`.
Those files contain the one-hour handoff ending 2026-09-16 07:44:57 UTC.
Do not mistake this original working tree's HEAD for the latest T4 implementation.
The branch is in this same Git repository and its runtime commit is `80dc1aa3`.

The forensic evidence is at `diagnostics/t4_strategy_reset/20260916/REPORT.md`.
Raw historical data and normalized scored records remain here in that directory;
Compact evidence, recovery manifests, and a lossless compressed copy of all
35,895 scored records are committed on the implementation branch.
Existing PMO and AGENTS.md changes in this original worktree belong to
ongoing user work and have deliberately not been reset or overwritten.

No claim that the new controller beats IVG is authorized without scored results.
The next agent must report what has actually run, not infer success from code.

Current result: five zero-oracle Modal probes finished; four cells produced
eligible candidates, FA7 did not. No scored pilot or new learned-prior fit was
launched. Nineteen focused tests pass; the broad repository verification is not
green. Read the exact failure report and support-probe results before deciding
what to launch. Frozen old controllers and runs were not changed.
