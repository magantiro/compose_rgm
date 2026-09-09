# Uncapped lookahead: launch record

Status at 2026-09-09 04:39:05 UTC: spawned and initializing. No result or docking
improvement is claimed. This directory will receive the immutable result and
compact review after preparation and exact selected-path verification finish.

This user-authorized comparison removes the planning/public-executor and cached
search-state cutoffs. It does not change the 16-edit terminal objective, 32-rollout
allocation, chemistry, predictor, priors, exploration floors or kappa. It uses
the same first parent as the earlier lazy-reference probe. See
../../docs/T4_UNCAPPED_LOOKAHEAD.md for scope and interpretation.

## Run identity

- Scientific commit: `de5f4d4fc18d530ea048ee77e4383f1f9d7a678d`.
- Clean worktree: `/private/tmp/compose-uncapped-lookahead.fYXgCN`.
- Call: `fc-01M227BJ38SYWKMHR6NDQ9QTXM`.
- Deployed app: `genmol-t4-opt`, session `compose_iclr`.
- Volume: `compose-v4-artifacts`.
- Prefix: `t4_uncapped_lookahead_probe/326d1235b652f31de8aa86e4d6a4ab9b2edf5b96fd064ebbeea8a6a4adea4c01`.
- Contract physical SHA-256: `b3be3bb8aeedacf3b6e1b9f9cfb0aa7833c68432e305206149f94a1c80816d8a`.
- Contract body SHA-256: `b1f356d5cb7601bd4568c3ac7abf2ae0f6d433ee1a905b426114854694bbe0df`.
- Image identity SHA-256: `bfaed5b44f88803eb1876f41f46d5bc3e66ef6e2d6e250afa7549f34e165ebe6`.
- Full launch/source manifest: spawn.json, byte-identical to the launch receipt.

Before launch the deployed app had zero active tasks, and no prior artifact
namespace existed for this new policy. The earlier capped result remains the
baseline, not a compatible completed preparation. Runtime verifies the same
exact archive, frozen model and predictor-check hashes before molecular work.

## Verification performed

In the clean exact-commit worktree, preflight passed with zero drift. The focused
dependency suite passed 198 tests, zero failures/errors/skips, in 15.189 seconds.
The XML receipt is focused_tests.xml, SHA-256
`48f1aaefc663d487043ef5b48724b27a34285c035adab546b660f7b66da032f7`.
Covered modules: lazy_reference, task_search, docking_value, t4_task_search,
option_continuation, ring_program, continuation, sampled_continuation,
t4_warm_continuation, fused_option, ring_expansion, t4_parent_budget,
continuation_profile and fused_reference_profile.

Ruff lint and format checks passed for the eleven touched core/test/launcher
Python files. The Modal app compiled; its pre-existing whole-file style debt
was not reformatted. Diff whitespace checks passed. The unrelated repository-wide
suite was not rerun under the scoped T4 development policy. No release/milestone
completion is claimed. Unrelated dirty scaffold/model work was excluded.

Deployment completed, then tools/t4_launch.py --uncapped-lookahead-probe spawned
the durable call. One CPU, 8 GiB, no retries, a two-hour administrative timeout,
zero fresh oracle calls. The 15-90-minute estimate is an extrapolation, not a
measured completion time. There is no automatic docking or next round.

Heartbeat fields report phase, executor calls, law enumerations, completed
rollouts and terminal/nonzero-terminal counts once planning begins. Inspect
result.json, failure.json and heartbeat.json at the prefix above; do not relaunch
an unfinished started parent merely because no candidate lock exists yet.
