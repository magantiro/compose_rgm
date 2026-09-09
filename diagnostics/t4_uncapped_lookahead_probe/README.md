# Uncapped lookahead: completed negative result

Completed at 2026-09-09 05:31:09 UTC. Removing executor/state cutoffs did not
activate task guidance in this fixed-parent probe. No docking improvement is
claimed. Authoritative outputs: result.json, verification.json and review.json.

## Outcome

- 32/32 reference rollouts completed, zero interrupted; 28 terminal evaluations.
- Zero task-dependent WHERE/WHAT/HOW decisions (10/10/16 decisions respectively);
  maximum total variation from reference was below 4e-16, numerical roundoff.
- 10 canonically unique harvested candidates, zero feasible, zero selected for
  docking, zero construction options completed, zero cycle-rank changes.
- Overlapping constraint failures: similarity 10/10, QED 7/10, SA 5/10.
- Proposal time 3003.178 seconds (50.05 minutes); total 3155.193 seconds.
  40,337 public executor calls; 369 completed law enumerations used 2870.870
  seconds. Frozen-law enumeration remains the dominant computational cost.
- All 16 committed edits passed exact production replay verification. No new
  oracle calls; the existing 51-call docking archive was not updated.
- Committed options: generic 3, open 2, append 1, aromatize 1, decorate 1,
  grow 1, rebuild 1. No constructive ring program was selected on this path.

Interpretation: full reference rollouts alone did not produce a useful task
signal here. This disproves the sufficiency of removing these cutoffs for this
parent/seed, not the potential of all lookahead controllers. The next authorized
work is the offline target-informed path diagnostic, not another blind run or
automatic docking. It separates executor reachability from guidance and
constraint barriers without fitting to released winners.

Result physical SHA-256:
`510200ff61e3a878e447b4a188f890e571ac9cae2534f8a59b596cb28b980943`.
Verification physical SHA-256:
`5efc23527d43f827352657e17e13f7d339419c03e1a51f7cce42483ded9491a9`.
Compact review physical SHA-256:
`295cbb8090de2b55484743bb04d34293cdfffc266c38c7733541f18fa91a4b7e`.
The 120 MiB full lock remains on the Modal volume below and in ignored
cache/candidate_lock.json, SHA-256
`27ee5f6824a53c6f9aa8896a7cd666c43feb27eb36129a1d32c3cf355ebf58d3`.
The committed review projector checked its envelope, physical lock hash and
verified result before extracting records; it performed no new molecular search.

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
zero fresh oracle calls. The original 15-90-minute estimate was an extrapolation;
measured completion time is reported above. There is no automatic docking or next round.

Heartbeat fields report phase, executor calls, law enumerations, completed
rollouts and terminal/nonzero-terminal counts once planning begins. Inspect
result.json, failure.json and heartbeat.json at the prefix above; do not relaunch
an unfinished started parent merely because no candidate lock exists yet.
