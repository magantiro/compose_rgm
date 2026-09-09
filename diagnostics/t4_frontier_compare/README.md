# Paired frontier comparison: ready, not launched

The implementation is frozen at `a6aa24b9a4b2b21883538a9301e609e83ff2435c`.
The self-hashed prospective recipe is `configs/t4_frontier_compare.json`,
contract body SHA-256 `c7036dfdd0d605f60d9bfef24a2032c50df79554d0881f1cc249beff6dd3651d`.

On the clean detached launch worktree, all 102 focused tests passed in 27.409 s,
with zero failures, errors or skips. Six changed/new Python files pass lint and
format checks. The legacy Modal application has 17 existing lint findings, with
no new issue types/counts or findings in the added block. No whole-repository
suite was run, under the bounded T4 development policy in AGENTS.md.
Tests use small synthetic reference rows and the production executor; they do
not establish behavior of the full learned reference on the T4 cell. Local
Python is 3.12.9 with the pinned chemistry overlay; remote Python is 3.11.

Strict preflight passed and deployment of `genmol-t4-opt` succeeded. The spawn
command was rejected by the approval reviewer because the current repository
milestone does not explicitly authorize the paired 40-call T4 comparison.
No scientific call was spawned, no call ID exists, and no new docking evaluations
were spent. This is an authorization boundary, not a negative scientific result.

Next required decision: explicit user authorization for this one warm-start round
per arm, at most 20 new dockings each (40 total), two parallel 1-core/8-GiB
preparation workers plus one driver, with a six-hour administrative timeout.
The expected preparation interval is 30–120 minutes, not a measured promise.
Do not bypass the rejection with another command or a different launcher.

The clean deployed source remains at
`/private/tmp/compose-frontier-pair.FmM9Iy/source`; its raw focused JUnit receipt is
`/private/tmp/compose-frontier-pair.FmM9Iy/focused.xml`. Verification and input
hashes are recorded in `verification.json`. All unrelated working changes remain
outside this revision. Nothing was pushed.
