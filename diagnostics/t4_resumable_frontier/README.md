# Stage-1 resumable frontier verification

Implementation: `28bad97040fed500f07e9985f3cdc4fd1e644166`.
Design and acceptance: `docs/RESUMABLE_FRONTIER_STAGE1.md`.
Machine-readable receipt: `verification.json`.

## Outcome

**Computed engineering evidence:** 97 focused tests passed, with zero failures,
errors or skips, in 21.304 seconds on the clean committed source. The environment
used RDKit 2024.03.5, NumPy 1.26.4, SciPy 1.13.1 and Python 3.12.9 on arm64 macOS.
The receipt binds the source/test hashes, exact command and JUnit report hash.
Ruff lint/format checks and `git diff --check` passed. This was not the repository
release suite or a docking experiment. No remote job or new docking call ran.

The tested implementation retains the existing local/global region distribution,
balanced option reference, generic exploration and kappa=1 primitive control.
Completed-option feedback can propagate before a complete primitive-horizon
rollout. Exact unfinished option states, cached full rows, primitive witnesses
and RNGs survive preparation slices and task-snapshot changes.

Split/resume equivalence was checked for the planning cursor and actual committed
T4 fixture paths, including an executor pause and a partially completed
multi-lineage slice. A task-model refresh retained the exact frontier without
re-enumerating chemistry. Codec tests covered noncontiguous occupied slots and
the previously omitted ring-expansion progress field in the shared decoder.

Initial fixture failures exposed a tuple/list lineage mismatch and the legacy
verifier's late schema check; both were repaired. The per-lineage pending-slice
census also prevents an interrupted preparation from repeating already completed
parent work. These are implementation repairs, not improvements in task score.

## Claim boundary and next action

Best-witness backups are an explicit search heuristic, not learned continuation
values, calibrated uncertainty or exact Doob control. Their usefulness under a
weak task predictor is still unmeasured. There is no proposal-throughput result,
InVirtuoGen recovery result or improved docking score from this change.

The new prepare-only result is deliberately not accepted as a legacy candidate
lock. Next, bind a compatible selected-path replay/launch contract and run a
bounded T4 development comparison against the same-generator post-hoc arm.
Retain the local/global hierarchy in that comparison and report intended versus
realized structural scale, ring/cycle changes, candidate diversity and proposal
time. Do not add the learned continuation head or expand the benchmark campaign
before that first comparison provides evidence.

Code changes and this verification record are separate local commits. No branch
push or unrelated worktree cleanup was performed.
