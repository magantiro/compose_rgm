# Dynamic COMPOSE v2.2 development revision

## Status and scope

Dynamic-v2.2 is an implemented, unlaunched development revision. It does not
modify or consume the live T4 or PMO Dynamic-v2.1 runs. Its proposed quick
pilot is limited to 256 charged calls each on T4 FA7 seed 0, BRAF seed 1 and
JAK2 seed 1, plus PMO GSK3B and C7H8N2O2 isomers. A separate launch lock is
required before scoring.

The scientific problem is that Dynamic-v2.1 often generates valid, eligible
candidates but allocates calls using global channel statistics rather than the
specific parent and edit. FA7 has a second failure mode: repeated zero-query
rounds regenerated the same 40 QED-failing endpoints instead of advancing a
proposal frontier.

The prospective hypothesis is that a persistent feasibility frontier plus a
small existing parent-edit model improves finite-budget endpoint allocation.
Implementation and zero-oracle tests do not establish this claim.

## Shared controller core

Both tasks retain the independent Dynamic-v2.1 shallow and structured
generators, exact execution, post-filter pooled competition, 3-module,
32-primitive and 8-block support, and an empty task-specific route archive at
initialization.

After at least 24 unique measured edit endpoints, the existing deterministic
`ParentEditFeatures` and `ParentEditModel` ridge ensemble is fitted to the
chronological run prefix. It is refitted only after 16 additional unique
measured endpoints. Each fitted model belongs to exactly one oracle protocol.
It predicts completed endpoint utility from the measured parent, constructor,
actual graph delta, attachment context, complete-program descriptors and
mutation records. Bootstrap disagreement is logged but never used for
optimism.

For an oversized eligible pool, the charged batch is allocated as follows:

| Share | Role |
| ---: | --- |
| 25% | Preserved within-pool shallow opportunity order |
| 25% | Score-blind structural diversity |
| 50% | Greedy predicted gain to the current task archive |

Before model readiness, allocation is 50% shallow spine and 50% score-blind
diversity. If the eligible pool fits the query allowance, every candidate is
scored directly. Empty quotas are released to score-blind diversity.

The shallow synthesizer is unchanged and has independent proposal randomness.
Parent scheduling also has its own persistent random stream. This preserves a
stable shallow opportunity backbone, but it does not claim to reproduce v0's
historical end-to-end trajectory, whose single random stream was interleaved
differently.

T4 uses oriented utility `-docking_score` and archive size `k=1`. PMO uses its
native higher-is-better reward and `k=10`. Models, observations and artifacts
are not shared across oracle protocols. Failures, endpoint-ineligible
molecules and unqueried proposals never receive task-utility labels.

## Persistent T4 feasibility frontier

The frontier stores at most 16 exact-valid but endpoint-ineligible program
prefixes. It records exact source, program, binding, trace, non-oracle
similarity/QED/SA margins, module count, attempt key and
independent shallow/structured random states. Subsequent proposal-only waves
alternate new root proposals with continuations from retained prefixes.

Every continuation binds on the exact prefix endpoint and is re-extracted as
one root-to-endpoint executable program. The complete route, not each suffix,
must remain within the same 3-module, 32-primitive and 8-block limits. Only an
eligible novel completed endpoint may receive a docking call. Per-gate
deficits are normalized before deterministic nondominated sorting, and
molecular diversity orders candidates within each frontier layer. Raw SA, QED
and similarity deficits are not compared as if they shared units. No docking
score, surrogate score, comparator result or known route enters this process.

This specifically prevents an interrupted or zero-eligible FA7 round from
restarting the same deterministic cold stream. The attempt-key set and both
proposal random states advance and are content-hashed after every wave.

## Runtime exclusions and evidence boundary

Runtime inputs contain no Full-146/v0/v1/v2.1 outcomes, winner molecules,
winner routes, comparator scores, target-to-program map or target-specific
complete program. Existing T4/PMO results may be joined only after a pilot is
complete for offline comparison.

The pilot must log full proposal pools, frontier transitions, feature/model
identities, selection roles, charged-query ledgers, failures, snapshots and
score-versus-call curves. A successful pilot would be prospective development
evidence on the declared units, not a full benchmark or held-out result.

The 256-call pilot does not use score-plateau stopping. A unit may end below
its call ceiling only for an explicit proposal/round shortfall or a durable
failure. Launch and live status are handled by `tools/dynamic_v22_pilot.py`;
the launch receipt binds the clean Git revision, material inputs, task locks,
oracle identities, seeds, and remote function-call identifiers.
