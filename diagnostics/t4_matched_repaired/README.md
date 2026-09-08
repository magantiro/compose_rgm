# Repaired matched T4 round

The repaired controller completed a six-membered pendant-ring program and
selected its feasible endpoint as the best molecule in both matched arms.
The same molecule scored -8.9 in the reference arm and -9.3 in the committor
arm. The inherited unseeded oracle makes this difference evidence of docking
variability, not a guidance advantage. This is one inspected PARP1 seed0,
d=0.4 development round, not held-out benchmark superiority.

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
The function returned `complete` after 1,137.571 seconds (18.960 minutes).

## Measured result

| Measure | Reference | Committor |
|---|---:|---:|
| Dockings / failures | 20 / 0 | 20 / 0 |
| Feasible candidates | 19 / 20 | 17 / 20 |
| Best feasible docking score | -8.9 | -9.3 |
| Completed pendant programs / selected | 1 / 3 | 1 / 3 |
| Emitted representatives / canonical pool | 32 / 30 | 32 / 30 |
| Distinct docked molecules / bundle owners | 20 / 20 | 20 / 20 |
| Docked mean pairwise Morgan distance | 0.557736 | 0.565637 |
| Pool mean pairwise Morgan distance | 0.569303 | 0.586535 |
| Proposal seconds | 430.617 | 449.687 |
| Docking seconds | 47.571 | 47.439 |
| Public executor calls | 12,372 | 16,335 |
| Law enumerations / sampled transitions | 79 / 123 | 78 / 123 |
| Mean recorded per-step KL | 0 | 0.224396 |

The requested allocation was one CPU core, 8 GiB RAM, no GPU, one proposal
worker, and one particle per bundle; model parameters were float32. Recorded
Linux process peak RSS was 4,101,532 KiB (about 3.91 GiB), not total container
memory. Proposal throughput was 0.286 and 0.274 sampled transitions/second
respectively; raw executor work includes enumeration, not only sampled steps.
The source has 19 active atoms and the constructed endpoint has 27, within
the frozen 40-active-atom support.

Both arms received the same 24 outer bundles: generic 5; build_ring_system,
grow, and decorate 3 each; rebuild and aromatize 2 each; append, open, restate,
scaffold_extend, shrink, and small_ring 1 each. Among 20 dockings per arm,
build_ring_system contributes 1, decorate 2, restate 0; other counts equal
their selected-bundle counts. Fused construction was clean-product-applicable
in 13/24 bundles but selected in zero. Cyclize and annulate were also not
selected. Their availability is not a sampled construction result.

The new best molecule is:

```text
CN(C)Cc1ccc2c(c1)CNC(=O)c1ccc(C3=NC(=O)NON3C)n1-2
```

It has +8 heavy atoms (C3N3O2), +1 graph cycle rank, and +1 ring system.
Its QED is 0.844474, SA 3.539611, similarity 0.540984, and constraint
violation is zero. Bundle `c808400777e6a0952c54` completed all 11 steps in
both arms. Offline replay verifies all 22 saved transitions against the
semantic production executor, preserving validity, connectivity, the frozen
region context, and each active macro contract. This is two matched program
instances producing one unique endpoint, not two independent discoveries.
The two restatement steps form a round trip: step 11 returns to the canonical
step-9 closure product. Successful execution does not establish beneficial
refinement.

The other two pendant bundles in each arm still stop after eight growth steps
with `no_clean_option_product`. No partial compound endpoint was docked.
The saved trace does not name its sampled mark: replay uses an executed ledger
mark with the same exact source and product as a witness. It does not infer
unsaved transition probabilities or claim exhaustive reachability.

## Intended scale and realized structure

Among docked candidates, intended release ranges from 0.052632 to 0.894737.
Its median is 0.157895 reference and 0.184211 guided. Realized coherent change
has min/median/max 0.052632 / 0.105263 / 0.473684 in both arms.

The successful program released only one original atom (0.052632) but added
eight atoms and realized 0.473684 coherent change. This is substantial pendant
growth from a small attachment region, not evidence that selecting a large
region reliably causes a large rewrite. Its original-atom change remains
0.052632. The local/global allocation is intact; proportional realized-scale
control remains unproven.

Four of 20 docked candidates per arm change topology. Two construct cycles:
the new six-membered pendant ring and a small-ring closure (+1 cycle rank,
+1 ring system each). The open and shrink candidates instead have -1 cycle
rank and +1 ring system, so their system-count increase is splitting, not
construction. Grow/rebuild candidates do not add cycles in this round.

## Interpretation and next boundary

The identity-unsafe closure predicate was a real obstruction: after its repair,
the previously blocked sampled bundle completes and reaches the oracle. Unlike
the earlier decoration-only best candidates, this round's best is constructive
ring chemistry. The previous paired round's best scores (-8.3 / -8.2) are
historical development observations, not noise-controlled effect estimates.
Do not compare this 20-call-per-arm round as a matched claim against the prior
500-call COMPOSE score or against IVG.

No guidance advantage is demonstrated: both arms follow the same successful
program, and the guided arm has fewer feasible dockings and more executor work.
No global-scale, fused-ring, online adaptation, synthetic feasibility, or
held-out superiority claim follows. Keep the repair frozen. A next objective
search experiment needs a separately fixed multi-round budget and an explicit
oracle-noise policy; a proposal-only ring-method panel would answer a different
question about completion efficiency. Neither next experiment is launched here.

## Result verification and reproduction

`attempt_1/review.json` contains the complete existing paired audit, including
per-option scale/topology statistics. `program_replay.json` records exact path
witnesses. `remote_inventory.json` binds all 29 remote JSON files (42,124,306
bytes); 11 downloaded remote files match byte-for-byte. Larger executor ledgers
and parent receipts remain on the volume. `verification.json` records hashes,
ordering checks, budgets, code identity, and the verification limitations below.
All six frozen runtime input hashes match. Both locks precede the common
oracle barrier, which precedes both docking-start receipts. No cap, timeout,
retry, training, support change, or docking-driven weight update occurred.

```sh
.venv/bin/python tools/t4_matched_audit.py diagnostics/t4_matched_repaired/attempt_1
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 .venv/bin/python \
  tools/t4_program_replay_audit.py diagnostics/t4_matched_repaired/attempt_1 \
  /private/tmp/t4-repaired-ledgers.xvQHqW
```

For replay on another machine, download each arm's `executor_attempts_0.json`
from the recorded volume namespace as `reference.json` / `committor.json` in
a chosen ledger directory. The auditor checks hashes before replaying.
Its initial offline pass incorrectly used the legacy executor and stopped at
`atom_restate_semantic`; the correction uses the same semantic executor as the
run and has a regression test. Six focused auditor tests, formatting, lint,
and whitespace checks pass. This read-only audit change does not modify the
frozen deployed scientific source, config, or launch surface. Its implementation
hash is recorded separately from its then-current Git revision and bound to
the subsequent auditor correction commit in the verification receipt.

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
between the verified revision and the launched revision. At launch, the main
checkout's later changes were documentation and recorded artifacts only.
Subsequent offline-auditor changes and their focused tests are separate from
the deployed scientific dependency closure.

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
