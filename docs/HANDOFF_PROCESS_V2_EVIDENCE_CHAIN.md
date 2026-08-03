# Process-V2 evidence chain: implementation handoff

**Branch** `codex/editing-v2-process-v2-evidence-chain` · **base** `b790024` ·
**worktree** `/private/tmp/compose-process-v2-evidence-chain`

This handoff describes a runnable implementation. It launches nothing. No Modal
job, V2 proof scan, Active8 materialization, Gate 0, T1, P50, or training run was
started at any point, and every artifact this work publishes carries its
authority fields explicitly false.

## 0. Scope decision, made by the implementer

**Phase 3 was narrowed, deliberately, and this is the record of that call.**

The approved plan named seven downstream stages. Two are wired here:

* the Process-V2 admitted source joined to the Active8 lane/role inventory, and
* the Gate-0 contract binding — **contract only, see below**.

**Gate 0 is not runnable under Process V2, and that boundary is tested rather
than implied.** The V2 structural contract exists, self-hashes, and validates
under the V2 chain module. No runner consumes it:
`load_semantic_gate_zero_structural_contract` hard-binds
`contract_id == "editing_v2_semantic_gate_zero_structural_v1"` and
`FROZEN_CONTRACT_SHA256`, and enforces an exact V1 field set, so it refuses the
V2 contract by construction. Teaching that loader to accept a structurally
different V2 body would entangle the two chains, which is exactly what keeping
them distinct is for; running Gate 0 under Process V2 needs a V2 runner. The
boundary is pinned by
`test_gate_zero_cannot_yet_consume_the_process_v2_structural_contract`, which
also asserts the V1 contract still loads unchanged through that same loader.

Five are **not** wired: candidate/successor caches, T1 panel and capacity, P50
recipe and prelaunch. The reason is not time. Those five sit *behind* Gate 0 in
the dependency order, and Gate 0 has never been run under Process V2, so their
inputs do not exist. Adapters written against them could not be exercised even in
a local fixture, and shipping unexercised adapters as "runnable" is the failure
mode this branch exists to avoid. The owner was offered the full seven and chose
the narrow cut on that reasoning.

What that means in practice: the chain is runnable up to and including the
Process-V2 admitted source, and the contracts for every later stage exist and are
verified, but the later *execution* wiring is future work.

## 1. Commit inventory

| Commit | Subject |
|---|---|
| `e5ad4f3` | tests: make the process v2 pin-memory and multiworker checks portable |
| `d5cfcaf` | scripts: freeze the process v2 rebind plan from its immutable payload |
| `5e46b9e` | data: join the process v2 overlay to the v1 active8 source inventory |
| `3625857` | tests: prove the rebind chain end to end and scope the sharding claim |
| `2e74582` | experiments: add the distinct process v2 downstream contract chain |
| `818f527` | docs: hand off the process v2 evidence chain |
| `44dddd1` | tests: pin the gate zero consumption boundary for process v2 |
| `d6f27f2` | docs: record the process v2 evidence chain verification results |
| `0e25bc0` | experiments: bind the admitted-source schema the adapter actually declares |

## 2. Identities

Every value below was recomputed live in this worktree, not quoted.

| Role | SHA-256 |
|---|---|
| current readable V1 implementation | `6c4721f0dd37132aae657e7aa5f1bfc01cef270662f228171c4587eb7dd48491` |
| **current Process V2** | `0c938177a34819e6e828920c1f66e240c6eb251fe7c9ea6cfe6757829dceb2dd` |
| historical completed V1 payload process (superseded) | `6b98ee21ef8b853deda9fa56a2963178208ecc893a397fb4aa412629fc2414d7` |
| rejected candidate identity (lineage only, no artifact) | `9fde14b59fc6bfb7be7aaf83564658a9a6758f479d9fd94c134206e84873319b` |

All four match the values supplied in the task. The rejected identity appears
only under lineage-tagged names; the chain verifier fails the build if a
historical value is carried under a live-sounding name, which it did once during
this work and which was fixed rather than suppressed.

## 3. Environment

```
python 3.14.2   torch 2.13.0   numpy 2.5.1   rdkit 2026.03.4
pytest 9.1.1    ruff 0.15.22   macOS-26.5.2-arm64
```

**Correction to the task's premise.** The Phase-0 defects were reported against
"macOS Python 3.14 / PyTorch 2.11". The repository-pinned interpreter
(`.venv/bin/python`, which `AGENTS.md` mandates) carries **torch 2.13.0**; bare
`python3` carries torch 2.11.0. Both were tested.

## 4. Exact commands

```bash
cd /private/tmp/compose-process-v2-evidence-chain
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src
VENV=/Users/rmaganti/Documents/Codex/2026-07-14/ok-so/compose_rgm/.venv/bin/python

# focused
$VENV -m pytest -q tests/test_process_v2_rebind_end_to_end.py \
                   tests/test_editing_process_v2_rebind.py \
                   tests/test_editing_v2_process_v2_active8_source.py \
                   tests/test_process_v2_atom_delete_mask.py \
                   tests/test_process_v2_runtime_checkpoint.py

# whole suite, both sides of the comparison
$VENV -m pytest tests/ -q --continue-on-collection-errors

# identity chain
$VENV scripts/verify_process_v2_hash_chain.py --base-revision b790024

# lint and whitespace
$VENV -m ruff check <touched files>
git diff --check

# freeze a rebind plan (writes at most the plan; --dry-run writes nothing)
$VENV scripts/plan_process_v2_rebind.py \
    --artifact-root /artifacts \
    --v1-payload-root <V1 migration payload root> \
    --dry-run
```

## 4b. Test results

Whole suite, same invocation on both sides:

```
base b790024 : 14 failed, 2470 passed, 2 skipped, 15 errors in 895.79s
head         : 14 failed, 2518 passed, 2 skipped, 15 errors in 925.33s
```

Compared as failure **sets**, not counts, because equal counts can hide an
equal-sized swap:

```
failures only on head (regressions) : none
failures only on base (fixed here)  : none
collection errors base vs head      : IDENTICAL
```

**+48 tests passing, zero regressions.** The 14 shared failures and the 15
collection errors are pre-existing on base: the errors are a missing PyYAML in
this venv, and the failures are dominated by the RingCore catalog fingerprint
drift, which fails identically on both sides.

Note: the whole-suite figures above were measured at `44dddd1`. The later
`0e25bc0` corrects a schema string in the seven V2 contracts and regenerates
them; its focused suite is green and the chain still verifies, but the whole
suite was not re-run after it. Stated rather than implied.

Focused suites: 32 (V2 contract chain) + 50 (rebind and end-to-end) + 7 (Active8
join) + 51 (Phase-0 mask and runtime/checkpoint, under **both** torch 2.13.0 and
torch 2.11.0).

Other gates: chain verifier `AGREES` with 178 literals agreeing and **0 stale**;
ruff clean on all 10 touched Python files and 64 errors on base equals 64 on head
with none from a touched file; `git diff --check` clean; the seven V2 contracts
regenerate deterministically and byte-match what is committed; **0 V1 configs
modified**, in the commits and in the working tree.

## 5. Phase 0: two verification portability defects

**Neither defect reproduced.** Tried both interpreters, with and without the
documented OpenMP guards, in isolation and whole-file. `pin_memory()` raised
`RuntimeError` every time; the multiworker test passed in ~9 s every time. That
is stated rather than papered over.

What was found instead is a real defect, arguably worse than the reported one:
**the pin-memory test asserted nothing on any Mac.** `Tensor.pin_memory()` takes
no device argument and dispatches to the current accelerator, which on these
machines is MPS, which has no pinned host memory. The test caught the resulting
`RuntimeError` and skipped, so its Process-V2 assertions never executed. Invoking
an unsupported backend to discover it is unsupported is also what made the call
unsafe in the first place.

The fix splits the concern. The accelerator-backed half is gated on
`torch.cuda.is_available()` — a capability, not a caught exception. The portable
half substitutes identity for the pin and asserts `pin_memory` rebuilds all 43
dataclass fields, which is the property worth testing: it is a hand-written
field-by-field reconstruction, the same shape that dropped
`atom_delete_admission_mask` from `_index_factorized_batch` in the previous
round. Mutation-proven: dropping that field now fails; previously nothing did.

For the multiworker hang, the mechanism was fixed rather than a symptom that was
never observed. `workers >= 1` builds a real DataLoader; macOS spawns a fresh
interpreter that re-imports the package and unpickles the dataset and collate
function; nothing on that path has a deadline. It now runs in a child process
under a 300 s bound, so a stall is a bounded failure with a diagnostic instead of
a hung session, and a companion test asserts the fixtures pickle with their
Process-V2 modes intact so a startup failure is localized rather than inferred.
Both proven: a legacy-collator mutation is killed with the child's stderr
surfaced, and the deadline path raises its intended message.

Production pinning semantics and the sampler are unchanged.

## 6. Phase 1: the rebind execution surface

`scripts/plan_process_v2_rebind.py` freezes the content-addressed plan;
`modal_apps/run_process_v2_rebind_app.py` maps and reduces it.

Three decisions a caller must not improvise, and why:

**Pinned identities are read from the payload, never computed live.** The
historical migration was built under the superseded V1 identity, and
`bind_v1_semantic_payload` requires exact dict equality with what each receipt
carries. Computing `editing_v2_process_identity()` today yields the current
value, and every task would be refused. The driver recovers the identities the
receipts carry, requires every receipt to agree so a payload root that mixes
migrations cannot bind whichever task was read first, and refuses unless the
process identity equals the expected historical one.

**`entries_per_task` is a data decision, never a fleet decision.** It is folded
into `run_identity_sha256`, so planning the same payload for 20 workers and for 1
worker addresses two different runs. Worker count is chosen at map time and is
deliberately not a plan input.

**The app reuses the library, and imports the driver rather than reimplementing
it**, so the pinned-identity discovery is identical locally and remotely. It maps
only tasks not already durably present, lets only the reducer declare completion,
and resolves the admitted source at the end so the downstream adapter is proven
able to read what was just published.

## 7. Phase 3 (narrow): the admitted-source join

`src/compose_v4/data/editing_v2_process_v2_active8_source.py`.

The rebind does not rewrite the packed shards; it publishes an **overlay** of
per-entry admission decisions over the same immutable payload. So the Process-V2
Active8 source is a join, and this is the one place that performs it. Both halves
keep their own authority, and the per-task census is taken from the published
task results rather than recounted, so this module never becomes a second census
authority.

The cross-check is the reason it exists. An overlay proved against one migration,
paired with a different migration's shards, produces a corpus whose chemistry and
whose admission decisions came from different runs — and **every count still
reconciles run-wide**, because the reduction only ever sees its own overlay. The
join refuses unless both name the same task set, the same semantic shard per
task, and the same lane and split, and unless source equals admitted plus
rejected per task as well as in total.

The V1 payload identity and the live Process-V2 identity are exposed under
distinct names. Collapsing them into one `process_identity_sha256` is how a V1
artifact would come to be read as a V2 one.

## 8. A correction to the approved Phase-4 specification

The plan called for a test proving that "a charge-policy-incompatible V1 delete
trace is rejected whole". **That case cannot exist, and the impossibility is
proven rather than asserted.**

`editing_charge_policy_constraint` is already a hard constraint of the V1
semantic rewrite system (`src/compose_v4/rewrite/kernel.py:379`), and Process-V2
gate 4 calls the same `charge_policy_preserved` over the same successor produced
by the same `apply_atom_delete`. A charge-violating delete therefore never
reaches a V1 payload: the V1 migration records it as rejected with
`semantic_action_rejected`. Six charged and aromatic molecules were surveyed;
every `charge_policy_violated` slot was also refused by the V1 runtime, with zero
counterexamples.

**Consequence worth carrying forward:** the charge policy is a *shared* invariant
across both processes, not a Process-V2 tightening. The genuinely V2-only surface
is aromatic and SCAR incidence. The whole-trace rejection path is therefore
exercised through the aromatic connected-nonleaf gate on a charged aromatic lead,
and the charge boundary itself has its own dedicated test.

## 9. Unresolved risks and open items

1. **Five downstream stages are not wired** (§0). Their contracts exist and are
   verified; their execution adapters do not.
2. **Gate 0 has never run under Process V2**, so no Process-V2 structural
   evidence exists and nothing downstream of it can be exercised end to end.
3. **The completion publishes the rejected-trace inventory as a hash only.**
   `result_inventory` is published in full alongside its hash; the rejected-trace
   inventory is not. When that hash moves across shardings a reviewer has nothing
   to diff and cannot tell that only the task address changed. The per-task
   manifests carry the rows; the completion, which is the document a reviewer
   reads, does not.
4. **The V1 T1 and P50 policies contribute zero checked pointer edges.**
   `editing_v2_semantic_t1_panel_policy_v1.json`,
   `..._t1_capacity_policy_v1.json` and `..._p50_recipe_policy_v1.json` bind
   their parents by bare hash with no adjacent `path` sibling, so the chain
   verifier discovers no edge for them and `_check_pointer_edges` never runs.
   They still appear in the chain report, which resolves by value ownership,
   which makes them look checked. A stale pin there is caught only by the
   base-revision sweep, and degrades to a silent "unclassified" when no base is
   available. `capability_cells_v1`'s classifier-source pin is invisible for a
   related reason. The V2 counterparts contribute 2, 4 and 2 checked edges.

5. **`p50_authorized` versus `bounded_p50_authorized`.** The admitted-source
   adapter spells its P50 flag one way and the plan driver envelope the other.
   Both are false, so nothing is wrong today, but a consumer grepping one
   spelling will miss the other.
6. **`entries_per_task` is part of the run address**, so a re-shard republishes to
   a different run root. This is deliberate provenance, but it means the shard
   size must be chosen once, deliberately, and recorded.
7. **The throughput projection below is a floor, not a prediction** (§10).
8. **Pre-existing and unrelated to this work:** the venv lacks PyYAML, which
   produces 15 collection errors on both sides of the comparison, and
   `build_production_ringcore_catalog` reconstructs a fingerprint that differs
   from the frozen constant, which fails several Gate-0-shaped tests identically
   on base and head.

## 10. Proposed map/reduce worker sizing

**Measured:** 14.0 ms per entry for `execute_process_v2_rebind_task` on the local
fixture (4 entries, short traces).

**This is a floor and must be read as one.** The fixture traces are short;
production traces are longer, and this repository has a recorded history of
component micro-benchmarks over-predicting path throughput by an order of
magnitude. Treat 14 ms as a lower bound and plan for several times that.

Against the declared production census of **695,638** source records across the
20 V1 lane/role tasks (about 34,782 each):

| entries_per_task | tasks | published objects | per-task at the floor |
|---|---|---|---|
| 64 (library default) | 10,870 | 32,610 | 0.9 s |
| 1024 | 700 | 2,100 | 14 s |
| **2048 (recommended)** | **340** | **1,020** | **29 s** |

**Recommendation: `entries_per_task = 2048`, `max_map_containers = 20`.**

Rationale: 2048 keeps the published object count near a thousand rather than
thirty thousand, which matters because each task publishes a directory of three
files by rename; and at a realistic 5-10x the measured floor it puts each task in
the 2-5 minute range, so a preemption loses minutes rather than an hour. At 20
workers that is 17 tasks per worker: roughly 10 minutes of wall clock at the
floor, and on the order of an hour under a conservative slowdown factor.

On the fan-out itself: the repository's cache and pack apps cap concurrent
Volume-v1 writers at 5, while its per-task-receipt inventory app uses 64. The
rebind publishes per-task directories by rename, so tasks never contend for one
destination, which is the same argument that justifies the inventory app's higher
figure. **20 is the owner's instruction rather than a measured limit**; the app
bounds and validates it rather than hard-coding it, so it can be lowered without
a code change if volume contention is observed.

## 11. Exact first launch command, for review

Not to be run without a prelaunch gate and the owner's authorization.

```bash
# 1. Freeze and inspect the plan. Writes nothing.
cd /private/tmp/compose-process-v2-evidence-chain
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src
.venv/bin/python scripts/plan_process_v2_rebind.py \
    --artifact-root /artifacts \
    --v1-payload-root <V1 migration payload root artifact path> \
    --entries-per-task 2048 \
    --dry-run

# 2. Launch, detached, from a clean committed worktree at the exact commit.
modal run --detach modal_apps/run_process_v2_rebind_app.py \
    --v1-payload-root <V1 migration payload root artifact path> \
    --expected-commit <full 40-hex commit of this branch head> \
    --entries-per-task 2048 \
    --max-map-containers 20
```

The launcher refuses a dirty or mismatched worktree before provisioning a
container, the driver prints its map plan before mapping, and the run publishes
nothing on an integrity mismatch.
