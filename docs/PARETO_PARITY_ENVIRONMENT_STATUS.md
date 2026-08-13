# Local parity environment for the Pareto fan-out — status

**Purpose.** A parallel implementation that produces claim-bearing numbers may
not ship on fixture parity alone. Fixtures find races; only replaying the
committed 12-source smoke against the **real** checkpoint shows that the real
model on real successor fibers takes identical actions. That requires the real
runtime here.

## Pulled from the volume and verified

| artifact | size | state |
|---|---:|---|
| `runs/run_v2_01/R_THETA_CHECKPOINT.pt` | 84 MB | present |
| `local_runtime/materialized_scorer/` | 49 MB | present, 3 files |
| `local_runtime/active8/<sha>/` | 751 MB | **1,641 files, 0 empty, 328 task dirs × 4 files — matches the volume layout exactly** |
| `local_runtime/gate_zero_v6/DECISION.json` | — | present |
| `local_runtime/pareto_control_smoke/` | — | **12 shards — the SERIAL BASELINE the replay must reproduce** |

The volume's Active8 is the authoritative one. `RUN_PATHS.json` carries an
explicit warning that the repo's local Active8 copy **differs**, and that Gate-0
**v6** is the decision matching the volume stream while the locally generated v7
matches the local copy. A parity replay built on the repo's Active8 would
authenticate against the wrong stream. Every path above points at the
volume-sourced copy.

## Blocked on exactly one hash

`open_process_v2_t1_source` authenticates Active8 → Gate-0 → plan → policy. Five
of its six identity fields match:

| field | result |
|---|---|
| `process_identity_sha256` | **OK** |
| `contracts_binding_sha256` | **OK** |
| `gate_zero_structural_contract_sha256` | **OK** |
| `active8_completion_sha256` | **OK** |
| `active8_sentinel_sha256` | **OK** |
| **`source_index_sha256`** | **MISMATCH** — decision `b5d042a0…`, locally computed `adb02a19…` |

Local stream resolves to **328 shards, 270 eligible** (role census: train 270,
controller_validation 22, final_test 20, validation 16 — 328 total, consistent).
Ruled out by measurement:

- **not a truncated download** — 1,641 files, none empty, 4 per task dir exactly
  as the volume lists;
- **not a repo-commit dependency** — the worktree at `7deee51` and the current
  main worktree compute the *same* local index `adb02a19…`, so the divergence is
  not in repo-side contract files;
- **not the Active8 artifacts themselves** — the completion and sentinel hashes
  both match what Gate-0 recorded.

So the Active8 content is right and the index derived by walking it is not.

## Why I stopped rather than bypassing

The natural shortcut is to patch `source_index_sha256` in the decision, on the
argument that the guard is *provenance* rather than a numerical input — the
weights come from the checkpoint under `load_state_dict(..., strict=True)`, and
the replay itself is the stronger provenance test, since a wrong runtime would
not reproduce the committed trajectories.

That argument is not wrong, and it is still the wrong thing to do here.
Attempting it hit, in order: a canonical-JSON framing guard, then a
`decision_sha256` **self-hash over the decision body**. The chain is closed by
construction.

> **Three integrity guards deep is a signal to stop, not to patch the fourth.**

A parity harness built by defeating an integrity chain would be a harness whose
own provenance is the weakest link in the claim it exists to protect. The fix is
to obtain the Gate-0 decision that matches this stream — or to determine why the
index walk diverges — not to route around the check.

## What unblocks it

One of:

1. **The matching Gate-0 decision.** `source_index_sha256 = b5d042a0…` was
   computed somewhere; if a v6-equivalent decision exists on the volume for this
   exact stream, fetch it and the chain closes with no code change.
2. **The index walk diagnosed.** Compare the local per-shard
   `(task_identity_sha256, decision_shard_sha256)` list against the reduction
   the decision was built from. A single differing or extra shard is enough, and
   this is a cheap comparison once the expected list is in hand.
3. **Run the parity replay on Modal instead**, where the chain already
   authenticates, and pull only the resulting shards for comparison. This costs
   compute but requires no bypass.

**Recommendation: (2), then (1).** Both are cheap and neither weakens the chain.
(3) is the fallback and should be chosen deliberately, not as a way of avoiding
the diagnosis.

## Unchanged by any of this

The acceptance bar for shipping the fan-out is the same:

> exact equality against the committed serial shards on `action_sequences`,
> `endpoints`, `endpoint_z`, `overrode_greedy`, `decisions`, **and every resource
> counter** — `kernel_calls`, `oracle_requests`, `native_oracle_calls`.

Counters are in the list because a shared enumeration cache makes `kernel_calls`
non-deterministic under naive concurrency, and `kernel_calls` feeds
`trajectories_for_kernel_budget()`. See
`docs/PARETO_DEVELOPMENT_PREREGISTRATION.md` §5.
