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

## RESOLVED — the diagnosis, with no guard bypassed

**The Gate-0 `source_index_sha256` embeds the absolute mount path.**

`src/compose_v4/data/editing_v2_process_v2_gate_zero.py:593` hashes a body whose
first field is the run root as a *string*:

```python
body = {
    "active8_run_root": str(root),          # <-- absolute filesystem path
    "active8_completion_sha256": ...,
    "shards": shards,
    "eligible_task_identities": sorted(eligible),
    ...
}
index_sha256 = canonical_sha256(body)
```

Confirmed by direct reconstruction — every other field held at its locally
computed value, only the root string swapped:

| root string used | index hash |
|---|---|
| `/Users/rmaganti/.../local_runtime/active8/<sha>` | `adb02a19…` |
| `/artifacts/editing_v2/process_v2_active8/<sha>` | **`b5d042a0…` — EXACT MATCH** |

So the content is **identical**: all 328 shards, every eligible task identity,
the role census, the completion and the sentinel. The tree is simply mounted
somewhere else, and the index hash is not content-addressed.

### This is a real portability defect, and it should be recorded as one

A "source index" digest that changes with **where you mounted the tree** cannot
verify content across environments — which is the one job that digest exists to
do. The design already anticipated relocation *for the plan*:

```python
mounted_root = mounted_process_v2_artifact_path(
    str(plan["run_artifact_root"]), artifact_root=artifact_root, ...)
```

`artifact_root` remaps the plan's recorded root onto a local mount. **The same
remap is not applied to the index body.** That asymmetry is the defect, and it
means any future attempt to verify these artifacts off Modal hits this same wall.

Fixing it properly means either dropping `active8_run_root` from the hashed body
or routing it through the same remap — both of which change the semantics of a
**frozen verification artifact** and would invalidate the recorded decision
hash. That is a deliberate change requiring its own review, not something to do
in passing while chasing a speedup.

## Consequence: parity replay runs on Modal

Not as a fallback guess — as the informed choice. The content is verified
identical, the sole obstacle is a non-portable path embedding, and Modal is
where the tree lives at the path the digest names. Running it there satisfies
the check **as written** rather than around it.

The local runtime stays: it is a working environment for everything that does
not need the authenticated chain, including scorer benchmarks and fixture-level
race testing of the compute-once cache.

## What this was NOT

Three hypotheses were on the table before the diagnosis. Recording which failed
matters, because each would have implied a different and more alarming problem:

| hypothesis | verdict |
|---|---|
| a truncated or partial download | **wrong** — 1,641 files, none empty, 4 per task dir exactly as the volume lists |
| a repo-commit dependency in the eligibility rule | **wrong** — the `7deee51` worktree and main compute the *same* local index |
| a differing or extra Active8 shard | **wrong** — the completion and sentinel digests both match; content is identical |

None of them was the answer, and the real cause was benign. That is worth
stating plainly: the chain did not detect corruption. It detected relocation,
and it could not tell the difference — which is exactly the defect above.

## Unchanged by any of this

The acceptance bar for shipping the fan-out is the same:

> exact equality against the committed serial shards on `action_sequences`,
> `endpoints`, `endpoint_z`, `overrode_greedy`, `decisions`, **and every resource
> counter** — `kernel_calls`, `oracle_requests`, `native_oracle_calls`.

Counters are in the list because a shared enumeration cache makes `kernel_calls`
non-deterministic under naive concurrency, and `kernel_calls` feeds
`trajectories_for_kernel_budget()`. See
`docs/PARETO_DEVELOPMENT_PREREGISTRATION.md` §5.
