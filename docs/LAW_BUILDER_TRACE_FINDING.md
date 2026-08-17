# Where a law call actually spends its time — and why the law-only builder is dead

**Status: the "build only the fields the law reads" branch is KILLED by measurement.**
Cost to find out: one 1-CPU Modal container (~$0.02) and one free local profile.

## What we believed

The profiler showed a 7.74 s law call spending 7.60 s (98%) inside
`prepare_factorized_mark_batch` and 0.042 s in the neural encode plus action
tables. A static AST pass then reported that of the batch's 43 fields,
`_encode_batch` reads 11, `_action_tables` reads 18, and **19 were read by
neither** — and those 19 were the ring-grow / ring-restate / graft / macro
fields, i.e. exactly the output of the expensive-looking call sites. The
inference was an encoder-style win: build the 24 fields the law needs, skip the
19 it does not.

Three premises. All three are false.

## 1. No field is unread

Recording every attribute read on the real batch during a real law call, with
frame attribution, gives **0 of 43 fields never read**. The static pass missed
consumers: `_coordinate_action` reads `graft_remove_neighbors` 333 times per
call and `ring_restate_actions` 22 times; `_ring_restate_logits` and
`_ring_restate_context_features` read the successor-group descriptors, ids and
multiplicities; `_graft_relation_features` reads `graft_mask`. Those are the
very fields the static pass proposed to delete. Deleting them would not have
crashed — `_coordinate_action` would have coordinated against absent structure,
which is a silently different law.

Fourteen fields are touched only by `_one_state_batch`, but that is
`dataclasses.replace` plus `.to(device)` plumbing, which reads every field. It
is evidence of nothing.

## 2. The expensive machinery never runs

The frozen model's actual configuration:

    compute_ring_system_delete   False
    compute_ring_grow_support    False
    compute_ring_restates        True
    compute_cyclic_graft         True
    compute_ring_opening         True

So `enumerate_clean_ring_system_deletes`, `enumerate_structured_ring_system_deletes`,
`_charge_preserving_macro_actions` and `ring_system_template_local_support_mask`
are **never called** — they do not appear in the timing table at all. The
machinery nominated as the thing to skip already costs zero. Skipping it saves
zero.

## 3. The cost is in two functions whose outputs ARE consumed

Per law call, measured on the container, mean over five real panel sources:

| producer | ms | share |
|---|---:|---:|
| `_semantic_cycle_close_admission_mask` | 2747.4 | **56.5%** |
| `_semantic_atom_restate_admission_mask` | 1626.4 | **33.5%** |
| `enumerate_ring_restate_semantic_groups` | 165.0 | 3.4% |
| `_semantic_cycle_open_admission_mask` | 142.7 | 2.9% |
| `compute_topology_features` | 34.4 | 0.7% |
| `_graph_application_masks` | 30.2 | 0.6% |
| `process_v2_atom_delete_mask` | 29.9 | 0.6% |
| sum of timed producers | 4776.4 | 98.3% |
| full law call | 4860.4 | 100% |

**Two functions are 90% of a law call.** Both produce fields
(`cycle_close_admission_mask`, `atom_restate_admission_mask`) that
`_action_tables` reads fifteen times each. They cannot be skipped. There is no
"build less" version of this call.

## What is actually inside the 90%

Both are thin wrappers that scatter an enumerated action set into a mask. The
cost is the enumeration. `enumerate_cycle_close_edges` walks every ordered pair
of real atoms crossed with the three micro bond classes and calls
`resolve_cycle_close_edge` on each — roughly 1,300 resolutions per state, of
which about 126 are admitted, so **90% of the work is spent rejecting**.

Profiled locally (no model needed, so this was free), mean per state:

    enumerate_cycle_close_edges          1582.9 ms
    enumerate_semantic_atom_restates      930.8 ms

    is_valid_state          2136 calls    689.7 ms   27.4% of both
    is_connected_or_null    2136 calls    155.6 ms    6.2% of both

`resolve_semantic_cycle_close` re-validates its INPUT state on every call:
`is_valid_state(state)` and `is_connected_or_null(state)`. The input state does
not change across the enumeration loop, and `prepare_semantic_cycle_close_context`
— which the loop already builds once and passes in — has itself already
required both. **46% of those 2,136 predicate calls are on the unchanged input
state**, recomputing an answer that a passed-in context has already established.

That redundancy is ~389 ms per state, about 15% of the two enumerators and 14%
of a law call: an exact, safe **1.16×**. Real, but not the ≥5× target on its own.
The remaining ~66% is genuine per-candidate resolution work.

## Correction to an earlier claim

I wrote that particle parallelism "would attack 1% of the cost." That was
wrong — it was reasoning from the neural encode being 1%. The 90% is per-particle
chemistry work and parallelises perfectly in principle.

It still is not the lever, for a different reason: the SMC app already runs
8 slots per container on 8 CPUs, so the cores are saturated at the slot level.
Parallelising particles would improve per-slot wall clock while leaving total
throughput unchanged, and for a 20-attempt benchmark throughput is what sets
both cost and finish time.

## What this leaves

Levers that reduce total compute rather than reshuffling it:

| lever | exactness | measured | status |
|---|---|---|---|
| law-only batch builder | — | — | **dead**, above |
| input-state validation memo | exact, no semantics touched | ~1.16× | ready to build |
| cross-attempt law cache | exact, no science change | 34% overlap | ready to build |
| cheap candidate pre-filter | **touches frozen chemistry semantics** | unknown, potentially large | needs a decision |

The pre-filter is the only remaining path to a large multiple: a cheap necessary
condition that rejects most of the ~90% of candidates that `resolve_*` rejects
anyway, before paying for the resolution. It is also the only one that edits the
frozen editing-process semantics, so it would need exact parity — identical
ordered marks and identical log probabilities on a broad banked-state sample —
rather than the trajectory-level check the other two need.
