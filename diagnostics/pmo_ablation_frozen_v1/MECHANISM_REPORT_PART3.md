# Part 3 — the generic-module constructor, and exactly what is coordinated in the shipped arm

## §2 — `synthesize_dynamic_program`, the constructor PMO and T4 share

`control/dynamic_program_synthesis.py:563`. Module docstring: *"Route-free synthesis of
bounded executable molecular edit programs. The sampler owns no stored molecule, endpoint,
or complete program."*

**Vocabulary — 13 generic modules** (`GENERIC_MODULES`, :52):
`segment_grow, segment_shrink, segment_replace, substituent_delete, append_ring, fuse_ring,
functionalize, carbonyl_insert, heteroatom_substitute, bond_reroute, cycle_open, cycle_close,
ring_system_restate`.

**Algorithm**

1. `near_capacity = source.n_real_atoms >= 36` (`CAPACITY_AWARE_THRESHOLD`, :92).
2. Draw module count `K` from `module_count_distribution` over
   `(0.4, 0.4, 0.2)`, or `(0.15, 0.6, 0.25)` when near capacity (:67-68).
   `MAX_GENERIC_MODULES = 8`; counts past the declared table decay geometrically
   (`EXTENDED_MODULE_DECAY = 0.5`) rather than uniformly.
3. For each module position: optionally offer an explicit region-replacement option FIRST
   (`replacement_option_rate`), then walk `_weighted_module_order(rng, near_capacity)` — a
   weighted random permutation of all 13 families — and accept the FIRST family that
   (a) compiles on the CURRENT graph, (b) `extract_program`s cleanly, (c) keeps
   `len(marks) <= max_primitives` and `len(blocks) <= max_blocks`.
4. If nothing compiles at a position: stop early keeping what exists; if nothing ever
   compiled, raise.
5. Finally re-extract the whole program, `compile_program_graph`, `execute_program_graph`,
   and assert the scheduled replay reproduces the incremental endpoint exactly, else
   `RuntimeError("dynamic composition replay changed its exact endpoint")`.

**Near-capacity module reweighting** (`NEAR_CAPACITY_MODULE_WEIGHTS`, :95-110) is a real
coordination signal: at >=36 heavy atoms the lottery is tilted toward shrinking
(`substituent_delete` 6.0, `segment_shrink`/`segment_replace`/`heteroatom_substitute` 2.0)
and away from growth (`segment_grow`/`append_ring`/`fuse_ring`/`cycle_close` 0.5).

**Each module compiles against `current`**, the graph left by the previous module — so
modules are sequentially dependent on the evolving molecule, not planned against the source.

**Planned-vs-completed length is already recorded**, which is what the ablation needs:
`requested_module_count`, `completed_module_count`, and per-module `primitive_edits`
in `metadata["modules"]`. `intermediate_task_evaluations` is hard-coded `0`.

## The coordination inventory — what IS and IS NOT coordinated in the shipped arm

PRESENT, verified in code:

- typed persistent atom handles with static liveness checking (Part 1 §3.1-3.2)
- producer/consumer dataflow edges between blocks (`edit_program_graph.py:109`)
- RAW/WAW/WAR hazard detection, resolved by **serialization, never by pruning** (:136,:143)
- peak-capacity reservation along the schedule, `1 <= atoms+min_delta`, `atoms+peak_delta <= 40` (:174-175)
- capacity-conditioned module-count law and module-family weighting
- sequential compilation against the evolving graph
- exact endpoint re-verification between incremental build and scheduled replay
- per-primitive executor revalidation (`"deterministic serial; executor revalidation after every primitive"`)

ABSENT in the shipped arm — and this is the finding that most changes the interpretation:

**Modules do NOT share a growth site.** `focus_policy` is the mechanism that would make a
later module return to the anchor an earlier one grew from, and it is passed by **no
production caller**: the only occurrence outside `anchor_focus.py` is the internal forward at
`dynamic_program_synthesis.py:634`, and `AnchorFocusPolicy` has zero consumers anywhere.
Because `focus` is only populated `if focus_policy is not None`, it stays `None` for the
whole program and every module receives `focus=None`.

`anchor_focus.py`'s own docstring states the consequence, MEASURED:

> *"v1 draws every module's growth anchor uniformly and independently over the whole molecule
> and then walks the tip forward, so a K-module program is K mostly unrelated local edits
> rather than one coordinated construction. … A tert-butyl is executable from three
> `atom_insert` primitives on one anchor — verified by construction,
> `CNCc1ccccc1 -> CC(C)(C)NCc1ccccc1` — and was observed 0 times across 2,803 production
> draws on every one of five live molecules, at every module cap from three to eight.
> Raising the cap does not help, because each added module still redraws its own anchor:
> the defect is coordination, not program length."*

This is the **seventh** built-but-inert mechanism on this repository's record.

### Which optional hooks are live in the scored PMO runtime

| hook | live in scored PMO? | evidence |
|---|---|---|
| `region_law` | **yes, warm-memory lane only** | `pmo_online_memory.py:929` passes it |
| `region_law` (T4) | **yes** | `t4_fiber_campaign.py:285` passes it |
| `replacement_option` | **no** | passed only from `scripts/` gates and probes |
| `focus_policy` | **no** | no caller anywhere |

## §6 — corrected statement of what the PMO ablation isolates

The structured arm's coordination is **program-level dataflow and resource coordination**:
handle-correct construction, dependency and hazard ordering, and capacity reservation — plus
module-family composition under a capacity-aware lottery.

It is **not** site coordination. By the module's own measurement, consecutive modules act at
independently drawn anchors.

So the comparison must be reported as:

> structured, dependency-scheduled multi-module construction under a capacity-aware module
> lottery, versus uniform legal single-primitive editing, at a matched planned allowance

and it removes, together: joint multi-module planning; dataflow/hazard/capacity scheduling;
catalogue and donor information (jump-lane plan library, transplant donors); and program
reuse (`_recombine`). It removes **no learned scoring**, because there is none in either arm.
It also does **not** remove anchor sharing, because the shipped arm never had it.
