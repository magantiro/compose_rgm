# De-novo ring dependency-block: costed retrain proposal

Status: **proposal only**. Nothing here has been launched. The acceptance
measurement it depends on is `acceptance_modal_n2000.json` in this directory.

## What changed and why

`compile_carbon_tree_to_target(..., event_schedule="ring_dependency_block")`
commits each ring system at the earliest graft prefix the executor accepts it
at, instead of deferring every ring transaction behind the whole graft phase
and the whole non-ring decoration phase.

The defect it addresses is a SUPPORT defect, not a reward or policy defect.
Measured on one real training trace (`diagnostics/denovo_schedule_probe_v1`),
the legal ring-template support collapses from 1034 templates to 295 at a
single `atom_restate` immediately before the first ring event, and small-ring
mass rises 0.1199 -> 0.2237 across that one step. Across the n=800 probe,
uniform small-ring mass tracks support SIZE
(`r(mass, log legal_template_count) = -0.87`) and not free slots
(`r = +0.13`, wrong sign).

## What the retrain requires

Lineage B's manifest records `require_path_cache: true` and
`path_cache_source_run: compose-v4-stage3-flexible-graft-prod-2be9258-v1`, so
training consumes PRE-COMPILED traces. Changing the schedule therefore means
recompiling the path cache; it is not a training-flag change.

Five additive plumbing hops, none of which exist yet:

1. recipe argument `event_schedule`
2. gate CLI flag in `train_tracelet_cnof_gate.py`
3. `build_tree_transport_path_records(..., event_schedule=)`
4. its two `compile_carbon_tree_to_target` call sites
5. Modal `build_tracelet_recipe_argv`

Hops 3 and 4 share a sink, so a single consultation test stays green when only
one is wired. Mutation-test at the CALL sites, per the region-law near-miss.

## Cost

| item | value | basis |
| --- | --- | --- |
| path-cache traces | ~100,000 | 50,000 train molecules x `tree_couplings_per_target: 2` |
| compile overhead | **1.087x** sequential | MEASURED, executor applications per compiled trace (50.2 -> 54.6, n=150 real train molecules). A load-independent counter: this machine is shared. |
| compile fan-out | CPU only, no GPU, no oracle | the original run used `path_workers: 16` |
| training | 3,000 steps, batch 64, lr 3e-4, warmup 500, seed 20260717 | `manifest.json -> recipe.arguments` |
| training hardware | 1x NVIDIA A100-SXM4-40GB | `manifest.json -> runtime.cuda_device_name` |
| training wall clock | **UNKNOWN** | the manifest records no duration; recover it from the run directory before quoting one |

The compile is the only new cost and it is ~9% on top of a compile the
original run already paid.

## What a retrain would and would not test

WOULD: whether supervising the ring decision on a large, undecorated carbon
skeleton moves the generated 3/4-ring rate toward the corpus rate of 5.26%.

WOULD NOT: anything about molecules whose ring system cannot be reordered.
Measured on 116 real molecules, **30% defer at least one ring system**, and a
deferred system keeps the legacy end-of-route position. A further ~2% fall back
to the sequential route entirely. Both are recorded per trace in metadata
(`ring_dependency_block_deferred`,
`ring_dependency_block_fell_back_to_sequential`) so the retrained corpus can be
stratified by them rather than reported as uniform.

## The finding that changes the recommendation

The dependency block is **statistically indistinguishable from the shipped
`exact_early_ring` scheduler** on the acceptance statistic. Matched three-arm
run, same molecules and same source draw (MEASURED, n=250 molecules / 590 ring
events, pinned kernel, zero oracle calls):

| arm | mean | median |
| --- | --- | --- |
| `sequential` (what Lineage B trained on) | 0.3089 +- 0.0069 | 0.2716 |
| `exact_early_ring` (shipped, never wired into training) | 0.1921 +- 0.0051 | 0.1381 |
| `ring_dependency_block` (this change) | 0.1957 +- 0.0051 | 0.1394 |

Paired, per ring event: repair minus shipped scheduler **+0.0037 +- 0.0017**,
with **515 of 590 events UNCHANGED**. That is marginally and detectably WORSE
than the shipped scheduler, not better -- about 2 standard errors, negligible
in size but stated with its sign rather than rounded to "equivalent". Repair
minus sequential is **-0.1132 +- 0.0071**, 462 improved against 96 worsened.

So the support gain is real and large against the corpus that was trained on,
and it is ALREADY AVAILABLE from a scheduler that shipped in July and was never
wired into the training recipe. The dependency block reproduces it at lower
compile cost and by construction rather than by 14,897 executor-verified
adjacent swaps per 800 traces, but it does not add support quality on top.

**Consequence: whichever schedule is chosen, the plumbing is the deliverable,
not the algorithm.** Do not present the dependency block as a further gain over
the shipped scheduler.

## Where the residual lives, exactly

Per ring-system ordinal within a molecule (MEASURED, block arm vs sequential):

| ordinal | n | dependency block |
| --- | --- | --- |
| 0 | 244 | **0.1388** |
| 1 | 194 | 0.1905 |
| 2 | 109 | 0.2755 |
| 3 | 37 | 0.3260 |
| 4 | 6 | 0.4274 |

The FIRST ring system of a molecule passes the <0.15 gate. Every later one
fails, and the repair's benefit decays to nothing by the fifth. The mechanism
is direct: committing a ring system constricts the legal ring-template support
the next one is decided against, and no reordering can avoid that in a
sequential trace with more than one ring system.

A schedule cannot fix this. Closing it needs either a support-level change
(what the catalog offers at a post-ring state) or an acceptance of the scope.

## Preconditions before spending GPU

1. **The acceptance gate FAILED.** 0.1957 +- 0.0051 against a threshold of
   0.15, on the mean the gate is stated over. It passes on the median (0.1394)
   and the FIRST ring system of a molecule passes on the mean (0.1388).
   Endpoint exactness held 250 of 250 in every arm. Do not spend GPU on the
   strength of the median alone.
2. The five plumbing hops wired AND mutation-tested at the call sites.
3. A decision on the residual: the repair is strongest on the FIRST ring system
   of a molecule and degrades for later ones, because each commitment
   constricts the support the next is decided against. If the aggregate misses
   the gate while the first-system figure passes it, that is a scoping
   question, not a reason to retune.
