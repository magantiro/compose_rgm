# Corrections and bindings, in response to the five directives
All statements MEASURED from code or saved artifacts unless marked INFERRED.
No docking rerun was launched. No dormant mechanism was activated.

---

## 1. Arm A reuse and the structured-vs-uniform-chain experiment: UNCHANGED

Kept as frozen in `frozen_config_v1.json`. Nothing is restarted because a different audit
inspected a different configuration.

Standing basis for the reuse, re-stated: across the three commits behind the existing
no-prescreen 1k table (`3e6baf94`, `c7fdddca`, `0b9666f4`), **zero files on the scored PMO
import closure differ** — the diffs are a Modal app wrapper, a canary script, a launcher and
diagnostics JSON. Arm A is therefore one configuration and is reusable; arm B is the only new
spend (18 campaigns, ~18,144 charged calls, seed-matched per task).

---

## 2. CORRECTION: "no learned scoring" was WRONG — narrow it to the reference model

**What I wrote:** "It removes no learned scoring, because there is none in either arm."

**What the code says.** `pmo_population_controller._credit_allocate` docstring (:584-589):

> *"The cell share comes from `PopulationCredit.allocate`… The credit decides WHERE to spend;
> **the fitted program value decides WHICH candidate inside the drawn cell**."*

and the fit is online and real:

| element | location |
|---|---|
| `_fit_value()` -> `ProgramValue(penalty=1.0)`; `value.fit(features, improvements)` | `pmo_population_controller.py:446,459,461` |
| `ProgramValue` class, `.fit`, `.predict` | `control/fiber_control.py:111,135` |
| feature vector `population_features` (16-dim) written onto each candidate as `fiber_features` | `pmo_population_controller.py:135-168`, set at :438 |
| value gates the allocation role: `allocation_role = "fiber_control" if value.weights is not None else "exploration"` | :542 |
| value used to rank inside a credit cell: `if value.weights is not None: … value.predict(features)` | :604-608 |

**CORRECTED CLAIM.** The scored PMO controller has **no learned REFERENCE model (R_theta) in
proposal construction** — that remains MEASURED (120-module closure taken by execution, zero
`torch.load`, zero rate-model instantiation). It **does** have an online-fitted linear
program-value model over `fiber_features`, fit on observed improvements, plus
`PopulationCredit` allocation.

**CONSEQUENCE FOR THE ABLATION.** Both arms must keep that machinery intact. Arm B therefore
must produce candidates carrying a well-formed `fiber_features` vector, which it does by
construction: the chain is materialised as a real `EditProgram` via `extract_program`, so
`primitive_edits`, `block_count`, the retention fields and the five-rule histogram are all
defined, and the channel one-hots take an existing valid encoding. No zero allocation, no
missing-feature penalty, no broken score.

---

## 3. Dormant mechanisms stay dormant; the proposer described as it runs

`focus_policy` is NOT activated. Nothing else dormant is activated.

The structured proposer as it actually runs (`synthesize_dynamic_program`,
`control/dynamic_program_synthesis.py:563`):

- **Module construction.** `near_capacity = n_real_atoms >= 36`. Module count K drawn from
  `(0.4,0.4,0.2)`, or `(0.15,0.6,0.25)` near capacity; `MAX_GENERIC_MODULES = 8` with
  geometric decay past the table. At each position, walk `_weighted_module_order` — a weighted
  random permutation of all 13 `GENERIC_MODULES` — and accept the FIRST family that compiles
  on the CURRENT graph, extracts cleanly, and fits `max_primitives=32` / `max_blocks=8`.
  Near capacity the lottery tilts toward shrinking (`substituent_delete` 6.0) and away from
  growth (`segment_grow`, `append_ring`, `fuse_ring`, `cycle_close` 0.5).
- **Information sources.** The current graph only, for the shallow/structured lanes. The jump
  lane additionally consumes a teacher-derived plan library
  (`diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json`,
  `fit_scope: shared_all_routes`). Transplant consumes another archive molecule's structure.
- **Program reuse.** `ProgramOptimizer._recombine` / `_mutate` over the online archive;
  `FRESH_SYNTHESIS_PROBABILITY = 0.5` splits fresh synthesis from reuse in the structured lane.
- **Actual scheduling behaviour.** `compile_program_graph` builds per-block footprints,
  producer/consumer dependency edges, and RAW/WAW/WAR conflict edges resolved by
  **serialization, never pruning**; `_topological` schedules greedily subject to
  `1 <= atoms+min_delta` and `atoms+peak_delta <= 40`; execution is **deterministic serial with
  executor revalidation after every primitive**, and a program that cannot be scheduled raises
  rather than emitting a partial endpoint.
- **Live optional hooks in scored PMO:** `region_law` yes (warm-memory lane,
  `pmo_online_memory.py:929`); `replacement_option` no (scripts only); `focus_policy` no.

---

## 4. CORRECTION to the §5 trace, plus channel contributions and ancestry

**Scope correction.** The saved trace
(`scored_ranolazine_mpo_seed20274925_20260924T202214Z`, round 31,
`structured_program_channel`) demonstrates **WITHIN-BLOCK molecular construction**. Its blocks
are `substituent_delete` (marks 1-8), `construct_substituted_ring` (9-13),
`ring_path_remodel` (14-15). The handle chain `created 0 -> 1 -> 2 -> 3` and the closure
`cycle_close(input 9, created 3)` all sit **inside block 2**.

Its three block footprints are disjoint on inputs (0-7 / 8-9 / 10-12), so this program has
**ZERO cross-block producer dependencies and zero serialization edges**. It is evidence that
the handle mechanism coordinates construction within a block; it is **not** evidence that
these blocks carry cross-block dataflow.

The molecular content stands: 40 -> 35 heavy atoms; four inserts build an N-C-C-N chain off a
retained atom and mark 13 closes it onto another retained atom, producing the `C3NCCNC32` ring
in the endpoint.

**Proposal-channel contributions** (same campaign, 63 rounds, MEASURED):

| channel | eligible pool | charged | selection enrichment |
|---|---|---|---|
| `structured_program_channel` | 3,760 (47.1%) | **634 (63.8%)** | 1.35x its pool share |
| `shallow_program_channel` | 4,212 (52.8%) | 359 (36.2%) | 0.69x |
| `joint_dependency_region_jump` | 8 (0.1%) | **0 (0.0%)** | never selected |
| `anchored_replacement_channel`, `transplant_program_channel` | absent | absent | not active in PMO |

**This matters for §6:** the teacher-derived plan library contributed **0 of 993 charged
calls** in this run. Its catalogue information is present in principle and inert downstream
here, so "the ablation removes catalogue information" is true of the configuration and
near-vacuous in realized effect for this task and seed.

**Final-result ancestry.** Best molecule `0.7949`
`CCCCCCOC(CCCCC(N)=O)CNc1cc(SC=CC(=O)C(C)C2=C(F)C=CC2)c(N)cc1C`, found in round 33 from
`structured_program_channel`, with `ancestral_primitive_edits = 47` accumulated over a
multi-generation lineage (`construction_ancestry` ancestors at 15, 11, … primitive edits each,
each carrying its own endpoint, `entry_id` and `source_state_sha256`). The winner is the
product of accumulated program reuse, not of one large program.

---

## 5. T4: the reported table bound to its campaign and library

**The winner-route library is real and it IS upstream of the T4 route experts.**
`tools/t4_route_distillation.py` — docstring *"Fit and audit the **answer-known** T4
route-distilled proposal policy"* — sets `WINNER_PAIRS = diagnostics/ivg_winner_paths/pairs`
(:59), reads them (:112), and `_teacher_traces` selects exactly the 77
`compiled_complete_transformation` teachers that `tools/t4_shared_retained_rewrite_gate.py`
then fits (`len(routes) != 77 or … != 15` guard). `diagnostics/ivg_winner_paths/audit.json`
records `origin: github.com/invirtuolabs/InVirtuoGen_results`, NC licence, 176 pair files.

**But it does NOT reach the reported table.** Two independent facts:

1. **Every one of the 16 replicate-1 contracts pins only the held-out expert.**
   Scanned `configs/t4_held_target_distilled_*.json`: `perTarget = 1`, `sharedAll77 = 0`, on
   all 16. The pinned artifact is
   `diagnostics/t4_held_target_distillation_quality_v1/<target>_checkpoint.json`, whose
   `split_audit` reports `split: leave_one_target_out` and
   `held_target_absent_from_training: true` for all five targets
   (fa7 59/18, braf 65/12, jak2 58/19, parp1 61/16, 5ht1b 65/12).
2. **The distilled artifact is general, not answer-bearing.** The expert payload is
   `templates` (`input_atoms` / `input_bonds` / `output_atoms` structural templates) plus a
   `marginal` (template probabilities, goal-count probabilities, `exploration_floor` 0.10),
   with `runtime_target_conditioning: false`. No endpoint or target molecule is stored.

Corroborating MEASURED evidence that it could not have moved the numbers anyway: the
`route_complete_region` lane returns **0 eligible endpoints in 8 of the 9 braf/fa7/5ht1b
cells** (5ht1b_1 returns 1), while every control cell survives on the model-free `shallow`
lane alone.

**VERDICT: the winner-route library issue does NOT apply to the reported T4 experiment.**
No docking rerun is warranted, and none was launched.

### One thing it DOES apply to — the in-flight replicates

The unified-controller replicate contracts pin a SECOND expert:

    t4_unified_controller_<target>_<delta>_r23_v1.json
      rung_zero_route_expert : diagnostics/t4_shared_retained_rewrite_v1/checkpoint.json
      round_one_route_expert : diagnostics/t4_held_target_distillation_quality_v1/<target>_checkpoint.json
      pinned -> perTarget=1  sharedAll77=1

`t4_shared_retained_rewrite_v1/checkpoint.json` carries `training_routes: 77`,
`training_scope: all_locked_t4_routes_shared_task_independent`, **no `split_audit` and no
holdout** — so for r2/r3 the evaluated target's own winner routes ARE in the rung-0 expert's
training set. `runtime_target_conditioning` is still `false` and the artifact is still
templates + marginal.

This does not affect the frozen table. It affects the replicate panel now running. Options,
for an owner decision rather than an agent one:
(a) report r2/r3 with the rung-0 expert disclosed as a declared difference from replicate 1
    — they are already a separately named phase;
(b) scope the replicate claim to reproducibility of the UNIFIED controller rather than of
    replicate 1's information conditions;
(c) re-fit the shared expert leave-one-target-out — moves every r23 payload hash and forfeits
    the ~1,500 charged calls already spent. Not recommended.
