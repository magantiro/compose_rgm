# Part 4 — §1 the complete task map, and §2 the remaining constructors

## §1 — reported task -> implementation that produced the numbers

| Task | Producer | Program layer | R_theta role | Shared with |
|---|---|---|---|---|
| **De novo** | `modal_apps/denovo_official_eval.py` (branches `denovo-generation-20260920`, `denovo-ring-marginal-20260922`) | none — direct CTMC sampling | **R_theta IS the entire generator.** `sample_tracelet_ancestral` from `DegreeBoundedCarbonTreePrior`, *"no target, no source molecule, and no controller"*. Checkpoint `lineageB/checkpoint.best_so_far.pt`, sha `c9d92751…`; typed ring catalog **deserialized from the checkpoint payload**, which the app notes is strictly stronger than the frozen-fingerprint check | — |
| **Fragments** (motif / decoration / superstructure / linker / morphing) | `modal_apps/fragment_pinned_sweep_app.py` -> subprocess `tools/run_fragment_constrained_suite.py` (fragment-* branches) | dependency layer (`EditProgram`) under a learned sampler | **R_theta present** — `load_factorized_rollout_checkpoint(ringcore_a7546e2_best.pt)`. Arms `frozen_sampler_baseline` vs `attachment_control` (+ `path_program`) | linker and morphing share one prompt column |
| **PMO** | `experiments/pmo_population_v1.py` -> `control/pmo_population_controller.py` | dependency layer | **none** (MEASURED: 120-module closure by execution; zero `torch.load`, zero rate-model instantiation) | shares `synthesize_dynamic_program` with T4 |
| **T4 lead optimization** | `experiments/t4_fiber_campaign.py` | dependency layer | **none in the proposal law**; route-expert checkpoints are teacher-route-derived structural priors | shares `synthesize_dynamic_program` with PMO shallow lane |
| **QED / ring construction** | `control/option_selector.py`, `option_policy.py`, `option_continuation.py` over `macro_engine` | macro rollout layer | **R_theta per step**, `(1-eps)·p^(1/T) + eps·uniform`, T=2.0, eps=0.15 | `ring_program`, `ring_expansion`, `carbonyl_option`, `fused_option` |

Three consequences worth stating plainly:

1. **R_theta's role is NOT uniform across tasks.** It is the whole generator for de novo, a
   per-step sampling law for the option/QED family, the sampler for fragments, and **entirely
   absent** from PMO and T4 proposal construction. Any sentence of the form "COMPOSE samples
   from its learned reference" is true of three task families and false of two.
2. **The reported de-novo and fragment numbers were produced on other branches**, not on the
   T4/PMO working branch. Their producers are named above.
3. **CLAIM-VALIDITY FLAG:** the fragment sweep pins `ringcore_a7546e2_best.pt`, which
   `configs/ringcore_v1_checkpoint_selection.json` marks `PROVISIONAL_EDITING_CHECKPOINT` and
   **forbids for frozen results** (selected by hazard-inclusive GM loss). That is fine for a
   mechanism probe and needs an explicit decision before it backs a published fragment row.

## §2 — the remaining constructors

### `_transplant_proposal` (`dynamic_program_synthesis_v21.py:105`)
*"Transplant a region from another ARCHIVE molecule into this parent."* Donor pool = every
OTHER archive entry's source plus a `_donor_reservoir()`; `transplant_program` attempts 24
bindings under the same `max_primitives`/`max_blocks` limits. Carries a **per-source donor
exclusion memory** (`optimizer._transplant_seen`, keyed by `entry_id`) so a deterministic
ranking cannot hand the same parent its own top transplant every round. Information consumed:
another molecule's **structure**, from the run's own scored archive.

### `realize_plan_binding` / `bind_realized_plan` (`control/pmo_realization.py`)
The jump lane's binder, and the most strongly coordinated constructor in the repo.
*"Given a parent molecule and one joint role plan — a sequence of ~16-31 typed roles, each
naming an executor rule, its parameters, and a per-operand descriptor — this module finds a
legal executable program on that parent that realizes EVERY role, in order, with the plan's
declared dataflow intact."*

- Replaces a width-4 beam. *"this search discards nothing for ranking — it orders children by
  one-step operand availability and keeps all of them reachable until an explicit budget is
  spent. A budget hit is reported as its own outcome and never as incompatibility."*
- The plan is drawn from the teacher-derived library
  (`diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json`,
  `fit_scope: shared_all_routes`) — the one genuine catalogue input on the PMO path.
- Outcome vocabulary: `proven_incompatible` / `search_budget_exhausted` /
  `completed_realization`.

### Constructor taxonomy, answering §2's question with the code's own semantics
- `synthesize_dynamic_program` — **samples primitives while maintaining a higher-level plan**
  (K modules, each a parameterized family compiled against the evolving graph).
- `_transplant_proposal` — **copies a donor subgraph and constructs its attachment**.
- `realize_plan_binding` — **chooses a declared role/dataflow target and searches for a route
  that realizes it** on this parent.
- `ProgramOptimizer._recombine` / `_mutate` — **recombines branches from, or mutates, earlier
  programs** held in the online archive.
- `macro_engine.rollout` — **extends a single-family rollout under an R_theta-tempered law**,
  no plan object at all.

These are genuinely different branches and should not be collapsed into one "macro" notion.
