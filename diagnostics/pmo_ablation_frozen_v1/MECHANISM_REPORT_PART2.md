# Part 2 — the two program layers, the reference-scoring formula, and the task map

Code-grounded. Static analysis unless marked MEASURED.

## The central structural finding: there are TWO distinct program layers, and only one uses R_theta

| | **Dependency layer** | **Macro rollout layer** |
|---|---|---|
| modules | `control/edit_program.py`, `control/edit_program_graph.py` | `control/macro_engine.py` |
| unit | `EditProgram` = typed-handle marks partitioned into blocks | `MacroRollout` = L primitive edits under one macro family |
| coordination | persistent typed handles + dataflow deps + hazard serialization + peak-capacity reservation | family restriction + support truncation; **no handles, no dataflow graph** |
| R_theta | **absent** — "not samples from a certified R_theta law" | **present per step** — see formula below |
| on failure | raises; **never emits a partial endpoint** (`edit_program_graph.py:216-217`) | sets `halted` and **keeps the steps already executed** |
| consumers | PMO population controller, T4 fiber campaign | `option_selector`, `carbonyl_option`, `fused_option`, `ring_program`, `ring_expansion`, `option_continuation` |

Conflating these is the main way a reader could get the method wrong: a "macro" in the
option family is an R_theta-sampled rollout, while a "program" in PMO/T4 is a
dependency-scheduled construction with no reference law anywhere in it.

## §4 — the exact implemented reference formula (macro layer only)

`macro_engine.proposal_support` (:81): keep indices in the **global top-300 UNION the top-20
within each family** (`GLOBAL_CAP = 300`, `FAMILY_FLOOR = 20`, :78-79). Pure function of
(families, probs).

`macro_engine.macro_action_distribution` (:100):

    q = (1 - eps) * renormalised R_theta^(1/T)  +  eps * uniform over the same set
    T = 2.0, eps = 0.15

restricted to actions that are (a) in support, (b) in `MACRO_FAMILIES[macro]`, (c) `clean`.
Degenerate input (non-finite or zero-sum) falls back to exact uniform.

Answering the specific questions asked:
- **Not** a sum of log probabilities, **not** length-normalised. R_theta enters **per step as a
  sampling distribution**; the program as a whole is never assigned a reference likelihood.
- Probabilities are over **raw enumerated edit marks** (`families`/`probs`/`handles` are
  per-action), **not** canonical molecular successors.
- Temperature flattens the head; the **uniform floor eps is what makes the deep tail
  reachable** — the docstring states that tempering alone cannot lift a rank-137 action.
- Multi-edit aggregation for selection does not exist: `rollout` docstring (:161) —
  *"Execute `length` primitive edits under one macro. NO intermediate scoring… Purpose is
  applied to `.endpoint` by the caller."*

### Three kinds of rejection, separated in the code (not all "learned preference")

1. **Fixed representation/family restriction** — `proposal_support` truncation and
   `MACRO_FAMILIES[macro]` membership. Not chemistry, not learning.
2. **Chemical validity** — `gate_fn(y)` applied to the PRODUCT, lazily: draw, apply, check,
   then mask the rejected action and redraw (up to 12 attempts), renormalising each time.
   MEASURED claim in the module docstring: masked bad chemistry is ~1/3 of actions by count
   but **under 1.5% of R_theta mass**.
3. **Synthetic-suitability preference** — `prefer_fn(y)`, deliberately separate from the gate,
   with a second pass of up to 8 draws that relaxes the preference but keeps the gate, so
   *"a preference must never make a macro unexecutable."* Justified in-code by a measurement:
   0/67 bridgehead endpoints were T4-feasible vs 3/8 (37.5%) with neither bridgehead nor
   stereocentre, against a 3.1% baseline.

Per-step provenance is recorded for traces: `MacroStep(macro, smiles, action_index,
r_theta_prob, within_family_rank)` (:131-138).

Halt reasons, all retained on the rollout: `enumerate failed: <Type>`, `no legal successors`,
`no legal {macro} action`, `no clean {macro} action after 12 draws`.

## §1 — task to implementation map (in progress)

| Task | Implementation that runs | Program layer | Reference-model role | Shared with |
|---|---|---|---|---|
| PMO | `experiments/pmo_population_v1.py` -> `control/pmo_population_controller.py` (`PmoPopulationController(DynamicV21ProgramOptimizer)`) | dependency layer | **none** (MEASURED: 120-module closure by execution, zero `torch.load`, zero rate-model instantiation) | shares `synthesize_dynamic_program` with T4 |
| T4 lead optimization | `experiments/t4_fiber_campaign.py` | dependency layer | **none in the proposal law**; route-expert checkpoints are teacher-route-derived structural priors, not R_theta | shares `synthesize_dynamic_program` with PMO shallow lane |
| Fragment family (motif / decoration / superstructure / linker / morphing) | task set + metrics in `benchmark/fragment_constrained.py` (`FragmentTask` enum, 5 members); `benchmark/fragment_constrained_runner.py` self-describes as *"a deliberately small proposal-support **gate**, not an optimization or benchmark result"* | dependency layer (`EditProgram`) | runner states it *"calls no objective, quality metric, docking function, or learned reference law"* | — |
| QED / ring construction (option family) | `control/option_selector.py`, `option_policy.py`, `option_continuation.py` over `macro_engine` | **macro layer** | **R_theta per step**, formula above | `ring_program`, `ring_expansion`, `carbonyl_option`, `fused_option` |

NOTE on the fragment row: `SCAFFOLD_MORPHING` and `LINKER_DESIGN` **share one prompt column**
(`_TASK_COLUMN`, `fragment_constrained.py:44-52`) — *"GenMol and InVirtuoGen use the linker
prompts/results for scaffold morphing."* So these are not two independent samplers; one
prompt set produces both rows. That is a reuse-of-outputs relationship of exactly the kind
§1 asks to be stated explicitly.

## Still open
- Which implementation actually emitted the reported fragment shards (the runner above is a
  gate; the benchmark producer is not yet identified).
- The de-novo generator path and where the Lineage B checkpoint enters.
- §2 per-constructor semantics for `segment_replace` / `segment_grow` / structured / anchored
  / transplant / jump realizer.
- §5 executed traces per macro family from saved artifacts.
