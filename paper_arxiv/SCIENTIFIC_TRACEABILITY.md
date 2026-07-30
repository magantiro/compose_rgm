# Scientific traceability

This note records the source and wording boundary for each load-bearing claim in
the current manuscript. It is an audit aid, not submission prose.

| Manuscript claim | Authoritative basis | Boundary enforced |
|---|---|---|
| The learned object is a marked stochastic rewrite process. | Pasted authoritative handoff §§3.2-3.4; `docs/HANDOFF_COMPOSE_TRACEABILITY_2026-07-29.md` §§3.2-3.4. | The manuscript does not call the completed diagnostic run successor-trained. |
| The semantic state space is a disjoint union across active cardinalities. | Pasted handoff §3.1; traceability §3.1. | Persistent padding is described as coordinates, not semantic fixed dimension. |
| The null state is not a molecule and is a reversible source. | Pasted handoff §§1, 3.1; traceability §§1, 3.1. | “Every state is a molecule” never appears; the claim is restricted to non-null committed states. |
| Molecular transitions are executable legal rewrites. | Pasted handoff §§3.2, 5, 7; system contract. | Valid graph states are not claimed to be synthetically accessible molecules. |
| The molecular kernel is the pushforward through execution and canonical identity. | Pasted handoff §4; traceability §4. | `segmented_successor.py` is not elevated from implementation to scientific definition. |
| Generic pushforward Generator Matching is prior work. | Billera, Nordlinder, and Murrell (2026), arXiv:2605.20547, especially Theorems 3.2 and 3.12. | COMPOSE does not claim the generic image-process theorem. Its selected-mark run and normalized productive-successor objective are not described as satisfying that theorem without separately checking its generator, loss, and regularity assumptions. |
| Refinement invariance is successor-level and within-fiber. | Pasted handoff §4. | Mark-level top-k, power, nucleus, and family tilts are explicitly excluded from the invariance claim. |
| Exact finite-horizon control is bounded-slice exact. | Pasted handoff §§8, 16-17, 20. | No full-scale exactness claim is made; learned molecular-scale values are called approximations. |
| Editing and de novo generation are separate regimes. | Pasted handoff §6; traceability §§6, 19-20. | The editing checkpoint is not used as evidence for timed unconditional generation. |
| The completed 16k run used mark-level training. | Pasted handoff §§3.4, 10-13; traceability §§3.4, 19. | A successor-aware repaired protocol is described as new, not as a reinterpretation of the old run. |
| Novelty lies in the formal composition, not individual ingredients. | Pasted handoff §§2, 19-20. | No “first” claim is made for atom edits, valid editing, Generator Matching, graph grammars, Pareto optimization, or Doob control. |
| The raw canonical pushforward may contain molecular self-event mass. | `src/compose_v4/experiments/production_successor_kernel.py`; `src/compose_v4/experiments/factorized_successor_training.py`. | The raw pushforward is denoted \(\bar p_\theta\), not presented as the fixed-budget editing kernel. |
| The productive editing kernel conditions on a non-self molecular jump. | `src/compose_v4/experiments/successor_kernel.py`; `src/compose_v4/experiments/factorized_successor_training.py`. | The denominator \(1-\bar p_\theta(x\mid x,t)\) is explicit, and the zero-productive-mass state is terminal. |
| The total marked-event hazard can differ from the molecular exit hazard. | `src/compose_v4/experiments/factorized_successor_training.py` (`productive_hazard`). | The molecular exit hazard is written as total hazard times productive mark mass. |
| Productive-successor identity loss is embedded-chain likelihood, not rate matching. | `src/compose_v4/experiments/factorized_successor_training.py`; `src/compose_v4/experiments/factorized_successor_objective.py`. | The manuscript does not call `productive_identity` Generator Matching or claim a hazard gradient. |
| Successor-generator Bregman loss matches the off-diagonal canonical generator. | Same objective sources as above. | It is distinguished from both productive identity and identity-plus-hazard. |
| The successor-aware bridge is development-bounded. | `src/compose_v4/experiments/editing_successor_trainer.py`. | The manuscript explicitly says the bridge does not authorize a full-corpus run. |
| The budget-indexed Doob law realizes the exact terminal tilt. | `docs/SYSTEM_CONTRACT.md` Appendix A; direct telescoping argument in Eqs. (path-telescope)--(terminal-tilt). | Exactness requires exact backward values and is claimed only at the declared horizon. |
| Doob control cannot create an unsupported molecular transition. | `docs/EXPERIMENT_INFRASTRUCTURE_PLAN.md` §2.2; follows directly from the multiplicative controlled kernel. | Support equality is not claimed under hard conditioning; zero-value successors can be pruned. |
| A state with zero remaining-budget value is unreachable under the declared target and budget. | Pasted handoff §8; `docs/HANDOFF_COMPOSE_TRACEABILITY_2026-07-29.md` §8. | The controlled row is declared undefined; no epsilon patch is licensed. |
| Dynamic retargeting is exact as a continuation from the current state. | `docs/SYSTEM_CONTRACT.md` Appendix A; telescoping argument restarted at the current state and remaining budget. | It is not claimed to reproduce the counterfactual law obtained by using the new objective from the original source. |
| General path constraints require killed or augmented-state semantics. | `docs/PAPER_MASTER_PLAN.md` §7. | A terminal desirability is not claimed to enforce arbitrary intermediate constraints. |
| Molecular-scale value control is approximate. | Pasted handoff §§8 and 16; experiment registry E7. | Exactness remains confined to the bounded enumerable verification slice. |
| The production molecular representation is 2D, broad-organic, persistent-slot, charge-preserving, and bounded at 40 active atoms. | Pasted authoritative handoff §§5.1 and 18; `configs/editing_gate_zero_runtime_v2.json`; active-element predicates in production state handling. | The manuscript does not claim stereochemical, isotopic, radical, protonation-state, 3D, or synthetic-accessibility support. |
| The bounded editing rebuild exposes eight active operator families. | `configs/editing_gate_zero_runtime_v2.json`; `configs/editing_t1_successor_gate_v3.json`; `diagnostics/coherence/editing_t1_successor_panel_v3_active8_2026-07-30.json`. | The paper describes the bounded rebuild, not a final trained checkpoint. `ring_system_grow` and `ring_system_delete` remain disabled; training authorization remains false until gates pass. |
| Atom insertion supports root and one-neighbor birth, not general multi-neighbor insertion. | Pasted handoff §5.3; active action factorization and Gate-0 contract. | Birth/death is trans-dimensional, but one-step inverse closure is not claimed for every deletion. |
| Ring-system restatement is currently a necessary coordinated accelerator under the primitive bond-reorder mask. | `diagnostics/coherence/ring_restate_primitive_path_audit_v1_2026-07-30.json` and its audited implementation. | The qualitative support conclusion is used; development counts are not promoted to paper results. Ring growth is not restored, and ring-system deletion remains excluded pending a separate reachability justification. |
| Corpus construction is capability-directed and leakage-aware. | `configs/editing_corpus_v2_contract.json`; Section 4 corpus contract. | Final lane counts, sampling coefficients, and checkpoint identity remain `\resultpending` fields until their frozen artifacts exist. |
| Long training is gated by deterministic, support, successor, and pilot checks. | Active Gate-0/T1/P50 contracts and the registered P500/P2000 protocol. | Passing development infrastructure does not authorize a long run; thresholds and training authorization must be frozen independently. |
| Experimental comparisons are divided into law-only, support-ablation, and external task-level comparisons. | `configs/experiment_registry.yaml`; `docs/EXPERIMENT_INFRASTRUCTURE_PLAN.md`. | External methods are not described as support-matched unless they actually share the production successor graph. |
| The exact verification graph has 967 states and 14,432 canonical directed edges. | Pasted handoff §§16-17 and the registered `carbon_6_slots` benchmark identity. | These are benchmark-identity facts, not chemical-performance results; the slice is carbon-only and non-representative. |
| Every rendered `XXX` is an unresolved empirical value or conclusion. | `COMPLETION_PLAN.md`; keyed `\resultpending{...}` calls in the section sources. | A placeholder is filled only from a frozen provenance-validated artifact; development-only E5 and T1 diagnostics cannot silently fill paper results. |

## Empirical-status policy

The current manuscript contains result tables and a complete interpretation
structure, but no filled endpoint-quality, likelihood, accuracy, benchmark,
Pareto, or calibration result. The only run-specific statement identifies the
completed run's training objective and diagnostic status, both fixed by the
authoritative handoff. Empirical claims remain visibly deferred until
checkpoint selection and the relevant frozen experiments pass.
