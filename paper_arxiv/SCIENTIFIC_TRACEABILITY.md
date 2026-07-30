# Phase 1 scientific traceability

This note records the source and wording boundary for each load-bearing claim in
the Phase 1 manuscript. It is an audit aid, not submission prose.

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

## Empirical-status policy

Phase 1 contains no endpoint-quality, likelihood, accuracy, benchmark, Pareto,
or calibration result. The only run-specific statement identifies the
completed run's training objective and diagnostic status, both fixed by the
authoritative handoff. Empirical claims are deferred until checkpoint selection
and the relevant frozen experiments pass.
