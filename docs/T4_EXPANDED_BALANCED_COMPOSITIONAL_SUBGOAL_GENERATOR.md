# Expanded balanced compositional structural-subgoal generator

This zero-oracle revision measures one change only: it adds the separately
sealed T4 delta-0.6 route corpus to the frozen compositional generator's
training distribution.

The baseline model, generic Active8 grammar, exact executor, structural
extractor and realizer, whole-source folds, candidate generation limits,
K=8/32/128 evaluation, metrics and acceptance gate remain unchanged. The
baseline source identity is commit
`184e4b4203853c28f9b2cc4acdd67227cf30746b`.

## Training evidence

The baseline distribution contains 77 complete T4 routes, 55 exactly converted
Full-146 auxiliary routes and 85 exactly converted PMO traces. This revision
adds 32 exact delta-0.6 COMPOSE witnesses containing 45 structural subgoals.
Those witnesses were compiled from reported IVG endpoints. They are inferred
COMPOSE realizations, not observed IVG trajectories.

The delta-0.6 corpus preserves 39 unique endpoint references and seven explicit
abstentions. One canonical source-endpoint pair contributes one route. Repeated
tied endpoint references remain provenance only and contribute no repeated
training mass.

For every fold, Full-146 and delta-0.6 routes are admitted only when their T4
source is in the training split. Every learned statistic is fit afterward.
Event weights assign equal mass in this order:

1. training domain;
2. source lineage, or PMO task-family lineage;
3. route;
4. dependency component;
5. decision within the component.

The fit report audits the mass at every level.

## Reproduction

The contract is
`configs/t4_expanded_balanced_compositional_structural_subgoal_generator_v1.json`.
Its payload is self-hashed and binds the baseline implementation, both existing
auxiliary corpora, and the new delta-0.6 corpus and result.

Fit from a clean committed revision:

```bash
PYTHONPATH=src .venv/bin/python \
  tools/t4_compositional_structural_subgoal_generator.py fit-all \
  --contract configs/t4_expanded_balanced_compositional_structural_subgoal_generator_v1.json \
  --output diagnostics/t4_expanded_balanced_compositional_subgoal_generator/attempt_1 \
  --pmo-corpus diagnostics/pmo_dependency_region_program_v2/attempt_1/training_dependency_region_corpus.json.gz \
  --pmo-result diagnostics/pmo_dependency_region_program_v2/attempt_1/result.json \
  --delta06-corpus diagnostics/t4_delta06_route_corpus/attempt_1/training_corpus.json.gz \
  --delta06-result diagnostics/t4_delta06_route_corpus/attempt_1/result.json
```

Then run `generate-all` and `evaluate-all` with the same `--contract`, output
root and immutable candidate-lock paths. The candidate lock is created before
the evaluator loads held teachers.

This revision makes no docking or task-oracle call and launches no Modal job.
It tests only whether the additional sealed training evidence improves
zero-oracle held-source structural proposal quality. It cannot establish
molecular utility or an IVG comparison.
