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
PYTHONPATH=.:src .venv/bin/python \
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

## Measured result

The frozen expanded fit emitted all 128 requested candidates for both policies
on all 15 held sources. This is 3,840 candidates in total, with zero candidate
shortfall. Every accepted candidate exactly realized and replayed: 1,920/1,920
for the uniform policy and 1,920/1,920 for the learned policy. The uniform arm
filled its pools in 1,922 compiler attempts. The learned arm required 2,041
attempts, with 73 `TypeError` and 48 `ValueError` compile abstentions recorded
before replacement candidates filled the frozen pools.

The table reports source-balanced granular component coverage. The baseline is
the separately frozen fit without the delta-0.6 routes. The grammar, compiler,
folds, seeds and candidate budgets are identical.

| K | Uniform | Baseline learned | Expanded learned | Expanded minus baseline |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 24.08% | 18.01% | 18.26% | +0.25 points |
| 32 | 37.86% | 34.63% | 35.13% | +0.50 points |
| 128 | 46.25% | 59.77% | 58.24% | -1.53 points |

The expanded learned arm passes the unchanged within-revision gate at K=128:
it exceeds uniform in every fold, 63.94% versus 46.28% in fold 0, 51.96%
versus 43.02% in fold 1, and 58.81% versus 49.44% in fold 2, while exact
realization precision remains 100%. It remains worse than uniform at K=8 in
all three folds and at K=32 in two of three folds.

The added delta-0.6 supervision does not establish a consistent improvement
over the baseline learned fit. It gives small aggregate gains at K=8 and K=32,
but lowers the primary broad-support K=128 measurement. At K=128, learned novel
whole-patch yield also changes from 92.67 to 90.40 per source, and unique patch
yield changes from 94.67 to 92.27 per source. This is a mixed, not a positive,
data-expansion result.

Both learned revisions recover 2/147 exact held patches at K=32 and K=128.
Neither the expanded learned arm nor its uniform control recovers any of the 77
complete teacher endpoints or any radius-2 teacher transformation class at
K=8, K=32 or K=128. These are material negative findings. They do not establish
poor docking utility because this milestone observes no task score.

The immutable candidate lock has physical SHA-256
`36af44427cab2fdbb76aa5b330707f048abef8cab74a4af62405a7571b1f69b6`
and payload SHA-256
`43b7470b73867d2a9805455465c3f594cd9c5edea6e6ed174b622a1b4cf328b6`.
The result has physical SHA-256
`9c305507d53db6b42933fe4b29fc3c791329cebe0280a79971a26ac2193635c5`
and payload SHA-256
`ef36d32fc691a2c9b98107e9e33917025c16b026695a48d58b1dc26f98b09f8d`.
The independent candidate-lock rebuild took 3,102.46 summed pool seconds on
one CPU worker and was byte-identical to the first lock. This is 20.22 seconds
slower than the baseline revision's independent 3,082.24-second build, a 0.66%
difference under the same candidate work.

## Decision boundary

The expanded model remains a valid candidate for a later score-blind utility
comparison because it generates broad, novel and exactly executable support.
The offline evidence does not select it over the baseline learned model. A
later utility lock must preserve both learned revisions and their matched
uniform controls. No model, compiler, support, split, budget or acceptance rule
is changed in response to this result.
