# Same-parent edit-choice diagnostic

2026-09-10: user authorized this saved-data follow-up to the failed chronological
prediction check. No new oracle calls, fitting, proposals, controller changes,
deployment, or scientific-support changes are authorized here.

Problem: distinguish ranking different parent molecules from choosing among
edits of one molecule. Output: matched-parent ranking and selection diagnostics
for the five already frozen predictors. This tests predictive association on
logged completed options, not future-value control or policy improvement.
COMPOSE remains a stochastic executable molecular editing platform, not this
particular diagnostic. Support remains the original bounded 1..40-heavy-atom
process, with its frozen charge and stereochemistry limits.

Inputs: the committed chronological report and both original 100-query PMO
development results. Verify their recorded hashes and join predictions to exact
archive parent IDs. No raw label is recomputed. These development outcomes have
already been inspected; this is not a newly sealed test set.

Fixed analysis, written before computing its results:

- Primary groups: identical archive parent ID and identical model fitting cutoff,
  within one arm. Each group needs at least two distinct queried endpoints.
  Separate groups whose parents were already observed at the fitting cutoff
  from those whose shared parent became available later. The original prediction
  contract permits both, but they are not interchangeable prospective settings.
- Report same-parent census across the full 80 predictions and exclusion counts
  for singleton parent/snapshot groups. Do not mix model snapshots to inflate
  primary coverage, or pool canonically equivalent exact parent variants.
- Compare every frozen model and both baselines. No model refit, new kernel,
  hyperparameter selection, target transform, or admission criterion.
- Pairwise concordance: higher endpoint score is preferred, true-score ties
  (absolute difference <=1e-12) omitted, prediction ties count half. Also report
  non-tied prediction coverage and conditional accuracy. Report micro averages
  and equal-weight parent/snapshot averages, since pairs are dependent.
- Selection: choose the highest predicted endpoint score. Average actual score
  over prediction ties, rather than picking a favorable tie using outcomes.
  Report selected score, regret to the best logged sibling, gain versus uniform
  choice within the same logged group, and chance of selecting a best sibling.
  Retain groups with equal true scores and report their count separately.
- Keep every pair/group, its source IDs, scores, predictions, option labels,
  scale bins, and cycle-increase flags. Report mixed cycle-increase/non-increase
  groups separately. Cycle rank is not SSSR ring count or a medchem judgment.
- Sequential logging may make later proposals depend on earlier outcomes, even
  when the predictor is frozen. These are not simultaneous randomized menus,
  counterfactual policy returns, or proof of improvement from deploying guidance.
- The previous `no_predictor_qualified` decision remains unchanged. Sparse or
  inconsistent evidence requires abstaining, not relaxing the previous rule.

Acceptance: source hashes and joins validate, focused grouping/tie/chronology
tests pass, all denominators and exclusions are persisted with implementation
identity and software provenance, and the result and interpretation are inspected.
This is an offline development diagnostic, not a deployment milestone; the
full-suite requirement at a completed scientific milestone remains unchanged.
