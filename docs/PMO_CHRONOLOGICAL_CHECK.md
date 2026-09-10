# Offline chronological prediction check

## Authorized scope and frozen recipe

2026-09-10: user approved the zero-new-oracle prediction check. This authorizes
small local CPU regressions on saved development labels only. It does not change
the original pilot contract's prohibition on model training during that pilot,
authorize reference-model training, deploy guidance, or launch new optimization.

Problem: determine whether scored molecular histories contain usable contextual
information beyond parent scores and pooled option preferences. Outputs are
predictions for already locked completed molecules and measured score changes.
The claim tested is chronological predictive association, not counterfactual
policy improvement, calibrated uncertainty, or full future-value control.

Inputs: the two complete 100-query perindopril development receipts, plus the
eight original root-score receipts matched by their recorded oracle-ledger SHA.
No winner structures, target molecule, task-specific scoring code, or new labels.
Both streams retain their original order and are evaluated independently. These
are inspected development runs, not an untouched test set. Existing molecular
support remains 1..40 heavy atoms with frozen charge/stereochemistry limits.

Freeze before fitting:

- First 20 charged unique queries are warmup, including all four roots. Fit at
  query counts 20,30,...,90; predict the following ten new molecules with that
  prefix model. No cross-arm labels or future observations enter training.
- A candidate's parent score is usable if observed before that candidate's query,
  even when it arrived after the current model snapshot. Every comparison uses
  this same available information. Duplicate queries are not repeated examples.
- Baselines: copy the parent score; parent score plus option-specific mean signed
  change with two observations of shrinkage toward the training-prefix mean.
- Context model: kernel ridge for signed score change, conditioned on parent
  structure, exact option label and intended scale bin. This uses information
  available before executing the option, but not the exact region/site.
- Endpoint model: kernel ridge for absolute score of the completed candidate.
- Edit model: kernel ridge for signed change, using parent context, endpoint
  structure and a normalized signed molecular-feature-change kernel. These last
  two models require a completed proposed molecule. They cannot by themselves
  guide unseen intermediate continuations or estimate pre-execution option value.
- All ridge penalties are 1.0, with targets centered by the training-prefix mean.
  Predicted endpoints are clipped to the known score range [0,1]. No model or
  hyperparameter sweep; no fitted uncertainty or pretrained task-value weights.
- State kernel: equal mixture of the existing Morgan/atom-pair fingerprint kernel
  and an RBF over fixed molecule counts (heavy atoms, cycles, aromatic rings,
  H-bond donors/acceptors). Count scales are [5,1,1,1,2], defined as structural
  units, not learned from this panel. No fitted feature preprocessing.
- Context kernel is the state kernel multiplied by (1+option equality)/2 and
  (1+scale-bin equality)/2. Edit kernel equally averages context, endpoint and
  normalized difference-feature kernels. All are positive-semidefinite kernels.

Report MAE and within-ten-query-window ranking concordance (ties count half),
plus sign-of-change accuracy and cycle-rank-increasing versus other edits.
Whole-window ranking is descriptive, not same-parent counterfactual ranking.
An offline candidate must beat both baselines on MAE and ranking in each arm;
ranking must also exceed 0.5. Report every model and all exclusions regardless of
this result. A positive result only nominates a bounded prospective comparison.
A negative result must not trigger more oracle spending or a relaxed threshold.

Acceptance: label/parent chronology and exact root identities validate, focused
future-label-invariance checks pass, all declared predictions are persisted with
input/model/implementation hashes and software settings, and final artifacts are
inspected. Only this offline assessment is authorized; no deployment milestone.
