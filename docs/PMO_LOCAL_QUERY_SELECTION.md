# Local query selection inside broad COMPOSE search

## Question and scope

The frozen reference's cached neighborhood contains improvements, but scoring
1024 successors yielded only 0.0027263 above the current controller champion.
Test whether an endpoint regressor trained on already-paid local labels allocates
small query batches better than uniform allocation. This is a component of the
broad local/global controller, not a replacement for multi-edit programs, a
future-value model, a Doob sampler, or a new executable chemistry contract.

The executor, reference checkpoint, exact source, all 1580 supported products,
task and prior generation lock remain fixed. No public winner is a model input.
Historical task labels retain their original accounting and provenance.

## Frozen recipe before fitting

Use the existing state kernel from `experiments/pmo_chronological.py`: half the
existing Morgan/atom-pair kernel and half the fixed molecular-count RBF. Fit
centered kernel ridge regression to observed endpoint scores with the existing
ridge constant 1.0. No hyperparameter selection or uncertainty-based acquisition.
Persist coefficients and exact training identities for replay.

Split the 1044 scored canonical neighbors by deterministic SHA-256 of
`local-query-v1/<canonical-smiles>`: first byte modulo five equals zero goes to
calibration, the remainder to training. Freeze assignments before features or
fitting. This is exposed within-parent interpolation, not scaffold- or
source-disjoint validation. It cannot authorize general task guidance.

Rank calibration products by prediction, take eight, and compare their mean
actual score against the uniform expected mean of the same calibration pool.
Also report selected parent improvers, precision, recall, best available,
selected best and absolute prediction error. Positive requires higher selected
mean and at least one actual improvement over the originating 0.649519 parent.
Negative: do not query this unchanged model. A passing component earns only the
following bounded prospective query comparison.

## Prospective component comparison

After calibration passes, refit on all paid neighbor labels. From the previously
unscored canonical neighbors, lock 16 top-prediction candidates and 16 candidates
sampled uniformly without replacement using seed 20261005. Arms may overlap;
charge the canonical union once, at most 32 new calls. Freeze both lists before
any new query. Do not update the model within the batch.

Primary decision: prediction arm exceeds uniform on both mean score and number
of source-parent improvements. Otherwise stop this unchanged allocation recipe.
Report best scores versus the observed 0.6747477698 champion separately; beating
uniform is not champion or PMO benchmark success. A positive permits testing the
unchanged allocation on new parent neighborhoods within broad search, not a
full campaign or generalization claim.

## Compute and provenance

One local CPU, float64, one numeric thread, 120-second bound, no GPU, no new
reference-law enumeration, no Modal workers, no docking. Reuse the complete
generation cache and exact primitive witnesses. Persist input hashes, split,
configuration, source snapshot, model, candidate lock, oracle receipts and
endpoint replay audit. Preserve all unqueried successors, failures and costs.
The final state kernel for at most 1580 molecules uses less than 32 MiB of
float64 entries, plus temporary arrays. Actual timing/memory is reported.
