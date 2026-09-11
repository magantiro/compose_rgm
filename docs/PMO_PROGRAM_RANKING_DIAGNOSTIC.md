# Reuse complete-program labels before another controller launch

Problem: the uniform donor/reference mixture improved PMO twice, but further
archive reuse did not improve best. Test whether existing scored complete edits
contain enough transferable preference signal to choose better programs.
Output is a preference over parent/product pairs, not a predicted calibrated
oracle score, future value, primitive transition law, or new molecular support.

Before fitting, assign the isolated donor probe and first matched comparison to
training. Assign the later replication and archive comparison to validation.
Deduplicate parent/product pairs, retain their origins, and remove every
validation edge whose parent or product occurred as either endpoint in training.
Freeze this reconciliation before extracting model features. This is an exposed
chronological development check, not a newly sealed test or scaffold holdout.
Do not change this split or model recipe after viewing its predictions.

Reuse `BranchPolicy` unchanged: same-parent score-gap-weighted logistic loss,
equal mass per informative parent, fixed graph/edit kernel and regularization.
Include incumbents as comparison anchors only. Features are deterministic;
coefficients and every learned quantity use training rows only. Audit positive,
negative and tied transitions by option before fitting. No hyperparameter sweep.

Compare model-ranked and uniform choices over the same saved completed
candidates within each new parent group. Report candidate/group coverage,
pairwise preference precision, precision and recall for positive parent-to-child
changes, mean selected gain, and best selected score. Report donor-only and all
options separately; groups with fewer than two distinct endpoints do not test
choice. A score decrease is not fabricated into a continuation failure.

A model that only identifies damage but cannot rank constructive donor changes
does not justify guidance deployment. A positive descriptive ranking result on
multiple new parents earns at most a small fresh proposal-ranking comparison;
uncertain coverage requires a fresh locked candidate pool, not a generalization
claim. A null stops this unchanged model. No future-aware interpretation follows.

Cost: local CPU fitting and feature evaluation only, zero new oracle calls, no
reference-model enumeration, no GPU, no Modal workers. Preserve exact source
result hashes, split/exclusion identities, recipe/model hashes, software,
timings, and all predictions. Broad chemistry and the frozen executor remain
unchanged. No public winner molecules or endpoints are injected.
