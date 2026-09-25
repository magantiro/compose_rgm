# Linker quality headroom on locked development panels

## Diagnostic question

Before changing linker proposals or selection, measure whether the saved
eight-offer development panels already contain model-supported, prompt-faithful
endpoints that pass the benchmark's public quality predicate when the deployed
novelty-4 selector returns an endpoint that fails. This is a read-only,
post-hoc diagnostic on seed 5, not a new benchmark or a method-selection gate.

## Frozen analysis

Use the 1,000 attempt records and ten prompt identities bound by the locked
novelty-4 development summary. Verify the summary, manifest, every attempt,
and the pinned evaluator module by physical SHA-256. For each distinct offered
canonical endpoint with `model_supported` status, compute the same upstream
QED and SA values as the published evaluator. The binary property predicate is
QED at least 0.6 and SA at most 4.0. Do not draw another candidate or replace
an output. Count, for each prompt, the attempts with any quality-qualified
offer, attempts with a quality-qualified offer that had not previously been
returned, and attempts where the selected endpoint fails despite either kind
of available offer. Track repeated selections separately because headline
quality counts unique qualifying outputs. Also record the selected endpoints'
QED-failure, SA-failure, and joint-failure counts.

Do not use this diagnostic as evidence that a quality-aware selector achieves
the offer-availability upper bound. A quality-aware policy would be a distinct
controller that consults the public benchmark properties during selection.
Its computational cost, scientific interpretation, and fresh-run performance
would require a separate predeclared evaluation. The current three-seed
novelty-4 evaluation remains unchanged.
