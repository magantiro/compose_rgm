# Raw-witness option imitation: do not promote

This local diagnostic used previously inspected IVG winners as supervised
development data. It made no new docking calls and did not run autonomous search.
`result.json` binds inputs, implementation hashes, source roles, configuration,
checkpoint, predictions and software. The producer revision is recorded along
with an explicit dirty-tree flag; this is development evidence, not a frozen
benchmark submission.

82 saved witnesses replayed in RDKit 2024.03.5, covering 1,576 primitive steps.
Nine source/target pairs without a supported saved witness were excluded with
their original reasons. Canonical decision deduplication left 1,271 examples
represented by 81 pair IDs; the `retained_pairs` field counts representatives
after deduplication, not replay failures. Training used 949 decisions from 11
source groups; 322 decisions from four source groups were excluded from fitting.
No shared intermediate molecular state required exclusion in this run.

| Source-balanced held-out diagnostic | Balanced reference | Demonstration marginal | State-conditioned actor |
| --- | ---: | ---: | ---: |
| Negative log likelihood, lower better | 3.668 | 1.863 | 2.737 |
| Mean demonstrated-option probability | 0.0351 | 0.1783 | 0.0887 |
| Top-label agreement | 0 | 0.530 | 0.330 |

The labels contain no recognized whole-ring construction segments. The two
compound segments are carbonyl insertions and both fall in the excluded source
groups. The training examples predominantly encode grow and shrink choices.
Consequently training longer on this representation would not supply the missing
ring-option demonstrations. The actor is not admitted for deployment.

The fixed diagnostic menu includes all unrefined pendant/fused five- and
six-member C/N/O descriptors with aromatic or nonaromatic electronics, plus
ordinary macro and carbonyl options. It is deliberately independent of winner
frequencies. It is not a product-applicability row. These probabilities are
therefore classification diagnostics, not full molecular route probabilities.

Preparation: 5.329 seconds; fitting: 0.537 seconds; total: 7.835 seconds.
Hardware: one CPU thread, float32. Focused checks: eight tests passed in
2.89 seconds across `test_option_demonstrations.py` and `test_option_policy.py`.
Ruff lint and format checks passed for the three new Python files. No full suite
or remote launch was performed.

Next: derive verified complete-option demonstrations from graph changes and
existing compiled routes. Keep proposal imitation separate from future-value
learning. Report whether better demonstrations change proposal mass and actual
continuations before claiming benchmark improvement.
