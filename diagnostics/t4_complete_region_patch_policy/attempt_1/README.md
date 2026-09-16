# Complete-region patch policy Gate 3

## Outcome

Gate 3 failed its frozen promotion rule. The learned conditional complete-patch
policy did not beat the same-grammar, source-balanced marginal control at
conditional teacher-patch Top-8 on any of the three held-source folds. Gate 4
was therefore not run or promoted.

The negative result is above the representation and compiler layers. All 147
held regions remained inside the complete-patch grammar and round-tripped
exactly (coverage 147/147; precision 147/147). No oracle, docking, Modal,
network, or GPU cost was incurred.

## Aggregate held-source evidence

Both arms used the same 128-candidate diagnostic fiber. Target-patch ranks are
conditional on the teacher WHERE and include a labelled teacher injection;
they are not autonomous-recovery measurements.

| Metric | Learned | Marginal |
|---|---:|---:|
| Conditional patch mean rank | 14.388 | 7.395 |
| Conditional patch median rank | 8 | 7 |
| Conditional patch Top-8 | 74/147 | 99/147 |
| Conditional patch Top-32 | 128/147 | 147/147 |
| Mean complete-patch NLL | 139.284 | 137.730 |
| Mean token NLL | 1.2351 | 1.2206 |
| WHERE mean rank | 49.898 | 50.667 |
| WHERE Top-8 | 8/147 | 14/147 |
| WHERE Top-32 | 49/147 | 47/147 |

The autonomous 127-candidate WHERE fiber contained the exact address-free
teacher WHERE for 0/147 held regions. The teacher was added only for the rank
diagnostic. This is support absence in the sampled WHERE fiber, distinct from
the conditional target-patch misranking measured after the correct WHERE was
provided.

Every learned factor had higher held-source NLL than the matched marginal:

| Conditional factor | Learned NLL | Marginal NLL |
|---|---:|---:|
| Atom attributes | 1.5960 | 1.5826 |
| Attachments | 0.6365 | 0.6212 |
| Bond attributes | 1.3194 | 1.3032 |
| Control | 1.7227 | 1.7211 |
| Dependencies | 2.9086 | 2.9085 |
| Target topology | 0.6350 | 0.6227 |

No full joint WHERE-plus-patch rank is reported because the frozen diagnostic
enumerated separate WHERE and teacher-conditioned patch fibers. Fabricating a
joint rank from those fibers would be invalid. Cross-region created-role policy
learning also abstains because this 147-region corpus contains no positive
cross-region created-role reference.

## Interpretation and next bounded change

The complete-region grammar, address-free target-patch representation, and
exact runtime remain supported. The sparse source-summary/prefix count
conditioner fails to transfer across held sources: small per-token NLL losses
compound into materially worse complete-patch ordering. The WHERE density also
does not concentrate early probability on the exact held region.

The smallest evidence-based next model revision is to keep the grammar,
runtime, realizer, folds, and candidate budgets fixed, but replace sparse exact
conditional-count lookup with a shared, lower-variance feature-conditioned
model. Its conditional contribution should be shrunk toward the balanced
marginal using training-only leave-one-source-out calibration. WHERE should be
trained/ranked over the actual connected-region candidate fiber with the same
shared source features, rather than relying on random region sampling plus a
single diagonal density. This is a proposed subsequent contract, not a result.

## Provenance

- Implementation commits: `8342d01e` (fit/evaluation), `dee223bb` (reducer and
  blocked-gate recorder).
- Gate 3 aggregate physical SHA-256:
  `8dd7382e01c3cea9b194e11a6781128d69cce0acb7813327f7080b87ff29f918`.
- Gate 3 aggregate payload SHA-256:
  `88bb3b72fece75b6bbfe4f4464f776d4a40a8439b736d2d988435676a2aa24a5`.
- Gate 4 blocked physical SHA-256:
  `a6ba2f48bd355876e55b3a3b701889fdcb341361444a2b83c41235c4c97ab665`.
- Gate 4 blocked payload SHA-256:
  `6f6b436367c1324a53d02f149e0bb72b8dc04814db856ef2196f69e3c136735c`.
- Three folds ran independently with one CPU worker each. Observed process wall
  times were 884.23 s, 920.26 s, and 633.49 s. The deterministic reduction used
  fixed fold order.

Authoritative machine-readable artifacts are `gate3_aggregate.json`, the three
`gate3/fold_*/gate3.json` files and checkpoints, and `gate4_blocked.json` in this
directory.
