# T4 proposal-prior corpus, split-clean preparation

Decision: **GO**. Contract `d8e5feb4a15bf0aa`.
Zero oracle calls, zero new labels, no model fit.

## Admission

- input rows: 35895
- admitted rows: 35895
- labelled constructions after collapsing repeat receipts: 34073
- excluded rows: 0 {}

The recovered pack was already filtered by the audit that produced it, so no
admission rule rejected anything here. The rules are therefore exercised only by
their tests, not by this data; that is a property of this input, not evidence that
the rules are unnecessary.

## Group disjointness

| Identity | Distinct values | Values crossing a cell |
| --- | ---: | ---: |
| endpoint | 30454 | 0 |
| entry_id | 34024 | 0 |
| input_state_sha256 | 4732 | 0 |
| protocol | 25 | 0 |

## Per-fold census

| Fold | Train records | Train cells | Held records | Held sources | Held lineages |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5ht1b | 27831 | 12 | 6242 | 1087 | 33 |
| braf | 28436 | 12 | 5637 | 837 | 17 |
| fa7 | 30129 | 12 | 3944 | 479 | 16 |
| jak2 | 24372 | 12 | 9701 | 1381 | 23 |
| parp1 | 25524 | 12 | 8549 | 948 | 44 |

## Construction-family support

Tier is the minimum training-side count across all five folds.

| Family | Minimum training records | Tier |
| --- | ---: | --- |
| dependency_branch | 6678 | dense |
| compiled_complete_transformation | 4622 | dense |
| bond_reroute | 3537 | dense |
| atom_restate_semantic | 3020 | dense |
| functionalize | 1576 | dense |
| substituent_delete | 1163 | dense |
| carbonyl_insert | 1023 | dense |
| segment_replace | 618 | dense |
| cycle_close | 565 | dense |
| cycle_open | 434 | dense |
| segment_grow | 421 | dense |
| append_ring | 383 | dense |
| heteroatom_substitute | 313 | dense |
| segment_shrink | 279 | dense |
| ring_system_restate | 192 | dense |
| fuse_ring | 137 | dense |
| construct_substituted_ring | 132 | dense |
| core | 102 | dense |
| ring_path_remodel | 39 | sparse |
| core_carbonyl_insertion | 36 | sparse |
| ring_carbonyl | 26 | sparse |
| pendant_benzene | 0 | absent |
| remodel_linker | 0 | absent |

## Statistical power of a held-out fold

Adaptive search descends from one bootstrap root, so a cell's scored
constructions collapse into very few genealogies. Record counts overstate power.

| Fold | Held records | Held lineages | Held sources | Weighted effective n |
| --- | ---: | ---: | ---: | ---: |
| 5ht1b | 6242 | 33 | 1087 | 676.3 |
| braf | 5637 | 17 | 837 | 299.1 |
| fa7 | 3944 | 16 | 479 | 151.1 |
| jak2 | 9701 | 23 | 1381 | 242.5 |
| parp1 | 8549 | 44 | 948 | 64.6 |

All 34073 records fall in 133 lineage components
(median 48, maximum 2116, 7 singletons).

Under the frozen target/cell/lineage/record hierarchy the heaviest single record
carries 0.0133 of corpus mass (454x uniform), the heaviest 100 records carry
0.233, and the whole corpus has effective
n = 805 against 34073 records.
Equalizing across lineage components over-corrects here, because a cell holds one
very large genealogy beside a few singletons. This is a measured property of the
corpus, reported as a finding; the hierarchy was frozen before it was computed and
was not changed afterwards. A capped or size-damped variant is a decision for the
separate fitting contract, taken with this number in hand.

## Gate

Every frozen hard criterion and required family held.

## Limits

- These are adaptive historical search observations, not independent samples.
- Held-target folds measure generic transfer of this corpus. They are not a
  held-out benchmark result, and a checkpoint fit on all cells stays trained-on-T4.
- Program payloads are bound by checkpoint hash and entry id, not copied. A later
  fitting step must resolve them inside its own training fold.
- Repeat dockings of one construction were collapsed; their range is preserved per
  record because REPORT section 7 measured non-trivial docking repeat variation.
