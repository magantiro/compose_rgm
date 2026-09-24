# Frozen C/N/O/F prior-scale split

This is a data-preparation result, **not** a trained prior or a de novo quality
improvement. The clean preparation checkout was commit `adf6d594994c07babf81033ce454e95af2169990`.

- Input: `/Users/rmaganti/compose_denovo_artifacts/guacamol/guacamol_subset_500000_seed0.smiles`, SHA-256 `70526d92f1f08d8e292cb31218f81b6924a2182f772c43348015110669d47791` (user-controlled GuacaMol corpus; redistribution not assumed).
- Output: `frozen_split.json`, physical SHA-256 `c06bff5e50f27bebe7ff4cfed260f76c2e1730113b3b7c213e5b13bbd7193f6e`, payload SHA-256 `06563ba7eefbae2a9591588e0e574d0b63aefcef44061512747f723d0483110c`.
- Census: 500,000 source rows; 225,149 unique eligible neutral connected C/N/O/F molecules with at most 40 atoms; 15 canonical duplicates; 274,836 excluded with row-level reason codes retained in the manifest.
- Frozen partitions: ordered train reservoir 216,149, validation 2,000, IID test 2,000, scaffold-disjoint test 5,000. The 50,000-molecule arm is the exact prefix of the larger train reservoir. The same validation and test identities apply to both arms.
- Validation: `validate_frozen_cnof_split(..., deep=True)` passed, including payload and partition hashes, source-row reconciliation, canonical support and zero scaffold-test group leakage. IID scaffold overlap is expected and reported separately in the manifest.

No model was fitted and no benchmark score was selected with these data. The
existing 50k Lineage B checkpoint cannot be treated as a clean small arm of
this comparison: it was trained under a different split, so warm-starting from
it would leak held-out identities. Both matched arms must fit from scratch
under a separately sealed training contract.
