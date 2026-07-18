# FCD transfer from `rank_d500k`

The empirical reference is the sibling `compose` recipe `rank_d500k`: reported
FCD 9.96 from 1,500 fresh generated molecules against 5,000 reference molecules.
It trained on about 466,000 usable GuacaMol examples for 30,000 steps. Its
generated ChemNet covariance trace was 25.7 versus 42.3 for the corpus (61%
coverage), so the result is a useful milestone rather than a solved diversity
problem. The recorded 5,000-reference slice is a training-set prefix; the
recipe names a separate held-out file, but 9.96 itself is not labeled as the
held-out result.

The predecessor is a mask/discrete-flow implementation. `compose_v4` does not
copy that process. It transfers only mechanisms whose rationale survives the
change to Generator Matching over executable valid rewrites.

| Measured predecessor lesson | `compose_v4` translation |
| --- | --- |
| One frozen 50k subset did not expose enough rare ring-system variety; ranking loss and 500k data were both needed for the 9.96 result. | Scale the corpus and report rule-family coverage. Family-stratified progress sampling remains exactly importance-correct. |
| A per-pair focal detector under-fired closures. Listwise ranking plus calibrated firing mass fixed recall. | Do not copy the focal/ranking loss. Poisson-KL Generator Matching already charges total hazard and rewards the complete teacher-successor rate with a proper rate objective. `quotient_energy` normalizes over distinct chemical successors, not aliases. |
| The model saw Kekulé open-ring paths in training but order-1 paths at decode; fixing this covariate shift improved FCD by about 3. | Use the aromatic bond view for model inputs while retaining executable Kekulé actions. Audit train/sampler state support before adding any decode patch. |
| Ring kind inferred from a partially rewritten graph silently mislabeled fused aromatics; clean-molecule ring kind plus live geometry fixed it. | Typed ring tracelets carry the complete atom/bond payload, while the current valid graph supplies junction context. Aromaticity is committed by coordinated ring rules, not isolated aromatic-bond edits. |
| Ring count alone did not fix FCD; learned placement and ring-system variety mattered. | Evaluate fused/spiro/bridged topology, scaffold diversity, and ChemNet covariance—not just aromatic-ring count. |
| The old source coupling had to match the sampler's prior. | Carbon-tree training and inference use the same `DegreeBoundedCarbonTreePrior`. Each endpoint can now be paired with multiple independent source-tree draws via `--tree-couplings-per-target`; the draw remains target-independent. |
| Exact recipes and identical sampling harnesses were necessary to avoid temperature/config confounds. | `recipes/tree_fcd_transfer_stage1.json` freezes the first quality-bearing tree experiment and `scripts/run_tracelet_recipe.py` materializes it through the normal trainer. |

## FCD protocol now recorded

`molecular_quality_report` records:

- generated and reference sample counts;
- total FCD;
- additive ChemNet mean and covariance components;
- generated/reference covariance traces and their ratio.

The first stage uses 2,500 generated molecules and an explicit 5,000-molecule
reference file. A final paper number should use at least 5,000 generated
molecules and multiple seeds.

## What is not yet comparable to full GuacaMol SOTA

The present chemistry gate is neutral connected C/N/O/F. A stage-1 FCD against
the full GuacaMol reference is intentionally harsh and is not directly
comparable to an unrestricted model until the element/charge vocabulary and
the 40-heavy-atom regime are supported. The stage-1 purpose is to test whether
structured carbon-tree noise improves the learned valid-rewrite generator and
whether the remaining FCD is mean-shift or under-dispersion.

Run the frozen stage with explicit datasets:

```bash
PYTHONPATH=src python scripts/run_tracelet_recipe.py \
  recipes/tree_fcd_transfer_stage1.json \
  /path/to/guacamol_v1_train.smiles \
  --quality-reference-file /path/to/heldout_5000.smiles
```
