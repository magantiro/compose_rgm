# AGILE/Ugi structural-bias and layered-sampling audit

**Date:** 2026-07-20  
**Decision:** retain AGILE virtual-only structures as a capped auxiliary Ugi layer; never merge them with R0 at their raw unique-union frequency.

## Result

The raw full-structure union contains 15,433 observed/measured R0 structures and 11,076 AGILE virtual-only structures. Treating every unique structure equally would assign AGILE **41.78%** of training draws. That is not supported by its structural coverage.

The frozen equal-priority realism-versus-coverage objective selects **22.56% AGILE and 77.44% R0**. This is a 46.00% relative reduction from AGILE's raw-union frequency. Eighteen sensitivity runs spanning 64, 128, and 256 R0-anchored structural cells, half/one/two R0-equivalent epochs, and minimax/L2 objectives select **20.47%–22.98%** (median 21.73%), so the conclusion is stable rather than a grid accident.

Within AGILE, sampling is not uniform. Each occupied AGILE structural cell receives the probability mass of its corresponding R0 cell, then divides that mass uniformly among AGILE structures in the cell. This prevents a densely enumerated Ugi neighborhood from receiving more probability merely because more variants were enumerated.

## Why raw AGILE frequency is biased

### Molecular profile

| Feature | R0 observed | AGILE virtual-only |
|---|---:|---:|
| Unique structures | 15,433 | 11,076 |
| Heavy atoms, range | 14–282 | 30–62 |
| Heavy atoms, median | 53 | 45 |
| Element signatures | 7 | 1: C/N/O only |
| Neutral | 96.70% | 100% |
| Any stereogenic atom | 74.37% | 100% |
| Ring-bearing | 52.66% | 59.43% |
| Aromatic-ring-bearing | 12.33% | 4.50% |
| No detected carbon branching | 50.69% | 89.08% |
| C–C unsaturation present | 38.27% | 39.94% |
| Exactly two long aliphatic termini | 47.83% | 100% |
| Ester motif present without another audited cleavable motif | 67.38% | 90.09% |

The long-tail count is a graph proxy: a non-aromatic terminal carbon at least six bonds from the nearest heteroatom through an aliphatic-carbon path. Cleavable motifs report substructure presence, not measured biodegradation.

### Scaffold and lipid-core diversity

Conventional Murcko scaffolds are insufficient for lipids because every acyclic molecule maps to the same empty scaffold. They are reported, but the primary family analyses use a heteroatom-connector core and R0-anchored ECFP4 cells.

- Murcko keys: 416 in R0 versus 7 in AGILE. Their Simpson effective counts are misleadingly similar (4.12 versus 4.22) because the acyclic bin dominates both.
- Heteroatom-connector cores: 2,229 in R0 versus 269 in AGILE.
- Simpson effective heteroatom-core count: 216.71 in R0 versus 22.67 in AGILE.
- At 128 R0-anchored ECFP4 cells, R0 occupies all 128 while AGILE reaches only **11 (8.59%)**.
- Raw Simpson effective structural cells: 51.25 for R0 versus 4.30 for AGILE.
- Raw R0–AGILE cell-distribution Jensen–Shannon divergence: 0.709 bits.

Even after within-AGILE cell calibration, divergence remains high because weighting cannot create chemistry in the 117 cells AGILE never reaches.

### Exact identity and near-neighbor redundancy

- Exact canonical cross-layer duplicates: **0**, by construction—the auxiliary set is the canonical AGILE-minus-R0 set.
- Median nearest-R0 ECFP4 Tanimoto for AGILE: **0.941**.
- AGILE structures with nearest-R0 Tanimoto ≥0.90: **6,935 / 11,076 (62.61%)**.
- Nearest-R0 Tanimoto ≥0.95: **4,545 (41.03%)**.
- Nearest-R0 Tanimoto exactly 1.0: **4,146 (37.43%)**.
- Within AGILE, 10,828 structures (97.76%) have another distinct AGILE structure with Tanimoto 1.0.

Tanimoto 1.0 means fingerprint equivalence, not exact molecular identity. ECFP4 can assign identical bit vectors to distinct molecules, especially homologous long-chain variants. The exact canonical duplicate count remains zero.

## Weight-selection objective

The maximum candidate AGILE probability is its raw unique-union fraction, 0.4178; optimization can only reduce it.

For each candidate probability:

1. **Realism penalty:** mean Jensen–Shannon divergence between the resulting mixture and R0 across size, charge, elements, stereochemistry, rings/aromaticity, branching, unsaturation, tail proxies, cleavable motifs, Murcko keys, heteroatom-connector cores, and R0-anchored ECFP4 cells. The curve is normalized so zero AGILE is 0 and the raw-frequency cap is 1.
2. **Coverage utility:** expected novelty-weighted number of unique AGILE structures encountered during one R0-equivalent epoch. Novelty is `1 − nearest-R0 ECFP4 Tanimoto`; within-AGILE draw probabilities are structural-cell calibrated. Coverage is normalized to its value at the raw-frequency cap.
3. **Selection:** minimize the larger of realism regret and unachieved-coverage regret. Both objectives are normalized, so no fitted or hand-set coefficient trades one against the other.

The primary structural-cell count is 128 because it is the member of `{64, 128, 256}` closest to `sqrt(15,433)`. The selected point has normalized realism penalty 0.381 and normalized coverage utility 0.620; worst regret is 0.381.

## Frozen split and eligibility contract

All grouping fields are computed before sampling probabilities.

- R0 retains its canonical leakage group and source study/component/reaction-family groups; a heteroatom-connector group is added.
- AGILE is frozen under `ugi_3cr_fixed_core`, an AGILE study group, a connector-core proxy, and a tail-architecture proxy.
- AGILE's exact A/B/C component identities cannot be recovered from the released virtual-only SMILES file. Therefore these are explicitly **structural-pretraining-only proxies**, not claimed synthetic component identities.
- AGILE cannot enter the observed layer and cannot support prospective labels.
- Any reaction-family, linker/core, component-proxy, or study holdout is assigned from the frozen group hash before a row is selected.

## Consequence for COMPOSE-Lipid

AGILE can teach a dense and useful Ugi 3CR neighborhood, but it cannot supply broad coverage of lipid size, elements, charge, branching, tail count, degradable chemistry, or core architecture. It should therefore be one auxiliary stratum alongside future independently qualified reaction families. Scaling AGILE alone to more enumerated homologues would increase row count without solving the coverage deficit.

## Artifacts

- `artifacts/datasets/compose_lipid_pretraining_v1/agile_ugi_bias_audit.json`
- `artifacts/datasets/compose_lipid_pretraining_v1/agile_virtual_only_nn_to_r0.csv`
- `artifacts/datasets/compose_lipid_pretraining_v1/layer_aware_sampler_rows.csv`
- `artifacts/datasets/compose_lipid_pretraining_v1/layer_aware_training_manifest.json`
- `scripts/audit_agile_ugi_bias.py`
- `src/compose_v4/lipids/corpus_bias.py`
- `tests/test_lipid_corpus_bias.py`

