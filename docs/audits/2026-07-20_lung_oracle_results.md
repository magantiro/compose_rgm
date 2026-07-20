# Lung-oracle results: verified LuT, LUMI, and LNPDB baselines

**Date:** 2026-07-20  
**Status:** reproducible first results, not a frozen prospective oracle

## Result in one sentence

Simple models already recover meaningful pulmonary structure--activity signal,
but the LuT round-transfer failure shows why COMPOSE-Lipid must use a
context-aware pan-lung multitask oracle rather than treating one combinatorial
library as the deployment distribution.

## LuT: direct systemic in-vivo lung task

The official source workbooks from the 2026 LuT study were downloaded, hashed,
rendered, and reconstructed into exactly 444 non-control compounds: 318 from
the first library and 126 from the second. Each row has paired endpoints:

- total lung luciferase expression after IV delivery of 0.1 mg kg-1 luciferase
  mRNA at 6 h, represented as log10 photons per second;
- lung selectivity as a fraction of the measured liver, spleen, and lung signal.

The source manifest is
[`configs/lut_444_systemic_lung_v1.json`](../../configs/lut_444_systemic_lung_v1.json).
Extraction and modeling are reproducible with
[`scripts/extract_lut_source_table.py`](../../scripts/extract_lut_source_table.py)
and
[`scripts/train_lut_oracle_baselines.py`](../../scripts/train_lut_oracle_baselines.py).

### Best first-pass results

| Endpoint | Split | Spearman | MAE | R2 | Top-decile enrichment |
|---|---|---:|---:|---:|---:|
| Lung expression | Random diagnostic | 0.577 | 0.478 log10 | 0.364 | 5.04x |
| Lung expression | Held head | 0.526 | 0.520 log10 | 0.244 | 3.07x |
| Lung expression | Held tail | 0.545 | 0.491 log10 | 0.348 | 5.26x |
| Lung expression | Round 1 to round 2 | 0.318 | 0.759 log10 | -0.232 | 3.73x |
| Lung selectivity | Random diagnostic | 0.815 | 0.183 fraction | 0.649 | 4.17x |
| Lung selectivity | Held head | 0.758 | 0.185 fraction | 0.585 | 2.41x |
| Lung selectivity | Held tail | 0.754 | 0.197 fraction | 0.577 | 3.95x |
| Lung selectivity | Round 1 to round 2 | 0.313 | 0.328 fraction | -0.130 | 0.75x |

The complete model-by-representation matrix is stored in
[`diagnostics/lut_444_in_vivo_baselines.json`](../../diagnostics/lut_444_in_vivo_baselines.json).

### Interpretation

- The held-component rankings are genuinely promising: the signal is not only
  a random-split artifact.
- Selectivity is substantially easier than absolute expression in this
  factorial library.
- Round-transfer ranking remains weak and absolute calibration fails. This is
  the critical negative result: even a rational second library from the same
  paper is sufficiently shifted that the first library alone is not a safe
  oracle for a new linker chemistry.
- These baselines use published structural/SAR annotations and component IDs,
  not full molecular graphs. They are a dataset qualification result, not the
  final representation benchmark.

## LUMI: airway-cell in-vitro task

The official 1,920-row LUMI 4CR table contains full structures, component codes,
and log2 RLU in human bronchial epithelial cells. The strongest ExtraTrees
baseline achieved random-fold Spearman 0.847 and held-component Spearman of
0.479, 0.737, 0.821, and 0.746 across R1--R4. Full results are in
[`diagnostics/lumi_lab_4cr1920_baselines.json`](../../diagnostics/lumi_lab_4cr1920_baselines.json).

LUMI is high-value representation learning and an airway-cell oracle head. It
is not relabelled as systemic in-vivo lung potency.

## LNPDB: every verified lung task, kept semantically separate

The frozen local LNPDB artifact contains 1,975 lung-associated observations
covering 291 unique ionizable lipids. It resolves into four non-interchangeable
tasks: 1,801 A549 in-vitro expression rows, 29 HBEC-ALI in-vitro expression
rows, 49 individual intratracheal functional-expression rows, and 96 systemic
IV barcoded-uptake rows. The leakage-resistant baseline uses held-lipid folds;
the complete representation-by-model results are stored in
[`diagnostics/lnpdb_lung_task_baselines.json`](../../diagnostics/lnpdb_lung_task_baselines.json)
and reproduced by
[`scripts/train_lnpdb_lung_oracle_baselines.py`](../../scripts/train_lnpdb_lung_oracle_baselines.py).

| Task | Best held-lipid cell | Spearman | MAE | R2 | Top-decile enrichment |
|---|---|---:|---:|---:|---:|
| A549 in-vitro expression | Combined / ExtraTrees | 0.568 | 0.638 | 0.368 | 2.36x |
| HBEC-ALI in-vitro expression | Descriptors / Ridge | 0.249 | 0.831 | -0.208 | 0.00x |
| Intratracheal functional expression | Morgan / Ridge | -0.221 | 0.957 | -0.517 | 0.00x |
| Systemic IV barcoded uptake | Combined / ExtraTrees | 0.044 | 0.852 | -0.137 | 0.00x |

This is an important negative result, not a reason to pool labels. A molecular
baseline works on the large A549 head, whereas the small in-vivo/local heads
do not support reliable isolated predictors. They therefore require shared
representation learning, explicit formulation and experimental context, and
uncertainty-aware multitask transfer. Barcode uptake remains an auxiliary head
and is never treated as functional expression.

## Locked modeling consequence

The lung oracle is a shared encoder with typed heads, not one pooled scalar:

1. systemic-IV lung expression;
2. systemic-IV lung selectivity and liver/spleen penalties;
3. local pulmonary delivery;
4. airway-cell in-vitro transfection;
5. cell-type and editing heads where available.

The final matrix axes are strictly **representation by model family**, not
dataset by model. Every executable cell is trained and scored over the entire
curated corpus with task/study/route/readout metadata and masked task-specific
heads. Representations are fingerprints/descriptors, component/region
features, molecular graph encoders, and full-formulation encoders; model
families are Ridge/Elastic Net, gradient-boosted trees, ExtraTrees, shallow
MLPs, graph models, and a shared formulation model. Dataset/source-specific
results above qualify endpoints and expose transfer failures; they are not
columns of the final model-selection matrix. Leave-study, chemical-family,
linker, and temporal splits are primary. Random folds remain diagnostics.

For the prospective Michael-addition linker, model selection requires a
novel-linker holdout and applicability-domain gate. If confidence is
insufficient, a small active-learning calibration set precedes expensive
in-vivo selection.
