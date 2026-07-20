# Pan-lung corpus manifest and oracle-matrix status

**Frozen:** 2026-07-20  
**Scope:** verified public pulmonary LNP evidence for COMPOSE-Lipid steering

## Ready now

The current frozen training boundary contains **4,339 measurement rows from
three primary artifacts**:

| Source | Rows | Structures | Typed use |
|---|---:|---:|---|
| LNPDB lung slice | 1,975 | 291 canonical lipids | A549, HBEC-ALI, local intratracheal expression, IV barcode uptake |
| LUMI 4CR | 1,920 | 1,920 full SMILES | human bronchial epithelial in-vitro expression |
| LuT | 444 | 444 head-tail compound IDs | individual IV lung expression and selectivity |

The row total is deliberately **not** presented as 4,339 independent lipids.
It combines different readouts and experimental contexts. The authoritative
machine-readable source registry is
[`../../configs/pan_lung_corpus_manifest_v1.json`](../../configs/pan_lung_corpus_manifest_v1.json).

## Provenance and duplicate decisions

- LNPDB is the canonical copy for its incorporated LiON, RCB/NRA, and CAD
  measurements. LiON has 9,194 canonical lipid structures, but 8,518 (92.65%)
  overlap LNPDB; those are provenance joins, not additive training rows.
- The RCB/NRA 720 and CAD 180 library headlines do not become 900 new lung
  labels. Their individual trainable LNPDB rows are already represented in the
  corresponding A549/local-pulmonary and barcode-uptake heads.
- CAD uptake is biodistribution supervision, not functional mRNA expression.
- PPZ contributes at most formulation/cell-type negative supervision: the 65
  tested formulations contain only eight ionizable lipid structures and the
  study reported no detected lung delivery.
- Branched-tail, split-Ugi, and the 22-lipid STAAR extrahepatic study remain
  pending until their row-level supplementary tables are frozen, hashed, and
  canonicalized. Their headline library sizes are not admitted as lung labels.

## LuT blocker, encoded rather than hidden

The official LuT Fig. 2/3 workbooks provide 444 paired IV expression and
selectivity rows, but not full product SMILES. The varied LuT compound is also
a quaternary cationic targeting component co-formulated with fixed ionizable
lipid 4A3-SC7. Therefore:

1. LuT is usable immediately through head/tail and published SAR features.
2. It is excluded from fingerprint/graph cells until the building-block
   products are reconstructed and chemically checked.
3. It remains a separate formulation-role branch even after reconstruction;
   it is not mislabeled as a conventional ionizable-lipid screen.

This avoids both invented structures and a biologically incorrect merge.

## Representation by model-family matrix

The matrix is defined in
[`../../configs/pan_lung_oracle_matrix_v1.json`](../../configs/pan_lung_oracle_matrix_v1.json).
Its columns are model families and its rows are representations. Dataset names
are provenance/split groups, never matrix axes. Every cell is scored across all
compatible typed heads with masked losses or per-head estimators under a shared
feature and evaluation contract.

| Representation | Linear | ExtraTrees | Boosting | Multitask MLP | Graph multitask | Ensemble |
|---|---|---|---|---|---|---|
| Morgan + descriptors + context | **qualified complete** | **complete** | **qualified complete** | queued | n/a | gated |
| Region/component + context | **LuT selectivity complete** | **LuT complete** | **LuT selectivity complete** | queued | n/a | gated |
| Frozen molecular encoder + context | queued | queued | queued | queued | n/a | gated |
| End-to-end molecular graph + context | n/a | n/a | n/a | n/a | deferred | gated |
| Full formulation set encoder | n/a | n/a | n/a | n/a | deferred | gated |

The **first executable matrix cell is Morgan/descriptors/context ×
ExtraTrees** over every full-SMILES typed head. It is now complete, alongside
the explicitly structure-blocked LuT component/SAR × ExtraTrees cell. The
joint exporter and frozen outputs are:

- [`../../scripts/export_pan_lung_matrix_cell_predictions.py`](../../scripts/export_pan_lung_matrix_cell_predictions.py)
  → [`../../diagnostics/pan_lung_matrix_R1M2_R2M2_predictions.csv`](../../diagnostics/pan_lung_matrix_R1M2_R2M2_predictions.csv)
  and [`../../diagnostics/pan_lung_matrix_R1M2_R2M2_metrics.json`](../../diagnostics/pan_lung_matrix_R1M2_R2M2_metrics.json)

The row-level table contains **11,683 held-group predictions**, with no null
predictions and no duplicate `source_id`/`measurement_id`/`typed_head`/split
keys: 1,975 LNPDB rows, 7,680 LUMI rows (1,920 under each of four component
holdouts), and 2,028 LuT rows (held-head, held-tail, and round transfer for two
typed heads).

### Frozen held-group results

| Source / typed head | Split | Rows | Folds | Spearman | R² | MAE | Top-decile recall |
|---|---|---:|---:|---:|---:|---:|---:|
| LNPDB A549 expression | held lipid | 1,801 | 5 | 0.568 | 0.368 | 0.638 | 0.238 |
| LNPDB HBEC-ALI expression | held lipid | 29 | 5 | -0.213 | -0.222 | 0.873 | 0.000 |
| LNPDB intratracheal functional expression | held lipid | 49 | 5 | -0.242 | -0.129 | 0.775 | 0.000 |
| LNPDB systemic-IV barcode uptake | held lipid | 96 | 5 | 0.044 | -0.137 | 0.852 | 0.000 |
| LUMI HBE expression | held R1 | 1,920 | 5 | 0.476 | 0.229 | 2.820 | 0.281 |
| LUMI HBE expression | held R2 | 1,920 | 4 | 0.735 | 0.534 | 2.138 | 0.500 |
| LUMI HBE expression | held R3 | 1,920 | 4 | 0.821 | 0.674 | 1.753 | 0.578 |
| LUMI HBE expression | held R4 | 1,920 | 5 | 0.751 | 0.510 | 2.138 | 0.359 |
| LuT IV lung expression | held head | 444 | 5 | 0.491 | 0.205 | 0.529 | 0.156 |
| LuT IV lung expression | held tail | 444 | 5 | 0.467 | 0.269 | 0.535 | 0.511 |
| LuT IV lung expression | round 1 → 2 | 126 | 1 | 0.297 | 0.045 | 0.659 | 0.385 |
| LuT IV lung selectivity | held head | 444 | 5 | 0.737 | 0.569 | 0.195 | 0.178 |
| LuT IV lung selectivity | held tail | 444 | 5 | 0.722 | 0.435 | 0.269 | 0.444 |
| LuT IV lung selectivity | round 1 → 2 | 126 | 1 | 0.313 | -0.123 | 0.328 | 0.077 |

The usable signals are A549, LUMI component transfer, and LuT selectivity.
HBEC-ALI, intratracheal expression, and barcode uptake are currently too small
or too weak under held-lipid evaluation to serve as standalone steering heads.
LuT round transfer is a warning against treating within-library performance as
prospective calibration.

## Qualified linear/tree comparison

The qualified signals were subsequently evaluated with Ridge, the frozen
ExtraTrees predictions, and XGBoost under identical fold assignments. Every
attempt is retained in
[`../../diagnostics/pan_lung_qualified_matrix_comparison_predictions.csv`](../../diagnostics/pan_lung_qualified_matrix_comparison_predictions.csv),
with calibration and ranking metrics in
[`../../diagnostics/pan_lung_qualified_matrix_comparison_metrics.json`](../../diagnostics/pan_lung_qualified_matrix_comparison_metrics.json).
The table contains **31,485 predictions**: 9,481 rows per full-structure model
and 1,014 rows per LuT component/SAR model.

| Signal / split | Ridge ρ | ExtraTrees ρ | XGBoost ρ | Current conclusion |
|---|---:|---:|---:|---|
| A549 held lipid | 0.526 | **0.568** | 0.558 | ExtraTrees leads rank and R² |
| LUMI held R1 | 0.355 | **0.476** | 0.437 | ExtraTrees leads |
| LUMI held R2 | 0.328 | **0.735** | 0.711 | ExtraTrees leads; Ridge is badly miscalibrated |
| LUMI held R3 | 0.764 | **0.821** | 0.793 | ExtraTrees leads |
| LUMI held R4 | 0.520 | **0.751** | 0.716 | ExtraTrees leads |
| LuT selectivity held head | **0.751** | 0.737 | 0.747 | Similar rank; XGBoost has lowest MAE |
| LuT selectivity held tail | 0.705 | **0.722** | 0.617 | ExtraTrees leads rank |
| LuT selectivity round 1 → 2 | 0.024 | **0.313** | 0.201 | ExtraTrees ranks best, but R² remains negative |

Ranking, enrichment, and calibration do not always select the same cell:

| Signal / split | Rank winner | Top-decile winner | Calibration readout | Selection status |
|---|---|---|---|---|
| A549 held lipid | ExtraTrees, ρ 0.568 | Ridge, recall 0.254 | **ExtraTrees:** bias -0.013, slope 1.032 | ExtraTrees candidate |
| LUMI held R1 | ExtraTrees, ρ 0.476 | XGBoost, recall 0.318 | **ExtraTrees:** bias 0.003, slope 0.940 | ExtraTrees candidate |
| LUMI held R2 | ExtraTrees, ρ 0.735 | ExtraTrees, recall 0.500 | **ExtraTrees:** bias 0.115, slope 0.951 | ExtraTrees candidate |
| LUMI held R3 | ExtraTrees, ρ 0.821 | ExtraTrees/XGBoost, recall 0.578 | **ExtraTrees:** bias -0.022, slope 1.023 | ExtraTrees candidate |
| LUMI held R4 | ExtraTrees, ρ 0.751 | XGBoost, recall 0.365 | XGBoost has better slope (0.961); ExtraTrees ranks better | Bootstrap paired ranking delta |
| LuT held head | Ridge, ρ 0.751 | Ridge, recall 0.244 | Ridge and ExtraTrees are both near unit slope | Ridge candidate |
| LuT held tail | ExtraTrees, ρ 0.722 | XGBoost, recall 0.489 | Ridge is best calibrated (bias 0.006, slope 0.952) | No single winner; bootstrap all three |
| LuT round 1 → 2 | ExtraTrees, ρ 0.313 | Ridge, recall 0.231 | Ridge is invalidly shifted (bias 1.031, slope -0.019); neither tree model is calibrated | No prospective oracle qualifies |

This does not justify pooling the signals. ExtraTrees is the current candidate
for A549 and LUMI. LuT selectivity requires split-aware uncertainty analysis:
the linear model is competitive on held heads but catastrophically shifted on
round transfer (mean prediction bias 1.031 and R² -8.123), while ExtraTrees is
the most consistent ranker across held-tail and round-transfer tests.

## Conservative ensemble qualification gate

The frozen OOF predictions now feed a leakage-resistant calibrated ensemble.
For every test fold, affine calibration and a 90% absolute-residual radius are
fit using only the other held-group folds. The reported pessimistic score is:

`calibrated ensemble mean - held-fold conformal radius - model disagreement`.

The disagreement term is an ensemble dispersion proxy, not a Bayesian
posterior standard deviation. Scores remain in assay-native units and cannot
be compared or added across typed heads. The machine-readable contract is
[`../../configs/pan_lung_qualified_ensemble_contract_v1.json`](../../configs/pan_lung_qualified_ensemble_contract_v1.json),
with the explicit qualification table at
[`../../diagnostics/pan_lung_qualified_ensemble_qualification.csv`](../../diagnostics/pan_lung_qualified_ensemble_qualification.csv).

| Domain / split | Rows | Mean ρ | Pessimistic ρ | Top-decile recall | Calibration bias / slope | Conservative coverage | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| A549 held lipid | 1,801 | 0.573 | 0.543 | 0.249 | 0.001 / 1.135 | 0.936 | qualified |
| LUMI held R1 | 1,920 | 0.331 | 0.297 | 0.260 | 0.065 / 0.692 | 0.896 | **diagnostic only** |
| LUMI held R2 | 1,920 | 0.696 | 0.688 | 0.443 | 0.021 / 0.918 | 0.891 | qualified |
| LUMI held R3 | 1,920 | 0.796 | 0.791 | 0.557 | -0.005 / 0.974 | 0.910 | qualified |
| LUMI held R4 | 1,920 | 0.726 | 0.745 | 0.286 | -0.088 / 1.000 | 0.904 | qualified |
| LuT selectivity held head | 444 | 0.749 | 0.759 | 0.200 | 0.000 / 1.016 | 0.917 | qualified |
| LuT selectivity held tail | 444 | 0.727 | 0.728 | 0.378 | 0.002 / 1.210 | 0.944 | qualified |

LUMI R1 is intentionally not promoted: cross-fitted calibration exposes a
rank drop to 0.331 even though its uncalibrated component models looked
stronger. LuT round transfer is absent from the qualification set by design;
it remains diagnostic evidence against prospective extrapolation. Six policies
are qualified only for domain-bounded filtering.

## Filtering-only serialized bundle

Eight full-data members are serialized under
[`../../artifacts/oracles/pan_lung_filtering_v1/manifest.json`](../../artifacts/oracles/pan_lung_filtering_v1/manifest.json):
three A549 models, two LUMI models, and three LuT-selectivity models. Every
artifact has a relative path, byte count, SHA-256 digest, software-version
record, and load-time hash verification.

The molecular applicability gate requires both:

1. maximum Morgan similarity to training chemistry above the fifth percentile
   of leave-one-out nearest-neighbor similarity; and
2. robust descriptor distance below the 99.5th percentile of training values.

This produces intentionally narrow thresholds: Tanimoto ≥ 0.805 for A549 and
≥ 0.875 for LUMI. These values reflect the dense combinatorial source libraries
and must not be relaxed using prospective outcomes. The LuT gate admits only
released heads, released tails, and round-1/2 contexts. It permits an unseen
pairing of known components but rejects any unseen component because full LuT
product structures are unavailable.

The integration audit at
[`../../diagnostics/pan_lung_filtering_admission_test.json`](../../diagnostics/pan_lung_filtering_admission_test.json)
verifies exact in-domain admission, leave-one-structure-out molecular
admission, molecular OOD rejection with no leaked score, a known LuT pair, an
admitted novel pairing of known LuT components, and rejection of an unseen LuT
head with no leaked score. The inference CLI is
[`../../scripts/score_pan_lung_filtering.py`](../../scripts/score_pan_lung_filtering.py).

Reward fine-tuning remains disabled. Before activation, the bundle must be
independently verified in the reward runtime, the reward wrapper must make OOD
scoring impossible, thresholds and budgets must be pre-registered, and domain
drift/score hacking must be monitored.

### Clean-runtime and reward-guard verification

The bundle was copied outside the repository and loaded in a fresh virtual
environment with user-site and `PYTHONPATH` disabled. The verifier reuses the
exact preinstalled binary dependency versions recorded in the manifest, but
installs the COMPOSE package into isolated site-packages and relocates every
model/AD artifact. All 15 relocated hashes pass. The result is frozen at
[`../../diagnostics/pan_lung_filtering_clean_runtime_verification.json`](../../diagnostics/pan_lung_filtering_clean_runtime_verification.json).

The hard wrapper in
[`../../src/compose_v4/oracles/reward_guard.py`](../../src/compose_v4/oracles/reward_guard.py)
returns `reward: null` for OOD candidates and raises on an admitted reward
request under the released bundle. Activation requires both a future bundle
authorization and an explicit frozen preregistration bound to that exact
manifest SHA. The checked-in
[`../../configs/pan_lung_reward_preregistration.template.json`](../../configs/pan_lung_reward_preregistration.template.json)
is deliberately non-authorizing. Four focused tests cover admission, OOD
no-reward behavior, admitted-request blocking, and the two-switch
authorization contract.

The earlier source-qualified implementations remain useful audit references:

- [`../../scripts/train_lumi_oracle_baselines.py`](../../scripts/train_lumi_oracle_baselines.py)
  → [`../../diagnostics/lumi_lab_4cr1920_baselines.json`](../../diagnostics/lumi_lab_4cr1920_baselines.json)
- [`../../scripts/train_lnpdb_lung_oracle_baselines.py`](../../scripts/train_lnpdb_lung_oracle_baselines.py)
  → [`../../diagnostics/lnpdb_lung_task_baselines.json`](../../diagnostics/lnpdb_lung_task_baselines.json)

LuT is evaluated in the adjacent component/SAR × ExtraTrees cell with
[`../../scripts/train_lut_oracle_baselines.py`](../../scripts/train_lut_oracle_baselines.py).
The new joint run preserves that conclusion under one shared ExtraTrees
contract; exact values differ slightly because it fixes one estimator and
hyperparameterization across cells rather than selecting each source's best
standalone baseline.

## Next measurable artifact

Filtering may now use the verified pessimistic scores. Reward fine-tuning is a
separate future decision and remains disabled unless thresholds and candidate
budgets are explicitly preregistered against the exact bundle hash, bundle
authorization is separately changed, and drift/score-hacking monitoring is in
place.
