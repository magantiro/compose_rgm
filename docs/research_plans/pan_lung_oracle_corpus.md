# Pan-lung oracle corpus and steering plan

**Status:** active evidence and implementation plan. Counts are admitted only
after direct inspection of a released artifact. This document does not collapse
in-vitro transfection, biodistribution, functional expression, and editing into
one interchangeable response.

## Decision

COMPOSE-Lipid will use a **deduplicated, multi-study pulmonary evidence
corpus**, not an LNPDB-only table and not a single-paper LuT model. The corpus
will include every accessible lung-relevant lipid/formulation record that has a
traceable structure, experimental context, and usable response. A shared lipid
and formulation encoder will learn across sources, while source-, route-,
cell-, and readout-specific prediction heads preserve what each experiment
actually measured.

The prospective steering reward will be derived only from the calibrated head
matching the locked target product profile. Related studies improve the shared
representation and uncertainty estimate; they do not silently redefine the
prospective endpoint.

## Current verified source registry

| Source | Verified usable scale | Experimental context | Role in the oracle |
|---|---:|---|---|
| LNPDB | 19,528 measured formulation rows overall; direct audit found 1,975 lung rows/291 supplied unique lipids across three studies and 145 in-vivo lung rows/88 unique lipids across two studies | Mixed in-vitro and in-vivo; multiple routes and readouts | Canonical cross-study anchor, formulation metadata, and overlap registry |
| LiON public release | 13,069 assay rows, 9,194 canonical ionizable-lipid structures overall; direct audit found lung epithelial in-vitro and smaller intratracheal/IV lung slices | Predictive-model training release spanning multiple studies and conditions | Reproducible preprocessing/model precedent and supplemental lung labels; deduplicate against LNPDB before use |
| LuT | 444 individually synthesized and evaluated LuT LNPs; official Fig. 2/3 workbooks are downloaded and hashed, yielding exactly 318 + 126 non-control rows with paired lung-expression and selectivity labels | Intravenous in-vivo luciferase expression and organ selectivity; later cell-type and functional lead studies | Largest currently verified single direct in-vivo lung structure--activity library; first baselines reach held-head/held-tail Spearman 0.53/0.55 for expression and 0.76/0.75 for selectivity, while round transfer falls to about 0.31 |
| LUMI-lab | Official Zenodo release contains 1,920 measured structures in `4CR-1920.csv`, with log2 RLU, full molecular SMILES, and Markush component code | Iterative human-bronchial-epithelial-cell in-vitro pulmonary screen with six leading lipids taken in vivo by intratracheal administration | Immediately usable, high-value pulmonary in-vitro task and AI-oracle precedent; retain as its own assay/study head and hash the frozen Zenodo artifact |
| RCB/NRA pulmonary library | 720 synthesized structures; released study contains the combinatorial library and tiered pulmonary evaluation | A549/in-vitro screening plus pooled/tiered and smaller individual intratracheal in-vivo evaluation | Broad pulmonary chemistry supervision; pooled and individual measurements receive different endpoint types |
| CAD/barcoded lung study | 180 synthesized CAD lipids; 96 LNPs entered pooled in-vivo barcode screening | IV barcode accumulation in lung and other organs, followed by individual functional validation of a small lead set | Biodistribution/selectivity head and off-target supervision; barcode uptake is not treated as functional expression |
| Branched-tail library | 580 synthesized lipidoids; 49 selected for individual in-vivo organ testing | Broad in-vitro/physicochemical screen followed by individual in-vivo expression | Structural and auxiliary-task supervision; only the 49 tested candidates receive in-vivo labels |
| PPZ/FIND | 65 formulations over a small number of distinct PPZ lipid structures, measured across 14 cell types | Functional in-vivo delivery/editing readouts with barcode-assisted identification | Cell-type/formulation supervision; useful for pulmonary immune/endothelial selectivity but not 65 independent lipid chemotypes |
| Split-Ugi pulmonary library | 155 synthesized ionizable lipopolymers; broad library is primarily screened in vitro and a lead is advanced in vivo | In-vitro pulmonary-relevant screening plus limited lead validation | Auxiliary pulmonary task; not a 155-example in-vivo dataset |

The registry will expand when a released source passes the same artifact audit.
Headline library size never substitutes for the number of independently usable
labels.

## Required canonical schema

Every raw measurement maps to a record with the following minimum fields:

```text
source_id, publication_id, experiment_id, batch_id
raw_lipid_id, raw_smiles, canonical_lipid_id
formulation_id, helper_lipids, component_ratios, N_P_ratio
cargo, cargo_format, dose, route, species, sex, timepoint
tissue, cell_type, assay, readout_type, raw_value, raw_unit
replicate_id, pooling_type, barcode_id
structure_family, reaction_family, head, linker, tail annotations
source_file, source_row, license, sha256
```

`pooling_type` distinguishes individual, pooled mixture, and barcode-resolved
records. `readout_type` distinguishes at least in-vitro transfection,
functional in-vivo expression, editing, whole-organ biodistribution, cell-type
delivery, physicochemical measurement, and toxicity.

## Deduplication without data loss

Deduplication occurs at four levels:

1. **Chemical identity:** versioned canonicalization groups exact molecular
   duplicates while preserving raw strings, stereochemistry, charge, and salt
   provenance.
2. **Formulation identity:** identical ionizable lipid does not imply identical
   LNP. Helper lipids, ratios, cargo, and process metadata remain part of the
   input.
3. **Measurement identity:** duplicated copies of the same paper row across
   LNPDB, LiON, or another compilation are joined by provenance, not counted as
   independent observations.
4. **Biological replicate:** technical replicates can inform measurement noise
   but are never split across train and test.

The overlap matrix is frozen before model splitting. A test study cannot rejoin
training through a second compilation.

## Oracle architecture

### Model--representation matrix

Following the practical pattern used in PeptiVerse and mRNAutilus, oracle
selection begins with a controlled cross-product rather than an assumption that
the deepest graph model will win.

Candidate representations:

- Morgan/count fingerprints plus curated RDKit lipid descriptors;
- graph-network embeddings from the COMPOSE or LiON molecular encoder;
- frozen chemical-language-model embeddings;
- explicit head/linker/tail and formulation descriptors;
- concatenated molecular, formulation, route, cargo, dose, assay, and study
  context.

Candidate predictors:

- regularized linear/Elastic Net models;
- random forest and ExtraTrees;
- XGBoost or another carefully tuned boosting baseline;
- SVM/SVR where scale permits;
- shallow MLPs;
- end-to-end message-passing or attention models.

Every representation--predictor pair receives the same nested hard splits and
budget. Model selection is endpoint-specific. A diverse calibrated ensemble is
retained only when its held-study errors are complementary; averaging many
correlated models is not treated as uncertainty. This reproduces the useful
PeptiVerse lesson that strong frozen representations plus simple regularized
heads can match or beat more complex predictors, while retaining the graph
models as genuine competitors rather than presumed winners.

### Shared representation

- ionizable-lipid molecular encoder;
- optional helper-lipid encoders with permutation-aware pooling;
- formulation-ratio, cargo, dose, route, species, tissue/cell, assay, and
  timepoint covariates;
- explicit missingness indicators rather than invented values;
- reaction-family and study embeddings used only where they improve held-study
  validation and do not leak the target.

### Task heads

- pulmonary in-vitro transfection;
- whole-lung functional expression;
- whole-lung accumulation/biodistribution;
- lung-versus-liver and lung-versus-spleen selectivity;
- lung epithelial, endothelial, immune, and macrophage delivery where labels
  exist;
- editing or disease-relevant functional endpoints;
- formulation quality/toxicity auxiliary heads.

Heads may be regression, ranking, ordinal, binary hit, or censored models as
required by the source assay. Study-specific normalization is learned or
fitted within training folds only.

### Prospective steering score

For a locked target profile `T`, reward is not an average of all lung labels.
It is a calibrated target-specific utility:

```text
utility_T = potency_T
            + alpha * lung_selectivity_T
            - beta * liver_off_target_T
            - gamma * spleen_off_target_T
            - delta * toxicity_T
            - uncertainty_penalty_T
```

The coefficients, hit threshold, applicability-domain rule, and abstention
policy are frozen before candidate generation. If the exact target head has
insufficient external performance, the campaign switches to a small
active-learning round rather than trusting an extrapolative reward.

## Leakage-resistant evaluation

Random splits are diagnostics only. Qualification requires:

1. leave-one-study/library-out evaluation;
2. scaffold, linker, head, tail, and reaction-family holdouts;
3. temporal holdout when dates permit;
4. formulation holdout for repeated lipids;
5. route/readout transfer tests reported separately, never hidden in a pooled
   score;
6. prospective-target calibration, uncertainty coverage, and abstention;
7. ranking metrics because selection quality matters more than global RMSE.

Baselines include Morgan/descriptor linear and tree models, simple multitask
MLPs, message-passing models, and reproducible LiON/LANTERN-style predictors.
The modern model is adopted only if it improves held-study ranking or calibrated
selection, not because it is architecturally newer.

## Interface to COMPOSE

The same valid-successor scoring API serves both papers:

1. Paper 1 qualifies conditioning and reward fine-tuning on matched
   small-molecule property/optimization tasks such as the frozen ZINC-QED gate.
2. COMPOSE-Lipid swaps in the frozen pan-lung oracle ensemble and lipid
   applicability-domain rules.
3. Arm A samples from the frozen base lipid generator and ranks candidates.
4. Arm B changes rates only among legal rewrite successors, with a trajectory
   or KL anchor and the same oracle-call/compute budget.
5. Both arms apply identical synthesis, uncertainty, diversity, and formulation
   filters before prospective candidate lock.

Every committed molecular state remains valid and connected; oracle guidance
changes probabilities, not chemical support.

## Immediate execution sequence

1. Download and hash every primary released lung artifact.
2. Reconstruct the source registry with exact molecule, formulation,
   measurement, and overlap counts.
3. Build the canonical row-level manifest and provenance joins.
4. Freeze study/library/scaffold/linker/reaction-family splits.
5. Reproduce simple oracle baselines before any reward fine-tuning.
6. Select the target product profile and train the multitask oracle ensemble.
7. Qualify calibration, ranking, uncertainty, and abstention on held studies.
8. Run Arm A first; activate Arm B only after the Paper 1 guidance gate and
   lipid-oracle qualification pass.

The first executable oracle results and their interpretation are frozen in
[`../audits/2026-07-20_lung_oracle_results.md`](../audits/2026-07-20_lung_oracle_results.md).

## Current blocker boundary

No user login is currently required for LNPDB, LiON, LuT source data, AGILE,
COMET, LUMI-lab, or the public model repositories. The official LUMI-lab
Zenodo record (DOI `10.5281/zenodo.17771224`) exposes `4CR-1920.csv`: 1,920
rows with `RLU (log2)`, full molecular SMILES in `mol`, and `Markush code`.
The remaining work is artifact hashing, chemistry validation, component-code
parsing, and overlap/provenance auditing—not author access.
