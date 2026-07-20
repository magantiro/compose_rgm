# COMPOSE-Lipid data and oracle lane: Claude handoff

**Handoff date:** 2026-07-20  
**Ownership:** Claude owns the structural-pretraining corpus and pan-lung oracle
matrix lanes. Codex retains ownership of the COMPOSE/RGM conditional and
unconditional generator implementation and training.

## Scope boundary

Preserve the existing scientific thesis: COMPOSE-Lipid imports the
validity-closed Rewrite Generator Matching model from Paper 1. This lane builds
the lipid structural corpus and the lung-selection oracle; it must not replace
the generator with a fragment, diffusion, DiGress, or GEM implementation.

Do not edit generator/model/training code outside `src/compose_v4/lipids/`,
`src/compose_v4/oracles/`, corpus/oracle scripts, configs, diagnostics, and the
corresponding research documents without coordinating with Codex. Work on a
separate branch or worktree and return commits or patches plus exact artifact
hashes.

## Mandatory read order and governing plans

This handoff is an index, not a substitute for the governing documents. Claude
must read these files completely, in this order, before changing code or
scientific plans:

1. `docs/research_plans/compose_two_paper_execution_plan.html` -- canonical
   program index, paper boundary, activation gates, experiment registry,
   decision rules, and dependencies between Paper 1 and Paper 2.
2. `docs/research_plans/paper2_compose_lipid.html` -- governing Nature
   Biotechnology thesis, six-figure story, selection arms, preregistration,
   translational bar, failure fallbacks, and exact novelty boundary.
3. `docs/research_plans/paper1_compose_methods.html` -- the method that Paper 2
   inherits, the evidence required before lipid-model activation, conditional
   benchmark contract, and claims that belong exclusively to Paper 1.
4. `docs/research_plans/compose_lipid_pretraining_corpus_audit.md` -- exact
   locally possessed corpus counts and the boundary between reported and
   obtained structures.
5. `docs/research_plans/defensible_lipid_corpus_spec.md` -- complete
   reaction-diverse 500k corpus design and qualification rules.
6. `docs/research_plans/lipid_dataset_intake.md` -- source-by-source access,
   licensing, structure, label, overlap, and blocker audit.
7. `docs/research_plans/pan_lung_oracle_corpus.md` -- multi-study schema,
   typed heads, representation-by-model matrix, prospective score, leakage
   controls, and COMPOSE interface.
8. `docs/audits/2026-07-20_lung_oracle_results.md` -- first reproducible oracle
   results and negative transfer findings.
9. `docs/audits/2026-07-20_agile_ugi_corpus_bias.md` -- why AGILE cannot define
   the 500k distribution.
10. `docs/CLAUDE_LIPID_HANDOFF_MANIFEST_V1.json` -- size and SHA-256 ledger for
    the canonical documents and result artifacts above.

If an HTML plan and a short Markdown note appear inconsistent, the HTML plan
and its explicit gates control unless a later dated, evidence-backed audit
records a deliberate change. Do not silently rewrite the HTML plans. Propose a
redline with the evidence and claim consequence.

## Full scientific context

### The two-paper program

**Paper 1: COMPOSE / Rewrite Generator Matching (ICLR methods paper).** It owns
the structured molecular RGM construction: executable graph rewrites are the
events of a learned continuous-time Markov chain; stochastic rewriting supplies
rules, matches, application conditions, inverses, and aggregation; Generator
Matching learns contextual marginal rates from certified endpoint-conditioned
programs. The intended state space contains complete, connected,
valence-admissible 2D molecular graphs. Inference is target-free ancestral CTMC
sampling, without beam search, an endpoint, terminal repair, or a compiler.
The owned novelty is the intersection--a validity-closed, non-monotone,
flexible-size molecular GM construction with a chemically admissible bridge,
coordinated executable events, and learned transitions between complete
connected molecules--not the invention of rewriting, CTMCs, GM, insertion,
deletion, or validity methods separately.

**Paper 2: COMPOSE-Lipid (Nature Biotechnology application paper).** It imports
the frozen RGM core and owns lipid chemistry support, structural pretraining,
heterogeneous pulmonary data, the study-aware oracle, prospective selection,
synthesis/formulation, in-vivo hit-rate enrichment, functional delivery, and
linker-specific SAR. It should explain the model sufficiently to establish
candidate provenance and genuine learned generation, but it is not a second ML
theory paper.

The papers share a frozen engine, not duplicated claims. Paper 2 data work may
proceed before Paper 1 is complete or accepted. Lipid generator training is
authorized after a credible Paper 1 base, recovery/revision qualification, and
one matched conditional/optimization proof. Prospective candidate lock requires
the full versioned lipid-ready core, oracle, selection, and wet-lab contracts.

### Paper 2 thesis and novelty boundary

Working thesis: COMPOSE-Lipid adapts RGM to ionizable-lipid chemistry,
generates complete molecules through chemistry-guarded stochastic rewrites
rather than searching a fixed virtual library, and prospectively tests whether
its generated candidates improve in-vivo lung-delivery hit rate under a matched
experimental budget.

Subject to a dated submission-time prior-art audit, the distinctive claim is:

> To our knowledge, the first learned de-novo ionizable-lipid generative model
> whose model-sampled candidates--rather than candidates selected from a
> pre-enumerated virtual library--are prospectively synthesized, formulated,
> and validated in vivo.

Do not claim the first AI lipid study, first lipid generator, first active
learning study, first combinatorial lipid campaign, or first AI-plus-in-vivo
study separately. LUMI-lab and MOLEA are translational AI comparators but not
learned molecular generators. LiGen and synthesis-aware lipid generators are
generative precedents but do not currently establish prospective in-vivo
validation of model-sampled candidates. The broad first claim must be removed
or narrowed if the final dated audit finds a genuine prior instance.

The team's GEM 2026 workshop paper, *Synthesis-constrained discrete diffusion
for ionizable lipid generation*, is citation-only prior work. COMPOSE-Lipid
reuses no GEM data, preprocessing, code, model components, checkpoints, text,
or figures. Do not import GEM's DiGress-style paradigm into this lane.

### What RGM contributes to the lipid campaign

- Hard linker/scaffold preservation can be an application condition that
  removes violating rewrites rather than a soft reward.
- Grow, shrink, retype/restatement, bond revision, Graft, and whole-ring-system
  events permit non-monotone changes in size and topology.
- Every committed intermediate is a complete connected molecule, enabling an
  oracle to score any event state and permitting anytime valid candidates.
- Arm B can change rates only among legal successors; guidance never expands
  chemical support.
- The same rewrite semantics can support de-novo generation and, after inverse
  operator/recovery training, local lead editing.

These are algorithmic affordances, not claims of synthesizability, strain-free
3D geometry, oracle accuracy, formulation success, or biological efficacy. A
rewrite path is not a synthetic route. Reaction certification is a separate
corpus/candidate requirement.

### Current Paper 1 dependency snapshot

The last developmental small-molecule checkpoint produced 2,000/2,000 valid,
unique, and novel neutral C/N/O/F samples with at most 40 heavy atoms. Its
full-support FCD was 16.08 (bootstrap 15.61--16.54) and matched-support FCD was
11.29 (10.82--11.76). Residual failures include covariance mismatch,
aromatic/fused-ring undercoverage, and flexible-chain overrepresentation. FCD
is a sufficiency diagnostic, not the Paper 1 headline and not a prerequisite
for beginning lipid data work by itself.

The current conditional gate is a frozen-base QED residual-sidecar pilot. The
canonical 2026-07-20 launch is Modal app
`ap-Hhfeh6k91K5Y0Fyd1vELUl`, call
`fc-01KY09913W095CCAPXCYD63D9Y`, artifact directory
`compose-v4-artifacts/compose-v4-griddd-qed-frozen-residual-pilot-20260720-v2-canonical`.
This run is owned and monitored by Codex and is recorded only so Claude
understands the activation dependency; Claude must not control it.

The locked QED optimization reproduction uses 800 starting molecules, 20
candidates per start, starting QED 0.70--0.80, target QED at least 0.90, and
fingerprint Tanimoto similarity at least 0.40 under matched preprocessing,
oracle/proposal budget, seeds, selection rule, and all-attempt denominator.
Paper 2 data/oracle preparation does not wait for the full Paper 1 matrix.

## Nature Biotechnology evidence chain

The six-figure plan in `paper2_compose_lipid.html` is locked as follows:

1. **Lipid RGM substrate.** Show legal state space, rules/matches/application
   conditions, GM, ancestral CTMC, flexible-size source, exact lipid element/
   charge/size/operator support, and representative de-novo and linker-frozen
   trajectories. Say rewrite-executable, not synthesis-executable.
2. **Unconditional lipid generation.** At least 10,000 ancestral samples;
   provenance and hard splits; every-intermediate validity/connectivity;
   coverage/density, memorization, scaling, and matched generative/enumeration
   comparators. Audit head/linker/tail architecture, ionizable amine class,
   charge, molecular weight, tail count/length/branching, linker/degradability,
   symmetry, and formulation-independent phenotype. FCD is supplementary.
3. **Useful and synthesis-eligible coverage.** Architecture map, rational and
   reaction-accessible chemistry, novel-linker campaign, failure atlas, route
   review, and blinded expert audit. Keep graph validity, lipid phenotype,
   synthesis feasibility, and biological plausibility as four separate claims.
4. **Frozen pan-lung oracle and candidate nomination.** Audited multi-study
   corpus, hard splits, simple and modern matrix cells, calibration,
   uncertainty, applicability domain, campaign, diversity-aware selection,
   matched controls, and locked panel.
5. **Prospective in-vivo hit-rate enrichment.** Preregistered contrasts,
   randomized/blinded formulation and testing, synthesis/formulation QC,
   matched route/cargo/dose/batch, primary lung endpoint, effect sizes and
   confidence intervals, biodistribution, target-cell localization,
   tolerability, and independent replication. Wet-lab results, not oracle
   scores, adjudicate success.
6. **Constrained optimization and functional delivery.** Prespecified lead,
   linker-frozen source-conditional RGM, prospective SAR, independent retest,
   functional cargo, cell specificity, durability, safety, and mechanistic
   support. Luciferase-only delivery is unlikely to carry the target venue;
   editing, protein replacement, or disease-relevant function is load-bearing.

## Prospective computational and experimental design

Four candidate-selection arms are desired if synthesis and statistical power
permit:

- **Arm A, required:** sample the frozen unconditional lipid RGM ancestrally;
  score with the frozen pan-lung oracle, applicability-domain/uncertainty rule,
  synthesis filters, and diversity-aware Pareto selector.
- **Arm B, gated:** use the identical base checkpoint and oracle to guide legal
  successor rates or KL/trajectory-anchored reward fine-tuning. It activates
  only after Paper 1 guidance and lipid-oracle gates. If it exploits uncertainty,
  leaves domain, or collapses diversity, it is dropped from the required wet
  lab and retained as a computational negative or round-2 method.
- **Random/DoE control:** matched chemistry envelope and experimental budget.
- **Expert/rational control:** matched synthesis and testing budget.

Primary prospective contrast: AI-selected candidates versus matched non-AI
controls for hit-rate enrichment. Arm A versus Arm B is prespecified secondary
unless powered. All arms share the same base checkpoint, oracle, chemistry
envelope, synthesis filters, compute/candidate budget, formulation policy, and
duplicate-resolution rule.

The intended biological target is non-liver pulmonary delivery. The exact
route, lung compartment/cell type, cargo, dose, formulation, primary endpoint,
hit threshold, and minimum functional claim are not yet locked and must be
preregistered before synthesis. The current novel linker derives from
Michael-addition chemistry. The desired transfer principle is that head and
tail knowledge can transfer across datasets while the new linker remains a
hard structural constraint; however, extrapolation is not assumed. If the
linker is outside the oracle applicability domain, run a small calibration/
active-learning round rather than reward-hacking an unsupported model.

## Source registry: structural pretraining

- **LNPDB:** obtained, MIT repository, 19,797 local rows and 12,837 canonical
  structures. Primary observed anchor and formulation/response source.
- **LiON/LNP_ML:** obtained, MIT repository, 13,069 rows and 9,194 canonical
  structures; 8,518 overlap LNPDB. It is mainly a released preprocessing/model
  view, not 9,194 independent additions.
- **AGILE measured:** 1,200 Ugi-3CR products with A/B/C components; 1,180/1,200
  overlap LNPDB. Useful for component and reaction reconstruction, not an
  independent broad distribution.
- **AGILE virtual:** 12,276 released candidates, 11,076 virtual-only after exact
  subtraction. The released code performs fixed-core wildcard stitching, not
  a general atom-mapped Ugi reaction. Retain as a labeled auxiliary layer only.
- **LUMI 4CR:** obtained from Zenodo DOI `10.5281/zenodo.17771224`; 1,920 unique
  measured full structures with component codes; CC-BY-4.0.
- **LuT 444:** obtained component/SAR and response tables but no full product
  structures. Excluded from structural pretraining until products and reaction
  are reconstructed exactly.
- **LiGen LNP-Virtual900k:** reports 900,351 structures, not obtained, no
  independent downloadable licensed artifact located. Request from authors;
  never count before receipt and audit.
- **LipidBERT/METiS 10M:** proprietary/in-house, unavailable, exclude as an
  executable dependency.
- **Ouyang fixed-template ~20M:** structures/predictions available on request;
  892 tails x 237 heads with Tail2=Tail3. Useful precedent and possible
  auxiliary source, not a general prior.
- **Bowen Li 4CR 40k:** reported but not recovered as a verified machine-readable
  licensed corpus. Request, audit, and do not count meanwhile.

The 500k registry should cover complementary families, not minor variants of
one scaffold: epoxide opening; aza-Michael addition; Ugi 3CR; Ugi 4CR;
Passerini 3CR; amine-aldehyde-alkyne/A3 coupling; iPhos/dioxaphospholane
chemistry; epoxide opening plus acyl-chloride esterification; the published
ricinoleic-acrylate/alcohol-carbonate pulmonary platform; and separately
validated esterification, amidation, carbamate/carbonate, disulfide, and
reductive-amination routes.

## Source registry: pan-lung evidence

- **LNPDB:** 1,975 lung observations and 291 unique canonical lipids across
  three studies. Typed heads: A549 in-vitro expression 1,801 rows/176 lipids;
  HBEC-ALI in-vitro expression 29/28; local intratracheal functional expression
  49/26; systemic-IV barcoded uptake 96/62.
- **LUMI:** 1,920 individual human-bronchial-epithelial in-vitro responses with
  full structures. Only six leads were advanced intratracheally; never relabel
  all 1,920 as in-vivo lung measurements.
- **LuT:** 444 individually measured component-defined compounds, 318 round 1
  and 126 round 2; IV 0.1 mg/kg luciferase mRNA, 6 h, mouse; expression and
  lung-selectivity heads. The varied LuT component is used with fixed 4A3-SC7
  and is not interchangeable with a conventional ionizable-lipid label.
- **RCB/NRA pulmonary:** 720 synthesized structures; broad A549 screen and 49
  individually usable intratracheal functional rows. Already represented in
  LNPDB; do not append duplicates.
- **CAD/barcoded lung:** 180 lipids, 96 barcoded IV formulations; barcode uptake
  is biodistribution, not functional expression. Canonical 96-row LNPDB copy.
- **Branched-tail library:** 580 synthesized lipidoids, only 49 individually
  tested in vivo. Supplement not yet frozen.
- **PPZ/FIND:** 65 formulations but only eight distinct ionizable lipids; 14
  cell types and no detected lung/kidney delivery. Potential negative/context
  supervision, not 65 positive lung chemotypes.
- **Split-Ugi pulmonary:** 155 lipopolymers, mainly in-vitro screening with one
  principal in-vivo lead. Do not inflate in-vivo count.
- **STAAR extrahepatic:** 22 individual systemic-IV lipids in the relevant
  table; freeze supplement before admission. The earlier 1,500-lipid screen was
  intramuscular, not a lung dataset.

Measurements must preserve formulation, cargo, dose, route, species, time,
tissue/cell, assay, readout type, study, batch, replicate, pooling/barcode, and
source provenance. Chemical identity deduplication must not erase distinct
formulation/measurement records. Technical or duplicated publication rows may
not cross train/test folds.

## Lane A: 500k structural-pretraining corpus

### Current verified state

- 15,433 unique measured/observed canonical ionizable lipids (`R0`).
- 26,509 unique structures after adding the released AGILE virtual candidates.
- 11,076 structures are AGILE virtual-only additions.
- Zero new molecules have been generated by our reaction enumerator.
- The AGILE virtual layer is highly redundant and cannot be multiplied to
  claim broad chemistry coverage.
- LiGen reports 900,351 virtual structures, but no public licensed structure
  artifact has yet been found.
- LipidBERT/METiS reports a proprietary 10M library and is not an executable
  dependency.

Ground-truth artifacts:

- `artifacts/datasets/compose_lipid_pretraining_v1/canonicalization_audit.json`
- `artifacts/datasets/compose_lipid_pretraining_v1/canonical_structure_union.csv`
- `artifacts/datasets/compose_lipid_pretraining_v1/r0_observed_real_structures.csv`
- `docs/research_plans/compose_lipid_pretraining_corpus_audit.md`
- `docs/research_plans/defensible_lipid_corpus_spec.md`
- `docs/research_plans/lipid_dataset_intake.md`

### Required outcome

Produce a reproducible release of approximately 500,000 unique,
route-certified ionizable-lipid structures. It must be heterogeneous across
reaction family and head/linker/tail architecture, not a blind Cartesian
product or one dominant Ugi family.

### Required execution

1. Assemble evidence packets for at least six complementary reaction families.
   Include exact primary-source substrate/product examples, reactant roles,
   chemoselectivity, stereochemistry, charge/protonation policy, and source
   locators. Never infer reaction SMARTS from a family name.
2. Promote a transform to `qualified_for_enumeration` only after freezing an
   atom-mapped transform and reconstructing source products exactly.
3. Freeze licensed building-block manifests with role, vendor/catalog snapshot,
   canonical structure, purity/availability metadata, and hashes.
4. Run a 1,000-product smoke release and audit validity, uniqueness, route
   replay, structural-region annotations, and COMPOSE eligibility.
5. Run a 50k--100k stratified pilot. Measure reaction-family balance,
   head/linker/tail coverage, nearest-neighbor redundancy, LNPDB coverage,
   property marginals, and held-component/linker/reaction-family leakage.
6. Expand to 500k only after the pilot gates pass. Preserve every rejected
   product and rejection reason separately from the selected release.
7. Freeze train/validation/test splits that jointly hold out exact structures,
   components, linkers/scaffolds, studies, and reaction families.

Existing fail-closed implementation:

- `configs/lipid_reactions/reaction_registry.schema.json`
- `configs/lipid_reactions/pilot_literature_specs_v1.json`
- `configs/lipid_reactions/pilot_enumeration_v1.json`
- `scripts/enumerate_lipid_reaction_pilot.py`
- `src/compose_v4/lipids/`

The three present registry entries--epoxide opening, aza-Michael addition, and
AGILE Ugi 3CR--remain literature specifications and are not qualified. Their
atom-mapped reaction SMARTS are intentionally null until reconstruction passes.

## Lane B: full pan-lung oracle matrix

### Current verified evidence

- LNPDB: 1,975 lung-associated observations, 291 unique supplied lipids;
  1,801 A549 rows are the strongest currently qualified molecular head.
- LUMI: 1,920 full structures with human bronchial epithelial-cell response
  and component codes.
- LuT: 444 systemic-IV in-vivo compounds with lung expression and selectivity;
  held-component prediction is promising but round-to-round transfer fails.
- Other small LNPDB lung heads are currently inadequate as independent reward
  models and must remain auxiliary typed heads.

The matrix axes are representations by model families--not datasets by models:

- Representations: molecular fingerprints/descriptors plus context; explicit
  region/component features; frozen molecular embeddings; end-to-end molecular
  graph; full-formulation set encoder.
- Models: Ridge/Elastic Net; ExtraTrees; boosted trees; shallow masked multitask
  MLP; masked graph model; calibrated ensemble.

Completed fundamental cells:

- `R1xM1`, `R1xM2`, `R1xM3`: molecular representation with Ridge,
  ExtraTrees, and XGBoost.
- `R2xM1`, `R2xM2`, `R2xM3`: region/component representation with the same
  simple model families where component data exist.
- Qualified ensemble/filtering bundles exist for A549, eligible LUMI axes, and
  LuT selectivity behind applicability-domain gates.

Representative qualified results:

- A549 ensemble: mean held-lipid Spearman about 0.573; top-decile enrichment
  about 2.47x.
- LUMI: best qualified held-component ensemble Spearman about 0.796;
  top-decile enrichment about 5.57x.
- LuT selectivity: held-head Spearman about 0.749 and held-tail about 0.727.
- LuT round transfer is prohibited as a deployment reward because it fails.

Ground-truth artifacts:

- `configs/pan_lung_corpus_manifest_v1.json`
- `configs/pan_lung_oracle_matrix_v1.json`
- `configs/pan_lung_qualified_ensemble_contract_v1.json`
- `diagnostics/pan_lung_qualified_matrix_comparison_metrics.json`
- `diagnostics/pan_lung_qualified_ensemble_metrics.json`
- `artifacts/oracles/pan_lung_filtering_v1/`
- `docs/audits/2026-07-20_lung_oracle_results.md`
- `docs/research_plans/pan_lung_oracle_corpus.md`

### Required outcome

Complete the representation-by-model matrix over the entire curated pulmonary
corpus using typed, semantically distinct heads and leakage-resistant splits.
Produce a calibrated, uncertainty-aware filtering oracle that can rank generated
lipids for the locked prospective target without pretending that in-vitro
transfection, biodistribution, local delivery, and systemic functional
expression are interchangeable labels.

### Required execution

1. Finish the canonical row-level multi-study manifest and duplicate/provenance
   joins before adding models.
2. Add `R3` frozen molecular embeddings and the shallow masked multitask MLP.
3. Evaluate end-to-end graph and full-formulation encoders only after the simple
   cells under the identical hard splits.
4. Keep source/study/route/readout as typed context and evaluation groups, not
   matrix columns.
5. Qualify by leave-study/library, held-lipid, held-head/tail/linker, held
   reaction family, temporal, and formulation splits. Random folds are
   diagnostic only.
6. Report ranking, top-decile enrichment/recall, calibration, uncertainty
   coverage, abstention, applicability domain, and clean-runtime serialization.
7. Select a prospective systemic-lung target utility only after qualification.
   Related in-vitro heads may improve representation learning but may not be
   relabeled as the in-vivo target.
8. Preserve a novel-linker applicability gate. If the new Michael-addition
   linker lies outside domain, require a small active-learning calibration set
   rather than silently extrapolating.

## Return contract

For every milestone return:

- exact commands and environment;
- raw-source URLs/DOIs, licenses, file hashes, and local artifact paths;
- canonical counts before and after every filter/deduplication operation;
- tests and failures, including rejected reactions/products/models;
- comparison against the frozen prior artifact rather than rewritten history;
- committed code/config/docs on a separate branch or a patch that Codex can
  review and merge.

Never report a paper's stated virtual-library size as locally possessed data.
Never invent missing chemistry or pool biologically incompatible endpoints to
inflate sample size.
