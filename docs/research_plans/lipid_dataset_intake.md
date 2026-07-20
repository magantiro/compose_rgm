# COMPOSE-Lipid dataset intake

**Status:** working evidence ledger, not a change to the canonical COMPOSE or
COMPOSE-Lipid thesis.  The source note was written for a different block-based
project and contains unverified claims.  This ledger extracts only usable data
leads and maps them onto the existing RGM program.

## Interpretation boundary

COMPOSE-Lipid remains a validity-closed, flexible-size molecular rewrite
generator trained through Generator Matching.  It will not be replaced by an
H/L/T grammar or a frozen fragment assembler.  Head, linker, and tail
decompositions are useful **annotations** for data splits, structural
constraints, evaluation, and prospective selection.  The rewrite system must
still be able to revise atoms, bonds, topology, size, and complete ring systems
outside a fixed block cross-product.

## Source ledger

| Source lead | What the supplied note says | Current evidence status | Proposed role | Required before use |
|---|---|---|---|---|
| AGILE public release | 1,200 labeled lipids; HeLa and RAW measurements; Ugi-derived components | **Downloaded from the official MIT-licensed repository and locally inspected.** The component-annotated CSV has 1,200 valid full-molecule rows and columns `combined_mol_SMILES`, `A_smiles`, `B_smiles`, `C_smiles`, `expt_Hela`, and `expt_Raw`; SHA-256 `1b3dd460125ba7d70d8cce266bd5febaabe09eeddc6a4622405c2706420b9686`. It contains 20 heads, 12 linkers, and five tails. RDKit audit: 720 ring-bearing molecules, exactly one 5- or 6-member ring when present, no fused/bridged/spiro systems, mean 47.77 heavy atoms. The current official `data.zip` also exposes 12,276 candidate rows and 1,200 labeled rows. The paper's 60k Ugi virtual library is not present as structures: the public enumeration notebook requires an unreleased `AGILE_1200_SMILES_Compontents.csv`, so the exact 60k cannot be claimed as reconstructible from the repository alone. The notebook itself is fixed-core wildcard stitching: it recognizes `[NH2]`, `N#C`, and aldehyde `[CX3H1](=O)` handles, replaces them with `*`, and attaches the three components to a predeclared core. It is not a transferable atom-mapped Ugi reaction or a source of chemoselectivity rules. | Measured conditional/oracle benchmark; component-combination and held-component tests; restricted lipid smoke corpus. The 12,276 candidate structures can augment a Ugi-specific structural layer, but neither set is sufficient alone as the claimed general lipid pretraining corpus. | Freeze both repository artifacts; verify label transforms, replicate handling, assay direction, formulation constants, split policy, and whether the candidate set is a strict subset/filter of the reported 60k. Reconstruct and validate the Ugi reaction independently from primary products before using new substrates. |
| Bowen Li / Nature Materials set | 584 experimentally labeled lipids and a 40,000-member virtual screen | **Paper-scale verified; no million-scale lipid corpus was found in this work.** The library arises from one four-component reaction platform. | Independent measured benchmark or external validation if structures, endpoints, and formulation context can be reconciled; reaction-specific virtual coverage layer only. | Resolve the official structure release and license; audit assay units, formulation metadata, overlap with LNPDB/AGILE, and molecule/study IDs. |
| LNPDB | Heterogeneous public formulation/response records, including lung measurements | **Official MIT-licensed CSV downloaded and locally audited.** SHA-256 `3493f27306419facd0958589030ed37f272f05c81ad47dd7bc1b8ce0284a57b3`. It has 19,797 rows: 19,528 formulation records plus 269 commercial rows, 12,845 unique supplied lipid strings, 12,837 canonical structures, 43 experiment IDs/42 PMIDs, 17,140 in-vitro and 2,388 in-vivo rows. There are 1,975 lung rows but only 291 unique lipids across three studies; the in-vivo lung slice is 145 rows/88 unique lipids/two studies. All structures parse. | Primary measured structural anchor and study-aware formulation/response oracle data; target-tissue slices must remain study-aware and must never be flattened into independent molecule labels. | Freeze the downloaded version and original strings; finalize endpoint/unit/replicate transforms and prospective study/scaffold/linker holdouts. |
| LUMI-lab | Human-bronchial-epithelial-cell screen from a four-component reaction library | **Official CC-BY-4.0 Zenodo release located and directly inspected.** DOI `10.5281/zenodo.17771224`; `4CR-1920.csv` contains 1,920 measured rows with `RLU (log2)`, full molecular SMILES in `mol`, and a component-level `Markush code`. | Large pulmonary in-vitro oracle head, structure--component generalization tests, and a strong held-component/held-family benchmark; never flatten into an in-vivo lung label. | Freeze and hash the artifact; validate/canonicalize structures, decode component identities, audit replicates and overlap, and recover full formulation/assay metadata from the paper/source data. |
| COMET / LiON | Additional public formulation and delivery data | **Unverified leads; exact artifacts not supplied.** | Possible oracle training and external study holdouts. | Resolve exact works and downloads; distinguish measured records from virtual structures and model predictions; audit overlap with LNPDB. |
| LiGen structural corpus | Approximately 900k virtual structures in the old note | **Paper count verified; public artifact not found.** LiGen reports `LNP-Virtual900k`, 900,351 ionizable lipids generated by applying reported synthetic reactions to commercially available substrates, and `LNP-Exp12k`, 12,467 experimental structures drawn from public sources. | Prior-art design pattern and comparator. It is not an experimental efficacy corpus, and it cannot be an executable dependency unless an auditable release appears. | Obtain an official release/hash/license or reproduce a separately named reaction-grounded corpus; audit overlap and supported reaction chemistry. |
| LipidBERT/METiS structural corpus | Approximately 10M virtual structures in the old note | **Paper count verified; corpus is described as in-house/proprietary and no public release was found.** | Prior-art scale reference only, not a dependency for the MVP and not an oracle-label source. | Do not claim use without an official artifact and license. Reproduce coverage independently if needed. |
| Ouyang et al. approximately 20M screen | 892 tail candidates combined with 237 heads under a fixed three-tail pattern | **Scale and construction verified, but this is a different group from Bowen Li and is chemistry/template specific.** The paper fixes Tail 2 = Tail 3 and exhaustively combines blocks. The article releases source measurements, while the full generated-library predictions are available only on request. | Enumeration precedent and a source of coverage targets; not a downloadable 20M training corpus and not a general lipid prior. | Request the structures if strategically useful, or build a separately named corpus from licensed building blocks and fully disclosed reaction rules. |
| Synthesis-aware lipid generator corpus | Fixed head/tail catalogs and a virtual library | **Prior-art lead; no artifact supplied here.** | Comparator and possible synthesis-eligibility audit. Do not import its block closure as COMPOSE's support. | Exact paper/data/code; common-subset definition; whether reported molecules are enumerated or sampled; route representation and license. |
| LIPID MAPS / commercial reaction substrates | Structural fragments and lipid-like molecules | **Lead only.** | Optional chemistry coverage and external fragment/linker novelty reference. Not transfection labels. | Scope/license/version; ionizable-lipid eligibility; salts, stereochemistry, and provenance. |
| Our GEM synthesis-constrained diffusion study | Related Ugi-scaffold work from the same group | **Citation-only prior work; not a data, code, model, checkpoint, or preprocessing dependency.** | Prior-art/lineage disclosure only. COMPOSE-Lipid is trained and evaluated independently. | Self-cite accurately and record the no-reuse boundary; no artifact import or matched reproduction is required. |
| Internal novel-linker campaign | Promising linker and eventual prospective measurements | **Not supplied yet.** | Prospective linker-constrained transfer and in-vivo validation set; never used in retrospective test tuning after lock. | Structure definition, synthesis constraints, target product profile, blinded candidate policy, and preregistered prospective lock. |

Numbers labeled unverified above must not enter a manuscript table, model card,
or training command until the underlying artifact is present and hashed.

The LNPDB chemistry audit also fixes the immediate model boundary.  Heavy atom
count has median 51, mean 60.60, 95th percentile 105, and maximum 282; only
15.96% of canonical structures fit the current 40-atom kernel and 67.48% fit a
64-atom kernel.  The corpus contains S, P, and Si in addition to C/N/O/F,
formal charges, 384 atom stereotags, and 23,864 stereo-bond annotations.  Rings
occur in 44.75% of structures, but only 14.33% contain an aromatic ring.  The
current kernel therefore needs a declared larger-size, element, charge, and
stereochemistry policy before broad lipid training.  AGILE is not independent
of LNPDB: 1,180 of its 1,200 canonical structures (98.33%) occur in LNPDB and
must not be double-counted across train and evaluation.

## Virtual-corpus decision

The executable reaction registry, route-certificate schema, sampling policy,
and qualification gates are specified in
[`defensible_lipid_corpus_spec.md`](defensible_lipid_corpus_spec.md).

COMPOSE-Lipid should enumerate a reaction-grounded auxiliary corpus, but it
should **not** enumerate every head × linker × tail combination and then treat
the resulting Cartesian product as the data distribution.  Full cross-products
create extreme homolog/duplicate weighting, magnify enumeration choices into a
false molecular prior, and include combinations that are formally connectable
but chemically or synthetically implausible.

The staged corpus is:

1. a high-weight real anchor of deduplicated LNPDB/verified experimental
   structures, with formulation rows retained separately for oracle training;
2. a reproducible reaction-grounded virtual layer beginning with released
   AGILE structures plus independently versioned, licensed building-block
   catalogs and named reaction rules (the exact AGILE component input file is
   not public), then expanding across multiple reaction families; and
3. stratified sampling over reaction family, head, linker, tail architecture,
   tail length/unsaturation/branching, charge, and physicochemical envelopes,
   with caps and exact provenance for every row.

Start with a 50k–100k unique pilot.  Scale toward hundreds of thousands or one
million only after uniqueness, chemistry coverage, current-kernel eligibility,
and path/sampling throughput pass.  Hold complete components, reaction
families, and combinations out before training.  The virtual corpus teaches a
lipid-shaped distribution; it never defines COMPOSE's hard generative support.

## Region-role ablation

A head/linker/tail/other role embedding is a legitimate lightweight ablation,
not part of the core thesis.  Roles must be recomputed from the current valid
molecule, must not encode target information, and symmetry-equivalent tails
share one `tail` role rather than `tail-1`/`tail-2` identities.  Retain it only
if it improves held-linker/held-combination coverage and architecture fidelity
under matched parameters and compute.  For the prospective novel-linker task,
a protected-atom/linker constraint is more direct than relying on a region
embedding alone.

## Four data layers (kept separate)

1. **Structural pretraining corpus:** molecule structures used to learn lipid
   chemistry.  Virtual structures may live here, but they carry no delivery
   label merely because they are lipid-shaped.
2. **Measured molecule records:** experimentally tested ionizable lipid
   identities and molecule-level endpoints, when a molecule-level endpoint is
   genuinely defined.
3. **Formulation-response records:** lipid plus helper lipids, ratios, cargo,
   dose, route, study, tissue/cell, time point, and assay.  These rows train the
   delivery oracle and cannot be collapsed to SMILES alone without confounding.
4. **Prospective locked set:** model-proposed candidates, synthesis/formulation
   records, and in-vivo outcomes collected after the candidate-selection lock.

## What the lipid generator must prove

### Base-generator evidence

- 100% committed-state validity and connectivity at lipid graph sizes;
- fidelity of atom, bond, formal-charge/protonation, size, tail count/length,
  branching, unsaturation, head, linker, and degradable-motif distributions;
- uniqueness, exact novelty, nearest-neighbor memorization curves, scaffold and
  linker novelty, and coverage/precision-recall—not Tanimoto novelty alone;
- practical source-to-target reachability and measured compilation/sampling
  throughput on molecules substantially larger than the current 40-slot model;
- stereochemistry policy stated explicitly; the present graph representation
  does not encode tetrahedral or E/Z stereochemistry.

### "Different combinatorial chemistries" evidence

This should be demonstrated, not asserted, through complementary splits:

1. **Held combination:** head, linker, and tail components are individually
   seen, but selected head-linker-tail combinations are withheld.
2. **Held component/scaffold:** withhold complete head or linker scaffolds to
   test structural transfer rather than memorized recombination.
3. **Held reaction family:** where reaction annotations exist, test outside a
   Ugi-only subset while preserving synthesis review.
4. **Flexible topology/size:** compare with independent released generators
   and matched enumeration controls where their support is genuinely
   comparable, then show additional lipid architectures reachable only by the
   RGM editor. GEM is citation-only and is not imported or reproduced.
5. **Novel-linker constraint:** condition on the prospective linker and measure
   diversity of the surrounding molecular structures without allowing the
   linker itself to leak into retrospective model selection.

H/L/T annotations enable these tests; they do not define the generator's hard
support.

### Conditional and oracle evidence

- Begin with the shared matched QED authorization test on the small-molecule
  model to qualify the guidance/recovery machinery cheaply.
- For lipids, condition or optimize on calibrated properties only under a
  declared oracle domain of applicability.  Molecular pKa and apparent LNP pKa
  are not interchangeable endpoints.
- Train delivery oracles with study-, scaffold-, linker-, reaction-family-, and
  temporal holdouts as available; report calibration, uncertainty, abstention,
  and performance versus distance from training chemistry.
- Compare unconditional selection/filtering and valid-rewrite guidance or
  reward fine-tuning under the same candidate and oracle-call budgets.
- Lock a diverse prospective set before synthesis; retain every attempted
  candidate in the hit-rate denominator.

## Immediate work once artifacts arrive

1. Copy each raw artifact into immutable storage and record source URL, access
   date, license, size, and SHA-256.
2. Profile schemas before making any molecule/formulation join; record missing
   fields and endpoint transforms.
3. Canonicalize under versioned salt, tautomer, protonation, charge, isotope,
   and stereochemistry policies while retaining the original strings.
4. Deduplicate at molecule, formulation, and study levels and generate a full
   overlap/leakage matrix across sources.
5. Produce chemistry-support tables: elements, charges, heavy atoms, rings,
   stereocenters, tail/linker/head annotations, and exact current-kernel
   eligibility/reachability.
6. Freeze structural, measured, formulation, and prospective manifests before
   launching any corpus-scale path/support build.

## Information still needed from the user

- Exact files or links for the intended broad structural corpus.
- Exact 584-lipid, COMET/LiON, and LNPDB artifacts to use.
- Whether AGILE's raw HeLa/RAW labels are already transformed and which endpoint
  direction constitutes improvement.
- The novel linker definition and allowed surrounding chemistry.
- Target tissue/cell, cargo, route, formulation policy, dose, primary endpoint,
  and minimum prospective hit definition.
