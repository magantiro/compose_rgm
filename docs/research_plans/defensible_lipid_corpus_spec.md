# Defensible COMPOSE-Lipid structural corpus

**Status:** executable design specification.  This does not change the COMPOSE
rewrite/generator-matching thesis and does not turn the model into a fragment
assembler.  It defines how to obtain a broad, auditable lipid training
distribution when the largest prior-art virtual corpora are not public.

## Decision

Build a heterogeneous, reaction-grounded structural corpus anchored to the
observed chemistry of LNPDB.  Do not use the raw size of a Cartesian product as
evidence of chemical coverage.  Each virtual product must carry a reproducible
route certificate, and no reaction family may dominate merely because it has
more compatible substrate combinations.

The first qualified release should contain 50,000--100,000 unique products.  A
500,000--1,000,000 product release is authorized only after the pilot passes
the chemistry, coverage, leakage, and COMPOSE-throughput gates below.  Ten
million structures are not intrinsically more defensible than one million
well-stratified structures.

## What the prior-art audit establishes

### LiGen

LiGen reports 900,351 structures in `LNP-Virtual900k`, described only as the
result of applying publicly reported synthetic reactions to commercially
available substrates using established in-silico enumeration protocols.  A
final search of the ACL Anthology record, paper, GitHub, and Hugging Face found
no official dataset or code release.  The paper does not identify the reaction
set, reaction transforms, substrate vendors/catalog versions, substrate lists,
product filters, deduplication policy, splits, or artifact license.  Its exact
corpus is therefore neither accessible nor independently reproducible from the
paper.  `LNP-Exp12k` is assembled from LiON and AGILE rather than providing an
independent million-scale experimental corpus.

### AGILE, Bowen Li, and LipidBERT

- AGILE's lipid-specific pretraining library is approximately 60,000 Ugi-derived
  structures.  The official repository releases 12,276 candidate rows and
  1,200 labeled structures plus an enumeration notebook, but the notebook's
  exact component input CSV is absent.  The paper's “millions” refers to the
  general MolCLR small-molecule warm start, not a million-lipid corpus.
- Bowen Li's Nature Materials work uses 584 measured lipids and a 40,000-member
  virtual screen from one four-component reaction platform.
- LipidBERT describes a 10-million-structure METiS library as in-house, produced
  by proprietary fragment/RL/generative workflows.  It is a scale precedent,
  not an available training dependency.
- A separate Ouyang-group study exhaustively combines 892 tails with 237 heads
  under a fixed three-tail architecture, constraining Tail 2 = Tail 3, to
  screen nearly 20 million structures.  The full virtual prediction library is
  available only on request and remains template-specific.

## LNPDB as the empirical anchor

The official LNPDB table contains 19,797 rows, 19,528 formulation records, and
12,845 unique supplied ionizable-lipid strings across 42 publications.  It
also provides explicit head, linker, and up-to-four-tail fields.  A current
molecule-level audit finds:

- 10,052/12,845 (78.3%) structures with a head annotation and 434 unique head
  strings;
- 2,833/12,845 (22.1%) with a linker annotation and 35 unique linkers;
- 9,601/12,845 (74.7%) with tail-1 and 2,451/12,845 (19.1%) with tail-2,
  yielding 267 unique tail strings across the tail fields;
- 9,177 distinct ordered annotated head/linker/tail tuples;
- 76.3% ester-bearing, 5.7% carbonate-bearing, and 2.9% disulfide-bearing
  full structures;
- 6,762 acyclic structures and 6,083 ring-bearing structures; and
- substantial study imbalance: the largest individual studies contribute
  hundreds to approximately 1,500 unique structures.

These annotations are not automatically interchangeable fragments.  Some are
reactants or publication-specific building blocks, and some contain attachment
placeholders.  They must be interpreted within the source publication and a
named reaction family before recombination.  LNPDB supplies observed support
and target marginals; it is not a license for a blind all-by-all cross-product.

## Corpus layers

### R0: observed real structures

Include every deduplicated, parseable LNPDB ionizable lipid under a frozen
canonicalization policy.  Retain original strings, protonated strings,
publication/study IDs, component annotations, commercial status, and all
formulation rows separately.  Structural pretraining uses one molecule once or
a declared study-balanced weight; oracle training continues to use the
study-aware formulation table.

Add other released real structures only after exact overlap analysis.  AGILE
is already 98.3% contained in LNPDB and must not be double-counted as an
independent distribution.

### R1: high-throughput combinatorial reaction families

Implement named, literature-grounded transformations independently.  The
initial reaction registry should cover complementary architectures rather than
minor variants of one scaffold:

1. amine opening of alkyl or alkenyl epoxides (amino-alcohol/lipidoid
   libraries);
2. aza-Michael addition of amines to alkyl acrylates (degradable
   amino-ester lipidoids);
3. isocyanide-mediated Ugi 3-component chemistry;
4. conventional Ugi 4-component chemistry;
5. Passerini 3-component chemistry for biodegradable asymmetric lipids;
6. amine--aldehyde--alkyne (`A3`) coupling for propargylamine lipids;
7. amine reaction with alkylated dioxaphospholane oxides for ionizable
   phospholipids (`iPhos`);
8. epoxide opening followed by acyl-chloride esterification for degradable
   branched four-tail lipidoids;
9. the published ricinoleic-acrylate/alcohol carbonate platform used for
   pulmonary lipid libraries; and
10. a separately validated set of modular esterification, amidation,
    carbamate/carbonate, disulfide, and reductive-amination routes needed to
    cover rationally designed clinical-like and degradable architectures.

Items 1--9 have explicit lipid-library precedents.  Item 10 must not be treated
as one generic reaction: each transform requires its own literature recipe,
reactive-handle constraints, and product validation.

### R2: rational-design neighborhoods

Enumerate small, provenance-preserving neighborhoods around observed LNPDB
heads, linkers, and tails only through an executable route in R1.  Vary:

- carbon-chain length, unsaturation, and branching;
- symmetric versus asymmetric tails and tail count;
- ester/carbonate/carbamate/disulfide placement;
- cyclic, acyclic, and heterocyclic ionizable heads;
- linker length, rigidity, and degradability; and
- observed P-, S-, halogen-, charge-, and stereochemistry-bearing classes.

This layer captures rational design without defining validity as “looks like a
lipid.”  A proposed analog without a route certificate is held out as a model
proposal, not inserted into the training corpus.

## Lessons from prior enumeration programs

The useful prior-art principles are separable from the exact private datasets:

- **AGILE:** vary head chemistry, two distinct tails, chain length (reported
  C6--C26), ester presence, and ester position, then narrow the screening set
  using tertiary-amine, tail-length, and reagent-availability rules.  Keep its
  explicit structural axes and synthesis logic; do not inherit its Ugi-only
  support.  The released notebook is not an atom-mapped Ugi reaction
  enumerator: it finds `[NH2]`, `N#C`, and aldehyde `[CX3H1](=O)` handles,
  replaces each with a wildcard, and attaches those wildcard-bearing fragments
  to a fixed core with RDKit's molecular enumerator.  That is adequate for its
  fixed library but does not supply general reaction SMARTS, selectivity, or
  incompatibility semantics.  COMPOSE-Lipid must validate the underlying Ugi
  transform against reported products instead of treating this stitching code
  as a transferable reaction rule.
- **Bowen Li pulmonary library:** use a factorial, experimentally tractable
  head-by-tail design (72 heads by 10 tails) around a fixed reaction platform.
  Keep the clean design-of-experiments logic for measured subsets; do not treat
  one fixed linker chemistry as the global lipid distribution.
- **Whitehead degradable lipidoids:** choose tail lengths using prior efficacy
  and solubility evidence before enumerating 280 amines by five acrylates.
  Keep the physicochemical feasibility envelope and transparent full-factorial
  counts.
- **Epoxide-derived lipidoids and iPhos:** encode reactive-site multiplicity and
  stoichiometry explicitly, because the number of reactive amines controls tail
  count and product architecture.  Do not generate products by joining fragments
  without reaction-valence semantics.
- **Passerini and A3 libraries:** use complementary multicomponent reactions to
  create asymmetric and degradable architectures inaccessible to a Ugi-only
  library.  Reaction-family diversity is more valuable than millions of near
  homologs from one family.
- **Ouyang approximately 20M screen:** select building blocks using empirical
  structure--activity evidence, but recognize that exhaustive permutation and
  fixed Tail 2 = Tail 3 amplify the chosen template.  Keep evidence-driven block
  selection; reject the resulting product-count distribution as a training
  prior.
- **LipidBERT/METiS:** iterative generative expansion can fill gaps after an
  initial corpus exists, but repeatedly training on model-generated structures
  risks self-reinforcing model bias.  Any later COMPOSE-generated augmentation
  must remain a separately labeled layer and pass independent route and coverage
  qualification.
- **LiGen:** commercially available substrates plus reported synthetic reactions
  is the right high-level constraint.  Our improvement is to publish the exact
  reaction registry, catalog snapshots, transforms, failures, selection weights,
  and splits that LiGen does not disclose.

Together these imply a two-axis corpus design: **reaction diversity** controls
what is actually synthesizable, while **distribution design** controls what the
model learns.  Neither axis can substitute for the other.

## Statistical design of the selected corpus

Let the candidate pool be partitioned by reaction family and architecture, and
let `p_real` denote the study-balanced empirical distribution of qualified
LNPDB structures over declared features.  The selected virtual distribution
must be a declared mixture of three strata:

1. **realism stratum:** products selected to cover and softly match `p_real`;
2. **coverage stratum:** products selected to guarantee minimum mass for every
   qualified reaction family, scaffold, and rare architecture; and
3. **exploration stratum:** route-certified products near, but not far outside,
   the observed physicochemical boundary.

The mixture weights are selected on the pilot and ablated; they are not inferred
from the number of available combinations.  A practical deterministic selector
can optimize feature-space coverage/optimal-transport distance subject to:

- lower and upper quotas per reaction family and architecture;
- caps per head, linker, tail, source publication, and near-duplicate cluster;
- minimum synthesis-confidence and maximum structural-liability thresholds;
- exact uniqueness and split-exclusion constraints; and
- positive inclusion probability for every qualified rare stratum.

Store each product's inclusion probability.  Training can then use explicit
mixture weights or inverse-selection corrections when estimating a desired
target distribution.  This prevents a large family or repeated building block
from becoming the model prior merely because it is combinatorially prolific.

The real LNPDB structures remain a separately weighted anchor throughout.  The
virtual layer teaches support and structured variation; it does not overwrite
the empirical molecular distribution or create biological labels.

## Building-block registry

Every substrate record must contain:

- canonical and original SMILES, stereochemistry, formal charge, salt policy,
  reactive-handle type and atom map;
- source artifact, vendor/catalog and snapshot date where applicable, license
  or permitted-use status, and stable source identifier;
- observed LNPDB/AGILE publication and component role, if any;
- calculated size, element, ring, tail-length, unsaturation, branching, and
  physicochemical descriptors; and
- compatible reaction IDs, with incompatibility reasons recorded rather than
  silently dropped.

Start from the audited LNPDB component strings and released AGILE components.
Augment them only with a frozen, legally usable catalog of commercially
available substrates.  The exact catalog snapshot is part of the dataset
identity; “commercially available” without a source/version is insufficient.

## Coverage sources beyond LNPDB

LNPDB is the real ionizable-lipid anchor, not the only source of structural
knowledge.  Complement it with clearly separated sources whose roles do not
leak into one another:

- **LIPID MAPS LMSD:** a downloadable CC BY 4.0 set of approximately 50,000
  curated and computational biologically relevant lipids.  Use its fatty-acyl,
  glycerolipid, phospholipid, sphingolipid, and sterol motifs to audit tail,
  branching, unsaturation, stereochemistry, and natural-lipid coverage.  It is
  not an ionizable-lipid target distribution and should enter, if at all, as a
  separately labeled representation/support layer.
- **SwissLipids/LMISSD:** large natural and computational lipid spaces useful
  for coverage reference and motif discovery.  Exact-structure versus
  species/sum-composition records must be separated; unresolved compositions
  cannot train a molecular graph generator.
- **clinical and commercial ionizable lipids:** retain an explicit reference
  panel (including commercial LNPDB rows) to ensure that common clinical-like
  architectures are covered despite their low frequency in combinatorial
  studies.
- **patent chemistry:** use public patent families to recover underrepresented
  clinical-like heads, linkers, tails, and routes.  Patent-extracted structures
  require document/claim provenance, structure-quality review, deduplication,
  and a declared license; automated extraction alone is not ground truth.
- **Open Reaction Database and source-paper supplements:** use open, structured
  reaction records to validate transforms, conditions, yields, and substrate
  compatibility.  ORD is CC BY-SA and downloadable, but only lipid-relevant
  reactions with verified product mapping enter the registry.
- **versioned purchasable-substrate catalogs:** supply building blocks after
  reactive-handle and availability checks.  These define a synthesis search
  space, not a molecular frequency prior.

General small-molecule corpora may warm-start atom/bond representations, as in
AGILE's MolCLR initialization, but they do not count as lipid structures and
must not be used to claim lipid-chemistry coverage.

Coverage is measured against a union of reference panels rather than a single
database: real ionizable lipids (LNPDB), clinical/commercial structures,
natural-lipid motifs, patent chemotypes, reaction families, and purchasable
building blocks.  Report recall separately for every panel so strong coverage
of one cannot hide collapse on another.

## Reaction-registry schema

Every reaction entry is versioned and reviewed independently.  At minimum it
contains:

The executable JSON Schema lives at
`configs/lipid_reactions/reaction_registry.schema.json`.  It makes a mapped
transform, positive and negative chemistry examples, executable tests, and an
artifact hash mandatory before a reaction can be marked
`qualified_for_enumeration`; the enumerator must fail closed on all earlier
statuses.

```text
reaction_id, reaction_version, status
source DOI/patent and exact scheme/example
ordered reactant roles
reactant-handle SMARTS with mapped reactive atoms
atom-mapped reaction SMARTS
stoichiometry and allowed reactive-site multiplicity
regio-/chemo-/stereoselectivity rule
required and forbidden functional-group conditions
reported conditions, yield range, purification assumptions
known-positive products used for validation
known-negative/incompatible substrate tests
enumerator implementation and unit-test hash
```

`status` progresses from `literature_spec` to `validated_transform` and then
`qualified_for_enumeration`.  A name such as “Ugi” or “Michael addition” is not
enough to qualify an entry.  The atom-mapped SMARTS must reproduce known paper
products exactly, reject deliberately incompatible reactants, preserve charge
and stereochemistry according to policy, and produce the expected number of
products.  Reaction-specific tests compare atom mapping and canonical products
against source examples before any large enumeration begins.

## Route certificate and product construction

Each enumerated product receives:

```text
product_id
reaction_id + reaction_version
publication/DOI supporting the transform
ordered substrate IDs and source catalogs
atom-mapped product and canonical product SMILES
stereochemical and protonation policy
sanitization/valence result
expected product multiplicity and selected regio-/stereoisomer
structural filters with pass/fail reasons
enumerator software/container versions
```

Products are created with atom-mapped reaction transforms and must pass RDKit
sanitization, exact valence/charge checks, single-product connectivity, and the
reaction-specific structural certificate.  Representative products from every
reaction family undergo manual chemistry review, and a stratified subset should
be checked with an independent retrosynthesis or expert route review before the
corpus is called synthesis-grounded.

## Sampling the virtual product space

Never materialize every compatible cross-product merely to train on it.  Use a
deterministic streaming enumerator, canonicalize online, and select a balanced
release with constrained sampling over:

- reaction family and source publication;
- head, linker, tail scaffold and component frequency;
- one/two/three/four-tail architecture and symmetric/asymmetric tails;
- size, charge, element set, ring/aromaticity, rotatable bonds, branching,
  unsaturation, degradable motifs, and physicochemical envelope; and
- distance to the real LNPDB manifold.

The target is a mixture: broad but positive coverage for each qualified
reaction family, plus soft matching to real LNPDB marginals.  No family is
weighted in direct proportion to the number of possible substrate tuples.
Preserve the raw enumerated counts separately so the selection distribution is
fully reconstructible.

## Frozen evaluation splits

Create all splits before model training:

1. **random molecule split** only as a low-bar diagnostic;
2. **study/publication split** on real LNPDB structures and formulation rows;
3. **held-combination split** where components are seen but selected
   head/linker/tail combinations are absent;
4. **held-component split** for complete head, linker, and tail scaffolds;
5. **held-reaction-family split** to test whether COMPOSE can leave the
   enumerated templates rather than memorize one chemistry;
6. **nearest-neighbor/scaffold exclusion split** preventing near duplicates
   across virtual and real partitions; and
7. **prospective novel-linker lock** created before any candidate ranking.

The full virtual corpus trains only the structural generator.  Biological
delivery labels remain attached to study-aware measured formulation records;
virtual products never inherit a delivery label from their components.

## Qualification gates

The 50k--100k pilot must pass all of the following before scaling:

- 100% parseable, single-component products after the declared salt policy;
- no duplicate product IDs or cross-split structure leakage;
- route certificates and exact regeneration tests for every row;
- reaction-family and component coverage reported both before and after
  selection;
- precision/recall and nearest-neighbor coverage against LNPDB, stratified by
  study and architecture rather than only a global embedding plot;
- plausible, non-collapsed size/charge/tail/ring/degradability marginals;
- expert review of a stratified product panel including rare families;
- exact COMPOSE source-to-target reachability on the eligible slice; and
- measured path compilation, training, and ancestral-sampling throughput at
  lipid graph sizes.

The paper should report the complete reaction registry, block-source manifest,
raw and selected product counts, sampling weights, failure counts, hashes, and
split manifests.  This is more defensible than an unreleased “900k” corpus even
if the first release is smaller.

## Immediate implementation order

1. Freeze LNPDB and AGILE raw artifacts with hashes and licenses.
2. Normalize the LNPDB component/attachment conventions without discarding the
   originals; map each annotated architecture back to its source study.
3. Implement and unit-test three high-yield pilot families first: epoxide
   opening, aza-Michael addition, and Ugi chemistry.
4. Add Passerini, A3, iPhos, branched epoxide/acylation, and the carbonate
   platform as independently tested registry entries.
5. Enumerate a 50k--100k stratified pilot and publish its audit card before any
   corpus-scale COMPOSE path build.
6. Expand toward 500k--1M only when a coverage analysis identifies a concrete
   missing region that additional enumeration resolves.
