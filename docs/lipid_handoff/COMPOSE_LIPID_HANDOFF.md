# COMPOSE → COMPOSE-Lipid: complete change handoff (for figure creation)

## 0. One-paragraph orientation
**COMPOSE / RGM (Rewrite Generator Matching)** is a generative model that builds a
molecule as a **continuous-time Markov chain (CTMC) over molecular graphs**: starting
from a *source-prior* graph, a learned **factorized rate model** predicts the rate of
each graph **rewrite** (atom insert/delete/restate, bond reorder/reroute, ring ops…),
and the trajectory of rewrites edits the graph into a finished molecule. It is trained
by matching model rewrite-rates to *teacher programs* (a Generator-Matching / Poisson–
Bregman objective). The rewrite families live in a registry (`MARK_RULE_NAMES`).
In COMPOSE's **carbon-tree architecture**, the source prior supplies the *carbon
skeleton* and rewrites only **decorate** it (turn carbons into heteroatoms, add
double bonds, attach rings). COMPOSE-Lipid = COMPOSE specialized to make **ionizable
lipids** (head + linker + tails) without breaking the shared RGM core.

---

## 1. What we changed in the GENERATOR / MODEL to make lipids work
Each item lists the *why*, the *change*, and the *file*.

### 1A. Lipid-shaped source prior  ← the single most important change
- **Problem:** de-novo tails came out structureless ("heads look like tails"). Root
  cause: in the carbon-tree architecture the **tail structure comes from the source
  prior, not from rewrites**, and the generic prior (`DegreeBoundedCarbonTreePrior`)
  samples uniform-ish degree-bounded random trees — which are *not* lipid-shaped.
- **Change:** `LipidCarbonTreePrior` (subclass) samples a **lipid topology**: a
  compact **hub** (head + linker-adjacent carbons) with **2–4 long linear tails**
  (tail length 6–18 C). Still a valid connected degree-≤4 carbon tree, so it remains a
  legitimate CTMC source prior and passes the `flexible_size_graft` type check.
- **Effect (measured):** longest-tail median 16 C (vs 4 generic); transport programs
  shorter (~14 vs ~26 steps); training GM-loss dramatically lower.
- **File:** `src/compose_v4/chem/lipid_source_prior.py`; wired via
  `--source-prior lipid_carbon_tree` in `scripts/train_tracelet_cnof_gate.py`.

### 1B. Region-aware rate model  ← "treat head / linker / tail differently"
- **Change:** `RegionAwareFactorizedTraceletRateModel` adds two lipid-native inductive
  biases on top of the frozen shared core (via two behavior-preserving hooks, no core
  edits):
  1. a **per-atom region embedding** {head, linker, tail, other} in the encoder;
  2. a **region-conditioned base rate for atom insertion**, so the element the model
     adds depends on region (**tails→carbon, linkers→O/N, heads→N**) — this corrects
     the global element prior that otherwise scatters heteroatoms into the tails.
- Region is a **deterministic function of the current partial graph**
  (`lipid_region_labels`), so it works identically at training and generation and adds
  **no new sampling degrees of freedom**. Embedding is **zero-initialized** → an
  untrained model is bit-for-bit the base model; region signal is learned as a residual.
- **Files:** `src/compose_v4/lipids/region_aware_rate_model.py`,
  `src/compose_v4/lipids/region_labels.py`,
  `src/compose_v4/lipids/head_region.py` (lightweight head detector, no oracle deps).
- **Input artifact:** `region_conditioned_prior_v1.json` (element-insertion prior
  conditioned on region, built from the corpus).

### 1C. BEAE linker as a frozen, "labeled-after-the-fact" scaffold
- **Why:** the target chemotype (BEAE = **branched E-enamine ester**) has a specific
  linker that forms *in situ* from an aza-Michael reaction; it is not built atom-by-atom
  by the generator but **labeled** onto the molecule.
- **Change:** `beae_linker.py` provides `is_beae`, `beae_decompose`,
  `beae_linker_atoms` — an **invariant frozen linker scaffold**; during generation the
  linker atoms are frozen and the enamine **N is treated as LINKER, not head**
  (head-agnostic region labeling + `basic_nitrogens` fix).
- **File:** `src/compose_v4/lipids/beae_linker.py`.

### 1D. Generation-time purity constraint (NO retrain)
- **Change:** a sampler wrapper that thins rewrites which (a) create a nitrogen beyond
  a **budget (N≤3)** or (b) restate a **deep tail carbon** (degree-2 carbon flanked by
  carbons, no heteroatom within 1 bond) into a heteroatom.
- **Effect (measured):** tail-region fraction ↑ to ~76%, amines/molecule ↓ to ~1.4
  (from ~69% / ~2.1 unconstrained) — clean tails without a retrain.

### 1E. AlkylGraft chain-rewrite family (built + tested, currently DORMANT — state honestly)
- **Change:** added an `alkyl_graft` rewrite family (registry `MARK_RULE_NAMES` 10→11):
  operator (`AlkylGraft`/`AlkylPrune`), teacher-program collapse of linear alkyl runs
  into grafts (verified graph-identical), and a factorized rate-model head mirroring
  `cycle_attach` (templates = chain lengths). Samples + applies; 15 model tests green.
- **Status:** on the *carbon-tree* architecture the skeleton already comes from the
  prior, so **0 grafts fire on lipid training traces** (`family_accuracy_alkyl_graft =
  0.0`). It is a clean, tested primitive kept for the null-source generation mode, but
  it is **not** on the critical path for the current lipid model. Do not claim it drives
  tail generation — the source prior (1A) does.
- **Files:** `src/compose_v4/rewrite/alkyl_graft.py`, `alkyl_teacher.py`;
  `src/compose_v4/model/factorized_tracelet_rate_model.py`.

### 1F. Infra (not modeling, but enabled the runs)
- **Distributed crash-safe path compile:** shard the (slow, CPU-bound) transport/path
  precompute across N containers (`--compile-shard-count/index`,
  `--compile-train-shards-only`), incremental volume commits, then a finalize pass
  resumes shards — verified byte-identical to single-container.
- **Zero-slowdown live peeks:** periodic checkpoint commits + off-GPU
  `sample_checkpoint`/`peek` that samples from the interim best checkpoint using the
  lipid prior — mid-training quality eval (validity / region balance / linker chemistry /
  physchem).
- **Files:** `scripts/train_tracelet_cnof_gate.py`, `modal_apps/lipid_smoke.py`.

---

## 2. The CORPUS — exactly what we did
- **Base:** a **reaction-grounded R1 corpus** of lipids
  (`r1_reaction_grounded_corpus_v1.csv`) — molecules tagged by the reaction family that
  makes them, not random molecules.
- **CNOF filter:** keep only molecules whose atoms are in {C,H,N,O,F}. (S/P-containing
  families become empty after this filter — expected.)
- **Michael skew (deliberate design choice):** the target chemistry is the **aza-Michael
  amine+acrylate** reaction, so we made that family the **plurality** of the corpus —
  raised aza-Michael from **3.6% → ~38.4%** of the mix — while **keeping all 9 CNOF
  reaction families** (a diversity constraint: the corpus must span ≥6 families; the
  trained model reports `represented_families = 6`). The remaining budget is
  **water-filled** across the other families (small families contribute all they have,
  the shortfall flows up). Michael unique structures cap ~16k (loader dedups), so past
  ~42k total the Michael *share* necessarily drops — by design, not a bug.
  - **Builder:** `scripts/build_michael_skewed_corpus.py`
  - **Outputs:** `generator_corpus_michael_v1.smiles`,
    `generator_corpus_michael_big_v1.smiles` (80k variant),
    `generator_corpus_cnof_v1.smiles`.
- **Region prior input:** `region_conditioned_prior_v1.json` — a region-conditioned
  element-insertion prior computed from the corpus (feeds change 1B).
- **BEAE fine-tune substrate (downstream, separate from the general corpus):** a
  **combinatorial enumeration head × acrylate-tail × propiolate-tail**
  (`beae_substrate_enumeration_v1.json`), **qualified** to reconstruct the measured lead
  lipids (RM-60, Example-2) via the propiolate aza-Michael route. Design principle:
  corpus stays broad/general; the Michael-BEAE linker is a **downstream fine-tune**, not
  the pretraining target.
- **Corpus provenance artifacts:** `corpus_coverage_card.json`,
  `reaction_families_qualification.json` (`all_qualified: true`),
  `canonical_structure_union.csv`, `r0_release_manifest.json`.

Manifest dir: `artifacts/datasets/compose_lipid_pretraining_v1/`.

---

## 3. The ORACLE — exactly what we did (and its honest limits)
- **What it is:** `pan_lung_filtering_v1` — a **filtering-only** potency oracle over
  **three domains**:
  | domain | what | train rows | unique structures | ensemble |
  |---|---|---|---|---|
  | `a549` | A549 lung adenocarcinoma transfection potency | 1801 | 176 | 1 |
  | `lumi` | luminescence potency | 1920 | 1920 | 3 |
  | `lut_selectivity` | lung-vs-liver (LUT) selectivity SAR | 444 | — | 2 |
  - **Representation:** `morgan_descriptors_plus_typed_context_v1` (Morgan fingerprint +
    physchem descriptors + typed structural context).
- **Applicability domain (AD) = a dual gate; BOTH must pass to be "admitted":**
  1. **Similarity:** max Morgan-Tanimoto to any training structure ≥ threshold
     (threshold = **5th percentile of leave-one-out max-Tanimoto** on the training set).
  2. **Descriptor distance:** robust RMS descriptor distance ≤ threshold
     (threshold = **99.5th percentile** of training robust distance).
- **Scoring:** per candidate the ensemble returns `ensemble_mean` (calibrated predicted
  potency), `total_uncertainty` (conformal radius + ensemble disagreement), and
  `pessimistic_score = mean − radius − disagreement`.
- **Governance:** `status = qualified_for_filtering_only`; **reward fine-tuning is
  DISABLED** (blocked pending: clean-runtime verification, frozen preregistration bound
  to the manifest hash, pre-registered AD thresholds + candidate budgets, a hard reward
  guard in the optimization path, and prospective score-hacking / drift monitoring).
- **KEY FINDING for BEAE — state carefully, do NOT over-claim novelty:** BEAE is **a
  Michael-addition product** (aza-Michael of the amine onto a **propiolate/alkyne
  acceptor**, giving the E-enamine ester). Michael/acrylate chemistry is **one of the
  combinatorial chemistries the oracle's ionizable-lipid training data is built from** —
  BEAE is a close variant (acrylate→propiolate swap), **not an alien chemotype.** What
  the AD actually reports is asymmetric: the **descriptor-distance gate PASSES** (BEAE is
  a normal lipid by physchem / size / typed context), and only the **Morgan-fingerprint
  similarity gate trips** (max training Tanimoto ≈ **0.45**, just under the conservative
  5th-percentile threshold) — a fingerprint artifact of the enamine C=C + branched ester,
  not evidence the model is blind to the chemistry. Because admission requires **both**
  gates, BEAE is formally "not admitted," so we don't quote it as a validated potency —
  but the right framing is **near-domain extrapolation on familiar Michael chemistry**,
  not blind abstention on a novel chemotype. We rank by `ensemble_mean` and treat the
  top-K as an **active-learning / confirm-in-wet-lab batch**.
- **Files:** `src/compose_v4/oracles/pan_lung_filtering.py`;
  manifest `artifacts/oracles/pan_lung_filtering_v1/manifest.json`.

---

## 3B. How candidates are RANKED — the composite (the oracle is only ONE term)
The BEAE candidate ranking is a **3-term composite**, NOT the oracle alone:

    composite  =  design_rule_score   +   3 × predicted_potency   +   1.5 × lead_similarity
                     (reliable)               (THE ORACLE, §3)           (Tanimoto to leads)
                                             provisional / near-domain

- **design_rule_score** — a transparent ionizable-lipid rule score (ionizable amine head
  present; two hydrophobic tails ~8–18 C; MW 550–1000; logP in range; penalize aromatic
  or polar tails, single tail, too-short tails). This term is **reliable** and is what
  actually separates good lipids from junk.
- **predicted_potency** — the pan-lung oracle `ensemble_mean` (§3). On BEAE this is
  **provisional / near-domain**, and — critically — **on its own it does NOT reject junk**:
  e.g. a PEG-tailed decoy receives the oracle's *highest* predicted potency (+0.81) while
  being chemically absurd. So the oracle can only **re-order within the already-good set**;
  it is not a quality gate.
- **lead_similarity** — Morgan-Tanimoto to the qualified leads (RM-60, Example-2); a mild
  pull toward validated chemistry.

This composite produces the **quality spectrum** in the figure (REALLY-GOOD / GOOD /
MARGINAL / POOR): the **design term makes the good-vs-poor boundary**, the **oracle orders
within the good set**. The two qualified leads land in **GOOD**; several generated
candidates score higher — a **wet-lab hypothesis, not proven** (the top tier is enriched
for a deapa-head / C18:1-tail motif, so it may be the oracle rewarding one feature).
- **Files (in `docs/lipid_handoff/`):** `beae_rank_composite.py` (composite + design
  score), `rank_beae.py` (oracle wrapper), `beae_png.py` (figure); ranked output
  `beae_candidates_ranked.json`; spectrum figure `beae_spectrum_figure.png`.

---

## 4. Suggested figure structure (COMPOSE vs COMPOSE-Lipid)
- **Panel A — architecture delta:** generic carbon-tree prior (messy tails) → lipid
  carbon-tree prior (hub + long tails). Show a generic random tree vs a lipid skeleton.
- **Panel B — region-aware decoration:** color atoms by region {head/linker/tail};
  arrows showing region-conditioned element choice (tail→C, linker→O/N, head→N).
- **Panel C — corpus:** stacked bar of the 9 CNOF families, aza-Michael 3.6%→38.4%.
- **Panel D — BEAE assembly:** frozen E-enamine-ester linker + head + two generator
  tails (head/linker/tail treated differently).
- **Panel E — oracle:** AD dual-gate schematic; BEAE (a Michael/propiolate product,
  same reaction family as the training lipids) sits **inside the descriptor gate** but
  just **outside the fingerprint-similarity gate** (Tanimoto ~0.45) → near-domain
  extrapolation → ranked as an active-learning batch. Do not draw it as "far outside /
  novel chemotype."

## 5. Honesty guardrails for whoever writes the paper/figure
- Tail *structure* diversity comes from the **source prior (1A)**, not from AlkylGraft
  (1E is dormant) and not from deterministic tail sampling.
- The **purity fix (1D) is a generation-time constraint, not a retrain.**
- The candidate **ranking is a 3-term composite (§3B), not "the oracle."** The oracle
  alone even scores some junk highly (PEG decoy +0.81); the **design-rule term** does the
  good-vs-junk separation, the oracle only re-orders within the good set.
- On the oracle + BEAE, **don't over-claim novelty**: BEAE is a **Michael-addition
  product** (propiolate acceptor), the same reaction family the oracle's ionizable-lipid
  training set is built from. It passes the descriptor gate and only trips the
  conservative **fingerprint-similarity** gate (~0.45) — so predictions are **near-domain
  extrapolation**, ranked as an **active-learning batch**, not "blind abstention on a
  novel chemotype." BEAE rankings are still **provisional / confirm-in-wet-lab**, and
  RM-60 / Example-2 are **structurally qualified leads**, not oracle-validated potencies.
