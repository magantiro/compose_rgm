# PHASE0B_DATA_RECIPE.md — B-edit universal-prior training corpus

The training-data recipe for the universal edit prior `Q_θ(y|x,t)` (master plan §0b). Percentages and
curriculum here are **PROPOSED, NOT LOCKED** — per the standing rule, final trace-mixture fractions are
chosen only after **at-scale (GuacaMol-TRAIN) statistics** exist. This document pins the *framework*: the
layers, the measured signals, the anti-domination safeguards, the data-driven curriculum, the versioned
manifest, and what still requires the gated Modal mining.

## Layers (three-way mixture)

1. **Synthetic corruption** (`rewrite/source_corruption.py`) — a real GuacaMol molecule walked a few legal
   micro edits to a nearby valid source, taught both directions (trim: real→corrupt; grow: corrupt→real).
   Covers the *reconstruction* signal and every model family (peripheral atom, bioisostere incl. ring-atom
   heteroatom scan, non-ring bond-order, aromatize↔de-aromatize, clean ring-open, cyclic graft).
2. **MMP-derived analogue traces** (`scripts/build_analogue_trace_pool.py` → `experiments/analogue_prior.py`)
   — real→real medicinal-chemistry transformations (one-cut matched molecular pairs), A→B and B→A. Covers
   *real transformation* structure that corruption cannot invent (the identity-bias counterweight).
3. **Sparse nearest-neighbour shared-scaffold traces** — real→real edits between same-scaffold neighbours
   (A2.3; Murcko-graph candidates mined, compilation is the resonance-invariant scaffold-pair compiler, still
   pending). Sparse NN only — **never all-pairs cliques within a scaffold group**.

## Measured characterization (pilot scale)

Source of truth: `diagnostics/composition/edit_trace_characterization.json` (regenerate with
`scripts/characterize_edit_traces.py`). Measured at pilot scale (350 GuacaMol-val molecules → 700 corruption
traces; the 98-trace A2.1 analogue pool):

- **Corruption layer (700 traces, 350 real→corrupt / 350 corrupt→real):** operator mix `atom_restate 32%`,
  `bond_reorder 15%`, `atom_delete 14%`, `atom_insert 14%`, `bond_reroute`(graft)`14%`, `ring_system_restate
  13%` (3.1 ops/trace, **6 active families**). Path length 1–5 (median 3); tercile bins local≤2 / lead-opt
  3–4 / scaffold >4. Source↔target Tanimoto median **0.403**, 50% in the [0.4,0.9] band (min 0.065 — deep
  corruptions drift far). **identity_fraction 0.29%** (no identity bias). Ring-*count* change **0%** (ring-
  count-preserving by design) but **aromatic-ring change 47%** (aromatize↔de-aromatize active). Size median
  26 heavy; charged 4.9%. Scaffolds 590 unique/700, **Gini 0.14** (diverse); 209 unique transformations,
  top-5 18%.
- **Analogue layer (A2.1, 98 traces = 49 pairs × 2 dirs):** 92.5% both-direction compile; operator mix
  `atom_delete 50%` / `atom_insert 50%` **only** (a one-cut MMP peel/graft is exactly leaf-to-root delete +
  attachment-outward insert). Path length 2–16 (median 8); Tanimoto median 0.455, 61% in band;
  **identity_fraction 0.0%**. Ring-count and aromatic change 0% (acyclic variable region, core preserved).
  Size median 17; charged 2.0%. Scaffolds **29 unique/98, top-1 37%, top-5 51%, Gini 0.39**; 29 unique
  transformations, top-5 51%.
- **Overall (798):** identity 0.25%, Tanimoto median 0.415 (52% in band), scaffold Gini 0.22, ring-count
  change 0%.

> Two load-bearing coverage facts: (1) **operator diversity comes entirely from the corruption layer** — the
> one-cut MMP analogue contributes only delete/insert, so a corruption-*only* or analogue-*only* corpus is
> wrong. (2) **Neither layer changes ring COUNT** (corruption preserves it; one-cut MMP preserves the core),
> so ring-topology-*adding* supervision comes from B's retained de-novo `ring_system_grow`, not the edit
> layers — consistent with the 07-25 learning that ring-opening is taught trim-only and ring-closing is
> retained from B.

## Proposed mixture (NOT LOCKED — starting point, adjust from at-scale stats)

| Layer | Proposed share | Rationale |
|---|---|---|
| Corruption | ~50% | family coverage + reconstruction; the only source of restate/bond-order/ring/graft supervision at pilot scale |
| MMP analogue | ~35% | real transformations; the identity-bias counterweight — but at pilot scale it is **53 pairs**, far too few to counterbalance ~50% corruption (see safeguards) |
| Scaffold NN | ~15% | real same-scaffold edits; **pending A2.3 compilation** |

These are the master-plan starting fractions. **Do not commit them to the trainer** until: (a) the MMP layer
is mined at GuacaMol-TRAIN scale (thousands–millions of pairs, not 53); (b) the per-layer statistics below
clear every safeguard; (c) the operator-family support (below) is non-degenerate for every enabled family.

## Anti-domination safeguards (each with a measured signal + a gate)

1. **Identity bias** — corruption can emit `source == target` (a no-op reconstruction). *Signal:*
   `identity_fraction` per layer. *Gate:* drop identity traces at build time; require corruption
   `identity_fraction ≈ 0` and enough *real-transformation* (analogue+scaffold) mass that the model is not
   an identity autoencoder. At pilot scale the 53-pair analogue layer is **too small** to counterbalance
   corruption — this is the single biggest reason at-scale MMP mining is required before training.
2. **Corruption domination** — corruption is cheap and unlimited; MMP is scarce. *Signal:* realized layer
   shares. *Gate:* cap corruption share and *floor* the real-transformation share; never let the sampler's
   uniform-over-records pick collapse to mostly corruption because it is the largest pool.
3. **Frequent-scaffold domination** — a few Bemis-Murcko scaffolds can dominate. *Signal:* scaffold Gini +
   top-5 share over source molecules. *Gate:* per-scaffold cap on traces; if Gini/top-k exceed a threshold,
   down-sample the head scaffolds.
4. **Transformation duplication** — the same transformation repeated. *Signal:* transformation-signature
   top-k share / #unique. *Gate:* de-duplicate by transformation signature; per-transformation cap.
5. **Missing supervision for rare operator families** — an enabled family (e.g. cyclic graft, ring-open,
   de-aromatization) with near-zero trace support trains a dead head. *Signal:* per-family trace count.
   *Gate:* every enabled family must have ≥ a floor count of supervising traces; if a family is rare,
   oversample corruption instances that exercise it (family-first corruption already supports this) rather
   than shipping an unsupervised head. Cross-check against the editing-sampler capability tests.

### Pilot findings — what the measured data triggered

- **Identity bias — CLEAN.** 0.25% overall (corruption 0.29%, analogue 0.0%). Identity traces are dropped;
  no identity-autoencoder risk at this composition.
- **Corruption diversity — CLEAN.** Scaffold Gini 0.14, 590/700 unique scaffolds, 6 active families,
  aromatize↔de-aromatize on 47% — the corruption layer is not dominated.
- **Analogue scaffold + transformation domination — TRIGGERED (expected).** One Murcko group supplies
  36/98 traces (top-1 37%, top-5 51%), only 29 unique transformations. This is the small-pool `C(n,2)`
  concentration the master plan warns about — **not a code defect**; the fix is at-scale mining + per-scaffold
  caps + uniform-group sampling. Do not train on this pilot pool as the analogue layer.
- **Rare-operator family — `ring_system_delete` zero support was a CHARACTERIZATION-CONFIG artifact, not a
  trainer gap (RESOLVED).** The characterization ran `make_edit_pair` *without* a ring `catalog`, so clean
  ring-opening is a no-op there (`source_corruption`: ring deletes are gated on `catalog is not None`). The
  **trainer already threads `catalog=ring_catalog`** at its corruption call site
  (`train_tracelet_cnof_gate.py`), so `ring_system_delete` *is* supervised in the real recipe: with a
  catalog, 92/200 corruption trims contain a ring delete, and a `ring_system_delete` teacher gives POSITIVE
  supervision (a short overfit raises its probability; the family head gets a nonzero gradient) — verified in
  `tests/test_ring_opening_supervision.py` (aromatic carbocycle / heterocycle / saturated / fused / a
  not-openable acyclic negative; replay validity; heteroatom preservation; inverse round-trip). The ring
  catalog is the SAME versioned definition used by enumeration / executor / sampling, and its
  `ring_catalog_fingerprint` is recorded in checkpoint metadata + this data manifest.
- **No ring-count change in either layer — BY DESIGN.** Ring-topology *adding* is retained from B's de-novo
  `ring_system_grow`; the edit layers preserve ring count. Acceptable, but the corpus alone does not teach
  ring growth — the warm-start from B carries it.

## Curriculum (data-driven, never preset)

Edit-budget bins are the **empirical terciles of compiled path length** (`_quantile_bins` in
`scripts/analogue_curriculum_stats.py`), refined by **ring complexity**, **attachment count**, and
**operator requirements** — never hand-set easy/medium/hard. Bin names track the edit regime
(micro / bioisostere / scaffold). The realized bins are recorded per layer in the characterization report
and pinned into the data manifest.

## Versioned data manifest

`scripts/build_edit_data_manifest.py` → `diagnostics/composition/edit_data_manifest.json`. Contains: source
GuacaMol partition; standardization hash (graph representation / SMILES policy); operator-registry hash
(`MARK_RULE_NAMES` + executor + operator payloads); compiler-version hashes (corruption + MMP + loader);
split identifiers; trace counts; and the curriculum-bin definitions. A training run records the manifest so
the corpus can never silently drift from the code that produced it.

## At-scale work (gated on Modal — not local)

- **Full MMP mining on the GuacaMol TRAIN partition** (the pilot's 53 pairs → thousands–millions), with
  scaffold-disjoint splits and no reverse-pair / MMP-core leakage.
- **A2.2 / A2.3 compilation** — the resonance-invariant fused-ring MMP fix and the sparse-NN scaffold-pair
  compiler (Murcko candidates are mined; compilation is pending).
- **Lock the mixture percentages and curriculum** from the at-scale statistics, then regenerate the manifest.

Nothing here launches Modal; the local deliverables are the characterization pipeline, the manifest, and
this proposed framework. The percentages are finalized only after the at-scale run.
