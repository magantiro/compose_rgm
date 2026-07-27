# Production Preflight Report — B-edit (max_atoms=40, broad-organic)

**Scope: PREFLIGHT ONLY.** Stops before any Stage-7 A100 optimizer update. Living document; each section is
filled as its stage completes. Frozen launch commit lineage: `f8894f2` (+ preflight commits below).

Status legend: ✅ done/verified · 🔶 in progress · ⏳ pending · 🚨 blocker.

---

## 1. Source commit & environment
- Frozen commit: `f8894f2` on `claude/control-closed-pareto-editing`; code tree (`src/ scripts/ modal_apps/
  tests/`) clean. Modal launches use a clean detached worktree.
- Production runtime (base checkpoint manifest): **A100-SXM4-40GB · Python 3.11.12 · torch 2.4.0+cu121 ·
  rdkit 2024.03.5**.
- Local verification env: `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src`; full suite 502 passed.

## 2. Corpus accounting (§1 of the max_atoms=40 reconciliation) ✅
`500,000 = 466,483 retained + 33 canonical_duplicates + 33,484 rejected` (exact, machine-checked). Rejected =
32,171 (>40 atoms) + 1,112 unsupported-element + 201 unparseable. `scanned_lines = 500,000` (no blank lines →
physical = nonempty = 500,000).

## 3. Base checkpoint identity (Stage 3) ✅
`compose-v4-stage3-flexible-graft-3k-1ac6f19-v1/checkpoint.best_so_far.pt`
- **SHA-256 `c9d927510360ec6eb84ff8dae1a222b0b693a9bef0ca23bb5d9cca063025876c`** (matches recorded Lineage B).
- `max_atoms=40` · `hidden_dim=256` · `message_passing_steps=6` · **CNOF 4-class heads** · 12-wide raw
  `atom_embedding` · **non-EMA** · `source_prior=carbon_tree` · `training_backend=factorized_marks` ·
  `ring_electronic_mode=factorized_local` · `rate_factorization=hierarchical` · `bond_representation=aromatic`.
- Reconstructed from its own stored config and **strict-loaded cleanly** (all keys matched) → exact architecture.

## 4. Base training subset provenance (§1) ✅ VERIFIED
The base model was trained on a **deterministic 50,000-molecule CNOF-neutral subset** of the version-pinned
500,000-molecule GuacaMol source corpus (`corpus_sha256 70526d92f1f08d8e…`).
- Selection: `load_cnof_corpus_split(train=50000, val=2000, test=2000, max_atoms=40, seed=20260717,
  scan_all=True, split_strategy="random")` — the first 50,000 of the seed-shuffled set of CNOF-neutral,
  ≤40-atom, canonicalized-deduplicated molecules.
- **CNOF-neutral ≤40 eligible in the 500k corpus: 225,149.** B trained on 50,000 of them (22% of eligible
  CNOF, 10% of the corpus).
- **`b_train_sha256 (ordered) = 83cebcef9a7a880a4d44f80dc67b6b5b5ed57dd0d16966da9c5738a0eb1bb00d`**;
  sorted = `650178e4150ea6c6…`. Artifact: `diagnostics/production_preflight/b_subset_provenance.json`.

**Four distinct universes (do not conflate):** physical corpus `guacamol_subset_500000_seed0.smiles` (500,000) ⊃
B's training subset (deterministic 50,000 CNOF-neutral ≤40) ; B-edit mining universe (466,483 broad-organic
≤40 eligible) → B-edit trace pool (363,456 broad-organic edit records). **B supplies a learned CNOF de-novo
initialization; B-edit adapts it to editing and expands it to broad-organic chemistry.**

## 5. Warm-start: strict flag crashes; semantic partial transfer (Stages 4, §2) ✅ impl+verified
- `--initialize-from-source-checkpoint` (strict `load_state_dict`) **RAISES** `size mismatch for
  restate_head/grow_root_head (4→15)`. **Production must use `--initialize-compatible-from-source-checkpoint`.**
- **Semantic partial transfer** (`_semantic_partial_checkpoint_initialization`, gate) copies shared classes by
  `(element,valence)` **label**, fresh only for genuinely-new classes:

  | Head | Layout | Shared C/N/O/F rows | New S/P/Cl/Br/I/B rows |
  |---|---|---|---|
  | `restate_head` (w+b) | per-class Linear | 4 bit-exact to B | 11 fresh |
  | `grow_root_head` (w+b) | per-class Linear | 4 bit-exact to B | 11 fresh |
  | `grow_option` | role-major 3×class Embedding | 12 bit-exact (4×3) | 33 fresh (11×3) |

  Body / `atom_embedding` / `total_hazard` = `COPIED_EXACT`. `restate_order_embedding (4,256)` correctly
  excluded (bond-order axis). Shared-support raw logits equal B **exactly** by construction. Transfer-map hash
  `bb89525b6021adb1`. Row-level table produced.

### 5b. Input embedding classification (§5) 🔶
`atom_embedding.weight (12,256)` copies shape-exact from B, but only the CNOF element rows were *trained* in
B's 50k CNOF run:
- CNOF element rows (C,N,O,F) → **INHERITED_TRAINED**.
- non-CNOF rows (B,P,S,Cl,Br,I) → **INHERITED_COLD** (present in B's architecture, never received training
  examples). Retained (finite, reproducible, trainable); Stage-7 must confirm they receive production
  gradients + a nonzero update. Not reinitialized.

## 6. Cache dependency resolution (§8) 🔶 — materially shrinks the compile
Production v2 = `--train-only --corrupted-prior-mix --organic-vocabulary --scaled-manifest … --analogue-trace-pool
… --initialize-compatible-from-source-checkpoint`. `denovo_keep = 0` under `--scaled-manifest` (gate:2279) →
`train_records = edit_records` only; the sampler builds no de-novo layer.

| Cache | Needed by v2? | Note |
|---|---|---|
| De-novo path cache `compiled_paths.pt` | **NO** | built+required then discarded (`[:0]`); zero-mixture branch must skip it |
| Training support cache | **NO** | guarded off for `--scaled-manifest`; on-the-fly |
| Eval cache (validation) | ⏳ OPEN | currently de-novo; decision: de-novo retention vs edit validation |
| MMP pool `edit_pool_full.jsonl` (363,456) | present | loaded from volume, not compiled |
| Corruption records | n/a | built in-memory from `split.train` at train start |

**Coupled fixes for the zero-mixture branch:** (1) source `ring_catalog` from **B's checkpoint** (not de-novo
recompile) so ring-template params stay `COPIED_EXACT`; (2) resolve validation. Statement of required caches
finalized once (2) is decided.

---

## 7. Warm-start wiring + dry-run (§2, §3) ✅
- Semantic transfer wired into `--initialize-compatible-from-source-checkpoint` (gate); transfer provenance
  (source SHA, impl version, map hash, source/dest vocab hashes, copied/fresh counts, row-table hash) stored
  in checkpoint metadata. The config-mismatch gate is unaffected (vocab/scope/edit flags not in `expected`).
- **Real-checkpoint dry-run gate `scripts/warmstart_dry_run.py` → `GO_WARMSTART_DRY_RUN`** (with the
  PRODUCTION catalog): (A) strict init fails on the widened heads; (B) semantic init loads; (C) all 114
  tensors accounted (**108 COPIED_EXACT + 6 widened**); (D) recognized; (E) finite no-update forward/sample.
  20 rows copied-by-map / 55 fresh; map `bb89525b6021adb1`. Artifact: `…/warmstart_dry_run.json`.
- **Ring-catalog compatibility (§8):** the production catalog is deterministic (5 fixed seeds) → rebuilds to
  **`639ff6078c32d43c`** = the manifest fingerprint. **B's checkpoint catalog `50337de077f374db` MISMATCHES**
  (built from B's CNOF de-novo paths at an older commit). B-edit uses the **production** catalog (authoritative,
  §8); B's is rejected. The one catalog-specific tensor `ring_system_template_key.weight` is correctly **fresh**
  (108 copied vs 109 with B's catalog), so B's stale template identity never leaks in.
- Tests: `test_semantic_partial_transfer.py` (10), `test_warmstart_launch_flag.py` (2, doc can't revert to
  the strict flag). Commits: #1 `30c92d0` (impl+tests), #2 `047bf03` (wiring), #4 `325dff4` (provenance/docs).

## 8. Calibration (§6) — diagnostic proposal, NOT applied
Fixed 8-molecule CNOF panel; B vs whole-head-reset vs semantic-transfer, on identical node features.
- **(A) Shared HEAD-CONDITIONAL fidelity — semantic preserves B's CNOF head-conditional distributions
  EXACTLY**: restate/grow_root shared TV = **0.0 / 0.0**, raw-logit max err = **0.0**. Whole-head-reset
  destroys it (TV 0.35 / 0.095). *(Complete-mark + canonical-successor shared fidelity: pending §5 — the
  "exact shared kernel" claim is deferred until then.)*
- **(B) New-support mass**: the 11 fresh S/P/Cl/Br/I/B rows **over-draw ~71–82%** of the restate/grow
  categorical at init — an **initialization failure**. A new-row-only bias correction reduces it without
  changing shared-row logits.
- **(C) DIAGNOSTIC_CALIBRATION_PROPOSAL (not applied):** `new_bias[c] = log(smoothed atom-freq[c]) + β`
  drives α_new to the **broad-corpus non-CNOF atom fraction (2.6%)** while keeping shared fidelity TV=0. This
  2.6% is an atom-occurrence PLACEHOLDER, **not** the action-conditioned selected-target rate. The
  **production** policy target = the operator-conditioned **selected-target** distribution from the locked
  sampler (§2), re-derived on a stratified held-out calibration panel (§4). **β values here are diagnostic;
  NOT applied to any optimizer update.** Artifact: `diagnostics/production_preflight/calibration.json`.

## 9. Catalog semantic audit (§1–§5) ✅ transfer cleared; 🚨 ring_system_grow finding
- **§1 dependency table:** built B-edit under 4 catalogs of different cardinality → **exactly 1** catalog-
  indexed tensor (`ring_system_template_key.weight`); the other **113 are catalog-INDEPENDENT** (constant
  shape) and copy safely. **No catalog-indexed tensor is positionally copied** → transfer CLEARED.
  Artifact: `diagnostics/production_preflight/catalog_semantic_audit.json`.
- **§2 template map:** base **4096** ring-system templates (hit the cap) vs production **5** (seeds);
  **SHARED_EXACT = 0**, no duplicate/ambiguous keys. **§3:** `ring_system_template_key` is fully fresh
  (all PRODUCTION_ONLY) — a safe FRESH_NEW_TEMPLATE, but **B's ring_system_grow template knowledge does NOT
  transfer.**
- **🚨 §5 finding — `ring_system_grow` is fresh AND unsupervised** (in `production_enabled`, but the
  corruption/MMP families never use grow, and the catalog mismatch makes its template rows fresh). Owner
  decision: **do not disable — verify uniqueness first, then retain + supervise if unique.**
- **DECISION: `RING_GROW_UNIQUE_AND_USEFUL`** (`diagnostics/production_preflight/ring_grow_decision.json`).
  `ring_system_grow` **cyclizes a chain into a ring** (Δrings=+1, Δatoms=0); `cyclic_graft` **only relocates**
  (verified: 16+34 grafts, every successor (Δatoms,Δrings)=(0,0)). grow uniquely increases ring count → a
  distinct, useful lead-opt capability → **retain + supervise** (§10). Also: the 0 operational-template
  overlap is a **source-encoding artifact** (templates encode the carbon-tree source), not chemical
  non-overlap — B knows benzene chemically, but its operational key differs, so `ring_system_template_key`
  is correctly fresh.
- **⇒ triggers a catalog/supervision rebuild (§4–§10):** real reversible ring-delete/grow dataset from
  GuacaMol; a **data-derived** production catalog (not the 5 seeds; size from a coverage curve); balanced grow
  supervision; fresh-row calibration; subtype/template-level tests; root-cause the Stage-6 false-positive;
  regenerate catalog/corruption/manifest + recompile catalog-dependent caches. **The production ring-catalog
  and operator-capability hash will change** → no final cache compile / Stage-7 until the rebuild lands.

## 10. §9 root-cause of the supervision false-positive ✅
`cold_vocab_audit.py` counted positive targets **only per `(element,valence)` atom class**, only from
`atom_insert`/`atom_restate` marks (line 71) — it had **no ring-template / operator-subtype notion**, so
`ring_system_grow` was never checked. "All families supervised" was really "all atom classes supervised."
**Fix (rebuild §8):** a subtype/template-level supervision gate + a regression test that fails when any
`production_enabled` operator subtype has zero positive selected targets.

### 10a. Subtype gate WIRED + run against the ACTUAL production record builders ✅ `GO_SUBTYPE_SUPERVISION`
`scripts/subtype_supervision_preflight.py` runs the exact production generators —
`build_corrupted_prior_records` (micro + cyclic graft + clean ring-open + de-aromatization) and
`build_cycle_op_records` (compositional cycle_close/cycle_open) — on **150** held-out-GuacaMol broad-organic
leads (4,670/5,000 eligible = 93.4%, scope `3721d69851110fdd`), then applies the subtype gate. The cycle ops
record executor names (`bond_insert`/`bond_delete`) that the model scores under `cycle_insert`/`cycle_attach`;
the gate aliases them (`_CYCLE_OP_EXECUTOR_TO_FAMILY`) so a supervised cycle op is not misread as unsupervised.
- **Positive selected targets / family** (1,437 records): `cycle_insert` 588 · `cycle_attach` 588 ·
  `atom_restate` 215 · `bond_reroute` 122 (cyclic graft) · `ring_system_restate` 116 (de-arom) ·
  `atom_delete` 106 · `bond_reorder` 92 · `atom_insert` 76 · `ring_system_delete` 44 (clean ring-open).
- **`compositional_support` → GO** (all 9 enabled families supervised; `unsupervised=[]`). The compositional
  cycle ops carry the ring-generation SUPPORT with **dense** supervision (588 each), so §9's "ring generation
  is fresh AND unsupervised" is resolved at the support level — ring growth/closure no longer depends on the
  unsupervised whole-ring macro.
- **`with_grow_macro` → NO_GO** (`ring_system_grow` unsupervised) — recorded on purpose: it proves the gate
  catches the still-open macro gap instead of silently passing it (the exact §10 false-positive). The
  `ring_system_grow` **K-macro** supervision is P4 (acceleration only; must not define support).
- Artifact: `diagnostics/production_preflight/subtype_supervision.json`. Tests:
  `test_operator_subtype_supervision.py` (+2 cycle-op alias cases).

## Ring-grow rebuild plan (§4–§10; the production catalog + operator hash WILL change)
| Phase | Where | Work |
|---|---|---|
| A (local) ✅ | §1, §9 | subtype-level supervision gate + regression test WIRED (§10a); run on the production builders → `GO_SUBTYPE_SUPERVISION` (compositional cycle ops carry ring support, 588 targets each; gate fails iff `ring_system_grow` is enabled unsupervised) |
| B (local, heavy) | §4, §5 | build the reversible ring-delete→grow dataset from GuacaMol; derive a **data-derived** catalog, pick size from a coverage curve (32…1024), round-trip verified |
| C (local) | §6, §7, §8 | balance grow supervision in the corruption recipe; calibrate fresh template rows from the selected-target dist; subtype/template tests |
| D (Modal) | §10 | regenerate catalog/corruption/manifest (new scope+operator+catalog hashes); recompile catalog-dependent caches; re-run preflight gates |
| E | — | cache + no-update throughput → cost; final GO/NO-GO |

## 11. Ring-catalog study (§0–§6) — root cause found: over-specific template identity
Catalog-free reversible grow data (`compile_carbon_tree_to_target` over GuacaMol → the `ring_system_grow`
steps are the grow events). Decisive comparison (`diagnostics/production_preflight/ring_catalog_study.json`):

| Template key | distinct | K=32 cov | K=64 cov | ≥50 support |
|---|---|---|---|---|
| **current `electronic_key`** | 2835 | 11% | 16% | **0** → `NO_GO_TEMPLATE_SEMANTICS` |
| **simplified ring-graph** | **121** | **97.1%** | **98.5%** | 4 (benzene = 56% of events) |

- The current `ring_system_grow` template key is **massively over-specific** (source/placement operands leak
  into the learned class) — 1.3× collapse, K=1024 covers only 52%, no template ever reaches ≥25 support. This
  is **why grow was unsupervisable** and quietly left unsupervised.
- A **canonical ring-graph identity** (element+aromaticity, context-free) collapses 31× → **121 chemical
  templates**, K=64 covers **98.5%**; at 400k scale the common templates are heavily supported.
- **Status: `NO_GO_TEMPLATE_SEMANTICS` (current key) → `GO_REBUILD_WITH_RING_CATALOG_K≈64` after a §6
  template-identity simplification** (move placement to the action instance; align model + enumeration +
  executor). Tentative **K=64** (hybrid rule), pending the full-corpus curve + support-learning floor.
- ⇒ the rebuild now requires a **model change to the `ring_system_grow` parameterization first** (changes the
  operator/catalog hashes + the ring-head checkpoint contract), then catalog + supervision + calibration +
  regenerate. Still pending: §3 support floor, §4 corrected topology strata, §5 branching/throughput, §7 priors.

## Pending (before the STOP report)
- §5/§7 Stage-7 gradient checks: cold non-CNOF embedding rows + fresh output rows receive gradients + nonzero
  updates (runs during the bounded preflight); apply the §7 bias policy at init.
- §8 zero-mixture branch (skip de-novo cache; ring-catalog from B's checkpoint) + validation decision.
- Cache compilation (much smaller than the old launch doc) + no-update throughput probe → 300/500/750/1000-step cost.
- Final GO/NO-GO.
