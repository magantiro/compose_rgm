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

## Pending (before the STOP report)
- §2 wire semantic transfer into `--initialize-compatible-from-source-checkpoint` + mismatch allow-list +
  checkpoint metadata + runtime map check.
- §3 real-checkpoint dry-run gate + launch-command regression test.
- §4 semantic-transfer tests (11) + final row-level artifact.
- §6 CNOF calibration panel (whole-head-reset vs semantic); new-support mass; policy if needed (§7).
- §8 zero-mixture branch (skip de-novo cache; ring-catalog from B) + validation decision.
- Cache compilation (much smaller than the old launch doc) + no-update throughput probe → 300/500/750/1000-step cost.
- Final GO/NO-GO.
