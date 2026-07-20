# Handoff: corpus + oracle → the COMPOSE-Lipid generator session

**From:** the lipid-corpus + pan-lung-oracle lane (`claude/lipid-corpus-oracle`)
**To:** the generator session (`compose_rgm_claude_generators`, `claude/generator-cond-uncond`)
**Coordination:** via pushed commits + this doc + the running status doc. No overlapping edits.

This is what the corpus lane provides for training + evaluating the general
(linker-agnostic) COMPOSE-Lipid generator, and how the oracle consumes its output.

## 1. What to train on

**Structural corpus = R0 (real) + R1 (reaction-grounded virtual).**

- **R0 real anchor:** `artifacts/datasets/compose_lipid_pretraining_v1/r0_observed_real_structures.csv` — 15,433 unique measured/observed canonical ionizable lipids. High-weight anchor. Never inherits biological labels.
- **R1 virtual:** reaction-grounded, route-certified products from the qualified registry over the building-block pool. Regenerate with:
  ```
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python3 scripts/enumerate_corpus_pilot.py --realism-target <N>
  ```
  ~464k unique products enumerable today (500k is a pool-size, not machinery, problem). Products carry `reaction_family` + route provenance.
- **Layering / sampling:** `artifacts/datasets/compose_lipid_pretraining_v1/layer_aware_training_manifest.json` (R0 77.4% / AGILE-Ugi 22.6%). Virtual layer teaches support/variation; it does not overwrite the empirical molecular distribution or create delivery labels.

**Distribution is realism-matched to R0 by a two-tier resample** (`enumerate_corpus_pilot.py --realism-target N --marginal-match`): (1) linker/family quotas weighted by R0 linker frequency; (2) **iterative proportional fitting** over the four axes family quotas can't control — `n_tails`, `tail_length`, `head_size`, plus `linker_type` — so every DOF matches R0 jointly; (3) a **per-family coverage floor** (drawn IPF-weighted within family) keeps all 12 reaction families visible for the generator, including linkers R0 barely contains. Verified by `structural_freedom_audit.json`: **all 10 DOF well-matched (JS-to-R0 ≤ 0.07, off-ratio axes: none)** and heteroatom-core Hill 188.0 vs R0 188.3. Honest residuals: the finer tail-*architecture* proxy is more concentrated than R0 (Hill 5.3 vs 11.6) and intra-corpus NN-Tanimoto is 0.956 vs R0 0.884 — expected of realism-matching (R0 itself is homologous-chain redundant); exact uniqueness stays 100%.

## 2. Chemistry the kernel must support (P2-G2)

From `corpus_pilot_v1/pilot_metrics.json → kernel_readiness_profile`:
- **Elements:** C, N, O, S, P (S = disulfide/thioether; P = iPhos phosphate).
- **Charges:** neutral products (protonatable ionizable heads).
- **Sizes:** median ~55 heavy atoms, up to ~128 (MW ~600–1,000, real range) — **the C/N/O/F small-molecule kernel is not lipid-ready**; extend elements + size before training.
- **Stereo:** ~50% carry stereocenters (policy must be declared).

## 3. Leakage-resistant splits (use these for train/val/test)

`artifacts/datasets/compose_lipid_pretraining_v1/splits_v1/` — `corpus_fold_assignments.csv` (per-structure fold for each split) + `manifest.json`. Splits (all verified leak-free — no group spans folds):
- `held_reaction_family`, `held_scaffold` (heteroatom-core), `held_head` (head region), `held_study` (R0 source), `random_diagnostic`.
Use held-scaffold/head/family for the Fig-2 non-memorization + reaction-family-generalization claims; random is diagnostic only.

## 4. Qualified generative substrate (what R1 is built from)

- **12 qualified reaction families** (`configs/lipid_reactions/qualified_reaction_families_v1.json` + `qualified_reactions_v1.json`): Ugi-3CR (validated 1200/1200 vs AGILE), aza-Michael, epoxide, Passerini, reductive amination, amide, thioether, disulfide, carbamate, urea, acetal, iPhos. Full linker palette; coverage card in `docs/audits/2026-07-20_corpus_coverage_card.md`.
- **410-block role-annotated pool** (`configs/lipid_reactions/building_block_pool_v1.json`): multi-source, multi-tail, 0/1/2 branches, C6–C22 × unsaturation.
- Enumerator: `src/compose_v4/lipids/reaction_enumeration.py` (fail-closed; only qualified transforms).

## 5. Oracle interface (scoring generated candidates)

Once generation runs, score candidates with the frozen filtering oracle:
```python
from compose_v4.oracles.nominate import OracleNominator
nom = OracleNominator.load(repo_root)            # head-aware gate + molecular AD + filtering bundle
nom.nominate(smiles, head_pka=None)              # -> rank | abstain_novel_head | abstain_off_domain | reject
```
- **Filtering only** — reward fine-tuning stays gated behind the preregistration guard (`reward_guard.py`).
- **Honest domain:** ranks within the ~1,183 known head-region scaffolds; novel heads abstain → small active-learning round. The novel Michael-linker campaign should keep **known heads × varied tails**.

## 6. Gates (do not skip)

- Lipid generator training is authorized only after the Paper 1 generator + recovery + matched-QED gate (P1-G7). This lane's data prep is authorized now.
- Requalify the corpus/oracle if the shared RGM core changes.
- The corpus stays general/linker-agnostic; the Michael linker is a downstream fine-tune, not baked into pretraining.
