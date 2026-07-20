# Handoff: corpus + oracle → the COMPOSE-Lipid generator session

**From:** the lipid-corpus + pan-lung-oracle lane (`claude/lipid-corpus-oracle`)
**To:** the generator session (`compose_rgm_claude_generators`, `claude/generator-cond-uncond`)
**Coordination:** via pushed commits + this doc + the running status doc. No overlapping edits.

This is what the corpus lane provides for training + evaluating the general
(linker-agnostic) COMPOSE-Lipid generator, and how the oracle consumes its output.

## 1. What to train on — MATERIALIZED, training-ready

**Canonical training corpus = R0 (real anchor) + R1 (reaction-grounded reachable support).**
Manifest: **`artifacts/datasets/compose_lipid_pretraining_v1/training_corpus_manifest_v1.json`** (`format: compose_lipid_training_corpus_v1`) — the single entry point; it has layer counts, per-layer sha256, realism-weighting spec, kernel profile, and split policy.

- **R0 real anchor:** `r0_observed_real_structures.csv` — **15,433** unique measured/observed canonical ionizable lipids. High-weight anchor; never inherits biological labels.
- **R1 reachable support:** **`r1_reaction_grounded_corpus_v1.csv`** — **464,265** unique route-certified products (sha `8a691e47…`), columns: `canonical_smiles, reaction_family, reactant_ids, reactant_roles, size_bin, charge_bin, ring_bin, n_tails_bin, tail_length_bin, linker_type, realism_weight`. This is the RGM rewrite-kernel target set (the reachable support the generator learns to cover). **Total corpus ≈ 479,698** ("~500k"; a larger genuine count needs a broader documented block source, not more combinations — see §4).
- **Realism sampling:** **sample R1 by the `realism_weight` column** to get an R0-faithful distribution while the whole reachable set stays available for coverage. Weights are IPF-fit to R0 on linker/n_tails/tail_length/size — verified weighted JS-to-R0: linker 0.005, n_tails 0.048, tail_length 0.003, size 0.002 (head_size validated on the pilot audit, JS 0.021). Do **not** treat raw per-family row counts as the training distribution — Ugi/Passerini dominate the raw combinatorial support; the weights correct them to R0's real linker mix.
- Regenerate deterministically (seed 20260720, ~30 min): `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src:. python3 -m scripts.materialize_training_corpus`.
- (Legacy `layer_aware_training_manifest.json` = the older 26.5k R0+AGILE union; superseded by the above for training.)

**Distribution is realism-matched to R0 by a two-tier resample** (`enumerate_corpus_pilot.py --realism-target N --marginal-match`): (1) linker/family quotas weighted by R0 linker frequency; (2) **iterative proportional fitting** over the four axes family quotas can't control — `n_tails`, `tail_length`, `head_size`, plus `linker_type` — so every DOF matches R0 jointly; (3) a **per-family coverage floor** (drawn IPF-weighted within family) keeps all 12 reaction families visible for the generator, including linkers R0 barely contains. Verified by `structural_freedom_audit.json`: **all 10 DOF well-matched (JS-to-R0 ≤ 0.07, off-ratio axes: none)** and heteroatom-core Hill 188.0 vs R0 188.3. Honest residuals: the finer tail-*architecture* proxy is more concentrated than R0 (Hill 5.3 vs 11.6) and intra-corpus NN-Tanimoto is 0.956 vs R0 0.884 — expected of realism-matching (R0 itself is homologous-chain redundant); exact uniqueness stays 100%.

## 2. Chemistry the kernel must support (P2-G2)

From `training_corpus_manifest_v1.json → kernel_readiness_profile` (full 464k corpus):
- **Elements:** C, N, O, **S, P** (S = disulfide/thioether; P = iPhos phosphate). NOT optional — degradable-linker families require them. The handoff's "(+P if in scope)" must resolve to **P in scope**; coordinate the shared kernel element set with Codex.
- **Charges:** all products neutral (protonatable ionizable heads); the kernel state must also represent the **protonated** amine (endosomal).
- **Sizes:** **median 48** heavy atoms, max **138**; 86.5% ≤64, 96.8% ≤96 — the size cap must go 40 → ~48+ with a tail for the upper range. **The C/N/O/F small-molecule kernel is not lipid-ready.**
- **Stereo:** **88%** carry stereocenters — a stereo policy MUST be declared (much higher than the drug-like ~50%; lipid tails are stereo-rich).

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
