# COMPOSE-Lipid corpus & oracle — running status

**Owner:** Claude (branch `claude/lipid-corpus-oracle`)
**Scope:** ~500k reaction-diverse lipid pretraining corpus + pan-lung representation×model oracle matrix.
**Last updated:** 2026-07-20

Run tests/scripts with `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src` (macOS OpenMP guard). Scripts importing the `scripts` package need `PYTHONPATH=src:.` and module mode (`-m scripts.<name>`).

**Ownership map (cross-stream coordination interface = pushed commits + this doc):** This lane owns the lipid corpus, pan-lung oracle matrix, and Paper 2 data. The COMPOSE generator (unconditional/conditional) is owned by a **separate Claude session** in worktree `compose_rgm_claude_generators` (branch `claude/generator-cond-uncond`) — do not edit generator-owned code. Lipid model training is gated behind the upstream Paper 1 generator + QED authorization (P1-G7).

---

## Completed

### Audit & reproducibility (verified live)
- Handoff manifest: **21/21 files hash-match**.
- Raw sources: 6/6 obtainable hash-verified. LUMI + LuT re-fetched from primary origins (Zenodo `10.5281/zenodo.17771224`, Springer source-data xlsx) — byte-exact to frozen SHA-256. LuT 444-row extraction regenerates to its exact hash.
- Frozen results reproduce: LuT baselines **bit-exact** (0.00 diff / 240 metrics); LUMI baselines to 4 decimals; corpus counts exact (R0=15,433, union=26,509, AGILE-only=11,076).
- Test suite green: **42 lipid/oracle/reaction tests pass**.

### Oracle — Milestone 4: component-transfer qualification → oracle design decision
Answers "does head/tail SAR transfer well enough for a single oracle to rank novel-linker candidates?" Held-component vs held-lipid on two independent in-vitro datasets:

| Axis held out | A549 Spearman | LUMI Spearman |
|---|--:|--:|
| held-lipid (baseline) | 0.568 | 0.830 |
| **head / R1 (amine-like)** | **0.158** | **0.475** |
| tail / R2–R4 (secondary) | 0.508 | 0.735–0.821 |

**Finding (2-dataset consistent):** the high-diversity **ionizable-head axis transfers poorly**; **tail / secondary axes transfer well**. Mechanistic (head drives pKa/escape potency; tails interpolate) and consistent with the LuT round-transfer collapse (0.31).

**Design decision (locked for the oracle lane):** *not* one global oracle applied blindly. Use a **head-aware applicability-domain gate** — candidates reusing in-distribution ionizable heads + varying linker/tails are rankable; novel-head candidates abstain. For the novel Michael-addition linker, run a **small active-learning calibration** (known heads × new linker) to anchor the linker offset, then rank within-family by transferable tail/secondary SAR. Matches the plan's calibrate-don't-extrapolate rule and the `reward_guard` OOD-abstention already in code.

### Oracle — Milestone 3: R1×M4 masked multitask MLP cell (+ negative-transfer finding)
- New matrix cell `R1×M4`: shared molecular trunk (Morgan2048 + RDKit descriptors) with typed per-head outputs, masked loss, held-lipid GroupKFold (global grouping). Trained multitask vs single-task under identical folds/features.
- Held-lipid Spearman (multitask / single-task): A549 **0.592**/0.576 · LUMI-HBE **0.885**/0.885 · HBEC-ALI −0.06/−0.03 · intratracheal −0.10/−0.17 · IV-barcode −0.06/−0.05.
- **Finding (reported negative transfer):** shared-representation multitask is competitive on the large full-SMILES heads (A549 edges the frozen ExtraTrees 0.568) but **does not rescue the small in-vivo/local heads** — they stay near-zero regardless of sharing. Confirms those heads must remain auxiliary/AD-gated (as the filtering bundle already does); they need formulation context or active-learning, not more representation sharing.
- LuT excluded (component-only → R2 lane); no molecular-graph imputation.

### Oracle — Milestone 2: canonical row-level pan-lung manifest
- Materialized `artifacts/oracles/pan_lung_canonical_v1/{canonical_rows.csv,manifest.json}` — the stated blocker for R3/R4/R5.
- **4,783 head-measurement rows** from **4,339 distinct measurements** (reconciles with the frozen corpus manifest). Grain = (measurement × typed head); LuT's 444 compounds → 888 paired expression+selectivity rows.
- All 7 typed heads at exact expected counts (A549 1801, LUMI-HBE 1920, HBEC-ALI 29, intratracheal 49, IV-barcode 96, LuT-expr 444, LuT-sel 444).
- Readout types kept **semantically distinct** (in-vitro transfection / functional in-vivo expression / biodistribution / selectivity — never pooled).
- `has_full_structure` boundary: 3,895 molecular rows (LNPDB+LUMI) vs 888 component-only (LuT); LuT graphs never imputed.
- **0 exact-structure overlap** LNPDB↔LUMI (no cross-source leakage). Covariates preserved (formulation/helper/ratios/cargo/dose/route/species/assay/timepoint/study/batch); missingness explicit; study constants recorded, not invented.

### Corpus — Milestone 1: first qualified reaction transform (Ugi-3CR)
- **Ugi-3CR (acid-free α-amino amide)** qualified end-to-end. Atom-mapped SMARTS:
  `[NX3;H2,H1:1].[CX3H1:2]=[OX1].[C;-1,+0;X1:3]#[N;+1,+0;X2:4]>>[N:1][CH1:2][C+0:3](=O)[NH1+0:4]`
- **Correctness gate: 1,200/1,200 (100%) exact reconstruction** of the AGILE measured library from A/B/C components.
- Chemoselectivity: 5/5 negatives rejected (tertiary amine, ketone, nitrile, alkane, carboxylic acid).
- Route-replay smoke: 20×12×5 block grid → **1,320 unique route-certified products; 1,200/1,200 (100%) exact replay** of released products + 120 verified alternative regiochemistries (2 multi-N-H amine heads × 12 × 5). Heavy atoms 38–60 (median 47).
- Fail-closed preserved: `pilot_literature_specs_v1.json` untouched (still 0 qualified).

---

## Quantitative results
| Result | Value |
|---|---|
| Ugi-3CR exact reconstruction (AGILE) | 1200/1200 = 100.00% |
| Ugi-3CR chemoselectivity negatives rejected | 5/5 |
| Smoke unique route-certified products | 1,320 |
| Smoke exact route-replay of released library | 1200/1200 = 100.00% |
| Qualified reaction families | **9** (Ugi-3CR, aza-Michael, epoxide, Passerini, red-amination, amide, + degradable: thioether/thiol-Michael, disulfide, carbamate) |
| Building-block pool | 320 role-annotated (25 AGILE, 75 curated, 210 programmatic); elements C/N/O/**S** |
| Pilot enumeration | 209k+ unique products (multi-tail, ≥2-tail biased); 500k = pool-size, not machinery |
| Multi-tail architecture | polyamine polysubstitution (C12-200 style), ≥2-tail biased (R0: 97% ≥2 tails) |
| Corpus faithfulness JS-to-R0 | size 0.062, tail-count **0.055**, unsat 0.007, branch 0.028, charge 0.017 (all low) |
| Corpus core diversity (Hill Simpson) | 702 vs R0 214 effective heteroatom-cores (exceeds real anchor) |
| Corpus NN vs R0 intra-NN | 0.977 vs 0.887 (high NN intrinsic to lipids, not duplication; exact uniqueness 100%) |
| LNPDB recall@Tanimoto 0.4 | 0.88 |
| Design-principle alignment | Whitehead & Arral 2026 (Nat Rev Bioeng): H/L/T ✓, multi-tail ✓, branch/unsat ✓, intrinsic pKa~9 / apparent~6.4 ✓, degradable linkers (ester/amide/thioether/disulfide/carbamate) ✓; still to add: ketal/ether/urea/phosphate(iPhos, P), MW-600-1000 check |
| Pan-lung oracle cells complete | R1/R2 × M1/M2/M3 + M6 ensembles + **R1×M4** |
| Head-axis transfer (A549 / LUMI held-head) | 0.158 / 0.475 (vs baseline 0.568 / 0.830) |
| Tests passing | 91 |

---

## Current blockers / gates
- Lipid **model training** gated behind Codex Paper 1 P1-G7 (not on my critical path; data/corpus/oracle prep is authorized now).
- R3 cell needs a justified **frozen molecular encoder**; leave-study-out needs **>1 source per airway head** (LiON lung slice is the next admission).
- Corpus pilot (50–100k) blocked until **≥3–4 families qualified** (diversity metrics require multiple families).

## Corpus positioning (Nature Biotech)
Two-stage story ([[generator-general-then-linker-finetune]]): (1) headline = a **general linker-agnostic** insane lipid generator (Fig 2/3, Arm A); (2) downstream = fine-tune / linker-freeze on the novel Michael linker (Fig 6, Arm B). The corpus stays broad/general. The corpus diversity/coverage metrics double as the **Fig 2/3 generator-qualification yardstick** (validity, non-memorization via NN curves, fidelity via marginals, coverage via LNPDB recall, architecture, synthesis-eligibility, family generalization).

## Exact next actions
1. **Corpus (top priority — surfaced by metrics):** add **multi-tail architecture** — iterative reaction application so polyamine heads carry 2–4 tails, closing the arch-Hill gap (5.5→~11) and the tail-count/size JS gaps to R0. This is the biggest fidelity lever.
2. **Corpus:** scale/diversify the building-block pool (more heads, cyclic/heteroatom tails; commercial catalog when available) → 50–100k pilot → 500k; add held-reaction-family + held-component splits.
3. **Oracle (paused):** finish the intrinsic head-pKa feature (fix MolGpKa batch bug; cross-validate vs xtb) and test held-head transfer lift; then the head-aware AD gate.
4. **Oracle:** integrate LiON lung slice (on disk) for leave-study-out on airway heads.

## Key artifact paths
- `configs/lipid_reactions/qualified_reactions_v1.json` — qualified Ugi-3CR registry (hash-bound).
- `configs/lipid_reactions/ugi_3cr_building_blocks_v1.json` — frozen role-annotated block manifest.
- `artifacts/datasets/compose_lipid_pretraining_v1/ugi_3cr_qualification.json` — reconstruction audit (sha `fc265a92…`).
- `artifacts/datasets/compose_lipid_pretraining_v1/ugi_3cr_smoke_v1/{products.csv,smoke_audit.json}` — smoke release.
- `src/compose_v4/lipids/reaction_enumeration.py` — qualified-transform enumerator.
- `scripts/qualify_ugi_3cr_transform.py`, `scripts/enumerate_ugi_3cr_smoke.py` — reproducers.
- Tests: `tests/test_ugi_3cr_qualification.py`, `tests/test_ugi_3cr_smoke.py`.

## Commit log
- `5dd44c4` M1: qualify Ugi-3CR transform + route-replay smoke.
- `ee87690` M2: canonical row-level pan-lung manifest.
- `d9c7ebf` M3: R1xM4 masked multitask MLP cell + negative-transfer finding.
- `9b72cce` M4: component-transfer qualification + head-aware oracle design decision.
- `b646133` M5: qualify 5 complementary reaction families (6 total).
- `20496ab` M6: building-block pool (220) + stratified pilot + diversity metrics harness.
- `0c819f1` M7: lipid-appropriate v2 metrics (core Hill + NN-vs-R0 + faithfulness JS); architecture-gap finding.
- `b646133` (families) + `20496ab`/`0c819f1` metrics; `dd4cc5e` status sync.
- `01b0812` M8: multi-tail architecture + degradable linkers (9 families, +S) — Whitehead design-principle alignment.
- `264cc33` M9: intrinsic head-pKa feature — lifts held-head transfer +42% (0.18→0.26).
- `3480a46` M10: 34k route-certified corpus scale demonstration.
- `013df74` M11: family-balanced within-architecture selection (arch Hill 6.4→9.3; fixes S-linker starvation).
- `18eab6d` M12: head-aware applicability-domain gate — 1,183 known head-region scaffolds; rank-within / abstain-outside (milestone-4 decision operationalized).
- `264cc33`/`3480a46`/`013df74`/`7492958` (pKa feature, 34k scale, balanced selection, status).
- `aed2bb6` M13: complete linker palette — urea/acetal/iPhos (12 families, +P); elements C/N/O/S/P.
- `56e8706` M14: freeze leakage-resistant corpus splits (family/scaffold/head/study; all leak-free) — Fig 2 contract + generator handoff.

## Oracle applicability-domain decision (locked)
Head axis doesn't transfer → the oracle's honest domain is **known heads (large: 1,183 head-region scaffolds from R0+corpus, ~435 LNPDB head SMILES)**. High-ranked candidates come from this rich known-head space; novel-head candidates **abstain → small active-learning round**. Uses **intrinsic** head-pKa (~9), not apparent (~6.4). Head-pKa feature lifts within-domain head ranking (+42%). Next: wire the gate + pKa into the deployed filtering bundle.
