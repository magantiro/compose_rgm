# COMPOSE-Lipid corpus & oracle — running status

**Owner:** Claude (branch `claude/lipid-corpus-oracle`)
**Scope:** ~500k reaction-diverse lipid pretraining corpus + pan-lung representation×model oracle matrix.
**Last updated:** 2026-07-20

Run tests/scripts with `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src` (macOS OpenMP guard).

---

## Completed

### Audit & reproducibility (verified live)
- Handoff manifest: **21/21 files hash-match**.
- Raw sources: 6/6 obtainable hash-verified. LUMI + LuT re-fetched from primary origins (Zenodo `10.5281/zenodo.17771224`, Springer source-data xlsx) — byte-exact to frozen SHA-256. LuT 444-row extraction regenerates to its exact hash.
- Frozen results reproduce: LuT baselines **bit-exact** (0.00 diff / 240 metrics); LUMI baselines to 4 decimals; corpus counts exact (R0=15,433, union=26,509, AGILE-only=11,076).
- Test suite green: **42 lipid/oracle/reaction tests pass**.

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
| Qualified reaction families | 1 of ≥6 target |
| Pan-lung oracle cells complete | R1/R2 × M1/M2/M3 + M6 ensembles (filtering bundle serialized) |

---

## Current blockers / gates
- Lipid **model training** gated behind Codex Paper 1 P1-G7 (not on my critical path; data/corpus/oracle prep is authorized now).
- Oracle R3/R4/R5 cells blocked on the **canonical row-level pan-lung table** (not yet materialized).
- Corpus pilot (50–100k) blocked until **≥3–4 families qualified** (diversity metrics require multiple families).

## Exact next actions
1. **Oracle:** materialize canonical row-level pan-lung manifest (LNPDB-lung 1,975 + LUMI 1,920 + LuT 444 = 4,339 typed rows) with preserved covariates + frozen split groups. Unblocks R3/M4.
2. **Corpus:** qualify breadth batch — epoxide opening, aza-Michael, Passerini — reusing the enumerator/registry machinery.
3. **Oracle:** R3 frozen embeddings + M4 masked multitask MLP over the canonical table vs frozen R1/R2 trees.
4. **Corpus:** stratified 50–100k pilot across qualified families with coverage/leakage gates.

## Key artifact paths
- `configs/lipid_reactions/qualified_reactions_v1.json` — qualified Ugi-3CR registry (hash-bound).
- `configs/lipid_reactions/ugi_3cr_building_blocks_v1.json` — frozen role-annotated block manifest.
- `artifacts/datasets/compose_lipid_pretraining_v1/ugi_3cr_qualification.json` — reconstruction audit (sha `fc265a92…`).
- `artifacts/datasets/compose_lipid_pretraining_v1/ugi_3cr_smoke_v1/{products.csv,smoke_audit.json}` — smoke release.
- `src/compose_v4/lipids/reaction_enumeration.py` — qualified-transform enumerator.
- `scripts/qualify_ugi_3cr_transform.py`, `scripts/enumerate_ugi_3cr_smoke.py` — reproducers.
- Tests: `tests/test_ugi_3cr_qualification.py`, `tests/test_ugi_3cr_smoke.py`.

## Commit log
- (pending) Milestone 1: qualify Ugi-3CR transform + route-replay smoke.
