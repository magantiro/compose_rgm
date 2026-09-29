# Reproducing the paper's results

**Start here.** This page maps every table in the ICLR 2027 submission to the
artifact that backs it and the command that regenerates it.

Nothing on this page needs cloud credentials, an oracle call, a docking run, or
a GPU. Everything it reads is committed in this repository.

```bash
python3 tools/reproduce_paper_tables.py          # PMO Tables 2 & 11, T4 Table 3
python3 tools/reproduce_paper_tables.py --pmo
python3 tools/reproduce_paper_tables.py --t4
```

The expected values are transcribed from the paper; the computed values come
from the result artifacts. Those are two independent sources, so a match is
evidence and a mismatch points at one cell.

---

## Status of each headline table

| Paper table | What it reports | Backing artifact | Reproduces |
|---|---|---|---|
| **Table 3** — T4 lead optimization | 30 cells vs IVG and GenMol, 250 calls | `diagnostics/T4_FROZEN_RESULT_v1.json` | **exact**, 30/30 |
| **Table 2** — PMO-1K final top-10 | 22 objectives × 3 seeds | `diagnostics/pmo_ablation_frozen_v1/inputs/pmo_1k_final.json` | 21/22, mean **exact** |
| **Table 11** — PMO-1K AUC-Top10 | same runs | same file | 21/22, mean **exact** |

### T4 (Table 3) — exact

All 30 cells match. COMPOSE is best in **10 of 15** at δ=0.4 and **13 of 15** at
δ=0.6, giving the abstract's **23 of 30**.

**One printed value is not a COMPOSE result.** FA7 seed 1 at δ=0.6 prints
**−6.4**, which is that cell's *seed score*: COMPOSE returned no feasible
molecule improving on the starting lead, and IVG wins the cell at −7.7. The
artifact records the cell as blank, which is correct; the reproducer surfaces
the substitution rather than silently filling it.

The paper states a **0.70 kcal/mol** docking replicate noise floor. A separate
measurement in this repo found a **1.3 kcal/mol** spread across three runs of one
molecule, because `obabel --gen3D` takes no seed while `qvina02` does. Read the
win counts and the aggregate; do not defend a single cell's 0.2–0.7 margin.

### PMO (Tables 2 and 11) — 42 of 44 values exact, both means exact

Means reproduce to the printed digits: **0.563** final top-10 and **0.482**
AUC-Top10 over the 22-objective protocol (valsartan SMARTS excluded).

Two details a reader needs:

- **Five objectives carry a fourth seed** in the artifact — albuterol,
  celecoxib, mestranol, thiothixene, troglitazone. The paper reports three.
  Dropping the seed whose id ends `983` (the `noprescreen_v2_rediscovery`
  replicate) reproduces the published mean *and* standard deviation on all five.
- **JNK3 differs by 0.001** on both metrics (artifact 0.246 / 0.210 against
  published 0.245 / 0.208). A fourth JNK3 run exists
  (`pmo_learned_trio_v1`, seed 20269928), so the committed artifact holds a
  different third seed than the paper used. The headline means are unaffected.
  This is the single known gap between the artifacts and the printed tables.

---

## What is *not* reproducible from this repository

Stated plainly, because the paper's reproducibility statement says the same
thing: *"a fresh checkout alone is insufficient to recover the reported
measurements."*

| Not here | Why | What survives |
|---|---|---|
| Raw PMO campaign rounds | live Modal volumes across three profiles | the per-seed reductions above |
| T4 per-round docking history | per-cell Modal volumes | per-cell best + SMILES |
| QuickVina2 binary, receptors, boxes | baked into the Modal image | the docked scores |
| TDC oracle pickles (drd2/gsk3b/jnk3), 71 MB | gitignored; PyTDC downloads on demand | — |
| Reference checkpoint `R_θ` | Modal artifact volume | — |

`python3 tools/preservation_inventory.py` reports what git does *not* protect
in the working tree, with hashes, and classifies each unprotected file as
regenerable-by-a-named-recipe or genuinely at risk.

---

## Two chemistry kernels, not one

The paper pins **rdkit 2024.3.5** (with torch 2.4.0, numpy 1.26.4, scipy 1.13.1)
for training and sampling. The **PMO runtime is different**: its image installs
`PyTDC==1.1.15 --no-deps` alongside **rdkit 2023.9.6**, because PyTDC pins
`rdkit<2024.3.1`. A PMO number computed locally under 2024.3.5 therefore needs a
parity statement. Neither reproduction command on this page depends on either
kernel — they read committed JSON.

---

## Other experiments in this repository

Per-benchmark guides, including how to *run* new work rather than verify
published numbers:

- [`experiments/t4/`](../t4/) — T4 task guide, environment, input manifest
- [`experiments/pmo/`](../pmo/) — PMO task guide, information boundary, reading hazards
- [`experiments/INDEX.md`](../INDEX.md) — every runnable entrypoint, generated from source
- [`experiments/region_resampling/`](../region_resampling/) — a campaign manifest

**A separate NeurIPS workshop package** (`paper_gem_neurips2026`) reports a
*different* T4 experiment: GenMol/RetMol/GraphGA at 500 calls per cell, backed by
`diagnostics/t4_combined_table.json` and reproduced with
`python3 tools/reproduce_t4_table.py --submitted`. Its numbers are not the ICLR
paper's and the two must not be mixed.
