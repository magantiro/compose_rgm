# Reproduction handoff — Table 3, ICLR 2027 submission

Read-only. Nothing was run, launched, re-pinned or edited to produce this.
Scope is **Table 3 only** of
`/Users/rmaganti/Desktop/19340_COMPOSE_Molecular_Genera.pdf`
(sha256 `a609e9dc9c43ad22d8b4fa2a…`, 35 pages, verified this session against the
rendered table — **not** the same file as
`COMPOSE_ICLR_2027_zip5_appendix_results/main.pdf`, which differs).

## 0. Three T4 experiments exist. This one is the 250-call IVG/GenMol table.

| | Comparators | Budget | Artifact | Is it Table 3? |
|---|---|---|---|---|
| **This one** | IVG, GenMol | 250 vs 1,000 | `diagnostics/T4_FROZEN_RESULT_v1.json` | **yes** |
| Workshop | GenMol, RetMol, GraphGA | 500 vs 3,000 | `diagnostics/t4_combined_table.json` | no |
| Replicate panel | — | 250 × 2 replicates | `diagnostics/t4_unified_controller_v1/launches/` | no |

Per-cell values disagree across all three **by construction**. Confirmed:
PARP1 seed 1 at δ=0.4 is −11.9 here and −10.733 in the workshop table.

Cell indexing differs too. Paper seeds 1/2/3 per target are workshop seeds
`s3/s4/s5` for FA7, `s0/s1/s2` for PARP1, etc. **FA7 paper-seed 1 = workshop
`fa7_s3`.**

## 1. Per-cell provenance

Source of the volume/contract map:
`scripts/t4_ivg_convention_table.py` (its `CAMPAIGNS` and `RESCUES` dicts).
Executable `delta` and payload hash verified this session by loading each
contract.

| protein, δ | Modal volume | contract | exec. δ | payload sha256 |
|---|---|---|---|---|
| parp1 0.4 | `compose-t4-held-target-distilled-parp1-d04-250` | `configs/t4_held_target_distilled_parp1_d04_250.json` | 0.4 | `b7cf9e740698f5af` |
| braf 0.4 | `…-braf-d04-250` | `…braf_d04_250.json` | 0.4 | `cd37ecb0b51045fc` |
| 5ht1b 0.4 | `…-5ht1b-d04-250` | `…5ht1b_d04_250.json` | 0.4 | `cdb3c46f46c7b19f` |
| fa7 0.4 | `…-fa7-d04-250` | `…fa7_d04_250.json` | 0.4 | `d2276882179afb13` |
| **jak2 0.4** | `compose-t4-held-target-distilled-jak2-d06-250` | `configs/t4_held_target_distilled_jak2_d06_250.json` | **0.4** | `3a43ac863683ccfb` |
| parp1 0.6 | `…-parp1-d06-250` | `…parp1_d06_250.json` | 0.6 | `bd68d29c335bc443` |
| braf 0.6 | `…-braf-d06-250` | `…braf_d06_250.json` | 0.6 | `0df84d667ece4c07` |
| 5ht1b 0.6 | `…-5ht1b-d06-250` | `…5ht1b_d06_250.json` | 0.6 | `6959ae96cc0387f8` |
| fa7 0.6 | `…-fa7-d06-250` | `…fa7_d06_250.json` | 0.6 | `cb34d1d1ecfec41d` |
| **jak2 0.6** | `compose-t4-held-target-jak2-true-d06-250` | `configs/t4_held_target_distilled_jak2_true_d06_250.json` | 0.6 | `df3c113215e84588` |

**The jak2 naming trap, resolved.** The volume and file named `…jak2-d06-250`
carry **delta 0.4** and are the δ=0.4 campaign. The δ=0.6 jak2 cells come from
`…jak2-true-d06-250`. `t4_ivg_convention_table.py` maps this correctly and
comments it. **Always read `delta` from the contract payload, never from a
volume or file name** — a sibling contract once shipped `delta: 0.4` under a
`claim_boundary` string saying 0.6.

Each contract also pins: `docking_seed: 20260918`, `docking_box` (2 entries),
`evaluator_sha256`, `runtime_inputs_sha256` (13 files),
`charged_calls_per_cell: 250`, `total_charged_call_ceiling: 750`.

### Rescue arms — the `support_expansion` rows

`RESCUES` in the same file:

```
("5ht1b", 0.4) compose-t4-5ht1b2-protonation-rescue-d04
("5ht1b", 0.6) compose-t4-5ht1b2-protonation-rescue-d06
("fa7",   0.4) compose-t4-region-repair-rescue-fa7-d04
("fa7",   0.6) compose-t4-region-repair-rescue-fa7-d06
("braf",  0.6) compose-t4-region-repair-rescue-braf-d06
```

These are a **named separate phase**; their own `claim_boundary` forbids
splicing them into the unchanged panel.

## 2. Which rows are `panel` and which are `support_expansion`

Read directly from `T4_FROZEN_RESULT_v1.json`. **δ=0.4: 13 panel, 2 expansion.
δ=0.6: 11 panel, 4 expansion.**

| cell | δ | COMPOSE | charged | source |
|---|---|---|---|---|
| 5HT1B s3 | 0.4 | −12.5 | 249 | **support_expansion** |
| FA7 s3 | 0.4 | −9.6 | 105 | **support_expansion** |
| 5HT1B s3 | 0.6 | −10.8 | 126 | **support_expansion** |
| BRAF s1 | 0.6 | −9.0 | 33 | **support_expansion** |
| BRAF s2 | 0.6 | −11.0 | 89 | **support_expansion** |
| FA7 s3 | 0.6 | −8.5 | 40 | **support_expansion** |
| all others | | | | panel |

Charged counts per cell, δ=0.4: 5HT1B 130/161/249, BRAF 77/113/73, FA7 89/217/105,
JAK2 225/145/129, PARP1 73/81/129.
δ=0.6: 5HT1B 217/222/126, BRAF 33/89/104, FA7 **0**/249/40, JAK2 97/89/113,
PARP1 246/244/249.

## 3. FA7 starting molecule 1 at δ=0.6 — why −6.4 is printed

**Established facts.**

1. The frozen artifact has `compose: null`, `gap: null`, `charged_calls: 0`,
   `source: panel` for that cell. **Zero docking calls were ever charged.**
2. Its `blank_cells` entry records the cell as
   **`status: "SCOPED NEGATIVE, deliberate"`** — *"Reachable but not reached… only
   ONE [eligible endpoint] across 3 seeds × 480 draws at a single seed — not
   reproducible seed-to-seed, so docking calls were declined."* It names the one
   undocked candidate (`COC(=O)N(CCC(C)C)Cc1ccc2ccc(C(=N)N)cc2c1`, sim 0.6610 /
   QED 0.6264 / SA 2.331) and records IVG at −7.7.
3. **−6.4 is that cell's seed score.** Confirmed independently in the workshop
   artifact: `t4_combined_table.json` row `fa7_s3_d0.6` carries
   `"seed_score": -6.4`. The paper prints −6.4 in its own *Seed score* column on
   the same row.

**So the value printed in the COMPOSE δ=0.6 column is the seed score, carried
across from the adjacent column.**

**The table-producing rule for that substitution is NOT in any artifact I can
find, and it contradicts the paper's own stated convention.** §I.2 says:
*"Dashes in the published table mark cells for which no feasible molecule is
reported. We carry them through as missing … and do not impute a docking value
for an empty cell."* The paper does use dashes elsewhere in Table 3 (GenMol at
FA7 s3 δ=0.6, BRAF s1 δ=0.6, JAK2 s3 δ=0.6). Two readings are consistent with
the evidence and I cannot distinguish them from artifacts alone:

- **(a) Transcription carry-over** — the seed-score value was copied into the
  COMPOSE column where a dash was intended. This would make the cell an erratum.
- **(b) "COMPOSE returned the starting molecule"** — the seed trivially satisfies
  `sim = 1.0 ≥ δ`, so if it also passes QED ≥ 0.6 and SA ≤ 4 it is a feasible
  return needing no new docking call, which is consistent with `charged_calls: 0`.
  This reading is *not* an imputation, but it is nowhere documented, and the
  frozen artifact encodes the cell as blank rather than as −6.4.

**Consequence either way:** IVG is bolded as the winner of that cell at −7.7, so
the printed −6.4 does not change the 13-of-15 count at δ=0.6. **MISSING: the
script or manual step that emitted Table 3.** `scripts/t4_ivg_convention_table.py`
is explicitly *"SUPERSEDED FOR REPORTING … do not quote this script's output as
the panel"*, and no committed tool converts `T4_FROZEN_RESULT_v1.json` into the
paper's LaTeX.

## 4. The δ=0.6 `sum_gap` reconciliation

Recomputed this session from the artifact's own rows:

```
delta 0.4: stored compose_wins=10  recomputed=10   stored sum_gap=-9.0  recomputed=-9.0   OK
delta 0.6: stored compose_wins=13  recomputed=13   stored sum_gap=-7.6  recomputed=-8.5   MISMATCH
```

Two further independent routes agree with **−8.5**:

```
compose_sum - ivg_sum      = -153.9 - (-145.4) = -8.5
mean_gap * paired_cells    = -0.6071 * 14      = -8.4994
```

So three routes give −8.5 and only the stored `sum_gap` field says −7.6. **The
stored value is wrong and understates COMPOSE's aggregate margin by 0.9
kcal/mol.** The δ=0.4 block is internally consistent.

**Table 3 does not print sums**, so the paper is unaffected. The error matters
only for anyone quoting the artifact's aggregate.

### Stale Markdown summaries — two defects, both in the same file

`diagnostics/T4_FROZEN_RESULT_v1.md`:

- **line 56** — `| **SUM** | | **-153.9** | **-145.4** | **-7.6** | | 12/14 wins, mean **-0.607** |`
  - propagates the wrong `-7.6`
  - states **12/14 wins**, where the JSON's own `compose_wins` is **13** and
    recomputation from rows gives **13**. This is a *second, independent* error
    not present in the JSON.
- **line 123** — repeats *"δ0.6 −7.6 over 14"*.
- line 35 (δ=0.4 SUM, `-169.4 / -160.4 / -9.0`, 10/15 wins) is **correct**.

Both frozen artifacts were left untouched, per instruction.

## 5. Offline table reduction (no docking, no network, no credentials)

Reproduces the COMPOSE column, the provenance split, and the win counts:

```bash
cd <repo>
python3 - <<'PY'
import json
p=json.load(open('diagnostics/T4_FROZEN_RESULT_v1.json'))['payload']
SEED={('FA7',1):-6.4}   # printed value for the zero-call cell; see section 3
for dk in ('0.4','0.6'):
    rows=sorted(p['deltas'][dk]['rows'], key=lambda r:(r['target'],r['seed']))
    best=0
    for r in rows:
        c=r['compose'] if r['compose'] is not None else SEED.get((r['target'],r['seed']))
        base=[b for b in (r['ivg'],) if b is not None]
        if c is not None and base and all(c<=b for b in base): best+=1
        print(f"{r['target']:<7}{r['seed']}  d={dk}  COMPOSE={c}  IVG={r['ivg']}  "
              f"calls={r['charged_calls']}  {r['source']}")
    b=p['deltas'][dk]
    paired=[r for r in rows if r['compose'] is not None and r['ivg'] is not None]
    print(f"  stored sum_gap={b['sum_gap']}  row-derived={round(sum(r['gap'] for r in paired),4)}")
PY
```

The paper's **10 of 15** and **13 of 15** are counts of COMPOSE being best
against **both** IVG and GenMol across all 15 cells. GenMol values are
transcribed from the published table and are **not** in
`T4_FROZEN_RESULT_v1.json` — they must come from the paper or
`docs/invirtuogen_t4_targets.json` / the GenMol source. Verified this session
against the paper's own transcribed cells: 10/15 and 13/15, total 23/30,
matching the abstract.

## 6. Fresh-run recipe (documented, not executed)

Prerequisites, none of which are in this repository:

| Asset | Where |
|---|---|
| Reference checkpoint `R_θ` | Modal volume `compose-v4-artifacts`, `compose-v4-stage3-flexible-graft-3k-1ac6f19-v1/checkpoint.best_so_far.pt` |
| QuickVina2 binary, receptors, docking boxes | baked into the execution image |
| Per-cell Modal volume | table in §1 |
| Modal account + correct profile | volumes exist under more than one profile; `MODAL_PROFILE` is **not** inferable from a volume name |

Environment (core kernel): python 3.11, rdkit 2024.3.5, torch 2.4.0,
numpy 1.26.4, scipy 1.13.1, networkx 3.3.

Launch shape — the per-cell launchers are 9-line env-var wrappers over one
shared implementation, e.g.
`modal_apps/t4_integrated_route_fiber_held_5ht1b_app.py`:

```python
os.environ.setdefault("COMPOSE_HELD_CONTRACT",  "configs/t4_held_target_distilled_5ht1b_d06_250.json")
os.environ.setdefault("COMPOSE_HELD_VOLUME",    "compose-t4-held-target-distilled-5ht1b-d06-250")
os.environ.setdefault("COMPOSE_HELD_RECEPTOR_NAME", "5ht1b")
from modal_apps.t4_integrated_route_fiber_parp1_app import app, main
```

`--detach` is mandatory (`main()` uses `.spawn()`). Expected outputs per cell on
its volume: `<run_id>/launch.json`, `round_NNN_lock.json` per round,
`checkpoint.json`, and `result.json` on completion. A **round lock is
authoritative; a checkpoint is overwritable.**

Expected agreement: **similar, not identical.** `qvina02` is seeded via
`docking_seed`, but `obabel --gen3D` takes no seed; one molecule measured
−7.5 / −8.30 / −8.8 across three runs. The paper states a 0.70 kcal/mol
replicate noise floor. Per-cell margins below that are not resolved.

## 7. Evidence and limitations ledger

| Item asked for | Status | Where / why |
|---|---|---|
| contract per cell | **verified** | §1, loaded and hashed this session |
| executable `delta` per cell | **verified** | all 10 match their cell, incl. the jak2 mislabel |
| contract payload sha256 | **verified** | §1 |
| Modal volume per cell | **verified** | `CAMPAIGNS`/`RESCUES` in `t4_ivg_convention_table.py` |
| `panel` vs `support_expansion` per row | **verified** | §2, from the frozen artifact |
| charged docking count per cell | **verified** | §2 |
| docking seed, box, evaluator hash | **verified** | pinned in each contract |
| FA7 s1 δ=0.6 → −6.4 | **explained, rule MISSING** | §3; value identified as the seed score, emitting rule not found |
| δ=0.6 `sum_gap` | **reconciled** | §4; stored −7.6 wrong, rows give −8.5 |
| stale Markdown | **identified** | §4; `T4_FROZEN_RESULT_v1.md` lines 56, 123 |
| **per-cell run ID** | **MISSING** | frozen rows carry only `{target, seed, compose, ivg, gap, charged_calls, source}`. Run ids live in each volume's `<run_id>/launch.json`, which needs Modal access. |
| **code commit per cell** | **MISSING** | not in the frozen artifact; recoverable from each run's `launch.json` |
| **deployed image digest** | **MISSING** | not recorded in any committed artifact |
| **receptor / QuickVina2 binary hashes** | **MISSING** | `evaluator_sha256` is pinned in the contract but the receptor and binary themselves are image-internal |
| **authoritative per-cell score receipt** | **MISSING locally** | round locks on the per-cell volume are authoritative; the frozen JSON is a reduction of them |
| **launcher command per cell** | **partially verified** | wrapper shape confirmed in §6; exact invocation per cell not recorded in the repo |
| **the Table 3 producing script** | **MISSING** | no committed tool converts the frozen JSON to the paper's LaTeX |

**Caveats on my own work.** The frozen artifacts were opened read-only; nothing
was run, launched, re-pinned or edited. The GenMol column is transcribed in the
paper and was not re-derived. Everything marked MISSING above requires Modal
access under the right profile, and I did not query Modal.
