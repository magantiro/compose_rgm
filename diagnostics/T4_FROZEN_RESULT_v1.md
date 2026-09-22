# T4 — FROZEN RESULT (authoritative)

**FROZEN — this is the authoritative T4 result. Do not re-derive from a builder.**

- Comparator: **InVirtuoGen (IVG), no-prescreen**
- Budget: **COMPOSE 250 charged docking calls per cell vs InVirtuoGen 1000**
- Objective: QuickVina2 docking of the best eligible lead, lower is better, subject to QED>=0.6, SA<=4.0, Tanimoto>=delta to the seed
- Frozen: 2026-09-22T03:59:40.736075+00:00
- Artifact: `diagnostics/T4_FROZEN_RESULT_v1.json` (payload sha256 acaf7dad784cf0ae…)

> Do NOT re-derive this from `scripts/t4_ivg_convention_table.py`. That builder pins one
> run per volume, and work migrates to a new `run_id` after a resume, so it silently
> reports an empty run as a cell's result. This file supersedes it.


## δ = 0.4 — coverage **15/15**

| Target | Seed | COMPOSE | IVG | gap | calls | source |
|---|---|---|---|---|---|---|
| PARP1 | 1 | -11.9 | -14.1 | +2.2 | 73 | panel |
| PARP1 | 2 | -12.3 | -13.4 | +1.1 | 81 | panel |
| PARP1 | 3 | **-14.1** | -9.0 | -5.1 | 129 | panel |
| FA7 | 1 | **-9.3** | -8.4 | -0.9 | 89 | panel |
| FA7 | 2 | **-9.4** | -8.9 | -0.5 | 217 | panel |
| FA7 | 3 | **-9.6** | -8.0 | -1.6 | 105 | RESCUE |
| 5HT1B | 1 | -12.4 | -13.3 | +0.9 | 130 | panel |
| 5HT1B | 2 | -11.9 | -12.0 | +0.1 | 161 | panel |
| 5HT1B | 3 | **-12.5** | -10.9 | -1.6 | 249 | RESCUE |
| BRAF | 1 | **-12.2** | -10.1 | -2.1 | 77 | panel |
| BRAF | 2 | -10.5 | -10.8 | +0.3 | 113 | panel |
| BRAF | 3 | **-10.9** | -10.6 | -0.3 | 73 | panel |
| JAK2 | 1 | **-10.4** | -10.2 | -0.2 | 225 | panel |
| JAK2 | 2 | **-11.3** | -10.5 | -0.8 | 145 | panel |
| JAK2 | 3 | **-10.7** | -10.2 | -0.5 | 129 | panel |
| **SUM** | | **-169.4** | **-160.4** | **-9.0** | | 10/15 wins, mean **-0.600** |

## δ = 0.6 — coverage **14/15**

| Target | Seed | COMPOSE | IVG | gap | calls | source |
|---|---|---|---|---|---|---|
| PARP1 | 1 | **-13.6** | -12.3 | -1.3 | 246 | panel |
| PARP1 | 2 | **-12.7** | -11.7 | -1.0 | 244 | panel |
| PARP1 | 3 | **-11.3** | -10.7 | -0.6 | 249 | panel |
| FA7 | 1 | — | -7.7 | — | 0 | panel |
| FA7 | 2 | **-7.8** | -7.5 | -0.3 | 249 | panel |
| FA7 | 3 | **-8.5** | -7.4 | -1.1 | 40 | RESCUE |
| 5HT1B | 1 | **-13.3** | -12.4 | -0.9 | 217 | panel |
| 5HT1B | 2 | -11.3 | -12.0 | +0.7 | 222 | panel |
| 5HT1B | 3 | **-10.8** | -10.6 | -0.2 | 126 | RESCUE |
| BRAF | 1 | -9.0 | -9.7 | +0.7 | 33 | RESCUE |
| BRAF | 2 | **-11.0** | -10.4 | -0.6 | 89 | RESCUE |
| BRAF | 3 | **-11.5** | -10.3 | -1.2 | 104 | panel |
| JAK2 | 1 | **-10.6** | -9.7 | -0.9 | 97 | panel |
| JAK2 | 2 | **-10.8** | -10.4 | -0.4 | 89 | panel |
| JAK2 | 3 | **-10.8** | -10.3 | -0.5 | 113 | panel |
| **SUM** | | **-153.0** | **-145.4** | **-7.6** | | 12/14 wins, mean **-0.543** |

## Correction applied to the previous table

**5HT1B seed 3, delta 0.4: -12.3 → -12.5.** Round locks record a molecule only once it becomes a PARENT, so the winner scored in the final round never appears in a lock. The completed run archive (249 scored molecules, status complete_budget) holds -12.5 as its best, and that molecule was independently re-scored under the pinned T4 kernel and passes every gate.

`Nc1nncn1C1=CC2C(CCCN3CCN(C(=O)Oc4ccc5c(c4)CCC5)CC3)=CN=C2C=C1`

Verified under the pinned T4 kernel: similarity **0.4000**, QED 0.6736, SA 3.9776, 36 heavy atoms. similarity sits EXACTLY on the 0.4 bound; it is eligible only because the shipped comparator is non-strict.
Round-lock lower bound was -12.3.

## The one blank cell

**FA7 seed 1, delta 0.6 — SCOPED NEGATIVE, deliberate.** IVG has -7.7.

Reachable but not reached. All five stored witnesses pass the production Fiber.check and a production configuration generated one eligible endpoint, but only ONE across 3 seeds x 480 draws at a single seed -- not reproducible seed-to-seed, so docking calls were declined. The barrier is architectural: expand does not gate the molecule a proposal law conditions (recovered_fraction = 0.0000 over 150 draws).

Record: `diagnostics/t4_fa7_0_scoped_negative.md (commit 01baa8a6)`

**Open option:** Dock the one credible eligible endpoint (COC(=O)N(CCC(C)C)Cc1ccc2ccc(C(=N)N)cc2c1, sim 0.6610 / QED 0.6264 / SA 2.331, retains the amidine pharmacophore) and mark it a single-witness rescue.

## Rescue phases

Cells marked `RESCUE` ran under their own versioned contract with a proposal law that
DIFFERS from the v1 arm. Their claim boundaries forbid splicing them into an
"unchanged-v1" panel without naming them. They are named here.


## Successor in progress (v1 remains authoritative)

A **v2** panel is being prepared from ONE FROZEN ALGORITHM — primary controller plus a generic
zero-support fallback available to all 15 cells, sealed prospectively before any docking
(branch `t4-generic-fallback-20260922`). Its purpose is to remove the `RESCUE` column: a row
produced by the declared algorithm is an ordinary result, and the fallback is described once in
Methods.

**Until v2 is sealed and run, the numbers above are the ones to quote.** Do not mix v1 and v2 rows.
