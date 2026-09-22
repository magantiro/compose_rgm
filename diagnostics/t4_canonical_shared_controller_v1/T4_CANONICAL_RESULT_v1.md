# T4 canonical shared controller — result

Run id: `3960cba7d11e2d9dddcbe0a56d8fd886373d59a6a654a48be18ecf4791db3721`  
Controller identity: `16a6647bc6ad0c621fe3894d014b7dd3f9424c1dc2e67828f778fdbcfd9d234d`  
Reconciled charged calls: **5160** (from immutable round locks, not checkpoints)

> Docking is not reproducible across runs: qvina02 is seeded, `obabel --gen3D` is not, and one T4 seed molecule has been measured at -7.5 / -8.30 / -8.8 across three runs. Every arm-vs-arm comparison below is WITHIN this run and is sound. The InVirtuoGen column is read-only context from a different run and a different docking invocation; per-cell margins against it are inside the noise, and only the aggregate is worth reading.

## Arm summary

| arm | cells | complete | exhausted | charged calls | scored | sum best | expansion rounds | invalid-chemistry oracle calls |
|---|---|---|---|---|---|---|---|---|
| A | 15 | 0 | 2 | 312 | 2 | -16.3 | 0 | 0 |
| B | 15 | 3 | 4 | 1443 | 7 | -68.7 | 0 | 0 |
| C | 30 | 10 | 1 | 3405 | 11 | -123.3 | 29 | 0 |

## Canonical controller (arm C), all 30 cells

A best in (parentheses) is a RUNNING cell's current incumbent from its checkpoint, not a final result.

| target | seed | delta | status | best | root | IVG (other run) | calls | rounds | expansions | eligible/1k draws |
|---|---|---|---|---|---|---|---|---|---|---|
| 5HT1B | 1 | 0.6 | complete_budget | -13.1 | 1.9 | -12.4 | 250 | 22 | 0 | 33.93 |
| 5HT1B | 2 | 0.6 | running | (-8.5) |  | -12.0 | 30 | 3 | 1 | 13.12 |
| 5HT1B | 3 | 0.6 | candidate_exhaustion | -9.8 | -9.8 | -10.6 | 1 | 0 | 1 | None |
| BRAF | 1 | 0.6 | running | (-9.9) |  | -9.7 | 88 | 12 | 7 | 6.71 |
| BRAF | 2 | 0.6 | running | (-10.6) |  | -10.4 | 49 | 4 | 0 | 41.13 |
| BRAF | 3 | 0.6 | running | (-10.7) |  | -10.3 | 190 | 20 | 3 | 11.36 |
| FA7 | 1 | 0.6 | running | (-8.1) |  | -7.7 | 20 | 6 | 6 | 3.67 |
| FA7 | 2 | 0.6 | running | (-7.5) |  | -7.5 | 215 | 23 | 4 | 9.95 |
| FA7 | 3 | 0.6 | running | (-8.4) |  | -7.4 | 48 | 8 | 7 | 5.63 |
| JAK2 | 1 | 0.6 | complete_budget | -10.1 | -7.6 | -9.7 | 250 | 21 | 0 | 232.38 |
| JAK2 | 2 | 0.6 | complete_budget | -11.4 | -8.1 | -10.4 | 250 | 21 | 0 | 93.49 |
| JAK2 | 3 | 0.6 | complete_budget | -10.8 | -8.6 | -10.3 | 250 | 23 | 0 | 18.02 |
| PARP1 | 1 | 0.6 | complete_budget | -11.0 | -7.3 | -12.3 | 250 | 21 | 0 | 143.3 |
| PARP1 | 2 | 0.6 | complete_budget | -11.0 | -7.8 | -11.7 | 250 | 21 | 0 | 72.62 |
| PARP1 | 3 | 0.6 | running | (-11.3) |  | -10.7 | 205 | 17 | 0 | 53.42 |
| 5HT1B | 1 | 0.4 | not_started |  |  | -13.3 | 1 | 0 | 0 | None |
| 5HT1B | 2 | 0.4 | not_started |  |  | -12.0 | 1 | 0 | 0 | None |
| 5HT1B | 3 | 0.4 | not_started |  |  | -10.9 | 1 | 0 | 0 | None |
| BRAF | 1 | 0.4 | not_started |  |  | -10.1 | 1 | 0 | 0 | None |
| BRAF | 2 | 0.4 | not_started |  |  | -10.8 | 1 | 0 | 0 | None |
| BRAF | 3 | 0.4 | not_started |  |  | -10.6 | 1 | 0 | 0 | None |
| FA7 | 1 | 0.4 | running | (-8.8) |  | -8.4 | 49 | 4 | 0 | 119.66 |
| FA7 | 2 | 0.4 | complete_budget | -8.2 | -6.3 | -8.9 | 250 | 21 | 0 | 83.76 |
| FA7 | 3 | 0.4 | not_started |  |  | -8.0 | 1 | 0 | 0 | None |
| JAK2 | 1 | 0.4 | not_started |  |  | -10.2 | 1 | 0 | 0 | None |
| JAK2 | 2 | 0.4 | not_started |  |  | -10.5 | 1 | 0 | 0 | None |
| JAK2 | 3 | 0.4 | not_started |  |  | -10.2 | 1 | 0 | 0 | None |
| PARP1 | 1 | 0.4 | complete_budget | -12.3 | -7.3 | -14.1 | 250 | 21 | 0 | 257.76 |
| PARP1 | 2 | 0.4 | complete_budget | -12.6 | -7.8 | -13.4 | 250 | 21 | 0 | 254.18 |
| PARP1 | 3 | 0.4 | complete_budget | -13.0 | -8.2 | -9.0 | 250 | 21 | 0 | 228.27 |

## A/B/C ablation at delta = 0.6 (same cells, same seeds, same run)

A vs B isolates coordinated structural programs. B vs C isolates adaptive use of structural support.

| target | seed | A local | B coordinated | C adaptive | B-A | C-B | C expansions |
|---|---|---|---|---|---|---|---|
| 5HT1B | 1 |  |  | -13.1 |  |  | 0 |
| 5HT1B | 2 | -9.4 |  |  |  |  | 1 |
| 5HT1B | 3 |  | -9.8 | -9.8 |  | +0.0 | 1 |
| BRAF | 1 |  |  |  |  |  | 7 |
| BRAF | 2 |  |  |  |  |  | 0 |
| BRAF | 3 |  |  |  |  |  | 3 |
| FA7 | 1 | -6.9 | -7.6 |  | -0.7 |  | 6 |
| FA7 | 2 |  | -7.6 |  |  |  | 4 |
| FA7 | 3 |  | -8.5 |  |  |  | 7 |
| JAK2 | 1 |  |  | -10.1 |  |  | 0 |
| JAK2 | 2 |  |  | -11.4 |  |  | 0 |
| JAK2 | 3 |  |  | -10.8 |  |  | 0 |
| PARP1 | 1 |  | -12.2 | -11.0 |  | +1.2 | 0 |
| PARP1 | 2 |  | -12.2 | -11.0 |  | +1.2 | 0 |
| PARP1 | 3 |  | -10.8 |  |  |  | 0 |
| **mean** | | | | | **-0.70** | **+0.80** | |

Negative is better (docking score). A negative B-A means the coordinated lanes helped; a negative C-B means the adaptive rule helped.

## Mechanism telemetry (per arm, summed over cells)

| arm | proposal draws | eligible pool rows | eligible/1k draws | mean depth | max regions | atoms created | atoms deleted | ring-family candidates | worker failures |
|---|---|---|---|---|---|---|---|---|---|
| A | 9888 | 1274 | 128.84 | 1.31 | 2 | 1519 | 3164 | 208 | 1 |
| B | 144480 | 9273 | 64.18 | 1.5 | 2 | 18223 | 34422 | 984 | 1 |
| C | 339264 | 35393 | 104.32 | 1.6 | 2 | 63849 | 157937 | 3966 | 6 |

## Frontier-improvement attribution (synthesis-time lane tag)

| arm | lane | improvements |
|---|---|---|
| A | shallow | 32 |
| A | unattributed | 10 |
| B | structured | 37 |
| B | shallow | 33 |
| B | anchored_replacement | 17 |
| B | unattributed | 15 |
| C | shallow | 48 |
| C | structured | 48 |
| C | anchored_replacement | 34 |
| C | unattributed | 20 |

## Built-in replicate control: the noise floor of this pipeline

Arms B and C share a vocabulary and a seed and differ only by the adaptive rule. On a cell where that rule NEVER FIRED they ran an identical algorithm from an identical stream, so any difference between them is produced by the oracle alone (`obabel --gen3D` is unseeded). These pairs are a free replicate experiment, and their spread bounds what a per-cell margin can mean.

| cell | B best | C best | \|B-C\| | C expansions |
|---|---|---|---|---|
| `parp1_0_d06` | -12.2 | -11.0 | 1.2 | 0 |
| `parp1_1_d06` | -12.2 | -11.0 | 1.2 | 0 |
| **n=2** | | | **median 1.2, max 1.2** | |

**Read every per-cell margin in this table against a noise floor of about 1.2 kcal/mol.** Differences below it are not evidence of anything. The comparisons that ARE sound are the structural ones -- whether a cell completed its budget or exhausted, and how many calls it spent -- and the aggregate over cells.

## Expansion trigger census (arm C)

- rounds that triggered the expansion: **29**
- distinct eligible endpoints the expansion added: **161**
- stop reasons: `{"ladder_exhausted": 3, "reached_target": 23, "wall_clock": 3}`

