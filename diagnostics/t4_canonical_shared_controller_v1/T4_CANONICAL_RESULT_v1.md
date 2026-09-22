# T4 canonical shared controller — result

Run id: `3960cba7d11e2d9dddcbe0a56d8fd886373d59a6a654a48be18ecf4791db3721`  
Controller identity: `16a6647bc6ad0c621fe3894d014b7dd3f9424c1dc2e67828f778fdbcfd9d234d`  
Reconciled charged calls: **2179** (from immutable round locks, not checkpoints)

> Docking is not reproducible across runs: qvina02 is seeded, `obabel --gen3D` is not, and one T4 seed molecule has been measured at -7.5 / -8.30 / -8.8 across three runs. Every arm-vs-arm comparison below is WITHIN this run and is sound. The InVirtuoGen column is read-only context from a different run and a different docking invocation; per-cell margins against it are inside the noise, and only the aggregate is worth reading.

## Arm summary

| arm | cells | complete | exhausted | charged calls | scored | sum best | expansion rounds | invalid-chemistry oracle calls |
|---|---|---|---|---|---|---|---|---|
| A | 15 | 0 | 0 | 0 | 0 | 0.0 | 0 | 0 |
| B | 15 | 0 | 1 | 302 | 1 | -7.6 | 0 | 0 |
| C | 30 | 0 | 1 | 1877 | 1 | -9.8 | 9 | 0 |

## Canonical controller (arm C), all 30 cells

A best in (parentheses) is a RUNNING cell's current incumbent from its checkpoint, not a final result.

| target | seed | delta | status | best | root | IVG (other run) | calls | rounds | expansions | eligible/1k draws |
|---|---|---|---|---|---|---|---|---|---|---|
| 5HT1B | 1 | 0.6 | running | (-13.1) |  | -12.4 | 109 | 9 | 0 | 49.4 |
| 5HT1B | 2 | 0.6 | running | (-8.5) |  | -12.0 | 30 | 3 | 1 | 13.12 |
| 5HT1B | 3 | 0.6 | candidate_exhaustion | -9.8 | -9.8 | -10.6 | 1 | 0 | 1 | None |
| BRAF | 1 | 0.6 | running | (-9.0) |  | -9.7 | 37 | 5 | 2 | 7.35 |
| BRAF | 2 | 0.6 | running | (-10.6) |  | -10.4 | 49 | 4 | 0 | 41.13 |
| BRAF | 3 | 0.6 | running | (-10.7) |  | -10.3 | 118 | 10 | 0 | 16.33 |
| FA7 | 1 | 0.6 | running | (-7.8) |  | -7.7 | 3 | 2 | 2 | 2.31 |
| FA7 | 2 | 0.6 | running | (-7.3) |  | -7.5 | 109 | 10 | 1 | 13.42 |
| FA7 | 3 | 0.6 | running | (-8.4) |  | -7.4 | 13 | 2 | 2 | 8.33 |
| JAK2 | 1 | 0.6 | running | (-10.1) |  | -9.7 | 121 | 10 | 0 | 262.58 |
| JAK2 | 2 | 0.6 | running | (-11.4) |  | -10.4 | 97 | 8 | 0 | 123.08 |
| JAK2 | 3 | 0.6 | running | (-10.3) |  | -10.3 | 128 | 11 | 0 | 23.8 |
| PARP1 | 1 | 0.6 | running | (-10.8) |  | -12.3 | 133 | 11 | 0 | 157.35 |
| PARP1 | 2 | 0.6 | running | (-10.9) |  | -11.7 | 157 | 13 | 0 | 91.06 |
| PARP1 | 3 | 0.6 | running | (-11.3) |  | -10.7 | 145 | 12 | 0 | 61.65 |
| 5HT1B | 1 | 0.4 | not_started |  |  | -13.3 | 1 | 0 | 0 | None |
| 5HT1B | 2 | 0.4 | not_started |  |  | -12.0 | 1 | 0 | 0 | None |
| 5HT1B | 3 | 0.4 | not_started |  |  | -10.9 | 1 | 0 | 0 | None |
| BRAF | 1 | 0.4 | not_started |  |  | -10.1 | 1 | 0 | 0 | None |
| BRAF | 2 | 0.4 | not_started |  |  | -10.8 | 1 | 0 | 0 | None |
| BRAF | 3 | 0.4 | not_started |  |  | -10.6 | 1 | 0 | 0 | None |
| FA7 | 1 | 0.4 | running | (-8.8) |  | -8.4 | 49 | 4 | 0 | 119.66 |
| FA7 | 2 | 0.4 | running | (-8.1) |  | -8.9 | 133 | 11 | 0 | 95.87 |
| FA7 | 3 | 0.4 | not_started |  |  | -8.0 | 1 | 0 | 0 | None |
| JAK2 | 1 | 0.4 | not_started |  |  | -10.2 | 1 | 0 | 0 | None |
| JAK2 | 2 | 0.4 | not_started |  |  | -10.5 | 1 | 0 | 0 | None |
| JAK2 | 3 | 0.4 | not_started |  |  | -10.2 | 1 | 0 | 0 | None |
| PARP1 | 1 | 0.4 | running | (-12.3) |  | -14.1 | 145 | 12 | 0 | 308.02 |
| PARP1 | 2 | 0.4 | running | (-12.5) |  | -13.4 | 145 | 12 | 0 | 301.31 |
| PARP1 | 3 | 0.4 | running | (-13.0) |  | -9.0 | 145 | 12 | 0 | 237.73 |

## A/B/C ablation at delta = 0.6 (same cells, same seeds, same run)

A vs B isolates coordinated structural programs. B vs C isolates adaptive use of structural support.

| target | seed | A local | B coordinated | C adaptive | B-A | C-B | C expansions |
|---|---|---|---|---|---|---|---|
| 5HT1B | 1 |  |  |  |  |  | 0 |
| 5HT1B | 2 |  |  |  |  |  | 1 |
| 5HT1B | 3 |  |  | -9.8 |  |  | 1 |
| BRAF | 1 |  |  |  |  |  | 2 |
| BRAF | 2 |  |  |  |  |  | 0 |
| BRAF | 3 |  |  |  |  |  | 0 |
| FA7 | 1 |  | -7.6 |  |  |  | 2 |
| FA7 | 2 |  |  |  |  |  | 1 |
| FA7 | 3 |  |  |  |  |  | 2 |
| JAK2 | 1 |  |  |  |  |  | 0 |
| JAK2 | 2 |  |  |  |  |  | 0 |
| JAK2 | 3 |  |  |  |  |  | 0 |
| PARP1 | 1 |  |  |  |  |  | 0 |
| PARP1 | 2 |  |  |  |  |  | 0 |
| PARP1 | 3 |  |  |  |  |  | 0 |

Negative is better (docking score). A negative B-A means the coordinated lanes helped; a negative C-B means the adaptive rule helped.

## Mechanism telemetry (per arm, summed over cells)

| arm | proposal draws | eligible pool rows | eligible/1k draws | mean depth | max regions | atoms created | atoms deleted | ring-family candidates | worker failures |
|---|---|---|---|---|---|---|---|---|---|
| B | 25632 | 2551 | 99.52 | 1.45 | 2 | 5092 | 8161 | 421 | 0 |
| C | 168288 | 21622 | 128.48 | 1.62 | 2 | 41633 | 86778 | 2533 | 2 |

## Frontier-improvement attribution (synthesis-time lane tag)

| arm | lane | improvements |
|---|---|---|
| B | structured | 12 |
| B | shallow | 8 |
| B | anchored_replacement | 8 |
| B | unattributed | 5 |
| C | shallow | 41 |
| C | structured | 40 |
| C | anchored_replacement | 32 |
| C | unattributed | 20 |

## Expansion trigger census (arm C)

- rounds that triggered the expansion: **9**
- distinct eligible endpoints the expansion added: **52**
- stop reasons: `{"ladder_exhausted": 1, "reached_target": 6, "wall_clock": 2}`

