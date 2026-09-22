# Canonical T4 preflight — zero oracle calls, all 30 cells

150 (arm, cell, lane) probes through the PRODUCTION proposal worker, 150/150 succeeded, 3569 s wall, **0 oracle calls**.

Controller identity across all 150 probes: `8c6f24344e6ccb43fbf61a650f33e37ebca223774b28fcaa83408494884d70db` — **one value**, which is what makes "the same controller ran on every cell" checkable rather than asserted.

Every probe ran the real `expand` on the real `Fiber`, at the contract's own draw count, with the region and completion laws resolved and PROVEN CONSUMED. A lane returning nothing is a measurement, not a failure: the complete controller is what has to work, not every channel from every state.

## Per-lane cost and root yield (at the probe's 480/512/480 draws)

| lane | probes | median s | max s | median s/draw | median eligible | empty lanes |
|---|---|---|---|---|---|---|
| `shallow` | 60 | 864 | 1525 | 1.800 | 42 | 3 / 60 |
| `anchored_replacement` | 45 | 336 | 699 | 0.657 | 0 | 27 / 45 |
| `structured` | 45 | 994 | 2023 | 2.071 | 21 | 8 / 45 |

`anchored_replacement` returns ZERO from 27 of 45 roots — every braf, fa7 and 5ht1b cell — and is productive only on jak2 and parp1. It was RETAINED: the preflight measures each cell's ROOT, not the parents a campaign visits, so dropping a declared program family on a root-state probe would select the vocabulary on evidence that does not cover the states it runs in. Round one of the scored run already contradicts the root reading on jak2/parp1 (53, 27, 181, 94, 71 eligible).

## Root eligible yield per cell, delta = 0.6

| cell | shallow | anchored | structured | total |
|---|---|---|---|---|
| `5ht1b_0_d06` | 51 | 0 | 42 | **93** |
| `5ht1b_1_d06` | 27 | 0 | 20 | **47** |
| `5ht1b_2_d06` | 0 | 0 | 0 | **0** |
| `braf_0_d06` | 13 | 0 | 1 | **14** |
| `braf_1_d06` | 26 | 0 | 6 | **32** |
| `braf_2_d06` | 22 | 0 | 3 | **25** |
| `fa7_0_d06` | 1 | 0 | 0 | **1** |
| `fa7_1_d06` | 19 | 0 | 7 | **26** |
| `fa7_2_d06` | 9 | 0 | 0 | **9** |
| `jak2_0_d06` | 181 | 209 | 192 | **582** |
| `jak2_1_d06` | 175 | 65 | 165 | **405** |
| `jak2_2_d06` | 28 | 0 | 21 | **49** |
| `parp1_0_d06` | 100 | 23 | 60 | **183** |
| `parp1_1_d06` | 53 | 24 | 70 | **147** |
| `parp1_2_d06` | 41 | 72 | 55 | **168** |

delta=0.6 median total 47, cells with zero: **1 of 15**.  
delta=0.4 median total 312, cells with zero: 0 of 15.

**The headline preflight finding.** On the earlier panel FIVE delta=0.6 cells terminated at `candidate_exhaustion` on round one — braf_0, braf_1, fa7_0, fa7_2 and 5ht1b_2. Under this controller only **one** root (5ht1b_2) produces nothing at all; braf_0 yields 14, braf_1 32, fa7_2 9, fa7_0 1. The difference is the merged completion law plus the `structured` lane. This is root yield, INFERRED to bear on campaign outcome and not a scored result — but it says the "exhausted cells" framing was a property of the proposal law rather than of those cells.

## A note on `root_passes_own_gate`

20 of 30 roots do NOT satisfy the endpoint gate themselves (QED >= 0.6, SA <= 4). That is expected and correct: the gate constrains molecules the controller RETURNS, not the lead it is given. It is recorded per row so nobody later reads it as a defect.

