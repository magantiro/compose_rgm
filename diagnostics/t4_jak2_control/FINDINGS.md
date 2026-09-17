# JAK2: where the gap actually is

Zero oracle calls. Frozen archive only.

## The winners are one-shot large programs, not long lineages

All 92 JAK2 molecules scoring <= -11.0 came from `full146`, the winner-bank arm.

| | edits in the final program | total ancestral edits |
| --- | ---: | ---: |
| the 92 best (<= -11.0) | median **14.0**, p90 17 | median **0.0** |
| worst (> -9.0), n=6,297 | median 2.0 | median 2.0 |

Ancestral 0 means they came straight off the root in a single coordinated program.
This CORRECTS the earlier "winners accumulate 15 ancestral edits" reading, which counted
lineage depth; the 15 was cumulative across generations and is not how the winners were
produced. It also explains why lineage depth measured anti-predictive prospectively
(AUC 0.332): the winners do not use depth.

## Size is necessary and nowhere near sufficient

| program size | autonomous median | autonomous best | bank median | bank best |
| --- | ---: | ---: | ---: | ---: |
| 1-2 | -8.10 | -10.5 | -8.80 | -11.6 |
| 5-7 | -8.20 | -10.2 | -8.30 | -10.8 |
| 12-20 | -8.80 | -10.1 | **-9.70** | **-11.7** |
| corr(size, score) | **-0.059** | | **-0.352** | |

The autonomous arms already emit 16 and 17 edit programs unaided -- so program-size
support is not the binding constraint -- and those programs score -7.2 to -7.8. In the
bank arm the same size band reaches -11.7. Large programs pay off only when they are the
RIGHT large programs.

55 of the 92 winners carry the block label `compiled_complete_transformation`, the bank's
compiled-route channel. Those programs are largely retrieved rather than synthesised, so
"the generator can produce a productive 14-edit program" remains UNPROVEN.

## Raising the module cap does not close it

`MAX_GENERIC_MODULES` is now 8 (default still 3), with counts beyond the declared table
decaying geometrically. Measured on the JAK2-1 root, 700 attempts per cap:

| cap | median primitives | >= 15 primitives | eligible | distinct endpoints |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 4.0 | 20 (2.9%) | 235 | 577 |
| 5 | 5.0 | 44 (6.3%) | 183 | 613 |

Reach into the payoff band doubles and eligibility falls. Given that large autonomous
programs are not better, this lever is not the fix.

## Selection is broken, and that part is solid

Ranking held-out JAK2 sibling transitions -- children of the SAME parent, so parent
quality cannot be read off:

| arm | pick-best top-1 |
| --- | ---: |
| random sibling | 0.404 |
| destination value `V_phi` (route-blind, basin-informed) | 0.449 |
| same, winner-blind | 0.446 |
| `V_phi` on Morgan fingerprints + descriptors | 0.498 |
| sibling-trained transition critic | 0.506 |

Withholding the good region changes nothing (0.446 vs 0.449), so the value was never
using it. Pool enrichment is the operational form of the same failure: over a pool of 800
archive molecules, docking the top 10 by `V_phi` returns -10.57 where the pool contains
-11.48, against -10.45 for a random ten.

CAVEAT, and it matters: those pools are drawn from the historical ARCHIVE, which includes
bank-initialised `full146` rows. This measures that selection fails on known-good
molecules. It does NOT show that the autonomous generator puts -11.48 candidates on the
table; that is a separate, still-open gate.

## What this leaves

Two independent gates, previously conflated:

- SUPPORT -- can the generator autonomously put a productive coordinated JAK2
  transformation on the table? Size support exists; productive-content support is unproven.
- RECOGNITION -- given such a set, can a target-aware controller identify it? Measured,
  and currently no: every selector sits between 0.45 and 0.51 against 0.40 chance.

---

# Allocation replay: would a hierarchical allocator have spent the budget better?

Zero oracle calls. Historical runs replayed causally: at charged call `t` the allocator
sees only outcomes with `query <= t`, and names the structural-hypothesis branch it would
fund next. Scored against the branch that eventually produced that run's best molecule.

## Both allocators fail once the circularity is removed

| allocator | share of budget to the winning branch | vs historical policy |
| --- | ---: | ---: |
| Thompson on frontier-advance rate | 0.001 - 0.460 | 0.00x - 20.9x, WORSE on 6 of 7 |
| extreme-value (branch best-so-far) | 0.189 - 0.938 | 1.8x - 45x on 10 of 11 |

The second looks like a pass and is not one. `branch_best` funds whichever branch holds
the best score so far, and the winning branch is BY DEFINITION the one holding the best
score, so most of that share is exploitation after the winner already appeared. Splitting
on the call at which the winning molecule was first observed:

| run | PRE-discovery share | decisions | POST-discovery share |
| --- | ---: | ---: | ---: |
| v0_braf_1_r0 | **0.000** | 287 | 0.989 |
| v0_5ht1b_0_r0 | **0.000** | 17 | 0.469 |
| v21_braf_1_r0 | **0.000** | 381 | 0.981 |
| v1_braf_1_r0 | **0.000** | 155 | 0.976 |
| v21_5ht1b_0_r0 | **0.000** | 44 | 0.925 |
| v1_5ht1b_0_r0 | **0.000** | 142 | 0.996 |
| v0d06_braf_2_r0 | **0.000** | 48 | 0.000 |
| v21_parp1_0_r0 | **0.000** | 125 | 0.999 |
| v0d06_parp1_0_r0 | **0.000** | 222 | 0.996 |
| v0_jak2_1_r0 | 0.017 | 709 | 0.996 |
| v21_jak2_1_r0 | **0.860** | 293 | 0.972 |

Mean 0.088 against 0.046 for uniform, but that is carried entirely by one run; without
`v21_jak2_1_r0` the pre-discovery share is worse than allocating at random.

GATE VERDICT: FAIL. The allocator does not find the productive branch before that branch
produces its result, so it is not ready for a JAK2 docking budget.

## Why the first allocator was the wrong shape

Frontier advances are too rare for a rate to exist. On the runs Dynamic won, the winning
branch advanced ONCE in 39 calls (BRAF, rate 0.026) and once in 64 (5HT1B, rate 0.016),
and every branch that ever wins does so with a single advance. A Bernoulli rate is then
`1/calls`, which PUNISHES whichever branch has been invested in. Thompson duly chased a
branch at rate 0.143 whose ceiling was -10.1 while the winner reached -11.2.

This is an extreme-value problem, not a rate-estimation problem. That correction is real
and is kept. It is simply not sufficient, because the branch maximum carries no signal
before the branch has produced anything.

## What this rules out

Neither of the two available signals identifies the productive structural direction in
advance:

- STRUCTURE -- every surrogate tried on whole transitions tops out near chance at the
  extreme tail (best 8.2% top-10 against 6.1%).
- SEARCH HISTORY -- within-run docking outcomes do not flag the winning branch before it
  wins, on 9 of 11 runs.

A better bandit cannot fix the second, because the information is not in the statistic.
Anything that claims to allocate toward the win must be checked PRE-discovery; three
separate arms in this work looked strong and collapsed under that split.

## Forensic: the one pre-discovery exception, and what it classifies

`v21_jak2_1_r0` allocated 0.860 of its pre-discovery budget to the winning branch while
nine other runs allocated zero. The cause is not early signal and not leakage.

That branch ALREADY HELD THE INCUMBENT. Its best before discovery was -10.10, the
incumbent standing just before the winner was -10.10 -- the same molecule -- and the
"win" was an incremental -10.10 to -10.7 inside the leading branch. It ranked 1 of 22 by
best-so-far. Checks requested and passed: deleting the winner and every later row leaves
it rank 1 of 22; it ranks 1 at every granularity (lane, lane+family, +scale, full path);
it is not a singleton (70 of 981 calls, 23 before discovery, against 80 for the
most-sampled branch).

Classifying every replayed run by where the winning branch stood at the moment the
decision had to be made:

| regime | runs | rank of the winning branch by best-so-far |
| --- | ---: | --- |
| CONTINUATION | 3 | **1** -- the winner extended the branch already leading |
| DISCOVERY | 8 | **3 to 12** of 8 to 29 -- the winner came from a branch that did not lead |

| run | winner | winning branch's best | incumbent | rank | regime |
| --- | ---: | ---: | ---: | ---: | --- |
| v0_braf_1_r0 | -11.2 | -10.40 | -10.80 | 6 of 21 | DISCOVERY |
| v0_5ht1b_0_r0 | -12.6 | -10.80 | -12.10 | 4 of 11 | DISCOVERY |
| v21_braf_1_r0 | -10.7 | -10.40 | -10.40 | 3 of 26 | DISCOVERY |
| v21_5ht1b_0_r0 | -13.7 | -9.50 | -12.30 | 10 of 13 | DISCOVERY |
| v21_parp1_0_r0 | -11.7 | -8.00 | -9.80 | 12 of 18 | DISCOVERY |
| v0d06_parp1_0_r0 | -11.0 | -7.00 | -8.90 | 11 of 11 | DISCOVERY |
| v0_jak2_1_r0 | -10.3 | -10.00 | -10.20 | 5 of 29 | DISCOVERY |
| v0d06_braf_2_r0 | -10.2 | -9.60 | -10.10 | 4 of 8 | DISCOVERY |
| v1_braf_1_r0 | -10.4 | -9.90 | -9.90 | 1 of 19 | CONTINUATION |
| v1_5ht1b_0_r0 | -14.0 | -12.70 | -12.70 | 1 of 15 | CONTINUATION |
| v21_jak2_1_r0 | -10.7 | -10.10 | -10.10 | 1 of 22 | CONTINUATION |

EIGHT OF ELEVEN WINS ARE DISCOVERY. At the moment the decision had to be made the
winning branch sat third to twelfth. No passive rule over the branches the historical
search happened to sample can find those, because the ordering information is not
present. The single exception is a continuation win, which is the case where allocation
is least consequential.

Two of the three continuation runs still scored 0.000 pre-discovery share. That is the
exploration constant in `branch_best_scores`, not the data: the bonus reaches roughly
0.87 for an unsampled branch against a quality term capped at 1.0, so thin branches
outrank the leader. It does not affect the eight discovery runs, which no constant fixes.

CONSEQUENCE: a replay over passively collected history can falsify a passive allocator,
and cannot validate an ACTIVE experimental-design policy, because the counterfactual
scores such a policy would request were never measured. The replay has done its job and
should not gate the next algorithm.
