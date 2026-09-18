# The two similarity lanes are different problems, and we lead one of them

The T4 endpoint criterion is `QED > 0.6, SA < 4, sim(endpoint, seed) >= delta` with delta
in {0.4, 0.6}. Similarity is Morgan radius 2, 2048 bits, Tanimoto -- verified identical to
`genmol_t4_opt_app`, so our gate and the benchmark's agree exactly.

JAK2 cell `jak2_1` is benchmark seed idx 13: `COC(=O)CC2Nc1ccccc1c3ccnc4[nH]cc2c34`.

## Bars, reconstructed over 5,622 scored molecules for that seed

| lane | best feasible | sim | margin | provenance |
| --- | ---: | ---: | ---: | --- |
| delta = 0.4 | **-11.70** | 0.426 | +0.026 | archive `full146` |
| delta = 0.6 | **-11.30** | 0.629 | +0.029 | **this session, route-assisted** |

Previous best at delta=0.6 anywhere in the archive was **-11.00**.

CONFIRMED over four fresh docking seeds under identical conditions:

| molecule | replicates | mean | sd |
| --- | --- | ---: | ---: |
| ours | -11.3, -11.4, -11.3, -11.3 | **-11.32** | 0.04 |
| archive delta=0.6 best | -11.0 x4 | -11.00 | 0.00 |
| second archive control | -10.6, -10.7, -10.6, -10.7 | -10.65 | 0.05 |

Separation is **0.32** against a per-molecule sd of 0.04, so it is not docking noise.

    CC1N(C(N)=O)CCN1C(=O)CC1Nc2ccccc2-c2ccnc3[nH]cc1c23
    -11.32 +/- 0.04   sim 0.629   QED 0.625   SA 3.79

A primary carboxamide where both archive leaders carry a carboxylic acid at the same
position: a small substitution the archive never made under the strict constraint.

## A correction that ran through this whole session

`-11.70` was used as THE JAK2 bar throughout. It is a delta=0.4 molecule at similarity
0.426 and is INFEASIBLE at delta=0.6. Every "short of target" statement made about -11.30
compared across incompatible lanes. The lanes have different feasible fibers --
`Q_0.6 ⊂ Q_0.4`, 1,162 feasible against 5,566 of the same 5,622 molecules -- and a
molecule may not be carried between them.

## Strong solutions hug the constraint boundary

| lane | median margin, top 20 | min |
| --- | ---: | ---: |
| delta = 0.4 | +0.062 | +0.003 |
| delta = 0.6 | **+0.019** | **+0.000** |

Three of the top ten delta=0.6 molecules sit at exactly 0.600. High reward concentrates
against the active constraint, which is why an edit that helps binding so often leaves the
fiber.

## Accessibility scales linearly; support is not the bottleneck

Free proposals from the -11.32 state, zero docking:

| proposals | unique molecules | delta=0.4 eligible | delta=0.6 eligible |
| ---: | ---: | ---: | ---: |
| 128 | 1,427 | 42 (2.9%) | 17 (1.2%) |
| 200 | 2,134 | 70 (3.3%) | 28 (1.3%) |
| 512 | 5,208 | 129 (2.5%) | 56 (1.1%) |

Eligibility fraction is flat, so a larger free fiber buys proportionally more feasible
candidates. Roughly 11 distinct molecules per proposal. Generation is not the constraint.

## -11.32 is a local optimum under single interventions

28 delta=0.6-eligible neighbours of the -11.32 state, selected toward the boundary and
docked: best **-11.00**, median -9.60, and **zero beat the parent**. The -11.00 is the
same carboxylic-acid variant the archive already held.

So the corridor immediately around -11.32 has been searched and contains nothing better
through one intervention. Beating it plausibly needs COORDINATED programs -- a
reward-directed change paired with a constraint-restoring change elsewhere, evaluated on
the completed endpoint -- which is exactly the capability still missing: 51 of 77 strong
routes are multi-subgoal and `joint_multi_site` is not wired into the controlled action.

## Standing caveats

- Route-assisted development, not autonomous. The basin was supplied.
- One molecule. Confirmed at four seeds, but a single point.
- delta=0.4 still trails: ours -11.40 against the archive's -11.70.
