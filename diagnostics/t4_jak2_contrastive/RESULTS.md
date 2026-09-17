# JAK2 contrastive mechanism experiment: results

> **CORRECTED.** The first version of this file omitted the `SA < 4` gate, which is part
> of the T4 endpoint criterion and which this work has cited all along. Applying it
> changes the headline: only 11 of the 28 docked candidates pass it, and NONE of the six
> control candidates do, so the controller-against-control comparison cannot be made on
> benchmark-eligible molecules. The corrected numbers are below; the original
> gate-omitting numbers are kept at the bottom so the correction is legible.

32 docking calls, exactly the preregistered budget
(`docs/JAK2_PROSPECTIVE_PREREGISTRATION.md`). Selections were committed before any
Stage-2 candidate was docked.

## Headline, on fully qualified molecules

A molecule is FULLY QUALIFIED when it passes every gate the benchmark applies -- QED > 0.6,
similarity > 0.4, **SA < 4** -- and carries no unstable motif.

    BEST FULLY QUALIFIED:  -10.00
    COC(=O)CC1Nc2ccccc2-c2ccnc3c2c1c1n3CCCC1      SA 3.35   QED 0.710

It came from STAGE 1, not from the controller's Stage-2 picks.

10 of 28 docked candidates are fully qualified. By arm:

| arm | fully qualified | best fully qualified |
| --- | ---: | ---: |
| controller (h3) | 2 of 6 | -8.90 |
| uniform control | **0 of 6** | none |
| Stage-1 bundles | 8 of 16 | **-10.00** |

The one comparison that survives the correction is that the controller produced two
qualified candidates and the control produced none. That is directionally favourable and
it is two against zero, which is not evidence of much.

WHAT THE ORIGINAL HEADLINE CLAIMED, and why it does not stand: the controller beat uniform
by 1.30 on best and 0.75 on mean. Both arms were drawn from the same pool, so the
comparison was internally fair, but the pool itself was never filtered for synthetic
accessibility. Seventeen of the twenty-eight molecules compared were not benchmark
eligible, including the -10.60 top hit's nearest rivals, so the margin describes a
population the benchmark would not accept.

The -10.60 that headlined the original version is an acyl-nitroso / mixed anhydride. It
passes SA at 3.85 and reproduces across seeds, and it is not a molecule anyone would
carry forward. It is now refused by the stability gate in `queryable_fiber`.

Confirmed at fresh docking seeds:

| molecule | replicates | mean | spread |
| --- | --- | ---: | ---: |
| `CC(=O)ONOC(=O)CC1Nc2ccccc2-c2ccnc3c2c1c1n3C(=O)CCC1` | -10.6, -10.5, -10.4 | **-10.50** | 0.20 |
| `COC(=O)CC1Nc2ccccc2-c2ccnc3c2c1c1n3C23CC(C2)NC3CC1` | -9.8, -9.9, -9.7 | -9.80 | 0.20 |

MEASURED DOCKING NOISE is about 0.2 total spread across three seeds, so sigma is nearer
0.1 than the 0.35 assumed in the belief. The belief was therefore CONSERVATIVE, not
optimistic; refitting it is future work and was not done after seeing these numbers.

## Stage 1: the contrasts are identifiable

16 calls, 4 matched scale bundles from the JAK2 root, all scored in 44 seconds.

| bundle | scale 2 | 3 | 4 | 5 | 6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | | -9.00 | -8.90 | **-10.00** | -9.50 |
| 1 | | -8.50 | -8.40 | -8.20 | -7.40 |
| 2 | | -8.80 | **-10.00** | **-10.00** | -9.40 |
| 3 | -8.90 | -9.10 | **-9.30** | -8.30 | |

Three of four bundles peak at scale 4-5 and one degrades monotonically. Because the arms
of a bundle share a parent, the parent term cancels and these differences identify the
coordinate effect. Posterior after Stage 1: scale 4 `+0.470`, scale 5 `+0.445`, scale 2
`+0.357`, scale 3 `+0.177`, scale 6 `+0.052`. The controller then concentrated Stage 2 on
scale 4-5, which is where its advantage came from.

## The queryable fiber held

Every one of the 32 calls was spent on a molecule already known to be a legal COMPOSE
program endpoint, exactly executable, connected, QED > 0.6 and similarity > 0.4. The
oracle was never asked whether a candidate was usable, only how well it binds. Stage-1
funnel from autonomous anchors: 165 interventions attempted, 82 refused as chemically
unsatisfiable, 83 admitted, 74 realized as molecules, 16 eligible.

## What this does NOT show

**Depth was uninformative.** The `h=3` and `h=1` controllers selected IDENTICAL
candidates, 6 of 6. With single-coordinate bundles the continuation term is a near
constant offset across candidates -- Q moved -10.958 to -11.853 almost uniformly -- so it
does not reorder them. This was predeclared as a possible outcome and is reported as
uninformative, NOT as a tie. Depth can only matter when candidates open genuinely
different continuation sets.

**Not significance.** Six against six, as predeclared.

**Not a benchmark result.** -10.50 confirmed is about the autonomous JAK2 record (-10.7)
and below the winner-bank arm (-11.6) and IVG. What is demonstrated is a mechanism, on 32
calls, not a competitive number.

**Two gates were missing, and both mattered.** `SA < 4` was never applied -- a plain
omission, since it is part of the endpoint criterion this work has cited throughout. And
nothing tested chemical stability, which let an acyl-nitroso species top the table. Both
are now enforced in `compose_v4.control.queryable_fiber` before a call is spent. The
runner-up praised as "clean" in the original version scores SA 5.37 and is itself
ineligible.

**The corrected result is weaker than reported.** Best fully qualified is -10.00 from
Stage 1, against an autonomous record of -10.7. The controller did not produce the best
molecule in this experiment; Stage-1 exploration did.

## Provenance

Fully autonomous on the JAK2 side: anchors came from the production sampler applied to the
JAK2 root, not from any teacher route. Route information entered only as intervention
GEOMETRY. No JAK2 winner route, endpoint or lineage was read.
