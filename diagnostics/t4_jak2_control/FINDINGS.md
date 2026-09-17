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
