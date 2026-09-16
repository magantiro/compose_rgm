# T4 construction scale: the value is where the sampler cannot go

Contract `951fb6961d880429`. Zero oracle calls, no model fit,
no production sampler modified.

## The winning routes decompose into a few coordinated regions

- 77 exact teacher routes, median 2 dependency regions, median 17 primitives
- primitives per region: p25 3, median 8, p90 19
- 65/77 reuse a handle they created (median 7 of 9)
- **76/77 have an ineligible interior**, median 11 prefixes

That last line is why a chain of small eligible programs cannot get there.

## What each family does

| family | primitives | changed originals | created |
| --- | ---: | ---: | ---: |
| compiled_complete_transformation | 18 | 11 | 8 |
| pendant_benzene | 9 | 1 | 8 |
| core | 11 | 2 | 7 |
| append_ring | 6 | 1 | 6 |
| dependency_branch | 9 | 5 | 4 |
| fuse_ring | 4 | 2 | 3 |
| remodel_linker | 6 | 4 | 3 |
| segment_replace | 5 | 3 | 3 |
| functionalize | 2 | 1 | 2 |
| segment_grow | 2 | 1 | 2 |
| carbonyl_insert | 1 | 1 | 1 |
| ring_carbonyl | 1 | 1 | 1 |
| atom_restate_semantic | 1 | 1 | 0 |
| bond_reroute | 1 | 3 | 0 |
| cycle_close | 1 | 2 | 0 |
| cycle_open | 1 | 2 | 0 |
| heteroatom_substitute | 1 | 1 | 0 |
| ring_system_restate | 1 | 6 | 0 |
| segment_shrink | 2 | 3 | 0 |
| substituent_delete | 1 | 2 | 0 |

Builders (create at least as much as they disturb): append_ring, carbonyl_insert, functionalize, fuse_ring, segment_grow, segment_replace. Only these move a program toward the
winning shape; the rest scatter edits and break similarity.

## Where the value is, and where the sampler is

| primitives | corpus n | best | p5 | median | sampler eligible (parp1) | (jak2) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-2 | 3434 | -13.4 | -10.2 | -8.2 | 82.4% | 50.0% |
| 3-5 | 1611 | -13.1 | -9.1 | -8.0 | 61.5% | 30.8% |
| 6-9 | 1222 | -12.5 | -9.9 | -8.6 | 43.5% | 19.2% |
| 10-14 | 1284 | -13.8 | -10.7 | -9.6 | 0.0% | 0.0% |
| 15-22 | 1137 | -14.3 | -11.9 | -10.2 | 0.0% | 0.0% |
| 23-32 | 39 | -13.9 | -13.6 | -12.3 | 1.2% | 0.0% |

**1 of 249** sampled proposals at >=15 primitives were eligible (0.40%); one-sided 95% upper bound 1.89%.

## But the region is densely populated

| cell | root-applied | >=15 primitives | share | distinct endpoints | best | p5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5ht1b_0 | 309 | 111 | 35.9% | 107 | -13.1 | -12.4 |
| braf_1 | 1101 | 1088 | 98.8% | 913 | -11.1 | -9.9 |
| fa7_0 | 526 | 329 | 62.5% | 271 | -9.5 | -9.0 |
| jak2_1 | 2578 | 669 | 26.0% | 628 | -11.7 | -10.9 |
| parp1_0 | 2250 | 479 | 21.3% | 454 | -14.3 | -12.5 |

Every corpus record is eligible by construction, so these are real eligible
endpoints reachable in one program from the benchmark root. The region is not a
needle; the sampler is aimed away from it.

## Limits

- This measures ACCESS, not docking value. Strategy report section 7 measured
  distance-to-best against score-gap at Spearman 0.12 on PARP1, so eligible large
  constructions still need real docking feedback to be aimed.
- The root-applied density describes what the historical search, holding the
  winner bank, chose to try. It proves the region is populated, not that it is
  easy to hit without the bank.
- Size bands, the 15-primitive threshold and the builder set were chosen after
  inspecting this corpus. The prospective lane gate in the contract was not.
- Local RDKit is newer than the pinned benchmark image.
