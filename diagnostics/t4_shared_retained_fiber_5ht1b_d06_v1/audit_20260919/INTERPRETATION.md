# 5HT1B-2 delta=0.6 candidate-exhaustion forensic

The frozen run stopped after the root call with status `candidate_exhaustion`. This audit made zero oracle or docking calls.

## Complete endpoint funnel

| Expert | Proposed unique | Exact endpoints | Structural | Sim | QED | SA | All eligible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| shallow | 4360 | 4360 | 3371 | 624 | 1368 | 140 | 0 |
| anchored_replacement | 2357 | 2357 | 1920 | 1362 | 225 | 1 | 0 |
| route_complete_region | 161 | 96 | 86 | 17 | 79 | 11 | 0 |

## Interpretation

The campaign stopped before FiberControl selection because every expert's completed-endpoint pool had an empty intersection between structural validity, similarity>=0.6 and SA<=4: shallow=0, anchored_replacement=0, route_complete_region=0. Exact construction itself was healthy: shallow built 4360 unique endpoints, anchored replacement built 2357, and the route expert committed 96/96 with exact precision. The root is valid with similarity=1 and QED above the floor, but SA=4.6872>4. The lanes can preserve similarity or repair SA, but not both in one protected program. This is an endpoint constraint-support failure, not a compiler, docking-value, or FiberControl-ranking failure. The answer-known strict corpus sharpens the support diagnosis: 3/3 released delta=0.6 endpoints change the source formal charge (+1 to 0), and all are recorded as unreachable by the current charge-preserving rewrite support.

Smallest general fix: Add one shared charge-aware retained-subgraph constraint-repair program family. It should explicitly extend the declared exact rewrite support with a chemically validated formal-charge/protonation restatement, retain the source scaffold, and use only free similarity/QED/SA margins to Pareto-search the rewritten complement as one protected program. Require nonzero diverse eligible yield under the production sampler on this root plus contrasting roots before scoring. Do not tune FiberControl first because no candidate currently reaches it.

Known endpoints are included only as labelled answer-known diagnostics. They were never injected into a proposal pool.
