# Shared retained-core support, nine-cell completion matrix

Decision: **PASS** for free exact support on all nine cells. This result uses one
target-independent retained-core pruning configuration everywhere and makes zero
oracle, docking, or Modal calls.

| Cell | Exact programs | Eligible unique |
|---|---:|---:|
| PARP1-0, delta 0.4 | 13 | 6 |
| PARP1-1, delta 0.4 | 8 | 2 |
| PARP1-2, delta 0.4 | 13 | 3 |
| BRAF-0, delta 0.4 | 101 | 19 |
| BRAF-1, delta 0.4 | 82 | 14 |
| BRAF-2, delta 0.4 | 76 | 16 |
| BRAF-0, delta 0.6 | 101 | 3 |
| BRAF-1, delta 0.6 | 82 | 6 |
| BRAF-2, delta 0.6 | 76 | 5 |

The two complete runs have byte-identical scientific projections. Operational
timing and code revision remain in the full results and are deliberately excluded
from that projection. The result establishes proposal and exact-execution support,
not docking utility.

The controller decision is to expose this generic operation as an internal
subproducer of the existing `route_complete_region` expert. It is not a fourth
target-specific lane. The existing program features already encode retained
deletion, so this does not require a new target or cell input.

Authoritative files:

- `attempt_1/result.json`, physical SHA-256
  `105a1ea623ce812f10258c74e847efa44d974d0fef55d36e5706e1a4637cb61d`;
- `attempt_1/scientific_result.json`, physical SHA-256
  `f85bc443ac99bdc176c602d293b2d45a27f8cb07fd23bb8a448a9c67ddb4867a`;
- scientific payload SHA-256
  `82989772a277410d783c3755422407617a542e3a893facb325276ca314b46429`.
