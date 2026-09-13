# PMO IVG-oracle parity results

## Result

The no-reselection audit completed all 150 locked queries under PyTDC 1.1.15,
RDKit 2023.09.6 and the other IVG-pinned oracle dependencies. It used zero
automatic retries. The separate precontract diagnostic accounts for one
additional call, so the complete parity-environment accounting is 151 calls.

COMPOSE exceeds the repository-reported IVG no-prescreen AUC in all 11 completed
tasks and the IVG prescreen AUC in 10 of 11. Median 1 is the only prescreen miss,
by 0.001550419 AUC.

| PMO task | Calls | COMPOSE AUC | IVG no-prescreen | Margin | IVG prescreen | Margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Albuterol similarity | 16 | 0.999200 | 0.949573 | +0.049627 | 0.974533 | +0.024667 |
| Celecoxib rediscovery | 16 | 0.880874 | 0.798138 | +0.082736 | 0.839263 | +0.041610 |
| GSK3B | 11 | 0.998451 | 0.951551 | +0.046900 | 0.987755 | +0.010696 |
| Isomers C7H8N2O2 | 11 | 0.999450 | 0.968379 | +0.031071 | 0.987885 | +0.011565 |
| JNK3 | 11 | 0.971465 | 0.825433 | +0.146033 | 0.897835 | +0.073631 |
| Median 1 | 11 | 0.384598 | 0.342036 | +0.042562 | 0.386148 | **-0.001550** |
| Mestranol similarity | 16 | 0.999200 | 0.797255 | +0.201945 | 0.990783 | +0.008417 |
| Perindopril MPO | 15 | 0.808404 | 0.645000 | +0.163404 | 0.753000 | +0.055404 |
| QED | 11 | 0.947920 | 0.942328 | +0.005592 | 0.943446 | +0.004474 |
| Thiothixene rediscovery | 16 | 0.886836 | 0.625030 | +0.261806 | 0.652193 | +0.234642 |
| Troglitazone rediscovery | 16 | 0.881903 | 0.594675 | +0.287227 | 0.852727 | +0.029175 |
| **Partial sum, these 11 tasks only** | **150** | **9.758300** | **8.439398** | **+1.318903** | **9.265568** | **+0.492732** |

The partial sum is descriptive and is not the 23-task PMO headline metric.
Twelve PMO tasks have not yet been evaluated in this development program.

## Protocol finding

Ten task reward sequences were numerically unchanged between the previous and
parity environments on these exact molecules. The C7 isomer sequence changed
materially: its AUC moved from 0.008877448 under PyTDC 0.3.6 to 0.999450 under
PyTDC 1.1.15. The largest single-query change was 0.992553417. This confirms
that the earlier C7 result was an evaluator-version artifact.

The downloaded GSK3B and JNK3 current-model pickles differ from the legacy
HN-GFN pickle identities used by the prior frozen forest lane, but their scores
on these particular locked panels were identical. That observation does not
establish global model equivalence.

## Receipt and provenance audit

- Authoritative started receipts: 150.
- Authoritative completed receipts: 150.
- Authoritative failure receipts: 0.
- Automatic retries or replacements: 0.
- Measured oracle evaluation time, excluding environment and asset acquisition:
  0.439435 seconds summed across 150 calls.
- Result artifact SHA-256:
  `e99d7b0a311693bd9f12509d62531444dc187f3b4e88a75ebd6a440479877675`.
- Query lock SHA-256:
  `056e6ca3ca4b80cd7bc9c1de577222560fab8d39852b3b011137d06fa395d4ae`.
- PyTDC GSK3B current asset: Dataverse file 6413412,
  27,791,877 bytes,
  `d3a20701b80e5179c88c3ad4dc3483dd7ab35c50dc055c6773a7f5b63e89b6d5`.
- PyTDC JNK3 current asset: Dataverse file 6413420,
  10,888,961 bytes,
  `cde8576fb4fa3f60b9f258ff9cf1b9ff346eb50d196d5cbbe25965efc1864889`.

Two pre-score failures are retained. The first rejected legacy model hashes;
the second rejected incorrectly transcribed current-model byte counts. Both
occurred before any authoritative query and changed no candidate, score,
comparator, metric or query budget.

## Scientific interpretation

These are measured evaluator-parity results for answer-known,
winner/panel-informed COMPOSE program curricula. They support exact arithmetic
comparisons for these locked development sets. They do not show autonomous
10,000-query search, held-out generalization, or superiority on the full
23-task PMO suite.

The immediate optimization target is Median 1, where only a 0.001551 AUC gain
is needed to exceed the IVG prescreen value. After that, the same
compile-lock-score discipline should cover the remaining 12 PMO tasks under
this parity evaluator.
