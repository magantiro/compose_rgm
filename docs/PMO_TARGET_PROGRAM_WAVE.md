# PMO exact-target program wave

## Problem and claim

This wave tests whether the executable-program mechanism that produced the
measured Perindopril result transfers to five additional published PMO task
definitions. The primary output is a locked collection of complete COMPOSE edit
programs and a counted task-oracle ledger.

The proposed claim is narrow: starting from the same unrelated exact molecular
root, COMPOSE can execute fifteen unique target-neighborhood programs per task,
and their early measured scores can be compared with IVG's reported top-ten
area under the curve (AUC). Exact public task targets are answer-known
development supervision. This is not held-out discovery.

## Tasks and evidence

The first wave contains:

- `albuterol_similarity` and `mestranol_similarity`;
- `celecoxib_rediscovery`, `troglitazone_rediscovery`, and
  `thiothixene_rediscovery`.

The target SMILES are copied from the pinned PyTDC 0.3.6 oracle source whose
SHA-256 is recorded in the contract. IVG comparators come from its official
results repository at commit
`b50bb3ae2bdcb9df581f0b219d79cf14b05d0fbb`; each three-run CSV is pinned by
SHA-256. The upstream repository is read-only and MIT licensed.

## Frozen procedure

1. Load one previously charged, unrelated neutral COMPOSE root and verify its
   physical hash.
2. Compile and replay a complete route from that root to each published target.
3. Without calling the task oracle, enumerate conservative one-edit variants of
   the target: supported atom substitutions and single terminal additions.
4. Compile each variant as a target route plus a short dependent suffix. Lock
   the first fifteen unique candidates that replay exactly, including the exact
   target.
5. Abort before scoring unless every task has fifteen unique supported program
   endpoints and all route/program receipts replay byte-for-byte.
6. For each task, score the unrelated root followed by its fifteen locked
   endpoints. Each query is reserved and completed once; ambiguous started
   calls are never retried or replaced.
7. Compute official PMO top-ten AUC with frequency 100, denominator 10,000, and
   `finish=True`. Report exact margins against both IVG regimes.

The total ceiling is 80 calls. The chemistry support remains neutral
charge-preserving graphs, at most 40 active atoms in 48 persistent slots. The
run uses no learned reference, surrogate selection, hidden prescreen, Modal, or
free task scoring during candidate construction.

## Decision rule and limitations

A task beats a comparator only when its measured official AUC is strictly
larger than the recorded comparator. Endpoint and final-top-ten scores are
reported separately. Plateau candidates remain distinct molecules but are not
described as independent optimization mechanisms.

Failure to compile fifteen candidates, invalid endpoints, oracle exceptions,
ties, and negative margins are retained. The five task objectives disclose
their targets; success therefore tests executable target-neighborhood program
construction, not whether COMPOSE inferred an unknown objective. Learned
property, isomer, scaffold-hop, and remaining MPO tasks require later bounded
waves with task-appropriate candidate evidence.

## Measured result, 2026-09-13

The structural gate replayed all 75 locked programs. All 80 counted queries
completed, with no retries or replacements. COMPOSE exceeded both IVG
comparators on all five tasks:

| Task | COMPOSE AUC | IVG no-prescreen | Margin | IVG prescreen | Margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| Albuterol similarity | 0.999200000000 | 0.949572587973 | +0.049627412027 | 0.974532957790 | +0.024667042210 |
| Celecoxib rediscovery | 0.880873684211 | 0.798137995354 | +0.082735688856 | 0.839263357912 | +0.041610326298 |
| Mestranol similarity | 0.999200000000 | 0.797255084892 | +0.201944915108 | 0.990782732417 | +0.008417267583 |
| Thiothixene rediscovery | 0.886835750916 | 0.625030138935 | +0.261805611980 | 0.652193299761 | +0.234642451154 |
| Troglitazone rediscovery | 0.881902608696 | 0.594675360870 | +0.287227247826 | 0.852727258675 | +0.029175350020 |

Both similarity tasks placed all 15 program outputs on the oracle's clipped
score plateau at 1.0. Each rediscovery task included the exact target at 1.0;
its remaining one-edit endpoints scored between 0.756098 and 0.888889. These
are answer-known target-neighborhood results and do not show blind target
inference. The authoritative result payload SHA-256 is
`6a20fe94e989eab09215792e7cbca3f7e3706c5dbef51c0de3836965e7213ec0`.
