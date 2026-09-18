# Anchored-replacement JAK2 scored pilot

## Outcome

The prospectively locked panel passed its promotion rule. All 12 novel JAK2 seed-1
delta-0.6 endpoints returned a docking score. The best score was **-11.1**.

| threshold | endpoints at or better |
| --- | ---: |
| -9.8 | 8 |
| -10.0, frozen promotion threshold | 4 |
| -10.4, published IVG mean | 3 |
| -11.0, released route-witness value | 1 |

The best endpoint was:

`CN1CCN(C(=O)O)CN1C(=O)CC1Nc2ccccc2-c2ccnc3[nH]cc1c23`

It has Morgan similarity 0.600, QED 0.6045, SA 3.6881 and 30 heavy atoms under
the frozen evaluator. The next two scores were -10.6 and -10.4.

Before this prospective panel, exact identity matching found one different endpoint
from the same frozen support pool in the historical Full-146 corpus at -10.2. That
historical rediscovery was excluded from the 12 charged calls.

## Interpretation

The result validates the proposal-support diagnosis. Coupling an existing pendant
deletion and substituted-ring construction at one retained anchor moved the generic
proposal law from zero eligible basin endpoints in both prior lanes to a pool containing
multiple molecules with strong measured JAK2 utility. The winning endpoint was generated
without loading a target route or endpoint at runtime.

This is one answer-known development cell and one docking seed. Single-call variability
on this target has previously been measured at about 0.5, so -11.1 is a strong signal,
not yet a replicated final score. It should not be described as a benchmark-wide IVG
win. The frozen next action is to integrate the promoted proposal family into the shared
FiberControl archive with the unchanged shallow exploration floor, then test whether
reward-guided refinement improves or stabilizes it.

## Query ledger summary

- Locked endpoints: 12
- Charged calls: 12
- Successful calls: 12
- Failed calls: 0
- Automatic retries: 0
- Replacement or backfill: none
- Docking elapsed time: 36.70 seconds
- New result file SHA-256:
  `770f3b69f8e1fd82a0b06c6b338899335c9d8c9403e5d33888a6cd407dc6e233`
- Candidate lock ID:
  `9154ebd97f40cd47fcf85d10cbc8ec94997da1b45261be149832b8203ec51520`
- Candidate lock file SHA-256:
  `def92f269bc002ead09bfac900a6775754497a5d8048ab8fcd1da1b3929d2173`
- Contract payload SHA-256:
  `927d769676576f01e8e48927df4d7f8dfe3b9c3d90f68fe7096f44f62057f706`
- Executed clean revision:
  `88b83df6923b30ae9012d3b71d5c5954b12acf1a`
- Modal run: `ap-FE9l0re6qX0iUMLvbHc8q8`

The preceding cold-image launch `ap-y11ZcoqEeYpD2RywDzSIIT` failed locally on a
missing `PYTHONPATH` before invoking the remote docking function. It spent zero oracle
calls. The unchanged lock was then launched with the documented `PYTHONPATH=src`.
