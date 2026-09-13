# PMO Median1 closing query

## Outcome

The single predeclared iodine analogue scored 0.39488251846572103 under the
IVG-core PyTDC 1.1.15 and RDKit 2023.09.6 environment. Appending this one
observation to the immutable eleven-query Median1 chronology increased the
official 10,000-query top-ten AUC from 0.3845976307197073 to
0.3963165611446791.

The resulting margins are:

| Comparator | AUC | COMPOSE margin |
| --- | ---: | ---: |
| IVG no prescreen | 0.34203551976134167 | +0.05428104138333745 |
| IVG prescreen | 0.3861480499242786 | +0.010168511220400522 |

## Query and accounting

- Candidate: `CC1CCC(C(I)C23CCC(CC2=O)C3(C)C)C(O)C1`
- New authoritative oracle calls: 1
- Automatic retries: 0
- Cumulative Median1 chronology: 12 calls
- Query-lock SHA-256: `bd1773f2ae806ff4cf5b77a4052d826f880ad00b94b29e71b7c9e3b1b82d4ac7`
- Result SHA-256: `e227e04aed1d71bc0137c52954906c2c92eddb2793f7a8c9f0030328b519c43a`
- Oracle receipt SHA-256: `7ecc6958763429b24ea40a50ae93598c504c73a064951a7cdf039805adde8de6`
- Code revision used for scoring: `3657823a0acb08e0e224a68f981b438b64a9135e`

## Scientific interpretation

This is a positive prospective panel-series extrapolation inside an
answer-known, winner-informed development regime. The candidate identity was
frozen before its score was observed, and the run used no retry, replacement,
or reselection. The result is not held-out evidence, autonomous PMO search, or
a claim about the remaining PMO tasks.
