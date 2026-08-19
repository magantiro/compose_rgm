# InversionGNN — vendored comparator source

    upstream  https://github.com/ivanniu/InversionGNN
    commit    cfdf1d9a981ca4ce5dd7293dc38373ddf9377718
    dated     2025-08-29 17:57:56 +0800
    vendored  2026-08-19

Vendored because the matched head-to-head requires RERUNNING this method, and a
`/tmp` clone does not survive a session. `.git` is stripped; the commit above is
the audited state.

## What the audit established (see docs/AMENDMENT_INVERSIONGNN_2OBJ.md)

`InversionGNN/molecular/denovo.py` is **not** a faithful reproduction script for
the paper's Table 3. It disagrees with the paper on three counts:

    start molecule    one hardcoded 'C1=CC=CC=C1NC2=NC=CC=N2'   paper: unspecified
    population_size   1                                          paper: C = 10
    preference        a single hardcoded [1, 3]                  paper: 5 weight vectors

The hardcoded start is not neutral: scored on our frozen oracle it is
JNK3 = 0.100 -- the maximum of our entire random-ZINC development cohort
(median 0.010) -- and is an anilinopyrimidine, a kinase hinge-binding chemotype.

Consequently the published 0.841 APS is carried as REPORTED CONTEXT only. The
load-bearing comparison is our matched rerun from a common frozen initialization
bank.
