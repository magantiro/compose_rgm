# T4 target-conditioned utility data acquisition

## Outcome

The authoritative v3 audit is an abstention: 84 rows enter global grouping, 37
rows are removed as one cross-fold connected component, and 47 remain. Only
JAK2 is supported, with two held-source strata. The frozen gate still requires
at least three targets and at least five jointly supported held-source strata.

The smallest immutable acquisition that closes the structural support gap uses
four requests:

- two exact `parp1_2` candidates from the compositional structural-subgoal lock
  at commit `7d87ff2959845d5c6273e743141ac9c0fe806594`;
- two exact `fa7_1` control candidates from the zero-oracle program-retrieval
  pool introduced at commit `74a14ee365bacaf3ac8bd614a24e5db0623803c9`.

Every candidate is replayed through the unchanged production executor. The
official strict endpoint calculation requires similarity greater than 0.4,
QED greater than 0.6, and SA below 4.0. All four pass. Their endpoint,
Bemis-Murcko scaffold, cell-and-source, final program or macro, and parent
program groups do not intersect the v3 poisoned component. The two selected
cells do not bridge one another.

## Why four is the minimum

V3 supports one target, JAK2. Each additional supported target needs at least
one held-source cell containing a strict measured pair, and therefore at least
two finite non-tied measurements. Reaching three targets requires two more
targets, so four requests are a lower bound. This lock meets that lower bound
with two requests in each of two independent cells.

## Conditional gate result

The simulation creates in-memory placeholder orderings solely so the unchanged
strict-pair and support code can be executed. Placeholder values are not written
to any artifact and are not evidence. If all four future calls return finite
scores and each within-cell pair is non-tied, all four rows survive grouping.
The projected final corpus has 51 rows, three supported targets (FA7, JAK2 and
PARP1), seven supported strata, and zero post-exclusion cross-fold groups.

The result is conditional. A failed call, null value, or tied pair does not
authorize replacement, backfill, a gate relaxation, or a positive claim.

## Separation and authorization

This revision acquires potential training data. It does not fit the utility
model and is not a prospective evaluation. No target-specific route lookup is
introduced at runtime; exact routes exist only as acquisition provenance.

No docking call is authorized by these files. A subsequent explicit
authorization must name the final physical and payload SHA-256 hashes of
`candidate_lock.json` and `request_lock.json` and authorize exactly four scored
calls. Automatic retries and replacements remain disabled. After acquisition,
labels must be frozen before any separate prospective evaluation is designed.

The final candidate lock is physically
`d790389696d729f19ce617b3dcf4afddf0f3ce06110882a1b628889e0c354249`
and has payload identity
`8b5acdbfca42a122d6a882d48e6bbe00468052c913f79e806b6dbb7164685471`.
The final request lock is physically
`e11441244c98d919d42315dcafbdede300bbb6232bdc5a911896bdb3d1010d05`
and has payload identity
`f78985599a7ff5e533384ac7d61c2cc77f7c6cf94e064f6a6ec0e3ed692bc279`.
