# T4 Dependency-Region Horizon Audit

## Question

The 77 exact T4 teacher routes contain 5 to 32 primitive rewrites, with a
median of 17. This diagnostic asks whether those traces contain far fewer
dependency-connected structural decisions than primitive executor steps.

The primary output is a decomposition

\[
P = R_1 \cup \cdots \cup R_K
\]

where each region groups primitive actions connected by changed atom
lifetimes, created-handle dependencies, cycle dependencies, or adjacency in
the lifetime molecular graph. Primitive actions remain in their recorded
order, and every route must replay exactly.

## Two definitions

The primary definition matches the PMO dependency-region analysis and joins
changed footprints that touch across a bond present at any point in the route.
Because that rule can merge nearby changes transitively, the audit also reports
a stricter sensitivity analysis that joins only shared atom lifetimes,
created-handle dependencies and cycle open/close dependencies.

Neither count is automatically a semantic subgoal label. The comparison
measures whether horizon compression is plausible and whether it is robust to
the connectivity definition.

## Required evidence

Report:

- primitive count, region count and region-run count per route;
- five-number summaries and full count distributions;
- component sizes, reentries and contiguous schedules;
- created-handle and cycle dependency edges;
- exact replay coverage and precision;
- coverage under the declared maximum of eight regions;
- abstentions and their reason codes;
- results by target and source group;
- exact input, code, software and environment identities.

Coverage and replay precision are separate. A route outside the eight-region
budget remains an abstention even if it replays correctly.

## Scope

This is a retrospective, answer-known and zero-oracle diagnostic. It does not
train a policy, generate candidates, change the current decoder, widen its
beam, alter a live run, or establish that a structural-subgoal controller will
recover docking performance.

The next architecture decision is made only after both this result and the
unchanged complete-program decoder comparison are observed.
