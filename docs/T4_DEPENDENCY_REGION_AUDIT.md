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

## Measured result

The authoritative zero-oracle result contains all 77 routes and 1,378
primitive transitions. Both definitions replayed 77 of 77 routes exactly, so
exact replay coverage and precision were 1.0.

Under the PMO-comparable lifetime-neighbor definition, 42 routes contained one
region, 34 contained two and one contained three. The median was one region,
the maximum was three, and all routes satisfied the declared eight-region
support. The 1,378 primitives collapsed into 113 regions, or 12.19 primitives
per region in aggregate.

Under the stricter shared-handle definition, 26 routes contained one region,
35 contained two, 13 contained three and three contained four. The median was
two regions, the maximum was four, and all routes again satisfied the declared
support. The 1,378 primitives collapsed into 147 regions, or 9.37 primitives
per region in aggregate.

The reduction is therefore robust to removing lifetime-bond adjacency.
However, regions are not always contiguous in primitive time. Under the
primary definition, only 47 of 77 routes had a one-pass contiguous component
schedule; the component-run median was one and the maximum was seven. Under
the strict definition, 33 of 77 were contiguous; the run median was three and
the maximum was ten. Primary and strict run counts were at most five for 74 of
77 and 67 of 77 routes, respectively.

The traces contained 643 created-handle dependency edges. All were internal to
the computed regions by construction. The primary and strict definitions
identified 25 and 14 cycle dependency edges, respectively. No route abstained
for primitive or region budget.

## Interpretation

The measured effective structural horizon is much shorter than the primitive
executor horizon. This supports testing structural-region or target-patch
prediction with exact compilation rather than treating all 5 to 32 primitive
steps as independent high-level decisions.

The interleaving result rules out a simplistic implementation that predicts
each region exactly once and never revisits it. A future macro controller
should preserve exact state checkpoints and support continuation, selective
repair or bounded region reentry. This result alone does not show that such a
controller will propose useful patches or recover docking performance.

Authoritative artifact:
`diagnostics/t4_dependency_regions/attempt_1/result.json`

Artifact payload SHA-256:
`ff024318edd2fa6c98254d03ddf38ab36a7781a874deb093264dc9193563788d`
