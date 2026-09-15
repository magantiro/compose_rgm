# T4 Compositional Structural-Subgoal Generator

## Scientific question

The grouped-source component audit recovered no complete held patch from a
training-fold patch vocabulary, while many atom, bond and small graph
components remained familiar. This milestone tests whether a split-first model
can generate new executable local patches compositionally. The generated object
is

\[
g=(R,H,\alpha,D),
\]

with an address-free source region `R`, target patch graph `H`, attachment edges
`alpha` and ordered construction dependencies `D`. The frozen factorization is

\[
\pi(g\mid x)=\pi_R(R\mid x)\,\pi_\theta(H,\alpha,D\mid x,R).
\]

The runtime support is the generic Active8 atom, bond and relative-role event
grammar. It contains no whole-patch identifier. One weighted diagonal density
selects a region through the first event. A second weighted density scores the
complete joint event vector conditional on the graph, selected region, partial
patch and prior event. One or two mutually local events jointly determine
`H`, `alpha` and `D`. The unchanged exact structural realizer then compiles the
patch within 32 primitives and 40 active atoms.

## Frozen data and split

The outer evaluation holds out each of 15 complete T4 source groups exactly
once across three predeclared folds. It contains 77 complete teacher routes and
147 structural subgoals. Every learned statistic is fitted after the whole-source
split. Training-only auxiliary data include 55 of 69 Full-146 applications with
exact origin-context bindings and 85 of 184 PMO routes converted exactly by the
existing immutable path. The remaining 14 Full-146 applications and 99 PMO
routes retain their recorded abstention reasons. Delta-0.6 routes are excluded.

The fold-specific training mixtures are:

| Fold | T4 routes | Full-146 routes | PMO routes | Joint events | Components |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 50 | 0 | 85 | 405 | 210 |
| 1 | 51 | 55 | 85 | 557 | 291 |
| 2 | 53 | 55 | 85 | 546 | 282 |

Fold 0 admits no Full-146 routes because their origin source is held in that
fold. This is a split consequence, not a dropped-data optimization. The fitted
weights give equal mass to domains, then lineages, routes, components and
decisions rather than following the raw counts in the table.

## Frozen autonomous comparison

The learned density and a uniform scorer use identical legal fibers, locality,
exact compiler, event-depth limit, beams and candidate budget. The autonomous
lock contains 15 held sources, two policies per source and 128 candidates per
policy. Teachers were loaded only after the lock was published. The table below
reports source-balanced measurements.

| K | Policy | Granular component coverage | Exact patch recall | Novel patches per source | Unique patches per source | Unique endpoints per source |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 8 | Uniform | 24.08% | 0.00% | 5.80 | 5.80 | 8 |
| 8 | Learned | 18.01% | 0.00% | 4.80 | 4.93 | 8 |
| 32 | Uniform | 37.86% | 0.00% | 18.93 | 18.93 | 32 |
| 32 | Learned | 34.63% | 0.92% | 17.33 | 18.47 | 32 |
| 128 | Uniform | 46.25% | 0.00% | 76.60 | 76.60 | 128 |
| 128 | Learned | 59.77% | 0.92% | 92.67 | 94.67 | 128 |

The exact patch metric compares the 147 held structural subgoals under the
frozen address-free attributed-isomorphism contract. The learned arm recovers
2/147 exact patch components at K=32 and K=128. The uniform arm recovers 0/147.
Neither arm recovers any of 77 complete teacher endpoints or any of 77 radius-2
transformation-equivalence classes at K=8, K=32 or K=128. Exact teacher recovery
is diagnostic and was not a required utility endpoint, but this zero remains a
material negative result.

At K=128, instance-weighted component results localize what improved and what
did not:

| Component family | Uniform | Learned | Learned minus uniform |
| --- | ---: | ---: | ---: |
| Atom-attribute tokens | 33.53% | 68.89% | +35.36 points |
| Bond-attribute tokens | 75.24% | 92.90% | +17.66 points |
| Source graphlets, at most 3 vertices | 20.30% | 49.17% | +28.87 points |
| Target-topology graphlets, at most 3 vertices | 40.09% | 47.63% | +7.55 points |
| Attachment-edge tokens | 39.09% | 30.45% | -8.64 points |
| Dependency tokens | 65.79% | 65.92% | +0.14 points |

The learned arm therefore expands atom, bond, source-region and local-topology
support, but it does not improve attachment coverage and barely changes
dependency coverage. Together with zero transformation-equivalent recovery,
this localizes the residual failure primarily to joint attachment and dependency
composition and to early ranking, not to a total absence of region or topology
support. The learned arm is also worse than uniform on aggregate component
coverage at K=8 and K=32. Its improvement appears only after the full K=128
support is exposed.

## Execution, work and gate decision

Every accepted candidate exactly realizes and replays: 1,920/1,920 per arm.
The uniform arm realizes 1,920 of 1,922 compiler attempts (99.90% compile
coverage). The learned arm realizes 1,920 of 2,053 attempts (93.52% compile
coverage), with 87 `TypeError` and 46 `ValueError` abstentions recorded before
the full candidate pool is filled. Both arms have zero candidate shortfall and
128 distinct endpoints on every source. The learned arm has at least 74 novel
whole patches on every source at K=128; the uniform minimum is 25.

The frozen scientific gate passes only at K=128. Source-balanced granular
component coverage rises from 46.25% to 59.77%, a 13.52-point improvement, while
accepted-candidate exact-realization precision remains 100% in both arms. No
gate was changed after observing the result. This pass establishes improved
generated local component support. It does not establish complete transformation
recovery, route recovery, docking utility, autonomous optimization or an IVG
comparison.

## Performance implementation

The first complete candidate lock took 3,080.16 seconds on one CPU worker. The
local-role optimization filters exact action references before executing
second-event candidates and preserves the same canonical local successor fiber.
A regression compares it directly with filtering the complete legal fiber.
There was no reduction in event depth, beams, action domains, candidates, folds
or evaluation support.

On one held-source first-event state with two mutable roles, complete enumeration
then filtering took 18.88 seconds and returned 23 local successors from a
699-successor fiber. Local prefiltering took 4.74 seconds, returned the same 23
canonical successors and was 3.98 times faster in this single representative
measurement. The measurement order and lack of repetitions limit this to
implementation evidence, not a general throughput claim. The full independent
candidate-lock rerun took 3,082.24 seconds and was byte-identical to the first
lock.

## Next safe gate

A scored pilot is not authorized here. A later contract may lock a small,
source-balanced and diversity-preserving learned panel together with the matched
uniform baseline, then score both under the same call budget. That prospective
utility test should keep exact teacher recovery as a diagnostic. The present
outer-test result must not be used to retune this fitted model. Any improvement
to early ranking or attachment/dependency composition requires a new split-first
development contract and selection evidence that does not reuse this outer test.

The self-hashed contract is
`configs/t4_compositional_structural_subgoal_generator_v1.json`. The complete
machine-readable lock, result, fit reports and runtime checkpoints are in
`diagnostics/t4_compositional_structural_subgoal_generator/attempt_1/`.
