# WHERE support audit: does the controller ever propose the region a strong route needs?

> **SUPERSEDED AND CORRECTED.** The figures below (10.6% overall, 0 of 42 single-region)
> compared two different objects. The teacher region was taken as every slot named in a
> region's primitive payloads, which includes CREATED slot indices -- atoms the
> transformation inserts, numbered at or above the source's real-atom count. 80.4% of
> teacher regions carry at least one. A runtime region is `assignment` plus
> `changed_original_slots` and by construction names only SOURCE slots, so four fifths of
> the comparison was uncoverable no matter what the sampler proposed.
>
> Restricted to the comparable quantity -- the ORIGINAL slots a transformation must
> occupy -- coverage is **68.1% overall and 96.6% for JAK2**, and single-region routes are
> 0.595 rather than 0.000. The corrected table is at the end of this file. The original
> numbers are preserved here as the record of the error, not as a result.
>
> The `0 of 42` was the tell. A quantity that is exactly zero across a large homogeneous
> class is far more often a definition mismatch than a finding, and it should have been
> chased before the number was reported.

Zero oracle calls. 77 strong routes, 113 structural regions, 100 proposals drawn from each
route's own source state, fanned across 40 containers (441s wall).

For each route region the question is only whether SOME proposal's region CONTAINS it. If
none does, reward was never offered that region and no acquisition rule could have chosen
it.

## Coverage is 10.6%

| | |
| --- | ---: |
| route regions fully contained in some proposal | **0.106** (12 of 113) |
| median best partial overlap | 0.609 |
| p90 best partial overlap | 0.982 |
| when covered, proposals containing it | median 14 of 100 |

| target | regions | covered | median best overlap |
| --- | ---: | ---: | ---: |
| fa7 | 29 | 0.172 | 0.714 |
| jak2 | 29 | **0.103** | 0.565 |
| parp1 | 22 | 0.091 | 0.265 |
| 5ht1b | 15 | 0.067 | 0.545 |
| braf | 18 | 0.056 | 0.832 |

## The sharpest cut: single-region routes are never covered

| route kind | regions | fully covered |
| --- | ---: | ---: |
| single-region | 42 | **0.000** |
| multi-region | 71 | 0.169 |

ZERO of 42. The p90 partial overlap of 0.982 says proposals get *almost* the whole region
and miss an atom, which is the worst possible failure mode: the region looks nearly right
and the decisive atom is outside it.

Coverage also collapses with region size -- 0.667 at size 2, 0.000 at sizes 6, 7 and 11 --
so the larger coordinated regions that strong routes actually use are the least reachable.

## What this establishes

The exploitation plateau at -11.40 is NOT a reward or acquisition failure. Nine times in
ten the controller never proposes a region containing the atoms the route needed, so the
winning candidate is never generated and selection is irrelevant.

Combined with the action-parity audit, the controller's interface was missing:

| capability | status |
| --- | --- |
| retained-role retype | ADDED this session |
| retained-role deletion | ADDED this session (109 route subgoals use it) |
| coordinated multi-region proposal | still one subgoal at a time (51 of 77 routes use it) |
| **region selection reaching the needed atoms** | **10.6% coverage** |

`q_where(R | G)` has to become an explicit, reward-tiltable factor of the action. Until it
is, the controller is choosing well among options that mostly exclude the answer.

## Method note

Container draws cost about 1.25s against 120ms locally, so a first fan-out at 300 draws
per state overran the 2400s task timeout. 40 shards at 100 draws finished in 441s. Measure
per-draw cost in the container before sizing a fan-out.


---

# CORRECTED: coverage over comparable source-slot regions

Same 77 routes, 113 regions, 100 proposals per source state, 40 containers, 504s.
Teacher regions restricted to ORIGINAL slots; created atoms are a consequence of the
program rather than a precondition of the region.

| | |
| --- | ---: |
| route regions fully contained in some proposal | **0.681** (77 of 113) |
| median best partial overlap | 1.000 |
| when covered, proposals containing it | median 16 of 100 |

| target | regions | covered | median best overlap |
| --- | ---: | ---: | ---: |
| **jak2** | 29 | **0.966** | 1.000 |
| parp1 | 22 | 0.727 | 1.000 |
| fa7 | 29 | 0.552 | 1.000 |
| 5ht1b | 15 | 0.533 | 1.000 |
| braf | 18 | 0.500 | 0.969 |

| route kind | regions | fully covered |
| --- | ---: | ---: |
| single-region | 42 | 0.595 |
| multi-region | 71 | 0.732 |

Coverage by region size is high through size 7 and dips hard at size 8 (0.222 over 9
regions) before recovering at 9 and 10. That dip is real and unexplained; it is not the
JAK2 bottleneck.

## Consequence for the JAK2 diagnosis

WHERE is NOT the JAK2 bottleneck. The controller proposes a region containing the atoms
JAK2's strong routes needed 96.6% of the time, yet the N-removed core molecule was never
generated in 227 admitted candidates from the -11.30 state. The failure is therefore
DOWNSTREAM of region selection -- in the program proposed within a correctly chosen
region, or in which source roles that program binds to.

Multi-region parity remains a separate open milestone: 51 of 77 routes use coordinated
subgoals and `joint_multi_site` is still not wired into the controlled action. It is not
the explanation for this one-step JAK2 gap and should not be conflated with it.
