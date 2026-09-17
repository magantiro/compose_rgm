# WHERE support audit: does the controller ever propose the region a strong route needs?

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
