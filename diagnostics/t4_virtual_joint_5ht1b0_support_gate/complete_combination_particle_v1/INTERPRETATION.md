# 5HT1B-0 depth-three complete-combination particle gate

## Outcome

The strict gate is **FAIL** because shard 0 repeated one realization abstention at
the unchanged 4,000-expansion per-target cap. The abstaining target is not one of
the three teacher endpoints. No planning shard, valid target, or failed realization
was dropped, and no budget was widened.

The canonical support result is nevertheless positive. The task-blind proposer
autonomously recovered all three exact canonical teacher endpoints. Across the five
shards it committed 192 exact proposal occurrences at 192/192 exact-realization
precision. Deterministic endpoint union produced 182 unique valid endpoints and 162
unique complete program identifiers. These measurements establish retrospective
canonical support under the declared binding-particle-3 and depth-three setting.
They do not establish docking utility or prospective optimization performance.

## Frozen computation

The runner used proposer commit `2a3062f7a918c5debf7bbaa1f28a28db8a00ee22`
and census commit `ad52c0bf92ec7b97b8ac324586b5389dc4b44b45`.
The frozen default `VirtualJointRegionBudgets` select 48 constituents and allow
4,096 planning expansions per call. Therefore the minimum complete particle count
is:

```text
ceil(C(48,3) / 4096) = ceil(17,296 / 4,096) = 5
```

The five disjoint combination ranges were `[0,3459)`, `[3459,6918)`,
`[6918,10377)`, `[10377,13836)` and `[13836,17296)`. Every range completed.
Their valid STOP counts were 15, 81, 81, 47 and 30, totaling 254 and reproducing
the committed exhaustive census exactly. The proposal calls received only the exact
source graph and generic target-independent expert. The three teacher identities
were joined afterward.

| Shard | Combinations | Valid STOPs | Unique valid targets | Committed | Runtime (s) |
|---:|---:|---:|---:|---:|---:|
| 0 | 3,459 | 15 | 7 | 6 | 110.64 |
| 1 | 3,459 | 81 | 61 | 59 | 43.10 |
| 2 | 3,459 | 81 | 69 | 65 | 31.96 |
| 3 | 3,459 | 47 | 41 | 40 | 25.04 |
| 4 | 3,460 | 30 | 23 | 22 | 19.52 |

Parallel proposal wall time was 117.18 seconds and complete end-to-end wall time
was 152.21 seconds on five local CPU worker processes. Committed program lengths
ranged from 1 to 19 primitives. All 182 union endpoints passed the frozen molecular
validity, connectedness, charge-preservation, size and canonical identity checks.
There were zero oracle, docking, Modal and teacher-action calls.

## Teacher recovery and the radius-two diagnostic

The exact canonical teacher endpoints were recovered at union ranks 154, 153 and
147 for routes `2b0aff16...`, `88e87055...` and `bbd7db0d...`, respectively.
The separately reported frozen radius-two transformation evaluator returned 0/3.
Forensics show that this is not a contradiction and must not be interpreted as an
optimization failure.

The exact metric compares canonical molecular identity. The radius-two evaluator
instead constructs a source-slot-aligned colored before/after neighborhood. Each
recovered endpoint is canonically identical to its teacher but uses a different
persistent-slot state. Realized versus teacher changed-slot counts were 19 versus
20, 21 versus 22, and 16 versus 19. Each pair had different Weisfeiler-Lehman
hashes and failed exact colored-neighborhood isomorphism. Therefore this radius-two
source-mapping diagnostic is non-monotone with canonical endpoint identity in this
case. It is not a valid failure criterion for canonically recovered endpoints
without an additional canonical slot-alignment step.

## Capacity abstention

Shard 0 contained seven unique valid targets. Six realized and one returned
`search_limit_abstention`. The failed target is depth three, changes the active-atom
count from 39 to 36, and has a conservative atom-event primitive lower bound of
nine: three deletions and six live-atom restatements. Its persistent-coordinate
target distance is 32. At the frozen cap it expanded 4,000 states, attempted 32,079
actions and reached best mismatch one. It is not canonically identical to any of
the three teacher endpoints. The complete deterministic rerun and the separate
forensic replay reproduced this abstention without changing a budget.

The other eight compiler abstentions were frontier-exhaustion outcomes rather than
declared capacity exhaustion. They remain in the denominator of realization yield
and are not hidden by the 192/192 exact precision statement, whose denominator is
the set of successfully realized targets.

## Reproducibility and claim boundary

The authoritative machine-readable result is `result.json`, with payload SHA-256
`7ef559c529f307975ff966bbae5cd47dd365f519a670fe5943063a89beb674e2`.
Its physical SHA-256 is
`516649fdb3330c99a429cb97469ec84705ad127fd6bef9ea86c350324fce62ec`.
An independent complete rerun produced a byte-identical `result.json`. The forensic
artifact is `forensics.json`, with payload SHA-256
`b2b8ac784c6c7a7950d083ce2a5eeb37b9e239cb2b3497422e580846e747603e`.

This is retrospective, answer-known, zero-oracle support evidence on one 5HT1B-0
source. It does not authorize budget widening, a scored launch, or a broader
controller claim. The strict gate remains failed because its predeclared rule treats
any shard capacity abstention as failure.
