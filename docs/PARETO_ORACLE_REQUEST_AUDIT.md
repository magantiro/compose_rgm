# Audit: is `gen_rank`'s doubled `raw_oracle_calls` algorithmic demand or harness overhead?

**Verdict: harness overhead. The fix is warranted — but the framing "the cache
suppressed the second native call, therefore it was double-billed" was WRONG,
and is withdrawn.**

## Why that framing was wrong

The frozen convention is explicit and was designed to survive exactly this
argument:

```
oracle_requests   every scoring request, duplicates included -- ALGORITHMIC DEMAND
evaluator_calls   expensive executions after caching        -- REAL WORK
```

> *"caching may reduce evaluator work, but it cannot erase wasteful algorithmic
> requests."*

So a cache hit **should** increment `raw_oracle_calls`. "The second call was a
cache hit" is not evidence of over-counting — under this convention it is
evidence of nothing at all. The only admissible question is:

> **Did the algorithm request the score twice, or did our harness?**

## The call graph, traced

**Call 1 — `unguided_run`, `pareto_control.py:217`**

```python
for _ in range(budget):
    rows = metered.successors(current)        # R_theta probabilities ONLY
    ...
    current = keys[rng.choices(...)]          # sampling, no objective read
return Trajectory("unguided", nan, states, current,
                  metered.z(current), complete, metered.ledger)   # <- CALL 1
```

`metered.z` is **never called inside the generation loop.** The trajectory is
fully determined by `R_theta` and the RNG before any objective is read — the arm
is preference-blind by construction, as its own docstring states. Call 1 happens
*after* generation is complete, purely to populate the `Trajectory.endpoint_z`
field.

**Call 2 — `generate_then_rank`, `pareto_control.py:353`**

```python
endpoints = [t.endpoint for t in pool]
z = metered.z_many(endpoints)      # <- CALL 2, every entry a cache hit
for weight in preferences:
    scores = scalarize(z, weight)
    pick = pool[_argmin_stable(scores, endpoints)]
```

Call 2 drives the ranking decision.

## Why call 2 is harness overhead and not algorithmic demand

`z_many` re-requests values **the harness is already holding**. Every element of
`z` is identical — the same cached array object — to the `endpoint_z` already
attached to the corresponding `Trajectory` by call 1. The identical ranking is
obtainable as

```python
z = np.stack([t.endpoint_z for t in pool])     # zero additional requests
```

with bit-identical results, because `t.endpoint_z` was populated from
`metered.z(t.endpoint)` and `z_many` looks up `metered._z[t.endpoint]` — the
same key into the same cache.

**A competent generate-and-rank implementation scores each generated molecule
once.** Nothing about the *method* requires asking twice; our `Trajectory`
dataclass eagerly attaches a score, and then the ranking step asks again for a
value it was handed. That is a data-structure convenience of ours, and
`gen_rank` is a **controlled internal comparator we author** — so its recorded
algorithmic demand must reflect a competent implementation of the method, not
our redundancy.

## The magnitude, and the part that actually matters

The redundancy is **not** specific to `gen_rank` in kind — it is specific in
*effect*, because it costs exactly one request per trajectory regardless of how
many the arm makes:

| arm | mean raw | mean native | requests/trajectory | redundant share |
|---|---:|---:|---:|---:|
| `gen_rank@verified` | 133.0 | 57.5 | **2.0** | **50%** |
| `gen_rank@greedy` | 5.5 | 2.6 | **2.0** | **50%** |
| `greedy_pref` | 21,004 | 10,936 | ~4,200 | 0.02% |
| `verified_pref` | 443,240 | 269,884 | ~88,600 | 0.00% |
| `unguided` | 5.0 | 5.0 | **1.0** | **none** |

`raw == 2 × n_trajectories` holds on **all 24** `gen_rank` arm-instances,
exactly.

Two corrections to my earlier report:

- **`unguided` is not affected at all** — raw 5.0 equals native 5.0, one request
  per trajectory, no cache hits. It never passes through `generate_then_rank`.
  An earlier table implied otherwise; that was an artifact of the ratio I
  printed, not a finding.
- **The COMPOSE arms carry the same one-per-trajectory redundancy** at the
  `Trajectory` construction site. It is real, and at 0.02% and 0.00% it is
  numerically invisible.

So the honest statement is not "gen_rank was singled out". It is: **an identical
shared-path redundancy is catastrophic for a low-demand arm and undetectable for
a high-demand one**, and `gen_rank` is the low-demand arm. Correcting it makes
the baseline look **cheaper** — again against COMPOSE.

## What is fixed, and what is deliberately not

**Fixed:** the new metered matcher reads `run.endpoint_z` instead of
re-metering. Regression is by construction — same generated molecules, same
scores, same ranking, same selected front — since both paths resolve the same
cache key to the same array.

**Deliberately NOT fixed: the COMPOSE arms' `Trajectory`-construction request.**
Correcting a 0.02% accounting artifact there would require **re-running the
COMPOSE arms**, and would change their committed counters — which would destroy
the serial baseline the fan-out parity replay must match exactly. The
redundancy is disclosed here and left in place. Removing it belongs with a
future re-run, not with a speed change and not with a baseline repair.

## Superseded

The P3/P4 oracle-demand accounting in `diagnostics/pareto_oracle_accounting_audit.json`
is **superseded for the `gen_rank` arms**: their `raw_oracle_calls` overstates
algorithmic demand by exactly 2×. Those contrasts were already withheld as
`INVALID_CONTRAST` on the kernel-budget defect, so nothing reported is affected.

The withdrawn statement, recorded so it is not repeated:

> ~~"the cache suppressed the second native call, so raw was double-billed"~~ —
> a cache hit incrementing `raw_oracle_calls` is the convention working as
> designed, not evidence of over-counting.
