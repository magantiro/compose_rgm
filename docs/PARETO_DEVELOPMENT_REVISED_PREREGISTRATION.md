# Pareto development, REVISED after repaired P3/P4 — preregistration

**Status: DESIGN ONLY. NOT LAUNCHED.** Supersedes the arm hierarchy in
`docs/PARETO_DEVELOPMENT_PREREGISTRATION.md`; its four frozen questions, its
escalation rule and its instrumentation list remain in force.

## What P3 changed

| contrast | HV, COMPOSE − fairly funded gen_rank | | algorithmic requests/source |
|---|---|---|---:|
| **P3 greedy closed-loop** | **+0.6678** [+0.5169, +0.8094] | 12W/0L | **17,722** |
| P4 verified closed-loop | +0.5668 [+0.4763, +0.6390] | 12W/0L | 397,706 |

**Greedy closed-loop beats a fairly funded generate-and-rank by MORE than
verified does, at 22× lower oracle cost.** So the Pareto mechanism is

> feed purpose back into each evolving molecular state

not

> do expensive deep lookahead everywhere.

### The hierarchy changes accordingly

- **Primary Pareto controller: greedy closed-loop.** It is the clean method
  result and the production path.
- **Verified: a secondary increment**, kept because *"does future-aware lookahead
  add anything beyond closed-loop feedback?"* is a real question. Current answer:
  modestly yes (+0.089 set-level over greedy), **not enough to justify making it
  the default.**

This is a *richer* result than "the big planner won", and it is consistent with
the rest of the paper rather than in tension with it: on exact-target recovery
future reachability mattered enormously; on Pareto, simpler state feedback is
sufficient for most of the benefit. **The control machinery can be matched to the
design problem rather than always invoking maximal planning.**

## Two budget-derived shortlists — not a sweep

Both `K` values come from the **same external 10,000-query-per-source rule**,
applied to scoring-event counts read from the frozen implementation. Neither was
chosen by looking at an outcome; `K_V` was fixed before P3 existed.

```
greedy    E_G = 5 prefs × 6 steps                        =  30 events
verified  E_V = 5 prefs × (6 + 8 rollouts × 15 steps)    = 630 events
```

Both event counts check out against measurement — implied fiber widths 590.7 and
631.3, against the census mean of 586.

| variant | K | retains | predicted req | reduction |
|---|---:|---:|---:|---:|
| **budgeted greedy** | **333** | **56.4%** of fiber | 9,990 | **1.8×** |
| budgeted verified | 15 | 2.4% of fiber | 9,450 | 42.1× |

**The asymmetry is the point.** Greedy needs only a mild cut to reach the
contemporary regime; verified needs 42×. Retaining 56% of the fiber is far more
likely to preserve goal-useful moves that `R_θ` alone might rank poorly — and we
already know from exact-target recovery that **plausible ≠ goal-useful**.

`K = 15` remains recorded as the aggressive stress test, but **the paper no
longer lives or dies on it.**

## Panel: FRESH sources, and why

**Do not use the original 12 as the scientific efficiency panel.** Those sources
have established the smoke, triggered the efficiency branch, exposed the oracle
problem, and motivated this hierarchy change. Even though `K_V` was derived
outcome-independently, **the branch exists because of what those sources said.**

- **Fresh held-in panel, 20–24 sources, frozen before any outcome.** All arms
  paired within each new source.
- **The original 12 are for regression and implementation checks only** —
  confirming the shortlist is not catastrophically broken. Never the load-bearing
  estimate of HV retention.

## One panel answers capability AND operating point

Running a huge exhaustive development, then an efficiency development, then a
scalable development is wasteful. Per fresh source:

| arm | role |
|---|---|
| **A — full-fiber greedy** | the reference ceiling |
| **B — budgeted `R_θ`-shortlisted greedy, K=333** | **primary scalable COMPOSE** |
| **C — budgeted `R_θ`-shortlisted verified, K=15** | secondary future-aware operating point |

Full verified across the whole panel is **questioned, not assumed** — its
scientific role is now secondary and it costs 22× arm A.

The two budgeted arms share approximately the same query ceiling, which makes a
genuinely useful algorithmic question available:

> **At the same contemporary oracle budget, is cheap closed-loop feedback or
> compressed future-aware control the better operating point?**

Either answer is informative.

### Reported outcomes

Preference responsiveness; final HV; HV-AUC; nondominated count and front spread;
**HV retention versus full-fiber greedy**; query reduction; chemical-envelope
fidelity; and kernel / trajectory / query resources **separately**.

### What would convince us

For budgeted greedy: **≥90% of full-greedy HV** while moving 17.7k → ≤10k
queries. 95–100% would be better. **It does not need a 42× reduction, because
greedy already removed most of verified's computational disaster.**

### If it fails

Do **not** design increasingly clever shortlist policies. Failure is believable —
exact-target recovery already established that plausible ≠ goal-useful. The
conclusion would be that COMPOSE provides powerful exhaustive finite-horizon
control while trading substantial objective-query cost for it, and we **drop the
oracle-budget-SOTA aspiration** rather than engineering for another month.

## The nondominated-cardinality deficit — diagnosis, not ceiling

Repaired P3/P4 both show COMPOSE losing on nondominated count (P3 −2.250, P4
−2.417, 0W/11L) while winning enormously on hypervolume.

**That is not an intrinsic limitation. It is evidence that the outer allocation
is primitive.** COMPOSE currently solves five *independent* preference-conditioned
control problems — `max U_{w_i}(x_H)` for a fixed grid `w ∈ {0.1,…,0.9}` — and
never optimizes `HV({x_H^(1)}, …)` or front spread. Generate-and-rank gets ~220
trajectories and cherry-picks five endpoints, so of course it assembles more
mutually nondominated points. **The surprising fact is that five directed
trajectories already crush 220 on hypervolume.**

There are two optimization layers, and only the first is currently good:

| layer | question | status |
|---|---|---|
| **inner control** | given a requested region, what edit next? | **excellent — P3 proves it** |
| **outer allocation** | given the front so far, where should the next trajectory go? | **primitive — a fixed grid** |

### `QUEUED_CONDITIONAL — NO DESIGN / NO RUN`: gap-filling aspiration control

**Trigger:** the fresh panel reproduces high HV + strong P3 + lower ND cardinality
/ clustering. Not before — we must not change candidate allocation, oracle
shortlist and controller at once, or we will not know what caused any improvement.

**Method, deliberately parameter-light.** Start from the two extremes. Then place
an **aspiration point in the largest uncovered objective-space gap**, direct
COMPOSE there, recompute the gap, repeat — five trajectories total.

```
A_2 → a_3 → x_3 → A_3 → a_4 → x_4 → …
```

**Preferred over an HV-acquisition estimator** (`argmax E[HV(A ∪ X) − HV(A)]`),
which is theoretically prettier but needs an expectation estimator and therefore
another model, rollout or acquisition mechanism. Largest-gap aspiration is
deterministic, transparent, parameter-light, easy to preregister and directly
visualizable — and it sidesteps the known problem that fixed scalar weights do
not map cleanly onto ordered discrete molecular regions.

**Comparison, with all methods returning exactly five molecules:** fixed-grid
COMPOSE vs adaptive-front COMPOSE vs generate-and-rank on the established
resource frontier. Measure HV@5, |ND@5|, front spread/coverage, HV-AUC over
trajectory count.

**BARRED: do not optimize ND count directly.** That produces five mediocre points
that happen not to dominate one another. The objective is **front coverage /
marginal hypervolume**, with ND cardinality as a **secondary diagnostic**. We
want *strong* **and** *spread out*, not merely *different*.

## Three orthogonal efficiencies — reported separately, never merged

| efficiency | mechanism | measure |
|---|---|---|
| **oracle** | `R_θ` shortlist | expensive property queries per decision |
| **trajectory** | archive-aware front filling | complete trajectories to build the front |
| **systems** | parallelism, caching, kernel engineering | wall time for identical computation |

Systems work remains bound by the standing rule: **no performance optimization
ships into a claim-bearing experiment unless it reproduces trajectories, actions
and accounting exactly.** The scorer-batching episode is why.

The resulting arc:

```
exact exhaustive controller   →  establishes capability
plausibility shortlist        →  makes each decision cheaper
archive-aware allocation      →  makes each trajectory more useful to the front
```
