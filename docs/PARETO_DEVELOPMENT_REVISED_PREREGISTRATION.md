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
| *full-fiber verified* | **NOT RUN on this panel** |
| **B — budgeted `R_θ`-shortlisted greedy, K=333** | **primary scalable COMPOSE** |
| **C — budgeted `R_θ`-shortlisted verified, K=15** | secondary future-aware operating point |

**SETTLED: full-fiber verified does NOT run on this panel.** It already did its
job in the 12-source smoke; paying ~22× again on 24 fresh sources would mostly
re-establish a ceiling we no longer need. Arm C is therefore measured against
arm A (full greedy) and against arm B at matched budget — never against a
full-verified ceiling, which is why C carries no noninferiority requirement.

The two budgeted arms share approximately the same query ceiling, which makes a
genuinely useful algorithmic question available:

> **At the same contemporary oracle budget, is cheap closed-loop feedback or
> compressed future-aware control the better operating point?**

Either answer is informative.

### Reported outcomes

Preference responsiveness; final HV; HV-AUC; nondominated count and front spread;
**HV retention versus full-fiber greedy**; query reduction; chemical-envelope
fidelity; and kernel / trajectory / query resources **separately**.

### The two arms answer DIFFERENT questions — do not conflate them

**Budgeted greedy is not a "42× efficiency" result. Its win is BUDGET
COMPLIANCE**: 17.7k → ≤10k with little or no loss of front quality. **The
dramatic scaling test is budgeted verified**: 397k → ~9.5k.

**SETTLED: full-fiber greedy is the correct ceiling, and the benchmark will NOT
be made stricter to create room for a win.** If K=333 retains 56% of the fiber
and performs almost identically to full greedy, that is not a boring null — it is
the production result:

> **Almost half of expensive successor scoring was unnecessary for preserving
> Pareto performance.**

`K_G = 333` came from an external 10,000-query budget *before outcomes*. That is
what makes the experiment defensible, and harshening it because 333 "looks easy"
would destroy exactly that property.

### Preregistered success criteria

**Primary — budgeted greedy (arm B vs arm A):**

> the lower bound of the source-level 95% CI on mean HV retention
> `HV_B / HV_A` remains **above 0.90**, while staying within the 10k-query budget.

Retention rather than absolute HV, because full-greedy difficulty varies by
source (smoke sd 0.261 across sources). Also required: **preference
responsiveness does not materially collapse**, and **chemical fidelity does not
worsen unexpectedly**.

**Secondary — budgeted verified (arm C vs arm B), at matched ≤10k budget:**

> `HV_verified@15 − HV_greedy@333`, plus preference response and resource use.

**No noninferiority-to-full-verified requirement**, because that ceiling is
deliberately not run here. Arm C's question is a practical operating-point one:
*at the same oracle budget, is mildly pruned greedy or aggressively pruned
future-aware control better?* Either answer is informative.

### Panel size — the first calculation was WRONG and n is NOT yet fixed

**Withdrawn.** The initial planning computed the lower CI *at* the true value:
`lower = r − 1.96·sd/√n`. That asks *"if the sample mean lands exactly on the
truth, does the interval clear?"* — a **~50%-power question by construction**,
because it ignores sampling variation in the mean itself. It also used a normal
approximation while the final analysis reports a **source-level bootstrap**, and
the two do not agree: the bootstrap of a ratio is skewed, the normal
approximation symmetric.

Redone with the actual estimand and the actual interval
(`scripts/pareto_scalable_power.py`, log-normal so a simulated retention cannot
go negative):

| true retention | n=16 | n=20 | **n=24** | n=30 |
|---|---|---|---|---|
| 0.95 | 0.437 | 0.515 | **0.604** | 0.667 |
| 0.97 | 0.708 | 0.804 | **0.867** | 0.923 |
| 1.00 | 0.948 | 0.981 | **0.992** | 0.998 |

**n=24 gives 60% power at a true retention of 0.95** — not the ~50% the bad
calculation implied it was clearing, but nowhere near adequate. The stated intent
was to size for the good-but-not-perfect case, and n=24 does not meet it.

**n required for 80% power**, across the true retention and the retention-ratio
sd:

| sd | ret=0.95 | ret=0.97 | ret=0.99 |
|---:|---:|---:|---:|
| **0.118** (conservative proxy) | **40** | 24 | 16 |
| 0.080 | 24 | 16 | 16 |
| 0.050 | 16 | 16 | 16 |

**The answer hinges entirely on a variance we are guessing.** The 0.118 proxy
comes from swapping *controllers* (verified vs greedy) — a far larger
perturbation than shortlisting to 56% of the fiber — so the true retention sd is
plausibly much smaller, and at sd ≤ 0.08 n=24 is already sufficient.

### Resolution: MEASURE the variance on the existing 12 first

The 12 are already designated for variance/power planning and implementation
checks, and never as the load-bearing estimate. Running arms A and B there is
**cheap** — full and budgeted greedy are ~16 kernel calls per source, against
arm C's ~393 — and it converts the guessed sd into a measured one.

**Then fix n from the measured sd, and only then select the fresh cohort.**

This costs little, removes the largest remaining uncertainty in the design, and
uses the 12 for exactly what they were sanctioned for.

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
