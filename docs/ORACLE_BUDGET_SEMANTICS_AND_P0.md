# Oracle-budget semantics, and the P0 amortization/partition audit

**Project-wide ruling plus the P0 outcome. Binds every cost claim.**

---

## 1. RULING: what the ≤10,000 budget counts

> **For any PMO-style external oracle-budget claim, the denominator is the number
> of UNIQUE CANONICAL MOLECULES ACTUALLY EVALUATED BY THE TRUE TASK ORACLE — not
> raw algorithmic requests.**

PMO defines sample efficiency as the number of molecules evaluated by the oracle,
and its 10K limit in those terms. A repeated lookup of a molecule whose oracle
value is already cached is **not another expensive experiment**. Counting it again
would penalise COMPOSE for implementation and control reuse rather than for oracle
sample complexity.

Any benchmark-only scoring that invokes the true oracle on a **previously unseen**
molecule **does** consume the same budget.

### Four costs, never one number

| # | cost | what it is | role |
|---|---|---|---|
| 1 | **unique true-oracle evaluations** | distinct canonical molecules actually scored | **the load-bearing external budget axis** |
| 2 | **algorithmic oracle requests** | every ask, cache hits included | important **secondary diagnostic** of controller information demand |
| 3 | **generative/process compute** | kernel enumerations, GNN forwards, rewrites, canonicalization | separate — and currently where most runtime lives |
| 4 | **total systems compute** | CPU/GPU, surrogate fitting, search/training overhead | separate — other methods can spend heavily here while using few true-oracle calls |

**Both of these are wrong:**

- ~~"COMPOSE uses 20× more oracle calls, therefore it is 20× more expensive."~~
- ~~"Our oracle is cheap, so query count doesn't matter."~~

The first confuses demand with cost; the second abandons the sample-efficiency
proxy that makes the number meaningful when the oracle is eventually docking,
simulation or assay.

### The measured reason this matters here

Profiled on this task: **~250k unique objective evaluations ≈ 12 minutes**, while
**~393 successor-kernel enumerations ≈ 46 minutes** — the kernel is ~80% of wall
time. **Property evaluation is not the current wall-time bottleneck.**

Report **measured oracle-seconds alongside** query counts to make that tangible —
but never *instead of* them.

### Where this leaves greedy COMPOSE

Per source, distinct molecules evaluated: **min 4,129 · median 10,352 · mean
10,936 · max 21,048 · 7 of 12 over 10k.**

So the primary controller is **already near the contemporary regime** without any
approximation — but it is **not** "10k compliant", because the budget is per run,
not a cohort average.

---

## 2. P0a — CLOSED. The sharing was already implemented

**Measured:** across the five greedy preference trajectories, 360 state
expansions collapse to 194 unique — a **1.86×** sharing factor.

**But `unique expansions == kernel_calls` on 12 of 12 sources.** The app builds
one `MeteredProcess` and passes it to all five preference runs, so the
enumeration and objective caches are **already shared across preferences**. The
saving is real and **already banked**; none of it is available.

### WITHDRAWN

> ~~"Full-fiber greedy under exact sharing costs 9,550 requests per source —
> already under the 10,000 budget."~~

That was **mean unique expansions × mean fiber width** — a product of averages,
which is not the average of products, and it concealed a 47,087 worst case. **No
source is under 10k on requests. 5 of 12 are under on evaluator calls.** Budget
compliance is **per source**; cohort averages may not be used for it.

## 3. P0b — KILLED

There is no shared-execution prototype to build. It would reproduce existing
behaviour with a different implementation and identical counters: engineering
with no scientific return.

## 4. P0c — the mathematics is exact, and the first measurement is a negative

**The partition is exactly computable.** With `g = utopia − z`:

```
s_y(w) = max(w·g₁, (1−w)·g₂) + ρ·(g₁+g₂)
```

The augmentation `ρ·(g₁+g₂)` is **constant in w** — a per-candidate offset. A max
of two lines in `w` is convex with exactly **one** structural breakpoint at

```
w*_y = g₂ / (g₁ + g₂)
```

so the controller's choice is the **lower envelope of 2N line segments** —
computable exactly, near-linear, **no combinatorial explosion at a state.**

**But the measured partition is tiny**, on real molecular objective vectors:

| set | N | preference intervals | distinct winners |
|---|---:|---:|---:|
| src000 endpoint pool | 1,025 | **2** | 2 |
| all 12 pools | median 220 | **median 5**, max 8 | median 5 |

**A thousand candidates collapse to two preference regions.** Five weights is not
obviously undersampling.

### Caveats that stop this being conclusive

1. These are **endpoint pools, not true one-step successor fibers.** The real
   fibers need the kernel, which is Gate-0 blocked locally.
2. **One-step sparsity does not prove trajectory-level collapse** over H=6.

### NEXT GATE — confirm on the real fibers

On the existing 12 greedy-development sources, using the **actual visited decision
states**, not selected examples: derive the exact preference regions over
`w ∈ [0,1]` with frozen deterministic tie-breaking, then recursively characterise
the distinct terminal trajectories and endpoints induced by the **complete
preference continuum**. Development/design evidence only; no fresh panel, no new
claim.

### The gate is THREE-way, not two-way

An earlier framing said a guard hit means "the continuum sweep is unaffordable,
which argues for aspiration just as directly as a collapsed partition." **That is
wrong, and it would have biased the branch toward aspiration.** A guard hit
establishes only that the exact preference tree is substantially richer than the
five-weight controller — and *rich* can mean two opposite things.

| # | outcome | consequence |
|---|---|---|
| 1 | **sparse tree, little extra breadth** | the scalarized controller effectively collapses the continuum → **aspiration branch earned** |
| 2 | **manageable tree, materially more breadth** | the **exact preference sweep is sufficient** → no aspiration controller |
| 3 | **tree exceeds the 240-expansion guard** | **do not decide yet.** Classify using the branches already generated |

For outcome 3, measure against cumulative expansions and unique-oracle cost:

```
k ↦ HV(A_k)        k ↦ |ND(A_k)|        k ↦ spread(A_k)
```

- **continued strong front gain** → the continuum is genuinely rich; use a
  **budgeted traversal of the existing tree**, not a new algorithm.
- **rapid saturation or redundancy** → scalarization-induced branching is
  inefficient → **aspiration earned**.

**Do not change the frozen traversal to make that curve look better, and do not
add another algorithm on the strength of this diagnostic.** The running probe
persists every fiber, every objective vector, and every leaf path in DFS
completion order, so all three curves are reconstructible **offline** from the
shard — no change to the running app is needed or permitted.

The endpoint-pool result makes the second outcome the likelier one.

**Claim discipline if the sweep is retained:** it is *"the complete set of distinct
trajectories and endpoints induced by the frozen COMPOSE preference controller
over the continuous preference range"* — **never "the true Pareto front."** Take
its nondominated subset and measure how well that covers the attainable front.

## 5. The escalation, in order

```
five weights  →  exact full preference sweep  →  adaptive gap-filling / aspiration
```

**Aspiration control is authorised only if the complete sweep still leaves the
front clustered.** If it is triggered, keep it deterministic and minimal: start
from the two extremes, place each next aspiration point in the largest uncovered
gap between neighbouring nondominated points in normalised property space, run
COMPOSE toward it, update the archive, repeat to five trajectories. **No learned
acquisition model, no hypervolume-optimizer network, no tuned heuristic stack.**

**BARRED: optimizing nondominated count directly.** That yields five mediocre
points that merely fail to dominate one another. **HV and front coverage are the
objectives; ND cardinality is diagnostic.** The target result is HV retained at
95–100% of current COMPOSE while coverage and ND roughly double.

## 6. `K_G = 333` — frozen, and deferred

Retained as `INDEPENDENT_EXECUTION_10K_SOLUTION`. **Do not run it or its variance
study until the final outer Pareto controller is chosen.** Query compression must
be evaluated on the algorithm we intend to publish, not on a fixed-grid
controller likely to be superseded — and choosing `K` for a controller we then
replace would waste the panel.

**Fresh-panel `n` and cohort execution remain paused.** If the outer controller
changes, the load-bearing estimand changes with it, and so does the variance and
therefore `n`.

## 7. A second, orthogonal saving this may unlock

If each trajectory is chosen to fill a missing region, **fewer useful
trajectories may be needed to construct a good front.** That is a saving on the
*trajectory* axis, independent of pruning successors on the *oracle* axis — and
it would make the ≤10k unique-evaluation target easier rather than harder.
