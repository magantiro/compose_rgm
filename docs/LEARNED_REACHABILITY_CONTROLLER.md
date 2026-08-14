# The learned reachability controller — design and preregistration

**Written after policy B failed 0/320 and BEFORE any implementation.** This is
not a replacement heuristic. It is the controller COMPOSE's own theory already
specifies, finally built at scale.

## Why B failed, precisely

B was `π ∝ R_θ(y|x)·QED(y)` — **prefer the locally nicer-looking street.** It
was never a GPS. Its 0/320 is therefore **not** evidence that COMPOSE cannot
optimize; it is evidence we were not using the control object COMPOSE was
designed around.

| | |
|---|---|
| exact rewrite kernel | which roads physically exist |
| `R_θ` | learned driving prior — which legal roads look chemically natural |
| goal `z` | the destination |
| **`h_b(x;z)`** | **the GPS** — from *here*, with `b` moves left, what is the chance I still reach the destination? |

B had the road network and the driving prior and **no destination-awareness at
all.**

## The object

```
h_φ(b, x; x_src, z)  ≈  Pr_{R_θ}( X_K ∈ B_z | X_{K−b} = x )      ∈ [0,1]
```

the **remaining-budget probability of eventual success under the frozen
reference process**. The controlled law is then exactly COMPOSE's:

```
P^φ_b(y | x, z)  ∝  R_θ(y|x) · h_φ(b−1, y; x_src, z)
```

For QED the terminal event is the **exact benchmark event** — no temperature,
no `QED^α`, no reward shaping:

```
g_z(x) = 1[ QED(x) ≥ 0.9  ∧  Sim(x, x_src) ≥ 0.4 ]
```

An imperfect `h_φ` can misallocate probability but **cannot leave the legal
support**, because support is set by the exact kernel, not by the controller.

## The exact rejection sampler

**Do not evaluate `h_φ` on all ~600 successors** — that rebuilds the
$24k full-fiber cost we already rejected. Because `h_φ ∈ [0,1]`, exact
rejection sampling is available:

At state `x` with `b` remaining:

1. compute the factorized `R_θ` marked law **once**
2. draw a legal mark `a ~ R_θ(·|x)`
3. execute **only that mark** → `y = T_a(x)`
4. reject and redraw if `y == x` (a virtual/self-loop mark)
5. evaluate `h_φ(b−1, y; z)`
6. accept `y` with probability `h_φ(b−1, y; z)`; else redraw from the
   **already-computed** base law

Then `Pr(Y=y | accepted) ∝ R_θ(y|x)·h_φ(b−1,y;z)` — **the learned Doob kernel
exactly, not a top-k approximation.**

### Why aliases come out right for free

`h_φ` is a function of the **canonical state** `y`, not of the mark. So if five
marks produce the same canonical `y`:

```
Σ_{a : T_a(x)=y} R_θ(a)·h_φ(y)  =  h_φ(y)·Σ_{a : T_a(x)=y} R_θ(a)  =  R_θ(y|x)·h_φ(y)
```

The alias masses sum **exactly** under the pushforward. This is why a
**mark-level control head would be dangerous** — its values could differ across
aliases of the same molecule and silently break the quotient. Measured alias
groups in COMPOSE are large, so this is not hypothetical.

### Batching that provably does not change the law

Draw `B` proposals and their uniforms **in advance**: `(a₁,u₁) … (a_B,u_B)`.
Execute the `B` rewrites concurrently, canonicalize, deduplicate repeated
successors, run **one batched `h_φ` forward** over the unique molecules, then
accept the **first `i` in original proposal order** with `uᵢ ≤ h_φ(yᵢ)`. If none
accepts, draw another batch.

Distributionally **identical** to sequential rejection sampling — the sequential
algorithm would examine the same proposals in the same order against the same
uniforms. We are only evaluating future proposals early.

> **Vectorize scheduling and neural computation. Never vectorize by changing the
> mathematical controller.**

`prepare_factorized_mark_batch` is still paid **once per committed state**, and
no 600-successor fiber is ever built for control.

## Training `h_φ`

`h` is a **probability**, not a reward model. Exact recursion:

```
h_0(x;z) = g_z(x)
h_b(x;z) = E_{Y ~ R_θ(·|x)} [ h_{b−1}(Y;z) ]
```

Two estimators of the **same** quantity, both used:

**Monte-Carlo future-event loss.** Run an ordinary base trajectory; every prefix
takes the terminal success/failure as a Bernoulli target. Cross-entropy has the
true reachability probability as its population optimum.

**Bellman consistency.** For sampled `x→y`, regress `h_φ(b,x,z)` on
`E_{R_θ} h_φ(b−1,Y,z)`, with a target network / stopped-gradient backup.

### Hindsight goal relabeling

One base trajectory trains **many goals**. From a terminal property vector
`F(X_K)` construct threshold goals, intervals, conjunctions, Chebyshev regions,
multiobjective boxes, and nearby positive/negative regions — then every prefix
becomes a training example for all of them.

So we do not train "a QED controller." We train a **universal molecular
reachability function** `h_φ(b, x; x_src, z)` over target regions in objective
space — which is exactly what the five-objective benchmark needs later.

**`R_θ` stays frozen throughout.**

## Architecture — do not make `h_φ` rediscover known chemistry

Inputs: frozen COMPOSE state embedding of `x` · remaining budget `b` · source
embedding and source-relative structure · **current normalized objective vector
`F(x)`** · source-relative similarity · goal-region descriptor `z` ·
path/protected-mask context where relevant.

Then a **small goal-conditioned head** → scalar logit → sigmoid.

Feeding `F(x)` and similarity directly is far easier than asking a GNN to infer
QED, similarity and the threshold from raw graph structure.

**Start with the frozen encoder plus a learned head.** If capacity is short,
**enlarge the controller, never `R_θ`.**

## The five-objective consequence

For MOLLEO Task 3, do **not** sweep weights. Maintain a nondominated archive
`A` and repeat:

1. identify an **under-covered / high-hypervolume-opportunity region** `B_z`
2. set that region as the goal
3. steer with the **same** `h_φ(·, z)`
4. insert every evaluated committed molecule into `A`
5. choose the next missing region

One learned GPS then yields QED editing, arbitrary property intervals,
conjunctions, dynamic goal changes, five-objective region targeting, front gap
filling, and path constraints **by masking support**.

## The validation ladder — frozen now

| stage | what | gate |
|---|---|---|
| **A** | exact small-state truth on the **enumerable 967-state system** | `h_φ` calibration vs **exact** `h`; Bellman residual; controlled terminal **TV vs exact Doob**; alias/quotient invariance; **rejection sampler vs explicit full-fiber self-normalization** |
| **B** | the existing **64×5 QED dev panel**, with B as the frozen weak baseline | QED actually climbs; **nonzero** `QED≥0.9 ∧ sim≥0.4`; `h_φ` calibrated on which states remain reachable; workable acceptance rate; **no support or validity change** |
| **C** | **fresh 128×20** disjoint validation | then and only then **freeze** |
| **D** | official GrIDDD **800×20, once** | ablation reuses the identical controller with size-changing successors disabled |

**Stage A is not optional.** It is the only place exact `h` exists, so it is the
only place the sampler and the learned value can be checked against truth rather
than against each other.

## The optimization invariant — frozen

> **Optimization may change how equivalent computations are scheduled, batched,
> cached, or vectorized. It may NOT change the legal support, the canonical
> quotient, the controller equation, the random law, or the accepted trajectory
> distribution.**

| permitted | barred |
|---|---|
| batch `h_φ` evaluation | shortlisting successors because it is faster |
| share the frozen graph encoder | replacing `R_θ·h_φ` with top-k |
| cache source representations | approximating aliases separately |
| deduplicate canonical successors before value evaluation | changing acceptance semantics |
| vectorize goal heads across states/goals | silently substituting an argmax policy |
| prefetch rejection proposals **retaining proposal order** | |
| batch label-generation rollouts | |

## The escalation ladder — do not jump to SMC

**One `h_φ`, two inference mechanisms.** They are not competing methods.

```
Level 1   learned h_φ + exact rejection sampler   →  MEASURE ACCEPTANCE RATE
          works and optimization improves?  STOP. Simplicity wins.

Level 2   twisted SMC, same h_φ as the twist      →  only if the value model is
          good but acceptance collapses on rare regions
```

SMC is the **escalation for rare events**, which is exactly what it is for.
Escalate on a **measured** acceptance rate, never on assumption.

## What is NOT promised

**This is a principled architecture with bounded-state precedent, not a result.**
It is not promised to beat GrIDDD or MOLLEO. If it fails Stage A it does not
proceed; if it fails Stage B the honest finding is that amortized reachability
does not transfer from the enumerable system to full molecular space, which is
itself worth reporting.

**Still barred:** `QED^α`, temperatures, shortlist sizes, reward shaping, or any
knob fitted to a benchmark-shaped number.
