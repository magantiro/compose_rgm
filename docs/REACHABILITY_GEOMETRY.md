# What the horizon diagnostic is actually for

**Canonical framing. The H24 run is not a hyperparameter search.**

It asks whether the frozen `R_θ` **contains useful routes to the target**, and
what the **geometry** of those routes looks like. That determines whether a
strong GPS can exist at all, and how to train it.

## The dream result is NOT high unguided reachability

> If unguided `R_θ` already hit 45 % QED success, we would barely need a
> controller — and the paper's central claim would evaporate.

What we actually want is a **well-behaved reachability hierarchy**:

| observation under unguided frozen `R_θ` | why it is ideal |
|---|---|
| `≥0.80` rises quickly, then saturates | easy regions are genuinely reachable |
| `≥0.85` rises more slowly, saturates later | harder goals need longer paths |
| **`≥0.90` rare but clearly NONZERO** | **the target is in the support, but needs strong control** |
| `≥0.95` extremely rare or absent | sensible difficulty ordering |
| first-hit time shifts later as the threshold rises | **remaining budget genuinely matters** |
| curves flatten by ~H12 rather than H6 | gives a principled, non-arbitrary horizon |
| similarity stays reasonable along successful paths | routes to high QED are not "destroy the source" |
| many genuine hits are later lost if execution continues | directly validates native STOP / anytime |

## The distinction this diagnostic draws

```
        bad GPS                 vs        destination unreachable
                                          on our road network
```

**A controller cannot manufacture edges `R_θ` does not support.** If `≥0.90` is
literally unreachable even at H24 while `≥0.85` saturates strongly, that is not
a controller failure — it says the target is an extremely rare tail of the base
process, or the frozen support makes it practically inaccessible. **Separating
those two diagnoses is scientifically essential**, and nothing else we run
distinguishes them.

## Why the ladder matters: the controller's actual job

At a molecule `x` with ~500 legal successors, suppose:

```
300 drift around QED 0.7–0.8
150 can reach 0.80
 30 can reach 0.85
  3 preserve a route that can eventually reach 0.90
```

`R_θ` correctly says **all of these are chemically plausible**. It *should not*
know which three we want — **QED was never part of its training.**

Policy B computed `R_θ(y|x) × QED(y)` and failed because **local QED barely
distinguishes the roads**. The controller instead learns

```
h_b(x,z) = P_{R_θ}( reach z within b moves | x )
```

and uses `h_{b−1}(y,z)` on each candidate:

```
P*(y|x,z)  ∝  R_θ(y|x) · h_{b−1}(y,z)
```

> **A successor can have LOWER immediate QED and still receive enormous control
> mass, because it is one of the few states from which 0.90 remains reachable.**

That is the GPS. **The controller's job is not to invent chemistry that is not
there — it is to concentrate probability onto the rare productive branches.**

## The three things the diagnostic feeds

**1 · Learn the whole reachability landscape, not `0.90` as a binary
classifier.** The thresholds × similarity floors × remaining budgets form a
structured goal language. The model sees *"from here with 2 edits left, 0.80 is
easy, 0.85 possible, 0.90 essentially not"* versus *"from there with 5 edits
left, 0.90 has a real route."* That is vastly richer supervision than one sparse
endpoint event, and it approximates a **molecular reachability field over
`(x, b, z)`** — exactly what gets reused for conjunctions and Pareto regions.

**2 · Spend labelling effort at the FRONTIER, not on more random roots.** If the
curve shows that states around QED 0.84–0.88 with good similarity are the
**gateways** through which `0.90` hits occur, then take *those exact molecules*
and run many frozen-`R_θ` continuations from them. That yields `h_4(x,z) = 0.37`
instead of one noisy Bernoulli label. **We are not changing the answer — we are
estimating the same conditional probability far more efficiently.**

**3 · Drive native inference.** `R_θ` says which edits are plausible; `h_φ` says
which plausible futures retain a route; COMPOSE samples their product; and the
instant `x_t ∈ B_z`, execute **STOP**.

## The result worth wanting: amplification

An exact Doob transform **conditions on a rare event**, turning the rare
productive subset *into* the controlled distribution. So low unguided
reachability is the setup, not the obstacle:

```
R_θ alone            ~2 %   reachability in the qualified horizon
R_θ × QED(y)   (B)    ~0 %   local tilt cannot distinguish the roads
R_θ × h_φ             large amplification  ← the claim
```

**That story is far stronger than discovering `R_θ` accidentally optimizes QED.**
It demonstrates the decomposition directly:

> ### **Plausibility is not purpose.**

And it would not be the first time: the exact-target work already found
**`R_θ`-only prioritization was the weakest prioritizer**. This would establish
the same principle for a **target-free property objective**.

## What we want from H24 — stated before reading it

**Not** "please give us lots of `0.90` hits." A hierarchy:

```
0.80 easy  >  0.85 harder  >  0.90 rare  >  0.95 very rare
```

with harder regions needing more remaining budget, and curves that saturate. That
tells us six things at once:

1. there are roads to the destination
2. the destination genuinely requires navigation
3. remaining budget matters
4. a non-arbitrary horizon exists
5. where to concentrate continuation labelling
6. we have the data structure needed to train `h_φ`

Then the target result is:

> **A universal region-conditioned `h_φ` learns these reachability contours and
> turns a rare event under the same frozen goal-independent `R_θ` into a
> high-probability controlled outcome — without retraining the molecular
> dynamics.**

If that holds, the QED experiment stops being "we got a decent benchmark number"
and becomes a clean empirical demonstration of the central COMPOSE architecture.
