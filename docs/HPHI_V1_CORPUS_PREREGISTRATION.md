# Region-`h_φ` V1 rollout corpus — preregistration

**Frozen BEFORE any rollout runs and BEFORE any goal-event census is read.**

## The corpus

> **Region-`h_φ` V1 rollout corpus: 1,024 × 8 × H6. No automatic scale-up.**

| field | decision |
|---|---|
| training sources | **1,024** |
| trajectories per source | **8** |
| horizon | **6 committed edits** |
| total trajectories | **8,192** |
| total transitions | **49,152** |
| source pool | non-Jin, RDKit QED ∈ [0.70, 0.80] |
| selection | deterministic sha256 rank, **QED-stratified** |
| official Jin 800 | **excluded** |
| 64-source controller dev panel | **excluded from training** |
| fresh controller validation | **128 sources, reserved now** |
| `R_θ` | **frozen** |
| first inference mechanism | **exact Stage-A1 rejection sampler** |

**Built and verified:** eligible pool 66,632 after excluding 1,516 canonical
molecules (official qed/logp04/logp06 test files + the 64-source dev panel).
Train QED 0.7003–0.8000 mean 0.7504; validation 0.7001–0.7999 mean 0.7506.
Train ∩ validation = ∅; both ∩ excluded = ∅.

**Stratification:** 10 strata of width 0.01, **equal allocation**, deterministic
sha256 rank within each. Equal rather than proportional allocation deliberately
— it guarantees coverage of the harder low-QED sources instead of following the
population skew.

## Why 8 rollouts and not 20

**Unique-source diversity matters more than precision on any single state's
reachability.** Eight rollouts per source across 1,024 sources buys breadth;
twenty across fewer sources would buy precision we do not yet need.

And if the hard event turns out to be rare, **more IID rollouts are the wrong
fix** — the right fix is targeted continuation labelling from informative
states, which directly estimates the quantity `h_φ` is meant to learn.

At ~7 s per transition this is ~96 serial core-hours, heavily parallelizable.
It buys a **reusable controller dataset**, not another benchmark run — every
trajectory yields multiple budget-indexed states, each relabelable against the
whole registered goal family, so the effective supervised set is far larger than
49k.

## The registered goal family — frozen before the census

Hindsight relabelling uses **exactly** this grid, so the head learns the *goal
language* rather than memorizing one benchmark threshold:

```
QED thresholds     τ_q ∈ {0.75, 0.80, 0.85, 0.90, 0.95}
similarity floors  τ_s ∈ {0.30, 0.40, 0.50, 0.60}

g_z(x) = 1[ QED(x) ≥ τ_q  ∧  Sim(x, x_src) ≥ τ_s ]        20 regions
```

The benchmark region `(0.90, 0.40)` is **one member of this grid, not the
target it was built around.** Fixing the grid in advance is what prevents the
goal language from being reshaped after seeing which events are populated.

Every prefix of every trajectory is a training state: remaining steps give `b`,
terminal membership gives the Monte-Carlo Bernoulli label, and the source gives
the similarity reference.

## The decision rule after 8,192 trajectories — frozen now

> **Do NOT automatically expand the IID trajectory census.**

Inspect the registered goal-event counts, then:

| observation | action |
|---|---|
| the hard QED/sim events carry **enough signal** | **train** |
| severely **tail-starved** | identify high-QED / high-similarity **prefixes** and obtain additional **conditional continuation labels from those exact states** under the same frozen `R_θ` |

Targeted continuation directly estimates `h_b(x)` where it is uncertain. It is
far more efficient than spraying another 50,000 source-rooted trajectories, and
**"run more of the same" is explicitly not the default.**

## What this corpus is not

**Never a benchmark result.** It is controller training data. The 64-source
panel remains the first genuine test of whether COMPOSE finally has a competent
GPS, and the 128 reserved sources are the fresh validation before anything
freezes.
