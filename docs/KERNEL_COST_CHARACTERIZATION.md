# What an executable chemistry-native process actually costs

**Systems characterization. Closes the master plan's standing open item
("bounded kernel profile — ~80 % of remaining cost is unmeasured"). No
benchmark outcome was inspected to produce any number here.**

## The measured decomposition of one fiber enumeration

Medians over 6 QED-task sources, ~600 marks each, one core:

| component | time | share |
|---|---:|---:|
| `runtime.apply` × ~600 legal rewrites | **4.33 s** | **61 %** |
| `prepare_factorized_mark_batch` (legal-mask construction) | ~2.6 s | 37 % |
| `canonical_state_key` × ~600 | 0.14 s | 2 % |
| **the neural network itself** | **0.06 s** | **1 %** |
| full kernel | 7.12 s | |

> **The cost of this process is not neural FLOPs, not scalar indexing, and not
> canonicalization. It is predominantly exact chemical-support construction and
> execution of the legal rewrite fiber.**

That is a genuinely useful characterization of what an executable
chemistry-native stochastic process costs, and it is the opposite of the
intuition for a learned generative model.

## Three optimization stories, each measured and each killed

Recorded because they were plausible, and because each was killed by a
measurement rather than an argument.

| hypothesis | verdict | evidence |
|---|---|---|
| the coordinate loop's per-mark tensor indexing dominates | ❌ **wrong** | 0.01 s, **0.1 %** |
| the neural forward dominates; batch it | ❌ **wrong** | the network is **0.06 s, 1 %** |
| batching states amortizes the forward | ❌ **wrong** | per-state flat: 3.378 s at batch 1, **2.930 s at batch 20** — 1.15× at best, and *worse* past batch 4 |

The batching result has a clean explanation: `prepare_factorized_mark_batch`
computes each state's legal mask independently, so preparation scales linearly
and a 60 ms forward has nothing to amortize. The training path had already
recorded the same shape from the other direction — batch 32 → 128 scaled
backward 14.3× for 4× the work.

## What remains available, and why it is not being taken now

**Direct mark sampling — real, exact, ~2.7×.** Each mark carries a *normalized*
`log_probability` and the canonical successor law is the **pushforward** of the
mark law with self-loops removed. So

```
sample mark ~ marked_law → apply ONE rewrite → canonicalize
reject self-loops        → accept with probability QED(y)
```

draws from **exactly** `π(y|x) ∝ R_θ(y|x)·QED(y)` — the same frozen policy, not
an approximation — while applying ~1 rewrite instead of ~600.

It removes `apply` and `canonicalize` (63 %) but **not** `prepare` (37 %), so
the ceiling is **2.7×, not the 50× first projected**. Worth building as a
**general COMPOSE inference improvement**, gated on exhaustive small-state
distributional equality against `successors()` — **not** inserted into the
critical path of a ~$9 benchmark that is ready to run.

**`prepare_factorized_mark_batch` is not to be touched.** It is 37 % of runtime
and it is the chemical-legality machinery. Optimizing safety-critical support
code to save a few dollars on one experiment is the wrong trade.

## What is being taken

**Exact memoization only.** Cache by the complete kernel input — canonical
molecule **plus every argument that changes the law** (the time point). All 20
replicates begin at the same source, so that fiber is computed once instead of
twenty times, and later collisions are free.

**Logical requests and actual cache misses are recorded separately**, so the
resource ledger reports work *requested* and work *done* without conflating
them.

Numerically identical by construction: a cache hit returns the same object the
kernel would have recomputed.
