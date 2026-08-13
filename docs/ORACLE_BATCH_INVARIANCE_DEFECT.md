# The DRD2 oracle is BATCH-SIZE DEPENDENT

**Status: DEFECT RECORDED. NOT FIXED. Fixing it re-baselines frozen results.**

Found while asking why batched scoring was not bit-identical. The answer turned
out to be one line, and the consequence is a reproducibility hazard rather than
a speed question.

## The one line

`src/compose_v4/drd2_oracle.py:213`

```python
return kernel @ self.dual_coef + self.intercept
```

Everything upstream is **bit-identical across batch sizes**, measured on 512
held-out molecules:

| stage | single-row vs batched |
|---|---|
| `cross = x @ support_vectors.T` | **512/512 exact**, max&nbsp;\|d\| 0.0 |
| `sq` (the ‖x−s‖² expansion) | **512/512 exact**, max&nbsp;\|d\| 0.0 |
| `kernel = exp(−γ·sq)` | **512/512 exact**, max&nbsp;\|d\| 0.0 |
| **`kernel @ dual_coef`** | **13/512 exact**, max&nbsp;\|d\| **1.82e−14** |

The heavy GEMM is *not* the culprit — that was the obvious hypothesis and it is
wrong. `(1, N) @ (N,)` dispatches to a dot product; `(B, N) @ (N,)` dispatches to
GEMV with different blocking. Same arithmetic, different accumulation order.

## The hazard, which is not about speed

**The same molecule scores differently depending on how many molecules you
happened to ask about at once.** Claim-bearing code mixes both call shapes:

| call site | batch |
|---|---|
| `pareto_gen_rank_topup_app.py:229` | `margin_many([key])` — **1** |
| `retarget_intervention_app.py:214` | `margin_many([key])` — **1** |
| `retarget_goal_calibration_app.py:203` | `margin_many([key])` — **1** |
| `experiment_c0_planning_signal_app.py:462` | `margin_many([key])` — **1** |
| `experiment_c0_planning_signal_app.py:527` | `margin_many(keys)` — **batched** |
| `retarget_goal_language_normalizers.py:54,126` | `margin_many(smiles)` — **batched** |
| `retarget_goal_movability_census.py:84` | `margin_many(members)` — **batched** |

1.82e−14 is chemically meaningless. It is **exactly the magnitude that flips an
`argmax` tie**, and controllers take `argmax` over ~586-wide successor fibers of
one molecule — structurally similar candidates, where exact ties are more likely
than in a diverse library (a 2,000-molecule held-out sample already contains 8
exactly-tied adjacent pairs).

**Where it demonstrably does not matter:** the frozen normalizers. Those are a
median and an IQR over 96,094 molecules; a 1e−14 perturbation per value moves a
scale constant of magnitude 2.6161 by ~1e−14. Mixing a batched normalizer with
B=1 controller scores is a relative error of ~1e−14 and is not a concern.

### The one mixed site, AUDITED AND CLOSED

`experiment_c0_planning_signal_app.py` uses both shapes, and worse than it first
looked: `margin_cache` is populated from **both** paths — `margin()` writes B=1
values (line 462) and the batched loop `setdefault`s batched ones (line 529). So
a molecule's cached value depends on **which path saw it first**, and the cache
holds a mixture.

The bounded audit asked the only question that matters: *did any claim-bearing
decision change?*

**No, on two independent grounds.**

1. **Every decision-bearing comparison reads one batched array.**
   `greedy_index = argmax(immediate)`, `immediate[chosen] < immediate[greedy_index]`,
   and the sacrifice statistic all index `immediate`, computed in a single
   batched call and therefore internally consistent whatever shape produced it.
   The mixed cache feeds only `rollout_prefix`'s running max (lines 485, 494,
   507).
2. **The experiment is superseded and its result was negative.**
   `docs/EXPERIMENT_PLAN.md:672` — *"Superseded en route, recorded so they are
   not re-run: the C0 planning-signal probe on DRD2 was **negative**."* And C0's
   `sacrifice_to_win` statistic was separately withdrawn as circular. **No live
   claim depends on any C0 number.**

**Closed. No requalification needed.**

And the standing rule that follows: **no sealed result is reopened because a
floating-point implementation can differ at 1e-14.** A result is reopened only
if an actual action or outcome is shown to have changed. Exact-target,
retargeting, Stage B and the Pareto smoke all score at B=1 throughout and are
untouched by this.

## The fix exists, is one line, and costs 0.4%

```python
return (kernel * self.dual_coef).sum(axis=1) + self.intercept
```

An elementwise multiply then a row-wise reduction. NumPy's pairwise summation
along the last axis has a fixed per-row order regardless of how many rows are
present, so the result is **batch-invariant**:

| | single vs batched | timing |
|---|---|---|
| `kernel @ dual_coef` | 13/512 exact | 0.29 µs/mol |
| `(kernel * dual_coef).sum(1)` | **512/512 exact** | 1.51 µs/mol |

1.2 µs/molecule against a ~1258 µs B=1 scoring cost is **0.4% overhead**, and it
would restore the 4.2× batched-scoring speedup that was rejected on fidelity
grounds.

## Why it is NOT being adopted now

**The new reduction does not reproduce the old one at B=1** — 19/512 exact,
max \|d\| 1.78e−14. So it is not a bug fix that leaves history intact; it is a
**re-baseline**. Every score computed under the current reduction — the sealed
exact-target panel, held-out retargeting, Stage B, the Pareto smoke — would
become non-reproducible bit-for-bit.

Trading that for a ~7% wall-time gain (scoring is ~20% of runtime; 4.2× of 20%)
is a bad trade, and this project's whole discipline is that trajectory identity
is worth more than throughput.

## What to do instead

1. **Pin B=1 at every claim-bearing controller call site.** This is already true
   everywhere except the two sites listed above, and it makes the oracle
   *consistent* even while it remains batch-*dependent*.
2. **Adopt the shape-independent reduction at the next natural re-baseline** —
   a full re-run of a frozen panel, not a speed change — and record the boundary
   so scores before and after are known to differ by ~2e−14.
3. **Never introduce a new batched call site** into a path that also scores at
   B=1, until (2) happens.

## What this says about the earlier rejection

The batching fix was rejected on the evidence that it perturbed scores by
2.6e−14 and that a perturbation of that size can flip a tie. That conclusion
stands. What was wrong was the *explanation*: I attributed it to the heavy GEMM,
and the GEMM is bit-identical. The cause is the final reduction, which is both
smaller and fixable — and finding that is what exposed the batch-dependence
hazard, which matters more than the speedup did.
