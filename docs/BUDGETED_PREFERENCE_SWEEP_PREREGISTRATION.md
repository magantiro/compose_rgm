# Budgeted preference sweep — preregistration

**Committed BEFORE implementation and BEFORE any quality outcome is computed.**
This is the **one and only** remaining Pareto scalability mechanism. If it
fails, the compression branch closes. There is no second rescue.

## Why this exists — the amendment P0c forced

P0c returned **outcome 2**: the frozen preference controller is rich and five
weights severely undersample it. Aspiration/gap-filling is **permanently
closed**.

But the verdict overreached in one sentence, now corrected:

> ~~The single earned front-construction branch is the exact preference
> sweep.~~

**The exact continuum sweep is a full-information CEILING, not a deployable
controller.** It consumes a median **75,910** unique true-oracle evaluations per
source against a frozen budget of **≤10,000** — 7.6× over on median, 32.8× on
the worst source, over budget on **11 of 12**. It cannot be placed beside
10k-budget methods and called fair.

## The question

> Can the preference-rich COMPOSE controller retain most of the full sweep's
> front quality while respecting **≤10,000 unique true-oracle evaluations per
> source**?

## The mechanism — `BUDGETED_PREFERENCE_SWEEP`

The **only** permitted pre-oracle prioritizer is the **frozen reference law**
`R_θ(y|x)`. This mirrors the COMPOSE decomposition exactly:

```
R_θ  →  plausibility: WHERE to spend evaluations
preference control  →  purpose: WHICH tradeoff to pursue
```

At every newly expanded state:

1. enumerate the **complete** canonical fiber (unchanged);
2. score **all** successors with frozen `R_θ`;
3. call the true objective oracle **only** on the top `K_sweep` *previously
   unevaluated* successors;
4. reuse all cached objective values **for free**;
5. compute the **exact** preference partition of that scored subset;
6. recurse under the unchanged traversal;
7. enforce a **global hard stop at 10,000 unique oracle evaluations**.

Everything else is preserved bit-for-bit: preference-partition semantics, the
deterministic `_argmin_stable` tie-break, the recursive traversal, frozen
`R_θ`, frozen objectives and scaling, and the **240-expansion guard**.

### Barred, explicitly

- objective-aware pruning or prefiltering
- aspiration control / gap-filling (closed by P0c)
- diversity bonuses or archive optimizers
- **a sweep over `K`** — one value, derived below, no alternatives run
- learned surrogates of any kind
- relaxing the 240-expansion guard
- raising the 10,000 budget

## `K_sweep` — derived from frozen constants only

```
K_sweep = floor(BUDGET / GUARD) = floor(10000 / 240) = 41
```

**Both inputs were frozen before P0c was run.** `BUDGET = 10,000` is the
standing per-source fairness rule; `GUARD = 240` is the preregistered expansion
guard. **`K_sweep` therefore does not depend on P0c data at all** — the
strongest provenance available, and it cannot have been tuned to an outcome that
did not yet exist.

**Why the guard and not the observed median.** With `K = 41`, the worst possible
traversal — a full 240 expansions with *zero* cache reuse — spends
`240 × 41 = 9,840 ≤ 10,000`. **The hard cap never binds.** Using the observed
median instead (136 expansions → `K = 73`) would fit a median source but would
hit the cap mid-traversal on a 240-expansion source, making the result depend on
**DFS order** — precisely the artifact this design must avoid. The hard cap
stays as a backstop, not as the operative constraint.

For context only, not as an input: `K = 41` retains ~6.8 % of the median fiber
(median width 606). This sits between the previously frozen `K_G = 333` (56 %)
and `K_V = 15` (2.4 %).

## The single comparison, run ONCE

On the **existing 12 P0c development sources only** — no fresh sources, no new
panel, no claim:

```
full exact sweep  (ceiling / reference)   vs   10k budgeted sweep
```

Reported outcomes:

- hypervolume
- nondominated set size
- objective-space spread / coverage
- distinct preference-induced endpoints
- **unique true-oracle evaluations** — must be **≤10,000 on every source,
  not on average**

## The stop rule, frozen now

| outcome | consequence |
|---|---|
| budgeted sweep **retains strong front quality** | **freeze it as the final Pareto controller**; proceed to the fresh panel |
| budgeted sweep **collapses** | **close the compression branch.** No second algorithm. |

On collapse the paper still separates honestly, and this is stated *before*
seeing the result so it cannot read as a consolation:

> the **full-information preference sweep** demonstrates the latent richness of
> the controller, while the **five-weight / budgeted** controller demonstrates
> what is achievable at contemporary sample budgets.

That is a publishable pair, not a failure.

## What this does NOT touch

**P3/P4 remain banked and unchanged.** P0c does not retroactively alter them,
and neither does this. They ran at five weights; their nondominated deficit
(−2.250, −2.417) stands **as reported**. The remedy counts only if a fresh panel
actually runs it.

**The two scientific questions stay separate.** Do not merge them into one
experiment:

| question | contrast |
|---|---|
| does state feedback matter? | closed-loop COMPOSE vs fair generate-and-rank *(banked)* |
| is the controller broad? | exact continuum vs five-weight COMPOSE |
| can breadth be exposed efficiently? | full sweep vs 10k budgeted sweep *(this document)* |

## Guard discipline on the fresh panel

The **240-expansion guard stays frozen**. If a fresh source hits it, that source
receives a **budget-truncated sweep** and is reported as such — **not** a
post-hoc larger guard.
