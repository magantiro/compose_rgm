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

## The stop rule — NUMERICALLY EXPLICIT, frozen now

"Retention → freeze / collapse → stop" was too vague to be a gate. Made exact
here, **before the K=41 run**, by **reusing the already-frozen standard** rather
than minting a threshold:

`RETENTION_FLOOR = 0.90` and the source-level paired bootstrap 2.5th percentile
are taken **verbatim** from `scripts/pareto_scalable_power.py`, where they were
frozen for the K333 study.

Two ratios per source, against the full-information ceiling:

```
r^HV_i = HV_i(budgeted)      / HV_i(ceiling)
r^ND_i = |ND_i(budgeted)|    / |ND_i(ceiling)|
```

| test | criterion |
|---|---|
| **R1 — quality** | bootstrap 95 % CI lower bound on `mean(r^HV_i)` **> 0.90** |
| **R2 — breadth** | bootstrap 95 % CI lower bound on `mean(r^ND_i)` **> 0.90** |
| **B — budget** | `unique_oracle_evals ≤ 10,000` on **every** source, not on average |

**Success requires R1 ∧ R2 ∧ B** — intersection–union, the same structure as the
cLogP `P1 ∧ P2` design, no multiplicity correction.

**R2 is not optional and does not get relaxed.** The entire P0c discovery is that
five weights collapsed breadth. A budgeted controller that preserves hypervolume
while losing nondominated breadth would **recreate exactly the problem P0c
diagnosed**, and must be recorded as a failure even if its HV looks excellent.

**This is a demanding gate and that is deliberate.** The ceiling reaches a median
of 11.5 nondominated endpoints, so R2 requires roughly 10 of them to survive a
~15× cut in oracle spend. **We accept a clean failure here.** Stating that before
the run is what stops a later "0.85 is basically 0.90" argument.

On collapse the paper still separates honestly, and this is stated *before*
seeing the result so it cannot read as a consolation:

> the **full-information preference sweep** demonstrates the latent richness of
> the controller, while the **five-weight / budgeted** controller demonstrates
> what is achievable at contemporary sample budgets.

That is a publishable pair, not a failure.

## Implementation qualification — passes BEFORE any K=41 outcome is opened

This is potentially the **final** Pareto algorithm, so the code is qualified
first. All three must pass before a single quality number is computed.

| # | test | what it protects |
|---|---|---|
| **Q1** | **full-fiber equivalence** — with `K ≥ |F(x)|`, the budgeted code reproduces the full continuum sweep **exactly**: same partition, same leaves, same endpoints, same order | proves budgeting is the *only* difference; any divergence is a bug, not a finding |
| **Q2** | **ranking correctness** — the shortlist is taken **after** canonical successor/alias construction, scored by frozen `R_θ`, with the same deterministic tie semantics | prevents shortlisting a pre-canonical or aliased fiber, which would silently change the candidate set |
| **Q3** | **worst-case budget fixture** — synthetic 240-expansion tree, **zero** cache overlap, 41 unseen successors at every state → the ledger must read **exactly 9,840** | proves the ≤10k guarantee holds at the worst case the guard permits |

### The headroom audit — Q3's real point

`10,000 − 9,840 = 160`. The formula is a true ≤10k guarantee **only if no other
true-oracle calls exist anywhere else in the controller.** Q3 therefore also
asserts that every algorithmic oracle call outside the shortlist path fits inside
that **160-query headroom**, and the ledger is checked against the frozen oracle
semantics — where **cached values are free** and the denominator is **unique
canonical molecules actually evaluated**, never raw requests.

If any hidden call path is found, it is reported — not absorbed into the
headroom silently.

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
