# P0c verdict — the preference continuum does NOT collapse

**Applied once, to all 12 sources, against the three-way rule frozen in
`ORACLE_BUDGET_SEMANTICS_AND_P0.md` §"The gate is THREE-way, not two-way".**
The rule was re-read before any shard was opened.

## Verdict: OUTCOME 2 — the exact preference sweep is sufficient

> **manageable tree, materially more breadth → the exact preference sweep is
> sufficient → no aspiration controller**

**The aspiration branch is CLOSED. It is not earned and will not be built.**

The single earned front-construction branch is the **exact preference sweep over
the continuum** — a traversal of machinery that already exists, not a new
algorithm.

## The result reverses the early signal

The endpoint-pool calculation had suggested collapse — 1,025 candidates → **2**
preference regions, median 5 across twelve pools — and the preregistration said
so plainly: *"The endpoint-pool result makes the second outcome the likelier
one."* It was right about which outcome, for the wrong reason: the pools
predicted **collapse**, and the real object shows the **opposite**.

This is exactly why the probe was run on real one-step successor fibers instead
of being decided from the pool arithmetic. **One-step sparsity does not imply
trajectory-level collapse over H=6**, and here it actively misleads.

## Gate 1 — is the tree manageable?

| | |
|---|---|
| COMPLETE within the 240-expansion guard | **11 / 12** |
| expansions | median **136**, range 14–241 |
| guard hit | source **004** only, at 241 |

Median 136 against a guard of 240: manageable, with room.

## Gate 2 — is there materially more breadth?

| per source | continuum | at the frozen 5 weights | gain |
|---|---|---|---|
| distinct endpoints | median **69.5** (9–106) | median **3.5** (2–5) | **+66.0 median, positive 12/12** |
| distinct trajectories | median 71.0 | median 3.5 | **+68.5 median, positive 12/12** |

Not marginal. The continuum induces roughly **twenty times** the distinct
endpoints the five frozen weights reach, on every source without exception.

## The breadth is useful, not just numerous

Breadth alone would prove little — more endpoints could be dominated clutter.
They are not:

| | median | positive |
|---|---|---|
| nondominated set size gain | **+8.0** | **12 / 12** |
| hypervolume gain | **+64.2 %** | **12 / 12** |

Per source: +145.7, +60.1, +23.3, +65.4, +5.3, +93.2, +87.4, +99.0, +4.3,
+34.9, +63.1, +119.6 %.

## What this settles about P3/P4

P0c existed to resolve one asymmetry: repaired P3/P4 won hypervolume **12/0**
but **lost nondominated cardinality** (P3 −2.250, P4 −2.417, 0W/11L). The two
candidate explanations were:

- **(a)** an artifact of sampling only five weights — sweeping the continuum
  fixes it, **no new algorithm needed**; or
- **(b)** the controller collapses the continuum onto a few decisions — weight
  density is not the limitation and **set-level allocation is**.

**The answer is (a).** At five weights the controller reaches a median of **3**
nondominated endpoints; over the continuum it reaches **11.5**. A median gain of
**+8** comfortably swamps the ~2.3–2.4 cardinality deficit that prompted the
question.

The ND loss in P3/P4 was a **sampling artifact of the five-weight grid**, not
evidence of a structural weakness in closed-loop control.

## Source 004 — the one guard hit, handled under outcome 3

Rule for a guard hit: *do not decide yet; classify using the branches already
generated.* Source 004 truncated at 241 expansions with 118 leaves, and shows
**ND +16** and **HV +5.3 %** on the partial tree. Both are **lower bounds** — the
untraversed remainder can only add.

It does not alter the cohort verdict, and it is the honest place to note that
004 carries the second-smallest HV gain in the set. Nothing here is rescued: the
truncation is reported, not smoothed.

## Claim discipline — what may and may not be said

**Permitted:** *the complete set of distinct trajectories and endpoints induced
by the frozen COMPOSE preference controller over the continuous preference
range.*

**Barred:** "the true Pareto front." The sweep is exhaustive over the
controller's preference response, not over chemistry.

**Also barred, in the other direction:** this result must not be read as
vindicating Chebyshev scalarization in general, any more than a collapse would
have indicted it. The statement is scoped to **this** frozen state geometry,
horizon, reference point and greedy closed-loop policy.

**Status of the numbers:** development and design evidence only. The
hypervolumes here use a per-source reference derived from the observed endpoint
sets, for the sole purpose of answering "is the extra breadth useful?" They are
**not** the frozen panel estimand and must never be quoted as a panel result.

## Consequences

1. **Aspiration control is not built.** Not deferred, not queued — **closed**,
   because the gate it depended on returned the opposite of what it needed.
2. The earned branch is the **exact preference sweep**, using the existing
   traversal.
3. The Pareto chain advances to: freeze the final controller **and** the
   comparator matrix together, then power and launch the fresh panel.
4. Fresh-panel `n` is still computed **after** that freeze, from the sanctioned
   Pareto sources, against the actual source-level paired bootstrap.

## Provenance

12/12 shards, `editing_v2/r_theta_run/pareto_preference_partition/`. Exact
partition by lower envelope of 2N segments with breakpoints `w* = g₂/(g₁+g₂)`,
bisected against the frozen `_argmin_stable`. Blinding held: the rule was read
from the preregistration before any shard was opened, and applied once.
