# Persistent search archive vs independent restarts — design, fixed before the data

**Hypothesis.** Preserving valid intermediates across candidates and branching
from them beats restarting SMC from x0 every time.

This is a principled dev-panel variant, which the protocol's path explicitly
licenses:

    64 x 5 dev -> iterate principled variants -> choose ONE
               -> 128 x 20 FRESH disjoint validation panel -> freeze
               -> official 800 x 20, ONCE

It does NOT license using a variant on the validation panel or the official 800
before one is chosen and frozen.

## Why 4 candidates and not 20

Reuse cannot help candidate 1 -- there is nothing to reuse yet. And the ladder
already shows independent attempts saturating fast: marginal conversion ran
29.7% -> 15.6% -> 2.6% across attempts 1, 2 and 3. Candidates 2 through 4 are
therefore exactly the window where an archive should show its advantage, and
where independent restarts are already known to be running out.

Four is also cheap enough to be a development experiment rather than a
production run: 12 sources x 2 policies x 4 candidates = 96 slots, roughly two
core-hours at the current 8x controller, about $0.20 and minutes of wall clock.
At the pre-lazy speed the same experiment would have been ~16 core-hours.

## Panel — 12 sources, stratified, chosen from banked results

    reliable  3, 15, 29, 60     solved at attempt 1
    marginal  5, 22, 46, 49     converted only at attempt 2 or 3
    hard      0, 25, 42, 63     never solved in 3 attempts

Stratified deliberately. A gain concentrated on the reliable stratum is nearly
worthless -- those sources already succeed. The experiment lives or dies on the
MARGINAL and HARD strata, and the readout must be reported by stratum, never
pooled into a single rate that the reliable four would flatter.

## The two policies

    BASELINE   4 independent SMC runs from x0, seeds as today
    ARCHIVE    1 persistent per-source archive of valid intermediates,
               producing 4 returned candidates

Same R_theta, same h_phi, same legal kernel, same N=32, same region, same four
returned outputs, same benchmark event.

## THE BUDGET CONSTRAINT, which is correctness and not fairness

`h_phi(x, b)` is BUDGET-CONDITIONED: it estimates reachability within b
REMAINING steps, and that is how it was trained. So a branch taken from an
archived intermediate at depth d must run with remaining budget H - d, never a
fresh H = 24.

Two things break otherwise. h_phi gets queried outside its training semantics,
so the twist stops meaning what it was fitted to mean. And the archive gets a
deeper total edit path from x0 than the four restarts do, so an apparent win
could be nothing but "more edits allowed" -- which a compute log would not
reveal, because it is a budget difference rather than a work difference.

Depth-respecting branching keeps the frozen horizon meaningful and keeps the
comparison honest on both axes.

## What is measured

  * source success after candidate 1, 2, 3, 4 -- BY STRATUM
  * sources rescued by the archive that the baseline never solved, and the
    reverse, since a policy can lose sources as well as gain them
  * law expansions and unique states visited per policy, so a win is attributable
    to REUSE rather than to more work
  * reuse rate: fraction of archive branches starting from a previously
    discovered intermediate
  * candidate diversity: distinct returned molecules out of 4. An archive that
    returns four near-copies of one branch is worth less than its success count
    suggests, because the benchmark event is "at least one of N succeeds" and
    correlated candidates do not buy independent chances
  * wall clock per policy

## Decision rule, fixed now

Kill or redesign if the archive fails to beat four independent restarts on the
MARGINAL plus HARD strata, whatever it does on the reliable four.

If it clearly wins there, go to 8 candidates on the same panel, or to a broader
32-64 source check, before anything else. Five candidates is not worth running
as an intermediate step unless the four-candidate result is ambiguous.

## What this experiment cannot settle

Rung 3 showed 37 of 37 hard-source failures with ZERO particles ever entering
the region under the current H=24, N=32 controller. If those sources are
unreachable within the frozen horizon, no search policy reaches them, and the
archive will lose on the hard stratum for reasons that say nothing about
archives. A null result on HARD is therefore weak evidence; a null on MARGINAL
is strong evidence, because those sources are demonstrably reachable.
