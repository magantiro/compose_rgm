# Amendment — MOLLEO Dev Gate 1: lazy verification (L1 vs L4)

Recorded before the data. Development only. Not an official MOLLEO result.

## Why this replaces the shortlist gate

The preregistered MOLLEO unblocker (`AMENDMENT_SHORTLIST_RETENTION.md`) assumed
a single `R_theta` top-K operating point could be chosen prospectively. That
assumption was FALSIFIED: per-transition recall looked ideal (median chosen rank
4 of a ~530 fiber; 90.4% within top-32 on hard sources) while **0 of 21
hard-source rescue trajectories were fully contained in top-128**, worst rank
median 271. At the compression the budget needs (~10x, K=64) roughly two-thirds
of 30-step routes break.

So that gate is closed, and the query saving moves to a different axis:

    OLD (dead):  save queries by DELETING possible moves.
    NEW:         save queries by KEEPING the moves and POSTPONING measurement.

`R_theta` remains a soft stochastic proposal prior over the FULL legal support.
No hard top-K anywhere.

## The question

> Can COMPOSE search four molecular edits for the price of roughly one
> ground-truth measurement cycle without materially losing Pareto quality?

## Frozen setup

- **Oracles:** the SHA-pinned self-contained bundle
  `artifacts/oracles/molleo_task3_v1` -- qed(max), jnk3(max), sa(min, (10-sa)/9),
  gsk3b(min, 1-gsk3b), drd2(min, 1-drd2). **No PyTDC at runtime**; the forests
  are extracted to .npz with recorded pickle sha256, and the runtime imports
  neither tdc nor sklearn. TDC's own QED agrees with the bundle to every digit.
- **Cohort:** development seed 100, 120 molecules, sha256 `06f70c84b97d577b...`,
  disjoint from every official set. All 120 are labelled ONCE (120 true calls)
  to seed the archive and the surrogate.
- **Roots:** 24, selected as the lowest `sha256(smiles)` -- objective-blind and
  fixed before any label was read. sha256 `5149c7c49633dc38...`.
- **Controller:** unchanged COMPOSE. N = 32, H = 12, exact legal fiber, soft
  `R_theta` proposal, frozen kernel (verified: R_theta reproduces a banked QED
  candidate bit-identically on this environment).

## THE ORACLE ACCOUNTING RULE, which is the experiment

One oracle call = one **distinct canonical molecule** measured by the frozen
oracle. Repeats are cached and free.

- Speculative depth: **surrogate only**. `navigation_lockout()` makes any true
  oracle access during navigation a hard error rather than a silent overspend.
- Verification depth: true oracle on the **unique realized particle states**
  only -- never on the successor fiber.

Per H12 block, before deduplication:

    L1 verifies at depths 1..12   ->  <= 12 x 32 = 384 true calls
    L4 verifies at depths 4, 8, 12 ->  <=  3 x 32 =  96 true calls

**The reported Pareto set and hypervolume contain ONLY oracle-verified
molecules.** A surrogate-predicted molecule may steer the search and can never
enter the reported front until it has actually been measured.

## Arms differ in ONE thing

Same roots, preferences, seeds, N, H, surrogate architecture and update rule,
and COMPOSE dynamics. Only the verification cadence differs: L1 every depth,
L4 every fourth.

## Stage 0 sentinel, before spending the cap

Two matched H12 blocks per arm. Report: actual unique true calls, true HV,
surrogate MAE by speculative distance 1..4, whether L4's checkpoint corrections
reorder the particle population, wall clock, and speculative states examined.

**Kill immediately if L4 is catastrophic here.** Otherwise proceed to the
matched-budget comparison at 2,000 true calls per arm, where L4 spends its ~4x
cheaper verification on more trajectory blocks.

Two readouts, and they answer different questions:

    matched search effort  -> does laziness preserve quality?
    matched oracle budget  -> does laziness buy more HV?

## What this does not license

Any official MOLLEO claim, any use of the official initialization sets (still
sealed behind `official_init_set(official=True)`), or selecting the
verification cadence after seeing which reads better.
