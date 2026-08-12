# Claim 2 — trajectory characterization — STATUS

**Status:** `SMOKE_HELD_IN` — the authorized 8-source instrument smoke ran and
**all six permitted gates pass**. No Claim-2 verdict was emitted, by design.

**Branch:** `codex/compose-claim2-trajectory`
**Base commit:** `04f1c46` · **Run commit:** `c2c95f2`
**Modal app:** `ap-TifBx7AHzBMd6x4cGH5BZN`, `ephemeral (detached)`, profile
`rahul-94866`, CPU only.

**Held-out data opened:** **NO.** The matched reserve is untouched.

**What is running:** nothing. The smoke finished 8/8.

## Smoke gate results — instrument only, NOT a Claim-2 result

| # | Question the run was allowed to answer | Verdict |
|---|---|---|
| 1 | Do all three laws diverge on real states? | **PASS** — min pairwise TV over 221 states: min 0.242, median 0.425, max 0.499. Zero degenerate states, zero with \|N+(x)\| ≤ 1. |
| 2 | Does this lane's learned-law construction agree with the canonical scorer? | **PASS** — 0 disagreements across all 8 sources. |
| 3 | Do trajectories complete under budget without instrument errors? | **PASS** — 8/8 sources, 0 failed tasks, 0 budget exhaustion, 0 enumeration failures, 0 truncated trajectories. 23–31 calls/source against a budget of 40. |
| 4 | Are the metric distributions non-degenerate? | **PASS** — both axes separate; 3 of 4 paired axis comparisons resolve even at n=8. |
| 5 | Are reversal / revisit / dead-end measurements behaving? | **PASS** — they discriminate strongly between arms (see below). Dead-end rate 0.000 for every arm. |
| 6 | Measured runtime and cost? | **16.43 s median per enumeration** (min 5.87, max 20.36). 221 enumerations, ~1.19 container-hours, wall 12.4 min. |

**Cost estimate was accurate.** Protocol predicted 14 s/call × 1.2–1.5 =
16.8–21 s; measured 16.43. Predicted 1.5 container-hours; actual ~1.19.

## Observation the main lane needs before authorizing 36 sources

**This is not a result. n = 8, below the 20-source floor, and the analysis
script emitted `UNDERPOWERED_NO_VERDICT` as designed.** It is reported because
it bears on whether the next run is worth its cost, and because a negative
characterization must not be discovered late.

On these 8 held-in sources `R_theta` **moved less and cycled more** than both
unlearned arms:

| arm | mobility | fidelity | revisit | reversal | heavy-atom Δ | families |
|---|---:|---:|---:|---:|---:|---:|
| `r_theta` | 0.623 | **1.000** | **0.281** | **0.300** | +0.06 | 8/8 |
| `uniform_canonical` | 0.750 | 0.781 | 0.000 | 0.000 | +2.31 | 6/8 |
| `empirical_family` | 0.766 | 0.979 | 0.010 | 0.013 | +0.06 | 8/8 |

Three things follow, all provisional:

1. **The mobility axis is doing exactly the job it was added for.** `R_theta`
   has perfect envelope retention *and* the lowest structural displacement —
   the "a process that barely moves wins fidelity trivially" failure mode.
   Without mobility as a co-equal axis this would have read as a clean win.
2. **If this pattern holds at n = 36, the frontier verdict is more likely
   `incomparable` than `dominates`** — mobility traded for fidelity. That is a
   declared, publishable outcome, not a failure, and it must not be scalarized
   away afterwards.
3. **The cycling numbers approach a stop rule.** 0.281 revisit / 0.300 reversal
   against ~0.00–0.01 for the unlearned arms is the direction the "pathological
   cycling" stop rule watches. It is not yet triggered — no verdict at n=8 —
   but it is the thing the 36-source run must adjudicate.

**Not** operator collapse: `R_theta` committed all 8 active families; uniform
committed only 6. Also note uniform's mobility comes partly with +2.31 heavy
atoms versus +0.06 for `R_theta`, which is the trivial-growth confound the
suite reports heavy-atom change beside Tanimoto distance to expose.

## Next action

**None without authorization.** Per the main lane: stop after the smoke, hand
back gates and measured cost, **no automatic promotion** to the 36-source run.

The likely next ask is the 36-source held-in development run, still no reserve:
47 worst-case enumerations/source, **~10 container-hours expected** at the now
*measured* 16.43 s/call. Route remains
`8-source smoke → 36-source held-in → main review → matched-reserve`.

**Nothing downstream is authorized**, including the reserve panel (~27
container-hours), which stays unopened and unmaterialized.
