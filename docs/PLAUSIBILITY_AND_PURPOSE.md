# Plausibility and purpose — the organizing frame, and what it fixes

**Canonical framing. Supersedes any description of the controller as an
add-on.**

## The architecture

```
legal molecular dynamics   +   R_θ  goal-INDEPENDENT prior   +   h_φ  goal-DEPENDENT
    what edits exist           what looks chemically             inference-time control
                               plausible
```

> **COMPOSE separates molecular plausibility from molecular purpose.** One
> goal-independent reference process learns which executable transformations are
> plausible; an inference-time finite-horizon controller redirects that same
> process toward new property regions, preferences or constraints **without
> retraining the molecular dynamics.**

That is a different claim from "a graph generator plus an optimizer."

**The controller is part of COMPOSE, not an accessory.** Evaluating bare `R_θ`
on an optimization benchmark is the **unnatural** evaluation — it deletes the
component that supplies purpose.

## What this fixes about policy B

Policy B's 0/320 was **not** "our generative model failed." It was **our first
trivial controller approximation failed.**

> ### ⛔ Policy B must NEVER be presented as "COMPOSE"
>
> Call it the **local one-step tilt** or the **myopic-control ablation**. The
> claim-bearing COMPOSE optimization method is the **finite-horizon learned
> controller**. Reporting a myopic ablation under the method's name would
> misrepresent the method to its own disadvantage.

## Why this is defensible against GrIDDD specifically

GrIDDD does **not** take an unconditional diffusion model and hope it wanders
toward QED 0.9. For property optimization it corrupts the source for 100 steps
and denoises **conditioned on the desired property vector**, with classifier-free
guidance (λ = 2 on ZINC), sampling 20 optimized candidates per source under that
conditioned procedure.

| | |
|---|---|
| **GrIDDD** | learns a **property-conditioned** denoising process; uses guidance during sampling |
| **COMPOSE** | learns a **reusable goal-independent** transition law; supplies the objective through a **separate finite-horizon controller** at inference |

Comparing GrIDDD's fully conditioned inference against bare `R_θ` would
**handicap COMPOSE by deleting the part of the method that supplies purpose.**
The difference in *where* conditioning lives is one of the interesting things
about the paper — not something to apologize for.

## Correction 1 — "20 candidates" is NOT "20 oracle calls"

**An earlier note of mine said "the protocol's 20 candidates" as though it
capped internal evaluations. That was wrong.**

The published protocol specifies **20 candidates per source**. It establishes
**no rule that only 20 QED evaluations may occur internally** — and GrIDDD
itself runs a 100-step conditioned denoising process per candidate.

So **intermediate property evaluations used to steer a trajectory are not
automatically a protocol violation.**

What they *are* is a **resource asymmetry**, so report separately and never
pooled:

| reported quantity | |
|---|---|
| **20 terminal candidates per source** | the matched benchmark allowance |
| intermediate property evaluations | ours, disclosed |
| kernel / executor calls | ours, disclosed |
| wall / compute | where useful |

> **We do not claim matched oracle efficiency against GrIDDD.** We claim task
> performance at the matched terminal-candidate allowance.

This is the defensible position, and it is much better than contorting COMPOSE
into imitating a diffusion model's inference structure.

## Correction 2 — SMC particle coupling is NOT a fairness problem

**A second note of mine argued that SMC's shared ancestry made its 20 outputs
unfair against GrIDDD's independent samples. Retracted.**

The benchmark metric is **best-of-20**: does *at least one* candidate satisfy
QED ≥ 0.9 and Tanimoto ≥ 0.4. That statistic **does not require independence**,
and the protocol does not impose it. GrIDDD's 20 happen to come from different
latents; that is a property of their method, not a rule for ours.

A population with shared ancestry is simply **how COMPOSE performs inference.**

What is required is **transparency**, not imitation:

> COMPOSE constructs its candidate set using adaptive controlled trajectories /
> population inference; GrIDDD generates conditioned diffusion samples.

**If SMC uses dramatically more objective evaluations, report that.** That is
the real asymmetry — the algorithms are *supposed* to differ, otherwise there
would be no point proposing COMPOSE.

## The inference hierarchy for the QED experiment

**Preferred — learned region `h_φ` + exact rejection sampling.** The cleanest
realization of the controlled kernel, and it naturally yields 20 separate
trajectories.

**If measured acceptance collapses — learned region `h_φ` + twisted SMC.** Still
COMPOSE, still legitimate, still the same frozen `R_θ`. Disclose that rare-event
inference required population control, and report the extra objective and
compute cost.

**Escalate on measurement, never on assumption.**

## The frame explains every experiment coherently

| experiment | in the frame |
|---|---|
| **GrIDDD QED** | can the same frozen process be given a competent GPS? |
| **retargeting** | change the destination halfway without changing the car |
| **Pareto** | ask the GPS to target different regions of objective space |
| **hard support** | close the roads that violate a constraint |
| **pathwise** | require every road actually traversed to satisfy a condition |

One learned prior, many destinations. That is the paper.
