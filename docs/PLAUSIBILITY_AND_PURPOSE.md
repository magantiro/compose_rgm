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

---

# The claim, at exactly the strength we can defend

## What to say

> **A single frozen, goal-independent molecular reference process can be reused
> across objectives. Changing the design objective changes only the
> inference-time control problem, not the learned molecular dynamics.**

Or memorably:

> **Train the molecular dynamics once; change the design problem at inference.**

`R_θ(y|x)` never needs to know QED, logP, DRD2, a Pareto weight, a constraint,
or which downstream problem we will eventually care about. It learns only *given
this molecule, which executable transformations are plausible?* The controller
supplies *given what I care about now, which plausible futures should I pursue?*

## What NOT to say

> ~~"COMPOSE automatically optimizes literally any unseen objective with zero
> additional learning."~~

Stronger than we have established. The honest tiering:

| situation | what changes |
|---|---|
| `h_φ` already understands the goal descriptor | **zero retraining** |
| new objective outside the learned goal language | learn a new **controller/value**, still **never** `R_θ` |
| exact / Monte-Carlo / SMC control | can use a newly supplied oracle **directly** |

**Every tier keeps `R_θ` frozen.** That is the invariant, and it is enough.

### And do NOT overclaim about GrIDDD

> ~~"GrIDDD must retrain a whole model for every new objective."~~

**Not demonstrated from their protocol — do not assert it.** The accurate
contrast is already major:

> **COMPOSE's `R_θ` is trained without knowing the downstream objective at all.
> GrIDDD's optimization capability relies on property-conditioned training and
> guidance.**

## Four axes — never collapse them into one number

| axis | question |
|---|---|
| **one-time learning cost** | how much task-specific training was required? |
| **objective dependence** | was the molecular model trained knowing QED/JNK3/…? |
| **adaptation cost** | what changes when handed a brand-new oracle tomorrow? |
| **inference cost** | how much computation does control require? |

**COMPOSE's potential win is on the middle two.** Oracle-call accounting
addresses only the fourth.

Oracle cost asks *how expensive is inference?* The decomposition asks *what has
to be retrained when the scientific objective changes?* Those are different
questions, and the second is a methodological advantage even if our first
scalable controller is not yet maximally oracle-efficient.

## Oracle efficiency — an engineering axis, not the fairness constraint

> **Oracle efficiency is a performance/engineering axis. It is not the central
> fairness constraint defining COMPOSE.**

**For QED it is close to bookkeeping.** QED and Tanimoto are cheap — no wet lab,
no 20-hour FEP. If the controller inspects many intermediates to steer six
edits, that is acceptable provided the 20 returned candidates obey the
benchmark, the official test set was never used for development, the inference
procedure is disclosed, and compute is not absurdly out of scale.

**Report evaluations for transparency. Do not organize the scientific story
around matching them.**

**Where it genuinely will matter:** docking, expensive learned ensembles, QM,
synthesis evaluation, real experiments — and on the Pareto benchmark, where
100× Graph-GA's evaluations for slightly better hypervolume is a legitimate
reviewer objection.

## Priority order

```
correct controller  →  competitive performance  →  efficient implementation
```

with `R_θ` frozen throughout. **Do not handicap the GPS because it has to look
at the map.** Batch it, amortize it, learn `h_φ`, escalate to SMC if rare-event
inference truly needs it. Optimize inference efficiency *after* demonstrating
competent control.

The scientifically embarrassing outcome is **not** "COMPOSE used 150 property
evaluations instead of 20." It is:

> *"We invented a framework whose entire novelty is reusable inference-time
> control, then deliberately used a pathetic one-step tilt so its inference
> budget would look small."*

That is optimizing the wrong thing.

## The table that makes the architecture impossible to miss

Not "COMPOSE QED = X vs GrIDDD 45.1 %." This:

| setting | dynamics retrained? | controller changed? | objective supplied when? | performance |
|---|---|---|---|---|
| QED | **No** | yes | inference | — |
| new single objective | **No** | yes | inference | — |
| conjunction | **No** | yes | inference | — |
| 5-objective MOO | **No** | yes | inference | — |
| preference switch midway | **No** | yes | **after the trajectory begins** | — |

> ### **Same frozen `R_θ` in every row.**

**This is why the five-objective benchmark earns its place.** If the same frozen
process that did QED handles a rich many-objective problem purely by supplying
new objective information to the controller, that demonstrates the architecture
far more forcefully than another two points of QED.

## Every experiment is one claim tested at increasing strength

| experiment | the version of the claim it tests |
|---|---|
| **GrIDDD QED** | can a frozen generic process be competitively steered toward an objective never used to train `R_θ`? |
| **retargeting** | can the objective change *after the trajectory has begun*, without restart or retraining? |
| **Pareto** | can the same frozen process be recontrolled across many preferences? |
| **five-objective** | can it handle a much richer objective vector without a new molecular generator? |
| **hard support** | can a new feasibility requirement change the reachable process at inference? |
| **pathwise** | can a constraint bind the entire realized trajectory, not just the final sample? |

**Not six capabilities — six consequences of separating purpose from dynamics.**

The performance tables answer *whether it works*. The frozen-`R_θ`,
interchangeable-control architecture answers **why COMPOSE is a different
framework at all.**
