# Horizon amendment — pre-controller-outcome

**Amends `HPHI_V1_CORPUS_PREREGISTRATION.md`. Made BEFORE any controller was
trained and BEFORE any controller outcome exists. The original H6
preregistration no longer applies and is not pretended to.**

## What the qualification returned

**The frozen selection rule did NOT fire.** It required *"the smallest horizon
at which the nontrivial reachability curve has essentially saturated."*

| tier | H3→H6 | H6→H12 | H12→H24 | decay ratio |
|---|---|---|---|---|
| ≥0.80 | +60 | +46 | +40 | 0.87 |
| ≥0.85 | +13 | +29 | **+29** | **1.00** |

At `≥0.85` the last doubling bought exactly as much as the previous one. **No
horizon in {3, 6, 12, 24} satisfies the criterion.**

> **Recorded verbatim: the saturation criterion returned "no saturation ≤ 24"
> and therefore did not choose a horizon.**

**We do NOT run H48 or H96.** Unguided base-process saturation is **not the
meaningful control criterion** — the controller's job is to concentrate
probability onto rare productive routes, not to wait for an unguided random walk
to exhaust them.

## The corpus amendment

```
    1,024 sources  ×  2 trajectories  ×  H24   =   49,152 committed transitions
    1,024 sources  ×  8 trajectories  ×  H6    =   49,152      (original)
```

**Exactly the same transition budget.** The change is depth, not spend.

| property | why it matters |
|---|---|
| identical transition/compute scale | the amendment is not a 4× compute grab |
| **1,024 sources preserved** | chemical diversity untouched |
| budgets now cover `b = 0 … 24` | the controller learns a real budget axis, not one truncated at 6 |
| far more distinct depth/state structure per trajectory | richer supervision from the same spend |
| no extra root-rollout computation | root Monte Carlo is the *wrong* place for precision |

**Local Monte-Carlo precision comes later from the already-preregistered
conditional continuations**, not from more roots.

At the measured ~1.2 % unguided `0.90/0.40` rate at H24, 2,048 root trajectories
yield on the order of a few dozen hard-region hits. **That is acceptable.** The
`0.80` and `0.85` tiers supply dense structured supervision, and if `0.90` needs
precision we **branch continuations from informative high-QED /
similarity-preserving states** rather than spraying roots.

## The native controller's budget

> **Maximum budget H24, with online first-hit STOP.**
> **H6 and H12 remain reported shorter-budget operating points.**

The controller's target is **finite-budget hitting reachability**:

```
h_b(x, z) = P( ∃ t ≤ b : X_t ∈ B_z | X_0 = x )
```

with the **exact boundary condition** `h_b(x,z) = 1` whenever `x ∈ B_z` — the
network is never asked to learn that from trajectories, and the policy executes
**STOP**.

Both `terminal` and `ever_hit / first_hit` labels remain stored; the native
controller trains on hitting, and fixed-horizon terminal stays the matched
ablation.

## What is explicitly NOT authorized

**Self-avoidance · novelty bonuses · beam search · QED temperatures or powers ·
reward shaping.** None is motivated by a measured failure. **`0.95` is an
auxiliary rare-tail fact, not a problem to solve.**

## ⭐ ONE CONTROLLER QUALIFICATION — then the full corpus

**Superseded an A/B/C/D ladder. That was caution turning into bureaucracy.**

We have already paid for the pilot trajectories and already established: H24
rollout mechanics, exact nesting, sensible region geometry, green feature tests,
the sampler parity gate, and the boundary condition. **We do not need another
ladder of mini-experiments.**

The **one** thing never yet shown: **that the new region-conditioned `h_φ` can
steer anything.** So one end-to-end qualification, on data we already own:

```
1  split the existing pilot BY SOURCE into throwaway train / held-out
2  train a small region-h_φ
3  verify on HELD-OUT sources that it learns nontrivial reachability --
   not a constant, and not a current-QED shortcut
4  plug that exact checkpoint into the Stage-A1 controlled sampler
5  test whether guidance improves the well-powered 0.80 / 0.85 regions
   over unguided R_θ
```

### The GO criterion

> **GO to the full 1,024 × 2 × H24 corpus if the learned value is demonstrably
> informative on held-out sources AND controlled sampling moves `0.80`/`0.85`
> in the correct direction.**

**Not** impressive benchmark numbers. **Not** `0.90` success — unguided
`0.90/0.40` is ~1.2 % at H24, so this scale is badly underpowered for it, and
demanding it would be measuring noise.

**If it passes: launch the full corpus immediately.** No second 128-source
pilot, no further architecture smoke, no 256→512→1024 creep.

**If it fails: that is valuable.** Spending more on labels before understanding
why the value learner cannot navigate the easy regions would be waste.

```
existing H24 pilot  →  ONE end-to-end h_φ qualification  →  full corpus
        ↓
real h_φ  →  64-source dev  →  128-source validation  →  800 official
```

**That is enough staging.**

**Note:** the ~900 molecules being encoded are *already-generated pilot states*.
No new molecules, no new sources, no scale-up.

## The remaining ladder — no further design

```
1  amend corpus to 1,024 × 2 × H24                         ← this document
2  tiny end-to-end h_φ engineering smoke on the EXCLUDED
   H24 pilot data; verify encoding, b ∈ [0,24], the exact
   in-region boundary, labels, batching, checkpointing and
   Stage-A1 sampling — then THROW THE WEIGHTS AWAY
3  generate the real corpus
4  train ONE universal region-h_φ: frozen encoder, budget
   conditioning, goal conditioning, MC reachability target
   + Bellman consistency. R_θ stays frozen.
5  if the 0.90 tail is under-labelled → conditional
   continuations from preregistered informative states
6  evaluate on the 64-source dev panel:
       unguided R_θ   vs   Policy B   vs   R_θ·h_φ
   report success, first-hit step, similarity, QED, value
   calibration, acceptance rate, and success at H6/H12/H24
7  ONE measurement decides: healthy acceptance → stop.
   Calibrated but collapsed acceptance → frozen twisted-SMC
   escalation. Same h_φ, same R_θ; SMC is inference.
8  fresh 128-source validation, ONCE → freeze
9  official 800, 20 native trajectories, max H24, STOP at
   first 0.90/0.40 → then size-fixed Experiment C, same
   controller and seeds
```

**Then move on. There is no QED rescue branch.**

## What counts as success — not merely beating 45.1 %

```
unguided R_θ       rare success
Policy B           fails
region h_φ         LARGE AMPLIFICATION
```

while `R_θ` never changes, successful controlled trajectories often **stop well
before H24**, H6/H12 performance shows the GPS **compresses search depth**, and
the same architecture then transfers to new regions and objectives.

That demonstrates the claim the whole paper rests on:

> ### **The molecular prior supplies plausible dynamics; future-aware control supplies purpose.**

---

## What the steering test showed, and what it does NOT forecast

### The steering test PASSED

| arm | hit rate `0.80/0.40` | median first hit | end similarity |
|---|---|---|---|
| unguided `R_θ` | 12/64 = **0.188** | 2.5 | 0.476 |
| **`R_θ·h_φ`** | **25/64 = 0.391** | **1.0** | 0.785 |

**~2.1× amplification, and faster.** Crucially it is **not** an endpoint
coincidence — the de-biased decision diagnostic over 110 audited decisions:

```
h_chosen                  0.5892
E_R[h] (fixed batch)      0.3063        ->  ~1.9x
predicted E_R[h²]/E_R[h]  0.5632        <-  the identity HOLDS
h_chosen > E_R[h] on      80.0% of decisions
```

The identity can only hold with a strict gap when `h_φ` has genuine variance
across successors. **The mechanism we need is functioning.**

### The scale of what is actually being asked

GrIDDD reports **45.1 % source-level success with 20 candidates.** Under a
homogeneous approximation, `1 − (1−p)²⁰ = 0.451` needs only

```
p ≈ 2.95 % per trajectory
```

Against unguided `1.2 %` at H24, that is **~2.46× amplification** — and the
steering test already achieved **2.1×** on an easier region with a deliberately
primitive controller.

> **We are not asking the controller for a 40× miracle.**

### ⚠️ But that arithmetic is NOT a forecast — heterogeneity dominates

Same mean per-trajectory probability, different spread across sources:

| distribution over sources | mean `p` | source-level success |
|---|---|---|
| all sources equal, 2.95 % | 0.0295 | **0.451** |
| half zero, half 5.9 % | 0.0295 | 0.352 |
| 90 % zero, 10 % at 29.5 % | 0.0295 | **0.100** |

**Identical mean, 4.5× spread in the benchmark number.** If many official
sources have essentially zero reachable `0.90/0.40` probability, averaging 3 %
over trajectories does **not** produce 45 % source success. **The distribution
ACROSS sources decides the benchmark**, which is exactly why the 64 / 128 / 800
source-level runs are the real test and this arithmetic is orientation only.

### Do NOT extrapolate 2.1× to `0.90`

The smoke controller was deliberately primitive: trained on the tiny excluded
pilot, almost no `0.90` supervision, tested at H6, already hitting its proposal
cap (32 cap hits), no targeted continuation labels for the tail, and no SMC
despite the rare-event regime. **The architecture was shown to work before being
given the machinery built for the hard region.** That is precisely when to
scale.

### What would make this pessimistic

1. the full `h_φ` calibrates well but moves `0.90/0.40` only from ~1.2 % to ~1.4 %
2. many official-like sources turn out to have **zero** reachable probability
3. `h_φ` cannot distinguish successors in hard-tail states **even with** targeted
   continuation labels

**None of these has been observed.** What has: rare-but-real `0.90`
reachability, clean difficulty ordering, strong budget dependence, real
state-conditional value variation, ~2× controlled amplification on a held-out
easier goal, earlier hitting, and a frozen rare-event escalation.

> **The position has moved from "maybe the controller idea does not work" to
> "the controller works — the question is how much tail amplification it gets
> when properly trained."**

### Correction: the identity check is de-biased, NOT independent

I called the predicted-vs-observed agreement "independent confirmation." **That
overstates it** — both quantities are evaluations of the *same* `h_φ` under the
*same* sampling mechanism. It is a clean, **de-biased mechanistic validation**,
which is what it was designed to be, and it should be described that way.

**What it does establish, exactly.** For positive `h`:

```
E_R[h²]/E_R[h] − E_R[h]  =  Var_R(h) / E_R[h]
```

so a strict gap means `h_φ` has **within-state variance across legal
successors**. Measured:

```
0.5632 − 0.3063 = 0.2569   →   Var_R(h) = 0.0787,  sd = 0.281
```

**Mean 0.306 with sd 0.281 on [0,1] — substantial within-state spread.** A model
that had learned only *"0.90 is globally harder than 0.80"* would assign
near-identical value to every road from a state, give `sd ≈ 0`, and **could not
steer at all.** Successor discrimination is the one thing control requires, and
it is present.

### The heterogeneity table is a WARNING, not a measurement

Those three rows are **hypothetical distributions with the same mean**. They
establish *sensitivity* to heterogeneity. **They do not show that COMPOSE has
the "90 % of sources unreachable" shape** — that has not been measured.

What *has* been established, weaker but favourable: across the development
population, `0.90/0.40` routes exist under an **unguided, goal-blind** `R_θ`.
How broadly they are distributed across sources is what the full controller
must tell us.

### Therefore: track INTENSITY and COVERAGE separately

**Do not optimize or celebrate pooled trajectory hit rate alone.** The
benchmark gives each source **one binary success** regardless of how many times
an easy source succeeds.

| axis | question |
|---|---|
| **intensity** | on sources where success is possible, how much does control raise the probability? |
| **coverage** | across how many *different* sources does control create a meaningful chance? |

> The dream is **not** *"ten sources become incredibly easy."* It is **"a broad
> fraction of sources move from low-but-nonzero reachability into useful success
> probability."** Then best-of-20 becomes powerful.

**The 64-source dev experiment must therefore report the source-level
distribution**, not only pooled trajectories:

```
fraction of sources with   0/20   ·   1+/20   ·   5+/20   successes
```

That distinguishes **broad amplification** from **hammering a few easy
molecules** — and a coverage failure is not something twenty candidates can
rescue, because a source with essentially zero reachable probability stays at
zero however often it is sampled.

**This is why the dev → validation → official sequence is source-based rather
than pooled.**

### Net

**Nothing here downgraded the project.** It killed one overly simple forecast —
*"3 % mean trajectory success = 45 % benchmark success"* — which is good,
because we now know the actual thing the controller must accomplish: **broad
amplification across sources, not merely higher pooled hit probability.** The
steering result is a legitimate reason to think that is achievable.
