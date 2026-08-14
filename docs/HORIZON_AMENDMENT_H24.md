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

## ⭐ STAGED CONTROLLER GATES — A, B, C before D

**The full corpus is justified only after the controller stack earns it.** We
know the rollout generator works. We do **not** know whether region-`h_φ` can
learn a useful GPS from those labels, so generating 49,152 transitions first is
backwards.

| stage | data | question | gate |
|---|---|---|---|
| **A · engineering** | existing excluded H24 pilot | does the feature/label/train/sampler pipeline run correctly? | **hard correctness only** |
| **B · learning** | the SAME 256 H24 trajectories | can `h_φ` actually learn reachability on `0.80`/`0.85`? | **must beat a trivial/constant value predictor**, and show sensible budget and goal ordering |
| **C · tiny controlled** | held-out subset of those pilot sources | does `R_θ·h_φ` steer better than unguided `R_θ` on `0.80`/`0.85`? | **directional improvement. NO `0.90` requirement** |
| **D · full corpus** | 1,024 × 2 × H24 | tail coverage and generalization for `0.90` | **launch only after A–C pass** |

### Do NOT demand `0.90` in the tiny smokes

Unguided `0.90/0.40` is ~**1.2 %** at H24, so a 64-source panel is **badly
underpowered** for it. Judge the learner on `0.80` and `0.85`, where the event
counts are real.

A convincing tiny result looks like:

```
h_φ(x, b, 0.80)  >  h_φ(x, b, 0.85)  >  h_φ(x, b, 0.90)     in sensible states
predictions rise with remaining budget where they should
calibration is nontrivial (beats a constant)
and R_θ·h_φ finds 0.80/0.85 MORE OFTEN or FASTER than unguided R_θ
```

> **If it cannot do that on the easy regions, do NOT generate 49,000
> transitions hoping more data fixes the implementation. Diagnose the learner.**

If it can, the full corpus is then bought for a specific reason — **tail
coverage and generalization**, not hope.

```
prove the GPS can learn on the roads we already mapped
        ↓
then map 1,024 sources to make the GPS good enough for the hard destination
```

**Note on the encode budget:** the ~900 molecules in stage A are *encodings of
already-generated pilot states*. No new molecules, no new sources, no scale-up.

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
