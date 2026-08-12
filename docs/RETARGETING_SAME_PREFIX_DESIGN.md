# Same-Prefix Goal Intervention — design, and what the census already decided

Status: **DESIGN + HELD-IN CALIBRATION INPUTS. Nothing launched, no panel carved,
no threshold selected.** Steps 1–2 of the execution order are specified here;
step 3 (held-in calibration) needs Modal compute and has not been run.

## The claim

> After a molecular trajectory has already been realized under goal **A**,
> COMPOSE accepts an unanticipated goal **B**, preserves the existing molecular
> history, and recomputes only the control law to continue from the exact
> current molecule under a finite remaining edit budget.

The novelty is not "generation under different conditions" — preference-conditioned
generators already span preferences, and iterative optimizers already continue
from a molecule. It is the conjunction: **unanticipated intervention +
realized-history continuity + finite-horizon replanning, on one frozen `R_theta`.**

## Three subclaims, on the goal structures where each is meaningful

Retargeting is not one claim, and **one goal pair must not carry all three**.
Forcing them together is what pushes a design toward picking the objective that
makes greedy fail, which would make the whole result read as engineered.

| | subclaim | goal structure | what it needs |
|---|---|---|---|
| **A** | **Intervention responsiveness** — COMPOSE responds coherently to an unanticipated goal change | natural switches `P → P∧D`, `D → P∧D` | retarget beats continue-A on B. **Verified control does NOT need to beat greedy here.** Greedy being decent is fine — the claim is about responding to the switch at all |
| **B** | **Future-aware adaptation** — remaining-budget reasoning beats myopic reasoning after the switch | bounded developability region: `L ≤ cLogP ≤ U` **and** `QED ≥ q` | a goal whose feasible set is bounded, so a locally attractive edit can consume future room. Gated on the held-in contrastive measurement below |
| **C** | **Prefix reuse** — the realized history is worth keeping | compatible `P → P∧D` vs conflicting `P → D`, reported separately | continue-from-`x_tau` vs restart-from-`x_0`. The prediction is **asymmetric**: compatible should often help, conflicting may favour restart. Continuation is not required to always win |

**Why a bounded region for B, stated as a principle rather than a convenience.**
Bounded feasibility is *inherently* non-monotone: moving toward the box is good,
overshooting is bad, and a locally attractive edit can consume the room needed to
land inside it later. That is a structural property of constrained design, not an
artifact chosen because greedy does badly on it. A monotone threshold ("more
potency is always better") has no such structure, which is exactly why C0 found
no planning signal on it.

**The region is `cLogP` box + `QED` floor, not `cLogP` alone.** A single scalar
box reads as a toy. A bounded developability region is a real medicinal-chemistry
object and target-free, with no privileged similarity heuristic.

**Anti-tuning rule, binding.** The held-in calibration asks *once* whether the
bounded region contains contrastive future-sensitive decisions. If it does,
freeze it. **If it does not, do not adjust the interval until it does** —
record that future-aware-vs-greedy is not a strong retargeting subclaim for that
goal and proceed with subclaims A and C. Subclaim B is allowed to fail.

## What the feasibility censuses established

Three local, kernel-free censuses were run before any controller was written.
All are counts; none selects a threshold.

- `diagnostics/retarget_goal_feasibility_census.json`
- `diagnostics/retarget_goal_movability_census.json`
- `diagnostics/retarget_goal_language_normalizers.json`

**1. Potency and developability are near-independent, not opposed.** The worry
that DRD2 ligands are lipophilic bases and so structurally excluded from a
developability box does not hold on this corpus: Spearman potency-vs-QED −0.24,
potency-vs-in-box −0.26, and the lift of P∧D over independence is 0.93–1.24
across every candidate threshold. P∧D is rare (0.36–1.28% of the reserve) but
**not structurally empty** — 60–135 molecules of 10,653 already satisfy both. The
conjunction is a real region, not an artifact.

**2. Panel construction is not a constraint.** The eligibility rule (a source
must satisfy *neither* phase goal at step zero) retains 57–76% of the held-out
reserve, i.e. 6,000–8,100 molecules. Any panel size under discussion is
comfortably available.

**3. Held-in and held-out are the same distribution.** IQR ratios 0.99–1.04,
medians within 0.02 on all three properties. A normalizer frozen on held-in does
not silently rescale the held-out goal.

## Goal language — frozen normalizers

Normalizers are the **held-in** IQR. IQR rather than standard deviation because
DRD2 log-odds has a long upper tail; a handful of actives would otherwise set
the scale for the 99% of molecules that are inactive.

| property | centre (held-in median) | `s_j` (held-in IQR) |
|---|---|---|
| DRD2 log-odds | −5.5217 | 2.6161 |
| QED | +0.5668 | 0.3139 |
| cLogP | +3.4211 | 2.0834 |

Margins, for a lower bound `c_j` and the obvious two-sided form inside a box:

```
m_j(x) = (f_j(x) - c_j) / s_j
```

Success and dense ranking are defined **separately**, because the asymmetry
below would otherwise let one hard objective drown out every other criterion for
the whole trajectory.

**Binary success** — the reported outcome, unclipped:

```
success_g(x) = 1{ m_j(x) >= 0  for all j }
```

**Dense ranking** — for guidance only, over *clipped* margins:

```
m~_j(x) = clip(m_j(x), -c, +c)
u_g(x)  = -tau_g * log sum_j exp(-m~_j(x) / tau_g)
```

Clipping is what keeps the soft-min a conjunction rather than a proxy for its
hardest term. Without it, a DRD2 margin sitting 2+ units below zero pins the
soft minimum for the entire trajectory and every developability improvement is
invisible to the controller. Clipping changes **no** success outcome — only
whether the dense signal is usable. `c` is frozen on held-in data with the other
parameters.

**DRD2 is scored in log-odds, never in P(active).** The pool sits at median
P(active) = 0.004, where probability is saturated and a probability delta
understates real movement.

### The asymmetry this creates — read before setting thresholds

In normalized units the two goals are not comparable in difficulty:

| goal | climb from the median molecule | in `s_j` units |
|---|---|---|
| potency, P(active) ≥ 0.5 | +5.54 log-odds | **2.12** |
| QED 0.565 → 0.70 | +0.135 | **0.43** |
| cLogP into [1, 4] | ~0.5–1.0 typical | **0.2–0.5** |

Potency is roughly **4–5× harder than developability** in the units the soft-min
sees. Consequence: `u_{P&D}` will be pinned to the potency margin almost
everywhere, and **the difficulty of the conjunction is essentially the difficulty
of potency.** This is the intended soft-min behaviour, not a bug, but it means
the P∧D switch classes inherit potency's risk profile wholesale and the
developability term will rarely be the binding constraint.

## The reachability question, and why it stays open

Whether a potency threshold is reachable in 4 edits was the main risk. The census
narrowed it but did **not** close it, and the honest state is worth recording
because two intuitive answers are both wrong.

Single-edit movement, measured over 972 real one-cut matched pairs on the
held-out reserve (favourable direction):

| property | median | p90 | p99 | max |
|---|---|---|---|---|
| DRD2 log-odds | +0.380 | +1.711 | +5.794 | +14.00 |
| QED | +0.048 | +0.168 | +0.325 | +0.448 |
| cLogP | +0.427 | +1.069 | +1.874 | +2.392 |

Against a required climb of +4.69 / +5.54 / +6.39 log-odds for P ≥ 0.3 / 0.5 /
0.7, only 8.5% / 6.7% / 5.6% of real single edits clear the per-edit bar
(climb ÷ 4), and 0.5–0.6% clear the whole climb in one jump.

That reads like "out of reach" — **but it is not**, for two measured reasons:

1. **Best-of-N.** Matched-pair mining sees a median of **1** neighbour per
   molecule; the real successor fiber is ~500 wide. Drawing N gains from the
   measured favourable distribution and taking the max, even N = 5 covers the
   climb over 4 steps (E[best step] +2.10, ×4 = +8.39 vs +5.54 required).
2. **The favourable tail is position-independent.** A best-of-N estimate assumes
   the gain distribution does not depend on where you start — exactly what an
   activity cliff would violate. Measured by starting band, it holds: from the
   deepest band (log-odds < −6) P(gain > 0) is 56.3% with p99 +4.04, and the
   largest single edit observed from a typical inactive start is **+14.0**. The
   big favourable moves are *not* confined to molecules already near an active.

So potency is plausibly reachable. **This must still be confirmed by the held-in
calibration**, because both arguments are optimistic in the same direction:
fiber successors are small, highly correlated perturbations rather than
independent MMP-scale draws, and the ×4 assumes gains add.

## The real risk is H2, not reachability — and C0 already measured it

The C0 planning-signal probe
(`diagnostics/editing_v2_experiment_c0_planning_signal_preregistered.json`) ran
lookahead against greedy on **DRD2 specifically** and returned
*STOP / INCONCLUSIVE — sacrificial actions pay off at about chance*: 8/14
sacrificial actions won against a null of 0.5, mean fresh regret −0.021.

That was not a reachability failure — greedy solved 7 of 12 sources. It was a
**planning-signal** failure: on DRD2, lookahead did not beat greedy.

Therefore: **do not stake H2 (future-aware adaptation) on the potency term.**
There is direct prior evidence it will not separate, and given the soft-min
asymmetry above, a P∧D goal is mostly a potency goal.

### What this does and does not license

It licenses **separating the subclaims**: potency switches remain the natural
setting for intervention responsiveness and prefix reuse (subclaims A and C),
where greedy being decent is not a problem, and the bounded developability
region is the setting where future-aware adaptation (subclaim B) is even
testable.

It does **not** license "replace potency with cLogP because cLogP makes greedy
fail." Two guards against that reading, both binding:

- Subclaim B is **gated on a measurement, not on an expectation.** The held-in
  contrastive analysis decides it, and subclaim B is allowed to fail.
- The interval is set **once** for feasible-region base rate, and is not
  retuned if the contrastive measurement comes back empty.

The structural argument for why a bounded region is the right setting stands on
its own: the feasible set is bounded, so overshoot is possible and a locally
attractive edit can consume the room needed to land inside later. For reference,
the median reserve molecule sits at cLogP 3.40 against a candidate ceiling of
4.0 with a median favourable single edit of +0.43 — the geometry that makes
overshoot reachable is present. Whether the controller's actual decisions are
future-sensitive is a separate, measured question.

## What the held-in calibration must decide (step 3)

Only these, and then freeze. Every one is chosen to **avoid floor/ceiling and
goal domination — never to maximise COMPOSE's advantage.**

1. Potency threshold `c_P` from the grid 0.3 / 0.5 / 0.7. The census cannot pick
   it: it measures the local landscape, not reachability under the real fiber.
2. QED floor `q` and the cLogP interval `[L, U]`, chosen for a reasonable
   feasible-region base rate.
3. Horizon: `H = 8, tau = 4` versus `3 + 3` — enough post-switch room without
   making the task trivial.
4. Soft-min temperature `tau_g` and clip `c`.
5. **The gate on subclaim B — the dynamic-retargeting analogue of C0.** On
   held-in post-switch states under the bounded developability goal, measure:

   > How often does greedy's best immediate action differ from the best
   > remaining-budget action, and when it differs, how often does that
   > difference actually pay off?

   Reported as C0 reported it: top-1 disagreement, sacrifice-to-win rate against
   a null of 0.5, and future regret of greedy. **This is the measurement that
   decides whether subclaim B is in the paper.**

   Unlike C0, the lookahead here is the **deterministic greedy continuation**
   `V_G`, not a Monte Carlo estimate — it is what the sealed-67 controller
   actually commits, it carries the policy-improvement guarantee, and it removes
   the MC noise that forced C0's independent selection/evaluation samples.

Calibrate on held-in sources only. Freeze the goal language **and** the switch
templates before any held-out panel is touched.

## Protocol (unchanged from the agreed design; recorded so it can be preregistered)

**Switch classes.** `P → P∧D` (late constraint addition), `D → P∧D` (reverse late
requirement), `P → D` (conflicting reprioritization). The first two test prefix
reuse; the third tests adaptation when history is actively inconvenient.

**Prefix policy.** One common policy for all post-switch arms: verified rollout
policy improvement under A for the first `tau` edits. Goal B must not be used in
producing the prefix. Physically separate stages — generate and save all
A-prefixes, hash and commit the prefix artifact, and only then inject B. That is
not ceremony: it is what proves the prefix was not selected with knowledge of the
future goal.

**Arms.** continue-A · greedy-retarget-B · local Boltzmann retarget
(`pi(y|x) ∝ R_theta(y|x) exp(beta u_B(y))`) · verified remaining-budget
retargeting · restart from `x_0` under B with the same post-switch budget ·
static compromise `u_A + u_B` from step zero · clairvoyant schedule (upper
reference; the gap to it is the **price of surprise**).

**No external baseline is qualified yet.** MARS-switch, GraphXForm-restart and
preference-conditioned generators wait until the internal causal table shows
intervention responsiveness, future-aware advantage where expected, prefix reuse
and a bounded price of surprise. Building adapters before the internal effect is
real spends the budget on plumbing for an effect that may not exist.

**Hypotheses.** H1 intervention responsiveness (retarget > continue-A) · H2
future-aware adaptation (verified > greedy) · H3 stateful prefix reuse
(retarget-from-`x_tau` > restart-from-`x_0`, **compatible switches only**) · H4
bounded price of surprise.

**Panel.** New sources from the matched reserve, disjoint from the exact-target
development 24, the sealed 65/67, and h_phi's training endpoints. 20–24
development sources; 60–80 held-out confirmatory. Every source evaluated under
all three switch classes, with statistics **clustered by source** — the branches
are not independent.

**Eligibility, pre-control only:** valid and representable · within the frozen
size range · does not already satisfy both phase goals at step zero. No source is
removed because a controller failed or a goal looked unreachable in exploratory
rollouts.

**Do not train another `h_phi` before this works.** First establish that
target-free future-aware retargeting buys something. If it does, the exact-target
lesson fixes the role: **proposal or compute allocation followed by verified
control**, never unverified direct action selection.

## Stop rules, with the census-informed trigger

- continue-A already does well on B → goals too correlated. The census says this
  is unlikely for P↔D (near-independent), so treat it as a live risk only for
  `D → P∧D`, where B contains A.
- every arm fails B → goal too hard or budget too short. **This is the live
  risk for the potency term**; adjust only in held-in calibration.
- greedy and verified retarget identical → dynamic switching works, future-aware
  planning adds nothing *for that goal*. Given C0, this is the expected outcome
  on potency and is the reason the box goal exists. Do not force a controller
  claim from it.
- restart always wins → the prefixes are not reusable under these switches.
  Weakens history reuse, not retargetability. Report it.
- only qualitative examples work → retargeting is not a headline claim.
