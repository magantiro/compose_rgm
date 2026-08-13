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

## Calibration outcome — 30 held-in sources, cohort `e402e318`

`diagnostics/retarget_calibration_result.json`. Ran once.

**1. Potency: all three thresholds usable, and reachability is settled.** 0/30
sources start satisfied; after the 4-edit prefix 17/13/10 of 30 reach P ≥
0.3/0.5/0.7. Median climb **+4.78 log-odds in four greedy edits** (p90 +8.59,
max +11.72). This closes the question the movability census left open — the
matched-pair pessimism was wrong and the position-independent best-of-N reading
was right. **P ≥ 0.5 at 13/30 is the operating point**: neither floor nor
ceiling.

**2. Developability at 4 post-switch edits: CEILING.** greedy 29/30, verified
29/30, binary headroom **0**. The region is simply easy to reach with four edits
and a ~500-wide fiber.

**3. Horizon:** contrastive decisions concentrate early — 22/30 states show
top-1 disagreement at remaining 4, then 16, 17, and 0 at remaining 1 (where
lookahead is vacuous by construction).

**4. THE GATE: CLOSED.** Not for absence of disagreement — lookahead would act
differently at 55/120 decision states (46%), on 27/30 sources. But the outcome
differed on **0** of those 27. A ceiling, not a null: a real planning advantage
would have had nowhere to show.

Per the anti-tuning rule the region is **not** retuned. This is the outcome the
stop rule anticipated: *"greedy and verified retarget identical → dynamic
switching works, future-aware planning adds nothing for that goal. Do not force
a controller claim from it."*

### Instrument defect found in this analysis — do not reuse the statistic

The intended headline, sacrifice-to-win, is **circular and was withdrawn**. The
lookahead action is chosen as `argmax(V_G)`, so `V_G(chosen) >= V_G(greedy)`
holds by construction and "the sacrifice won" is true whenever the two disagree
and do not tie. The data confirm it: of 55 disagreements, 50 higher, 5 tied,
**0 lower** — a genuine measurement would show losses.

C0 avoided this by scoring on an independent evaluation sample. That safeguard
was dropped here on the reasoning that a deterministic `V_G` has no winner's
curse. That reasoning is wrong: the bias is selecting and scoring with the same
function, and determinism does not remove it. Only two quantities from this run
are admissible — **top-1 disagreement** (non-circular: it says the controller
would act differently) and the **endpoint comparison**.

Any future planning-signal probe must either score on an independent
continuation or compare endpoints. Deterministic lookahead does not license
dropping the split.

### The 4+4 and 3+3 verdicts above are WITHDRAWN — there was no verified arm

Both runs compared greedy against greedy. `verified_land` was computed as
`rollout(switch_key, POST_STEPS, develop_score)` — a pure greedy continuation —
while the decision loop committed `current = keys[greedy_index]`. The loop
computed the lookahead's preferred action at every step and then discarded it.

The tell was in the 3+3 output: identical final utility on **all 30** sources
while the lookahead disagreed at 38/90 decision states across 27/30 sources.
Disagreement that never once changes an outcome is not a ceiling; it is an arm
that was never run. Headroom 0 was definitional.

## Calibration outcome — 3+3, verified arm actually implemented

`diagnostics/retarget_calibration_result_3plus3_fixed.json`. Same cohort
`e402e318`, same thresholds, region, utility, clip, temperature and candidate
law. The verified arm now commits `argmax V_G` under the sealed-67 strict-
improvement rule and re-plans from the committed state.

**Binary endpoint — still a ceiling.** greedy 28/30, verified 28/30, headroom
**0**. The lookahead overrode greedy on most sources and changed the outcome on
**none**.

**Continuous utility — verified is better, but the direction is guaranteed.**
Mean +0.0361, median +0.0177, higher on 23/30, lower on **0**. The zero is the
**policy-improvement theorem**, not a measurement: greedy's action is always in
the candidate set, `futures[greedy_index]` *is* greedy's own landing value, and
strict improvement never commits a lower `V_G`, so by induction the verified
landing cannot be worse. Only the **magnitude** is admissible — and it is
**+3.2% of the typical post-switch movement**, never enough to flip a success.

**GATE: CLOSED.** Per the pre-committed rule: no budget 2, no tighter box, no
harder threshold. Subclaim B is dropped for this goal.

> On this target-free developability goal, retargeting is easy enough that
> myopic control suffices. Future-aware control is never worse and is slightly
> better in utility, but the advantage is small and never decides an outcome.

This is not damaging. The sealed exact-target result (40% → 62%) already
establishes that future-aware control matters where the problem is hard. Not
every goal has to reproduce it.

### Three instrument defects in one experiment — the shared shape

All three reported a quantity whose sign was fixed in advance:

1. **sacrifice-to-win** — action chosen as `argmax V_G`, then scored by `V_G`.
2. **no verified arm** — both arms were greedy, so headroom was 0 by definition.
3. **sign test on paired utility** — policy improvement guarantees `verified ≥
   greedy`, so the null of 0.5 was false before any data existed.

The lesson is procedural: **before reporting a statistic, ask what value it
could take if the hypothesis were false.** If the answer is "none", it is not a
measurement. C0's independent-evaluation-sample design was the right pattern and
should not have been dropped.

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

---

# Frozen interpretation of the development result

The development panel is complete: 30 sources x 2 histories, H=6, switch at 3,
`B = P AND D`, max-min goal language, controller parity audited. The
interpretation below is frozen. Further descriptor exploration on these 30
sources has rapidly diminishing scientific value and is not authorised.

## Three levels, in order

**1. Primary — intervention.** Changing the objective at the realized switch
state redirects the trajectory. Greedy-matched contrast, so the comparison
varies only the objective:

| | paired difference | wins |
|---|---|---|
| P-first | +0.716 [+0.512, +0.922] | 23 / 1 |
| D-first | +1.406 [+1.181, +1.620] | 30 / 0 |

**2. Stateful — the value of history.** Retaining the realized molecular state
has positive average value against restarting with the same remaining budget,
and this survives a change of controller class:

| | greedy-matched (sensitivity) | verified-matched (primary) |
|---|---|---|
| P-first | +0.513 [+0.423, +0.610] 30W/0L | +0.444 [+0.348, +0.550] 30W/0L |
| D-first | +0.246 [+0.044, +0.442] 20W/10L | +0.290 [+0.104, +0.484] 22W/8L |

Only 4 of 60 source-history observations disagree in sign between controller
classes, so this measures a property of the history rather than of the
controller that exploits it.

**3. Heterogeneity — and its explicit limit.** The value of history depends on
what the history accomplished: uniformly positive for potency-first, which
pursued the eventual bottleneck, and heterogeneous for developability-first,
which solved the already-easy requirement. Seven D-first sources are hurt under
BOTH controller classes.

> **The present development panel does not establish what molecular feature
> predicts when history becomes harmful.**

That sentence is load-bearing and must survive into the paper.

## The explanation that was attempted and failed

Potency margin at the switch state was examined as a candidate mechanism. It is
**recorded as exploratory and unsuccessful**, and is NOT frozen as a held-out
hypothesis.

- Split by outcome, it looked suggestive: median −2.163 for the seven harmful
  sources against −1.702 for the other 23, with prefix displacement, heavy-atom
  count and cLogP essentially identical.
- Measured correctly, as a continuous association across all 30 D-first
  sources: **Spearman rho = +0.119, p = 0.53**, OLS slope +0.156.

The median split looked convincing only because it dichotomised on the outcome
being explained, which inflates a near-zero continuous relationship. Once the
seven are *defined* by having a negative history effect, asking what
distinguishes them is outcome-conditioned: it can suggest a hypothesis, it
cannot establish one. The all-source correlation is the appropriate check and
it erased the signal.

**No further descriptor search on these sources.** No fingerprint scans, no
RDKit descriptor panels, no operator-frequency sweeps, no classifiers. If a
mechanism falls out naturally from the reference-dynamics or preference-control
work, it can be revisited there with a fresh panel. Until then this is
unexplained heterogeneity, and saying so is stronger than carrying a weak
mechanism into confirmation.

## What is barred from the write-up

- Calling the seven an *outcome-defined subset* is correct; calling them a
  *subgroup* is not, because a subgroup must be recognisable from a
  pre-existing feature.
- Pooled "x/60" counts describe source-history OBSERVATIONS. The independent
  unit is the source, n=30 within each history.
- `verified_retarget` vs `greedy_retarget` is not reportable as a claim:
  policy improvement guarantees its direction.

---

# The claim wording, after external qualification

Two phrasings are now **barred**, both killed by verified source reading of
REINVENT 4 (Lane 3, `COMPARATOR_MATRIX.md`):

- ❌ "no existing method changes objective mid-run" — REINVENT 4's staged
  learning does exactly that, natively.
- ❌ "the baseline must retrain / without retraining" — REINVENT's stage
  boundary is far softer than that implies. The Agent is created once outside
  the stage loop; the Adam optimizer is constructed once and the *same object*
  enters every work package, so `exp_avg`, `exp_avg_sq` and the step counter
  carry across the switch unbroken; there is no LR scheduler in the RL path;
  and the inception replay buffer is never cleared, carrying stage-N molecules
  with stale stage-N scores into stage N+1. A stage boundary changes the
  scoring function and the termination criterion and essentially nothing about
  the optimizer state.

"Retraining" is additionally a word a reviewer can argue about — whether
continued RL under a new score counts as retraining is a definitional dispute
we would lose time on. Drop it.

**The operational distinction, which is sharp and survives:**

> COMPOSE performs inference-time intervention on an explicit realized
> molecular state while all learned model parameters remain fixed. Changing the
> goal changes only the control computation.

REINVENT carries forward a **learned policy and its optimizer state**, and
continues updating that policy under the new score. COMPOSE carries forward the
**actual molecule `x_3`** and performs zero parameter updates.

The state object is the difference, not the training. Lane 3's Mol2Mol finding
reinforces it: an input molecule can condition subsequent generation, but
"the scaffold can change within the limits of the given similarity", so it is a
**similarity anchor**, not the exact state of one ongoing executable
trajectory.

**Do not write "the goal changes at an arbitrary step."** The formal machinery
operates from any realized state, but the scalable experiment fixes `tau=3,
H=6`. Invariance to switch time is NOT established. Say instead:

> After a realized molecular prefix has accumulated, COMPOSE can change the
> active objective and continue from the exact current molecular state while
> keeping all learned parameters fixed.

**Paper-level wording, preferred:**

> COMPOSE represents molecular design as control of a learned executable
> stochastic process, allowing goals and trajectory requirements to be changed
> at inference time without updating the learned reference model.

Avoid "repeatedly training objective-specific generators" as the foil — not
every baseline does that, and the sentence is not needed for the claim to land.
