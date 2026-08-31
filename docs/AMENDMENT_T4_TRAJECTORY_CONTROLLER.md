# Amendment — T4 long-horizon trajectory controller

Status: **frozen on paper. No docking call is authorized against this spec until
Gates 1-3 pass.** Supersedes arm E (`traj`, tags trajE/trajE2/trajF) which is
retained as diagnostic only.

## 0. What the diagnostics established

Three findings, each with the evidence that produced it:

1. **Every production controller to date was short-horizon.** Structural
   comparison of official seeds to published constraint-satisfying winners:
   parp1 s0 d0.4 needs >=14 edits (+14 heavy, +3 rings), 5ht1b s7 d0.4 needs
   >=15 (+15 heavy, +3 rings), braf d0.6 needs 8-14 deletions. Controllers
   searched 1-4 edits, macro to 8.
2. **Long horizons compete when actually sampled.** trajE2, after the dilution
   fix: 12 of 13 round-winning endpoints came from L>=2; per-visit win rate
   1.5/7.3/6.8/2.9/7.9% for L=1/2/4/8/16. n=13 against 5 horizons, uniform
   expectation ~2.6 wins each -- **directional, not significant.** Not a result.
   Attribution is sound here because endpoint-only docking makes realized k == L.
3. **Unguided R_theta walks cannot navigate terminal constraints.** braf d0.6
   produced `feas 0` on every round: `_step` samples from the R_theta mark
   distribution with zero awareness of QED/SA/sim. braf requires a precise
   8-14 atom deletion while *holding* QED and similarity.

Finding 3 is why "dock the best state on the random path" (trajF) is a
diagnostic patch and not the solution: it improves *selection* over a proposal
distribution that is still blind to the constraints. Fix the proposal.

## 1. The controller

Not a single path per proposal. A **particle population over complete
molecules**, `{x_t^(i), w_t^(i)}_{i=1..N}`, propagated by the R_theta-controlled
process, weighted by cheap future feasibility, resampled when the population
degenerates, with a diverse frontier archive and staged state reuse. Because
every particle is a complete molecule, at any moment it can be inspected,
constrained, stopped, branched, archived, retargeted, or docked -- that is the
substrate advantage being demonstrated, and the long-horizon deployment of the
Feynman-Kac/Doob story already in the paper.

    pi(a | x, z, b)  ∝  R_theta(a | x)
                        * exp[ beta * V_cheap(T(x,a), z, b-1) + S_dock(x, a) ]

`V_cheap` is a long-horizon future value for the **deterministic free**
constraints (QED, SA, similarity, remaining budget, structural direction),
estimable from enormous numbers of executable rollouts at zero oracle cost.
`S_dock` is the online task-specific controller, updated **only** from counted
docking rewards. Gates 1-3 run with `S_dock = 0`.

### 1a-0. What the method IS, precisely

    twisted SMC WITHIN stages  +  diverse frontier reuse ACROSS stages

Within a stage the weighting carries the true local normalisers, so that part is
a proper Feynman-Kac / twisted SMC targeting the conditioned process. The
depth-stratified banking between stages is an **adaptive optimisation/restart
policy**: it selects states by criteria outside the FK target and is not
accounted for in the weights. The staged procedure as a whole therefore does NOT
sample one exact FK distribution and is not claimed to. Both parts are useful;
only the first is exact.

Open question worth testing before the reportable run: if resampling under a
good h_b already retains deep particles, the hand-designed banking may be
largely unnecessary. Prefer removing it to justifying it.

### 1a-0b. The displacement LADDER (no single D)

Rejected: a production target of the form `g_z(x) = 1[v(x)=0 and |dN_heavy|>=D]`
with D ~ 8-15. That would bake a competitor's endpoint statistics into our own
optimisation target, and displacement is not value -- fifteen useless atoms
satisfy it. The exact-target branch is closed (2026-08-12) precisely because a
goal with a privileged target-derived heuristic cannot settle anything.

Instead a generic ladder, ALL rungs carried simultaneously:

    d in {0, 4, 8, 12, 16}
    h_b(x; z, d) = Pr( v(X_tau)=0 AND disp(X_tau, x_0) >= d
                       for some tau <= b | X_0 = x )

d = 0 is plain feasibility, so the ladder strictly contains the previous
controller as its lowest rung. No rung is selected in advance; the COUNTED
DOCKING REWARD decides which rung pays for a given receptor. One receptor may
find d=2 sufficient and another d=16, and the algorithm discovers that -- we
never encode "braf needs deletion".

Displacement is kept as a VECTOR for the frontier,

    Delta(x) = (edit depth, |dN_heavy|, |dN_rings|, 1 - sim(x, x_0)),

with Pareto-nondominated banking per rung, so "explore" cannot collapse into
"keep adding carbon". Resampling is WITHIN rung: pooling would let the
highest-weight rung consume the others' particles.

### 1a-0c. Two measured facts that corrected the implementation

**The proposal support was never the blocker.** One-step probe of the frozen
R_theta:

| cell | shrink mass | grow mass | first shrink rank |
|---|---|---|---|
| braf (39 heavy) | 0.241 | 0.161 | **0** |
| parp1 (19 heavy) | 0.271 | 0.457 | 0 |

Deletion is the TOP-RANKED mark for braf with 24% of the mass. The hypothesis
that deletion fell outside a top-32 pool was wrong and is recorded as wrong.
Per-rule pool coverage (PER_RULE_K) was kept anyway as a cheap guarantee, but it
is not what braf needed. Untwisted net drift is -0.24 + 0.161 = -0.08
atoms/edit, so ~24 edits gives about -2: the RATE was the problem, not the
support.

**The two pressures need different budget schedules.** Dividing the displacement
term by (1+b) alongside the feasibility term made one atom of progress worth
0.125/21 ~ 0.006 of V at b=20 -- a 1.02x reweight under beta=4, numerically
nothing. Feasibility pressure must relax with b (bridge states allowed to look
bad); displacement pressure must not, because reaching d atoms requires
sustained pressure over the whole path. Decoupling them doubled braf's drift
rate (-1.1 -> -2.4 over 10 edits) and improved min_v 0.343 -> 0.2105.

The remaining scale is CALIBRATED from the measured mass split rather than
swept: selecting progress ~60% of the time against 0.241/0.161 needs an odds
ratio ~7, i.e. dV = log(7)/beta = 0.49 per atom at beta = 4.

### 1a-1. V_cheap is the finite-horizon value, not a hand schedule

The reportable controller uses the object the theory already gives:

    h_b(x; z) = Pr_R[ g_z(X_b) = 1 | X_0 = x ]        (or hitting within b)
    h_0(x)    = g_z(x)
    h_b(x)    = sum_y R(y|x) h_{b-1}(y)

    q_b(y|x, z) ∝ R_theta(y|x) * h_hat_{b-1}(y; z)

Budget dependence lives INSIDE h_b. A bridge state is allowed to look bad now
precisely when its future reachability is good, with no guessed 1/(1+b)
pressure. The local normaliser of this twist is h_b(x) itself, which is the
sharpest available check on the weighting and is unit-tested as such.

`h_b` is not exactly computable on the molecular space, so it is estimated by
**fitted value iteration on free rollout data**: label a visited state y with
remaining budget b by whether its own continuation actually reached the feasible
set within b, and regress on (QED, SA, sim, dheavy, drings, b). Zero oracle
calls. Credit flows backward from outcomes, which is exactly what makes a bridge
state defensible.

Implementation note: V := log h_hat, so **beta = 1 is the exact transform** and
beta = 0 is the untwisted reference process. The analytic 1/(1+b) form is
retained ONLY as the stage-0 bootstrap and as the beta-sweep mechanism probe; it
is not the reportable controller.

### 1a-2. The earlier bootstrap (diagnostic only)

The hinge form in section 2 needs measured `rho_j`, but `rho_j` is measured by
Gate 2 -- circular. Stage 0 therefore uses an analytic form needing no measured
rate, which still satisfies both required limits (`b` large => no pressure,
`b` -> 0 => full pressure):

    V_cheap^(0)(y, z, b) = - sum_j c_j(y) / (1 + b)

Stage 1 onward *fits* V_cheap by Monte Carlo regression on stage-0 rollouts:
from a state y with b edits remaining, the regression target is the best
feasibility actually achieved downstream,

    target(y, b) = - min_{t <= b} v(x_t)   along rollouts continuing from y

on features (QED, SA, sim, dN_heavy, dN_rings, b). This is fitted value
iteration on free data, so it costs no oracle calls and is the mechanism by
which the controller learns to *sustain* growth rather than let R_theta undo it.

### 1b. Staged state reuse

A 24-edit transformation is three manageable stages, not 24 fresh steps:

    x_0 --8 edits--> x_8*   (bank)
    x_8* --8--> x_16*       (bank)
    x_16* --8--> x_24*

Banking is what makes 20-30 effective edits affordable under the oracle cap.
If a strong state appears at step 11, the next stage continues from it rather
than rediscovering eleven edits.

## 1c. Original single-path form (retained for reference)

For state x, goal z = (QED>=0.6, SA<=4, sim(.,x0)>=delta), remaining edit
budget b:

    q_psi(y | x, z, b)  ∝  R_theta(y | x)
                           * h_feas(y, z, b-1)^alpha
                           * exp(s_psi(x, y))

Three jobs, three factors, deliberately separated:

| factor | role | source | learned? |
|---|---|---|---|
| `R_theta(y|x)` | executable chemical transition prior | frozen process model | no |
| `h_feas(y,z,b)` | can y still reach z with b edits left | **free** QED/SA/sim only | no (analytic; its rates are *measured*, not fit to docking) |
| `exp(s_psi(x,y))` | which feasible directions suit this protein | **counted docking rewards** | yes |

The separation is the point: no docking call is spent teaching the controller
what QED and similarity already say for free, and no free property is asked to
know anything about the protein.

Normalization is over the top-`APPLY_CAP` mark pool that `_step` already
enumerates, so q is a proper distribution over a known finite support and the
score function below is exactly computable.

## 2. h_feas — future-aware terminal feasibility

Shortfall vector on the *returned* molecule (T4 constrains only the returned
molecule):

    c(y) = ( (0.6 - QED(y))_+ / 0.6 ,
             (SA(y) - 4)_+ / 4 ,
             (delta - sim(y,x0))_+ / delta )

Per-coordinate per-edit improvement rate `rho_j` is **measured** in Gate 2 from
cheap trajectory statistics; it is not a free parameter.

    h_feas(y, z, b) = exp( - sum_j ( c_j(y) - b * rho_j )_+ / T )

Required properties, and they hold by construction:

* **b large => h_feas -> 1.** Bridge states are allowed to look bad. This is the
  explicit requirement that constraint pressure NOT be applied uniformly.
* **b -> 0 => h_feas -> exp(-sum_j c_j(y)/T).** Full terminal pressure.
* Pressure rises monotonically as the horizon closes.

Similarity needs separate treatment and this is where a naive reading fails:
sim decreases roughly monotonically with edits and is not cheaply recovered, so
for the sim coordinate the quantity of interest is not "can it improve" but
"will it still clear delta at the stopping time". Under adaptive stopping the
correct predicate is existential -- does there exist t <= b with x_t feasible --
not "will x_b be feasible". Gate 2 measures the sim-vs-edit decay to set
rho_sim, and reports it as a decay rate rather than an improvement rate.

Note the tightness this predicts for growth: +14 heavy atoms on a ~16-atom seed
roughly doubles size, and Morgan-Tanimoto to x0 falls steeply. The corrected
audit reached only 0.5500 (parp1) / 0.5098 (5ht1b) best similarity. delta=0.4
growth is feasible but narrow; delta=0.6 growth may be genuinely infeasible and
Gate 2 must be allowed to say so.

## 3. Adaptive stopping time

A rollout of sampled length L that selects its molecule at step k <= L:

* dock `x_k`;
* record realized horizon **k**, never L;
* credit reward only to transitions 1..k;
* discard steps k+1..L for that observation.

Otherwise a rollout whose useful chemistry finished at step 4 is filed as
evidence for L=16. (trajF already implements this: it stores `(steps[:k], k)`
and telemetry reads that k.)

## 4. The trajectory objective, stated explicitly

The earlier code summed path features, was changed to averaging because a
16-step path took ~16x the gradient at equal reward, and that switch was made
for stability without declaring the objective. Declaring it now.

Ordinary expected-terminal-reward policy gradient is

    grad J = E_tau [ A(tau) * sum_{t=1..k} grad log pi(a_t | x_t) ]

so **the sum is the trajectory log-likelihood, not a bug.** A 16-decision path
*should* contribute 16 decisions' worth of gradient.

The actual defect was elsewhere: the score function was never centered. With
s_psi(x,y) = psi . phi(y) linear in the fingerprint,

    grad_psi log q_psi(y|x) = phi(y) - E_{y' ~ q_psi(.|x)} [ phi(y') ]

The implementation used `phi(y)` alone. Summing *uncentered* dense fingerprints
over a path accumulates a length-proportional common-mode component -- every
molecule's fingerprint shares most of its mass -- so the update grew with path
length for reasons unrelated to what made the path good. Averaging masked that
common mode instead of removing it. **Centering removes it at the source, and
then summing is correct.** The baseline expectation is computable over the same
mark pool q normalizes on (fingerprints are ~0.1ms each against a 6.74s R_theta
call, so the pool expectation is free at this scale).

Frozen objective -- per-transition clipped PPO, batch-normalized by total
transitions:

    L(psi) = (1 / sum_tau k_tau) * sum_tau sum_{t=1..k_tau}
                 min( r_t A_tau , clip(r_t, 1-eps, 1+eps) A_tau )

    r_t = q_psi(a_t|x_t) / q_psi_old(a_t|x_t)

Deliberate choices, each with its reason:

* **Per-transition ratios, not a whole-trajectory product.** The product of k
  ratios has variance growing with k; that, not the summation, is what
  destabilizes long paths.
* **Normalize by total transitions in the batch**, not per path. Long paths keep
  the greater influence they have earned by making more decisions, while the
  step size stays scale-free.
* **Horizon-bucketed advantage normalization.** A_tau is standardized within its
  realized-k bucket, so a horizon cannot look good merely by having a higher
  mean reward than other horizons.
* **Explicit KL guard** on q_psi against q_psi_old, in addition to clipping;
  long-horizon updates are the case where clipping alone is known to drift.
* A_tau uses terminal reward only (no learned value function at 1000 oracle
  calls); it is therefore Monte Carlo and shared across the prefix.

## 5. State reuse

A found prefix is not rediscovered. Having reached x_8, later cycles continue
from x_8 rather than restarting from x_0, which is what makes effective 20-30
edit optimization affordable under the oracle cap. The asymmetry that makes long
paths viable at all:

    16 cheap edits  ->  1 counted docking call

## 6. Gates. No Vina before these pass.

Gate 1-3 spend **zero** oracle calls.

**Gate 1 — can it navigate the constraints at all?**
Thousands of paths from the constraint-aware sampler. braf d0.6: does it produce
feasible endpoints? If ~0 feasible after >=10,000 cheap trajectories, the
proposal process is still wrong and **nothing is docked.**

**Gate 2 — can it make sustained structural progress?**
On growth cells, measure d(N_heavy), d(N_rings), sim, QED, SA against realized
horizon. Requirement: L=8/16/24 must produce qualitatively different chemistry,
not wander and undo itself. This gate also *measures* rho_j for h_feas.

**Gate 3 — does path credit reach bridge transitions?**
Synthetic already-paid reward. The policy must raise the probability of the
*bridge* transitions, not only the endpoint motif. The synthetic test must keep
the unrewarded bridge (the first version was rigged: the motif appeared at every
step, so endpoint credit already saw it).

**Gate 4 — small docking diagnostic, only after 1-3 pass.**
parp1 growth, 5ht1b growth, 2x braf d0.6, one control. 100-200 calls. Scale only
if the plateau breaks.

## 7. What is not claimed

Reproducing InVirtuoGen's -14 numbers is **not** promised. Published winners are
used only to infer the generic requirement -- support long growth/deletion
trajectories under terminal constraint control -- and their endpoint structures
are discarded during reportable optimization. The optimizer never targets them.

The objective is to build the correct optimizer for COMPOSE's executable
transition geometry and measure how far that substrate goes.

---

## 8. Overnight results, 2026-08-24 (ZERO oracle calls)

### 8.1 The barrier correction

A linear similarity shortfall normalised by delta is far too flat at the wall.
Measured: at sim 0.388 against delta 0.40 the shortfall contributes 0.03 while
the displacement term contributes 0.5 -- displacement outweighed the binding
constraint 16:1, and particles walked out of the feasible set while still
"being penalised" for it. Growth reached |dheavy| = +12 with sim 0.277: the
displaced states existed and were simply unreturnable.

Similarity is a hard endpoint constraint, quasi-monotone and unrecoverable, so
it takes a quadratic barrier with a safety margin (0 above delta+0.05, 1 at
delta, 4 at delta-0.05) instead of a linear term. QED and SA keep the (1+b)
relaxation because they ARE recoverable.

This is also the mechanism behind a general chemical lesson, and no extra
machinery was needed to get it: large size change while preserving Tanimoto
requires LOCALISED edits -- one contiguous substituent destroys few fingerprint
bits, the same atom count removed scatter-wise destroys many. Defending sim at
every b is what selects localised edits.

### 8.2 Ladder reach, twisted vs untwisted (feasible AND |dheavy| >= d)

| cell | beta | d4 | d8 | d12 |
|---|---|---|---|---|
| parp1 s0 d0.4 | **4** | 176 | **55** | **5** |
| parp1 s0 d0.4 | 0 | 40 | 0 | 0 |
| 5ht1b s7 d0.4 | **4** | 99 | **15** | **1** |
| 5ht1b s7 d0.4 | 0 | 9 | 0 | 0 |

Raw evidence of the mechanism: `parp1/b4 st2 d17: dheavy +5.4 / max +13,
sim 0.450, feas 38` -- growth past +13 heavy atoms WHILE holding sim above
delta=0.4. Before the barrier the same controller reached +11 at sim 0.388 with
17 feasible. The beta=0 control still exits the feasible set (sim 0.317-0.342).

State reuse compounds across stages rather than collapsing shallow (parp1 d4:
25 -> 84 -> 160 -> 176 across stages), lineages stay at 24-48 of 48, and ESS
holds at 4-9.6. No collapse-to-one-lineage failure mode.

### 8.3 What is NOT established

* **No docking has been run inside the search.** There is no T4 score here and
  nothing in this section may be reported as a benchmark result.
* braf (deletion, delta=0.6) has NOT passed: the twist drives deletion
  (dheavy -4.4 vs +0.2/-1.1 for control) but no feasible displaced state was
  produced, because delta=0.6 leaves very little similarity slack.
* d16 is unreached on every cell.
* The held-out validation (jak2, fa7 -- winners never opened) was still queued
  behind the container cap at time of writing. Until it reports, the claim that
  these are GENERAL lessons rather than lessons fitted to the cells whose
  winners were inspected is UNTESTED. The prediction is registered in advance:
  the same controller, with no per-cell tuning, should raise d4/d8 reach on
  jak2/fa7 by a margin comparable to parp1/5ht1b.

---

## 9. Route audit: the bridge is ~5-6 unrewarded edits (2026-08-24, ZERO oracle calls)

Native operator audit of legal routes toward development landmarks. Guidance is
SHARED FINGERPRINT BITS with the landmark, not Tanimoto -- Tanimoto divides by
|A union B| so every intermediate growth step reads as a regression, which is
what pinned the earlier bidirectional beam at 0.55 while it was supposed to grow.

### 9.1 Three hypotheses eliminated

| cell | steps | zero/negative-gain steps | steps outside APPLY_CAP |
|---|---|---|---|
| parp1 s0 d0.4 | 28 | **89%** | **0** |
| 5ht1b s7 d0.4 | 34 | 50% (a clean 22<->21 cycle) | **0** |
| braf s10 d0.6 | 3 | 0% | 0 |
| braf s9 d0.6 | 4 | 25% | 0 |

* **Candidate truncation / reference rank is not what causes the observed
  greedy plateau.** Zero actions SELECTED BY THIS GREEDY AUDIT fell outside
  APPLY_CAP. Note the weaker claim: the audit never produced a complete legal
  path to a landmark, so this does NOT establish that a full path exists inside
  the truncated support. It establishes only that truncation is not what caused
  the cycling. The earlier ring-rank finding (first cycle_close at rank 305 vs a
  cap of 300) remains real and separate.
* **Reference-law bias is not the primary bottleneck.** Chosen actions ranged
  from rank 0 to 277 with probabilities 2.5e-1 down to 1e-6; the law offers them.
* **Operator-family allocation is not the bottleneck.** The families used are a
  reasonable mix of insert/delete/restate/reroute.

### 9.2 What IS the bottleneck

parp1 requires 17 atom_insert and the greedy route selected **2**, spending 26
steps on atom_restate_semantic while shared bits sat at 31 for 24 consecutive
steps. 5ht1b oscillated insert/delete 22<->21 for 17 rounds.

The mechanism: inserting ONE atom of a 14-atom fragment creates no shared
environment with the goal, because a Morgan radius-2 environment does not match
until the new atom's own neighbours exist. The first several atoms of any
appended fragment therefore return ZERO reward, and the shared structure appears
in a block only as the fragment closes.

    OBSERVED: one-step credit is INADEQUATE. parp1 held 31 shared bits for 24
    consecutive steps; 5ht1b oscillated 22<->21 for 17 rounds.

The plateaus observed here run ~5-6 edits, but that number is NOT adopted as a
constant. What the audit establishes is that a one-step guide cannot cross these
routes; the useful temporal scale is to be LEARNED, by carrying a ladder of
segment lengths and letting the reward decide, exactly as with the displacement
rungs.

This is the growth-vs-deletion asymmetry in one number: deletion changes shared
structure immediately (braf: 0-25% stalled steps, clean progress), growth does
not (parp1: 89% stalled).

### 9.3 Consequences

* Any MYOPIC value function stalls here by construction. Every scalar term added
  on 2026-08-24 -- ring displacement, similarity efficiency, scaffold
  preservation, gamma debiasing -- is a 1-step signal applied to a problem with
  no 1-step signal, which is why none cleared the 0.70 docking noise band.
* The trajectory arm was directionally right but under-ranged: L in {1,2,4} is
  shorter than the bridge, so most of its rungs cannot cross it.
* A value function needs >= 6-step lookahead, or the controller needs composite
  moves that traverse a whole fragment before being scored.
* |dHeavy| as a horizon metric undercounts by ~40% (parp1: 23 operations
  required vs 14 reported), so depth budgets were also short.

---

## 10. WITNESS ROUTE: parp1 s0 d0.4 (2026-08-24)

A complete legal executable path from the parp1 seed to InVirtuoGen's -13.6
winner, found by executor-only backward search in 70 seconds.

**Reachability was never the obstacle.** 18 edits, every transition a legal
executor rewrite. The route was found with NO model call: reachability is a
property of the executor, not of R_theta's probabilities, and dropping R_theta
from route search made it ~1000x faster (20 min -> 1 s per attempt).

### 10.1 The route

    depth  0  CN(C)Cc1ccc2c(c1)CNC(=O)c1cccn1-2                 seed, fused tricycle
    depth  3  CCNCc1cccc(CNC(=O)c2ccc[nH]2)c1                   TRICYCLE OPENED
    depth  6  C=CCCCCc1cccc(CNC(=O)c2ccc[nH]2)c1                chain grown
    depth  9  O=C(NCc1cccc(CCCc2ccccc2)c1)c1ccc[nH]1            phenyl added
    depth 12  Cc1ccc(CCCc2ccc3c(c2)CNC(=O)c2cccn2C3=O)cc1       TRICYCLE RECLOSED
    depth 15  CC(CCc1ccc2c(c1)CNC(=O)c1cccn1C2=O)c1ccc2c(c1)CNC2
    depth 18  O=C1NCc2cc(C3CCCC3Cc3ccc4c(c3)CNC(=O)c3cccn3C4=O)ccc21   winner

The program is: **BREAK the seed scaffold, GROW, then RECLOSE it.**

### 10.2 The bridge, measured in the protein objective

    depth   0    3    6    9   12   15   18
    ds   -7.2 -7.5 -7.6 -8.8 -10.8 -11.7 -13.6

Binding moves 0.4 kcal/mol over the first SIX edits -- flat -- while the
scaffold is open, then pays 6 kcal/mol once the ring recloses.

CORRECTION: an earlier reading of a BROKEN over-deleted route showed a monotone
curve and was used to claim "the delayed-credit problem is only in the
fingerprint proxy, docking gives usable gradient". That was WRONG. On the real
route the protein objective carries the same flat bridge the cheap proxies do.

### 10.3 Why every controller in this project fails here

During the break-and-grow phase EVERY signal we have steered by moves the wrong
way: similarity falls, shared bits fall, ring count falls, and docking is flat.
Our best-ever parp1 result is -10.5, which sits at depth ~12 on this curve --
exactly the reclose. We reach the reclose and never get past it.

This is structural, not a tuning failure. Any controller that scores individual
edits by any of these quantities is blocked by construction.

### 10.4 What this does NOT establish

* One cell, one landmark. 5ht1b and the held-out cells are untested.
* The route was found WITH the answer in hand. It proves the basin is
  reachable and characterises the bridge; it says nothing yet about whether a
  target-blind controller can find it.
* No controller change has yet been shown to cross the bridge.

---

## 11. WHY THE DETOUR IS FORCED: saturated anchors (2026-08-25)

The parp1 target needs new material attached at two positions which, IN THE SEED
ITSELF, are already valence-saturated:

    target atom 28 (C) must bond to target 27 and 16, which map to
      seed atom 18:  N, degree 3, implicit H 0
      seed atom  7:  C, degree 3, implicit H 0
    atom_insert actions reaching either anchor, at the seed: ZERO

This is not a construction-order problem and no build ordering can avoid it --
there is no state in which those anchors have free valence until a bond is
broken.

### SCOPE CORRECTION (measured across all five landmark cells)

Only a SMALL MINORITY of the new material is anchor-blocked:

| cell | new atoms | anchor-blocked |
|---|---|---|
| parp1 s0 | 17 | **1** |
| 5ht1b s7 | 18 | **2** |
| braf s10 | 7 | 1 |
| braf s11 | 5 | 0 |
| braf s9 | 1 | 0 |

So "the target requires material at saturated positions, therefore the scaffold
must be opened" is TOO STRONG as first written. Sixteen of parp1's seventeen new
atoms attach without obstruction -- which is exactly why the compiler reached
31/33 atoms and stalled on one.

The consequence matters: the detour is forced for EXACT REPRODUCTION of the
landmark, but the benchmark does not ask for their molecule. It asks for a good
molecule under the constraints. With 16/17 of the growth unobstructed, a
controller may well reach comparable chemistry without ever opening the
scaffold, and whether the detour is necessary FOR SCORE is untested.

### SECOND CORRECTION: the blocked atom is NOT what causes the infeasible stretch

Tested by docking every T4-feasible state on the compiler's own build:

    +0 heavy (seed)  -7.3      +4  -7.7
    +1               -7.7      +5  -7.7
    +2               -7.7      +6  -7.8
    +3               -7.6      +7  -7.8      vs our banked -10.5

Two distinct effects were being conflated:

* The one anchor-blocked atom forces a ring to be OPENED. Local, real.
* The INFEASIBLE STRETCH has a different cause. The compiler's build leaves the
  feasible set after ~8 atoms -- long before it reaches the blocked atom.
  Partial ring construction destroys similarity because a half-built ring system
  shares almost no fingerprint environments with the seed, and similarity only
  recovers when the ring CLOSES. That is precisely why the witness route is
  infeasible from depth 2 to 9 and feasible again at depth 10, immediately after
  the reclose.

So the general statement is NOT "the scaffold must be opened". It is:

    A PARTIALLY BUILT RING SYSTEM IS UNRECOGNISABLE TO A FINGERPRINT, SO
    FEASIBILITY ONLY RETURNS ON COMPLETION.

Any controller requiring per-edit feasibility therefore cannot build a ring
system at all -- a partial ring is always worse than no ring. That covers the
growth cells generally, not just the one blocked atom on parp1.

NOTE ON PROCESS: this claim has now been narrowed twice (first from "most new
material is blocked" to 1-2 atoms, then from "the blocked atom forces the
detour" to the above). Treat the mechanism as established and the framing as
provisional until a controller test confirms it.

### The causal chain for exact reproduction, every link measured

1. One required atom (parp1 atom 28) attaches only at seed positions that are
   valence-saturated.
2. The scaffold must therefore be OPENED to free valence. Forced by chemistry.
3. Opening the fused ring collapses similarity to 0.128 against delta = 0.40,
   so the molecule leaves the feasible set for ~9 states.
4. Re-closing the ring restores similarity to 0.42+ and recovers feasibility;
   8/8 bridge states escape in ONE closure.
5. Across the detour EVERY signal is flat or adverse: similarity, shared
   fingerprint bits, ring count, and docking (-7.2 -> -7.6 over six edits,
   then -13.6 after the reclose).
6. Therefore no controller scoring individual edits can traverse it. This is
   structural, not a tuning failure.

### What this retro-explains

* Why five controller variants (rebuild option, bridge-aware banking,
  returnability trigger, fragility trigger, segment strata) all scored -8.3 to
  -8.6 against a -10.5 baseline.
* Why weighted A* stalled at D=39 for 26 expansions: its heuristic is flat over
  exactly this detour.
* Why the greedy compiler stopped at 31/33 atoms.
* Why growth cells fail and deletion cells (braf) succeed: deletion frees
  valence rather than requiring it.

### What it implies for the controller

A controller must be able to commit to OPEN -> ATTACH -> RECLOSE as a single
decision whose purpose is evaluated only at the endpoint. That is the compiled
macro / temporal-abstraction design: purpose selects a program, a compiler
orders its dependencies, and R_theta realises each primitive edit. Every
underlying transition stays inside the frozen process; no new executor
primitive is required.

---

## 12. METRIC ERROR: every "realized" figure in sections 10-11 was inflated

The compiler's progress measure used

    rdFMCS.FindMCS(..., bondCompare=BondCompare.CompareOrder)

which compares a KEKULE view. An aromatic ring and a saturated ring of the same
connectivity therefore score as a FULL MATCH. The compiler was hill-climbing a
number that could not see the difference it was supposed to close.

### What the numbers actually are

| measure | reported | strict (CompareOrderExact) |
|---|---|---|
| final build vs target | "33 atoms + 38 bonds of 33+38" | **13 atoms / 14 bonds** |
| build progress | "100% of target material" | **24 + 27 of 33 + 38 (72%)** |
| earlier milestones | "93%", "66/71" | inflated by the same factor |

The final molecule under the loose metric was C27H29N3O3 against the target's
C27H25N3O3 -- same heavy atoms, four more hydrogens, i.e. wrong saturation. Its
strict overlap with the target (13 atoms) is LOWER than the SEED's (16), so by a
correct measure that build moved AWAY from the target while the metric reported
completion.

Adding an aromatic-atom-count penalty did NOT fix this: it charges a global
count while leaving the MATCHING loose, so the compiler kept optimising the
broken objective.

### Consequences

* `x_seed ~> x_IVG` is **NOT solved**. There is no verified forward teacher
  program, so nothing downstream of it (macro extraction, information-removal
  ladder, teacher distillation) can start.
* Corrected: 24 + 27 of 33 + 38 in 14 ops / 2.4 s, blocked by saturated anchors.

### What is NOT affected

* The backward WITNESS ROUTE (section 10): verified by canonical SMILES equality
  at the seed, independent of this metric. Its docking curve stands.
* The speed result: direct action selection is ~7,000x faster than
  enumerate-and-score. That is a wall-clock fact.
* Mechanism findings measured independently of the MCS: saturated anchors,
  one-edit bridge escape (8/8), operator-set inversion asymmetry.

### Process note

This is the fourth narrowing of a claim in this lane in one session. The failure
mode each time was reporting a derived number before checking what the measure
could actually distinguish. Any future progress metric on this lane must be
validated against a case where the answer is known -- here, "does it call the
seed closer to the target than a wrong-saturation molecule?"
