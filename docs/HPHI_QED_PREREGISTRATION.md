# QED / GrIDDD region-`h_φ` controller — PREREGISTRATION

**Binding implementation protocol for the GrIDDD/QED experiment.**
The paper-level ruling lives in section J of
`MASTER_PLAN_EXPERIMENTS_AND_COMPARATORS.md`; this document is the *how*.
Task definition, source cohort, success criterion, 20-candidate budget and the
size-fixed ablation remain governed by `GRIDDD_JIN_PROTOCOL.md`.

> **FROZEN 2026-08-14, before any official Jin test outcome exists.**
> `h_φ`'s weights do not yet exist at the time of freezing. That is the point:
> the controller cannot be chosen after seeing the official result.

---

## 1 · Why Policy B was rejected

`GRIDDD_JIN_PROTOCOL.md` preregistered **`stochastic one-step COMPOSE control`**
(Policy B) — a receding one-step QED tilt, `π(y|x) ∝ R_θ(y|x)·QED(y)` over the
full legal fiber.

It was evaluated on the **disjoint 64-source development panel** and produced
**0 successes in 320 trajectories**. No official Jin test outcome had been
observed at any point.

**Diagnosis, not merely a failure count.** A one-step terminal tilt reweights by
the *immediate* QED of each successor. The benchmark region (QED ≥ 0.90 **and**
similarity ≥ 0.40) is typically not reachable in one edit from a source at
QED ≈ 0.75, so the quantity Policy B optimizes is nearly uninformative about
whether the region is reachable *at all* within the remaining budget. It is a
greedy approximation to a finite-horizon reachability problem, and on this task
the approximation carries almost no signal.

### The wording rule — binding on the manuscript

Region-`h_φ` is a **pre-outcome amendment consistent with the pre-existing
finite-horizon COMPOSE controller**, motivated by a developmental failure of the
cheap one-step approximation.

**Do NOT write** that `h_φ` "restores" an originally-frozen policy, or that
compute cost was the only thing ever separating them. `GRIDDD_JIN_PROTOCOL.md`
contains a section explicitly titled *"Why NOT the rollout `ĥ`"* — the protocol
**declined** it. The stronger phrasing is refutable from our own document and is
barred. The amendment's validity rests solely on predating official evaluation,
and that is sufficient.

**Policy B is retained as a reported developmental negative control**, not
deleted.

---

## 2 · Frozen `R_θ` — the paper-wide invariant

| | |
|---|---|
| checkpoint | `sha256 c979cdb3d7b0b403bfbf7bfb0aa5098b2588c6d4217770c2c58292b7c4e53de8` |
| retrained for this experiment? | **NO** — and not for any experiment in the paper |
| role of `R_θ` here | reference transition law **and** frozen feature extractor (encoder used unchanged; gradients never flow into it) |

---

## 3 · Data — exclusions and disjointness

| split | n | role |
|---|---:|---|
| `hphi_train_1024` | 1,024 | `h_φ` training corpus roots |
| `hphi_valid_128` | 128 | fresh validation, used **ONCE** |
| `dev_panel_qed_64` | 64 | controller development panel |
| official Jin test | 800 | **never touched until the final run** |

**Excluded from the eligible pool before sampling:** the official Jin
`qed` / `logp04` / `logp06` test sources **and** the 64-source dev panel —
**1,516 excluded** from **66,632 eligible**.

**Verified disjoint** (recomputed at freeze time):
`train ∩ valid = 0`, `train ∩ dev = 0`, `valid ∩ dev = 0`.

**Root selection:** QED-stratified into 10 strata of width 0.01 over
`[0.70, 0.80]`; deterministic `sha256` rank within each stratum; **no RNG**.
Equal allocation per stratum, deliberately *not* following the population skew,
so the harder low-QED sources are covered.

---

## 4 · The corpus

| | |
|---|---|
| identity | `train_1024x02_H24`, `sha256 647f8265f5143e2b15dc047de42e837203593c42beb7e45e2b94e55e41602581` |
| shape | 1,024 sources × 2 trajectories × H24 = **49,152 transitions** |
| relation to the original plan | **identical transition count** to the original 1,024 × 8 × H6; depth was bought with replicates, not with new budget |
| why | no reachability saturation was observed at budgets ≤ 24, and H6 captured only **43 %** of QED-0.85 reachability (`HORIZON_AMENDMENT_H24.md`) |
| realized content | 39,663 unique states · 51,200 prefix examples · budgets evenly covered |
| benchmark-region incidence | **3.12 %** of trajectories, **5.7 %** of sources, 64 hits, 687 gateway states |

**This corpus is never a benchmark result.** It is controller training data.

---

## 5 · Region goal language

The controller is conditioned on a **region**, not on QED alone — this is what
makes it a general controller rather than a QED controller.

```
region  = (τ_q, τ_s)          goal set  B = { x : QED(x) ≥ τ_q  AND  sim(x, x₀) ≥ τ_s }
τ_q ∈ {0.75, 0.80, 0.85, 0.90, 0.95}      τ_s ∈ {0.30, 0.40, 0.50, 0.60}
```

**20 registered regions**, trained jointly. The GrIDDD benchmark region is
`(0.90, 0.40)` — one member of the family, given no special weight in training.
`sim` is similarity to the trajectory's own source `x₀`.

---

## 6 · `h_φ` — architecture and training contract

**Target: finite-budget HITTING reachability.**

```
h_φ(x, x₀, region, b)  ≈  P( ∃ t ≤ b : X_t ∈ B_region  |  X_0 = x,  X ~ R_θ )
```

First-passage within the remaining budget — **not** terminal occupancy. This is
the right object for an anytime controller that STOPs on first qualification.

### Features — `INPUT_DIM = 1055`

| block | dims | content |
|---|---:|---|
| embeddings | 1024 | `[e_x, e_x₀, e_x − e_x₀, e_x ⊙ e_x₀]`, `EMBED_DIM = 256` |
| properties | 4 | `QED(x)`, `sim(x, x₀)`, `τ_q`, `τ_s` |
| **margins** | 2 | `QED(x) − τ_q`, `sim(x, x₀) − τ_s` |
| budget | 25 | one-hot over `b ∈ [0, 24]` |

Embeddings are the frozen `R_θ` encoder's permutation-invariant graph-level
`global_state` at `TIME_POINT = 0.5`, `CANONICAL_SLOTS = 48`.

### Head

`Linear(1055→512) · ReLU · Dropout(0.1) · Linear(512→256) · ReLU · Dropout(0.1)
· Linear(256→128) · ReLU · Linear(128→1)`, sigmoid to `[0, 1]`.

### Boundary condition — enforced, never learned

```
h_φ(x, ...) := 1   whenever  x ∈ B_region,  regardless of network output
```

In-region states are **excluded from the training set** rather than supervised
to 1, and `h_with_boundary()` enforces it at inference.

### Losses

| term | weight |
|---|---:|
| cross-entropy against the Monte-Carlo hitting label | 1.0 |
| Bellman consistency on **observed** transitions | `BELLMAN_WEIGHT = 0.3` |

### Optimization — frozen

`EPOCHS = 40` · `PATIENCE = 5` (early stopping) · `VAL_FRACTION = 0.15` held out
**by source**, never by example · **best-validation checkpoint is selected and
saved**, never the last epoch.

> The first trainer saved the *last* model while validation loss rose
> monotonically from epoch 0 (0.1713 → 0.1821). Best-checkpoint selection is
> frozen here so that defect cannot recur silently.

---

## 7 · Inference — the native controller

### Exact rejection sampling from `R_θ·h_φ`

```
1. sample a mark from the frozen R_θ         4. reject self-loops
2. apply one rewrite                          5. accept with probability h_φ(y)
3. canonicalize                               6. else resample
```

**Precisely stated** — this matters, and an earlier wording here was loose.
`h_φ ∈ [0, 1]` and `h_φ` is a *state-level* function, so aliased successors
aggregate correctly, and **conditioned on acceptance the accepted move has
exactly the `R_θ·h_φ` law**.

But the cap is finite. When all `MAX_PROPOSALS` are rejected the sampler
**fails** and the trajectory ends, so the capped procedure as a whole is **not**
an exact realization of the controlled kernel — it carries an explicit
sampler-failure event. **Do not describe the capped sampler as exact.**

This is acceptable *because* the preregistration treats cap pressure as a
**diagnostic**, with a frozen SMC escalation for the case where it becomes
severe (§8). It is a measured quantity, not a defect to tune away.

### STOP semantics

`τ = min{ t ≤ H : X_t ∈ B_region }` — an **online stopping time**, evaluated on
the realized state at each committed edit. The trajectory halts at the first
qualifying molecule and returns it.

This is a **policy, not retrospective selection**: nothing is chosen after
seeing the whole trajectory, so it carries no selection advantage. Max `H = 24`.

### Returned candidates

**20 per source**, matching GrIDDD's returned-candidate budget — matching the
task interface, *not* GrIDDD's internal denoising shape.

---

## 8 · Escalations — permitted ONLY under pre-existing triggers

Neither may be invoked at discretion. Both are governed by rules frozen **before**
this preregistration; nothing here adds latitude.

| escalation | governing trigger |
|---|---|
| rejection → **twisted-SMC** | `HORIZON_AMENDMENT_H24.md` step 7 — healthy acceptance → **stop, do not escalate**; *calibrated but collapsed* acceptance → frozen twisted-SMC. Same `h_φ`, same `R_θ`. **SMC is inference, not retraining.** |
| **conditional continuation labels** | `HPHI_V1_CORPUS_PREREGISTRATION.md` decision rule — only when goal events are *severely tail-starved*, and only from preregistered informative prefixes under the same frozen `R_θ`. *"Run more of the same" is explicitly not the default.* |

---

## 9 · The evaluation ladder

```
64-source dev panel  →  128-source fresh validation (ONCE)  →  official 800
```

Reported at each rung: success rate, first-hit step, similarity, QED, value
calibration, and acceptance rate. The H6 / H12 / H24 compute frontier is a
**separate prespecified set of runs**, defined in section 12 — it is **not**
reconstructed from this ladder's H24 trajectories.

**Comparators on the dev panel:** unguided `R_θ` · Policy B · `R_θ·h_φ`.

### No-rescue rules — binding

1. **The 128-source validation is used exactly ONCE.** If it fails, the result
   is reported as a failure. There is no second validation set.
2. **There is no QED rescue branch.** If `h_φ` does not qualify, the experiment
   is reported as negative and **this controller path closes** under its own
   preregistered rules — no patched variant is tried to rescue it.

   **State the scope precisely.** A negative result closes *the current QED
   region-`h_φ` controller path*. It does **not** retroactively invalidate
   COMPOSE, the already-banked Pareto results, or any other frozen finding, and
   it does **not** establish that no multi-objective controller is possible.
   What it does mean, given the current paper plan, is **stop and reassess
   before building any new claim-bearing pipeline on top of it** — in
   particular, do not respond by inventing a five-objective workaround.
3. **The official 800 are touched once**, after the ladder completes.
4. No hyperparameter, region, threshold, or horizon may be re-chosen after
   observing an official-test outcome.
5. Development-panel evidence (steering 0.188 → 0.391; `Var_R(h)` sd 0.281) is
   **developmental**, never reported as a result.

---

## 10 · Compute accounting — this experiment's instance

Per the paper-wide doctrine (master plan section I), three axes are reported and
**none is traded against another**. Specifically for this experiment:

| axis | reported quantity |
|---|---|
| training | `R_θ` training cost (**shared, amortized, unchanged**) + `h_φ` training cost (**this objective only**) |
| adaptation | what a *new* objective would require — stated honestly, not assumed to be zero |
| inference | kernel calls · objective (QED/similarity) evaluations · realized STOP/edit distribution · acceptance rate · wall time · SMC particles and ESS **if** the escalation fired |

**The oracle cost is never offered as compensation for not retraining `R_θ`.**
The conclusion drawn is only: *COMPOSE obtains its performance from a reusable
goal-independent molecular process, which carries a particular inference cost,
reported separately.*

---

## 11 · Execution budget ledger

**Run-level compute authorizations live here, not in the master plan.** The
master plan carries the scientific ruling; this section carries what was
authorized to spend executing it. The two are deliberately kept apart.

### Amendment 1 — encode authorization $2 → $5 (2026-08-14)

| | |
|---|---|
| **job** | `modal_apps/hphi_encode_app.py` — frozen `R_θ` embeddings for the 39,663 unique corpus states |
| **prior authorization** | **$2**, a hard ceiling |
| **measured cost** | ~70 min × 80 containers × 1 CPU = **93.5 core-hours ≈ $4.39** |
| **amended authorization** | **$5** — margin over the measured figure, not a blank check |

**Why this is preprocessing, not an experiment.** The corpus is already frozen
(`sha256 647f8265…41602581`) and `R_θ` is frozen
(`sha256 c979cdb3…4e53de8`). The encode is therefore a **deterministic
function of two frozen inputs**: it has exactly one correct answer, produces no
scientific result, and admits no researcher choice. It is computed once and
reused forever.

**What happened, recorded plainly.** A first attempt was launched at $4.39
against the $2 ceiling. It was stopped ~8 minutes in, before any shard had
persisted, at a cost of roughly **$0.50** and with **nothing recoverable**. The
cap was exceeded because the cost was not priced before launch — an earlier
estimate of ~21 min / ~$1.35 was wrong by more than 3×. **This is not recorded
as having been under cap.**

**What changed so this cannot repeat destructively.** Encoding is now its own
job (`encode → persist → exit`), and **every shard writes and commits its own
file the moment it finishes** rather than one merged write at the end. The job
is idempotent: a shard whose file exists returns without loading the model, and
the driver skips persisted shards. An interruption now costs at most one shard,
not the whole run.

> Prior to this split, the same encode was destroyed **twice** by stopping a
> trainer that held completed embeddings in memory — 93.5 core-hours each time.
> See the `persist-expensive-deterministic-artifacts` rule.

### Standing rule

Price a run **before** launching it and state the figure. If it would exceed the
standing authorization, do not launch — request an amendment to a specific
number and record the reason here first.


---

## 11b · Optimization sequencing — BEFORE the 128, not after

**Frozen 2026-08-15.** Engineering optimization has a place in the ladder, and
it is **not** between the 128 and the 800.

```
64 dev  →  SELECT the frozen inference branch (rejection vs SMC)
        →  implement + qualify exact optimizations on the SURVIVING branch
        →  128 fresh validation  →  800 official
```

**Why not after the 128.** The 128-source validation exists to validate the
**actual implementation that will touch the official test set**. Running it on a
slow reference implementation and then introducing a faster one for the 800
would mean the official run uses code that no validation rung ever exercised —
even if the speedup is mathematically exact.

**Why not before the 64.** The 64 selects which branch is production. Optimizing
the rejection path before knowing whether rejection survives may be optimizing
code that gets discarded.

| after the 64 says… | then |
|---|---|
| rejection is **healthy** | keep rejection; implement exact batched scoring if profiling justifies it |
| cap pressure fires the **frozen SMC trigger** | move to SMC **first**; do not optimize the rejection path as though it were production |

Then, on whichever branch survives: profile it · implement **only**
semantics-preserving speedups · **qualify the fast implementation against the
reference** · **freeze it** · run the 128 · run the 800 unchanged.

**Batching changes how many `h_φ` scores are computed per pass. It must never
change how many proposals the frozen sampler may draw** (cap = 40, drawn in
chunks of 16). Any expected speedup is an engineering hypothesis until profiled.

---

## 12 · Inference-cost reporting and the compute frontier

**Frozen 2026-08-15, before the 64-source development result exists.**

### 12.1 Resource logging — descriptive, secondary, and analyzed afterwards

> **Inference-cost reporting.** All claim-bearing runs will record wall-clock
> inference time, kernel/support evaluations, objective evaluations, committed
> edits, STOP step, and returned candidates. These resource quantities are
> **secondary descriptive outcomes** and do not alter the primary task protocol.
> Performance and resource use will be reported separately; differences in
> training cost will not be used to offset differences in inference cost.

**Why this is preregistered even though it changes nothing.** Measuring cost
after the fact is legitimate — these are descriptive quantities, and no analysis
choice about them can alter the primary outcome. What would *not* be legitimate
is **choosing which compute budgets to showcase after seeing performance**. That
is cherry-picking, and freezing the operating points below removes the
opportunity before any result exists.

### 12.2 ⬇️ DEMOTED by master-plan AMENDMENT II — no compute-frontier study

An earlier version of this section preregistered `b_max ∈ {6, 12, 24}` as a
**frozen compute-frontier study**. **AMENDMENT II supersedes that.** Inference
cost is **logged prospectively** (§12.1) and reported; it is **not** a pillar of
the paper, and no frontier study is preregistered or required.

The claim-bearing runs use **max H24 with native anytime STOP**, per AMENDMENT
II §3. Nothing else about the protocol changes.

**If** inference cost later becomes an important result, a small prespecified
frontier may be added — and if it is, the method note in §12.3 binds.

### 12.3 Method note (binding IF a frontier is ever run)

`h_φ(x, z, b)` **conditions on remaining budget**. An H24 trajectory truncated
at step 6 is therefore **not** the same object as a controller actually started
with `b = 6`:

* the truncated run was steered at every step by `h_φ(·, ·, b)` for `b` counting
  down from **24** — a controller that believed it had budget to spare and could
  afford an excursion away from the region;
* a genuine H6 run is steered by `h_φ(·, ·, b)` counting down from **6** — a
  controller under pressure, which is a **different policy**, not the same
  policy observed earlier.

Truncation would report the cheap operating point using the expensive
controller's decisions and would systematically misstate the frontier, most
likely flattering H6. **Any frontier must therefore be separate controller
runs**, never reconstructed from H24 prefixes.

---

## 13 · SMC escalation completion — FROZEN BEFORE ANY SMC PERFORMANCE IS OBSERVED

The 64-source development panel triggered the preregistered twisted-SMC
escalation: region-`h_φ` showed useful navigation (coverage 15.6 % → 28.1 %,
13 sources new vs unguided, first hit at step 3.4 vs 10.0) while capped
rejection failed mechanically (**94.3 % cap-hit rate**, mean 2.4 edits of 24,
34.2 % of trajectories never committing a single edit).

The preregistration fixed the SMC **construction** — molecular-state particles,
learned-value twist, systematic resampling, ESS `< N/2` trigger — but did **not**
fix the particle count or the mapping from an SMC population to the Jin
20-candidate output contract. **Those two quantities are frozen here, before any
SMC performance exists.** The 64 triggered the transition and gave **no**
evidence about either.

### Particle count

> **`N = 32`**, inherited as the **single pre-existing SMC operating point
> already instantiated in the repository** before this gate fired. **No
> particle-count sweep is permitted.** It was **not** selected using SMC
> performance on the QED development panel.

### Candidate budget — the load-bearing rule

> Each source receives **exactly 20 independent SMC runs**. Each run produces
> **exactly one** returned molecule.
>
> ```
> SMC^(j)(x₀; N=32, H=24)  →  1 molecule,   j = 1 … 20
> 20 independent SMC runs   →  20 returned candidates
> ```
>
> The 32 internal particles are **inference state**, exactly like the internal
> steps of any other generative algorithm. They are **not** 32 returned-candidate
> opportunities.

**What one run returns:** one molecule sampled from the **normalized terminal
particle measure**, `J ~ Categorical(w₁/Σw, …, w₃₂/Σw)`, returning `x_J`.
Goal-region hits retain the already-defined absorbing/STOP semantics at the
particle level.

### Explicitly barred

* ❌ best-of-32 / highest-QED particle
* ❌ return-all-particles
* ❌ first-successful-particle
* ❌ top-20 from one population
* ❌ the old `budget_per_lead = 1000` semantics — **disallowed for GrIDDD/Jin**

Any of these would convert internal particle population into extra candidate
budget and break the external contract.

### ⚠️ AUDIT FINDING — what must be preserved vs replaced

`scripts/griddd_value_guided_smc_controller.py` already implements the SMC
correctly and **must not be rewritten**:

| component | status |
|---|---|
| `ess = 1.0 / np.sum(weights**2)`; `if ess < n_particles / 2` | ✅ **matches the frozen spec — preserve** |
| `_systematic_resample()` (low-variance) | ✅ **preserve** |
| FK potential / value twist plumbing | ✅ **preserve**, rebind the twist to frozen `h_φ` (V1; the prior run was V0, `value_twist: None`) |
| **output rule: tracks `best_qed`, `best_state`, returns `best_feasible_qed`** | ❌ **this is best-of-population — REPLACE** per the rule above |
| `budget` as distinct oracle calls (the 1000-per-lead path) | ❌ **REPLACE** — 20 independent runs, one molecule each |

This is the three-layer architecture (§2 of the governing plan) doing its job:
**same molecular dynamics, reusable SMC primitive, new task policy** matching the
Jin contract. Separate the inference machinery from the old task-specific
wrapper; preserve the former, replace only the latter.

### Everything else is unchanged

Frozen `R_θ`, frozen `h_φ`, `H = 24` maximum, region `(0.90, 0.40)`, hard
support semantics, STOP/absorption, weighting rule, ESS threshold, systematic
resampling.

**No SMC hyperparameter is selected using the 64-source results.** After
mechanical qualification the SMC implementation is frozen and evaluated on the
**fresh 128-source validation panel**.

### Qualification is mechanical, NOT parity with rejection

SMC is *supposed* to produce different trajectories from rejection, so demanding
trajectory parity would be incoherent. Qualification establishes that the frozen
mathematics is implemented correctly: particle weights and potentials,
systematic resampling, ESS trigger, STOP semantics, region, budget accounting,
reproducible seeds, deterministic replay where expected, kill/resume
losslessness, and exact checks wherever a small enumerable case allows one.
