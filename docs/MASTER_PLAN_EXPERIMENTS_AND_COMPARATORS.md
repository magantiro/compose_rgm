# COMPOSE — master plan: every experiment, every comparator, what each is compared to

**Canonical and consolidating.** This is the single place that answers, for each
locked experiment: *what question does it ask, what is it compared against, and
why that comparator rather than another.*

It does not reopen anything. It supersedes no ruling. Where an older document
said something now overtaken — `docs/EXPERIMENT_PLAN.md`'s "must-run" external
list is the main case — this records **what changed and why**, rather than
quietly dropping it.

Companion documents, all still authoritative in their own scope:

| doc | holds |
|---|---|
| `EXPERIMENT_PLAN.md` | the canonical plan: claims, phases, statistics, gates |
| `SCOPE_LOCK_AND_KERNEL_COST.md` | the six-piece lock; the kernel-cost blocker |
| `COMPARATOR_ROLES_CANONICAL.md` | the four roles; framework-first selection |
| `AMENDMENT_PUBLISHED_NUMBER_FIRST.md` | the four tiers |
| `PARETO_COMPARATOR_MATRIX_REQUIREMENT.md` | the per-block map; the Pareto gate |
| `BENCHMARK_RULINGS.md` | DDSBM tier 1, DRD2 demoted, T3 stopped, InVirtuoGen restricted |
| `workstreams/EXTERNAL_BASELINE_INDEX.md` | where every primary-source audit lives |

---

## Part 0 — How a comparator gets chosen

Two orthogonal questions, asked in this order.

### First: what is this comparator *for*? (role)

| role | what it tests |
|---|---|
| `FRAMEWORK_NEIGHBOR` | the closest alternative generative abstraction. **Primary external evidence for methodological novelty.** |
| `MATCHED_CAUSAL_CONTROL` | an internal counterfactual changing ONE mechanism, holding executor, state, objective and budget fixed. **Primary evidence for why each component matters.** |
| `TASK_COMPETENCE` | strong native methods showing COMPOSE performs credibly on ordinary molecular design. **Does not define the novelty claim.** |
| `CONCEPTUAL_LINEAGE_ONLY` | scientifically close, but a faithful common task would require substantial adaptation. |

A method may hold **different roles in different experiments**.

### Second: do we run it, or cite it? (tier)

| tier | condition | action |
|---|---|---|
| **1 — exact alignment** | task, oracle, budget, metric, normalization, validity/canonicalization, seed aggregation and applicability all align | **run COMPOSE only**; cite the authors' numbers, labelled *reported* |
| **2 — partial** | some protocol details differ | contextual literature numbers, **never** head-to-head statistics |
| **3 — rerun** | our custom task, and the method natively supports it through a thin adapter | rerun **the smallest necessary set** |
| **4 — do not reconstruct** | code broken, checkpoints absent, license restrictive, or adaptation requires method invention | cite and discuss |

> **Published-number-first when there is exact protocol alignment; official-native
> rerun when a load-bearing comparison requires matched conditions unavailable in
> published results; otherwise contextual only.**
>
> **Do not reconstruct a method solely to increase baseline count.**

### The two standing prohibitions

**No manufactured ports.** Building someone's method ourselves to add a table row
invents the parts they never released, and we would be benchmarking our
reconstruction.

**No mixing incompatible published protocols.** The standing example: HN-GFN's
own paper and InversionGNN attribute **70,000 vs 1,000** oracle calls to the same
method, under incompatible oracle representations (1024-bit ECFP4 against private
forests vs TDC's 2048-bit). Quoting either beside ours would have been wrong. The
correct finding was that *the comparison does not exist yet.*

---

## Part I — The six locked experiments

One experiment = one question, one primary contrast, one stop rule. A negative
result closes a branch; a positive one is banked and we move on.

---

### 1 · Executable-valid dynamics and edit expressivity

**Question.** Does the process produce healthy, executable, *every-state-valid*
trajectories, with genuine variable-size restructuring rather than a decorated
add-only walk?

| | |
|---|---|
| **Primary contrast** | construction + operator-use characterization |
| **External** | validity numbers only where directly comparable |
| **Comparator rationale** | there is nothing to *beat* here — the claim is that the state is valid **at every step**, which unconditional generators do not report because their intermediates are not molecules |
| **Reported from** | the DDSBM, Pareto and pathwise panels; **no standalone run** |
| **State** | ✅ satisfied by design; harvested from other panels |

This is the piece a reviewer checks and moves past. It is deliberately not given
its own experiment.

---

### 2 · DDSBM conventional endpoint competence

**Question.** Can frozen COMPOSE solve an established external task set by
someone else, without any objective-specific retraining?

| | |
|---|---|
| **Task** | DDSBM's ZINC250k logP 2→4, their exact held-out split, all **5,984** test sources |
| **Objective** | transport adapter `τ(x₀) = logP(x₀) + 2`, `u(y;x₀) = −|logP(y) − τ(x₀)|` |
| **Controller** | greedy closed-loop, horizon 6, frozen `R_θ`, no objective-specific update |
| **Comparator** | **DDSBM** — `FRAMEWORK_NEIGHBOR`, **tier 1** |
| **We run** | COMPOSE alone. DDSBM's published table is cited *as reported*. |
| **Metrics** | benchmark-native only: logP `W₁`, QED MAD, SA MAD, validity — plus trajectory-wide validity and edit expressivity, which **no row in their table can report** |
| **Stop rule** | none; this is a competence demonstration, not a superiority claim |
| **State** | 🟢 protocol frozen, gate 5,984/5,984 passed, pilot clean — **full run LAUNCHED** |

**Why DDSBM and not GraphXForm.** GraphXForm asks *is COMPOSE competitive with a
powerful sequential graph model?* DDSBM asks *why COMPOSE's executable graph CTMC
rather than another graph CTMC/bridge?* — the second is the question near the
novelty. GraphXForm is separately **capability-disqualified: it cannot delete.**

**The objective is a transport adapter, not a point target.** DDSBM moves a
marginal at logP ~ N(2, 0.5) to one at N(4, 0.5) and scores `W₁` between
*marginals*. A point objective at 4 would collapse the generated spread toward
the mode and inflate `W₁` against a target whose sd is 0.5. Verified on the
frozen test split **before** launch: source mean 2.015 sd 0.499, τ mean 4.015
sd 0.499.

**FCD and NSPDK are deliberately omitted**, and that selection was frozen before
any COMPOSE result existed — which is what stops it being cherry-picking. They
ask whether a bag of molecules resembles a dataset: DDSBM's question as a
*distribution* model, not ours. This is therefore **not a reproduction of DDSBM
Table 1**, and is labelled as such in the payload itself.

**Barred:** the CSV's randomly paired `PRB-SMI`. That coupling exists for DDSBM's
training; the task is transport between marginals. Only `REF-SMI` is read.

#### The cost decision, measured rather than estimated

| | earlier estimate | **pilot-measured (16 sources)** |
|---|---|---|
| per-source | — | median **60.0 s**, max 219.0 s |
| kernel calls | 6 | 6 (full horizon, **0 dead ends, 16/16 OK**) |
| candidates scored | — | median 2,642, max 3,490 |
| core-hours for 5,984 | ~50 | **120.2** |
| **cost** | ~$2.35 | **$5.65** |
| wall at 80 containers | — | 1.50 h |

The estimate was wrong by 2.4×; the pilot existed precisely to find that out,
and it did, for about fourteen cents. That exceeded the standing $2 Modal cap,
so it was put to a human decision rather than absorbed silently.

> **Decision, 2026-08-14: cap raised, full 5,984 authorized and launched.**
> App `ap-PRq38dAVAxPbshXxq57Efd`, driver `fc-01KZZGNABNX6KMSYTYN4MZQ2EY`,
> 81 tasks, detached. CPU only — `cpu=(1.0, 1.0)`, no GPU anywhere in this run.

The two rejected alternatives are recorded because the reason matters: a
preregistered random subsample (~1,500 sources, ~$1.42) would have preserved the
tier-1 claim **only if frozen before launch** — a subsample chosen after seeing
results would not be. Taking the full split removes that hazard entirely, which
is worth more than the $4 saved.

---

### 3 · Exact-target control and mid-trajectory retargeting

**Question.** Does finite-horizon reasoning improve hard target attainment, and
is a realized intermediate state still useful once the objective changes?

| | |
|---|---|
| **Primary contrast** | greedy · verified · similarity controls · `R_θ`-top1; then continue-from-`x₃` vs restart-from-`x₀` vs clairvoyant |
| **External** | **none — and this is the right answer.** No published method instantiates "the goal changed at step 3." |
| **Comparator rationale** | `x₃ →ᴮ ⋯` vs `x₀ →ᴮ ⋯` directly answers whether prior molecular progress has value. Inserting DDSBM or GraphXForm there would isolate **nothing** |
| **State** | ✅ **banked, sealed and closed** |

This branch is finished. It is not to be reopened, extended, or re-litigated.

---

### 4 · Pareto — target-free multi-objective control

The longest chain, and the one carrying the conceptual claim.

```
P3/P4 banked  →  P0c resolves  →  exactly ONE earned branch  →  comparator matrix FROZEN  →  fresh final panel
```

#### 4a · P3/P4 — does closed-loop control itself matter? ✅ banked

The primary conceptual comparison. **Not** a fairness footnote.

| | |
|---|---|
| **Contrast** | closed-loop COMPOSE vs **fairly funded** generate-and-rank, at **matched kernel calls** |
| **Role** | `MATCHED_CAUSAL_CONTROL` |
| **Result** | **P3 +0.6678 [+0.5169, +0.8094], 12W/0L · P4 +0.5668 [+0.4763, +0.6390], 12W/0L** on hypervolume |
| **Integrity check** | the repair **lifted** the baseline 0.308 → 0.371 — exactly what a *broken* matcher would have failed to do |
| **Budgets** | `K_G = 333`, `K_V = 15`, both derived from the frozen ≤10k rule |

**Reported against us, unrescued:** nondominated set size favours the baseline
(**P3 −2.250, P4 −2.417, 0W/11L**). Resource axes disagree by ~10³ in opposite
directions — COMPOSE uses **44× fewer trajectories** and **2,133× more oracle
requests** at matched kernel calls. All four costs stay separate; the oracle
denominator is **unique canonical molecules actually evaluated**, never raw
requests.

#### 4b · P0c — does a preference continuum exist at all? 🟡 11/12

A feasibility question that must be answered before any front-construction
branch is chosen. Exact preference partition via Chebyshev scalarization,
`s_y(w) = max(w·g₁, (1−w)·g₂) + ρ(g₁+g₂)`; the augmentation constant in `w`, so
the front is the lower envelope of 2N line segments with breakpoints
`w* = g₂/(g₁+g₂)`. Bisected against the frozen `_argmin_stable`.

**Early signal:** the endpoint pool suggests the continuum **collapses** —
1,025 candidates → **2 regions**. A frozen **three-way rule** decides what
happens next; I remain blinded to outcomes until source **004** lands.

#### 4c · The external comparator matrix — frozen *before* the panel launches

**DDSBM does not fill this slot.** It covers conventional source-conditioned
endpoint transformation, not target-free front construction.

| method | role / status | verdict |
|---|---|---|
| **HN-GFN** | `TASK_COMPETENCE` | GPU-only, no released checkpoint, empty `BlockMoleculeDataExtended()` start, surrogate-budget asymmetry |
| **InversionGNN** | tier 4 | not runnable as shipped, no LICENSE, 70× budget disagreement |
| **OP-GFN** | excluded | CC BY-NC-**ND** |
| **MOG-DFM** | `CONCEPTUAL_LINEAGE_ONLY` | same-lab lineage |
| **AReUReDi** | candidate | the only same-lab numerical candidate; gated on native/thin/author-validated |
| **pCoMole** | ⚠️ `UNVERIFIED` | OpenReview not machine-reachable, not on arXiv — **needs a human login** |
| **ParetoFlow** | ⚠️ **never audited** | new work for this gate |
| **A-GPS** | ⚠️ **never audited** | new work for this gate |

Six carry prior findings to be **reused, not re-derived**. Two are new work.

**Target: 1–2 genuine external numerical comparators, and only on exact
alignment. One or zero is an acceptable outcome** — the block already carries a
fair matched causal baseline in repaired P3/P4, which tests the claim more
directly than any external row.

---

### 5 · SA hard support

**Question.** Does enforcing admissibility *during* generation beat
generate-then-filter?

| | |
|---|---|
| **Primary contrast** | exact support vs **the identical unconstrained controller** + post-hoc filter |
| **External** | CDD · PRODIGY · ConStruct as `CONCEPTUAL_LINEAGE_ONLY` |
| **Comparator rationale** | CDD's useful object is the **published constraint definition**, which transfers verbatim; its implementation does not (59-byte placeholder README). PRODIGY has no LICENSE |
| **Precedent** | ConStruct's own §G.1 planarity negative mirrors our scaffold stop — cited as lineage, not competition |
| **Stop rule** | runs **only if** the frozen census passes |
| **State** | 🔴 designed; **blocked on the Gate-0 mount-path defect** |

---

### 6 · cLogP pathwise confirmation

**Question.** Does constraining the *route* matter beyond endpoint
admissibility?

| | |
|---|---|
| **Primary contrast** | endpoint-only vs pathwise-constrained, **on the same process** — close to an ideal causal experiment |
| **External** | none, and none would help |
| **Earned by** | Stage A2 (corridor excursions prevalent and spread, 5/5) and Stage B (**two-thirds of sources hide a forbidden intermediate**) |
| **Design** | held-out, **n = 48**, source-clustered paired bootstrap, source as the independent unit |
| **Tests** | **P1** prevalence — reuses A2's frozen `V5a > 1/3` verbatim, strengthened to a CI lower bound (declared as a change) · **P2** noninferiority |
| **Inference** | intersection–union (P1 ∧ P2), no multiplicity correction, joint power |
| **δ** | **0.25 as a tolerance on the frozen held-in DRD2 scale**, with **δ = 0.20 mandatory sensitivity** |
| **State** | 🔴 frozen; blocked on the checkpoint regression |

**Two corrections are baked into this design and must not be quietly undone.**

First, **δ is a tolerance, not a potency threshold.** The medicinal-chemistry
two-fold framing was rejected: it imports an experimental convention onto a
classifier's odds, where it has no such meaning.

Second, **"little or no potency cost" was withdrawn.** A CI spanning zero is not
evidence of equivalence. That is the entire reason this confirmation exists.

The earlier n = 24 was a **~50%-power** calculation — it had not been done
against the final estimand. The bootstrap gave 0.604 power at true retention
0.95, and n = 24 was withdrawn.

**The blocker is real and self-inflicted.** The Stage B checkpoint was
**write-only**: Modal preempted at t = 800 with 317/368 kernel calls done and 76
minutes were lost. Real resume now exists with a
`test_resume_is_LOSSLESS_not_merely_safe` regression — and the bar was then
raised to a **forced real-kill and torn-write** test before this launches.

---

### Queued — the navigation capstone

**Not part of the six.** After the final Pareto controller produces a usable
state/front map: can a newly requested tradeoff be reached better by **selecting
and continuing from an already-realized mapped state** than by restarting from
`x₀`?

**Do not run before the map exists.** The bar is higher than banked retargeting,
which already showed an intermediate state retains value after a goal change.
What is new is **choosing which** mapped state to reuse.

---

## Part II — The full comparator ledger

Every external method ever considered, with its verdict and where the primary
audit lives. `configs/comparator_registry_v3.json` — 20 comparators, 6 with
full qualification records.

### Numerical — a real head-to-head exists or may

| method | role | tier | status |
|---|---|---|---|
| **DDSBM** | `FRAMEWORK_NEIGHBOR` | **1** on ZINC logP 2→4; 4 on DRD2 | published numbers cited; COMPOSE runs alone |
| **AReUReDi** | Pareto candidate | TBD | gated on native/thin/author-validated |

### Task competence — credible, but does not define novelty

| method | status |
|---|---|
| **HN-GFN** | GPU-only, no checkpoint, empty-molecule start, surrogate asymmetry |
| **InVirtuoGen** | selected editor — ⚠️ **its output may never become `R_θ` training data** |
| **GraphXForm** | ⛔ **capability-disqualified — cannot delete** |
| **MolEditRL** | task competence |
| **GraphGA · REINVENT** | qualification records, 2026-08-12; PMO-official implementations |
| **MARS** | designed, not built; gated on an oracle preflight |

### Conceptual lineage — cited, never ported

| method | why not numerical |
|---|---|
| **Edit Flows** | edit-based CTMC over **sequences**; a molecular-graph port would require inventing the chemical-support machinery COMPOSE contributes |
| **Expanding Flow Maps** | variable-size *de novo*, not source-conditioned legal rewriting |
| **GrIDDD** | its **own ablation** shows insert/delete inert on DRD2 |
| **MOG-DFM · pCoMole · PepTune** | same-lab lineage |
| **CDD · PRODIGY · ConStruct** | hard-constraint lineage; numerical only under a native common protocol |
| **Morph** | 3D geometric; a 2D rewrite adaptation is not automatically fair |

### Excluded outright

| method | reason |
|---|---|
| **OP-GFN** | CC BY-NC-**ND** |
| JTVAE · GDSS · DiGress · GruM · DISCO · Cometh · DeFoG · CoCoGraph | the old unconditional-generator matrix — belonged to a GuacaMol-centered paper that no longer exists |

### Unaudited — open work

| method | need |
|---|---|
| **ParetoFlow** | full audit against the ten schema fields |
| **A-GPS** | full audit against the ten schema fields |
| **pCoMole** | ⚠️ **a human OpenReview login.** All ten fields `UNVERIFIED` |

### What changed from `EXPERIMENT_PLAN.md`'s "must-run" list

That list predates the editing-competence and benchmark audits. Recorded, not
silently dropped:

| then | now | why |
|---|---|---|
| MARS — must-run | designed, not built | oracle preflight gate unmet; not load-bearing for any locked claim |
| DDSBM — must-run, native rerun | **tier 1, published numbers, COMPOSE alone** | exact benchmark alignment makes a rerun pure cost |
| GraphXForm — must-run | **capability-disqualified** | cannot delete; cannot instantiate the variable-size claim |
| GraphGA / REINVENT — must-run sanity checks | qualified, unscheduled | PMO track is not in the six-piece lock |

---

## Part III — The comparison map at a glance

| block | question | primary comparator | external numerical? |
|---|---|---|---|
| reference law | did `R_θ` learn useful dynamics? | uniform canonical · empirical-family · learned-family/uniform-ID · empirical-family/learned-ID | **none needed** |
| reference dynamics | what behaviour does learned `R_θ` induce? | uniform + empirical-family processes | **none needed** |
| conventional editing | can frozen COMPOSE solve an external task? | **DDSBM** | **yes — published values** |
| exact-target | does finite-horizon reasoning help? | greedy · verified · similarity · `R_θ`-top1 | **no natural external task exists** |
| retargeting | is the realized state useful after a goal change? | continue-from-`x₃` vs restart-from-`x₀` vs clairvoyant | **matched causal is better** |
| hard support | does enforcing beat filtering? | identical unconstrained controller + post-hoc filter | lineage only |
| pathwise | does the *route* matter? | endpoint-only vs pathwise on the same process | **matched causal** |
| validity / expressivity | variable-size restructuring, every state valid? | construction + operator-use characterization | where directly comparable |
| Pareto | does closed-loop control itself matter? | **matched generate-and-rank at equal kernel calls** | 1–2, only on exact alignment |
| map reuse *(queued)* | can prior search effort be reused? | best mapped state vs restart from `x₀` | **matched causal** |

**This is not weak baseline coverage.** For several blocks an unrelated external
model would be a *worse* control.

### The internal arms — the most important comparators we have

Every full-scale control task compares the **same frozen `R_θ`** under: unguided
· hard-mask-only · greedy one-step · local Boltzmann tilt · beam/best-first ·
untwisted SMC · immediate-potential SMC · learned bridge controller · Monte
Carlo value/MPC (small subset).

> Does future-aware bridge control outperform ordinary generation plus
> ranking/search under the same legal kernel, reference model, oracle budget and
> source molecules?

Without that, a reviewer can attribute gains to the rewrite kernel or the oracle
rather than the bridge.

---

## Part IV — Open items, and who owns them

| item | owner | blocking |
|---|---|---|
| **pCoMole** OpenReview login | 🧑 **human** | the Pareto matrix cannot be frozen with an `UNVERIFIED` row |
| **ParetoFlow / A-GPS** audits | lane | same |
| ~~**DDSBM cost** — $5.65 vs the $2 cap~~ | ✅ resolved 2026-08-14 | cap raised, full run launched |
| **P0c source 004** | running | the whole Pareto chain |
| **cLogP checkpoint** — forced real-kill + torn-write | agent | piece 6 |
| **SA census** — Gate-0 mount-path defect | agent | piece 5 |
| **Bounded kernel profile** | agent | ~80% of remaining cost is unmeasured |
| **Fresh Pareto panel `n`** | open | powered from Stage B variance, not from the 14/24 result |

---

## Part V — Barred

**Scope.** New objective pairs · new scalarization families · new constraint
classes · architecture changes · baseline zoos.

**Comparators.** Manufactured ports · mixing incompatible published protocols ·
reconstructing a method to increase baseline count · bending a published method
into an unnatural variant to manufacture a leaderboard.

**Rescues.** No smaller-core rescue after the Bemis–Murcko feasibility stop. No
soft-guidance arm. No unpreregistered rescue of any negative result. The T3
1,000-evaluation budget is not to be rescued.

**Claims.** Not "essentially free." Not "little or no potency cost." A CI
spanning zero is not evidence of equivalence.

**Provenance.** No InVirtuoGen-generated output may ever become `R_θ` training
data.
