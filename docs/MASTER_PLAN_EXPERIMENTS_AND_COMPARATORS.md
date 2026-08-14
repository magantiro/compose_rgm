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

### 2 · ~~DDSBM~~ → **GrIDDD** conventional editing competence

> ⬇️ **AMENDED.** DDSBM ran to completion (5,984/5,984) and is **out of the
> manuscript** — its native question is distribution-to-distribution transport,
> not source-conditioned editing, and with Edit Flows and GrIDDD seated it
> fills no remaining role. The slot is now **GrIDDD on the Jin/ZINC QED task**.
> The DDSBM result is preserved unrescued; the removal is recorded with the
> fact that it was proposed *after* an unflattering FCD, so a later reader can
> judge that themselves. See `AMENDMENT_EDITFLOWS_GRIDDD_HEADLINE.md`.

#### The completed DDSBM stress test, kept for the record

**Question.** Can frozen COMPOSE solve an established external task set by
someone else, without any objective-specific retraining?

| | |
|---|---|
| **Task** | DDSBM's ZINC250k logP 2→4, their exact held-out split, all **5,984** test sources |
| **Objective** | transport adapter `τ(x₀) = logP(x₀) + 2`, then `u(y;x₀) = −\|logP(y) − τ(x₀)\|` — i.e. **negative absolute deviation** from the shifted source, written out in full below |
| **Controller** | greedy closed-loop, horizon 6, frozen `R_θ`, no objective-specific update |
| **Comparator** | **DDSBM** — `FRAMEWORK_NEIGHBOR`, **tier 1** |
| **We run** | COMPOSE alone. DDSBM's published table is cited *as reported*. |
| **Metrics** | benchmark-native only: logP `W₁`, QED MAD, SA MAD, validity — plus trajectory-wide validity and edit expressivity, which **no row in their table can report** |
| **Stop rule** | none; this is a competence demonstration, not a superiority claim |
| **State** | ⬇️ **EXECUTED 5,984/5,984, then REMOVED FROM THE MANUSCRIPT.** Developmental transport stress test; result preserved. See `AMENDMENT_EDITFLOWS_GRIDDD_HEADLINE.md` |

**The frozen objective, in full** — this is the authoritative statement; the
table cell above abbreviates it:

```
tau(x0)  = logP(x0) + 2
u(y; x0) = -| logP(y) - tau(x0) |
```

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

#### The first full launch failed 100%. Why, and what it exposed

**Every one of the 5,984 sources died** with

```
ModuleNotFoundError: No module named 'compose_v4.experiments.pareto_control'
```

**The pilot had passed 16/16 on the same code path minutes earlier.** The
difference was never the code — it was the *launch directory*.

| | pilot | full run |
|---|---|---|
| launched from | the `/private/tmp` scratchpad worktree | `/Users/rmaganti/compose_v2_work` |
| `pareto_control.py` in mounted `src/` | **present** | **absent** |
| result | 16/16 OK | 0/5,984 |

The module is committed on `codex/compose-pareto-control` and was present in the
worktree; it was simply not on the branch the full run mounted. Modal mounts the
launch directory's `src/`, so the workers imported a tree that had never
contained the file.

**Three lessons, all recorded rather than smoothed over:**

1. **A green pilot proves nothing unless it ran from the same tree the full run
   will mount.** This pilot's clean 16/16 actively created false confidence.
2. **A load-bearing source file for a launched experiment existed only in
   `/private/tmp`.** Nothing was lost — it was on origin via the lane branch —
   but the scratchpad was on the critical path for a real experiment.
3. **`retries=3` turned one defect into thousands of paid container inits**
   (~160 s each) before it was caught.

**Fixed in `7020dc3`:** the module is committed to this branch unmodified
(39,193 bytes, purely additive). All seven `compose_v4` imports the app needs
were checked to resolve, and then the chain was **actually imported** rather than
statically checked — `_argmin_stable` returns index 2 for values `[3,1,1,2]` over
keys `[d,b,a,c]`, i.e. the lexicographically smallest key among tied minima, so
the frozen deterministic tie-break is intact.

**Standing rule going forward: launch from the committed working tree, and
verify imports resolve in the tree that will actually be mounted.**

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
| **Arms as run** | `greedy_pref` and `verified_pref` on the **full fiber**, against fairly funded `gen_rank@greedy` / `gen_rank@verified` |
| **How matched** | **kernel calls matched by construction** by the metered matcher — not by any shortlist budget |

> ⚠️ **`K_G = 333` and `K_V = 15` are NOT part of this banked result.** They are
> the budgeted shortlist sizes for arms **B** and **C** of the *fresh scalable
> panel* — the ≤10k-query operating points defined per fresh source in
> `PARETO_DEVELOPMENT_REVISED_PREREGISTRATION.md`, whose own estimand is
> `HV_verified@15 − HV_greedy@333`, a different contrast entirely. P3/P4 ran the
> **full fiber**. Attaching those budgets here would retroactively rewrite what
> was actually executed. See §4d.

**Reported against us, unrescued:** nondominated set size favours the baseline
(**P3 −2.250, P4 −2.417, 0W/11L**). Resource axes disagree by ~10³ in opposite
directions — COMPOSE uses **44× fewer trajectories** and **2,133× more oracle
requests** at matched kernel calls. All four costs stay separate; the oracle
denominator is **unique canonical molecules actually evaluated**, never raw
requests.

#### 4b · P0c — does a preference continuum exist at all? ✅ RESOLVED — it does NOT collapse

A feasibility question that must be answered before any front-construction
branch is chosen. Exact preference partition via Chebyshev scalarization,
`s_y(w) = max(w·g₁, (1−w)·g₂) + ρ(g₁+g₂)`; the augmentation constant in `w`, so
the front is the lower envelope of 2N line segments with breakpoints
`w* = g₂/(g₁+g₂)`. Bisected against the frozen `_argmin_stable`.

**Early signal:** the endpoint pool suggests the continuum **collapses** —
1,025 candidates → **2 regions**. A frozen **three-way rule** decides what
happens next.

**All 12 shards landed 2026-08-14** and the rule was applied once. **See
`docs/P0C_VERDICT.md`.**

> ### ✅ VERDICT: OUTCOME 2 — the continuum is rich; aspiration is CLOSED.
>
> **Amended after the cost measurement:** the exact sweep is the
> **`FULL_INFORMATION_PARETO_CEILING`**, *not* the deployable controller — it
> costs a median **75,910** unique oracle evaluations/source against the frozen
> **≤10,000** rule (over budget on 11/12). One bounded compression step remains:
> **`BUDGETED_PREFERENCE_SWEEP`**, `K_sweep = 41`, frozen `R_θ` ranking only.
> See `docs/BUDGETED_PREFERENCE_SWEEP_PREREGISTRATION.md`.

**The result reverses the early signal.** The endpoint pool predicted collapse;
the real successor fibers show the opposite. One-step sparsity does not imply
trajectory-level collapse over H=6 — here it actively misleads, which is exactly
why the probe was run on the real object.

| | continuum | at the frozen 5 weights | gain |
|---|---|---|---|
| distinct endpoints | median **69.5** | median **3.5** | **+66.0**, positive 12/12 |
| nondominated size | median 11.5 | median 3 | **+8.0**, positive 12/12 |
| hypervolume | — | — | **+64.2 %**, positive 12/12 |

11/12 completed inside the 240-expansion guard (median 136). Source 004 hit it
at 241 and is classified under outcome 3 from its partial tree — ND +16, HV
+5.3 %, both lower bounds.

**This settles the P3/P4 asymmetry as hypothesis (a):** the nondominated-
cardinality loss (P3 −2.250, P4 −2.417) was a **sampling artifact of the
five-weight grid**, not a structural weakness in closed-loop control. A +8
median ND gain swamps a ~2.3 deficit. **No new algorithm is needed.**

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

> ### ⚠️ AMENDED — the ceiling applies to RERUNS, not to published rows
>
> An earlier version capped this at *"1–2 genuine external numerical
> comparators."* **That was too restrictive and is retracted.** The correct rule
> separates two very different costs:
>
> **Include EVERY method for which an exact published-number comparison is
> possible.** If four papers report on the same task, oracle, budget, HV
> normalization, canonicalization and seed aggregation, then **one COMPOSE run
> buys four legitimate rows** and we retrain nothing. That is
> published-number-first at its best.
>
> **Cap only what we rerun ourselves: at most 1–2 official-native reruns**, and
> only when a load-bearing comparator lacks compatible published numbers *and*
> supports the common task through a thin adapter. **No manufactured ports.**
>
> Standards do not drop to fill rows: if only one method survives the audit,
> only one gets a head-to-head row. Partial alignment stays **contextual**.

### The audit's real job: find benchmark INTERSECTIONS

The instruction to the audit is **not** "find a baseline for COMPOSE." It is
**find clusters of papers already sharing a protocol.** For each Pareto method
record:

molecular dataset/task · exact objectives · exact oracle implementation and
version · source-conditioned vs de novo · oracle-evaluation count *and what
counts as one* · number of generated candidates · hypervolume reference point
and normalization · ND/spread metrics · seed aggregation ·
validity/canonicalization · the published numerical values.

Then look for where those coincide. A single shared benchmark carrying several
prior methods is worth far more than several separately-argued near-matches.

**We do not yet know that these methods share a task** — that is precisely what
is unfinished. Do not assume it.

The block already carries a fair matched causal baseline in repaired P3/P4,
which tests the COMPOSE-specific claim more directly than any external row. That
is a reason the *rerun* count stays near zero; it is **not** a licence to stop
auditing. Four rows are still open.

### External methods belong in the FINAL panel, not in every subexperiment

Each Pareto subexperiment isolates one mechanism, and an external model would
answer none of them:

| subexperiment | question | correct contrast | external methods? |
|---|---|---|---|
| **P3/P4** | does closed-loop state feedback matter? | COMPOSE vs fair generate-and-rank | **no** — they don't isolate feedback |
| **P0c** | was five-weight sampling hiding richness? | 5 weights vs complete sweep | **no** |
| **K41** | can `R_θ` expose that front under 10k? | full-information ceiling vs budgeted sweep | **no** |
| **fresh final panel** | is COMPOSE competitive? | COMPOSE final vs all exact-alignment published methods | **yes — this is where they go** |

#### 4c-bis · K41 budgeted sweep — **FAILED**, branch closed

> ### ❌ `R1 ∧ R2 ∧ B` → **FAIL.** The compression branch is closed. No second
> algorithm, no re-specification, no rescue. See `K41_BUDGETED_SWEEP_RESULT.md`.

| test | criterion | observed | |
|---|---|---|---|
| **B** budget | ≤10,000 on every source | max **6,888**, cap never binding, 12/12 | ✅ |
| **R1** quality | CI lower on `mean(r_HV)` > 0.90 | 1.425 | ⚠️ see below |
| **R2** breadth | CI lower on `mean(r_ND)` > 0.90 | **0.737** | ❌ |

**The budget mechanism itself worked.** `R_θ` shortlisting cut oracle spend
from a median **75,910 → 4,113** (~18×) with no violation anywhere, and
`cap_was_binding` was false on all 12 — so nothing is a DFS-order artifact.
That vindicates deriving `K_sweep` from the guard.

**R1 "passes" for the wrong reason.** `mean(r_HV) = 5.018`, exceeding 1 on 10
of 12 sources and reaching **40×** on source 006 — impossible for a genuine
retention ratio. The budgeted sweep is **not a sub-traversal** of the full
sweep: it scores a different subset, partitions differently, and descends a
different tree, sometimes expanding *more* states (source 006: **14 → 76**).
The "ceiling" is the full-information *partition*, **not** an upper bound on
attainable hypervolume.

**That does not rescue the verdict.** R2 fails on its own terms — genuinely
fewer nondominated points on 6/12 sources, as low as 0.515 — and
intersection–union fails regardless of R1.

**What stands:** P0c is untouched. What K41 does *not* establish is that the
richness can be exposed at a 10k budget with breadth preserved. The
preregistered fallback, written before the result, applies: the full sweep
demonstrates latent controller richness; the five-weight/budgeted controller
demonstrates what is achievable at contemporary budgets.

#### 4d · The scalable operating points — RECLASSIFIED after P0c

> ### ⚠️ `K_G = 333` / `K_V = 15` are now HISTORICAL, not a scheduled experiment
>
> These were derived for the **five-weight, 30-scoring-event** controller. P0c
> replaced that structure: the continuum tree expands a **median 136 states**,
> not 5 weights × 6 steps. **The scoring-event arithmetic they came from no
> longer describes the object**, so they cannot be reused mechanically for
> continuum control.
>
> **Do not launch a K333/K15 panel alongside the K41 work** merely because an
> older plan lists them. `K_sweep = 41` now answers the more important final
> scalability question. If K41 passes it becomes the **production controller**,
> and K333/K15 remain **recorded development designs** — preregistered honestly,
> superseded honestly. Whether they still support a *distinct* claim is a
> question to ask **after** K41 resolves, not before.

Recorded as originally preregistered. Both `K` values come from the **same
external 10,000-query-per-source rule** applied to scoring-event counts read from
the frozen implementation. Neither was chosen by looking at an outcome, and `K_V`
was fixed **before P3 existed**.

```
greedy    E_G = 5 prefs × 6 steps                     =  30 events
verified  E_V = 5 prefs × (6 + 8 rollouts × 15 steps) = 630 events
```

| arm | K | retains | predicted req | reduction |
|---|---:|---:|---:|---:|
| **B — budgeted `R_θ`-shortlisted greedy** | **333** | 56.4% of fiber | 9,990 | 1.8× |
| **C — budgeted `R_θ`-shortlisted verified** | **15** | 2.4% of fiber | 9,450 | 42.1× |

**The asymmetry is the point.** Greedy needs only a mild cut to reach the
contemporary budget regime; verified needs 42×. **Settled: full-fiber verified
does not run on the fresh panel** — it did its job in the 12-source smoke, and
paying ~22× again would re-establish a ceiling we no longer need. Arm C is
therefore measured against arm A and against arm B at matched budget, never
against a full-verified ceiling, which is why C carries no noninferiority
requirement.

**`K_G` came from a budget rule before outcomes existed. It is not to be
harshened because 333 later "looks easy."**

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

> **Each claim-bearing task uses the smallest matched-control set required to
> isolate its mechanism. The exact arms are frozen in that experiment's own
> section above, and nothing here adds to them.**

This replaces an earlier formulation that read as though *every* full-scale task
must carry all nine candidate arms — unguided, hard-mask-only, greedy one-step,
local Boltzmann tilt, beam/best-first, untwisted SMC, immediate-potential SMC,
learned bridge controller, Monte Carlo value/MPC. **That list is a menu of
available controls, not a required set**, and reading it as required would
authorize exactly the internal baseline zoo the scope lock bars.

Whichever arms a task does carry, they exist to answer one question:

> Does future-aware bridge control outperform ordinary generation plus
> ranking/search under the same legal kernel, reference model, oracle budget and
> source molecules?

Without a matched control of that shape, a reviewer can attribute gains to the
rewrite kernel or the oracle rather than the bridge. **One well-chosen matched
arm answers this; nine do not answer it nine times.**

---

## Part IV — Open items, and who owns them

| item | owner | blocking |
|---|---|---|
| **pCoMole** OpenReview login | 🧑 **human** | the Pareto matrix cannot be frozen with an `UNVERIFIED` row |
| **ParetoFlow / A-GPS** audits | lane | same |
| ~~**DDSBM cost** — $5.65 vs the $2 cap~~ | ✅ resolved 2026-08-14 | cap raised, full run launched |
| ~~**P0c source 004**~~ | ✅ landed 2026-08-14; 12/12 complete | unblocked — verdict not yet read |
| **cLogP checkpoint** — forced real-kill + torn-write | agent | piece 6 |
| **SA census** — Gate-0 mount-path defect | agent | piece 5 |
| **Bounded kernel profile** | agent | ~80% of remaining cost is unmeasured |
| **Fresh Pareto panel `n`** | open | **power AFTER the final Pareto algorithm and estimand freeze** — see below |

---

### How the fresh Pareto `n` gets set — and a conflation to avoid

**There is no Pareto "Stage B."** Stage A2 and Stage B belong to the **cLogP
pathwise** line only. An earlier draft of this document said the Pareto panel
would be "powered from Stage B variance"; that was a conflation of two separate
experimental lines and is **struck**.

The correct procedure:

1. **Freeze the final Pareto algorithm and estimand first** — `n` cannot be
   computed against an estimand that does not exist yet.
2. Draw the variance from the **sanctioned Pareto development sources** — the 12
   smoke sources supply the spread, and nothing else about them is used.
3. Power against **the actual final source-level paired bootstrap**, the same
   procedure the final analysis will report.

Step 3 is not a formality. **The earlier n = 24 was withdrawn precisely because
planning used a normal approximation while the analysis used a bootstrap** — the
two do not agree in general, the bootstrap of a ratio is skewed, and the real
power turned out to be 0.604 where 24 had implied adequacy. `scripts/
pareto_scalable_power.py` was rewritten to run the actual bootstrap for this
reason. **Do not re-introduce a closed-form shortcut.**

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

---
---

# ⭐ CURRENT AMENDMENT — reference THIS section, not the blocks above

**Everything above remains the historical record. Where it conflicts with this
section, this section governs.** Two things changed: policy B was refuted on
the development panel, and the vague custom MOEA block is replaced by a real
published five-objective molecular benchmark.

## A · The self-contained experiment / comparator table

| Block | Experiment / question | What we claim | Correct comparison | External numerical? | Status |
|---|---|---|---|---|---|
| **1A** | **Every-state executable validity** — are all realized states complete valid molecules? | support/execution property of COMPOSE | trajectory-wide validity, dead ends, operator usage | **No** — external *endpoint* validity is not the same object | harvest from all runs; no standalone benchmark |
| **1B/C** | **Trans-dimensional utility** — do size-changing edits matter? | insert/delete is empirically useful, not padded-slot bookkeeping | **full COMPOSE vs identical size-fixed COMPOSE** | context only — GrIDDD runs the analogous ablation | nested inside the GrIDDD QED benchmark |
| **2** | **Standard molecular editing** — can frozen COMPOSE solve a normal source-conditioned task? | ordinary editing competence | Jin QED task, 800 sources × 20 outputs | **GrIDDD**; JT-VAE/CG-VAE/GCPN as reported context | primary external editing benchmark |
| **3A** | **Finite-horizon control** — does future reachability beat local decisions? | remaining-budget reasoning matters | verified/lookahead vs greedy, same `R_θ`/kernel/source/goal | **No** | ✅ banked |
| **3B** | **Mid-trajectory retargeting** — after goal B appears, keep the realized `x₃`? | stateful recontrol from a realized molecule | continue-`x₃` vs restart-`x₀` vs clairvoyant | **No** | ✅ banked |
| **4A** | **Closed-loop Pareto control** — does purpose *during* generation beat generate-and-rank? | closed-loop feedback matters | COMPOSE closed-loop vs **matched** generate-and-rank | **No**, for the causal claim | ✅ banked (P3/P4) |
| **4B** | **Preference richness (P0c)** — does the controller produce a continuum? | one frozen process supports meaningfully different preferences | continuous sweep vs frozen 5-weight grid | **No** | ✅ positive |
| **4C** | **K41 compression** — can continuum richness be recovered cheaply? | efficient approximation of the full sweep | K41 vs its frozen retention gates | **No** | ❌ failed, closed, no rescue |
| **4D** | **Standard high-dimensional MOO competence** | COMPOSE produces a strong molecular Pareto set, not just mechanism | **MOLLEO Task 3** (5 objectives) | **YES** | ⬆️ **replaces the vague custom MOEA block** |
| **5** | **Hard-support control** — exact constraint vs encouraging/filtering | restricting feasible support itself buys something | post-hoc filter vs fixed soft guidance vs hard support | **No** | internal three-arm test |
| **6** | **Pathwise constraint** — does the whole route matter, not just the endpoint? | executable intermediates enable genuinely pathwise constraints | endpoint-only vs pathwise, same process | **No** | internal causal test |
| **Queued** | **Map reuse / navigation** — reuse explored states when preference changes? | search effort becomes reusable and stateful | continue from best saved state vs restart | no, unless a native same-task method appears | only after a usable map exists |

### The external structure, entire

```
framework level          →  Edit Flows + GrIDDD
ordinary source editing  →  GrIDDD QED (Jin ZINC-250k)
ordinary multiobjective  →  MOLLEO Task 3 (five objectives)
unique COMPOSE claims    →  matched internal controls
```

## B · The LOCAL ONE-STEP TILT is refuted on the development panel

> ⛔ **Never call this "COMPOSE."** It is the **myopic-control ablation** —
> COMPOSE's first trivial controller approximation, not COMPOSE's optimization
> method. The claim-bearing method is the finite-horizon learned controller.
> See `docs/PLAUSIBILITY_AND_PURPOSE.md`.

Read once, under the criterion frozen before any number existed.
64 disjoint sources × 5 replicates = **320 trajectories**, ~$0.18.

| | |
|---|---|
| **PRIMARY** `1[QED≥0.9 ∧ sim≥0.4]` | **0 / 320**, source-clustered CI [0.0000, 0.0000] |
| best-of-5 source success *(descriptive)* | 0 / 64 |
| Tanimoto ≥ 0.4 alone | **0.6719** ✅ |
| **QED ≥ 0.9 alone** | **0.0063** ❌ |
| terminal QED | median **0.7470** vs source median ~0.7576 |
| mechanics | 0 dead ends, 0 zero-denominator fallbacks, 31.4 % kernel work saved |

**Similarity is not the problem; optimization is.** `π ∝ R_θ(y|x)·QED(y)` does
not move QED at all — with QED clustered in 0.6–0.8 across a ~600-wide fiber,
the multiplicative factor barely reweights `R_θ`, the reference law dominates,
and the walk is effectively unguided. GrIDDD reports **45.1 %**.

**Nothing failed mechanically.** The policy simply does not optimize.

**Barred response:** `QED^α`, temperatures, shortlist sizes — any knob fitted to
this number. Permitted: a controller variant statable independently of it.

**What it tells us structurally:** a one-step tilt against a strong reference
law cannot move a bounded objective. DDSBM said the same thing from the other
side — it optimized hard and left the chemistry. **The finite-horizon question
set aside on cost grounds is now the actual scientific obstacle, not a nicety.**

### ➡️ The response: build the controller the theory already specifies

**Framing, canonical:** the controller is **part of COMPOSE, not an add-on**.
COMPOSE separates **plausibility** (`R_θ`, goal-independent) from **purpose**
(`h_φ`, inference-time). Evaluating bare `R_θ` on an optimization benchmark is
the *unnatural* evaluation — GrIDDD does not run an unconditional diffusion
model either; it denoises **conditioned on the property vector** with
classifier-free guidance. `docs/PLAUSIBILITY_AND_PURPOSE.md`.

**B was not a GPS.** It had the road network and the driving prior and no
destination-awareness. So 0/320 does not show COMPOSE cannot optimize — it
shows we were not using COMPOSE's control object.

The answer is **not another heuristic**. It is
`h_φ(b,x;x_src,z) ≈ Pr_{R_θ}(X_K ∈ B_z | X_{K−b} = x)`, giving
`P^φ_b(y|x,z) ∝ R_θ(y|x)·h_φ(b−1,y;z)`, sampled by an **exact rejection
sampler** — exact because `h_φ ∈ [0,1]`, and alias-correct for free because
`h_φ` is a function of the canonical state, so alias masses sum exactly under
the pushforward. Batched proposals preserve the law exactly by fixing proposal
order and uniforms in advance.

Full design and the frozen A→B→C→D validation ladder:
**`docs/LEARNED_REACHABILITY_CONTROLLER.md`**. Stage A checks against **exact**
`h` on the enumerable 967-state system and is not optional. **Stage A1 is
DONE** — the rejection sampler is proven to reproduce the exact Doob kernel,
with aliases aggregating for free and batching provably law-preserving (6/6).

> ### 🔍 AUDIT FIRST — much of this already exists
> `docs/AUDIT_EXISTING_CONTROLLER_LANE.md`. A trained 4-seed `h_φ` ensemble, a
> value-guided SMC controller, an RTB trainer and a best-first reference were
> found **after** this design was written.
>
> **Reuse the pattern, not the weights.** The existing `h_φ` conditions on a
> target **molecule** (`e_z`, `sim(y,z)`) — the banked exact-target controller —
> where QED needs an objective **region**. SMC ran on **Lineage B**, not frozen
> `R_θ`, and its 0.5 success used **budget 1000 per lead** against the
> protocol's 20 candidates, so it is **not** comparable to GrIDDD's 45.1 %.
> Its `value_twist` was `None` — V0, no learned twist at all.
>
> **Two words to stop using:** the "33 % ceiling" is not a ceiling — SMC beat it
> at 0.5. Call it the best-first search reference.
>
> **The signal that matters:** something reached 0.5 where policy B reached 0.0.
> The failure was the controller, not COMPOSE.

## C · MOLLEO Task 3 — the five-objective external MOO benchmark

**We do not invent a five-objective molecular task.** One already exists, from
**MOLLEO (ICLR 2025)**, and it is far better suited than copying MOG-DFM's five
*peptide* objectives — affinity, hemolysis, non-fouling and so on are
peptide-specific and COMPOSE cannot run them fairly.

```
max QED      max JNK3      min SA      min GSK3β      min DRD2
```

Minimization objectives are transformed higher-is-better; all five normalized
to [0, 1]. Chemically this is a real design problem: **a drug-like,
synthesizable, JNK3-selective molecule that avoids GSK3β and DRD2.**

| protocol field | value |
|---|---|
| initialization | **120 random ZINC-250k molecules** |
| budget | **≤10,000 oracle calls** |
| seeds | **5** |
| metric | **hypervolume** (Pareto-set selection variant) |
| published baselines | **Graph-GA** + three MOLLEO variants |

This lets us say *"here is an established ICLR five-objective molecular task; we
run COMPOSE under its exact oracles, initialization, budget, normalization and
metric"* — instead of *"we invented DRD2 + QED and bolted NSGA-II onto our
executor."*

### Classical many-objective algorithms — secondary, with a real caveat

NSGA-III, SMS-EMOA, SPEA2 and MOPSO are useful, and **NSGA-III now fits better
than NSGA-II** because five objectives is genuinely many-objective.

> ⚠️ **The selection algorithms need no reimplementation. The molecular
> proposal layer does.** NSGA-III knows how to select nondominated individuals;
> it does **not** know how to mutate a molecular graph. Any such arm requires a
> **single frozen common variation operator**, and that design choice is
> load-bearing.

**Priority: Graph-GA → MOLLEO → COMPOSE**, with classical MOO optional and only
under one frozen common operator. **Not mandatory** before we know whether the
published MOLLEO comparison already supplies the competence evidence.

Later work reuses this five-property setting at a 5,000-call budget, so the task
is recognizable rather than a one-off — **audit protocol differences before
mixing any numbers.**

## D · The Pareto policy gets its own development pipeline

The bi-objective work answered *mechanistic* questions — does preference
recontrol exist, does closed-loop beat generate-and-rank. It was **never
designed to produce the best five-dimensional front.**

> **Do not point the existing five-weight Chebyshev controller at five
> objectives and call the output "COMPOSE."**

The development question is: *given frozen `R_θ`, what is the best scalable
inference policy for constructing a high-quality many-objective front under a
fixed oracle budget?*

Candidate **families** — compared on a disjoint development task, a small number
of principled options, never 50 scalarization weights until HV rises:

- the current preference-conditioned controller
- region / reference-point targeting
- archive-aware hypervolume expansion
- a population/particle realization of the controller

The last three are more COMPOSE-native than sweeping fixed weights: the theory
already frames goal-conditioned value as **reachability to a region in objective
space**, which supports a loop that identifies an under-covered high-contribution
region, targets it, steers there, and folds the result into the nondominated
archive.

**P3/P4/P0c stay untouched as mechanistic validations.**

```
disjoint development  →  freeze ONE policy  →  MOLLEO Task 3 protocol, ONCE
```

---

## E · The paper-wide invariant: `R_θ` is never retrained

> ### **Do not retrain `R_θ` because the downstream question changed.**

This is close to being *the* architectural experiment of the paper, so it must
be a **visible methodological fact**, not something reviewers infer from methods.

### The pinned checkpoint

```
runs/run_v2_01/R_THETA_CHECKPOINT.pt
sha256  c979cdb3d7b0b403bfbf7bfb0aa5098b2588c6d4217770c2c58292b7c4e53de8
bytes   87,804,652
```

**Every claim-bearing table carries a permanent column:**

> **Base molecular process: same frozen `R_θ` checkpoint — YES**

with **this identical hash across the entire paper**. Pinning the hash makes the
invariant *checkable* rather than asserted.

### What may change, per experiment

| experiment | frozen `R_θ`? | what may change | why |
|---|---|---|---|
| GrIDDD QED | **Yes** | goal `z`, controller inference | new single-objective task |
| another single property | **same checkpoint** | new goal / oracle | objective reuse |
| conjunction | **same checkpoint** | goal becomes region/intersection | compositional control |
| Pareto preferences | **same checkpoint** | objective-space region/preference | recontrol, not retraining |
| **5-objective MOO** | **same checkpoint** | many-objective regions + Pareto acquisition | **strongest general optimization test** |
| mid-run preference switch | **same checkpoint** | `z_A → z_B` during the trajectory | dynamic recontrol |
| hard support | **same checkpoint** | legal-support restriction + controller | constraint injection |
| pathwise constraint | **same checkpoint** | support condition at every step | trajectory-level control |
| map reuse | **same checkpoint** | starting realized state + new goal | reuse of prior search |

### Baseline fairness follows from this, not despite it

**Do NOT handicap COMPOSE by retraining it task-by-task merely because a
comparator was trained task-specifically.** On a shared external task:

- GrIDDD runs exactly as published
- Graph-GA / MOLLEO use their native search
- NSGA-III and friends use their native algorithms
- **COMPOSE uses the same frozen `R_θ` it uses everywhere else**, plus its
  proper controller

Then report the training/adaptation distinction **separately**.

**This is more impressive if they beat us somewhere.** They may be specialized;
the question we are asking is whether a *single reusable molecular process* can
be competitive across all of them.

## F · Build `h_φ` as a GENERAL controller, not a QED controller

QED is the development vehicle because it gives a standard external benchmark
and an immediate test of whether the GPS works. **The object being built is
not a QED controller.**

```
h_φ(b, x; x_src, z, m)          z = a general goal-REGION descriptor
```

`QED ≥ 0.9 ∧ sim ≥ 0.4` is merely **one value of `z`** — which is exactly why
the registered goal family was frozen as a 5×4 grid rather than a single
threshold.

**The design target, stated as a target and not a claim:**

> **One `R_θ`. One universal goal-conditioned `h_φ`. Many design problems.**

If that holds, we do not train another controller network for the five-objective
experiment either — the same `h_φ` receives a different objective-region
descriptor. **We do not yet know whether one `h_φ` generalizes from simple QED
regions to a five-dimensional objective space, and we must not claim it before
testing it.**

If it does hold, the paper's claim rises from *"we don't retrain the generator"*
to:

> **The learned molecular process AND its amortized control machinery are both
> reused; a new design problem is specified predominantly through its objective
> functions and goal descriptor.**

## G · MOG-DFM — precedent, not competitor

**MOG-DFM's five-objective experiment is peptide design.** It is presented as
guidance placed on a *pretrained discrete flow-matching generator*, demonstrated
on peptide and enhancer-DNA generation — **not small-molecule graph editing.**

**Acknowledge honestly:** "separate reusable base generation from
objective-specific guidance" is **not uniquely ours** as a philosophy. MOG-DFM
is serious precedent that modular guidance is a real idea rather than a
convenient evaluation choice of ours.

**The structural difference:**

```
MOG-DFM   pretrained sequence DFM              →  multiobjective guidance
COMPOSE   learned stochastic process on an     →  finite-horizon STATEWISE control
          exact executable molecular rewrite
          graph
```

Our leverage is that **every state is a valid molecule and every edge an
executable legal rewrite.** That is what makes remaining-budget reachability,
exact continuation from a realized molecule, support restriction, and pathwise
constraints *consequences of the framework* rather than bolted-on property
guidance.

> **Position: strong conceptual precedent for reusable inference-time
> multiobjective guidance, in biological sequence generation rather than
> executable molecular graph rewriting. NOT a competitor to beat numerically on
> QED.**

### The punchline

Not "COMPOSE has six tricks." It is:

> **One learned molecular world model; many objectives, preferences and
> constraints supplied at inference.**

---

## H · ⭐ THE AUTHORITATIVE EXPERIMENT MAP

**This is the canonical table. Reference THIS, not the blocks in the historical
body above.** Supersedes the first synthesis draft, which conflated external
competence baselines with restricted COMPOSE ablations.

### Train molecular plausibility once; change molecular purpose at inference.

> COMPOSE deliberately separates a **goal-independent learned molecular
> transition law** from the **control problem**. Every claim-bearing task uses
> the same frozen `R_θ`; any objective-specific controller learning, oracle use,
> or inference computation is **reported separately**.

| | |
|---|---|
| **Molecular dynamics retrained?** | **NO** — same frozen `R_θ` throughout, `sha256 c979cdb3…4e53de8` |
| **Control adaptation?** | **experiment-dependent, and reported explicitly** |

**That second line is not a hedge — it is the honest form of the claim.**
Depending on what the region-`h_φ` work establishes, a new objective may still
need a new oracle, new value estimation, or controller calibration. Whether one
universal `h_φ` generalizes with nothing but a new goal descriptor is an
**empirical result to earn**, never an assumption. Reviewers must not be able to
say we hid controller training.

### The three layers

| layer | rule |
|---|---|
| **1 · external fairness = TASK-level parity** | match source cohort, objectives, feasibility criterion, returned candidates / budget *where genuinely specified*, metrics. **Never manufacture equality of internal algorithms.** |
| **2 · COMPOSE primary = NATIVE inference** | stopping · continuation · branching · support restriction · adaptive goal changes · archive reuse · future-aware control |
| **3 · matched restriction = remove EXACTLY that affordance** | the gap says whether the architecture earned its complexity |

**Three distinct objects, never merged:** *native* (claim-bearing) ·
*matched COMPOSE restriction* (causal) · *external* (competence/context).

### The map

| experiment | frozen `R_θ` | **native COMPOSE — claim-bearing** | **matched COMPOSE restriction / causal control** | **external competence / context** | what it establishes |
|---|:---:|---|---|---|---|
| **QED / GrIDDD** | ✅ | 20 native controlled trajectories; **STOP at the first preregistered qualifying molecule**, max H6 | force trajectories through H6; separately size-fixed support in Exp. C | **GrIDDD reported QED benchmark** | ordinary editing competence + value of anytime execution |
| **Trans-dimensional utility (Exp. C)** | ✅ | full legal successor fiber, including heavy-atom-count changes | remove every successor changing heavy-atom count; everything else identical | GrIDDD's insert/delete ablation as **contextual precedent** | whether variable-size execution actually contributes |
| **Exact-target control** | ✅ | remaining-budget verified / future-aware control | myopic greedy + the already-frozen matched controls | none needed | value of future reasoning |
| **Dynamic retargeting** | ✅ | switch goal after observing the realized intermediate and continue | restart from `x₀`; clairvoyant reference; ignore-switch control as frozen | none needed | purpose can change without retraining dynamics or discarding state |
| **Pareto mechanism — P3/P4** | ✅ | closed-loop objective-aware molecular control | **fairly funded generate-and-rank** *(internal control, NOT external)* | none needed for the causal claim | value of closed-loop control |
| **Preference richness — P0c** | ✅ | continuous objective-preference family | frozen sparse / five-weight representation | none | the process expresses a genuine continuum rather than collapsing |
| **K41 compression** | ✅ | frozen budgeted approximation | full-information preference analysis as registered reference | none | ❌ **negative; closed** |
| **5-objective / ordinary MOO competence** | ✅ | native region controller + adaptive front/region construction under the frozen task protocol | same COMPOSE substrate with the preregistered native affordance removed (e.g. non-adaptive / fixed-region search) — **only if already frozen** | **native molecular benchmark comparators + the final frozen MOEA set** | whether reusable COMPOSE control is competitively useful on hard many-objective optimization |
| **Hard support** | ✅ | infeasible transitions **absent from the controlled support** | fixed soft guidance + post-hoc filtering | CDD / PRODIGY / ConStruct as related-work lineage unless exact task alignment exists | value of making feasibility structural |
| **Pathwise constraint** | ✅ | predicate enforced at **every committed molecular state** | endpoint-only same-process control | methods without meaningful molecular intermediates: **`N/A`** path metric | value of an executable molecular trajectory |
| **Map reuse** — 🕓 **QUEUED / conditional** | ✅ | select and branch from a useful previously realized state | restart from `x₀` | none unless a native same-task method exists | whether accumulated molecular search is reusable |

**Preference switching is a subclaim under retargeting / Pareto, not its own
row.** It and retargeting are the same principle — *purpose changes while
molecular dynamics remain fixed*. Creating a separate row would silently mint
another required experiment while we are in execution mode.

### The architectural table that goes under it

| setting | `R_θ` retrained? | goal / controller changes? |
|---|:---:|---|
| QED | **No** | yes |
| new single objective | **No** | yes / goal descriptor |
| conjunction | **No** | yes |
| 5-objective MOO | **No** | yes |
| mid-run objective switch | **No** | yes, **during inference** |
| hard / pathwise constraint | **No** | support & control only |

### Three worked cases where the doctrine bites

**Pareto — do not let the ablation define the method.** Five independent
weights, launching from `x₀` every time, intermediates discarded, endpoints
returned is a **useful matched restriction**; it must not *be* COMPOSE. Native
COMPOSE may say *"the archive is missing this region, I have a realized molecule
near it, branch from there."* Burying that because NSGA-III cannot operate that
way would delete the contribution.

**Pathwise — the fair-looking move destroys the experiment.** *"Other generators
only provide endpoints, so evaluate COMPOSE only at endpoints to be fair"*
defeats the point. Shared task: produce a successful final molecule.
COMPOSE-native stronger task: additionally guarantee forbidden chemistry never
appears along the executable path. Methods without meaningful intermediates get
**`N/A`**, **not zero**.

**Retargeting — the purest case.** COMPOSE has a molecule **literally in hand at
step 3**. Forcing a restart because another model would restart is bizarre;
**restart is the counterfactual** that quantifies statefulness.

### The standing question

Whenever we catch ourselves saying *"but baseline X can't do that, so maybe
COMPOSE shouldn't either"* — the answer is **not automatically "use the
affordance."** It is:

> **Was this affordance part of the method AND prospectively specified before
> seeing the result?**

If yes: use it natively and ablate it separately. That single test blocks **both**
failure modes — artificially handicapping COMPOSE to resemble a comparator, and
retrospectively inventing a convenient capability after seeing an outcome.
