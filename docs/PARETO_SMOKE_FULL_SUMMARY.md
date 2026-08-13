# PARETO_CONTROL_SMOKE_RESULT — final 12-source summary

Target-free Pareto control, held-in smoke. Pair `potency_vs_developability`,
K=8, 12 sources, **held-out never opened** (`held_out_opened: false`).
Lane branch `codex/compose-pareto-control`, result at `7deee51`.
Analysis artifact `diagnostics/pareto_control_analysis.json`
(`schema: compose.pareto.control_analysis`, `status: SMOKE_HELD_IN`).

Every instrument was frozen before these numbers existed; timestamps verified
against zero `DONE` lines at commit time.

---

## 1. Frozen gates

### 1a. Stage 0 task-geometry gates — admitted the pair before any control ran

Instruments: **I-A** (primary) the real model-gated canonical successor fiber,
120 decision states, mean fiber width 586, depths 0 and 1, support only, no
transition probability read. **I-B** (cross-check) 81,500 one-cut MMP pairs
over 5,592 decision states, no kernel at all.

| gate | statistic | value | threshold | verdict |
|---|---|---:|---|---|
| G1 alignment | Spearman rho | −0.274 | < +0.70 | PASS |
| G2 local tradeoff | tradeoff-move fraction (I-A) | 0.512 | ≥ 0.20 | PASS |
| G2 local tradeoff | states offering each direction (I-A) | 1.000 / 1.000 | ≥ 0.25 | PASS |
| G3 no domination | max binding share | 0.760 | ≤ 0.90 | PASS |
| G4 no saturation | reach fractions P / D | 0.700 / 0.000 | ≤ 0.85 | PASS |
| G5 front richness | distinct selections of 5 (I-A) | 2.34 | ≥ 2.0 | PASS |
| G5 front richness | states where all 5 agree (I-A) | 0.100 | ≤ 0.50 | PASS |

C5 budget-exhaustion check, single-objective greedy rollouts at full budget on
20 held-in sources: **P** reach 0.700, median endpoint +3.047, median movement
+3.320 over 6 edits. **D** reach 0.000, median endpoint +0.627, median movement
+0.773. **S** median absolute movement 0.0000 in z_S units, `inert: True` — no
ceiling reachable (T=1 needs zero edits), so the axis was tested for inertness
instead.

G3 binding share 0.76 and G4 potency reach 0.700 were logged as **live risks**
going into the smoke. Neither bit.

### 1b. Analysis-discipline gates — checked at analysis time

| check | pass | meaning |
|---|---|---|
| D1 selection/scoring distinct | **PASS** | no statistic selected and scored by the same function |
| D1b falsifying range declared | **PASS** | every statistic has a range that could refute it |
| D2 arms are distinct | **PASS** | |
| D3 no false null | **PASS** | no guaranteed-sign statistic reports a p-value |
| D4 manifest hashes | **PASS** | |
| D5 contrast parity | **PASS** | each PRIMARY contrast varies exactly one mechanism |
| **D6 HV budget matched** | **FAIL** | → P3 and P4 barred, see §7 |

`global_checks_passed: true`; `barred_contrasts: [P3_closed_vs_open_loop,
P4_closed_vs_open_loop_verified]`.

### 1c. Frozen analysis hierarchy

1. preference responsiveness — ordered, distinct regions
2. final HV and HV-AUC
3. verified vs greedy **per-preference** scalarized value (magnitude only; sign guaranteed)
4. verified vs greedy **set-level** HV, independent, two-sided
5. trajectory and oracle resource curves, separately, never merged

**Gap, stated rather than omitted: HV-AUC was declared in item 2 and in the
reporting rules ("both native and raw conventions are reported; neither may be
substituted for the other") but was NOT computed in this analysis.** Only final
HV exists. Item 2 is therefore half-delivered.

### 1d. The six contrasts and their parity status

| contrast | arm vs base | varies | isolates one mechanism | status |
|---|---|---|---|---|
| P1 unguided floor | `greedy_pref` vs `unguided` | controller **and** objective | **no** | CONTEXT_ONLY — may not carry a headline |
| P2 future awareness | `verified_pref` vs `greedy_pref` | controller | yes | **PRIMARY** |
| P3 closed vs open loop | `greedy_pref` vs `gen_rank@greedy` | controller | yes | **INVALID_CONTRAST** |
| P4 closed vs open loop (verified) | `verified_pref` vs `gen_rank@verified` | controller | yes | **INVALID_CONTRAST** |
| P5 preference responsiveness | `greedy_pref@w=0.9` vs `@w=0.1` | objective | yes | **PRIMARY** |
| P6 same-prefix branching | `branch@w_i` vs `branch@w_j` | objective | yes | **PRIMARY** |

---

## 2. Preference responsiveness — real, and imperfectly ordered

### P5 — objective_0 gap, w=0.9 minus w=0.1
Falsifying range (−inf, +inf); a preference-blind controller gives 0.

| n | mean | median | 95% CI | W/L/T |
|---:|---:|---:|---|---|
| 12 | **+0.5782** | +0.5169 | **[+0.1659, +0.9898]** | 10 / 2 / 0 |

### P6 — same-prefix branching
Falsifying range [1, 5]; 1 means the prefix determines the future.

- mean distinct endpoints from the **byte-identical** branch point: **3.75 / 5**
- `all_five_identical_fraction`: **0.0**
- `all_branches_started_at_the_branch_point`: **true**

### Per-arm ordering statistics — the honest limitation

| arm | median Spearman ρ(w, objective) | monotone fraction | correct adjacent fraction | mean distinct endpoints | preference-blind |
|---|---:|---:|---:|---:|---|
| `verified_pref` | 0.5643 | 0.167 | 0.556 | 4.17 | no |
| `greedy_pref` | **0.6357** | 0.167 | 0.639 | 3.83 | no |
| `unguided` | **0.4321** | 0.000 | 0.507 | **4.92** | **YES** |
| `gen_rank@greedy` | 0.7071 | 1.000 | 0.583 | 1.67 | no |
| `gen_rank@verified` | 0.8944 | 1.000 | 1.000 | 2.58 | no |

Three things this table says that the headline does not:

- **Guided ρ +0.636 sits above the preference-blind floor of +0.432 without
  dominating it**, and that floor is noisy at n=12.
- **Monotone on only 1 source in 6** (0.167) for both guided arms.
- **`verified_pref` orders WORSE than `greedy_pref`** (0.564 vs 0.636). Future-aware
  control improved the set (§5) while not improving the ordering.
- **The preference-BLIND arm produced the MOST distinct endpoints** (4.92 vs 3.83).
  This is the cleanest possible demonstration that distinctness is not evidence
  of preference control — which is why ordering, not distinctness, was the test.

`gen_rank`'s high ρ and 1.000 monotone fraction come with fronts of 1.67 and
2.58 distinct points: ordering a near-degenerate candidate pool is easy.
Descriptive only — no formal contrast exists (§7).

> **Verdict.** Preferences produce **different futures, loosely ordered** by the
> requested tradeoff. **Five cleanly ordered Pareto regions is NOT supported by
> this smoke.**

---

## 3. Final held-in-scaled hypervolume

Frozen nadir = held-in p5; utopia = held-in p99. Both frozen before any HV number.

| arm | HV (mean over 12) | feasibility |
|---|---:|---:|
| `verified_pref` | **0.9378** | 1.00 |
| `greedy_pref` | 0.8488 | 1.00 |
| `gen_rank@verified` | 0.3083 | 1.00 |
| `unguided` | 0.1912 | 1.00 |
| `gen_rank@greedy` | 0.1488 | 1.00 |

**p99 exceedance, descriptive and unclipped:** 12.67% of endpoints exceed `z*`;
max excess **0.9093** IQR units. `z*` is a normalisation scale, not an attainable
ceiling. `no_clipped_variant` — introducing a clipped or robust HV after seeing
these results is exactly what the freeze exists to prevent.

**HV-AUC: not computed.** See §1c.

### P1 unguided floor — CONTEXT_ONLY, may not carry a headline

Varies controller **and** objective, so it isolates nothing.

| metric | mean | median | 95% CI | W/L/T |
|---|---:|---:|---|---|
| normalized HV | +0.6576 | +0.6508 | [+0.5286, +0.7817] | 12/0/0 |
| preference coverage | +0.1500 | +0.1000 | [−0.0667, +0.3667] | 6/3/3 |
| nondominated set size | +0.0833 | 0.0000 | [−0.5833, +0.7500] | 5/3/4 |
| endpoint diversity | +0.1015 | +0.1047 | [−0.0436, +0.2213] | 9/3/0 |

---

## 4. Per-preference scalarized value — GUARANTEED SIGN, magnitude only

Statistic: mean over the five preferences of `s(greedy endpoint | w) −
s(verified endpoint | w)`; Chebyshev is minimised, so positive favours verified.

| n | mean | median | 95% CI | W/L/T |
|---:|---:|---:|---|---|
| 12 | **+0.1431** | +0.0763 | [+0.0576, +0.2768] | **12 / 0 / 0** |

`sign_is_guaranteed: true`. **The 12–0 is definitional** — greedy's action is
always in the shortlist and strict improvement never commits a lower `V_G`.
`pvalue_NOT_REPORTED`: policy improvement fixes the sign for each requested
preference. **Reported as a magnitude, never as evidence.**

---

## 5. P2 set-level hypervolume — the contrast that could have failed

`HV(verified endpoint set) − HV(greedy endpoint set)`. `sign_is_guaranteed:
false`; falsifying range (−inf, +inf), negative values real and live.

| metric | mean | median | 95% CI | W/L/T |
|---|---:|---:|---|---|
| **normalized HV** | **+0.0890** | +0.0750 | **[+0.0379, +0.1449]** | **10 / 2 / 0** |
| preference coverage | +0.0333 | 0.0000 | [−0.1167, +0.2000] | 4/4/4 |
| nondominated set size | −0.1667 | 0.0000 | [−1.0000, +0.7500] | 2/4/6 |
| endpoint diversity | +0.0427 | +0.0169 | [−0.0221, +0.1082] | 6/6/0 |

Only hypervolume resolves. Coverage, set size and diversity all span zero, and
**nondominated set size has a negative mean** — verified control does not widen
the front, it pushes it outward.

The sign was free. Sources **000 (−0.001)** and **010 (−0.073)** went the other
way. A registry keyed on the contrast alone had suppressed this comparison
entirely until it was rekeyed on (contrast, metric); source 000 is now a
permanent regression fixture.

> **Verified control improves the endpoint SET, not merely each trajectory
> individually.** Pointwise improvement does not imply set-level improvement —
> five individually better points can enclose less dominated area.

**This contrast is NOT a pass condition for Pareto.** Exact-target recovery
already carries "future-aware reasoning matters"; Pareto's job is target-free
reuse across preferences.

---

## 6. Resource accounting — three axes, never merged

`internal_axis`: completed controlled trajectories, COMPOSE arms only.
`external_axis`: unique valid canonical evaluations, and oracle requests.
External methods are **never** placed on the trajectory axis.

### N_90 — trajectories to reach 90% of POOLED ATTAINABLE hypervolume

`HV_star_internal` = pooled terminal nondominated union of the predeclared
internal arms at the fixed maximum trajectory count. **Never one arm's own
front.** Its value is known only after every arm runs, but the RULE is
preregistered and method-symmetric: a strong arm that expands the pooled
frontier raises the bar for everyone including itself.

| arm | median (uncensored) | n uncensored | n censored | censoring rate |
|---|---:|---:|---:|---:|
| `verified_pref` | **2.00** | **12** | **0** | 0.000 |
| `greedy_pref` | 2.00 | 7 | **5** | 0.417 |
| `unguided` | — | 0 | 12 | **1.000** |
| `gen_rank@greedy` | — | 0 | 12 | **1.000** |
| `gen_rank@verified` | — | 0 | 12 | **1.000** |

Censored sources are counted, **never imputed at B_max**; a median over
uncensored sources alone is meaningless without the censoring rate beside it.

Name discipline, enforced: never "reference HV" and never "90% of COMPOSE's
front" — the name is where this bias survives review.

### Preference-region coverage

| arm | mean |
|---|---:|
| `verified_pref` | 0.433 |
| `greedy_pref` | 0.367 |
| `gen_rank@verified` | 0.167 |
| `unguided` | 0.150 |
| `gen_rank@greedy` | 0.150 |

### Cost per source — real and unflattering

| arm | kernel calls | oracle requests | unique valid canonical evaluations |
|---|---:|---:|---:|
| `verified_pref` | 398.9 | 443,240.5 | **269,883.8** |
| `gen_rank@verified` | 167.1 | 133.0 | 57.5 |
| `unguided` | 18.6 | 5.0 | 5.0 |
| `greedy_pref` | 16.2 | 21,004.0 | 10,936.3 |
| `gen_rank@greedy` | 10.8 | 5.5 | 2.6 |

Counter soundness: **612.2 unique evaluations per kernel call** against a census
mean fiber width of **586**. The counter is sound.

**The oracle-demand ratio against `gen_rank` remains WITHDRAWN as a cost claim**
pending a matched comparison.

Instrumentation defect recorded, not a result: `drd2_batch_size: 1` —
`objective_vector()` calls `oracle.margin_many([key])` with a list of exactly one
molecule, so every evaluation is its own scorer invocation. Counts are honest;
batching is left on the table. And `N_drd2 == N_descriptor == N_all` only because
nothing short-circuits — the counters must be split at their own call sites
before any cost claim.

---

## 7. Inadmissible comparisons and withdrawn statistics

### P3 and P4 — WITHHELD, `status: INVALID_CONTRAST`

`barred_by: D6_hv_budget_matched`; `reason: BUDGET_ASYMMETRIC_KERNEL`.
All four metrics withheld on both contrasts.

Kernel ratios **1.492** and **2.388** against a **1.25** limit. The matcher
assumed 6 fresh kernel calls per trajectory, but unguided trajectories from a
shared root collide in the enumeration cache, so `gen_rank` was **underfunded —
an error in the direction that flatters us.**

> **`gen_rank`'s poor numbers must NOT be read as defeating that baseline.**

There is no admissible closed-loop versus generate-and-rank comparison in this
smoke. The fix is additive — iterate the matcher until the ledger reaches target
and top up `gen_rank` trajectories — **not** a rerun of the COMPOSE arms, and
**not** a change to COMPOSE because the baseline was underfunded.

### Withdrawn statistics

| statistic | reason |
|---|---|
| `sacrifice_to_win` | circular: action chosen as `argmax V_G`, then scored by `V_G` |
| `verified_vs_greedy_sign_test` | policy improvement makes the null of 0.5 false |
| `best_candidate_reaches_pooled_p99` | with a ~589-wide fiber this fires with probability 0.997 whatever the truth is — no falsifying range. Withdrawn from census gate G4 **before** any pair verdict was read (DECISION_LOG D-007) |

### Superseded

The n=4 exploratory pattern "HV sign may track saturation" was **FALSIFIED at
n=5** and marked superseded. It is not replaced by a weaker version — not by
"maybe saturation predicts magnitude", not by anything.

### Watch-items — the live risks did not materialise

W1 1.00 · W2 last productive step 5.0/6, no early stall · W3 spread ratio 0.427,
neither axis flat · W4 3.83/5 distinct, none collapsed · W5 1.00, no saturation
stall.

---

## 8. Formal escalation decision

### Framing licensed by this smoke

**Barred from Figure 5:**
> ~~"COMPOSE precisely sweeps the Pareto frontier according to user preference."~~

**Provisional caption, until a larger development says otherwise:**
> The same frozen molecular process can be recontrolled across target-free
> property preferences, producing a diverse Pareto set with **measurable
> preference responsiveness**.

"Measurable" is upgraded only by data, not by a better sentence.

### The distinction preserved

The current experiment optimizes `U_{w_i}(x_H^{(i)})` independently per
trajectory. It does **not** optimize `HV({x_H^{(1)},…,x_H^{(5)}})`. That is the
first-principles reason perfect frontier sweeping should not be expected
automatically — and this smoke shows the gap is real, since verified control
improved individual preference trajectories on 12/12 while reducing set
hypervolume on 2/12.

**The current experiment is the right first experiment for proving target-free
preference control. It is not necessarily the optimal COMPOSE algorithm for
deliberately sweeping a frontier.**

### Escalation levels

| level | what it is | status |
|---|---|---|
| **1 — fixed preference control** | five predetermined weights; current design | **PRIMARY. Stays primary.** |
| **2 — aspiration / reference-point control** | five predeclared *property-space* targets `a_1…a_5`; terminal desirability from normalized shortfall (max-min / Chebyshev, matching the frozen goal algebra). Still target-**free** — regions of property space, not target molecules | conditional, **not built** |
| **3 — archive-aware outer control** | choose the next preference to maximise expected marginal hypervolume against the current archive | **not built, probably beyond this paper** |

### The decision rule — fixed before the development reports

| development outcome | consequence |
|---|---|
| preference response **and** HV both strong | keep the controller. The one scaling follow-up is efficient reference-based shortlisting |
| HV strong, ordering still mediocre | **one** aspiration-point experiment (Level 2), preregistered as a distinct cleaner formulation of frontier targeting |
| both weak | **demote Pareto.** Do not build machinery to rescue it |
| fixed preferences cluster badly *despite* strong individual optimization | only then consider Level 3, as a motivated extension — **never as a rescue** |

Level 2 must be motivated as a cleaner formulation and preregistered before
running. It may **not** be introduced because a particular source or an n=12
result is annoying.

### Next actions authorised, in order

1. **Instrumentation only, controller untouched** — exact vectorized batching
   gated on identical-trajectory regression tests; split counters
   (`N_drd2_requests`, `N_drd2_unique`, `N_drd2_evaluator_batches`,
   `N_descriptor_requests`, `N_scorer_batches`, `N_kernel_calls`,
   `N_complete_trajectories`); fix the `gen_rank` matcher per the frozen resource
   convention and top up only the missing baseline computation.
2. **The reference-law ablation now definitely runs here** — Pareto passed its
   task-geometry gate and shows genuine target-free behaviour, so the
   preregistered primary host is live
   (`docs/REFERENCE_LAW_ABLATION_PREREGISTRATION.md`). It may matter more than
   P2, because it answers why learn molecular plausibility at all if control
   supplies purpose.
3. **Only if the capability is established: shortlisting.** Control currently
   asks ~586 successors for objective values at nearly every decision. The
   architecture suggests legal support → plausibility shortlist via `R_θ` →
   expensive goal evaluation → control, i.e. legality → plausibility → purpose.
   A scaling follow-up **gated on full-fiber control proving the capability
   first**, not a substitute for proving it.

### Explicitly not the goal

> ~~five weights → five perfectly monotonically ordered points~~

Molecules are discrete and the reachable front is irregular; adjacent weights may
legitimately map into the same basin. ρ ≈ 0.6 can be respectable if the front is
discrete, endpoints span the actual tradeoff, HV is strong, and the
preference-blind floor is substantially weaker.

---

Sources: `docs/PARETO_CONTROL_SMOKE_RESULT.md`,
`docs/PARETO_DEVELOPMENT_PREREGISTRATION.md`,
`diagnostics/pareto_control_analysis.json`,
`diagnostics/pareto_oracle_accounting_audit.json`,
`docs/workstreams/pareto-control/STATUS.md` — lane branch
`codex/compose-pareto-control` at `7deee51`.
