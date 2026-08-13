# Workstream E — Target-free Pareto / preference control

**Artifact status: `DESIGN_ONLY`.** No held-out panel has been opened. No Modal
run has been launched. Everything below is frozen *before* the Stage 0 census
result is read; the census verdict is recorded separately in `DECISION_LOG.md`.

---

## 1. Claim

> The same frozen COMPOSE process — one `R_theta`, one rewrite kernel, no
> retraining — pursues **user-specified tradeoffs between competing objectives
> when no answer molecule exists**. Different preference weights over the same
> objective pair, applied from the same realized molecular state under the same
> edit budget, produce genuinely different endpoints, and future-aware
> remaining-budget control covers the achievable front better than myopic
> control at matched oracle budget.

This is the third scalable control capability, alongside exact-target recovery
(sealed, 40% → 62%) and realized-state retargeting.

**Why it is not a repeat of exact-target recovery.** Exact-target recovery asks
"can control reach a known molecule?" — and a known molecule hands the
controller a privileged local heuristic, Tanimoto-to-the-answer, which is not
available in any real design problem. Pareto control removes the answer. The
only guidance is the objective pair and a preference weight. That is where
goal-conditioned desirability genuinely belongs.

## 2. Non-claims

- No claim that COMPOSE beats a dedicated multi-objective molecular optimizer.
  No external baseline is qualified in this lane (Workstream D owns that).
- No claim of medicinal-chemistry plausibility for any endpoint.
- No claim about learned amortization. **There is no `h_phi` in this lane.**
  Whether future-aware preference control provides a real gain is established
  first; a large amortization project before that is exactly the mistake this
  project already made once.
- No claim that a good hypervolume implies good control — see §7.1, the
  HV inflation channel.

## 3. Frozen inputs

| object | path | sha256 |
|---|---|---|
| held-in / held-out split | `diagnostics/editing_v2_matched_validation_reserve_ids.json.gz` | `ba9270fa8bea1a67...` (full below) |
| goal-language normalizers | `diagnostics/retarget_goal_language_normalizers.json` | `187d1ccc60b00c85...` |
| DRD2 oracle manifest | `artifacts/oracles/drd2_svm_v1/drd2_oracle_manifest.json` | `423737efb6e4c942...` |
| DRD2 SVM parameters | `artifacts/oracles/drd2_svm_v1/drd2_svm_parameters.npz` | `7c9224c12c423ba2...` |
| one-cut MMP miner | `scripts/build_analogue_trace_pool.py` | `05a7326365c11021...` |
| committed reachability | `diagnostics/retarget_calibration_result_3plus3_fixed.json` | `5fdca52fcba5e281...` |

Full hashes are in `handoff.json`. `R_theta` is **FROZEN**: no retraining, no
`h_phi` training, no corpus edits, no operator changes, no objective redesign.

**Held-in only.** The census and every Stage 1 test use
`training_source_keys` (n = 96,094). `reserve_source_keys` (n = 10,653) is
**held-out and is not opened in this lane**.

## 4. Objective definitions — frozen

All objectives are **maximized** and expressed in held-in IQR units so that a
Chebyshev scalarization is meaningful across them.

**O-P — potency.**
```
z_P(x) = (drd2_logodds(x) - (-5.5216576)) / 2.6160789
```
Log-odds, never P(active): the pool sits at median P(active) = 0.004 where
probability is saturated.

**O-D — developability.** The frozen retargeting goal object, unchanged:
```
m_QED (x) = (QED(x)   - 0.60) / 0.3139246
m_logP(x) = min(cLogP(x) - 1.0, 4.0 - cLogP(x)) / 2.0833575
z_D(x)    = -tau * log sum_j exp(-clip(m_j(x), -1.5, +1.5) / tau),  tau = 0.25
```
with `MARGIN_CLIP = 1.5`, `SOFTMIN_TAU = 0.25`, `QED_FLOOR = 0.6`,
`LOGP_BOX = (1.0, 4.0)` — the exact constants of
`modal_apps/retarget_goal_calibration_app.py`.

> **Named escape hatch, closed in advance.** `z_D` is *clipped*, so it has a
> hard ceiling. An unclipped soft-min, or QED alone, would not saturate. If O-D
> fails the saturation gate G4, **this lane may not switch to the unclipped
> variant.** Redefining an objective after it fails a gate is tuning, not
> calibration, and it is precisely how the "29/30 vs 28/30" ceiling was
> originally mistaken for a result. Any change to the frozen goal language is a
> main-lane decision.

**O-S — source similarity.**
```
z_S(x; x_0) = (T(x, x_0) - centre_S) / s_S
```
`T` = ECFP4 (Morgan radius 2, 2048 bits) Tanimoto to the source. `centre_S` and
`s_S` follow the **same recipe as the frozen normalizers** — held-in median and
held-in IQR — computed over the census candidate set in census Phase 0, written
to the census artifact, and fixed before any gate statistic is evaluated. The
recipe is predeclared here; the number is not, because it does not exist yet.
Gate statistics G1–G2 are rank/sign based and therefore scale-invariant, so the
normalizer cannot influence them.

## 5. Predeclared objective-pair order — **BINDING, set before the census ran**

The census is run on all three pairs, and the **FIRST pair in this order that
satisfies every gate criterion in §6 is adopted.** Controller performance is not
inspected before the choice. The order is not reordered after the census.

1. **potency vs developability**  (O-P, O-D)
2. **potency vs source similarity**  (O-P, O-S)
3. **developability vs source similarity**  (O-D, O-S)

DRD2-potency vs developability is *not* assumed good merely because retargeting
used it. Retargeting used it as a **switch**, which needs only that the goals be
distinguishable; Pareto control needs a genuine **front**, which is a strictly
stronger requirement.

## 6. Stage 0 — tradeoff census and the headroom gate

### 6.1 Instruments — two, run on the same sources

Both are **fully local**. No Modal run, no GPU, no `R_theta` weights.

**I-A — the real canonical successor fiber (PRIMARY).** The model-gated
canonical successor support is enumerable on this machine:

```
open_process_v2_t1_source(<local Active8>, gate_zero_decision_path=<v7>, ...)
build_process_v2_score_revised_scratch_runtime(source, materialized_state=<matscorer>)
canonical_successor_result(model, pad_molecular_graph(g, 48), 0.5).batch.successors
```

Measured cost on this machine: **~5.5 s per state, ~334 canonical successors**
for a drug-sized molecule, after a 63 s one-time runtime build.

> **Why the support is the frozen support even without the frozen checkpoint.**
> The legal mask comes from the model's structural action tables, not from its
> weights; only the *probabilities* need the trained `R_theta`. The census reads
> **support only** — which candidates exist and what their properties are — and
> never reads a probability. The local Active8 pairs with gate-zero **v7** while
> the volume's `RUN_PATHS` pins **v6**, so this was checked rather than assumed:
> `process_identity_sha256`, `gate_zero_structural_contract_sha256`,
> `contracts_binding_sha256`, `active8_completion_sha256`,
> `enforced_structural_clauses`, `model_family_counts` and
> `executor_rule_counts` are **byte-identical between v6 and v7**. The two
> decisions differ in corpus accounting, not in the executable support.

A **decision state** is a held-in molecule with heavy-atom count in [18, 38]
(the frozen cohort size band, outcome-independent) with a non-empty successor
set. Two depths are censused per source: `d0` = the source itself, and `d1` = a
**uniformly random** legal successor of the source under a fixed seed. The
depth-1 state is drawn objective-independently so that the same states serve all
three candidate pairs and no pair gets a state selected in its favour.

**I-B — one-cut matched molecular pairs (CROSS-CHECK).** `mine_one_cut_pairs`
over the full held-in pool: 81,500 pairs, 30,976 molecules with >= 2 neighbours,
51.6 s, no kernel at all. An MMP pair is one local structural change on a shared
core — a real medicinal-chemistry analogue of an executor move, drawn from real
molecules rather than from the executor's own action set.

### 6.2 Why both, and what a disagreement between them means

The two instruments have different and partly opposite biases, so agreement
between them is worth more than either alone.

- I-B's neighbourhood has median degree 1 and p90 degree 5 against I-A's ~334.
  I-B therefore **understates** tradeoff availability and preference
  distinguishability.
- I-A is the executor's own action set, so it is the right support for the
  claim, but it is exactly the set the controller will search — it cannot tell
  us whether the tradeoff structure is a property of chemistry or an artifact of
  the operator inventory. I-B, drawn from real molecules, can.

Decision rule, predeclared:

- **Both pass** → the gate passes; conclusive.
- **Both fail** → the gate fails; conclusive.
- **I-A passes, I-B fails** → gate passes, flagged
  `OPERATOR_SET_DEPENDENT` — the tradeoff exists in the executable support but
  is not visible in real analogue pairs, which is a caveat the paper must carry.
- **I-A fails, I-B passes** → gate **fails**; the controller can only act
  through I-A, so a tradeoff invisible to the executor is not actionable.

G1 (pool correlation) and G4 (saturation) do not use either neighbourhood
instrument: G1 is a pool statistic and G4 is read from **real-fiber rollouts**,
so both are conclusive on their own.

> **I-A is the instrument of record. I-B corroborates and never decides.**
> Measured on the adopted pair: I-A gives **2.342** mean distinct selections and
> **0.100** unanimous states; I-B gives **1.656** and **0.412**. I-B's value sits
> *below* the predeclared G5 floor of 2.0 — a proxy-only census would have read
> borderline where the executable support is comfortable. The two agree closely
> on sign statistics (tradeoff-move fraction 0.512 vs 0.535) and diverge on
> selection statistics, exactly as a neighbourhood of median degree 1 versus 586
> predicts.
>
> This project has already been misled by a matched-pair proxy once: the
> movability census implied the DRD2 threshold was out of reach in four edits,
> and the real fiber then climbed **+4.42 log-odds in three**. The asymmetry
> here runs the same way. A controller can only act through the executable
> support, so that is what the gate is judged on.

### 6.3 The five measurements

| id | measurement | instrument | definition |
|---|---|---|---|
| C1 | objective correlation | pool + I-A | Spearman `rho` between the two objectives over all census candidates; and, where both objectives are pool-defined, over the raw held-in pool |
| C2 | Pareto-front width | I-A | size of the nondominated set over census candidates in normalized coordinates; the range of each objective along that front; and the number of front points selected by at least one weight on a fine Chebyshev grid (a front that is effectively a point selects 1) |
| C3 | local successor tradeoff frequency | I-A + I-B | per **move**: the fraction landing in each sign quadrant of `(dz_1, dz_2)`. Per **state**: the fraction of legal decision states offering at least one `(+,-)` candidate, and separately at least one `(-,+)` candidate |
| C4 | preference distinguishability | I-A + I-B | per state, the augmented-Chebyshev argmax under each `w in {0.1,0.3,0.5,0.7,0.9}`; report the mean number of **distinct** candidates selected and the fraction of states where all five preferences select the same candidate |
| C5 | reachable floors/ceilings under a 6-edit budget | I-A rollouts + committed shards | **single-objective greedy rollouts at the full 6-edit budget**, run on I-A. Ceiling: analytic for O-D (the clipped soft-min cannot exceed `1.326713`), held-in pool p99 for O-P. O-S has no reachable ceiling — `T = 1` needs zero edits — so it is tested for **inertness** instead: six real edits that cannot move an axis make it uncontrollable. `retarget_calibration_result_3plus3_fixed.json` corroborates at 3 edits. See DECISION_LOG D-007 for the statistic this replaced and why. |

### 6.4 The headroom gate — numeric thresholds, predeclared

All five must pass. Each threshold is stated so that the statistic could
plainly fall on either side of it.

| gate | criterion | reject if | judged on |
|---|---|---|---|
| **G1** alignment | objectives must not be near-redundant | Spearman `rho >= +0.70` | pool; conclusive |
| **G2** local tradeoff | tradeoff moves must be common | quadrant fraction `(+,-) + (-,+) < 0.20`; **or** either direction alone `< 0.05`; **or** fraction of states offering a `(+,-)` candidate `< 0.25`; **or** fraction offering a `(-,+)` candidate `< 0.25` | I-A primary, I-B cross-check, §6.2 rule |
| **G3** no domination | no single objective may be binding for every preference | the same objective is the binding Chebyshev term in `> 0.90` of (state, preference) decisions | I-A primary |
| **G4** no saturation | a 6-edit budget must not exhaust either axis | either objective's reach fraction `> 0.85` (for O-S: the axis is inert) | I-A 6-edit rollouts; conclusive |
| **G5** front richness | preferences must select different candidates | mean distinct selections across the five preferences `< 2.0`; **or** the fraction of states where all five agree `> 0.50` | I-A primary, I-B cross-check, §6.2 rule |

**Why this gate exists.** A calibration in this project reported
"29/30 vs 28/30" and called it a result. It was a **ceiling**: the task was easy
enough that every arm reached the top, so no controller difference could have
appeared. A task where different preferences do not create genuinely different
optimal futures cannot support the Pareto claim, and the failure is invisible
after the fact unless the thresholds were written down first.

## 7. Stage 1 — arms, scalarization, budget

### 7.1 Scalarization — frozen

**Primary: augmented weighted Chebyshev**, minimized.
```
s(x | w) = max_j [ w_j * (z*_j - z_j(x)) ]  +  rho * sum_j (z*_j - z_j(x))
w = (w, 1 - w),   rho = 1e-3
```
`z*` is the utopia point, `r` the reference point:
```
z*_j = held-in p99 of z_j          r_j = held-in p5 of z_j
```
both frozen from held-in in census Phase 0, before any arm runs.

Chebyshev, not only a weighted sum: **every** Pareto point is the Chebyshev
optimum for some weight, whereas a weighted sum can only ever select points on
the convex hull of the front. Using a weighted sum alone would systematically
miss nonconvex regions and would make the front look smaller than it is.

**Secondary, reported for diagnosis: weighted sum** `sum_j w_j z_j(x)`. The
statistic of interest is the fraction of Pareto points selectable by Chebyshev
but by no weighted-sum weight — a direct measurement of how nonconvex the
achieved front is.

**Preference grid:** `w in {0.1, 0.3, 0.5, 0.7, 0.9}` on objective 1.
**Budget:** `H = 6` productive edits.

### 7.2 Arms — no learned `h_phi`

| arm | controller | description |
|---|---|---|
| `unguided` | none | sample 6 edits from frozen `R_theta`, preference-blind. Five branches from five fixed seeds. The floor: whatever preference coverage this achieves is chance. |
| `gen_rank@<budget>` | open-loop | generate `K` unguided 6-edit trajectories, then for each preference return the Chebyshev-best endpoint. `K` is **computed from the control arm's own cost ledger at run time**, never chosen by hand, on whichever budget axis the contrast declares. |
| `greedy_pref` | myopic | at each state, commit the fiber `argmin s(y | w)`. |
| `verified_pref` | future-aware | shortlist `S` candidates; for each, compute `V_G` = the deterministic greedy-under-`w` continuation of the remaining budget; commit `argmin` under **strict improvement** (ties keep greedy); re-plan from the committed state. |

`gen_rank` is instantiated at **two** budget levels, `gen_rank@greedy` and
`gen_rank@verified`, because a single instance cannot hold budget parity against
two control arms with different costs. This is a parity requirement, not an
extra arm type.

> **No arm may be described as globally "budget matched".** With a ~600-successor
> fiber there is no single notion of matched compute, so Pareto quality is
> reported against **both** counters — unique valid oracle evaluations and
> kernel / reference-process calls — and each arm's position on both is stated.
> A kernel-matched generate-then-rank baseline may be vastly oracle-richer than
> the closed-loop arms; that makes it a strong baseline **in one resource
> dimension**, which is how it must be described.
>
> **The two budget axes buy ~600x different amounts of search, and the contrast
> must be reported as a bracket.** One kernel call yields ~600 candidate
> molecules on this process, measured. So matching `gen_rank` on **native oracle
> calls** hands it roughly 600x the closed-loop arms' kernel budget — that is
> the hypervolume inflation channel wearing a benchmark convention's clothes.
> Matching on **kernel calls** instead gives it far fewer distinct molecules
> than the closed-loop arms see, which understates it on the axis the
> multi-objective literature actually budgets. Neither matching is "the fair
> one." P3 and P4 are therefore declared as a **bracket** — `gen_rank` at both
> matchings — with the unclaimed axis reported in each case. Quoting one matching
> alone would be a reporting choice that decides the winner, the same failure as
> quoting one HV-AUC convention.
>
> **Only the kernel-matched end is affordable in the smoke.** A native-matched
> `gen_rank` needs ~2,600 unguided trajectories per source, ~15,600 kernel calls,
> ~30 h per source. It is costed and deferred to main, not quietly dropped — and
> until it runs, **P3 and P4 are one-sided and must be read as such.**

Shortlist for `verified_pref`, frozen: 4 by immediate scalarized score, 2 by
`R_theta` reference probability, 2 uniformly at random — the same 4/2/2
stratification the calibration used. The random stratum is retained
deliberately: a candidate universe defined entirely by the greedy score cannot
show that a non-greedy action was worth taking.

### 7.3 Metrics — frozen before any result

1. **Normalized hypervolume (HV).** 2-D, per source, over that arm's committed
   **endpoints only** — exactly one endpoint per preference branch, so every
   arm contributes exactly 5 points. Reference point `r` (held-in p5), utopia
   `z*` (held-in p99); HV divided by the HV of the box `[r, z*]`, so
   HV in [0, 1].

   > **The HV inflation channel, and how it is closed.** An arm that simply
   > generates more molecules inflates HV without better control. The control is
   > structural, not statistical, and it separates two things that are easy to
   > conflate:
   >
   > - **Equal endpoint counts, enforced on every HV contrast.** HV is computed
   >   over committed endpoints only and every arm contributes exactly 5. An arm
   >   may cost more; it may never contribute more points.
   > - **Compute parity, enforced on one axis at a time.** Compute is *not* one
   >   of the four parity dimensions — `budget` there means the **edit** budget
   >   `H = 6`, which every contrast holds. Each contrast declares which compute
   >   axis it claims, and only that axis is enforced; the other is reported with
   >   the flag `COMPUTE_ASYMMETRIC_ON_AN_UNCLAIMED_AXIS_REPORT_THE_RATIO`.
   >     - **P2 claims neither.** A lookahead controller intrinsically spends
   >       more compute than a myopic one, and throttling it to greedy's compute
   >       would delete the mechanism under test.
   >     - **P3/P4 claim `kernel`, and cannot also claim `native`.** One kernel
   >       call yields ~600 candidates, so the two axes are unsatisfiable
   >       together: kernel-matched `gen_rank` sees ~4 endpoints against the
   >       control arm's ~15,600 scored candidates, and native-matched
   >       `gen_rank` would need ~2,600 trajectories and ~30 h per source. Each
   >       instance claims one axis; the pair is the declared bracket.
   >
   > An earlier version of the gate failed P2 for its compute asymmetry, which
   > would have made verified-vs-greedy unrunnable. That is recorded rather than
   > silently corrected: a check that forbids the experiment is as wrong as one
   > that permits a confound.

2. **HV-AUC — two conventions, reported together, never substituted.**
   - `HV-AUC@native`: x-axis = **distinct molecules scored** by the property
     oracle (the benchmark-native convention).
   - `HV-AUC@raw`: x-axis = **every scoring invocation**, including cache hits
     and every candidate evaluated inside a lookahead rollout.

   Both appear in the same table with their ratio. Native counting alone hides
   what verified control spends inside rollouts; raw counting alone hides
   caching efficiency. Quoting one without the other is a reporting choice that
   can decide the winner, so it is forbidden.

   Kernel calls (`R_theta` forwards) are reported as a third cost axis because
   they dominate wall time, but HV-AUC is defined against the two oracle
   conventions as above.

3. **Preference coverage.** The fraction of the five preferences for which the
   arm's endpoint is the unique Chebyshev argmin *within that arm's own endpoint
   set*. Range [0.2, 1.0]; 0.2 means all five preferences collapsed to one
   molecule.

4. **Nondominated-set size.** `|ND|` over the arm's 5 endpoints per source, in
   {1..5}; mean over sources.

5. **Feasibility.** Fraction of committed trajectories completing all 6 edits
   with every state valid, representable, and non-dead-end.

6. **Source similarity.** ECFP4 Tanimoto of endpoint to `x_0`, as **context**.
   If O-S is one of the adopted objectives, source similarity moves into the
   objective set and heavy-atom drift replaces it as the context metric — a
   metric may not simultaneously be the objective and the independent check.

7. **Endpoint structural diversity.** `1 - mean pairwise ECFP4 Tanimoto` among
   the arm's five endpoints per source. This is the direct quantitative form of
   "one molecular history, many preference-dependent futures".

### 7.4 Statistics — the source is the independent unit

Source-level **paired bootstrap**, 2000 resamples, seed 20260813, 95%
percentile interval. A resample draws **sources**; all five preference branches
of a source move together. Preference branches are repeated measures, not
independent examples.

Wins/losses/ties are reported at source level. **No p-value is emitted for any
statistic whose sign is guaranteed** — `verified_pref >= greedy_pref` in
scalarized landing value follows from policy improvement, so only the
**magnitude** of that difference is admissible, and it must be read against the
scale of the total post-start movement.

## 8. Contrast parity table — every contrast varies exactly one dimension

Parity is mandatory. A parity audit in the main lane found a contrast
(`verified_retarget` vs `continue_A`) that varied **controller and objective**
at once and inflated its effect by roughly 11%.

| id | status | question | arm | base | controller | start | budget | objective | **varies** |
|---|---|---|---|---|---|---|---|---|---|
| **P1** | **CONTEXT_ONLY** | unguided floor | `greedy_pref` | `unguided` | greedy / **none** | `x_0` = `x_0` | H=6 = H=6 | `s(.\|w)` / **none** | **controller + objective** |
| **P2** | PRIMARY | does future-awareness help? | `verified_pref` | `greedy_pref` | verified / greedy | `x_0` = `x_0` | matched native oracle calls | `s(.\|w)` = same | **controller** |
| **P3** | PRIMARY | closed loop vs generate-then-rank | `greedy_pref` | `gen_rank@greedy` | closed / open loop | `x_0` = `x_0` | matched native oracle calls | `s(.\|w)` = same | **controller** |
| **P4** | PRIMARY | closed loop vs generate-then-rank, future-aware | `verified_pref` | `gen_rank@verified` | closed / open loop | `x_0` = `x_0` | matched native oracle calls | `s(.\|w)` = same | **controller** |
| **P5** | PRIMARY | do different preferences give different futures? | `greedy_pref @ w=0.9` | `greedy_pref @ w=0.1` | greedy = greedy | `x_0` = `x_0` | H=6 = H=6 | w=0.9 / w=0.1 | **objective** |
| **P6** | PRIMARY | same realized prefix, preference-dependent futures | `branch @ w_i` from `x_3` | `branch @ w_j` from `x_3` | greedy = greedy | `x_3` = `x_3` | H-3 = H-3 | w_i / w_j | **objective** |

**P1 is confounded and is labelled so.** An earlier draft of this table claimed
P1 varied only the controller. `scripts/pareto_instrument_gate.py` rejected it:
an arm with no controller cannot have an objective either, so `greedy_pref` vs
`unguided` unavoidably varies **both**. That is the same shape as the confound
the main lane's audit found in `verified_retarget` vs `continue_A`, which
inflated its effect ~11%. P1 is therefore kept as the floor, reported with the
confound stated out loud, and **may not carry a headline**. The parity-clean
forms of the same question are **P3** (closed vs open loop, same objective) and
**P5** (objective only).

The gate caught this before any run, which is the only reason it is a
documentation note rather than a retraction.

P2 is deliberately **not** `verified_pref` vs `unguided`: that would vary
controller and objective together and reproduce the main lane's confound
exactly.

## 9. Instrument discipline — executable

`scripts/pareto_instrument_gate.py` runs before any statistic is reported and
fails the run rather than the reader. It encodes the five defects already caught
in this project, each of which was a quantity whose sign was fixed before any
data existed:

| id | defect it prevents | check |
|---|---|---|
| **D1** | an action selected by `argmax V_G` then scored by `V_G` | the selection function and the scoring function of any reported advantage statistic must be distinct registered objects |
| **D2** | a "verified" arm that was secretly greedy | the two arms' committed action sequences must differ on at least one source, else the run is `INVALID_INSTRUMENT` |
| **D3** | a sign test against a null that a theorem makes false | statistics on the guaranteed-sign registry may report magnitude only; requesting a p-value for one is an error |
| **D4** | a fabricated SHA-256 in a manifest | every hash in `handoff.json` is recomputed from the file on disk |
| **D5** | the parity confound | every declared contrast must vary exactly one of {controller, start, budget, objective} |
| **D6** | hypervolume inflated by generating more molecules | endpoint counts must be equal across the two arms of **every** HV contrast; compute parity is enforced only on the **one axis** a contrast claims (`kernel` ratio <= 1.25, or `native` gap <= 10%), and the unclaimed axis is reported |

Additionally, every reported statistic must declare its **falsifying range** —
the values it could take if the hypothesis were false. A statistic with an empty
falsifying range is not a measurement and the gate refuses to emit it.

## 10. Stop rules

- **Gate fails on all three predeclared pairs** → report that target-free Pareto
  control is not supported by this objective language, and hand the choice of a
  fourth pair back to main. Do **not** invent a fourth pair in this lane after
  seeing census results.
- **`greedy_pref` already covers the front** → preference control works,
  future-aware control adds nothing here. Report it; do not force a controller
  claim. This is the retargeting outcome and it is not damaging.
- **`gen_rank` matches the closed-loop arms at matched budget** → the sequential
  claim fails; the preference-responsiveness claim (P5/P6) may still stand.
- **All five preferences produce the same endpoint for every arm** →
  `INVALID_INSTRUMENT` for the front claim; the census gate should have caught
  this and did not, which is itself reportable.
- **Feasibility below 0.8 for any arm** → the budget or the shortlist is wrong;
  stop and report rather than retune.

## 11. Allowed calibration vs forbidden adaptation

**Allowed (held-in, once):** the shortlist size `S`; the `gen_rank` trajectory
count `K` needed to hit budget parity; the number of census sources; the
normalizer `centre_S`/`s_S` under the predeclared recipe.

**Forbidden:** changing any frozen objective constant; changing the clip;
reordering the pair list; shrinking the budget until a controller wins;
selecting the preference grid after seeing coverage; retraining anything;
training an `h_phi`; opening `reserve_source_keys`.

## 12. Scope boundary

This lane stops **before any held-out run**. Stage 0 census and Stage 1
implementation with local tests only. The held-in smoke is costed in
`HANDOFF.md` and requires main-lane authorization before launch.
