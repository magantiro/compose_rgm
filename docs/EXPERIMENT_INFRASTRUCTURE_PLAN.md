# Experiment infrastructure plan v2 (E1–E7)

Build plan for the paper's experimental program, executed while the RingCore-V1 editing prior trains.
Experiment specs live in `configs/experiment_registry.yaml`; this document is the *build* plan.

**v2 incorporates the owner review's five required corrections.** v1 is commit `3cdc56d`; the corrections are
recorded in §2 rather than silently applied, so the revision is auditable.

Status legend: **[now]** buildable without the trained checkpoint · **[frozen-gated]** needs the selected
editing checkpoint · **[lane]** parallel de-novo lane with its own gates.

---

## 0. Ground rules

1. **Nothing touches the live run.** `compose-v4-ringcore-v1-scientific-a7546e2-v1` is pinned to commit
   `a7546e2` in its own clean worktree. Do not modify `scripts/train_tracelet_cnof_gate.py`,
   `modal_apps/train_tracelet_gm.py`, `scripts/ring_core_identity.py`,
   `configs/ringcore_v1_production.json`, or `src/compose_v4/data/**`. `_source_fingerprint()` hashes `src/`,
   `scripts/`, `recipes/` and the Modal app, so edits there change run identity; `configs/`, `docs/`,
   `diagnostics/`, `tests/`, `results/` are outside the fingerprint.
2. **From scratch.** No reuse of `scripts/exact_doob_enumerable_benchmark.py`,
   `scripts/doob_guidance_ground_truth.py`, `scripts/e0_toy_h_exactness.py`; the untracked
   `diagnostics/exactness/exact_doob_enumerable_cap{4,5}.json` are **non-evidence**.
3. **One evaluator.** No experiment reconstructs successor probabilities. Enforced by test, not intent.
4. **The evaluator validates, not just reports.** Capability flags are compared against the checkpoint's
   scientific contract *and* the experiment registry; disagreement fails the run.
5. **Atomic commits**; subject `<module>: <one-line imperative, lowercase, no trailing period>`; never
   mention Claude/AI in commits or PR bodies.
6. **Gates.** `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src python3 -m pytest tests/ -q` and
   `./.venv/bin/ruff check` (ruff is NOT on PATH). Read the SKIP count, not just PASS.
7. **No result claims before the checkpoint is selected.** Harnesses may be complete while their numbers are
   unproduced.
8. **Test isolation.** No test-set oracle output is used to train, tune, or select anything.

---

## 1. Verified facts this plan rests on

| Fact | Evidence |
|---|---|
| Loader reconstructs vocab **15** + `ring_restates/cyclic_graft/heteroatom_scan/ring_opening=True` from `organic_vocabulary=True, corrupted_prior_mix=True` | ran on `_bedit_ckpt.pt` |
| Loader reconstructs vocab **4** + all capability flags **False** when those keys are absent | ran on `_denovo_ckpt.pt` |
| `expected_scope_hash` **fails loudly** when the payload has no `corpus_scope_hash` | raises `ValueError` |
| `enable_cycle_ops` is **False** in both local fixtures — production sets it | neither payload has `cycle_op_mix` |
| Snapshots every 500 steps as `checkpoint.step<N>.pt`, each a full `exact_training_recovery` payload | `_step_snapshot_path`; live run at step 500 |
| Loader already accepts `exact_training_recovery` payloads | explicit branch |
| Canonical aggregation exists with loop reference **and** vectorized path, per `(example, successor)` pair | `segmented_successor.py` |
| **Doob prunes but never creates support**; equality iff next-state value is strictly positive; kernel undefined where `h_k(x)=0` | numerical check, §2.2 |
| **Mark-level power tilt is not refinement-invariant**: aggregate mass 0.500 vs 0.200 at β=2, 0.500 vs 0.667 at β=0.5, equal only at β=1 | numerical check, §2.7 |

**Consequence:** the fixtures cover the vocab-width and capability-flag branches but **not**
`enable_cycle_ops=True`. A1 needs a dedicated test constructing that branch directly, and must not claim
coverage it lacks.

---

## 2. Owner review: required corrections (all accepted)

### 2.1 Protocol freezing hoisted to Phase A0
B2 (oracle/task protocol) and B3 (dynamic protocols) move **before** any E2–E7 output is inspected, in
parallel with A1. Definitions may carry a placeholder for the selected checkpoint hash; everything else is
content-hashed first.

### 2.2 Doob support statement corrected — v1 was mathematically wrong
v1 said the transform "neither creates nor destroys support." For
`P^g_k(x,y) = P_k(x,y) · h_{k+1}(y) / h_k(x)` the correct statement has **four** parts:

1. **Always:** `supp P^g_k(x,·) ⊆ supp P_k(x,·)`. No transition absent from the base kernel is ever created.
2. **Equality** iff `h_{k+1}(y) > 0` for every `y ∈ supp P_k(x,·)`; sufficient condition is strictly positive
   desirability on reachable terminal states with all reachable backward values positive.
3. **Hard conditioning prunes:** support is restricted to transitions from which the target event remains
   reachable within the remaining budget. Verified: base `[0.5,0.3,0.2]` with `h_{k+1}=[1,0,0.4]` gives
   controlled `[0.862,0,0.138]` — three base edges become two.
4. **Undefined where `h_k(x)=0`** (0/0); the controlled chain lives on `{x : h_k(x) > 0}`.

Propagates to: theorem wording, A5.4, acceptance criteria, tests, paper prose.

### 2.3 Learned-controller training added as a workstream (§5)
v1 defined the `learned_value` *interface* but never said how the value model is trained — the largest
omission. New Phase B1.5.

### 2.4 Development vs final task artifacts (§7 A3)
Two immutable artifacts: a **development** set from the validation partition (debugging, controller
training/selection, thresholds, preliminary plots, failure analysis) and a **final** set from the test
partition generated by the already-frozen algorithm, used once.

### 2.5 De-novo promoted to a parallel critical path
Unconditional generation is a spotlight gate, not an appendix. Preparation starts now in a separate worktree;
training still waits for its own gates.

### 2.6 Exact benchmark = K-step reachable graph, not the full bounded state space
`S_K(x_0) = {x : reachable from x_0 in ≤ K steps}`, expanded breadth-first with canonical merging. Two-stage:
solver verified now on a simple fixed kernel, then instantiated on the *same* graph with the selected learned
checkpoint. Plus independent verification (DP vs explicit path enumeration) on the smallest slice.

### 2.7 Mark-level negative result added (A4.7)
Successor-level control is refinement-invariant; mark-level power/top-k/nucleus/family controls are not.
Verified numerically above. This sharpens the quotient contribution from "our aggregation is consistent" to
"successor-level is the only level at which these controls are well-defined."

### 2.8 Metric hierarchy, compute accounting, semantic panel IDs
1–3 preregistered primary metrics per experiment (§6); full compute accounting for controller fairness (§5.4);
semantic panel IDs instead of hard-coded figure numbers (§6.2).

---

## 3. DELIVERABLE 1 — updated dependency DAG

```
PHASE A0  (freeze first, no model output inspected)
  A0.1 objective + oracle registry (identities, versions)
  A0.2 benchmark protocol: sources, normalization bounds, similarity floors,
       edit + oracle budgets, constraint definitions, seeds, statistical units
  A0.3 multi-objective predeclarations: HV reference points, dominance/epsilon,
       diversity measure, source-similarity floor, 2/3/4-5-objective task sets
  A0.4 dynamic protocols: switch times, restart variants, regret definitions
  A0.5 primary/secondary metric table (§6)
  A0.6 compute-accounting schema (§5.4)
  A0.7 semantic panel IDs (§6.2)
  A0.8 exact-sizing policy, preregistered (§4)
  A0.9 task-generation algorithm + config hash (consumed by A3)
        │
        v
A1.0  FREEZE THE SHARED KERNEL PROTOCOL          <-- gates BOTH branches below
      CanonicalSuccessorKernel: the single definition of the molecular process.
      A2 may proceed in parallel ONLY against this contract, never around it.
        │
        ├─────────────────────────────────────────────┐
        v                                             v
PHASE A1  shared kernel infrastructure          PHASE A2  exact control
  A1.1 evaluator skeleton, provenance,            A2.1 sizing sweep (uses A0.8)
       versioned schema, contract validation      A2.2 K-step reachable graph
  A1.2 PRODUCTION implementation of the           A2.3 exact finite-horizon solver
       protocol: enumeration + canonical          A2.4 independent DP-vs-path check
       aggregation                                A2.5 support theorem tests (§2.2)
  A1.3 CLI + registry-driven metrics              A2.6 tilts
  A1.4 single-evaluator enforcement test          A2.7 controller arms vs exact optimum
  A1.5 uniform canonical-successor baseline            │
       (a PROBABILITY WRAPPER over the protocol,       ├── A4.6 quotient check 6
        not a second enumeration)                      └── A4.7 mark-level counterexample
  A1.6 development task generator (A0.9)               │
  A1.7 quotient checks 1-5                             │
        │                                              v
PHASE B  controller system
  B1   same-base controller interface (+ compute accounting)
  B1.5 learned value/controller training pipeline (§5)   <-- needs A2 for calibration
  B1.6 controller checkpoint selection on controller-validation only
        │
        v
PHASE C  post-training selection
  C1 leaderboard over checkpoint.step<N>.pt   C2 predeclared selection rule
  C3 freeze editing prior                     C4 low-cost diagnostics gate
        │
        v
PHASE D  final runs  [frozen-gated]
  final task artifacts generated from frozen algorithm -> E2, E3, E4, E7
  A2 re-instantiated with the selected checkpoint (stage 2 of §2.6)

PARALLEL LANE (starts now, own gates)          PHASE F  artifacts
  L1 de-novo bond_reorder repair                 F1 external baseline environments
  L2 timed-CTMC hazard semantics                 F2 immutable result schemas
  L3 broad-organic reference distribution        F3 plot/table generators
  L4 de-novo source process + manifest/contract  F4 manuscript mapping by panel ID
  L5 unconditional metrics + baselines
  L6 de-novo gate -> training -> E1
```

**Critical-path note.** A0 gates everything that produces numbers. B1.5 needs A2 for its calibration target.

**Parallelism correction (owner review).** v2 originally said A2 "needs a kernel, not *the* kernel." That
phrasing was loose and would have licensed exactly the duplication this plan exists to prevent. The rule is:

> A2 may proceed in parallel **only against the frozen `CanonicalSuccessorKernel` protocol (A1.0)**, testing
> against a fixture implementation of it. A2 must **not** independently implement legal mark enumeration,
> canonicalization, mark-to-successor aggregation, capability resolution, or slot masking. When A1.2 lands,
> the production evaluator supplies the implementation and A2 switches to it with no change to A2's code.

Same rule for A1.5: the uniform baseline is a *probability wrapper* over the protocol, not a second
enumeration. Enforced by A1.4's scan test.

---

## 4. DELIVERABLE 2 — exact-sizing policy (preregistered)

Frozen **before** any sizing result is used, so the exact benchmark cannot be chosen because one candidate
gives nicer controller numbers.

**Enumeration target.** `S_K(x_0)` = states reachable from source `x_0` in ≤ K steps under the **real
production executor**, expanded breadth-first, identical molecular states merged by canonical key. Record
per-depth layer sizes and branching factors. Not the full bounded molecular space.

**Operator set.** RingCore-V1: compositional `cycle_close`/`cycle_open` (executor `bond_insert`/`bond_delete`,
scored as families `cycle_insert`/`cycle_attach`); `ring_system_grow` macro **disabled**.

**Acceptance window (owner-approved).** State count alone is *not* a sufficient tractability criterion — a
20,000-state graph with enormous branching is harder than a larger sparse one — so the window bounds both
states and edges:

```yaml
exact_sizing:
  N_min: 500          # large enough not to be a hand-written toy
  N_max: 20000        # sparse exact DP stays manageable
  E_max: 2000000      # max distinct canonical DIRECTED edges in the reachable graph
  selection: smallest_qualifying_candidate
```

Per-budget transition operators stored sparsely, under a hard memory cap.

**Non-degeneracy conditions (all required).** A qualifying candidate must contain:
1. at least one atom-**birth** transition;
2. at least one atom-**death** transition;
3. at least one **cycle-changing** transition;
4. at least **two distinct atom counts** represented;
5. at least **two distinct cycle-rank values** represented;
6. **at least one terminal event reachable by multiple distinct molecular paths.**

Condition 6 is load-bearing for the control comparison: with a single route to each target, several controllers
become indistinguishable and the panel would measure nothing.

**Selection rule (preregistered).** Choose the **smallest** candidate satisfying the state window, the edge
cap, and all six non-degeneracy conditions. Ties broken by smaller `|S_K|`, then by smaller edge count, then
lexicographically by candidate id. Rationale for "smallest": for the exact panel, computational exactness and
auditability matter more than chemical realism — the full-scale experiments supply the realism.

**Shrinking sequence**, applied in this order until the window is met:
1. lower heavy-atom cap; 2. lower edit budget K; 3. restrict element vocabulary; 4. simpler source molecule.

**Fallback.** If no single feasible graph contains both cardinality and cycle behaviour, build **two** exact
benchmarks — a trans-dimensional slice and a cycle/topology slice — rather than omitting either capability or
reporting an intractable graph. If even the smallest candidate exceeds the cap, report **infeasible**; never
approximate and call it exact.

**Reporting.** Every candidate evaluated is reported with its layer sizes, whether it met the window, and why
it was or wasn't selected — so the selection is auditable and no silent search happened.

---

## 5. DELIVERABLE 3 — learned controller training plan (Phase B1.5)

### 5.1 Object
```
h_φ(b, x; x_src, z, m) ≈ E_{P_θ}[ g_z(X_K) | X_{K-b} = x ]
```
`b` remaining budget · `x` current molecule · `x_src` source · `z` goal descriptor · `m` protected mask.
Base kernel `P_θ` is the **frozen** editing prior; the controller never fine-tunes it.

### 5.2 Targets (both implemented; the exact benchmark decides)
1. **Monte-Carlo terminal desirability** under base-prior rollouts from `(b, x)`.
2. **Bootstrapped backward target** `h_φ(b,x) ← Σ_y P_θ(y|x) h_φ(b-1,y)`, using the canonical-successor
   kernel from A1 — never a mark-level surrogate.

Selection between them is made on controller-validation calibration and backward-residual, not on test tasks.

### 5.3 Architecture, loss, data

**Encoder policy (owner decision): `frozen_base_with_trainable_adapter`.** The molecular encoder from the
selected editing prior stays **frozen**:

```
e_x = E_θ(x),   e_s = E_θ(x_src)          E_θ frozen
h_φ = H_φ(e_x, e_s, e_x - e_s, b, z, m)   H_φ trained
```

Trained components only: remaining-budget embedding · objective/preference embedding · protected-mask /
constraint embedding · source-vs-current comparison module · a small residual adapter · the value head.

Why this is the primary configuration: the base successor kernel stays genuinely fixed across controllers, so
learned control cannot improve by silently altering the molecular prior; exact-vs-learned discrepancies are
attributable to *value approximation* rather than a changed process; and it reduces overfitting to a small
rollout-value dataset. Goal, budget and source information still enter through trainable modules.

**Secondary ablation only:** `copied_encoder_top_block_adaptation` — a separately copied, lightly adapted
encoder. It must **never write back into the base generator**. Report whether adaptation improves value
calibration, but the frozen-encoder controller remains the main same-base causal test.
- Regression in log-desirability with an explicit non-negativity/positivity treatment, since `h=0` is exactly
  the support-pruning case in §2.2 and must be representable.
- **Three disjoint splits:** controller-train goals/sources · controller-validation goals/sources · final test
  sources and objective schedules. No test oracle output touches training or selection.

### 5.4 Validation metrics and compute accounting
Validation: value calibration · backward-equation residual · successor ranking quality · **terminal-law TV
versus exact Doob on the enumerable graphs** · reachability-event calibration · control quality versus the
exact optimum · robustness across budgets and goals.

Compute accounting recorded for **every** controller arm, because "same prior and budget" is insufficient when
methods score different numbers of successors or particles:

| counted separately | why |
|---|---|
| property-oracle calls | the primary cost axis for E7 |
| model forward passes | learned controllers pay here, reranking does not |
| successors scored | reranking/greedy inflate this |
| particles propagated | SMC inflates this |
| accepted edits | committed-state count |
| wall-clock, GPU-hours | reported separately, never as the primary axis |

Primary E7 curve is **hypervolume versus all oracle calls**. Counting committed states for one method while
ignoring evaluated particles or reranked candidates for another is explicitly disallowed.

---

## 6. DELIVERABLE 4 — primary metric table

One to three preregistered primary metrics per experiment; everything else is a diagnostic.

| Exp | Primary (load-bearing) | Secondary / diagnostic |
|---|---|---|
| **E1** unconditional *(owner-decided)* | `frechet_chemnet_distance`; `held_out_chemical_space_recall`; `ring_topology_distribution_distance` | novelty, internal diversity, nearest-neighbour similarity, memorization rate, size + physicochemical distributions, throughput, scaffold novelty, precision |
| **E2** learned transport | held-out canonical-successor NLL; target/analogue recovery **at matched edit budget**; path overhead vs shortest/compiler path | reversals, repeated states, net vs gross displacement, endpoint fidelity, diversity |
| **E3** cardinality adaptation | target atom-count success; edits/oracle calls to success; retained source similarity | overshoot/correction, steps to enter interval, validity, connectedness, per-step cardinality trace |
| **E4** topology adaptation | target topology success; valid-path rate; edit efficiency | cycle rank, ring-system count, ring-size distribution, ring-op utilization, unnecessary open/close, final property/similarity |
| **E5** quotient invariance | successor-mass invariance residual under slot relabeling and action refinement (exact to tolerance); **mark-level divergence demonstrated** | raw mark multiplicity, sampled-frequency agreement at declared n and seed |
| **E6** exact control | terminal-law TV to the analytic target | support-subset conformance, per-arm regret vs exact optimum, backward-value residual |
| **E7** multi-objective / dynamic | normalized hypervolume; **HV-AUC versus all oracle calls** | IGD+, feasible count, source similarity, diversity, held-out evaluator performance, adaptation regret, prefix reuse |

### 6.1 E1 specification (owner-decided, expanded)

**Three primary metrics, no more.**

1. **FCD against the held-out test distribution** — the primary global distribution-fidelity measure.
   Protocol: 10,000 molecules per seed · five fixed seeds · compared against a fixed held-out test sample of
   **equal size** · report mean, standard deviation **and every individual seed** · identical sample count for
   every baseline.
2. **Held-out chemical-space recall / coverage** — distinguishes genuine coverage from a narrow high-precision
   mode. **One frozen implementation** (recall/coverage in ChemNet or another declared molecular embedding);
   the definition may not be switched after results arrive.
3. **Ring/topology distribution distance** — required because RingCore is a defining architectural
   contribution, so unconditional generation must reproduce real topological diversity. Frozen topology vector:
   cycle rank · ring-system count · ring-size histogram · fused-ring incidence · spiro incidence · bridged
   incidence (where representable) · aromatic vs non-aromatic ring fractions. Preregistered aggregate: mean
   Jensen–Shannon divergence across categorical features **plus** Wasserstein distance for ordered count
   variables.

**Hard gates — required but NOT headline metrics** (a new tier this plan did not previously have):
endpoint validity · all-step validity · connectedness · uniqueness · basic sanitization success.
These must pass, but validity is largely guaranteed by the executor, so headlining it would undersell the
paper. Recorded as gates, reported as gates.

**Secondary diagnostics:** novelty · internal diversity · nearest-neighbour similarity · memorization rate ·
molecular-size distribution · physicochemical-property distributions · throughput · scaffold novelty · precision.

### 6.2 Semantic panel IDs (no hard-coded figure numbers)
`GEN_UNCONDITIONAL` · `TRANSPORT_LEARNED_VS_UNIFORM` · `CARDINALITY_ADAPTATION` ·
`TOPOLOGY_ADAPTATION` · `EXACT_TERMINAL_TILT` · `QUOTIENT_INVARIANCE` · `PARETO_SAME_BASE` ·
`DYNAMIC_SWITCH` · `PARETO_FAN` · `PATHWISE_CONSTRAINTS`

Result artifacts carry the panel ID; the manuscript build maps IDs onto whatever final layout fits the page
budget, so merging exact and dynamic panels needs no code change.

---

## 7. DELIVERABLE 5 — revised registry schema (v2)

`configs/experiment_registry.yaml` moves `schema_version: 1 → 2`. Additive except where noted.

```yaml
schema: compose.experiments.registry
schema_version: 2

protocol:                       # extended
  successor_kernel: canonical
  editing_budget_convention: fixed_step_embedded_jump_chain
  held_out_policy: <ringcore-v1 scaffold key, 0.9/0.05/0.05, verified disjoint>
  similarity: {fingerprint: ECFP4, radius: 2, bits: 2048, metric: tanimoto}
  seeds: [0, 1, 2, 3, 4]
  normalization_policy: declared_before_any_model_output_inspected
  independent_evaluation_oracle: required_where_available
  # --- new in v2 ---
  protocol_freeze:              # A0
    frozen_at_commit: <sha>
    content_hash: <hash of the frozen protocol block>
    # Provenance ONLY. Record the ACTUAL highest snapshot at freeze time -- the run is past step 1500, so
    # writing 0 here would be false. The structural guarantee does not depend on this number:
    #   "No checkpoint outputs or rollout results were inspected in selecting tasks, thresholds,
    #    objectives or reference points."
    # Enforced by construction: the task builder accepts NO checkpoint argument at all.
    highest_existing_snapshot_step: <int, measured at freeze>
  hard_gates:                   # E1 tier: required, never headline (§6.1)
    [endpoint_validity, all_step_validity, connectedness, uniqueness, sanitization_success]
  objectives:                   # A0.1
    - {id, definition, oracle_id, oracle_version, normalization_bounds, direction}
  multi_objective:              # A0.3
    hypervolume_reference_points: {<task_id>: [...]}
    dominance_convention: <pareto|epsilon>
    epsilon: <float|null>
    diversity_measure: <id>
    source_similarity_floor: <float>
    task_arities: {two: [...], three: [...], stress: [...]}
  dynamic:                      # A0.4
    switch_steps: [...]
    restart_variants: [continuation, restart_from_source, restart_under_separate_sampler,
                       static_compromise]
    regret_definition: <id>
  compute_accounting:           # A0.6 -- all required per arm
    [oracle_calls, forward_passes, successors_scored, particles_propagated,
     accepted_edits, wall_clock_seconds, gpu_hours]
  task_generation:              # A0.9
    algorithm_id: <id>
    config_hash: <hash>
    development_partition: validation
    final_partition: test
    preregistered_exclusions: [invalid_source, oracle_failure, unreachable_under_claimed_support]

experiments:
  E<n>:
    title, question, maps_to, priority
    checkpoint: none | any | editing_prior | denovo
    status: READY | BLOCKED | FROZEN_GATED
    # --- changed in v2: primary/secondary split replaces a flat metric list ---
    primary_metrics: [...]        # 1-3, load-bearing
    secondary_metrics: [...]      # diagnostics
    panel_id: <semantic id>        # replaces `panel: "Figure 2 (size)"`
    task_artifacts:                # new
      development: configs/tasks/<id>.development.json
      final: configs/tasks/<id>.final.json
    arms: [...]           baselines: [...]        ablations: [...]
    budgets: {...}        failure_criterion: <str>
    outputs: [...]        per_step_records: [...]
    support_caveats: [...]   # e.g. spiro/bridged marked unsupported if unreachable

controllers:                    # new top-level, directive item 8
  - id, kind, requires_checkpoint, oracle_uses, compute_profile
learned_controller:             # new top-level, §5
  target_object, target_estimators, splits, architecture, loss,
  validation_metrics, selection_rule, budget
post_training_queue:            # new top-level, directive item 13
  snapshot_glob: "checkpoint.step*.pt"
  steps: [wait_for_artifact, verify_provenance, production_weighted_successor_nll,
          balanced_family_nll, update_leaderboard, apply_selection_rule, freeze_winner,
          low_cost_diagnostics, full_suite]
  selection_rule: <predeclared>
dag: [...]                      # §3
```

Breaking change: `metrics`/`panel` → `primary_metrics`/`secondary_metrics`/`panel_id`. A loader test will
fail on a v1 registry rather than silently misread it.

---

## 8. DELIVERABLE 6 — implementation order

| # | Piece | Phase | Gate |
|---|---|---|---|
| **0** | **A1.0 freeze the `CanonicalSuccessorKernel` protocol + fixture + duplication test** | A1 | **[now]**, required by owner review before piece 1 |
| 1 | A0.1–A0.9 protocol freeze + registry v2 | A0 | needs 0; blocks all numbers |
| 2 | A1.1 evaluator skeleton, provenance, versioned schema, contract validation | A1 | needs 0 |
| 3 | A2.1 sizing sweep under the preregistered policy | A2 | needs 0; parallel with 2 |
| 4 | A1.2 production implementation of the protocol (enumeration + canonical aggregation) | A1 | needs 2 |
| 5 | A2.2–A2.3 K-step graph + exact solver on a fixed kernel | A2 | needs 3 |
| 6 | A1.3–A1.4 CLI, registry-driven metrics, single-evaluator enforcement | A1 | needs 4 |
| 7 | A2.4–A2.5 independent DP-vs-path check + support theorem tests | A2 | needs 5 |
| 8 | A1.5 uniform canonical-successor baseline harness | A1 | needs 4 |
| 9 | A1.6 development task generator (validation partition) | A1 | needs A0.9 |
| 10 | A1.7 quotient checks 1–5 | A1 | needs 4 |
| 11 | A2.6–A2.7 tilts + controller arms vs exact optimum | A2 | needs 7 |
| 12 | A4.6–A4.7 quotient check 6 + mark-level counterexample | A2/A4 | needs 11 |
| 13 | B1 same-base controller interface + compute accounting | B | needs 6, A0.6 |
| 14 | B1.5 learned value/controller training | B | needs 13, 11 |
| 15 | B1.6 controller selection on controller-validation | B | needs 14 |
| 16 | C1–C4 leaderboard, selection rule, freeze, diagnostics gate | C | needs 6 |
| 17 | Final task artifacts from the frozen algorithm | D | needs A0.9, 16 |
| 18 | E2, E3, E4, E7 final runs; A2 stage-2 with selected checkpoint | D | **[frozen-gated]** |
| 19 | F1–F4 external baselines, result schemas, generators, panel mapping | F | needs 18 |
| L | L1–L6 de-novo lane, **parallel from now** | lane | own gates; E1 after L6 |

De-novo lane (L) runs in its own worktree concurrently with 1–18 and never blocks on them.

---

## 9. Owner decisions — RESOLVED, now preregistered

```yaml
E1_primary_metrics:                       # §6.1
  - frechet_chemnet_distance              # 10k mols x 5 seeds vs equal-size held-out sample
  - held_out_chemical_space_recall        # one frozen implementation, never re-defined post hoc
  - ring_topology_distribution_distance   # frozen topology vector, mean JSD + Wasserstein

exact_graph:                              # §4
  N_min: 500
  N_max: 20000
  E_max: 2000000                          # distinct canonical directed edges
  selection: smallest_qualifying_candidate
  non_degeneracy: [atom_birth, atom_death, cycle_change, two_atom_counts,
                   two_cycle_ranks, multi_path_terminal_event]

learned_controller:                       # §5.3
  encoder_policy: frozen_base_with_trainable_adapter
  secondary_ablation: copied_encoder_top_block_adaptation   # never writes back to the base
```

These are frozen as of this commit. Changing any of them after a corresponding result is inspected would
defeat the purpose of A0, so revisions must be recorded as an explicit contract update with a reason — the same
discipline applied to the scheduler hash supersession.

## 10. Remaining open item

**Nothing blocking.** One judgement call is deferred to its own piece rather than guessed now: the exact
`held_out_chemical_space_recall` implementation must be *named and frozen* during A0.1 (candidate: ChemNet
embedding recall). It is listed as a decision inside A0, not left implicit.
