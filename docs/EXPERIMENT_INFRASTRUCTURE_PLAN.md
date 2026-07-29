# Experiment infrastructure plan (E1–E7)

Build plan for the paper's experimental program, executed while the RingCore-V1 editing prior trains.
Authoritative experiment specs live in `configs/experiment_registry.yaml` (commit `310e5b7`); this document
is the *build* plan for the harnesses that registry describes.

Status legend: **[done]** committed · **[now]** buildable without the trained checkpoint · **[frozen-gated]**
needs the selected editing checkpoint · **[lane]** separate de-novo lane with its own gates.

---

## 0. Ground rules

1. **Nothing touches the live run.** The training run `compose-v4-ringcore-v1-scientific-a7546e2-v1` is
   pinned to commit `a7546e2` in its own clean worktree. Do not modify
   `scripts/train_tracelet_cnof_gate.py`, `modal_apps/train_tracelet_gm.py`, `scripts/ring_core_identity.py`,
   `configs/ringcore_v1_production.json`, or `src/compose_v4/data/**`. Note `_source_fingerprint()` hashes
   `src/`, `scripts/`, `recipes/` and the Modal app, so edits there would change run identity; `configs/`,
   `docs/`, `diagnostics/`, `tests/` and `results/` are outside the fingerprint and safe to commit.
2. **From scratch.** Per owner instruction, the exact-control work does **not** reuse
   `scripts/exact_doob_enumerable_benchmark.py`, `scripts/doob_guidance_ground_truth.py`, or
   `scripts/e0_toy_h_exactness.py`, and the untracked `diagnostics/exactness/exact_doob_enumerable_cap{4,5}.json`
   are treated as **non-evidence**. No imports from those modules.
3. **One evaluator.** No experiment reconstructs successor probabilities. Enforced by a test, not intent (A1.4).
4. **Atomic commits**, subject `<module>: <one-line imperative, lowercase, no trailing period>`. Never mention
   Claude/AI in a commit message or PR body.
5. **Gates.** `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src python3 -m pytest tests/ -q` and
   `./.venv/bin/ruff check` (ruff is NOT on PATH). `src/` stays ruff-clean. Read the SKIP count, not just PASS.
6. **No result claims before the checkpoint is selected.** Harnesses may be complete while their numbers
   remain unproduced; the plan says so explicitly per piece rather than implying results.

---

## 1. Verified facts this plan rests on

Measured, not assumed:

| Fact | Evidence |
|---|---|
| `load_factorized_rollout_checkpoint` reconstructs vocab **15** + `enable_ring_restates/cyclic_graft/heteroatom_scan/ring_opening = True` from `organic_vocabulary=True, corrupted_prior_mix=True` | ran on `_bedit_ckpt.pt` |
| Same loader reconstructs vocab **4** + all capability flags **False** when those keys are absent | ran on `_denovo_ckpt.pt` |
| `expected_scope_hash` **fails loudly** when the payload carries no `corpus_scope_hash` | raises `ValueError: checkpoint corpus_scope_hash None != expected …` |
| `enable_cycle_ops` is **False** in both local fixtures | neither payload has `cycle_op_mix`; the production run sets it |
| Snapshots are written every 500 steps as `checkpoint.step<N>.pt`, each a full `exact_training_recovery` payload | `_step_snapshot_path`; dead run wrote step500/1000/1500 |
| The loader already accepts `exact_training_recovery` payloads | explicit branch in the loader |
| The live run has written **0** checkpoints as of writing | volume listing |
| Canonical aggregation exists with a loop reference **and** a vectorized path, aggregating per `(example, successor)` pair | `src/compose_v4/model/segmented_successor.py` |

**Consequence for A1:** the two local fixtures cover the vocab-width and capability-flag branches, but **not**
the `enable_cycle_ops=True` branch — that is only exercised by the production checkpoint. A1 must therefore
carry a test that constructs a model with cycle ops on directly, and must not claim coverage it lacks.

---

## 2. Adversarial findings against an earlier draft of this plan

Recorded so the corrections are auditable.

1. **Dependency error.** Quotient check 6 ("exact control on the quotient is invariant to encoding
   refinement") requires the exact solver. Fixtures were ordered *before* the solver. → A5 precedes A4;
   checks 1–5 may land earlier.
2. **Overstated checkpoint independence.** Doob exactness is a property of the kernel algebra over the real
   state graph and needs no trained weights; the `learned_value` controller arm genuinely does. Stated per arm.
3. **"Enumerable" is empirical.** The reachable-set size under the real executor must be *measured* before the
   bounded chemistry is fixed. If it explodes there is no exact solver. Sizing is a deliverable (A5.1).
4. **Uniform baseline had no stopping rule** → path length, reversals and oracle cost were meaningless.
   It inherits the fixed-step embedded jump chain, applied identically to both arms.
5. **Baseline fairness is flag-sensitive.** Identical executor, identical capability flags, identical
   enumeration; only the probability assignment differs. A flag mismatch silently invalidates E2.
6. **"Frozen before the checkpoint" as a timestamp is fragile.** Replaced with a structural property: splits
   are built from corpus statistics with **no model output inspected**, provable by construction. The highest
   existing snapshot step at freeze time is recorded alongside (currently 0).
7. **Single-evaluator rule needed enforcement**, not intent → A1.4 test.
8. **Output schema needs versioning**, additive-only, else Phase B/D break Phase A consumers.
9. **Evaluator must be validated against a real payload**, not only a fresh model, or the vocab-width,
   capability-flag and scope-hash paths go untested.
10. **A2 is a harness, not a result.** Its metrics cannot produce numbers until the checkpoint freezes.

---

## 3. Dependency graph

```
A1 evaluator ──┬── A2 uniform baseline harness ──┐
               ├── A3 frozen task splits ────────┤
               └── A4 quotient fixtures (1-5)    ├── Phase D: E2, E3, E4, E7  [frozen-gated]
A5 exact graph + Doob ──┬── A4 check 6           │
                        └── E6 results           │
Phase B controller iface + oracle protocol ──────┘
Phase C post-training queue ── selects the checkpoint Phase D consumes
Phase E de-novo lane ── E1  [lane, own gates]
Phase F external baselines, figure artifacts
```

---

## Phase A — buildable now

### A1. Unified checkpoint evaluator  *(priority 2, directive item 2)* **[now]**

The single entry point every experiment consumes. Its value is that it exists **once**: divergent
reimplementations would make the paper's numbers incomparable.

**Files**
- `src/compose_v4/experiments/checkpoint_evaluator.py` (new)
- `src/compose_v4/experiments/evaluate_checkpoint.py` (new, CLI `python -m …`)
- `tests/test_checkpoint_evaluator.py` (new)

**CLI**
```
python -m compose_v4.experiments.evaluate_checkpoint \
    --checkpoint <path> --registry configs/experiment_registry.yaml \
    --experiment E2 --seed 0 [--expected-scope-hash 3721d69851110fdd] [--output <json>]
```

**Centralized responsibilities** (each exactly once, for all experiments)

| # | Responsibility | Implementation note |
|---|---|---|
| 1 | model construction | delegate to `load_factorized_rollout_checkpoint`; do **not** reimplement |
| 2 | capability configuration | report the flags actually set; `ring_system_grow` macro stays disabled |
| 3 | persistent slot state | mask with the real-element predicate; **never** slice `[:n_real_atoms]` |
| 4 | legal-successor enumeration | production enumerators only |
| 5 | canonical aggregation | `segmented_successor.py`; per `(example, successor)` pair |
| 6 | budget convention | fixed-step embedded jump chain; editing discards the hazard |
| 7 | oracle loading | interface here; concrete oracles frozen in Phase B |
| 8 | metrics | per-experiment, driven by the registry |
| 9 | provenance | checkpoint sha256, registry hash, capability flags, operator-registry hash, seed |
| 10 | output schema | versioned, additive-only |

**Namespace hazard.** Executor rule names and model family names are different namespaces:
`bond_insert`/`bond_delete` are executor names; the dense head scores them as `cycle_insert`/`cycle_attach`.
Use `_CYCLE_OP_EXECUTOR_TO_FAMILY`; never hand-map.

**Representability hazard.** `AtomInsert` is scoreable only with 0 neighbours (`grow_root`) or exactly 1
(`grow_connected`). ≥2 is outside production support.

**Sub-pieces / commit granularity**
- **A1.1** module skeleton + provenance + versioned output schema + registry loading; unknown experiment id
  fails loudly.
- **A1.2** enumeration + canonical aggregation; kernel sums to 1 per example; matches the loop reference.
- **A1.3** CLI + metrics dispatch driven by the registry.
- **A1.4** enforcement test: fails if any experiment script calls the rate model directly instead of routing
  through the evaluator (pattern mirrors the existing anti-concatenation scan test).

**Acceptance criteria**
- Loads both local fixtures, reporting vocab 15/flags-on and vocab 4/flags-off respectively.
- Loads a model constructed with `enable_cycle_ops=True` (fixtures do not cover it).
- `sum_y P(y|x) == 1` per example to float tolerance, on benzene/toluene/pyridine via the real chemistry stack.
- Aggregated log-probs equal `reference_successor_logprobs` exactly.
- Unknown experiment id, missing checkpoint, and scope-hash mismatch each fail loudly.
- No trained checkpoint required for tests.

**Risks.** Fixtures are `hidden_dim=32` toy models — they validate *plumbing*, not numerics. Real-checkpoint
numerics are validated in Phase D.

---

### A2. Uniform legal-successor baseline harness  *(priority 3, directive item 3)* **[now]** (numbers **[frozen-gated]**)

E2's control arm. Assigns **uniform probability over canonical molecular successors**, `1/|N(x)|`, *not* over
raw action marks — the registry encodes this and it is the whole point of the comparison.

**Files** `src/compose_v4/experiments/uniform_successor_baseline.py`, `tests/test_uniform_successor_baseline.py`

**Fairness contract (load-bearing).** Both arms share: the same executor, the same capability flags, the same
enumeration, the same step budget, the same seeds. The *only* difference is the probability assignment. The
harness asserts flag equality between arms and refuses to run on mismatch.

**Stopping rule.** Fixed-step embedded jump chain, identical for both arms. Without this, path length,
reversal counts and oracle-cost-per-endpoint are undefined for the baseline.

**Metrics** (all 11 from the directive): held-out canonical-successor NLL; analogue/target recovery;
productive molecular displacement; fraction of steps improving endpoint similarity/objective; path length;
path overhead; immediate reversals; longer cycles; net vs gross edits; endpoint distribution fidelity;
oracle calls per successful endpoint.

**Acceptance.** Uniform mass is over distinct canonical successors (a molecule reachable by 5 marks gets the
same mass as one reachable by 1); mass sums to 1; per-step records carry every field the metrics need.

---

### A3. Frozen task splits  *(priority 1b/4, directive items 4–5)* **[now]**

**Files** `src/compose_v4/data/adaptation_tasks.py`, `scripts/build_adaptation_tasks.py`,
`configs/tasks/*.json`, `tests/test_adaptation_tasks.py`
*(new module under `src/compose_v4/data/` is permitted; existing files there are not modified)*

**Split policy — reuse, do not invent.** ringcore-v1 scaffold key (murcko + carbonized-wl3, salt
`ringcore-v1`, 0.9/0.05/0.05); tasks drawn from validation/test only; disjointness **verified numerically**.
Similarity ECFP4 r=2, 2048 bits, Tanimoto. Seeds `[0,1,2,3,4]`.

**Anti-tuning property.** Structural: constructed from corpus statistics with no model output inspected.
Recorded per artifact: `frozen_before_checkpoint`, the highest existing snapshot step at freeze, the training
commit, the corpus scope hash, the split-policy identity, and a content hash.

**E3 families** source_too_small_target_larger · source_too_large_target_smaller ·
grow_then_switch_to_smaller_preferred_range (target interval changes mid-trajectory) · held_out_atom_count_bins ·
**matched_sources_unsolvable_at_fixed_cardinality** (load-bearing: makes cardinality change provably necessary;
the verification argument must be documented, not asserted).
Per-step records: active atom count, insertions, deletions, net change, overshoot, correction, steps to enter
interval, terminal success, similarity retained, validity, connectedness.

**E4 families** cycle_creation · cycle_opening · ring_size_change · fused_system_modification ·
spiro_or_bridged_where_supported · topology_simplification · topology_addition_then_reversal_after_switch.
**If the production operator set cannot reach spiro/bridged changes, those tasks are marked unsupported —
not fabricated, and the disabled `ring_system_grow` macro is not re-enabled to manufacture them.**

**Slot hazard.** Atom counting must mask on the real-element predicate; slot-stable states leave nulls mid-array
and `[:n_real_atoms]` silently drops trailing real atoms.

**Acceptance.** Held-out disjointness verified; deterministic under seed; content hash changes when any task
changes; the unsolvable set is unsolvable by the documented argument.

---

### A5. Exact enumerable RingCore graph + Doob solver  *(priority 5, directive item 7)* **[now]**, `learned_value` arm **[frozen-gated]**

**Written from scratch.** No prior-art imports; prior JSONs are non-evidence.

**Files** `src/compose_v4/experiments/enumerable_ringcore.py`,
`src/compose_v4/experiments/exact_finite_horizon_control.py`,
`scripts/run_e6_exact_control.py`, `results/E6_exact_control.json`, `tests/test_e6_exact_control.py`

**A5.1 Sizing first (gate on the rest).** Measure the reachable-set size and branching under the **real
production executor** for candidate bounded chemistries; report the numbers; only then fix the chemistry.
Operator set is RingCore-V1: compositional `cycle_close`/`cycle_open` (executor `bond_insert`/`bond_delete`,
families `cycle_insert`/`cycle_attach`), `ring_system_grow` macro **disabled**. If no candidate is
enumerable, that is the finding and the exact panel is reported as infeasible rather than approximated.

**A5.2 Precompute** every valid state; every legal action; canonical successors; aggregate rates; transition
matrices **indexed by remaining budget** (finite horizon ⇒ time-inhomogeneous); connected components;
terminal distributions; exact backward values.

**A5.3 Exact finite-horizon Doob transform** + primary metric
`terminal_law_total_variation_residual` vs the analytic tilt. Report the **actual magnitude**; at solver
tolerance expect ~1e-10. A tolerance is never loosened to make a test pass; a non-exact result is reported
as the finding.

**A5.4 Support preservation** — the transform neither creates nor destroys support.

**A5.5 Controller arms** scored against the exact optimum: `endpoint_reranking`, `greedy_reward`,
`local_boltzmann`, `smc_feynman_kac`; `learned_value` interface defined and reported checkpoint-gated.

**A5.6 Dynamic** objective replacement, exact continuation from an intermediate state, restart control —
these underpin the paper's dynamic-switching claim, so exactness here is load-bearing.

**A5.7 Tilts** rare motif · atom-count interval · property interval · two simultaneous constraints ·
Boltzmann reward · small objective-space region.

**Acceptance.** Enumeration provably closed under the executor (completeness); every transition matrix's rows
sum to 1; Doob terminal law matches the analytic tilt to stated tolerance; support preserved.

---

### A4. Canonical-quotient invariance fixtures  *(priority 6, directive item 6)* **[now]**, check 6 needs A5

**Files** `src/compose_v4/experiments/quotient_invariance.py`, `tests/test_quotient_invariance.py`

Symmetric molecules where several marks produce the same molecular successor. All six checks:
1. raw mark multiplicity differs;
2. aggregate canonical successor mass is identical;
3. sampled successor frequency matches aggregate mass — needs a **declared sample size and fixed seed** so
   the test is not flaky; this is a sampler property and needs no trained weights;
4. renaming persistent slots does not change the molecular kernel;
5. splitting one internal action encoding into equivalent sub-actions does not change the state-level law;
6. **exact control computed on the quotient is invariant to encoding refinement** — requires A5's solver.

---

## Phase B — protocol freezing (must land before any result is inspected)

- **B1. Same-base controller interface** *(item 8)*. One interface for: unguided prior · endpoint reranking ·
  greedy one-step reward · local Boltzmann · scalarized guidance · MOG-DFM-style control · SMC/Feynman–Kac ·
  learned Doob/value. Records generated endpoints, all intermediate states, base-kernel log probability,
  controller correction, oracle calls, compute, effective sample size, constraint violations, success and
  Pareto metrics.
- **B2. Frozen oracle and task protocol** *(item 9)*. Per benchmark: source molecules, property objectives,
  scaffold/source splits, similarity policy, objective normalization, constraint thresholds, optimization
  budget, oracle-call budget, seeds, success definitions. Multi-objective predeclarations: 2-objective tasks
  for Pareto visualization, 3-objective, 4–5-objective stress tests, **hypervolume reference points**,
  epsilon/dominance convention, diversity measurement, source-similarity floor. Independent evaluation oracle
  where available. **Normalization declared before any model output is inspected; post-hoc renormalization
  forbidden.**
- **B3. Dynamic experiment protocols** *(item 10)*. Dynamic preference switching (continuation vs restart vs
  restart-under-separate-sampler vs static compromise; adaptation regret, retained old-objective gain, new
  gain, edit/oracle cost, time to adapt); Pareto fan (prefix reuse, endpoint spread, hypervolume, region
  coverage, branch diversity, cost vs independent restarts); pathwise constraints (scaffold protection,
  pharmacophore retention, atom-count interval, charge policy, structural-alert exclusion, optional similarity
  corridor) compared against endpoint-only filtering, reporting invalid/forbidden intermediate states, wasted
  oracle calls, endpoint yield, path feasibility, cost.
- **B4. Registry extension** to cover items 8, 10 and 13, which it currently does not.

---

## Phase C — automated post-training queue *(item 13)*

`scripts/checkpoint_leaderboard.py` + `scripts/post_training_queue.py`.
Steps: wait for checkpoint + evaluation artifact → verify provenance → production-weighted canonical-successor
NLL → balanced-family NLL → update leaderboard → apply the **predeclared** selection rule → freeze the winner
once the run completes → launch low-cost diagnostics first → launch the full same-base suite after freeze.
Consumes `checkpoint.step<N>.pt` snapshots (every 500 steps).

## Phase D — experiment runs **[frozen-gated]**

E2, E3, E4, E7 against the frozen checkpoint, seeds `[0,1,2,3,4]`, outputs to `results/E*.json`.

## Phase E — de-novo lane **[lane]** *(item 11, priority 8)*

Repair + regression-test the deferred de-novo `bond_reorder` defect; verify timed CTMC hazard semantics;
define the de-novo source distribution; build its training manifest and contract; confirm RingCore-V1
compositional cycle operations active and legacy whole-ring growth disabled; unconditional evaluation scripts;
baseline environments. Unconditional metrics: all-step validity, endpoint validity, connectedness, uniqueness,
novelty, internal diversity, FCD, precision/recall or coverage, scaffold novelty, size/property distributions,
ring/topology distributions, nearest-neighbour and memorization analysis, throughput.
**E1 only behind its own gates. The editing checkpoint is never evidence for unconditional generation.**

## Phase F — baselines and figures *(items 12, priorities 9–10)*

External baseline environments; then figures as executable artifacts: Figure 2 (generation, size, topology,
learned transport), Figure 3 (exact control), Figure 4 (conditional/dynamic), tables for unconditional
benchmarks / transport ablations / same-base controllers / external baselines, Extended Data for quotient
fixtures, operator support, held-out bins, failure cases, full Pareto fronts.

---

## 4. Open questions for the owner

1. **Phase B timing.** The priority list puts A5 (5) before dynamic protocols (7). But "freeze the protocol
   before looking at results" argues for B2 earlier. Current plan follows your priority order; say the word to
   hoist B2.
2. **E6 scope if sizing fails.** If no bounded chemistry under the real executor is enumerable at reasonable
   cost, preference: report infeasible, or shrink the chemistry until it is enumerable even if less chemically
   interesting?
3. **Spiro/bridged E4 tasks.** Confirmed intent: mark unsupported rather than re-enable the macro.
