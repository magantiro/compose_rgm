# Handoff: RingCore-V1 post-run diagnosis and repair plan

Self-contained handoff. Assumes no prior context from the session that produced it.

**Branch:** `claude/control-closed-pareto-editing` · **HEAD at handoff:** `4489370`
**Full test suite:** 967 passed, 0 skipped
**State:** 16,000-step training run COMPLETE. Two capability failures diagnosed. No checkpoint selected.
No architecture changed. Repair is a training-recipe problem, not an operator-design problem.

---

## 0. TL;DR for whoever picks this up

1. The run finished cleanly: 32/32 snapshots, zero preemptions, aggregate loss −30%.
2. **Two families failed: ring-opening (`cycle_attach`) and graft (`bond_reroute`).** Both were WORKING at
   initialization and were destroyed during training. This is catastrophic forgetting + training competition,
   NOT broken operators.
3. **A micro-overfit test proves `cycle_attach` is fully learnable** (family probability → 1.0 in 50 steps on
   56 examples). Do NOT re-enable the `ring_system_grow` macro as a substitute — it cannot open rings at all.
4. The fix is: family-balanced sampling + base-B replay/distillation + differential learning rates +
   per-family abort gates. Then staged pilots (500 / 2,000 steps) before any full run.
5. The blocking omission that let this reach step 16,000 undetected: monitoring tracked only AGGREGATE loss
   and balanced accuracy. Per-family retention/learnability gates did not exist. Build them first.

---

## 1. Project context

COMPOSE/RGM: a continuous-time Markov chain over molecular graphs whose committed states are complete,
chemically valid, connected molecules and whose transitions are executable chemical rewrites. Generator
Matching learns contextual firing rates. Repo conventions in `CLAUDE.md`, `.claude/context/*.md`.

**Two regimes.** De-novo (build from a carbon-tree source prior) and editing ("B-edit": start from
lightly-corrupted real molecules). The editing prior is the subject of this run.

**RingCore-V1** replaced the finite whole-ring macro (`ring_system_grow`) with compositional cycle
operations: `cycle_close` / `cycle_open`. Executor rule names are `bond_insert` / `bond_delete`; the dense
head scores them under family slots 5/6 named `cycle_insert` / `cycle_attach`. Mapping lives in
`_CYCLE_OP_EXECUTOR_TO_FAMILY` (`src/compose_v4/model/factorized_tracelet_rate_model.py:112`).
**These are different namespaces and conflating them has caused real bugs. Always translate.**

| family slot | executor rule | meaning |
|---|---|---|
| `cycle_insert` (5) | `bond_insert` | ring **closing** (add a ring bond) |
| `cycle_attach` (6) | `bond_delete` | ring **opening** (remove a non-bridge ring bond) |

---

## 2. The completed run

- **Run label:** `compose-v4-ringcore-v1-scientific-a7546e2-v1`
- **Commit:** `a7546e2` · **Modal workspace:** `nitya` · **Volume:** `compose-v4-artifacts`
- **Warm start:** base B = `compose-v4-stage3-flexible-graft-3k-1ac6f19-v1/checkpoint.best_so_far.pt`,
  loaded with `--initialize-compatible-from-source-checkpoint` (strict fails on the 4→15 head widening)
- **Scheduler:** AdamW, peak LR 3e-4, warmup 500, cosine to 0.05 floor over 16,000 steps,
  hash `dafd4b5092414394`
- **Batch size 64**, bf16, `evaluation_every=250`, `recovery_every=500`, early stopping DISABLED
- **Throughput:** ~0.726 s/step (better than the 0.8331 projection)

### Artifact inventory (verified)

| check | result |
|---|---|
| snapshots | **32 / 32** (`checkpoint.step500.pt` … `checkpoint.step16000.pt`, every 500) |
| missing / unexpected / duplicate steps | NONE / NONE / NONE |
| `metrics.json`, `checkpoint.pt`, `checkpoint.best_so_far.pt`, `checkpoint.recovery.pt` | all present |
| `manifest.training.json` | present |
| preemptions / resumes after relaunch | **0 / 0** |

**Still owed:** SHA-256 for each of the 32 snapshots. Do this Modal-side; do not download ~32 checkpoints.

### Headline metrics

| | initial validation | best validation (step 8500) | final test |
|---|---|---|---|
| `factorized_gm_loss` | 3.9712 | **2.7919** | 2.9708 |
| balanced-family accuracy | 0.1089 | 0.1889 | 0.1937 |
| balanced-family top-3 | 0.3228 | 0.4318 | 0.4522 |
| represented families | 9/10 | 9/10 | 9/10 |

`represented_families = 9/10` is CORRECT: `ring_system_grow` has 0 teacher examples because the macro is
deliberately disabled. Loss plateaued from ~step 8000 (28 evaluations without improvement) — consistent with
the cosine tail; LR at step 15,000 was ~1.8e-5, ~6% of peak.

**NOTE:** the final-test table has now been INSPECTED. It cannot serve as a sealed holdout for the repaired
model. Construct a fresh untouched holdout before any paper claim.

---

## 3. Per-family results (final test) — the failures

| family | accuracy | top-3 | teacher examples | share |
|---|---|---|---|---|
| `atom_insert` | 0.512 | 0.791 | 43 | 2.5% |
| `atom_restate` | 0.500 | 0.958 | 118 | 6.9% |
| **`cycle_insert`** (ring close) | **0.502** | 0.680 | 681 | 39.7% |
| `atom_delete` | 0.214 | 0.696 | 56 | 3.3% |
| **`cycle_attach`** (ring open) | **0.015** | **0.109** | 650 | 37.9% |
| `bond_reorder` | 0.000 | 0.171 | 35 | 2.0% |
| **`bond_reroute`** (graft) | **0.000** | **0.000** | 50 | 2.9% |
| `ring_system_restate` | 0.000 | 0.464 | 56 | 3.3% |
| `ring_system_delete` | 0.000 | 0.200 | 25 | 1.5% |
| `ring_system_grow` | 0.000 | 0.000 | **0** | 0% (expected) |

Total teacher examples 1,714. **Cycle ops = 78% of all supervision. Graft = 2.9%.**

`family_accuracy_X` semantics (verified in code, `factorized_mark_conditional.py:952`): among examples whose
TEACHER family is X, the fraction where `argmax(family_log_probabilities) == X`. It is 10-way family
classification, not within-family mark accuracy.

### FAILURE 1 — ring opening collapsed from a working initial state

| | at initialization | final test |
|---|---|---|
| p(`cycle_attach`) family prob | **≈ 0.253** | 0.026 |
| p(`cycle_insert`) family prob | ≈ 0.243 | 0.350 |
| mean teacher MARK probability (attach) | — | 0.00157 |
| mean teacher MARK probability (insert) | — | 0.3092 |

At initialization, on real ring molecules (cyclohexane, benzene, paracetamol, naphthalene), the untrained
model assigns ring-opening **slightly more** probability than ring-closing. Training drove opening down ~10×
and closing up. **The capability existed at step 0 and was destroyed.**

### FAILURE 2 — graft was catastrophically forgotten

`bond_reroute` top-3 accuracy: **0.821 at initial validation → 0.000 at final test.** The initial value
comes from base B, which WAS trained on flexible graft. So the warm start transferred the capability
correctly and 16,000 steps of training erased it, on 2.9% of the supervision, with no retention machinery.

---

## 4. Diagnostics — what has been RULED OUT (with measurements)

Do not re-run these. All were measured on real drug-like leads using the production executor.

### 4.1 Aliasing is NOT the explanation (predicted the wrong sign)

| | raw marks/state | canonical successors/state | alias multiplicity | frac aliased |
|---|---|---|---|---|
| `cycle_insert` (close) | 23.25 | 12.75 | 1.824 | 0.569 |
| `cycle_attach` (open) | 8.00 | 5.75 | 1.391 | 0.283 |

Opening has LESS aliasing and FEWER candidates — the easier family on both axes.

### 4.2 Structural masking is NOT the explanation

Forward pass on an untrained model with `enable_cycle_ops=True`: both `cycle_insert` and `cycle_attach`
masks are populated (12–22 cycle edges), neither family is `-inf`, and p(open) ≈ 0.253 > p(close) ≈ 0.243.
`batch.cycle_edge_mask` is computed in `_state_features` per state and read identically by both paths.

### 4.3 Cross-direction target ambiguity is NOT the explanation

`build_cycle_op_records` on 10 leads: 40 open teachers, 40 close teachers, 47 distinct source states,
**0 states carrying both directions**. Open lives on real molecules, close on precursors.

### 4.4 A real structural asymmetry exists, but it does NOT explain the family-level failure

| | targets per source state | distinct source states | competing families at those states |
|---|---|---|---|
| open | **4.00** | 6 (the real molecules) | **2.83** (max 3) |
| close | **1.04** | 23 (precursors) | 1.23 (max 2) |

Example real-lead teacher set: `{bond_delete: 4, atom_delete: 1, bond_reroute: 2}` — four "open this bond"
answers, two graft answers, one atom delete, all at the SAME state.

**Why this does not explain it:** `bond_delete` has the MOST teacher mass at those crowded states (4 vs
graft's 2). Under a rate-matching objective it should WIN the family argmax there. It has the lowest accuracy
of any supervised family. The story is contradicted by its own data. Record this as measured-but-insufficient.

### 4.5 DECISIVE — `cycle_attach` is fully learnable (micro-overfit)

Trained a fresh model (hidden_dim 64, 2 MP steps, cycle ops on, macro off) on **56 ring-opening teachers
only**:

| step | loss | family accuracy | mean p(attach) |
|---|---|---|---|
| 0 | 4.40 | 1.0000 | 0.570 |
| 50 | 2.72 | 1.0000 | **1.000000** |
| 300 | 2.66 | 1.0000 | 1.000000 |

Family probability saturates at 1.0 within 50 steps. (Accuracy 1.0 at step 0 is not learning — `cycle_attach`
is the argmax family at init. The signal is p: 0.57 → 1.00.)

**Conclusion: no representational, masking, labelling or gradient-routing defect.** Head, mask, teacher
encoding, executor round-trip and the gradient path into slot 6 all work. The full-run collapse is
optimization / data competition.

### 4.6 Mechanism still UNRESOLVED

What remains consistent with all evidence: the family head reallocates mass away from `cycle_attach` toward
`cycle_insert` when the full mixture is present. Why it would, given precursors vs ring-bearing molecules are
trivially separable by cycle count, is not explained. **This is the open question.** It does not block the
repair (the repair addresses competition and retention directly), but it should be understood.

---

## 5. Decision: keep RingCore, do NOT restore `ring_system_grow`

1. `cycle_attach` passes its unit test (§4.5). You do not replace a component that works.
2. `ring_system_grow` **adds** rings; there is no ring-OPENING capability in the macro vocabulary. It cannot
   substitute, and ring opening is a declared E4 task family.
3. This run produced **zero** evidence about the macro — 0 teacher examples. Re-enabling would be a guess.
4. Ring CLOSING via the compositional path demonstrably works (0.502 accuracy, family prob 0.24 → 0.35),
   which is positive evidence for the compositional design.

---

## 6. The repair plan

All recipe, no architecture.

| # | change | notes |
|---|---|---|
| 1 | **Per-family sentinel + abort gates** | build FIRST; makes every later pilot trustworthy; reusable for the de-novo lane |
| 2 | **Family-balanced sampling** | graft got 2.9% of signal against 78% cycle ops |
| 3 | **Base-B replay + legacy-logit distillation** | the only genuinely new machinery; no precedent in this repo |
| 4 | **Differential learning rates** | low for inherited body/heads, higher for new vocab rows + cycle heads |
| 5 | **Initial frozen-body phase** | protect inherited representation early |

Suggested objective shape (coefficients to be screened on a development split in short pilots, NEVER chosen
from the final test):

```
L = L_ringcore_edit + lambda_replay * L_base + lambda_retain * KL(q_base || q_theta)
```

with the KL taken on a frozen legacy-family probe set.

### Staged protocol — do not launch a blind full run again

```
Gate 0  initialization parity   every inherited family vs base B, retention tolerance enforced,
                                new heads identified, transferred params verified, every family has
                                valid teachers AND candidates
Gate 1  per-family micro-overfit  each family memorizes a tiny set to a successor-recovery threshold
                                  (cycle_attach DONE, §4.5; graft and the rest still to run)
Gate 2  500-step pilot          ~10 min GPU, ~$0.40. Abort on: inherited-capability collapse, zero
                                gradients into a family head, absent teachers, exploding candidate
                                support, new family showing no movement
Gate 3  2,000-step pilot        ~25 min, ~$1. Require: improvement in every load-bearing new family,
                                acceptable legacy retention, successor-level (not just aggregate) gains
Gate 4  full 16k                ~3.5 h, ~$8, with alarms and a stop rule live throughout
```

### The two alarms that would have caught this by minute 10

1. **Capability retention alarm** — any inherited family dropping materially below its initialization value
   (would have fired on graft: 0.821 → falling).
2. **New-family learnability alarm** — a family with substantial teacher mass showing no movement or
   negative movement (would have fired on `cycle_attach`).

### Per-checkpoint forensics still to run on the completed run

Do NOT discard the 32 snapshots. On a fixed validation probe, for each checkpoint, compute for
`cycle_insert`, `cycle_attach`, `bond_reroute`, `bond_reorder`:
mark-level NLL · canonical-successor NLL · top-1/top-3 · MRR · raw and canonical candidate counts · alias
multiplicity · effective teacher weight · gradient statistics where logs allow. This locates WHEN each
collapse began and whether an intermediate checkpoint retained graft.

---

## 7. Checkpoint selection — NOT yet done, and constrained

`metrics.json` reports `selected_step = 8500`, but that is the TRAINER's internal pick by its own
early-stopping reference loss. It is **not** the preregistered rule.

**The frozen rule:** production-weighted canonical-successor NLL (primary); balanced-family
canonical-successor NLL (secondary); held-source/scaffold/topology safeguards; calibration and rollout
behaviour as tie-breakers. Hazard calibration reported SEPARATELY and never used to select.

Constraints:
- Select on **validation**, never the final test (already inspected).
- Do NOT select by raw loss, by family accuracy, or by "step 8500 because the trainer said so".
- Do NOT retroactively add the new family metrics to the frozen primary rule. If the selected checkpoint
  fails a load-bearing capability safeguard, declare the run **scientifically insufficient** and repair.
- The plateaued metric is the MARK-level objective; selection is on the SUCCESSOR-level pushforward. Different
  quantities — the plateau says little about the latter.

---

## 8. Infrastructure that already exists (do not rebuild)

All committed, `src/` ruff-clean, 967 tests green.

| module | purpose |
|---|---|
| `configs/experiment_registry.yaml` | frozen E1–E7 protocol, schema v2. Protocol content hash `b4cd905a640ab32a` |
| `src/compose_v4/experiments/registry.py` | validating loader; rejects v1, detects protocol drift, enforces 1–3 primary metrics, semantic panel IDs |
| `src/compose_v4/experiments/successor_kernel.py` | the shared `CanonicalSuccessorKernel` protocol, 12-field `SupportSignature`, `UniformSuccessorKernel`, comparison types |
| `src/compose_v4/experiments/reference_successor_kernel.py` | slow dictionary-based aggregation oracle. **TEST ONLY** — a scan test fails if production imports it |
| `src/compose_v4/experiments/checkpoint_evaluator.py` | A1.1: construction, capability validation against the registry, support signature, versioned output envelope |
| `src/compose_v4/experiments/enumerable_ringcore.py` | E6 exact-graph sizing, `cycle_rank` (sole definition), graph fingerprint, composite benchmark identity |
| `scripts/run_e6_sizing_sweep.py` | runs the sizing sweep, writes `diagnostics/exactness/e6_sizing_sweep.json` |
| `tests/test_slot_safety.py` | enforces `is_element`/`is_occupied` for molecular occupancy; ratchet elsewhere |
| `docs/EXPERIMENT_INFRASTRUCTURE_PLAN.md` | the full build plan with two rounds of adversarial review recorded |

**E6 benchmark selected:** `carbon_6_slots` — 967 states, 14,432 canonical directed edges,
`structural_graph_fingerprint = 84121ff86cbc1ba8`, `benchmark_semantics_hash = 3647e87f8f75b038`. All six
non-degeneracy conditions hold; 964/967 states have multiple distinct parents. Null included as a
vocabulary-restricted distinguished source (`∅ ⇄ C`).

---

## 9. Conceptual precision the manuscript must preserve

**The trained objective is MARK-LEVEL.** `factorized_mark_bregman_loss` is
`total_hazard − teacher_rate·(log_hazard + selected_mark_log_probability)` — a Poisson-KL Generator Matching
loss "in normalized marked-rate form" scoring the selected teacher MARK.

The molecular kernel is the **pushforward**:

```
y = pi_x(a),    P_theta(y|x) = sum_{a : pi_x(a)=y} q_theta(a|x)
```

Correct formulation: *COMPOSE learns a stochastic process over executable rewrite marks. Pushing this process
through execution and molecular canonicalization induces a well-defined kernel over molecular successors. All
control and state-level comparisons operate on this quotient kernel.*

- E5 licenses the **quotient-level interpretation and control** of the pushforward. It does NOT make the
  training objective successor-level. Saying E5 "licenses the derivation" is an overstatement.
- No production canonical-successor aggregator is wired into training or rollout. `model.segmented_successor`
  and `canonical_successor_distillation.aggregate_canonical_successor_rates` have test callers only.
  `segmented_successor` *computes* the pushforward; the dictionary oracle *verifies* it. No source file is
  the definition.
- The graft family is already quotiented at training time via `graft_successor_groups`, so the trained law is
  mark-level EXCEPT for graft.

**Vocabulary scope — three distinct things, never conflate:**
1. **Production editing prior: 15-class broad-organic** element-valence table, `max_atoms=40`.
2. **E6 carbon-only six-slot: an exactly enumerable verification slice**, not the trained vocabulary and not
   representative drug-like chemistry.
3. **Legacy CNOF-4 checkpoint: an initialization/testing artifact only.**

The de-novo lane must also ultimately be broad-organic; the CNOF fixture must not stand in as the
unconditional result.

---

## 10. Standing constraints

- **Never mention Claude/AI in commits or PR bodies.** No co-author trailer.
- Commits atomic; subject `<module>: <one-line imperative, lowercase, no trailing period>`.
- Never launch Modal before `scripts/prelaunch_gate.py` is green; launch only from a clean committed detached
  worktree. `--detach` is MANDATORY (`main()` uses `.spawn()`).
- `_source_fingerprint()` hashes `src/`, `scripts/`, `recipes/` and `modal_apps/train_tracelet_gm.py`, so
  edits there change run identity. `configs/`, `docs/`, `diagnostics/`, `tests/`, `results/` do not.
- Do not change production operator semantics to make a result work. Report before implementing.
- Do not add multi-neighbour `AtomInsert` (vertex subdivision) — outside production support.
- Tests: `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src python3 -m pytest tests/ -q`.
  `ruff` is at `./.venv/bin/ruff`, NOT on PATH. `src/` stays ruff-clean; `scripts/` are drivers.
- Read the gate's SKIP count, not just PASS.

---

## 11. Traps that have actually bitten this project

1. **Slot-stable states.** A delete leaves a NULL slot mid-array, and deletion can leave a SCAR (occupied but
   NOT a real element). `atom_types >= 0` counts padding (NULL_IDX == 0); `atom_types != NULL_IDX` counts
   scars; `[:n_real_atoms]` drops trailing real atoms. Use `is_element` / `is_occupied`. Enforced by
   `tests/test_slot_safety.py`.
2. **Executor rule names ≠ model family names.** Always translate via `_CYCLE_OP_EXECUTOR_TO_FAMILY`.
3. **A teacher must lie in the dense mask that scores it.** Outside → grouped logsumexp `-inf` → rate 0 →
   `+inf` loss that survives forward and backward as inf/nan. Smoke every new teacher source for FINITE loss,
   split by direction, on a CHARGED drug-like lead.
4. **Any batch builder scoring editing teachers must enable the same editing-family flags as training.**
   Train worked / eval didn't, once already.
5. **Cycle rank ≠ SSSR ring count ≠ ring-system count.** `CalcNumRings` is symmetrized SSSR and returns 3 for
   bicyclo[2.2.2]octane where the cycle rank is 2. Use the Betti number `|E|−|V|+c`.
6. **An empty SMILES parses to a zero-atom Mol**, so `is_rdkit_valid(empty)` is True. Validity does not block
   the null state.
7. **Micro-benchmarks lie about paths.** A component speedup is not a path speedup until the whole path is
   timed (this repo has a 630× → 2× → 98× sequence on record).
8. **A test whose reference is a copy cannot fail usefully**, and a scan-style test does not catch a broken
   import — add `py_compile`.
9. **`nohup cmd &` reports the wrapper's exit, not the job's.** Do not read an empty log as success.
10. **Preemption + warm start collided once** and killed a run at step 1500: the retry carried
    `--initialize-compatible-checkpoint` together with `--resume-checkpoint`, which the gate rejects as
    mutually exclusive. Fixed in `2ffb4dd` / `a7546e2` (resume supersedes initialization; retries 1→5).
    Resume is now EXACT: weights, AdamW moments, `completed_steps` (LR position), dataloader offset, CPU+CUDA
    RNG, guarded by a required-keys check, an `optimizer_kind` discriminator and a 15-field provenance match.

---

## 12. Recommended order of work

1. **Per-family sentinel + abort gates** (nothing else is trustworthy without it).
2. SHA-256 the 32 snapshots Modal-side; freeze the run identity record.
3. Per-checkpoint family forensics across all 32 snapshots — locate when each collapse began.
4. Preregistered checkpoint leaderboard on VALIDATION canonical-successor NLL.
5. Remaining per-family micro-overfits (graft especially).
6. Implement retention machinery (replay + distillation + differential LRs + frozen-body phase).
7. 500-step pilot → 2,000-step pilot → full run, with alarms live.
8. Construct a fresh sealed holdout before any paper claim.

## 13. Open questions

1. **Why does the family head reallocate mass away from `cycle_attach` under the full mixture**, when the
   family is learnable in isolation and its source states are separable from close's by cycle count? (§4.6)
2. Does replay + distillation actually hold graft? No precedent in this repo.
3. Is the ~2.5× wall-clock tax from dataloader stalls worth fixing before the repaired run? Measured:
   data_wait is 65% of mean step time but the MEDIAN wait is 0.4 ms — rare, large stalls (shard opens), not
   steady starvation. Typical step 0.29 s vs effective 0.73 s.
