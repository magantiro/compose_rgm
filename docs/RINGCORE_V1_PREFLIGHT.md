# RingCore-V1 preflight — bank the compositional ring core

**Mandate:** freeze the supervised compositional ring core as a separately-versioned production candidate
`RING_CORE_V1` (`enable_cycle_ops=true`, `enable_ring_macros=false`, legacy `ring_system_grow` macro
**disabled**), run the bounded preflight, and return a macro decision. **Do NOT** implement P4 into this
production path. **Do NOT** launch the full A100 run — bounded preflight only (≤750 steps).

Ring reachability is defined by the compositional operators; a finite ring catalog is a LATER
(`RING_HYBRID_V2` / P4) acceleration layer, never the definition of support.

Status legend: ✅ done · 🔶 in progress · ⏳ pending.

> **Full-run prerequisite (scaling gate, NOT a bounded-preflight blocker):** corruption records are
> regenerated **in-memory, serially (~0.5 s/sample on Modal)** at train time — fine for the bounded preflight
> (200 source mols ≈ 3–4 min) but a hard bottleneck at production scale (~100k+ source mols ≈ **14+ h**,
> GPU-blocking). The MMP pool is already precompiled to the volume (`edit_pool_full.jsonl`); corruption is not.
> **Before any full run:** mine the corruption pool once to the volume via a parallel fan-out (extend
> `mine_edit_traces_app.py`), applying the teacher-representability filter at mine time, then load it like
> `--analogue-trace-pool`. RingCore's zero-mixture path has no de-novo cache (`denovo_keep=0`), so the
> historical reason corruption stayed in-memory (preserve B's carbon cache under `--train-only`) no longer
> applies — precompiling is clean. This is item 1 of the post-`GO_FOR_FULL_RINGCORE_TRAINING` scale-up.

---

## 1. Freeze — RING_CORE_V1 ✅
- **Git tag `ring-core-v1`** → commit `491121d` (manifest); frozen core code at `0016873`.
- **Capability hash `330473e319bfec19`** · operator-registry hash `9197401e8dc3a7ae` · cycle-op semantic
  hash `27a823aeb6cf7548` · ring-support-semantics-version `1`.
- Config: `enable_cycle_ops=True`, `enable_ring_grow_macro=False`, `enable_ring_macros=False`,
  `corrupted_prior_mix=True`, `organic_vocabulary=True` (15 classes), scope `3721d69851110fdd`, production
  ring catalog `639ff6078c32d43c`, base `c9d927…`.
- **Legacy grow disabled** (`enable_ring_grow_macro=False`): grow family masked dead — 0/400 draws vs 74/400
  enabled; grow head params retained (warm-start-safe, byte-identical when enabled). Ring ADDITION is purely
  compositional (`cycle_close`).
- **De-ambiguated public↔internal operator map** (internal `cycle_insert`/`cycle_attach` are engineering
  aliases, never exposed): public `cycle_close`(bond_insert) · `cycle_open`(bond_delete) ·
  `ring_delete`(whole removable-ring, decoration-preserving) · `ring_aromaticity_restate`. Deferred to P2:
  `ring_ear_insert`, `ring_spiro_attach`. Disabled: `ring_grow_macro`.
- **Latent Modal-crash blocker fixed:** the real training path (`FactorizedMarkDataset.__getitem__`) rejected
  the cycle-op executor teacher names (`bond_insert`/`bond_delete`) — the prepare-batch tests bypassed it. Now
  aliased to `cycle_insert`/`cycle_attach`. RingCore training smoke: loss finite, grad_norm 0.75. Full suite
  **530 passed**.
- Artifacts: `diagnostics/production_preflight/ring_core_v1_manifest.json`; `scripts/freeze_ring_core_v1.py`.

## 2. Ring-topology capability gate ✅ `GO_RING_TOPOLOGY_CAPABILITY`
Executable compositional construction (`cycle_close`) + deletion (`cycle_open`) per topology class, decomposed
via a spanning tree (non-tree edges = ring-closing bonds), with all 14 required properties.

**Topology matrix — 14/14 `SUPPORTED_BY_CORE`, 0 `REQUIRES_STRUCTURED_P2_MODE`, 0 `UNSUPPORTED`:**

| Class | ring bonds | Class | ring bonds |
|---|---|---|---|
| simple_saturated | 1 | fused_saturated (decalin) | 2 |
| aromatic_carbocycle (benzene) | 1 | spiro | 2 |
| aromatic_heterocycle (pyridine) | 1 | bridged_bicyclic (norbornane) | 2 |
| saturated_heterocycle (piperidine) | 1 | polycyclic_cage (adamantane) | 3 |
| macrocycle (12-ring) | 1 | sulfur_ring_aromatic (thiophene) | 1 |
| fused_aromatic (naphthalene) | 2 | sulfur_ring_saturated | 1 |
| phosphorus_ring | 1 | charge_preserving_ring (N⁺ + cyclohexane) | 1 |

Every class passed all of: valid start, legal compositional sequence, all-intermediate validity+
connectedness, ≤40 atoms every state, endpoint isomorphism, correct cycle-rank changes, charge preservation,
canonical-successor identity, exact inverse (open↔close), positive supervision.

**⇒ The mandate's P2 conditional is NOT triggered — the compositional core is support-complete across all
topology classes; no minimal structured P2 mode is required for support.** Fused/spiro/bridged/cage
construction is reachable compositionally (efficiency is measured separately in §3).

**Core mechanism (topology-agnostic, once over the union):** finite gradients ✅ (grad_norm 1.18) ·
forced sampling executes ✅ · natural sampling after a 120-step overfit ✅ (**100%** of draws on precursor
states are cycle ops) · save/reload equivalent ✅. Artifact:
`diagnostics/production_preflight/ring_topology_capability.json`.

## 3. Compositional path-cost study ✅
The pilot analogue pool has **no** ring-changing pairs (53 pairs, all `variable_ring_atoms=0`, atom_insert/
delete only), so path cost is measured directly on **800 real held-out GuacaMol ring systems** via the
executor-verified spanning-tree decomposition (ring-system ↔ ring-opened precursor is the canonical
ring-changing transformation; its compositional edit length is exactly the cycle_close/cycle_open count).
**100% compilation success**, max temp atoms 40 (bound respected). Budget = 16 (sampler horizon).

Two costs (`diagnostics/production_preflight/ring_path_cost.json`):
- **Ring restructuring** (cycle_close/open when the ring atoms are present — the common lead-opt ring edit:
  open/close/reshape/fuse): median **3**, p75 4, p90 5, **max 7**. **86.6% ≤4 edits, 100% ≤8, 0% exceed the
  budget.** Cheap for *every* topology (per-topology medians 2–4.5, max 7 incl. fused/spiro/bridged/aromatic/
  hetero). ⇒ the whole-ring macro adds little to ring restructuring.
- **Ring introduction** (build a whole new ring system from fresh atoms = atom_inserts + closures — the
  UPPER bound a macro collapses to ~1 step): median **20**, **73.5% exceed the budget** (fused/hetero 24 /
  93.6%, spiro 26 / 100%, monocarbocycle 14 / 15%). ⇒ de-novo whole-ring construction is where a macro
  would help — but that is a de-novo move, not the dominant lead-editing operation.

**Read for the macro decision:** restructuring existing rings (the editing regime's ring work) is
compositionally cheap and always within budget; only building a *whole new* fused/hetero ring de-novo is
expensive. Whether that bites RingCore in practice depends on how often trained ring edits are whole-ring
introductions vs restructurings — measured in §6.
## 4. P5 calibration for the compositional core ✅
Calibrates the fresh cycle heads to the **selected-target** statistics (NOT raw corpus ring frequency),
verified at three levels (`diagnostics/production_preflight/ring_core_calibration.json`).
- **Selected-target teacher rate** (precise, 2,570 targets): cycle family **61.0%** (cycle_close 30.5% /
  cycle_open 30.5%, a 50/50 within-family split) — the dominant edit family; other families atom_restate
  10.2%, bond_reorder/bond_reroute ~5.7%, ring_system_restate 4.8%, ring_system_delete 3.0%.
- **Initial sampler mass** (representative model): cycle family 30.3%.
- **Calibration policy** (hash `f534c0233e3bf4a0`): the model is **hierarchical** — the family distribution
  is `family_head`, independent of the within-family cycle-head partition — so the cycle-family mass is
  calibrated on `family_head`'s bias at the cycle indices (5/6), refined for the softmax renormalization.
  Shifts the sampler cycle mass **30.3% → 58.5%** (target 61.0%). **No macro/template prior.**
- **Shared-kernel drift = 0.0**: the calibration touches only the cycle family_head bias; every delete/
  restate/reorder head and every non-cycle family logit is byte-identical (structural, verified).
- **Canonical-successor level**: mean multiplicity **1.47** raw legal cycle marks per distinct successor
  (cycle ops are near-injective; per-coordinate scoring conserves rate — no successor-group over-count).
- *Caveat:* absolute masses use a representative model; the policy + multiplicity + drift are
  backbone-independent, finalized at Modal load with B's warm-started `family_head`.
## 5. Regenerated RingCore manifest + verification battery ✅ `GO_RING_CORE_REGEN`
`diagnostics/production_preflight/ring_core_data_manifest.json` regenerates only the artifacts affected by
the compositional operator change (operator-registry + cycle-op semantic hashes, capability hash, trace
schema, subtype-supervision counts, selected-target distribution, calibration-policy hash, training recipe,
cache-schema discriminators `{corrupted_prior_mix, cycle_op_mix, disable_ring_grow_macro, organic_vocabulary,
analogue_trace_pool}`, checkpoint-metadata contract). SYSTEM_CONTRACT §4 updated (cycle_close/cycle_open
repurposed slots; grow disabled; public-alias policy).
- **Reuse proof:** the MMP/analogue pool uses only `atom_insert`/`atom_delete` (measured 0/98 traces touch a
  ring/cycle op) — families UNCHANGED by the compositional-ring change, so the pool is reused as-is.
- **Verification battery:** operator fuzz **0 corrupt/crash** (357 valid / 2,043 cleanly rejected — the
  executor never corrupts on random cycle ops) · trace replay **464/464 exact** · inverse open↔close
  **232/232 exact** · all-state validity 100% (§2/§3) · subtype gate `GO_SUBTYPE_SUPERVISION` ·
  canonical-successor normalization (multiplicity 1.47, rate-conserving) · tiny-overfit natural sampling ✅ ·
  data-sampler distribution ✅ · save/reload equivalent ✅.

## 5b. Zero-mixture launch orchestration + the eval-collator representability fix 🔶
The bounded run is `--scaled-manifest` (zero-mixture: `denovo_keep=0`, no de-novo path cache). Getting the
REAL trainer to the first editing batch surfaced a chain of orchestration + one genuine correctness bug, all
found by a **CPU-only `--dry-launch`** (early exit from the real trainer after the first training forward + one
edit-validation forward, before backward/optimizer; instrumented so `denovo_*`/`optimizer_steps`=0 are proven,
not assumed). Fixes (each pre-optimizer, ~0 A100 cost):
- Zero-mixture gate path: skip de-novo compilation + carbon-tree dataset; reconstruct the production ring
  catalog from the 5 fixed seeds (`639ff6078c32d43c`); skip the empty-partition guard; build edit VALIDATION
  from `split.validation` (disjoint from `split.train`). Modal `train_stage`/`dry_launch_stage` drop the
  de-novo path/support cache requirement under `--scaled-manifest`.
- **Root cause of the first-forward crash (`teacher ring restate is outside exact dynamic candidates`):** the
  evaluation/validation/test collator constructed its dynamic candidate set with default **de-novo** capability
  flags, while teachers were generated under the editing operator support. This excluded valid editing-family
  teachers (`ring_system_restate` / `ring_system_delete` / `bond_reroute`) from the exact candidate set.
  **Threading the active model capabilities (`OperatorCapabilities`) through all batch-building paths restored
  one shared representability contract.**
- **Charge retraction:** formal charge was NOT causal. Neutral, cationic, anionic, and zwitterionic examples
  are scoreable under the corrected candidate configuration (proven: 0/13 fail with capabilities on vs 5/13
  without), while existing local charged-atom protections (`_touches_charged`) preserve the intended charge
  semantics. No new charge limitation belongs in `SYSTEM_CONTRACT.md`. Verdict:
  `GO_LOCAL_CHARGE_PRESERVING_RESTATE`.
- **Safeguards added:** (1) `OperatorCapabilities` immutable object + `model.operator_capabilities` — batch
  builders take the capability object instead of silently defaulting editing families off; (2) a global
  `teacher ∈ A_exact(x)` invariant (`assert_teachers_in_exact_candidates`) checked immediately after batch
  construction (training + eval), raising rich context (molecule, teacher, candidate count) before the scoring
  loss; (3) eval-batch-cache key now includes the capability fingerprint + operator-registry hash (not just
  `FORMAT_VERSION` 1→2), so a cache built under one capability set is never reused under another.
- §7 data check: the invalid teachers are DYNAMIC-only (the stored MMP/scaffold pool has 0 editing-ring
  teachers), so no stored artifact needs regeneration. Regression: `tests/test_teacher_in_candidates.py`
  (reproduces the failure + 0 mismatches across neutral/cation/anion/zwitterion/S/Cl/fused strata).

## 5c. CPU dry-launch — `GO_CPU_DRY_LAUNCH` ✅
The CPU-only `--dry-launch` of the REAL trainer (commit `898c4dd`, run `compose-v4-ringcore-v1-dry-898c4dd-v1`)
reached the first editing training batch + one held-out editing validation batch with **zero optimizer steps**
and passed every required counter:
- **Zero-mixture (all de-novo counters 0):** `denovo_path_compile_calls`, `carbon_tree_prior_constructions`,
  `denovo_cache_resolution_calls`, `denovo_cache_open_calls`, `denovo_support_cache_resolution/open_calls`,
  `denovo_dataset/dataloader_constructions`, `denovo_training/validation_records_sampled` — **all 0**.
- **Editing path reached:** `production_catalog_constructions=1` (fingerprint `639ff6078c32d43c`),
  `edit_manifest_loads=1`, `edit_pool_open_calls=1`, `edit_dataset/dataloader_constructions=1`,
  `edit_training_batches_emitted=1`, `edit_validation_batches_emitted=1`, `model_forward_calls=2`,
  `gm_loss_calls=2`.
- **Finite losses:** train GM **16.36**, validation GM **3.96**. `optimizer_steps=0`, `backward_calls=0`.
- **`ring_system_grow` absent** (`teacher_examples_ring_system_grow=0`, family prob 0); **cycle families
  present + dominant** (`cycle_insert=310`, `cycle_attach=350` teacher examples).
- **Teacher-in-candidate invariant clean:** `dropped_unrepresentable_traces=13 / 351` at the data source; no
  `TeacherOutsideCandidatesError` reached scoring. Warm-start verified (atom_restate head 94% accurate from B).
- `zero_mixture_ok: true`, `failing_zero: {}`, `failing_positive: {}`.

## 5d. Teacher-filter versioning + characterization (pre-launch finalization §1–§3) ✅ `GO_TEACHER_FILTER_CHARACTERIZED`
The teacher-in-exact-candidates filter is a DECLARED production data contract (version
`TEACHER_REPRESENTABILITY_FILTER_VERSION=1`, hash `f8137995181eeadc`, eval capability fingerprint
`5bedd317328c9ed7`), recorded in the checkpoint metadata, the scaled manifest, and here. Characterized on
400 broad-organic held-out molecules (`diagnostics/production_preflight/teacher_filter_characterization.json`):
- **Removal rate 2.04%** (28/1370 traces) — the measured full-scale rate (the pilot 3.7% was a small-sample
  estimate); report this measured rate.
- **Removals are surgical + one-dimensional:** 28/28 `ring_system_restate`, 28/28 `corrupted_source_grow`
  (inverse) direction, 28/28 `restate_outside_dense_candidates`. Topology 21 fused/aromatic + 5 fused/saturated
  + 2 mono/aromatic (93% fused). Charge 26 neutral + 2 zwitterion (NOT charge-driven). Every removed teacher is
  a genuinely unscoreable inverse restate on a fused ring; no representable teacher is removed.
- **No family collapses (§3):** post-filter every production-enabled corruption family keeps positive selected
  targets — atom_restate 1043 · atom_delete 631 · bond_reorder 551 · bond_reroute 513 · atom_insert 442 ·
  `ring_system_restate` 463 (healthy — only the ~5.7% unrepresentable grow-fused restates removed) ·
  ring_system_delete 256. `unsupervised_after_filter: []`.
- **Launch identity (§4):** the gate prints a `ring_core_launch_identity` line at startup (all frozen hashes +
  flags + `denovo_keep=0` + cache-format v2); eval-cache v1 is unconditionally rejected (v2 + capability-hash
  key). The invariant `teacher ∈ A_exact(x)` is a hard failure condition, not a zero-weight example.

## 6–8. Bounded RingCore preflight — COMPLETE ✅ `GO_AFTER_PRODUCTION_SCHEDULE_CHECK`
Run `compose-v4-ringcore-v1-bounded-e38210b-v1` (commit `e38210b`, A100, 500 steps). First launch (`4204c9d`)
crashed at warm-start on **DEV-CUDA** (`_cksum` called `.numpy()` on a CUDA tensor — the CPU dry-launch could
not catch it); fixed (`.cpu()` before `.numpy()`) and relaunched. NON_SCIENTIFIC_PREFLIGHT.

**§6 learning (fixed diagnostic IMPROVES monotonically):** validation GM loss **3.97 → 2.79 → 2.30**
(step 1/250/500), teacher-mark prob 0.0019 → 0.032 (16×), not early-stopped. **The compositional cycle
families LEARN** (top-3 acc): cycle_close 0.48→0.75, cycle_open 0.15→0.91; atom_insert 0.04→0.85,
atom_delete 0.00→0.56. `ring_system_grow` 0 examples / 0 acc (correctly dead). Some non-dominant families
drop from their B-warmstart values (atom_restate 0.98→0.21, bond_reroute 0.82→0.00, ring_system_restate/delete
→0) — consistent with the family head re-weighting off B's de-novo preferences onto the editing mixture during
warmup + small per-family counts; too early (500 warmup steps) to call a collapse.

**§5 schedule (the decisive caveat): the ENTIRE 500-step run is WARMUP.** `warmup_steps=500=training_steps`, so
LR ramps 6e-7 → 1.5e-4 → 3e-4 and reaches full value only AT step 500 — it never enters the decay phase. The
run validates that the model learns + the mechanics hold, but the loss trajectory is **not representative of the
full-run schedule** (warmup→decay). This is the `GO_AFTER_PRODUCTION_SCHEDULE_CHECK` condition.

**§7 rollout hard gates (trained checkpoint, hardened harness) — ALL PASS:**
- Identity gate PASSED (`identity_ok`, capability `330473e319bfec19`, cycle-ops on, grow off, SHA recorded).
- **hard_gate_violations = {} (zero):** all-state validity **1.0**, cycle-rank correctness **1.0**, charge
  preserved **1.0** (net + protected centers, 0 unexplained mutations), executor/successor match, kernel
  normalized (1.0), no over-40, no legacy-macro access.
- **Grow exclusion:** grow family log-prob −∞, mass 0.0, 0 legal marks, out of capability set.
- **Natural compositional ring editing:** the trained model samples cycle ops in **100%** of rollouts (vs
  uniform 17%) — it LEARNED to prefer ring editing. cycle_close 35 / cycle_open 16, **all productive**,
  `toggle_dominated=False` (net accumulation 7 / loss 4, only 1 close→inverse-open), ring_delete 2.

**§10 macro decision: `MACROS_OPTIONAL_ABLATION`** (confirmed with trained-model evidence). The compositional
core is support-complete (§2: 14/14 topologies), ring restructuring is cheap + in-budget (§3), and the trained
model naturally + productively uses compositional cycle ops without toggling. Macros (P4/RING_HYBRID_V2) would
only accelerate de-novo whole-ring construction (not the dominant editing operation) → optional efficiency
ablation, not a correctness/support prerequisite.

**Training verdict: `GO_AFTER_PRODUCTION_SCHEDULE_CHECK`** — every hard correctness gate passes, the model
learns, and the defining compositional cycle families learn + are naturally + productively used; BUT the
warmup-compressed 500-step schedule cannot authorize the full-run schedule. **Smallest next action:** a
schedule check — a longer run that includes the decay phase (or a warmup+decay schedule with
`schedule_steps > warmup_steps`) — before the full run. NOT `GO_FOR_FULL_RINGCORE_TRAINING` (schedule
unconfirmed); NOT `GO_AFTER_RECIPE_ADJUSTMENT` (cycle ops occur naturally + aren't toggle-dominated; the
non-dominant family drops are warmup re-weighting, to be re-checked with the decay-phase run).

## 6a. Production-scheduler decision — OWNER-LOCKED (2026-07-28) ✅
The full-run LR schedule was **not** previously locked for RingCore: the base recipe declares
`schedule_steps=30000`, but that is base-B's *de-novo* recipe and B (a warm-start fine-tune) self-limited at
~1–3k steps. Over that window a 30000-horizon cosine stays ≥98% of peak — it never decays, so it cannot test
the decay behaviour that blocks full-run authorization. Owner decision:

**RingCore-V1 production scheduler (LOCKED, `scheduler_config_hash = 0b832985c65de1cc`):**
`AdamW` · peak_lr `3e-4` · weight_decay `1e-5` · `warmup_steps=500` · **`schedule_steps=3000`** ·
`minimum_learning_rate_fraction=0.05` · cosine with linear warmup. **500 warmup + 2500 decay.**
- **Not `30000`** (de-novo horizon; LR ~peak over the fine-tuning window, never tests decay).
- **Not `2000`** (compresses decay too aggressively; risks reading rapid-cooling under-training as a
  model/data failure for the *initial* production model).
- Historical B early-stop (~1–3k) is **supporting evidence, not a guarantee** RingCore stops there.

**Expected relative LR trajectory** (fraction of peak): step 0 ≈ 0 · 250 = 0.50 · 500 = 1.00 · 750 = 0.977 ·
1000 = 0.909 · 1500 = 0.672 · 3000 = 0.05. The schedule-check and the eventual full run **share this hash**
(only `training_steps` differs); the gate aborts any RingCore launch whose scheduler args do not reproduce it.

Recorded in: production config (`configs/ringcore_v1_production.json`), config registry
(`scripts/ring_core_identity.py` `PRODUCTION_SCHEDULER`/`SCHEDULER_CONFIG_HASH`), schedule-check baseline
(`diagnostics/coherence/schedule_check_baseline.json`), `SYSTEM_CONTRACT.md`, the paper reproducibility
appendix (`numbers.tex`), and every RingCore checkpoint's metadata.

## 6b. Schedule-faithful check — spec (start fresh from base B) 🔶
`training_steps=1000`, `schedule_steps=3000`, `warmup_steps=500`, **fresh semantic warm-start from base B**
(NOT the superseded step-500 preflight checkpoint — it followed a `schedule_steps=500` all-warmup trajectory).
One-step **GPU smoke first** (catch CUDA-only failures the CPU dry-launch cannot; cf. DEV-CUDA). Eval at steps
**0/250/500/750/1000**, recording LR at each. Purpose: verify the warmup→decay transition; stability under the
production scheduler; continued validation improvement after step 500; survival/recovery of lower-frequency
edit families; family-mass calibration; productive (non-toggle-dominated) compositional cycle editing; and
unchanged rollout correctness. **Not** for convergence or final molecular quality. Verdict targets:
`GO_FOR_FULL_RINGCORE_TRAINING` / `GO_AFTER_RECIPE_ADJUSTMENT` (only for a demonstrated family-distribution
imbalance) / `NO_GO_PRODUCTION_SCHEDULE` (instability from the warmup→decay transition). Do not extend past
step 1000 automatically; a step-1500 continuation under the same scheduler may be authorized as a diagnostic if
lower-frequency-family recovery is genuinely ambiguous.

## 6-prev. Bounded RingCore preflight (≤750 steps) — launch recipe (superseded by the result above)
§6 needs base checkpoint **B** (`compose-v4-stage3-flexible-graft-3k-1ac6f19-v1`, SHA `c9d927…`), which lives
on the `compose-v4-artifacts` Modal volume — so the bounded run is a **Modal** job, not local. All local prep
(§1–§5) is committed; the code gates are green; the launch is a `--disable-ring-grow-macro --cycle-op-mix`
adaptation of `docs/BEDIT_TRAINING_LAUNCH.md`, warm-started + bounded:

```bash
# from a CLEAN detached worktree at tag ring-core-v1; prelaunch_gate green first.
# 1) recompile the training-support cache over the mixed tuple (signature changed: +cycle_op_mix,
#    +disable_ring_grow_macro) — CPU
modal run modal_apps/train_tracelet_gm.py --support-compile-only \
  --corrupted-prior-mix --cycle-op-mix --disable-ring-grow-macro --organic-vocabulary \
  --analogue-trace-pool /artifacts/edit_mining_full_broad/edit_pool_full.jsonl \
  --scaled-manifest /artifacts/edit_mining_full_broad/scaled_edit_data_manifest.json \
  --run-label compose-v4-ringcore-v1-<COMMIT>-v1 \
  --source-run-label compose-v4-stage3-flexible-graft-3k-1ac6f19-v1
# 2) bounded warm-start fine-tune (A100), hard cap 750 steps — nominal 500
modal run --detach modal_apps/train_tracelet_gm.py --train-only \
  --corrupted-prior-mix --cycle-op-mix --disable-ring-grow-macro --organic-vocabulary \
  --analogue-trace-pool /artifacts/edit_mining_full_broad/edit_pool_full.jsonl \
  --scaled-manifest /artifacts/edit_mining_full_broad/scaled_edit_data_manifest.json \
  --corrupted-prior-count <N> \
  --run-label compose-v4-ringcore-v1-<COMMIT>-v1 \
  --source-run-label compose-v4-stage3-flexible-graft-3k-1ac6f19-v1 \
  --initialize-compatible-from-source-checkpoint \
  --initialization-source-run-label compose-v4-stage3-flexible-graft-3k-1ac6f19-v1 \
  --checkpoint-name checkpoint.best_so_far.pt \
  --training-steps 500 --schedule-steps 500
```
Schedule: step 0 diagnostic eval · 250 eval + recovery ckpt + family/subtype gradient audit · 500 eval +
ckpt + fresh-process save/reload + unseen-lead + ring-topology rollout panels. **Do not exceed 750; do not
launch the full run.** Post-run gate: `broad_preflight_gate.py`. Rollout metrics to collect: all-state
validity, cycle-rank correctness, ring-op target counts, natural ring-op usage, topology classes reached,
ring-edit path length, reversal/cycle/return-to-source rates, net structural displacement, canonical
branching, legal-mark enumeration cost, throughput, broad-element/charge preservation.
## 7. Unseen-lead + topology-stratified rollouts 🔶 `ANALYSIS_READY_FOR_RINGCORE_CHECKPOINT`
`scripts/ring_core_rollout_panel.py --checkpoint <ckpt> [--step0-checkpoint <ckpt0>] [--training-log <jsonl>]`
— **hardened per the 10-requirement mandate** and self-validated end-to-end (all sections present, hard gates
all-zero, verdict machinery returns `GO_AFTER_PRODUCTION_SCHEDULE_CHECK` when no training log is supplied —
it does NOT return GO merely because states are valid):
1. **Strict identity** (`ring_core_identity.verify_checkpoint_identity`) — SHA-256, step, source-B hash,
   max_atoms=40, scope, capability `330473e319bfec19`, operator-registry `9197401e8dc3a7ae`, cycle-op
   `27a823aeb6cf7548`, calibration `f534c0233e3bf4a0`, `enable_cycle_ops=true`, `enable_ring_macros=false`,
   `ring_system_grow=false`, sampler-config hash — **fails loudly** on any mismatch. 10 negative tests
   (`test_ring_core_identity_gate.py`). The trainer now self-identifies RingCore checkpoints.
2. **Support-level legacy-macro exclusion** — per state: grow absent from legal enumeration (0 marks), 0
   family mass (log-prob −∞), out of capability set, kernel sums to 1.0. Any access = hard fail.
3. **Charge split** — net_formal_charge_preserved / protected_charged_centers_preserved (slot-tracked) /
   unexplained_charge_mutations (require 0) / charged-context rollout count. Equal net charge alone ≠ pass.
4. **Paired A/B/C** — identical leads+seeds for A (step-0 semantic init), B (step-500 trained), C
   (uniform-legal). A runs when `--step0-checkpoint` is given.
5. **Compositional-ring behaviour** — per-subtype opportunities/mass/counts/distinct-sources/rank-changes;
   productive close/open/delete vs close→inverse-open toggling, repeated toggling, net accumulation/loss;
   **dead-head detection** (no natural cycle op ⇒ `GO_AFTER_RECIPE_ADJUSTMENT`, not GO).
6. **Exact-zero hard gates** — invalid/disconnected/>40 states, illegal actions, executor/successor
   mismatch, unexplained charge, non-normalized kernel, legacy-macro access.
7. **Learning diagnostics** — parses `--training-log` for steps 0/250/500 diagnostic loss + grad/update
   norms (schema finalized against the real log).
8. **Stable 13-metric schema** + provenance (harness commit, lead-panel/seed-list/checkpoint/eval-config
   hashes, `schema_version=ringcore_rollout_v2`).
9. **Interpretation gate** → `GO_FOR_FULL_RINGCORE_TRAINING` / `GO_AFTER_RECIPE_ADJUSTMENT` /
   `GO_AFTER_PRODUCTION_SCHEDULE_CHECK` / `NO_GO_RINGCORE_TRAINING` + smallest required next action.
10. Every artifact tagged **`NON_SCIENTIFIC_PREFLIGHT`** (excluded from paper/benchmark/model-selection).

**On launch:** run against the step-500 checkpoint (+ `--step0-checkpoint`, `--training-log`) → fills §6–§9
and the final report requirements below.

### Final-report requirements to fill from the run (owner mandate)
- Explain exactly what `--schedule-steps 500` controls (LR schedule / warm-up / curriculum / time sampling /
  loss weights) and whether the 0→500 schedule reproduces the first 500 production steps or is a compressed
  coverage/stress schedule (label it honestly).
- Decompose the **61% cycle-family selected-target mass** by MMP/scaffold/corruption/cycle-op layer,
  curriculum bin, subtype, topology stratum, source scaffold, rate-weighted vs unweighted — and state whether
  61% is an enriched warm-up distribution or the intended long-run edit prior.
- Step-0 init diagnostics · step-250 (grad/update norms by block + per cycle subtype + per fresh broad-element
  row, mixture realized-vs-requested, recovery ckpt) · step-500 (eval, save/reload equality, fixed-seed
  trajectory equality, unseen + topology + charged-context rollouts, cost projection).
## 8. Measured throughput ⏳
## 9. Projected full-run cost ⏳
## 10. Decision ⏳ (final call gated on §6)
`CORE_SUFFICIENT_FOR_FULL_RUN` / `MACROS_REQUIRED_BEFORE_FULL_RUN` / `MACROS_OPTIONAL_ABLATION` /
`NO_GO_RINGCORE_CORRECTNESS`.

**Provisional read from §1–§5 (not final — §6 measures the trained-edit distribution + throughput):**
leaning **`MACROS_OPTIONAL_ABLATION`**. Rationale: the compositional core is **support-complete** (§2:
14/14 topologies, no P2 needed) and **correct+robust** (§5: fuzz 0-corrupt, replay/inverse exact,
save/reload); ring **restructuring** — the dominant lead-editing ring operation — is **compositionally cheap
and always within budget** (§3: median 3, 0% over budget, every topology); the cycle family is the dominant,
well-calibratable, drift-free edit family (§4: 61% of targets, mass calibratable 0.30→0.59). The **only**
expensive case is building a *whole new* fused/hetero ring de-novo (§3: median 20, 73.5% over budget) — a
de-novo move, not lead editing; a macro would help there, so macros are a plausible **efficiency ablation**,
not a correctness/support prerequisite. §6 flips this to `MACROS_REQUIRED` only if trained ring edits
routinely exceed the budget or ring edits are rarely completed; to `CORE_SUFFICIENT` if whole-ring
introductions are rare in practice.

## Paper position (§9)
The method is written around the compositional ring core; the theoretical support claim does NOT depend on a
finite macro dictionary. **Precision (R1) — "catalog-independent" is scoped to ring GENERATION, not all ring
support:** ring GENERATION is compositional (`cycle_close`/`cycle_open`, catalog-independent) and ring
RESTATE (`ring_system_restate`) is catalog-independent, but ring DELETION (`ring_system_delete`/`ring_delete`)
remains catalog-bounded to the 5 seed topologies via `enumerate_clean_ring_system_deletes(state, catalog)` — a
live catalog dependency, retained by design. So "macros disabled / catalog acceleration-only" is only
partially realized: whole-ring deletion is a catalog-bounded operator. Macros, if later adopted
(`RING_HYBRID_V2`), are optional learned shortcuts for frequent ring transformations. The compositional-only
model is the **support-complete** variant for ring GENERATION, not a reduced-capability one — macros affect
efficiency and finite-budget reachability. Required ablation: compositional-only / catalog-only /
compositional+macros.
