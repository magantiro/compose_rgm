# RingCore-V1 preflight — bank the compositional ring core

**Mandate:** freeze the supervised compositional ring core as a separately-versioned production candidate
`RING_CORE_V1` (`enable_cycle_ops=true`, `enable_ring_macros=false`, legacy `ring_system_grow` macro
**disabled**), run the bounded preflight, and return a macro decision. **Do NOT** implement P4 into this
production path. **Do NOT** launch the full A100 run — bounded preflight only (≤750 steps).

Ring reachability is defined by the compositional operators; a finite ring catalog is a LATER
(`RING_HYBRID_V2` / P4) acceleration layer, never the definition of support.

Status legend: ✅ done · 🔶 in progress · ⏳ pending.

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

## 6. Bounded RingCore preflight (≤750 steps) 🔶 READY TO LAUNCH (bounded Modal run)
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
finite macro dictionary. Macros, if later adopted (`RING_HYBRID_V2`), are optional learned shortcuts for
frequent ring transformations. The compositional-only model is the **support-complete** variant, not a
reduced-capability one — macros affect efficiency and finite-budget reachability. Required ablation:
compositional-only / catalog-only / compositional+macros.
