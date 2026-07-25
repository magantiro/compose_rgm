# Generator lineage map — definitive, evidence-backed audit

**Status:** read-only audit, 2026-07-23. **Author tooling:** git history (all branches) + on-disk
SHA256 of every checkpoint + full code read + docs/diagnostics cross-check.
**Scope:** the COMPOSE / RGM molecular generator lineage in
`compose_rgm_claude_generators`. Every factual claim below cites a commit SHA and/or a
`file:line`. Where a claim could not be verified, it is marked **[UNVERIFIED]** rather than guessed.

> How to read the SHAs. "SHA `c9d927…`" is a **checkpoint content hash** (`shasum -a 256` of a
> `.pt` file on disk). "commit `b41a8bd`" is a **git commit**. These are different namespaces and
> are never interchangeable. Checkpoint binaries are **not** tracked in git — none exist under the
> repo tree (`git ls-files '*.pt'` → empty; `find . -name '*.pt'` → empty). Every generator
> checkpoint lives either in `/private/tmp/…` (local, ephemeral) or on the Modal
> `compose-v4-artifacts` volume (remote). This is the single biggest provenance hazard and the
> root of the confusion this document exists to end (see §6).

---

## 0. TL;DR — the answers

1. **Canonical / shipped base generator = "Lineage B"**, the quotient-correct flexible-Graft +
   whole-ring-system model, **step-1,000 `best_so_far`**, SHA
   **`c9d927510360ec6eb84ff8dae1a222b0b693a9bef0ca23bb5d9cca063025876c`** (38,387,536 bytes).
   On disk at `/private/tmp/lineage_b_checkpoint/checkpoint.best_so_far.pt`; on Modal at
   `compose-v4-artifacts:compose-v4-stage3-flexible-graft-3k-1ac6f19-v1/checkpoint.best_so_far.pt`.
   Evidence: `README.md:91-93,109`; `docs/GENERATOR_RESULTS_SUMMARY.md:10-16`;
   `docs/HANDOFF_LIPID_GENERATOR_FROM_GENERATORS.md:88`; every paper/editing script defaults to it
   (§4); decision commit `b41a8bd` (2026-07-20).

2. **step-1,000 vs step-2,500 — resolved: step-2,500 is NOT confirmed better. It is
   schedule-confounded, still defective, and was never promoted.** step-2,500 is numerically better
   (small-ring 51%→40%, matched-CNOF FCD 23.044→19.180) but on the *same 3,000-step cosine schedule
   whose LR had already decayed to 4.22e-5*, so "a new scientific conclusion must not be drawn"
   (`docs/audits/2026-07-19_pancake_to_quotient_run_audit.md:38-42`). The fresh-optimizer
   continuation "failed cleanly" (val loss rose 11.69→12.01→12.08→12.08, early-stop;
   audit `:203-206`) and the legacy-transfer rescue also lost to the incumbent (`:245-251`). The
   **pinned** base is the step-1,000 `c9d927…`, never the step-2,500. See §3.

3. **Correction to the "bb53e00" anchor.** `bb53e00…` is **real and on disk** — it is the SHA256 of
   **both** `/private/tmp/quotient_best.pt` **and** `/private/tmp/step2500_checkpoint.pt`, which are
   **byte-identical** (`cmp` clean; 38,387,728 bytes each). It is the step-2,500 "selected quotient"
   inference checkpoint. It is **absent from git history, docs, and any tracked file** — an untracked
   local artifact. (A git-only search will report it "not found"; that search is incomplete, not
   evidence the file is fake.)

4. **Other raw-`t` bug sites (same class as the `value_guided_smc` bug you already found):** at least
   **six** additional call sites pass raw operational time (which reaches 10–16) to the model instead
   of `frozen_time ∈ [0,1]`. Full table in §4.3. The canonical ancestral eval path is **correct**.

5. **Biggest surprises:** (a) the canonical base has an **uncommitted working-tree fix** for the
   frozen_time bug right now; (b) the conditional QED lane is built on the **pancake (A)** base, not
   Lineage B, and its steering is measured-weak; (c) the pancake-6,250 checkpoint exists in **four
   byte-identical copies** under four different names; (d) **two** live docs disagree on the canonical
   base (the older ones still say pancake — they are explicitly superseded).

---

## 1. Checkpoint inventory

### 1a. Checkpoints that EXIST ON DISK NOW (SHA256 computed 2026-07-23)

| Canonical name | Path (all under `/private/tmp/`) | SHA256 | Bytes | Step | Role |
|---|---|---|---|---|---|
| **Lineage B (quotient, step-1000)** — CANONICAL BASE | `lineage_b_checkpoint/checkpoint.best_so_far.pt` | `c9d927510360ec6eb84ff8dae1a222b0b693a9bef0ca23bb5d9cca063025876c` | 38,387,536 | 1,000 | Shipped unconditional base |
| **Quotient step-2,500** (`quotient_best.pt`) | `quotient_best.pt` | `bb53e00237bd4be60436e149c7f92cd75b30ce555ae96d9264164e303150abb1` | 38,387,728 | 2,500 | Later ckpt of B's run; **not promoted** |
| **Quotient step-2,500** (dup, `step2500_checkpoint.pt`) | `step2500_checkpoint.pt` | `bb53e00237bd4be60436e149c7f92cd75b30ce555ae96d9264164e303150abb1` | 38,387,728 | 2,500 | **Byte-identical to `quotient_best.pt`** |
| **Pancake / Lineage A (step-6250)** | `compose-pancake6250-checkpoint.pt` | `47716924f7798ed24556c5aa8fb10c533c55dbd1f02f8f53a463cf2ad80ae2bf` | 100,130,172 | 6,250 | Legacy pre-quotient base; conditional incumbent |
| Pancake A (dup 2) | `compose_pancake_checkpoint/checkpoint.recovery.pt` | `47716924…ae2bf` | 100,130,172 | 6,250 | **Byte-identical** to pancake6250 |
| Pancake A (dup 3) | `pancake_checkpoint/checkpoint.recovery.pt` | `47716924…ae2bf` | 100,130,172 | 6,250 | **Byte-identical** (script default for conditional scripts) |
| Pancake A (dup 4) | `pancake_recovery.pt` | `47716924…ae2bf` | 100,130,172 | 6,250 | **Byte-identical** |
| **"current2500" recovery** | `compose_current2500/checkpoint.recovery.pt` | `071431d4d6d17ec76b8d016133ae5eecfbc97a945f639e1f0162fbf7716527e2` | 113,567,252 | 2,500 | Recovery (optimizer-state) form of a step-2,500; run-ID untracked **[UNVERIFIED same run as `bb53e00`]** |
| **QED-conditioned sidecar (step-500)** | `compose-v4-qed-conditioned-step500.pt` | `e0624f163c53893fcdcbbb565c4aae5aacad7b3b4247f0d1a45741e8f61bb229` | 38,656,540 | 500 | Conditional; **init from pancake A** (§3.4) |
| QED-conditioned step-500 recovery | `compose-v4-qed-conditioned-step500.recovery.pt` | *(not hashed; recovery twin)* | — | 500 | recovery form |
| Property-conditioned smoke | `compose-property-conditioned-smoke.pt` | `416266a4e55905a90b500faaa0b7e4dd13b0fb3e4163c1dbcf2bb35a61cef033` | 146,621 | smoke | Tiny smoke-test checkpoint |
| Property-conditioned smoke (best) | `compose-property-conditioned-smoke.best_so_far.pt` | `8cf0ecb952e09a55685869dd909377b465455886834dae6087b8d8e71257a0c8` | 146,149 | smoke | Tiny smoke-test checkpoint |
| Distillation head-delta | `canonical_successor_qualification_probe/distilled_head_delta.pt` | `d1ae1ba791d18ba6df0623b49ba6bff89691da989bd64b6eab22e5afe573dc22` | 541,571 | — | Head delta from pancake→quotient distillation (§5) |

Notes:
- **Pancake-6250 exists in 4 byte-identical copies** (SHA `47716924…`), **all exactly 100,130,172
  bytes** (`compose-pancake6250-checkpoint.pt`, `pancake_recovery.pt`,
  `compose_pancake_checkpoint/checkpoint.recovery.pt`, `pancake_checkpoint/checkpoint.recovery.pt`) —
  one content hash, four names. They carry optimizer state (recovery-format), hence 100 MB vs the
  38 MB inference-only Lineage-B / step-2500 checkpoints.
- **`quotient_best.pt` and `step2500_checkpoint.pt` are the same bytes** under two names.
- The **only** difference between the canonical base (`c9d927…`, 38,387,536 B) and the step-2,500
  quotient (`bb53e00…`, 38,387,728 B) is 192 bytes — consistent with two steps of the **same
  architecture** (same model family), reinforcing that step-2,500 is a later checkpoint of Lineage B's
  run. **[UNVERIFIED]** by manifest: no run-ID sidecar binds `bb53e00…` to
  `compose-v4-stage3-flexible-graft-3k-1ac6f19-v1`; the neighboring `step2500_preview_metrics.json`
  and `step2500_ring_calibration_12.json` are the only local provenance.

### 1b. Checkpoints REFERENCED ONLY (not on local disk; Modal volume or lost)

| Name | Location referenced | SHA / step | Source |
|---|---|---|---|
| **Lineage C — factorized-tree** | `compose-v4-artifacts:/artifacts/tree_fcd_transfer_stage1_factorized_v1/checkpoint.pt` (+`.recovery.pt`) | step-7,200 (trained to 8,000); **no SHA published** | `HANDOFF_GENERATOR_RUN_LINEAGE_CORRECTION_V2.md:173-174,349-372` |
| QED-conditioned step-1000 continuation (selected) | `/artifacts/compose-v4-qed-conditioned-step1000-continuation-20260720-v1/checkpoint.pt` | SHA `76ddbba66838d4e3173a9e9278aea8453d54cb845384ea51241590773452b5c4`, step-1,000 | `diagnostics/qed_step1000_continuation_status.json:36-37` |
| QED-conditioned pilot v4 step-500 recovery (init source) | `/artifacts/compose-v4-qed-conditioned-pilot-20260720-v4` | SHA `6e5d14e4ca3d110aec8ae93cd43f3d5d03b5f761bcc66670a6a2592f6c91eebc` | `diagnostics/qed_step1000_continuation_status.json:5-6` |
| QED frozen-residual sidecar v5 (canonical) | `/artifacts/compose-v4-griddd-qed-frozen-residual-pilot-20260720-v5-canonical/checkpoint.best.pt` | best step 500, val loss 11.1448; config SHA `6a63c154…fba01` | `docs/CLAUDE_GENERATOR_HANDOFF_MANIFEST_V1.json:20-34`; `HANDOFF_CLAUDE_GENERATORS.md §13.2` |
| "Combined chem+topology" fine-tune | `compose-v4-uncond-chem-topology-train-20260720-v1` | selected_step 400 | commit `fca5bd5`; `HANDOFF_LIPID…:234` |
| Local GM gate (from scratch) | `results/tracelet_gate_from_scratch_seed20260714.pt` | seed 20260714 | `results/*ring_restate*` JSONs; **not on disk** |
| Local quotient-kekulé gate | `results/tracelet_quotient_kekule_800_seed20260714.pt` | 800-step | `results/…kekule…json`; **not on disk** |
| Tiny-rate exactness gate | `results/tiny_rate_gate.pt`, `results/tiny_rate_quotient_pilot.pt` | overfit gate | `scripts/overfit_tiny_rate_model.py:33`; **not on disk** |
| CNOF conditional gate | `results/cnof_gate.pt`, `results/cnof_gate_late.pt` | gate | `scripts/train_cnof_conditional_gate.py:80`; **not on disk** |
| CNOF scale-16 ablations | `cnof_scale16_random.pt`, `…prior_tilt.pt`, `…topology_deferred.pt` | ablation | `results/README.md:41-53`; **not on disk** |

---

## 1c. Complete training-run ledger (every run + its honest result + what used it)

Modal run labels enumerated by grepping every `compose-v4-*` / `tree_fcd_*` token across
`docs/ diagnostics/ configs/ scripts/ src/ modal_apps/ results/`. Honest results are from the
cited docs/diagnostics; **[UNVERIFIED]** marks anything I could not confirm from a primary artifact.

**Unconditional base lineage**

| Run label | Lineage | Step selected | Honest result | Used by |
|---|---|---|---|---|
| `compose-v4-stage3-full-ring-hierarchical-v1` | **A / pancake** | 6,250 (`47716924…`) | Best FCD (11.288 matched-CNOF) **but** thrashes: 76.07% events Graft/reroute, one trace with 89 canonical self-events, rings dumped in last decile (pos 0.978), small-ring 29.9% vs 5.26% ref. Needs `-0.5/-1.5` calibration band-aid. `HANDOFF_GENERATOR_RUN_LINEAGE_CORRECTION_V2.md:60-100`; `diagnostics/pancake_step6250_eval2000_event_audit.json` | Conditional-QED scripts; `griddd_valid_fiber_controller_panel.py` default |
| `compose-v4-stage3-flexible-graft-3k-1ac6f19-v1` (commit `1ac6f19`) | **B / quotient** | 1,000 `best_so_far` (`c9d927…`); also 1,500, 2,500 (`bb53e00…`) | step-1,000: 0 canonical self-events / 2,750, 0 delete-to-one, fused 48% vs 52.18% — clean editor, **but** small-ring 51%, only n=100, FCD 23.044 matched; **3,000-step cosine schedule confounded** (LR still decaying). `…CORRECTION_V2.md:102-168` | **Canonical base**; all paper/editing scripts (§4) |
| `compose-v4-stage3-step1000-preview100-v1` | B (preview eval) | 1,000 | 100-sample preview eval of the pinned base | `…CORRECTION_V2.md:108-112` |
| `compose-v4-stage3-flexible-graft-prod-2be9258-v1` (commit `2be9258`) | B (prod attempt) | — | **FAILED**: duplicate full-set validation OOMed the 40 GB GPU, **no optimizer updates**. `docs/HANDOFF.md:73-78`; `CURRENT_MODEL.md:110-112` (pre-failure snapshot) | — (dead) |
| `compose-v4-stage3-flexible-graft-prod-7ea89d0-v1` (commit `7ea89d0`) | B (prod attempt) | — | Prod/preflight attempt **[UNVERIFIED result]**; appears in `docs/PROJECT_STATUS.md` | — |
| `compose-v4-stage3-exact-support-preflight-08fe2b1-v1` (commit `08fe2b1`) | B (preflight) | — | Exact-support compile preflight **[UNVERIFIED result]** | — |
| `compose-v4-stage3-best6250-eval2000` | A (eval) | 6,250 | 2,000-sample eval of pancake (produces the event audit) | `diagnostics/pancake_step6250_eval2000_event_audit.json` |
| `tree_fcd_transfer_stage1_factorized_v1` | **C / factorized-tree** | 7,200 (to 8,000) | FCD 14.577, family acc 89.5%, fused 52.2% (near ref) — good topology **but** severe initial delete-collapse (mean 19.18-event delete run) and off-thesis ring-ear substrate. `…CORRECTION_V2.md:170-219` | Comparison arm only |

**Ring-fix / chemistry pilots (all warm-started from pancake A, none blessed)**

| Run label / commit | What it changed | Honest result |
|---|---|---|
| `compose-v4-unconditional-ring-topology-pilot-20260720-v1/-v2` (commit `526811c`) | `ring_template_factorization="topology_cycle_hierarchical"` head | Fixes topology **mix** (fused 0.09→0.168, small-ring 0.525→0.351) but **cycle_rank unchanged** (2.46 vs 3.35). "repairs the topology MIX … but not total cyclization." Not promoted. |
| `compose-v4-uncond-chem-topology-train-20260720-v1` (commit `fca5bd5`) | chemistry + topology heads jointly (selected_step 400) | "improved, not solved": triples 2.7× ref, fused still 2.7× too low, small-ring ~5×, cycle_rank unchanged. `diagnostics/unconditional_combined_chem_topology_nocalib_eval180.json`. Not promoted. |
| `compose-v4-unconditional-chemistry-{marks-only,contextual}-20260720-v1`; `…-p1p2-pilot-…-v3` | P1/P2 empirical-prior + topology-mass corrections | Negative/partial evidence; "Do not promote them over the three primary lineages without a matched panel." `…CORRECTION_V2.md:244-262` |
| `compose-v4-unconditional-ring-topology-paths-20260720-v1` (app `ap-R32pM7XOzNxeptbjNmWfuM`) | CPU path-compile only | Infra/compile run, not a model |
| **§9.3 fix (superposed + topology_cycle_hierarchical)** (commit `b41a8bd`) | `rate_factorization="superposed"` warm-started from **B** | **Mechanism proven by unit test only** (`tests/test_topology_group_intensity_ring_hazard.py`). **No trained checkpoint — Modal failed 3×.** `GENERATOR_RESULTS_SUMMARY.md:43-60` |

**Conditional (QED) lineage — all init from pancake A's `…full-ring-hierarchical-v1/checkpoint.recovery.pt`**

| Run label | Step | Honest result | Artifact |
|---|---|---|---|
| `compose-v4-qed-conditioned-pilot-20260720-v4` | 500, 1000 | Weak direct steering: target-0.9 mean 0.6418 vs 0.5745 control (+0.067), **0/10 at QED≥0.9**. `2026-07-20_conditional_backbone_selection.md:27` | `compose-v4-qed-conditioned-step500.pt` (`e0624f16…`, on disk) |
| `compose-v4-qed-conditioned-step1000-continuation-20260720-v1` | 1000 (`76ddbba6…`) | val loss 11.0227; direct target-0.9 shift **+0.0196, 0/10 at QED≥0.9**; late-window +0.0636, 2/10 but CI includes 0. "Do not select as final conditional backbone." `…selection.md:28,33-35` | Modal `/artifacts/…` |
| `compose-v4-griddd-qed-frozen-residual-pilot-20260720-v1…v5-canonical` | v5 best 500 | v5 completed, best val loss 11.1448; the "v5 sidecar." Trained QED sidecar showed **NO molecule-level efficacy** (0/5 lift) — commit `a9bcbfb` | Modal `/artifacts/…` |
| `compose-v4-qed-conditioned-step{500,1000}-target0{70,90}-eval10-…`, `…-dose1{10,30}-…`, `…-successor-gate100-…`, `…-controlled090-late13to24-…` | eval-only | Dose/target/window response evals feeding `diagnostics/qed_step{500,1000}_*` | `diagnostics/qed_*` |

**Local gates / ablations (results/*.pt — referenced, not on disk; produced by `scripts/`)**

| Checkpoint | Producer | Purpose / result |
|---|---|---|
| `results/tiny_rate_gate.pt`, `tiny_rate_quotient_pilot.pt` | `scripts/overfit_tiny_rate_model.py` | Exactness overfit gate (E0-adjacent). Evals: `results/tiny_rate_gate_eval_*.json` |
| `results/cnof_gate.pt`, `cnof_gate_late.pt` | `scripts/train_cnof_conditional_gate.py`, `train_tracelet_cnof_gate.py` | CNOF conditional gate. Evals: `results/cnof_gate_*_diagnostics.json` |
| `results/tracelet_gate_from_scratch_seed20260714.pt` | `scripts/train_tracelet_gm.py`-family | GM training gate; ring-restate audits |
| `results/tracelet_quotient_kekule_800_seed20260714.pt` | (kekulé quotient gate) | Aromatic-closure ambiguity audits |
| `cnof_scale16_{random,prior_tilt,topology_deferred}.pt` | scale-16 driver | CNOF scale-16 ablations (`results/cnof_scale16_*.json`) |

---

## 1d. Complete eval / diagnostic ledger (what each measures; honest headline)

Committed under `diagnostics/` (result JSONs). Headlines are the *honest* status per the source docs.

- **Exactness (rigor foundation — PASSED):** `diagnostics/exactness/e0_toy_h_{cap3,contains_N,size_max}.json`,
  `doob_guidance_ground_truth.json`. Driver `scripts/e0_toy_h_exactness.py` (self-contained, **no
  checkpoint** — `README.md:97`), `scripts/doob_guidance_ground_truth.py`. Commit `375761e` /
  `2be5d28`.
- **Value-guided SMC / conditional (measured V0):** `diagnostics/conditional_smc/` (17 files) —
  `smc_panel12_zerocalib_b640.json` (+`_b1000`, `_b640_twist`), `tier1_pathwise_safety.json`,
  `physchem_box.json`, `property_dial.json`, `scaffold_opt_panel12.json`, `anytime_panel12.json`,
  `usable_oracle_efficiency.json`, `conjunction_funnel.json`, `constrained_diversity_diagnostic.json`,
  `lineageB_zerocalib_sufficiency.json`. Honest V0 headline: success **25% (3/12)**, mean best QED
  **0.880**, population diversity **0.62** (`GENERATOR_RESULTS_SUMMARY.md:68-71`). Pathwise-safety
  67%→0% (magnitude base-dependent, guarantee base-robust — `README.md:117-120`).
- **Base decision panel (measured):** `diagnostics/griddd_valid_fiber_controller_panel12_ceiling.json`
  (pancake) vs `…_panel12_lineageB_ceiling.json` (B). B **33.3% vs 16.7%** (`…LIPID…:70-84`).
- **Pancake characterization:** `diagnostics/pancake_step6250_eval2000_event_audit.json`,
  `pancake6250_calibration_eval600_metrics.json`, `pancake_ring_*`, `lineageB_calibration_ablation.json`.
- **Reachability (paper spotlight — barrier survived, recovery retracted):**
  `diagnostics/reachability/{pareto_reachability_cap4,cap5,barrier_interior_boundary,barrier_objective_robustness,frontier_recovery_cap4}.json`
  (`.claude/context/learnings.md` 2026-07-22).
- **Ring calibration:** `diagnostics/ring_calibration/step2500_exact_support_audit_12.json` (the
  step-2,500 support-vs-logit diagnostic; 3,092 templates, 285 small-ring, 3.017% prior mass).
- **Canonical-successor backbone qualification — FAILED:**
  `diagnostics/canonical_successor_backbone_qualification.json` (`qualification_passed:false`,
  source `47716924…`, weight_source `step6250_pancake_plus_panel_quotient_distilled_family_hazard_heads`).
- **Conditional gate diagnostics:** `diagnostics/qed_step500_*`, `qed_step1000_*`,
  `griddd_trained_sidecar_molecule_qed_smoke.json`, `griddd_valid_fiber_controller_panel12_*`.

---

## 2. Lineage tree (commit-cited)

```
eaf328b (07-18) import COMPOSE RGM snapshot
   │
6f6bc96 (07-18) preserve pre-quotient Graft churn  ← the pre→post "quotient" boundary
   │
   ├─ Lineage A "pancake": run compose-v4-stage3-full-ring-hierarchical-v1
   │     step-6250  →  47716924…                 (hierarchical ring templates = the small-ring defect)
   │
fa5483c (07-19) archive step-1000 diagnostics (Lineage B preview)
c82e270 (07-19) controlled fresh-optimizer continuation   ← later REJECTED
c30ef12 (07-19) bounded compatible-checkpoint fallback    ← main branch tip
   │
   ├─ Lineage B "quotient": run compose-v4-stage3-flexible-graft-3k-1ac6f19-v1
   │     step-1000 best_so_far → c9d927…  ← PINNED CANONICAL BASE
   │     step-1500, step-2500 → bb53e00… (quotient_best.pt == step2500_checkpoint.pt)  NOT promoted
   │     + recovery: 071431d4… (compose_current2500)
   │     prod attempts: 2be9258 (OOM, 0 updates), 7ea89d0, 08fe2b1 preflight
   │
   └─ Lineage C "factorized-tree": run tree_fcd_transfer_stage1_factorized_v1
         step-7200 (to 8000) → [no SHA published]   comparison arm only

5088053 (07-20 13:42) snapshot RGM + lipid; first pancake6250/step2500/quotient strings;
                       sets PANCAKE defaults in conditional scripts   ← agent-snapshot parent
8740349 (07-20 14:34) Rev-1 handoff: PANCAKE = "retained incumbent"  ← agent/compose-rgm-full-snapshot tip
526811c (07-20 17:17) ring-fix #1 (topology head)   — mix fixed, cyclization not
fca5bd5 (07-20 18:00) ring-fix #2 (chem+topology)   — improved, not solved
517889f (07-20 19:30) valid-fiber controller (A-vs-B panel)   ← manifest-V2 observed_head
62e4321 (07-20 20:28) merge gen→lipid (only up to 517889f, i.e. PRE-decision)
b41a8bd (07-20 20:39) ★ DECISION: §9.3 ring fix + "Lineage-B base decision" (base = B, not pancake)
b1e87b1 (07-20 20:42) lipid branch independently adopts Lineage B
642f111 (07-21)       conditional suite (value-guided SMC); FIRST hard-codes lineage_b default in scripts
89fdb17,bd6a91e (07-21) corpus provenance + reproducibility fields
ca9fce9 (07-21)       commit V2 lineage-correction doc + pristine README (base = B)
… (07-22) control-substrate / Pareto-editing reframe
```

**Branch tips and whether they carry the decision (`b41a8bd`):**

| Branch | Tip | Has `b41a8bd`? | Canonical stance |
|---|---|---|---|
| `main` | `c30ef12` (07-19) | **No** | Pre-decision; no A/B/C blessing |
| `agent/compose-rgm-full-snapshot-20260720` | `8740349` (07-20) | **No** | Rev-1: **pancake** incumbent |
| `claude/generator-cond-uncond` | `e601521` (07-22) | **Yes** | Uncond/editing = **B**; conditional = pancake |
| `claude/control-closed-pareto-editing` | `f98e759` (07-22) | **Yes** | Same as above (**checked out at audit time**, with uncommitted changes) |
| `claude/lipid-corpus-oracle` | `d3ee593` (07-21) | **No** (reached B independently via `b1e87b1`) | B by parallel adoption; its `HANDOFF_LIPID…` is still **Rev-1** |

> **Merge hazard:** the authoritative decision + `docs/HANDOFF_GENERATOR_RUN_LINEAGE_CORRECTION_V2.md`
> live **only** on `generator-cond-uncond` and `control-closed-pareto-editing`. `main`, the snapshot
> branch, and `lipid-corpus-oracle` still present pancake or have no blessing. A human should merge the
> decision forward (§6).

---

## 3. THE canonical / latest generator (definitive)

**Canonical unconditional base = Lineage B, step-1,000 `best_so_far`, SHA `c9d927…5876c`.** Trail:

1. `README.md:91-93` — "base checkpoint (Lineage B) … `compose-v4-artifacts :
   compose-v4-stage3-flexible-graft-3k-1ac6f19-v1/checkpoint.best_so_far.pt` … scripts default to
   `/private/tmp/lineage_b_checkpoint/checkpoint.best_so_far.pt`"; `:109` "Base = Lineage B … a
   **step-1,000 preview** of a 3,000-step schedule."
2. `docs/GENERATOR_RESULTS_SUMMARY.md:10-16` — "Selected **Lineage B** … over the pre-quotient pancake
   (A) and factorized-tree (C). Checkpoint: … SHA `c9d927510360ec6e…` (step-1,000 best_so_far,
   verified)."
3. `docs/HANDOFF_LIPID_GENERATOR_FROM_GENERATORS.md:88` — full SHA + **38,387,536 bytes** (matches my
   on-disk hash exactly).
4. Code: every paper/editing driver hard-codes `…/lineage_b_checkpoint/checkpoint.best_so_far.pt`
   (§4.1). Decision commit `b41a8bd` (2026-07-20).

### 3.1 step-1,000 vs step-2,500 — definitively resolved

**step-2,500 (`bb53e00…`) is NOT confirmed better. It is schedule-confounded, still defective, and was
never promoted.** Evidence, all primary:

- **Numbers did improve** at step-2,500 (n=100): small-ring 51%→40%, full FCD 28.688→24.708,
  matched-CNOF 23.044→19.180 (`…CORRECTION_V2.md:145-151`;
  `docs/audits/2026-07-19_pancake_to_quotient_run_audit.md:180-183`).
- **…but on a confounded schedule:** "the 3,000-step cosine schedule is confounded … its learning rate
  had already fallen from 3e-4 to 4.22e-5 while validation loss was still improving … **A new
  scientific conclusion must not be drawn from this schedule mismatch**" (audit `:38-42,184-185`).
- **The clean-improvement attempts failed:** the fresh-optimizer continuation "failed cleanly … validation
  loss rose from 11.6914 to 12.0112, 12.0760, and 12.0837, and early stopping fired" (audit `:203-206`);
  the legacy-transfer rescue "did not resolve the failure … the selected quotient incumbent remains
  better at loss 11.6977 and 58.08%" (audit `:245-251`). 48h tracker U1: "Completed, **no promotion**"
  (`48_HOUR_RESULTS_TRACKER.md:16`).
- **The defect persists** at step-2,500: 40% of molecules still carry a 3/4-member ring vs 5.26% ref,
  every new small ring localized to `ring_system_grow` (audit `:224-235`).
- **The pinned checkpoint is step-1,000**, `c9d927…` — never step-2,500 — in every current doc.

**Conclusion:** step-2,500 is a *directionally interesting but scientifically confounded, unblessed*
later checkpoint of Lineage B's run. The canonical base remains step-1,000. Do **not** silently swap in
`bb53e00…`.

### 3.2 The one open defect on the canonical base

Lineage B **overproduces small rings** (~49–51% carry a strained 3/4-member ring vs 6% reference;
`GENERATOR_RESULTS_SUMMARY.md:29-41`). Root cause: the Boolean "ring family enabled" head renormalizes
rare small-ring templates to probability one. The **§9.3 fix** (`rate_factorization="superposed"` +
`ring_template_factorization="topology_cycle_hierarchical"`) is **mechanism-proven by unit test only**;
**no trained ring-fixed checkpoint exists** (Modal failed 3×; `:43-60`). `CLAUDE_GENERATOR_READ_NOW.md:12-13`:
"do not assume that rings are globally solved."

### 3.3 What is NOT the canonical base (explicit)

- **Pancake / step-6,250 (`47716924…`)** is the *legacy pre-quotient* base and remains the *conditional*
  incumbent, but is **not** the unconditional base. `CLAUDE_GENERATOR_READ_NOW.md:12-13`: "Do not assume
  that the step-6,250 pancake checkpoint is the correct final backbone."
- **Factorized-tree / step-7,200 (Lineage C)** is a comparison arm only.
- **The quotient-distilled canonical-successor backbone FAILED qualification**
  (`diagnostics/canonical_successor_backbone_qualification.json`: `qualification_passed:false`).

### 3.4 The conditional generator is a SEPARATE lineage (built on pancake, not B)

The QED-conditioned models init from `compose-v4-stage3-full-ring-hierarchical-v1/checkpoint.recovery.pt`
(pancake A), **not** Lineage B (`2026-07-20_conditional_backbone_selection.md:9-17`). Their steering is
**measured-weak** (target-0.9 shift +0.02…+0.07, 0/10 at QED≥0.9; `:27-28`), and the doc says "Do not
select it as the final conditional backbone" (`:33-35`). This matches the standing memory that
"generator conditioning is weak." The deployable conditional method is instead the **value-guided SMC
controller over B's CTMC** (`GENERATOR_RESULTS_SUMMARY.md §4`), which is mechanism-driven and
base-independent.

---

## 4. Code map (define / load / sample / train / eval) + the time-convention audit

### 4.1 Key symbols (file:line)

| Role | Symbol | Location |
|---|---|---|
| Base rate model | `class FactorizedTraceletRateModel` | `src/compose_v4/model/factorized_tracelet_rate_model.py` (`sample_rewrite_mark` :2938, `_conditioned` :2955) |
| Canonical direct sampler | `class AnalyticPancakeQuotientSampler` | `src/compose_v4/experiments/canonical_successor_distillation.py:568` (`rate_table` :605, `sample_rewrite_mark` :620) |
| Loader (controller) | `_load_base_sampler` | `scripts/griddd_value_guided_smc_controller.py:60` → wraps `load_factorized_rollout_checkpoint` + `AnalyticPancakeQuotientSampler` |
| Loader (fiber panel) | `_load_base_sampler` | `scripts/griddd_valid_fiber_controller_panel.py:37` |
| De-novo ancestral sampler (CANONICAL) | `sample_tracelet_ancestral` | `src/compose_v4/experiments/tracelet_conditional.py:1119` |
| Parallel ancestral wrapper | `sample_tracelet_ancestral_many` | `src/compose_v4/experiments/parallel_tracelet_sampling.py:52` (delegates to `sample_tracelet_ancestral` :90) |
| SMC controller | `value_guided_smc` | `scripts/griddd_value_guided_smc_controller.py:114` |
| Training entrypoint (Modal) | `train_tracelet_gm.py` | `modal_apps/train_tracelet_gm.py` (guacamol vol :105, artifact vol :106, ckpt :134/380, **step-1000 preview gate** :250,403) |
| Training entrypoint (local gate) | `train_tracelet_cnof_gate.py`, `train_cnof_conditional_gate.py`, `overfit_tiny_rate_model.py` | `scripts/` |
| Eval / rollouts | `evaluate_tracelet_rollouts.py` | `scripts/evaluate_tracelet_rollouts.py` (`sample_tracelet_ancestral_many` :394; `load_factorized_rollout_checkpoint` :119) |
| Paper drivers | — | `scripts/tier1_pathwise_safety.py`, `physchem_box.py`, `property_dial.py`, `pathwise_precheck.py`, `pareto_editing_hero.py`, `e0_toy_h_exactness.py`, `doob_guidance_ground_truth.py`, `make_paper_figures.py` |

**Checkpoint→script binding (code-verified defaults):**
- **Lineage B** (`/private/tmp/lineage_b_checkpoint/checkpoint.best_so_far.pt`):
  `tier1_pathwise_safety.py:48`, `physchem_box.py:49`, `property_dial.py:43`, `pathwise_precheck.py:38`,
  `pareto_editing_hero.py:61`, `griddd_value_guided_smc_controller.py:330`,
  `griddd_reward_finetune_train.py:83`, `denovo_learned_vs_uniform.py:52`,
  `composition_learned_vs_uniform.py:47`, `collect_twist_trajectories.py:86`.
- **Pancake A** (`/private/tmp/pancake_checkpoint/checkpoint.recovery.pt`):
  `griddd_valid_fiber_controller_panel.py:136`, `qualify_analytic_pancake_quotient_backbone.py:31`,
  `run_griddd_analytic_zero_sidecar_smoke.py:50`.

### 4.2 The `frozen_time = 1 - exp(-t/2)` convention — where correct, where missing

The model's time feature must be in **[0,1]**. The canonical map is
`frozen_time = 1 - exp(-(operational_time + interval_end)/2)` (midpoint of each step, "k=2").

**Correctly applied (k=2):**
- `src/compose_v4/experiments/tracelet_conditional.py:1155` → fed to the model at `:1161` and `:1176`
  (this is the canonical `sample_tracelet_ancestral`; **the base eval path is correct**).
- `src/compose_v4/experiments/tiny_rate.py:492` → `:507`.
- `src/compose_v4/experiments/cnof_conditional.py:488` → `:506`.
- `scripts/diagnostics/render_compose_full_trajectories.py:51` → `:53`.
- `scripts/griddd_value_guided_smc_controller.py:48-57` (`_frozen_time`) → `:175,:232`
  — **UNCOMMITTED working-tree fix** (see §4.4).

**Separate "k=1" map `1 - exp(-t)` (still in [0,1] — NOT the raw-t bug, but a convention
inconsistency worth reconciling):**
- Training-label generation: `tracelet_conditional.py:508`, `cnof_conditional.py:230`,
  `factorized_mark_conditional.py:190` (late-time example labels).
- Inference/guidance eval: `scripts/evaluate_qed_successor_guidance.py:119`
  (`model_time = 1.0 - exp(-operational_time)`) — uses k=1 where the ancestral path uses k=2.
  Both stay in [0,1]; flag for a human to standardize k.

### 4.3 Raw-`t` bug sites (pass RAW operational time — which reaches 10–16 — to the model)

`sample_rewrite_mark` / `rate_table` expect the **frozen_time [0,1]** feature. Passing raw operational
time (which exceeds 1 once t > ~1.4, and the horizons here are 16) is out-of-distribution and silently
collapses the productive hazard. Every call site was inspected:

| # | file:line | Time passed | Horizon / range | Verdict |
|---|---|---|---|---|
| 1 | `scripts/griddd_value_guided_smc_controller.py:175,232` (**HEAD/committed**) | `node_time`, `times[i]` raw | to 16 | **BUG — the one you found**; fixed only in the uncommitted working tree (§4.4) |
| 2 | `scripts/pathwise_precheck.py:70` | `t` raw (accum `+=0.1`) | `horizon=16.0` (:64) | **BUG** |
| 3 | `scripts/pareto_editing_hero.py:99` | `t` raw = `uniform(t_lo,t_hi)` | **default `--t-lo 10.0 --t-hi 16.0`** (:180-181) | **BUG (severe: t∈[10,16])** |
| 4 | `scripts/denovo_learned_vs_uniform.py` (imports `pareto_editing_hero.propose`, :44) | `t` raw (inherited) | 10–16 | **BUG (inherited)** |
| 5 | `scripts/collect_twist_trajectories.py:61` | `time` raw (accum `+=0.1`) | `--horizon 16.0` (:96) | **BUG** |
| 6 | `scripts/griddd_valid_fiber_controller_panel.py:104` | `node_time` raw (frontier, `+0.1`/depth :122) | grows past 1 at depth>10 | **BUG** (uses pancake ckpt) |
| 7 | `scripts/griddd_reward_finetune_train.py:59` | `t` raw (accum) | `--horizon 16.0` (:93) | **BUG-class**; caveat: RTB needs sampler==scorer, so if `forward_mark_batch` also uses raw t it is self-consistent for the gradient though still OOD **[verify the scorer's time]** |

**Correct / not-a-bug call sites** (for completeness): the canonical ancestral path
(`tracelet_conditional.py:1161,1176`); `render_compose_full_trajectories.py:53`; the
`model_time`-from-descriptor replays (`compare_pancake_quotient_repairs.py:92,100,103,107`,
`qualify_analytic_pancake_quotient_backbone.py:105`, replay stored frozen_time);
`run_griddd_analytic_zero_sidecar_smoke.py:147-155` (probes fixed model_time 0.0/0.2/0.4);
`composition_learned_vs_uniform.py:58` (`UniformLegalSampler` **ignores time** — its learned arm calls
`value_guided_smc`, so it inherits #1's fix/bug). Internal wrapper delegations
(`guided_rewrite_sampling.py`, `calibrated_rewrite_sampling.py`, `griddd_conditional.py`,
`property_conditioned_sampling.py`, `canonical_successor_distillation.py:1345`) are pass-through — correct
iff their caller passes frozen_time.

> **Severity:** the pancake event-time audit shows real operational event-times with means ~1.7
> (`atom_delete`), ~3.3 (`atom_restate`), ~5.4 (`ring_system_grow`) — i.e. routinely **>1**
> (`diagnostics/pancake_step6250_eval2000_event_audit.json:27-75`). Passing those raw is deep OOD.

### 4.4 Working-tree state at audit time

`git status` shows `scripts/griddd_value_guided_smc_controller.py` **modified but uncommitted**. The
diff adds `_frozen_time` and converts the two call sites (`:175,:232`) from raw `node_time`/`times[i]`
to `_frozen_time(...)`. So **HEAD still contains the bug; the fix is unstaged.** Several referenced
scripts are **untracked** (`??`): `composition_learned_vs_uniform.py`, `denovo_learned_vs_uniform.py`,
`pareto_editing_hero.py`, `corruption_pairs.py` — meaning bug sites #3, #4, #6 are in files not yet in
git.

---

## 5. Dead ends / confounds / mislabeled artifacts (do not trust)

1. **step-2,500 quotient (`bb53e00…` = `quotient_best.pt` = `step2500_checkpoint.pt`)** — schedule-confounded,
   unblessed; continuation "failed cleanly." §3.1.
2. **The `flexible-graft-prod-2be9258-v1` "production" run** — OOMed with **zero optimizer updates**
   (`HANDOFF.md:73-78`); `CURRENT_MODEL.md:110-112` describes it mid-compile (pre-failure). Do not cite
   it as a trained model.
3. **Canonical-successor / quotient-distilled backbone** — **FAILED qualification**
   (`diagnostics/canonical_successor_backbone_qualification.json`: `qualification_passed:false`,
   `heldout_rate_gate:false`, `rollout_gate:false`). `distilled_head_delta.pt` is the artifact of this
   dead end.
4. **Ring-fix pilots** (`526811c`, `fca5bd5`, P1/P2 pilots) — "improved, not solved"; explicitly
   "Do not promote them … without a matched panel" (`…CORRECTION_V2.md:261`).
5. **Fresh-optimizer continuation (`c82e270`) and legacy-transfer gate** — both "retired"
   (`48_HOUR_RESULTS_TRACKER.md:30-31`; audit `:203-216,245-252`).
6. **Rev-1 "pancake is the retained incumbent" claim** (`HANDOFF_CLAUDE_GENERATORS.md:237-294`) — the
   specific claim the V2 correction exists to reverse; `README.md:132-137` marks
   `HANDOFF*.md` / `48_HOUR_RESULTS_TRACKER.md` "Historical / superseded."
7. **Old step-6,250 metrics as "evidence for the corrected model"** — "diagnostics, not evidence"
   (`HANDOFF.md:65-67`).
8. **Ring-fix definition drift:** `HANDOFF_CLAUDE_GENERATORS.md §7.4` proposes
   `topology_cycle_hierarchical` *alone* as the fix; `GENERATOR_RESULTS_SUMMARY.md §3` later states
   hierarchical *alone is the defect* and `superposed` is required. The docs contradict; the newer one
   governs.
9. **QED-conditioned sidecars** — trained but "no molecule-level efficacy" (commit `a9bcbfb`;
   `…selection.md:33-35`). Real artifacts, weak result; keep as negative evidence.
10. **Four byte-identical pancake copies** and **two byte-identical step-2,500 copies** under different
    names — pure naming debt (see §6).
11. **`compose_current2500/checkpoint.recovery.pt` (`071431d4…`)** — a step-2,500 recovery whose run-ID
    is untracked; **[UNVERIFIED]** whether it is the recovery twin of `bb53e00…` or of the failed
    continuation. Do not reuse until its manifest is located.

**Could not verify (honesty):** Lineage C's checkpoint SHA (remote, none published); the exact run-ID
behind `bb53e00…`/`071431d4…`; the Modal `compose-v4-artifacts` volume contents (no access; Modal launch
prohibited); the `results/*.pt` gate checkpoints (referenced, absent from disk); `forward_mark_batch`'s
time convention in bug site #7. These are stated as open, not guessed.

---

## 6. Cleanup recommendations (do NOT execute — for a human)

Ordered by leverage. Each ties to a concrete failure this audit found.

**P0 — stop the active bug from shipping**
1. Commit the `griddd_value_guided_smc_controller.py` frozen_time fix (currently uncommitted, §4.4),
   then fix raw-`t` sites #2–#7 in §4.3 (route every `sample_rewrite_mark`/`rate_table` call through a
   single shared `_frozen_time`). Add a regression test that asserts the sampler is never called with
   `t>1`.
2. Get sites #3/#4/#6 (`pareto_editing_hero.py`, `denovo_learned_vs_uniform.py`,
   `composition_learned_vs_uniform.py`) **into git** — they are currently untracked, so any paper
   result they produced is unreproducible.

**P1 — end the checkpoint-location chaos (the root cause of "getting confused")**
3. **Move canonical checkpoints off `/private/tmp`.** `/private/tmp` is wiped on reboot; the shipped base
   lives there. Establish `checkpoints/` (git-ignored) + a **git-tracked registry**
   `docs/CHECKPOINT_REGISTRY.md` (or `.json`) mapping: canonical name → SHA256 → bytes → Modal run-ID
   → training config path → producing commit → status(canonical/superseded/dead). This audit's §1 tables
   are a ready seed.
4. **De-duplicate:** collapse the 4 pancake copies to one (keep `compose-pancake6250-checkpoint.pt`,
   delete the 3 `recovery`/`pancake_recovery` twins) and the 2 step-2,500 copies to one (keep
   `step2500_checkpoint.pt`, delete `quotient_best.pt` or make it a symlink). Record the SHA so a future
   reader knows they were identical.
5. **Rename by content, not vibe:** `quotient_best.pt` implies "best/blessed" — it is neither. Rename to
   `lineageB_flexgraft_step2500_UNPROMOTED_bb53e00.pt`. Same for `compose_current2500` once its run-ID is
   found, else quarantine it.

**P2 — reconcile the docs + branches**
6. Merge `b41a8bd` + `docs/HANDOFF_GENERATOR_RUN_LINEAGE_CORRECTION_V2.md` forward into `main` and
   `lipid-corpus-oracle` (which still carries Rev-1). Delete or clearly stamp the superseded
   `HANDOFF_CLAUDE_GENERATORS.md §6` pancake-incumbent claim.
7. Resolve the ring-fix definition drift (§5.8) in one place.
8. Publish Lineage C's checkpoint SHA (or mark it lost).

**P3 — mislabeled diagnostics**
9. `diagnostics/canonical_successor_backbone_qualification.json` points at `…/compose_rgm/artifacts/…`
   (the **sibling** Codex repo path), not this repo — note the cross-repo provenance so it is not read
   as a local result.

### 6.1 Proposed experiment-hygiene RULES (lean set — for promotion into `CLAUDE.md`)

A deliberately **small, low-friction** set — each is mostly "add a SHA field / a status word," not
process overhead, and each maps to a concrete failure this audit found. **Recommendation only — not
applied here** (this audit may write only this file). Adopt the subset that helps; drop any that would
slow a run down.

1. **One registry, SHA-keyed.** Every checkpoint that outlives a session gets one row in a git-tracked
   `docs/CHECKPOINT_REGISTRY.md`: `name → SHA256 → bytes → run-ID → producing-commit → init_from-SHA →
   status`, where `status ∈ {canonical, candidate, superseded, dead}` and **exactly one row per lane is
   `canonical`.** This one rule kills 80% of the confusion here (untracked `bb53e00…`, 4 pancake dups,
   "locate remotely by run-ID"). §1 is a ready seed. *(Cost: one row per real checkpoint.)*
2. **Name by content, not aspiration.** No `best`/`final`/`current`/`quotient_best` in a filename unless
   the registry says `canonical`. Include the step and short-SHA
   (`lineageB_flexgraft_step2500_bb53e00.pt`).
3. **Pin the base by SHA in every experiment that uses one** (train, eval, or conditional). A
   `base_checkpoint_sha256` field in the result/manifest makes "the conditional lane is silently on
   pancake, not B" (§3.4) impossible to miss.
4. **Every result carries an honest status + its budget.** Tag `proven | measured | pending` (the pattern
   already in `GENERATOR_RESULTS_SUMMARY.md`), and record `n_samples` + the panel/budget. A-vs-B claims
   use a matched budget, per-sample (`.claude/context/learnings.md`). *(Cost: three fields.)*
5. **Flag confounds explicitly.** If the LR schedule / init isn't held fixed across a comparison, the
   result is a `candidate`, never a promotion — one line in the manifest. *(Prevents the whole
   step-2,500 episode.)*
6. **Sampler-time CI guard.** One test asserting no `sample_rewrite_mark`/`rate_table` call ever gets
   `t>1` (frozen_time ∈ [0,1]). Catches the raw-`t` bug class (§4.3) automatically, forever.

Everything else (FCD subset labeling, keeping negative results, manifest configs) the repo already does
well; those stay as-is. The six above are the additions worth the keystrokes.

---

## Appendix — exact SHA / size / date table for on-disk generator checkpoints

```
c9d927510360ec6eb84ff8dae1a222b0b693a9bef0ca23bb5d9cca063025876c  38,387,536  Jul20 20:06  lineage_b_checkpoint/checkpoint.best_so_far.pt   [CANONICAL, step-1000]
bb53e00237bd4be60436e149c7f92cd75b30ce555ae96d9264164e303150abb1  38,387,728  Jul19 18:31  quotient_best.pt                                 [step-2500, UNPROMOTED]
bb53e00237bd4be60436e149c7f92cd75b30ce555ae96d9264164e303150abb1  38,387,728  Jul19 19:04  step2500_checkpoint.pt                           [byte-identical to quotient_best.pt]
47716924f7798ed24556c5aa8fb10c533c55dbd1f02f8f53a463cf2ad80ae2bf 100,130,172  Jul20 07:17  compose-pancake6250-checkpoint.pt                [pancake/A, step-6250]
47716924f7798ed24556c5aa8fb10c533c55dbd1f02f8f53a463cf2ad80ae2bf 100,130,172  Jul19/20     {compose_pancake_checkpoint,pancake_checkpoint}/checkpoint.recovery.pt  [dups]
47716924f7798ed24556c5aa8fb10c533c55dbd1f02f8f53a463cf2ad80ae2bf 100,130,172  —            pancake_recovery.pt                              [dup]
071431d4d6d17ec76b8d016133ae5eecfbc97a945f639e1f0162fbf7716527e2 113,567,252  Jul19 17:24  compose_current2500/checkpoint.recovery.pt       [step-2500 recovery, run-ID UNVERIFIED]
e0624f163c53893fcdcbbb565c4aae5aacad7b3b4247f0d1a45741e8f61bb229  38,656,540  Jul20 08:34  compose-v4-qed-conditioned-step500.pt            [conditional, init from pancake]
416266a4e55905a90b500faaa0b7e4dd13b0fb3e4163c1dbcf2bb35a61cef033     146,621  Jul20 07:11  compose-property-conditioned-smoke.pt            [smoke]
8cf0ecb952e09a55685869dd909377b465455886834dae6087b8d8e71257a0c8     146,149  Jul20 07:11  compose-property-conditioned-smoke.best_so_far.pt [smoke]
d1ae1ba791d18ba6df0623b49ba6bff89691da989bd64b6eab22e5afe573dc22     541,571  —            canonical_successor_qualification_probe/distilled_head_delta.pt [distillation dead end]
```

*End of lineage map. Every claim above is sourced to a commit SHA and/or `file:line`; unverifiable items
are marked `[UNVERIFIED]` and enumerated in §5.*
