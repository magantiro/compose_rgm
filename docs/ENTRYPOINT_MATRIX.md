# Entry-Point Matrix — Program Coherence Audit (HEAD 4204c9d)

106 scripts + 6 diagnostics + 5 modal apps + 1 src module-CLI + 90 tests + 1 Makefile. **Nothing UNKNOWN.**
No `console_scripts` in pyproject.

## Production RingCore-V1 call graph (proven from code)
`modal_apps/train_tracelet_gm.py` (WRAPPER) → subprocess → `scripts/train_tracelet_cnof_gate.py`
(PRODUCTION_CANONICAL) `--corrupted-prior-mix --cycle-op-mix --disable-ring-grow-macro --organic-vocabulary
--scaled-manifest ... --train-only`:
- **Model builder:** `FactorizedTraceletRateModel` (gate:3005), guarded by `model==from_scratch &&
  training_backend==factorized_marks`. enable_ring_restates/cyclic_graft/heteroatom_scan/ring_opening=True,
  enable_cycle_ops=True, enable_ring_grow_macro=**False**, atom_vocabulary=ORGANIC (15 classes).
- **Data builder:** `load_organic_corpus_split(BROAD_ORGANIC_V1)` (asserts max_atoms==40) +
  `build_corrupted_prior_records(ORGANIC)` + `build_cycle_op_records`; zero-mixture (denovo_keep=0, no de-novo
  path cache); validation rebuilt from split.validation.
- **Candidate enumerator (shared training+sampling):** `prepare_factorized_mark_batch` via
  `train_factorized_mark_model → factorized_mark_loader → FactorizedMarkCollator`, reading `model.enable_*`.
  Cycle ops scored via `_CYCLE_OP_EXECUTOR_TO_FAMILY` when `self.enable_cycle_ops`.
- **Executor:** one `RewriteSystem.apply` (`kernel.py:51`) via `de_novo_rewrite_system()` — SAME fiber+executor
  produce the teacher marks AND the dense mask they're scored against (DRY = the correctness guarantee).
- **Sampler (eval, not under --train-only):** `evaluate_tracelet_rollouts.load_factorized_rollout_checkpoint`
  reconstructs the identical model from checkpoint metadata → `sample_tracelet_ancestral` → same
  `prepare_factorized_mark_batch` → same executor. RingCore eval = `ring_core_rollout_panel.py` (canonical loader).

## Classification (summary)
- **PRODUCTION_CANONICAL:** train_tracelet_cnof_gate.py, evaluate_tracelet_rollouts.py,
  tracelet_sampling_worker.py, mine_edit_traces.py (data), build_scaled_edit_manifest.py,
  build_ring_core_v1_manifest.py, build_edit_data_manifest.py, freeze_ring_core_v1.py,
  regenerate_ring_core_manifest.py, ring_core_identity.py (lib).
- **PRODUCTION_WRAPPER:** train_tracelet_gm.py, run_tracelet_recipe.py, evaluate_rollout_shards.py,
  mine_edit_traces_app.py.
- **DIAGNOSTIC_ONLY:** the RingCore preflight gates (ring_core_rollout_panel, broad_preflight_gate,
  prelaunch_gate, warmstart_dry_run, subtype_supervision_preflight, ring_topology_capability_gate,
  ring_core_calibration, ring_path_cost_study, characterize_teacher_filter, cold_vocab_audit,
  stage6a_coverage_smoke) + all Paper-1 result/figure drivers + the audit/benchmark/render scripts + paper build.
- **EXPERIMENTAL:** the conditional-control subsystem (griddd_* SMC controller, reward-FT, twist), the older
  CNOF path (train_cnof_conditional_gate), the analogue-layer data (build_analogue_trace_pool), and the whole
  COMPOSE-Lipid program (separate paper lane).
- **LEGACY_QUARANTINED:** the "pancake quotient" backbone line (7 scripts — superseded by Lineage B, NOT
  reachable from the production recipe).
- **TEST_FIXTURE:** 90 tests. **DEAD_CODE:** none among entry points.

**No CRITICAL/HIGH deviation in the RingCore-V1 training path itself.**
