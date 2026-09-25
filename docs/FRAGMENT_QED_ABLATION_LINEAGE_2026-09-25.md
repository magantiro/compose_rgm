# Fragment and QED ablation lineage, 2026-09-25

This report binds the current manuscript rows to named run contracts and
records which causal intervention is supported by each implemented sampler.
It does not promote a development or incomplete result to an official run.

## Manuscript fragment rows

| Manuscript row | Current COMPOSE values (quality, uniqueness, diversity, validity) | Bound source | Evidence role |
| --- | --- | --- | --- |
| Motif extension | 43.0, 97.5, 0.669, 100.0 | `diagnostics/fragment_motif_focused_dev_v1/summary.json`, SHA-256 `71c962306249da7c5a040aba1b505086eb1c7bbe5732645643bf843e5747a88a`; qualified development precursor of `configs/fragment_motif_official_v1.json` | 200 attempts, one seed; the official 3,000-attempt run is separate and was still in progress at inspection |
| Superstructure | 39.03, 97.33, 0.725, 100.0 | `configs/fragment_superstructure_official_v2.json`, payload SHA-256 `77c994c8364d088c6e5bf1fe688794cac5869c7e4abf98c92a87d08e1b1c751e`; `diagnostics/fragment_superstructure_official_v2/result.json` | complete 10 prompts x 100 attempts x 3 seeds; 3,000 committed and prompt-compliant outputs |
| Linker design | 35.0, 91.5, 0.548, 100.0 | `diagnostics/fragment_training_linker_metric_pilot_v2/summary.json`, recorded in `docs/FRAGMENT_LINKER_METRIC_PILOT_V2_RESULT_2026-09-24.md`; development precursor of `configs/fragment_linker_official_v1.json` | 200 attempts, one seed. The completed 3,000-attempt official run instead reports 28.6, 72.4, 0.562, 100.0 |
| Scaffold morphing | same as linker | linker output alias by identical prompt inputs in `configs/fragment_linker_official_v1.json` | not independent sampling |
| Scaffold decoration | 22.5, 99.0, 0.628, 100.0 | `diagnostics/fragment_pendant_decoration_dev_v1/summary.json`, SHA-256 `e4d064a9786854ae9e5187654d3be7c1c65efd139aea3bacc458825ab897cf88` | 200 attempts, one development seed; no later pilot is substituted |

The motif contract's serialized status is
`prepared_pending_payload_specific_scored_evaluation_authorization`, whereas
its matching manifest is present and a separate user authorization launched
the official run. The stale preparation status does not establish completion;
the final `summary.json` and all 30 sealed rows do. The linker and
superstructure contracts record authorized status, but likewise require their
actual completed artifacts for result claims.

## Operative samplers and checkpoint equivalence

Motif and linker construct eight complete, structurally valid programs per
attempt from training-derived construction information. The frozen RingCore
model scores each admitted program by the mean native log-mark probability
along its primitive path. A softmax selects one unique endpoint. Structural
offer construction, exact compilation, endpoint deduplication, and native
finite-fiber membership depend on the current graph and frozen operator
configuration, not the learned tensor values. The numerical-finiteness guard
must remain visible; a nonfinite score is an abstention, never a new support
element. Uniform selection over the *same recorded model-supported panel* is
therefore a conditional panel-ranking ablation. It does not remove the
training-derived construction catalogue and does not replace a primitive
transition reference.

The completed linker panels support an inexpensive paired replay on all 3,000
attempts at `diagnostics/fragment_common_panel_linker_v1/result.json`.
The learned arm exactly reproduces the official summary. Learned versus
uniform common-panel selection gives quality 28.6% versus 23.33%, uniqueness
72.4% versus 73.37%, and diversity 0.56215 versus 0.57390. Both arms retain
the original no-output slots and all eight recorded offers. These are
prompt-level descriptive means; no statistical claim is inferred here.

Superstructure uses `sample_completion` in
`src/compose_v4/benchmark/fragment_conditioned_sampler.py`. It draws
primitive marks from the model, executes them under the attachment controller
and effective-chemistry lock, and preserves no-output attempts. It is the
primary learned-transition-reference ablation. The exact nonlearned law and
replaced probability-dependent stages were frozen before outcome inspection
in `docs/FRAGMENT_SUPERSTRUCTURE_REFERENCE_ABLATION_LAW_2026-09-25.md`.

All three named fragment contracts pin the same RingCore file SHA-256
`24117dfeaee91729bb4ebccb5eb5b993b4a6218605045e51917823d086b4c1e4`.
Loading the checkpoint confirms 118 parameter tensors, width 256, six message
passing steps, hierarchical rate factorization, primitive cycle operations,
and disabled ring-grow macro. The fragment task constructors and conditioning
settings nevertheless differ. The RingCore file comes from the 16,000-step
scientific diagnostic lineage; its filename alone is not a checkpoint-selection
gate. The deployed QED Editing-V2 checkpoint is different: it has 120 selected
parameter tensors, and none of the 117 shared, shape-compatible tensors equals
the RingCore tensor. The deterministic sorted-tensor SHA-256 digests are
`c5923a3948199cbd9666004a34438e4161eb4517cb0141971e21098fb88ca61e`
for RingCore and
`078e6a5678ca0a3ea09fc0dac1fc3447bf2937dfc0a922bdeaa9e3639d9e0eb1`
for Editing-V2. Thus checkpoint equivalence holds across these fragment
contracts, not across the fragment and QED experiments.

## Actual 800-source QED lineage and fixed-size intervention

`scripts/hphi_official800_launch.py` obtains the exact 800 Jin sources through
`official_tasks()` in `modal_apps/hphi_h40head_ab_app.py` and deploys the
`run_source` restart arm. The paper-era run uses horizon 40, 32 particles,
eight independent terminal returns per source, head directory `hphi_v2` with
budget input clamped to 24, and `seed_for("restart", source, k)`. The model is
the Editing-V2 `R_THETA_CHECKPOINT.pt`, SHA-256
`c979cdb3d7b0b403bfbf7bfb0aa5098b2588c6d4217770c2c58292b7c4e53de8`.
The head `head.pt` and `norm.json`, `RUN_PATHS.json`, materialized scorer and
source-level shards are volume-bound inputs; their exact local bytes are not
present in this worktree. The generic option/macro controller is not the
source of the 800-source row.

The latest local per-source rederivation,
`diagnostics/qed_griddd_arm_a_rederivation_v1.json`, has 798 scored sources,
446 solved and sources 135 and 408 unscored. It verifies the available source
and returned-molecule metrics, but it is not a complete 800-source paired
baseline. Counting 446 over the full 800 gives 55.75%, while the conditional
798-source rate is 55.89%; the 55.8% manuscript value uses the former rounded
number. The two missing source records must be completed or explicitly
accounted for before a fixed-size comparison is evaluated over all 800.

The fixed-size switch already prepared in the *same H40 runner* removes every
primitive family that changes heavy-atom count before its family draw,
renormalizes remaining probabilities, and asserts unchanged count after
execution. An empty restricted fiber kills the particle without replacement.
The frozen head, runtime inputs, seeds, horizon, particles, resampling, and
eight-terminal-return rule are unchanged. This is a capability-disabling test
of the deployed sampler, not a new option/macro sampler. No scored fixed-size
job has been launched. Exact volume inputs and the two missing baseline
source records are required before the full paired 800-source analysis can
run. Successful-path atom-count histories are also absent from existing
returned-endpoint archives and require authentic ancestry logging on a
parity-checked full-arm rerun.
