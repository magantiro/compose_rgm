# Fragment checkpoint-selection provenance

The named fragment contracts pin
`/Users/rmaganti/compose_fragment_ckpt/ringcore_a7546e2_best.pt`, SHA-256
`24117dfeaee91729bb4ebccb5eb5b993b4a6218605045e51917823d086b4c1e4`.
The file is a RingCore-V1 checkpoint from the completed 16,000-step
`a7546e2` training run. Its payload records
`checkpoint_kind=interim_best_evaluation_model`,
`completed_steps=16000`, and `selected_validation.selected_step=8500`.
Thus the weights used by these fragment evaluations are the trainer's
interim-best step-8,500 weights, not automatically the final step-16,000
snapshot.

The separate frozen editing-selection policy in
`configs/ringcore_v1_checkpoint_selection.json` specifies productive
canonical-successor NLL and hard executor/family gates, not the trainer's
hazard-inclusive GM validation loss. The available readiness record,
`diagnostics/coherence/ringcore_v1_successor_leaderboard_readiness_2026-07-30.json`,
states `ready_to_execute=false` and lists unfinished panel and scoring
prerequisites. No completed all-snapshot canonical-successor selection
decision was identified in the current repository. The fragment benchmark
run completion and its frozen checkpoint identity are established; a claim
that this checkpoint won the separate canonical-successor selection protocol
is not established by those run records.

The deployed QED Editing-V2 reference is numerically distinct from the
RingCore fragment checkpoint, as recorded in
`docs/FRAGMENT_QED_LATEST_ABLATION_REPORT_2026-09-25.md`. This provenance
point concerns the fragment checkpoint label and selection rule, not whether
the completed fragment evaluations ran.
