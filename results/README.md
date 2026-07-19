# Result file ledger

Files described below as “current” are current only within their historical
development gate. None is the quotient-correct Stage-3 production checkpoint;
that run has not yet launched. See `docs/PROJECT_STATUS.md` for live status.

- `tree_source_reachability_primitive_tracelet_uniform_size_seed20260717.json`
  is the current structured-source reachability gate: 512 independently
  size-sampled tree/target attempts over 128 GuacaMol molecules. All 496
  attempts for the 124 chemistry-supported targets reached the exact endpoint;
  the remaining 16 are four known hypervalent-sulfur compiler failures. Mean
  absolute source/target size difference is 5.86 atoms, mean path length is
  20.40 events, and scalar ring closures are zero.
- `tree_source_reachability_seed20260717.json`,
  `tree_source_reachability_primitive_seed20260717.json`, and
  `tree_source_reachability_primitive_uniform_size_seed20260717.json` are
  superseded diagnostics from before root tracelet conversion. They are kept
  only to measure the optional reroute/path-length and scalar-closure boundary.

- `tiny_rate_quotient_pilot.pt` is the current quotient-corrected learned
  checkpoint used for the final tiny-gate evaluations.
- `tiny_rate_quotient_pilot.json` is its 100-rollout training report.
- `tiny_rate_gate_eval_1000.json` is the primary frozen-checkpoint evaluation.
- `tiny_rate_gate_eval_dt0p2.json` and `tiny_rate_gate_eval_dt0p05.json` are the
  time-step checks around the primary `dt=0.1` evaluation.
- `tiny_rate_gate_pilot.*`, `tiny_rate_marginal_pilot.*`, and
  `tiny_rate_gate.*` are retained diagnostic runs from before canonical
  chemical-state marginalization. They must not be reported as the final gate.
- `cnof_gate_late.pt` and `cnof_gate_late.json` are the current held-out
  C/N/O/F corpus checkpoint and primary 100-rollout report. They use the 50/50
  physical-time/late-operational-time training mixture.
- `cnof_gate_late_sampler_diagnostics.json` contains the frozen-checkpoint
  `dt=0.20/0.10/0.05` checks and the untrained baseline, each at 100 rollouts.
- `cnof_gate.pt` and `cnof_gate_sampler_diagnostics.json` are the earlier
  uniform-physical-time control. Its late-time hazard extrapolation produces
  excess cycles and it is not the current checkpoint.
- `cnof_gate_context.*` is the negative rewrite-context ablation. Explicit path
  distance/global cycle features did not repair the endpoint calibration error.
- `cnof_gate_pilot.*` and `cnof_gate_sampler_smoke.json` are development pilots,
  not reportable final evaluations.
- `cnof_scale16_random.pt` and `cnof_scale16_random.json` are the full-scan
  1,333-molecule random-split gate for the uniform-base neural generator. This
  run has 1,200 updates and 200 learned plus 200 matched-baseline rollouts.
- `prior_tilt_smoke.*` verifies that the neural residual generator exactly
  inherits the fitted corpus-marginal rewrite prior at initialization.
- `cnof_scale16_prior_tilt.pt` and `cnof_scale16_prior_tilt.json` are the first
  600-update prior-tilted comparison on the identical split. The neural
  tilt improves several structural distribution metrics over the matched prior
  but not FCD or QED; see `docs/SCALE16_GATE.md`.
- `topology_deferred_smoke.*` is the 96/16/16 integration pilot for topology-
  aware action features, trust-region regularization, deferred bond-order
  teachers, and aromaticity diagnostics.
- `cnof_scale16_topology_deferred.pt` and
  `cnof_scale16_topology_deferred.json` are the current primary 1,333-molecule
  checkpoint and report. The learned model beats its matched prior on FCD,
  QED, SA, ring-size TV, cycle-rank TV, and fused/bridged/spiro calibration,
  while retaining 100% valid/connected/non-null/unique sampling.
