# Artifact index

This index separates current model artifacts from historical diagnostics.

## Canonical research plans

| Artifact | Purpose |
|---|---|
| [`research_plans/paper1_compose_methods.html`](research_plans/paper1_compose_methods.html) | COMPOSE RGM methods-paper plan |
| [`research_plans/paper2_compose_lipid.html`](research_plans/paper2_compose_lipid.html) | COMPOSE-Lipid translational plan |
| [`research_plans/compose_two_paper_execution_plan.html`](research_plans/compose_two_paper_execution_plan.html) | Integrated execution and dependency plan |

## Current audits

| Artifact | Status |
|---|---|
| [`audits/semantic_ring_teacher_audit_v4.json`](audits/semantic_ring_teacher_audit_v4.json) | 128 examples; zero teacher-support failures |
| [`PROJECT_STATUS.md`](PROJECT_STATUS.md) | Dated completed/in-flight/queued workstream and claim boundary |
| [`CURRENT_MODEL.md`](CURRENT_MODEL.md) | Source, teacher, operator, generator, sampler, and evidence contract |
| [`NOVELTY_POSITIONING.md`](NOVELTY_POSITIONING.md) | Scoped distinctions from Morph, Edit Flows, DDSBM, grammars, and fragment methods |
| [`TREE_SOURCE_TRANSPORT.md`](TREE_SOURCE_TRANSPORT.md) | Carbon-tree source, flexible-size transport, and Graft semantics |
| [`RING_REWRITE_ARCHITECTURE.md`](RING_REWRITE_ARCHITECTURE.md) | Coordinated ring-system operator design |

## Legacy pre-quotient trajectory bundle

All files below are under
[`trajectory_diagnostics/legacy_prequotient/full_trajectories_step6250/`](trajectory_diagnostics/legacy_prequotient/full_trajectories_step6250/).
They document the checkpoint that motivated the successor quotient; they must
not be presented as results of the corrected model.

| Artifact | Interpretation |
|---|---|
| [`full_trajectory_step6250_index32.png`](trajectory_diagnostics/legacy_prequotient/full_trajectories_step6250/full_trajectory_step6250_index32.png) | Full 118-event “pancaking” trajectory: 103 Grafts, 89 molecular self-transitions |
| [`full_trajectory_step6250_index32_page01.png`](trajectory_diagnostics/legacy_prequotient/full_trajectories_step6250/full_trajectory_step6250_index32_page01.png) | First of four readable pages for index 32 |
| [`full_trajectory_step6250_index32.json`](trajectory_diagnostics/legacy_prequotient/full_trajectories_step6250/full_trajectory_step6250_index32.json) | Exact per-state/action/SMILES replay for index 32 |
| [`full_trajectory_step6250_index75.png`](trajectory_diagnostics/legacy_prequotient/full_trajectories_step6250/full_trajectory_step6250_index75.png) | Efficient 37-event comparator with no molecular self-transitions |
| [`trajectory_event_audit_step4750.png`](trajectory_diagnostics/legacy_prequotient/full_trajectories_step6250/trajectory_event_audit_step4750.png) | Earlier event-family raster and atom-count curves; checkpoint step 4,750 |
| [`eval2000_summary_step6250.png`](trajectory_diagnostics/legacy_prequotient/full_trajectories_step6250/eval2000_summary_step6250.png) | Exact 2,000-sample legacy dashboard: rewrites, rings, and FCD decomposition |
| [`full_trajectory_summary.json`](trajectory_diagnostics/legacy_prequotient/full_trajectories_step6250/full_trajectory_summary.json) | Compact cross-trajectory statistics and deterministic seeds |

## Diagnostic scripts

- [`../scripts/diagnostics/render_compose_full_trajectories.py`](../scripts/diagnostics/render_compose_full_trajectories.py)
  replays a preserved checkpoint and rollout cache into full trajectory records.
- [`../scripts/diagnostics/rerender_saved_compose_trajectories.py`](../scripts/diagnostics/rerender_saved_compose_trajectories.py)
  recreates PNGs from the portable JSON replays.
- [`../scripts/diagnostics/analyze_compose_preview_trajectories.py`](../scripts/diagnostics/analyze_compose_preview_trajectories.py)
  renders the earlier event-family/atom-count audit from an explicit rollout-cache path.

The trajectory JSONs use relative asset names and contain no machine-specific
paths. The PNG bundle is small enough to remain versioned with the code.
