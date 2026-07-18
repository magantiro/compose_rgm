# COMPOSE trajectory diagnostics archive

This folder is the canonical saved bundle for the trajectory diagnostics discussed on 2026-07-18. Keep the full-resolution PNGs, paginated PNGs, and machine-readable JSON together.

## Step-6250 full ancestral trajectories

- `full_trajectory_step6250_index32.png` — gauge-churn failure case: 118 recorded events, 103 Grafts (`bond_reroute`), 30 unique molecular states, 89 canonical molecular self-transitions, and zero strict nontrivial two-cycles.
- `full_trajectory_step6250_index75.png` — efficient comparison: 37 events, 22 Grafts, 8 inserts, 38 unique molecular states, and no self-transitions or strict two-cycles.
- `full_trajectory_step6250_index32_page01.png` through `page04.png` — readable pages for trajectory 32.
- `full_trajectory_step6250_index75_page01.png` through `page02.png` — readable pages for trajectory 75.
- `full_trajectory_step6250_index32.json` and `full_trajectory_step6250_index75.json` — complete per-state and per-event records.
- `full_trajectory_summary.json` — compact cross-trajectory statistics and seeds.

## Context figures retained in the same bundle

- `trajectory_event_audit_step4750.png` — earlier event-family/atom-count trajectory audit used to diagnose Graft-heavy behavior. This is checkpoint step 4,750, not step 6,250.
- `eval2000_summary_step6250.png` — exact 2,000-sample step-6,250 summary, including rewrite activity, ring distributions, ring phenotypes, and FCD decomposition.

The checkpoint labels are intentionally preserved so the figures are never mistaken for measurements from the same model state.

The first rendering called every equality at lag two an “immediate reversal.” That conflated repeated molecular self-transitions with genuine `A → B → A` cycles. The preserved PNGs and JSON now use the corrected definitions above.
