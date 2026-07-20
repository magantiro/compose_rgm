# Conditional-backbone evidence table

**Date:** 2026-07-20  
**Scope:** frozen existing artifacts, including the completed events-13--24
controller result.

## Provenance boundary

The immutable Modal manifest for
`compose-v4-qed-conditioned-pilot-20260720-v4` records compatible
initialization from
`compose-v4-stage3-full-ring-hierarchical-v1/checkpoint.recovery.pt`.
It does **not** record initialization from the retained step-6,250 pancake
checkpoint, `empirical_mark_prior_mode`, or `ring_family_mass_mode`.
Step 1,000 exactly resumes the step-500 recovery checkpoint.  Consequently,
neither conditioned checkpoint can be credited with the calibrated-pancake
inference wrapper or the P1/P2 empirical-prior and topology-mass corrections.
The current local recipe text is not a substitute for the immutable run
manifest when describing the completed artifact.

## Evidence comparison

| Candidate | Proposal support | Self-transition / thrashing risk | Ring chemistry | Family / mark learning | Measured controllability | Repair boundary |
|---|---|---|---|---|---|---|
| Raw step-6,250 pancake | Broad typed executable catalog and 110 reusable learned tensors; four later ring-role tensors are absent | Severe presentation churn: one preserved 118-event trace has 89 canonical molecular self-events; the 2,000-sample audit assigns 76.07% of events to Graft/bond-reroute and 93.25% of trajectories are Graft-majority | Ring commitment is extremely late: mean normalized first-ring position 0.9784 and 99.31% of ring events are in the final event decile; total cycle rank is underproduced | Pre-quotient marked-rate law; no QED input and no current conditioned teacher metrics | Not a suitable raw oracle controller: self successors consume proposal/oracle budget and its condition is absent | Canonical no-op handling, CTMC thinning, and endpoint-preserving commuting are executor/control fixes. Chemistry rate, ring-role, and topology learning require training |
| Calibrated step-6,250 pancake | Same positive support; calibration only thins existing actions and never enables an action | Operational failure is removed: 600 endpoints and 18,646 states are valid and connected, with zero molecular self-events, immediate backtracks, collapse, or event-budget failure | Improved but not faithful: cycle rank 2.46 versus 3.35; fused 28.5% versus 58.0%; small-ring prevalence 11.3% versus 6.0%; mean QED 0.439 versus 0.602 | The checkpoint weights are unchanged by the inference-only delete (-0.5 log-rate) and small-ring (-1.5) calibration; the legacy ring-role gap remains | Best measured executable substrate, but it has no learned target input and its proposals retain large composition/rate errors | Existing no-op and bounded thinning are executor/control fixes. Triple-bond, heteroatom, ring-electronic, and topology residuals require P1/P2 training |
| QED-conditioned step 500 | Validity-closed, de-novo carbon-tree support; actual initialization is the full-ring hierarchical checkpoint, not pancake/P1/P2 | All 50 target/control endpoints are valid and unique within condition. Saved high-target trajectories have all valid, connected states and no exhaustion or virtual events | No matched unconditional chemistry audit exists, so calibrated-pancake chemistry claims cannot be transferred to this checkpoint | Immutable BF16 test top-1 family accuracy 60.50%. CPU replay: top-3 86.74%, balanced top-1 55.25%, teacher-family probability 0.5351, exact-teacher-mark probability 0.0738. Insert and reorder top-1 are only 35.7% and 33.3% | Real but weak direct response: target 0.9 mean 0.6418 versus 0.5745 control (+0.0673), with 0/10 at QED >=0.9. On fixed states, the target-0.9 family law moves TV 0.077 from classifier-free | Valid-successor QED tilting is a control-layer option. Weak exact-mark/family learning and endpoint-conditioning strength require training/objective changes |
| QED-conditioned step 1,000 | Strongest measured controller support: the frozen gate produces 400/400 valid, connected proposal successors; soft expected QED improves 0.00956 and offline best-of-four improves 0.02906 | Direct and both controlled windows are 10/10 valid and unique; every audited state and proposal is valid and connected | Still lacks a matched unconditional chemistry audit. Moving control from events 1--12 to 13--24 exposes restate/ring-grow proposals but has a severe support-search tail | Validation loss improves to 11.0227, but frozen-test top-1 is 58.56%, top-3 86.19%, balanced top-1 53.32%, and exact-teacher-mark probability 0.0767 | Direct target-0.9 shift is +0.0196 with 0/10 at QED >=0.9. The first-window controller gives paired +0.0237 and 0/10 at QED >=0.9. The late window gives +0.0636 and 2/10 at QED >=0.9, but its interval [-0.0703, 0.1716] includes zero and realized marks are unmatched (384 versus 392) | Window timing, K, and beta are controller-level. The late result supports the mechanism but does not select a controller or repair the direct steering/family/mark plateau; those require training/objective changes |
| P1/P2-compatible corrected backbone | Designed to preserve positive rare-chemistry support through smoothed corpus bases plus learned residuals and topology-group absolute rate mass | Compatible with the successor-correct executor and exact CTMC thinning | P1 targets triple/heteroatom/ring-electronic errors; P2 prevents a tiny all-small residual support set from inheriting the whole ring-family probability | Implementation and focused tests exist, but no trained checkpoint or rollout result exists | Conceptually the best future conditional basis, but currently unmeasured and therefore not selectable | P1/P2 inherently require the authorized 250--500 cached training pilot and fixed 200-rollout gate; executor/control changes cannot substitute for this evidence |

## Selection implication

Use step 1,000 only for the currently bounded controller-mechanism experiment,
because it has the strongest measured valid-successor support.  Do not select
it as the final conditional backbone: additional direct training lowered
Generator-Matching loss without improving average high-QED steering, and it
lacks the P1/P2 corrections.  The leading **candidate lineage**, not yet a
qualified checkpoint, is a P1/P2-corrected QED-conditioned model after its
unconditional 200-rollout chemistry gate and matched conditional evaluation.

## Evidence sources

- `/artifacts/compose-v4-qed-conditioned-pilot-20260720-v4/manifest.json`
- `/artifacts/compose-v4-qed-conditioned-pilot-20260720-v4/manifest.training.json`
- `/artifacts/compose-v4-qed-conditioned-step1000-continuation-20260720-v1/manifest.training.json`
- `diagnostics/pancake_step6250_eval2000_event_audit.json`
- `diagnostics/pancake6250_calibration_eval600_metrics.json`
- `diagnostics/qed_step500_family_diagnostics.json`
- `diagnostics/qed_step500_condition_sensitivity.json`
- `diagnostics/qed_step500_target_response.json`
- `diagnostics/qed_step1000_target_response.json`
- `diagnostics/qed_step1000_successor_guidance_gate.json`
- `diagnostics/qed_step1000_controlled_rollout_analysis.json`
- `diagnostics/qed_step1000_controlled_rollout_late_window.json`
- `docs/audits/2026-07-20_unconditional_chemistry_failure_audit.md`
