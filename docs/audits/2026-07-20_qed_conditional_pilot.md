# Direct-QED conditional pilot audit

**Date:** 2026-07-20  
**Checkpoint:** `compose-v4-qed-conditioned-pilot-20260720-v4/checkpoint.pt`  
**Checkpoint SHA-256:** `e0624f163c53893fcdcbbb565c4aae5aacad7b3b4247f0d1a45741e8f61bb229`

## Decision

The 500-update model has learned a real, bidirectional QED response while
retaining perfect final validity and within-condition uniqueness in this
50-attempt pilot.  It is not yet a high-QED optimizer.  Target means are
strictly monotonic, but the high-target paired effects are small and noisy,
and none of the 20 rollouts requested at QED 0.7 or 0.9 reached QED 0.9.

This authorizes bounded conditional development.  It does **not** authorize
the 800-start, 20-candidate GrIDDD benchmark or a claim of constrained
molecular optimization.

## Frozen training result

- The selected checkpoint is the final step 500.
- Validation Generator-Matching loss improved from 38.7554 to 11.6031.
- The immutable CUDA-BF16 artifact reports 60.4972% test top-1 family
  accuracy on 362 nonterminal examples.
- A CPU-float32 diagnostic replay reports 60.2210% top-1, 86.7403% top-3,
  55.2549% macro top-1, and 83.3047% macro top-3 across the six represented
  families.  One near-tie accounts for the top-1 precision difference.
- Mean test probability assigned to the teacher family is 0.5351; mean
  probability assigned to the exact teacher mark is 0.0738.  Exact mark
  probability includes the family and the within-family site/template choice.

The complete family diagnostic, including counts, is
`diagnostics/qed_step500_family_diagnostics.json`.

## Matched target-response table

Every arm uses the same checkpoint source prior, trajectory seed 20260721,
ten attempts, 40 atom slots, horizon 16.0, time step 0.1, and 128-event cap.
Validity and uniqueness use all attempts as the denominator.

| Condition | Valid | Unique | Mean QED | SD | Range | MAE | Within ±0.10 | Mean events | Shift vs control |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Classifier-free | 10/10 | 10/10 | 0.5745 | 0.1365 | 0.3122–0.7338 | — | — | 30.7 | — |
| QED 0.30 | 10/10 | 10/10 | 0.4611 | 0.2050 | 0.2266–0.7981 | 0.1970 | 5/10 | 34.2 | −0.1134 |
| QED 0.50 | 10/10 | 10/10 | 0.4735 | 0.1177 | 0.2546–0.6705 | 0.0977 | 5/10 | 31.4 | −0.1010 |
| QED 0.70 | 10/10 | 10/10 | 0.6124 | 0.1415 | 0.3902–0.8478 | 0.1383 | 4/10 | 26.7 | +0.0379 |
| QED 0.90 | 10/10 | 10/10 | 0.6418 | 0.1669 | 0.2061–0.8158 | 0.2582 | 1/10 | 24.0 | +0.0673 |

The four requested-target means have Spearman correlation 1.0 and span
0.1807 from target 0.3 to 0.9.  This is an aggregate monotonicity diagnostic,
not an uncertainty-adjusted effect estimate.  In paired per-seed analysis,
the QED-0.7 mean shift has a bootstrap 95% interval of [−0.0767, 0.1549], and
the QED-0.9 shift has interval [−0.0118, 0.1460].

The `within ±0.10` metric at target 0.9 includes outputs in [0.8, 1.0].  It
must not be described as QED-0.9 success: the QED-0.9 arm produced **0/10**
outputs at or above 0.9.

The machine-readable aggregate is
`diagnostics/qed_step500_target_response.json`.  Persistent high-target
artifacts are:

- `/artifacts/compose-v4-qed-conditioned-step500-target070-eval10-20260720-v1`
- `/artifacts/compose-v4-qed-conditioned-step500-target090-eval10-20260720-v1`

Their saved trajectory diagnostics cover 277 QED-0.7 states and 250 QED-0.9
states with zero invalid states, disconnected states, event-budget
exhaustions, or virtual events.

## What the condition changes

On the same 512 test states, increasing requested QED from 0.3 to 0.9 lowers
mean hazard from 8.2448 to 5.7746, lowers mean atom-insert family mass from
0.0904 to 0.0419, raises atom-delete mass from 0.0649 to 0.0923, and raises
ring-system-grow mass from 0.2019 to 0.2386.  The QED-0.9 family distribution
has mean total variation 0.0771 from the classifier-free distribution.

The model therefore does not ignore its target.  The rate response is modest,
which agrees with the weak rollout shift.  Exact values are in
`diagnostics/qed_step500_condition_sensitivity.json`.

## Straggler diagnosis and evaluator hardening

With 10 attempts and 12 requested workers, each attempt occupies a separate
worker.  Most QED-0.5 attempts completed in about one minute, while the full
arm took 1,021.6 seconds.  The old evaluator waited for every worker and
persisted nothing until the complete multi-condition run ended.  A single
support-search long tail could therefore hide every completed condition.

The evaluator now:

- prints per-attempt completion progress;
- atomically saves JSON and rollout-cache partials after each condition; and
- permits target-isolated jobs without recomputing the classifier-free arm.

This does not change the checkpoint, CTMC, rewrite support, seed derivation,
or all-attempt accounting.

## Condition-dose diagnostic and next bounded pilot

Two isolated, ten-attempt off-range condition doses tested whether simple
condition extrapolation could strengthen the result.  These are diagnostic
inputs, not attainable QED targets or benchmark arms.

- Dose 1.1 produced 10/10 valid and unique outputs with mean QED 0.7003,
  maximum 0.8794, and a paired mean shift of +0.1258 from control.  Its paired
  bootstrap interval was [0.0031, 0.2489].
- Dose 1.3 produced 10/10 valid and unique outputs with mean QED 0.5310,
  maximum 0.8084, and a paired mean shift of −0.0435.
- Neither arm produced QED≥0.9.  The response therefore improves at dose 1.1
  and then reverses; naive extrapolation is saturated.

The next bounded pilot is now an exact-optimizer continuation from step 500 to
a hard total cap of 1,000 updates.  It reuses the same paths, support cache,
evaluation tensors, optimizer state, RNG/data-stream position, and 15%
condition dropout.  The original 500-step cosine schedule remains at its
2e-5 floor after step 500.  Validation and recovery occur every 100 steps, and
two non-improving evaluations stop the run.  The live immutable run is:

- `/artifacts/compose-v4-qed-conditioned-step1000-continuation-20260720-v1`
- `recipes/tree_fcd_transfer_qed_conditioned_step1000_continuation.json`

The resume validation reproduced step 500 exactly (loss 11.6031, top-1
59.1623%, top-3 85.8639%).  Step 600 did not improve model selection.  Step
700 was promoted at loss 11.3182, step 800 did not improve on it, step 900 was
promoted at loss 11.1174, and the hard-cap step 1000 was selected at loss
11.0227.  Relative to step 500, the selected loss is 5.00% lower.  The final
validation top-1 and top-3 family accuracies are 60.2094% and 84.8168%; the
balanced values are 53.9864% and 80.1836%.  On the frozen test split, top-1
is 58.5635%, top-3 is 86.1878%, balanced top-1 is 53.3240%, and balanced
top-3 is 83.4973%.  In independent matched eval10 arms, step 1000 produced
mean QED 0.6037 at target 0.7 and 0.5980 at target 0.9, versus 0.6124 and
0.6418 at step 500.  Both arms remained 10/10 valid and unique, but neither
produced QED≥0.9.  Continued training therefore improved Generator-Matching
loss without improving average high-QED steering.  Exact comparisons are in
`diagnostics/qed_step1000_target_response.json`; live machine-readable status
is in
`diagnostics/qed_step1000_continuation_status.json`.

## Valid-successor QED guidance gate

The direct-conditioned evaluator never scores intermediate successors with
QED; it computes QED only after a trajectory ends.  A frozen-state gate now
tests a distinct non-beam controlled CTMC on 100 saved step-1000 states.  At
each state it draws four marks from the learned rate model, commits them with
the validity-closed executor, scores each canonical successor once, and
stochastically resamples while preserving sampled multiplicity and total
hazard:

`q_beta(m|x,t) ∝ p_theta(m|x,t) exp(beta U(QED(F_m(x))))`.

With `beta=8` and `U(q)=-|q-0.9|`, all 400 proposed marks produced valid,
connected successors.  They represented 343 within-state canonical groups and
338 unique QED evaluations.  Relative to the oracle-consuming beta-zero
control, the soft tilt improved expected QED by 0.00956; offline best-of-four
selection improved paired QED by 0.02906 on average.  The support therefore
contains exploitable moves, although the stochastic effect is modest.  The
only authorized follow-up is a ten-seed, 48-proposal-mark-per-attempt
controlled rollout.  Exact gate results are in
`diagnostics/qed_step1000_successor_guidance_gate.json`.

That controlled rollout produced 10/10 valid and unique molecules in both
arms, with all scored proposals and intermediate states valid and connected.
The QED mean rose from 0.6666 in the oracle-consuming beta-zero arm to 0.6903
under the target-distance tilt, a paired increase of 0.0237.  Three seeds
improved, two worsened, and five were unchanged; a ten-sample paired bootstrap
interval was [−0.0494, 0.1047].  Neither arm reached QED 0.9, so this remains a
positive developmental mechanism result, not benchmark authorization.

Persisted-log analysis shows why the effect is limited.  The first twelve
controlled events cover only 47–48% of each path and their proposal sets have
mean QED spread 0.0245, versus 0.0532 across the all-progress frozen-state
gate.  They include essentially no atom-restate or ring-grow proposals, even
though those families occur frequently later.  The sole recommended next
controller change is therefore to move the same twelve-event, K=4, beta=8
window to events 13–24 while holding the 48-mark budget fixed.  That exact
single-change run completed as
`compose-v4-qed-step1000-controlled090-late13to24-eval10-k4-b48-beta8-20260720-v1`
(Modal app `ap-JOHZnG1YUxtUhcRYq4beZW`); no sweep or other controller change
was launched.  Both arms were 10/10 valid and unique with all states and
scored proposals valid and connected.  The late target-distance controller
raised mean QED from 0.7096 to 0.7732, a paired increase of 0.0636; seven seeds
improved, one worsened, and two tied.  Two guided outputs reached QED 0.9.
However, the 10-seed bootstrap interval was [-0.0703, 0.1716], and short
trajectories consumed only 392 beta-zero and 384 guided marks.  The declared
48-mark per-attempt budget was therefore a ceiling, not an exactly matched
realization.  The stronger point estimate supports the valid-successor control
mechanism, but it does not select the late window definitively or authorize the
GrIDDD benchmark.  Exact analysis is in
`diagnostics/qed_step1000_controlled_rollout_analysis.json`.

If it improves the frozen validation loss, rerun only independent control,
0.7, and 0.9 eval10 arms.  Require 100% validity/connectivity, a QED-0.9 paired
shift larger than +0.0673, and at least one QED≥0.9 output before considering
a constrained-optimization smoke test.  Exact dose results are in
`diagnostics/qed_step500_condition_dose_response.json`.

## GrIDDD boundary

The present experiment is de-novo generation from a carbon-tree source prior.
It does not take a QED-0.70–0.80 lead molecule as input and does not enforce
fingerprint similarity to that lead.  A faithful GrIDDD comparison still
requires 800 starting molecules, 20 candidates per start, QED in [0.9, 1.0],
fingerprint Tanimoto similarity at least 0.4, and success over all 800 starts.
That full benchmark must not be launched until the input-lead and similarity
protocol are implemented and pass a small, all-start smoke test.
