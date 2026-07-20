# Qualified analytic zero-sidecar GrIDDD rewrite smoke

**Date:** 2026-07-20
**Status:** execution/accounting gate passed; learned QED efficacy not tested

## Modal failure diagnosis

The attempted canonical training app `ap-Hhfeh6k91K5Y0Fyd1vELUl` spawned call
`fc-01KY09913W095CCAPXCYD63D9Y` with run label
`compose-v4-griddd-qed-frozen-residual-pilot-20260720-v2-canonical`.
The frozen pilot config names
`compose-v4-griddd-qed-frozen-residual-pilot-20260720-v1`, and
`modal_apps/train_qed_frozen_residual.py` rejects a different label before it
creates the Volume run directory. Modal therefore reported the app as stopped
with zero tasks, and `modal volume ls` confirmed that no `v2-canonical`
artifact directory existed. No training checkpoint or QED-sidecar metric was
produced by that call.

The local entrypoint now performs the same run-label check before spawning, so
a future mismatch fails visibly without emitting a misleading function-call ID.

## Smoke definition

The bounded local gate uses:

- the passed analytic qualification manifest and retained checkpoint SHA-256
  `47716924f7798ed24556c5aa8fb10c533c55dbd1f02f8f53a463cf2ad80ae2bf`;
- one real lead with QED `0.7435014993311442`, inside `[0.70, 0.80]`;
- target QED `0.90` and Morgan-radius-2 Tanimoto threshold `0.40`;
- one candidate in each of the direct, controller, and combined arms;
- four real canonical rewrite events per arm; and
- an exact allowance of six QED calls per arm: one lead call, four guidance
  calls, and one terminal call.

The QED residual sidecar is zero initialized and untrained. Its target-present
and missing-condition family rates are exactly identical to the qualified
analytic base at model times `0.0`, `0.2`, and `0.4`. Thus this gate tests real
rewriting, conditioning plumbing, safety, and oracle accounting. It does not
test a learned direct-conditioning effect.

## Result

The execution gate passed:

- all three arms applied four real events and returned valid, connected states;
- each arm used exactly `6/6` QED calls;
- direct used zero selection-influencing calls, while controller and combined
  each used two selection-influencing calls and two explicit padding calls;
- each final molecule had Tanimoto similarity `0.42857142857142855`, above the
  frozen `0.40` threshold; and
- all three final molecules had QED `0.39605397258456815`, so none reached
  `0.90` and the observed success rate was `0/1` in every arm.

The final QED decrease is not surprising evidence about a trained conditional
model: there is no trained sidecar in this smoke. It is also not positive
efficacy evidence for the controller. The run authorizes only the execution and
exact-accounting claim. It does not authorize a GrIDDD-comparable result, the
800-by-20 benchmark, or any QED-optimization claim.

## Durable artifacts

- `scripts/run_griddd_analytic_zero_sidecar_smoke.py`
- `diagnostics/griddd_analytic_zero_sidecar_rewrite_smoke.json`
- `modal_apps/train_qed_frozen_residual.py`

At generation time, the metrics artifact SHA-256 was
`5aa1753f92bb29fffe7113fcb401d2ed358892d5d550203ab198d25e532abb63`.

## Next gate

Launch the frozen 500-step QED sidecar with the exact config run label, confirm
that step-0 and step-100 artifacts are written, and then rerun this same
three-arm smoke using the best trained sidecar checkpoint. Only the trained
rerun can begin to answer whether direct or combined conditioning improves QED.
