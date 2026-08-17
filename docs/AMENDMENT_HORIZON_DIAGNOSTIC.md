# Amendment — horizon reachability diagnostic

**Amends preregistration section 13**, which freezes H = 24. Recorded before the
data.

## The question

Rung 3 found 37 of 37 remaining hard sources failing with **zero particles ever
entering the region**. Two worlds explain that, and they imply opposite next
steps:

    World A, horizon-limited     routes exist but need more than 24 edits
    World B, search-limited      routes of <= 24 edits exist and the controller
                                 misses them

The archive line is closed: forced-diverse branching moved hard sources 12/37 ->
13/37, null on the pre-registered strata. So reallocating candidates within
H = 24 does not open the inaccessible part of the space. Distinguishing A from B
is prior to designing any better controller, because a rare-event method aimed
at World B is wasted effort if we are in World A.

## THE CONSTRAINT THAT SHAPES WHAT THIS CAN CONCLUDE

`h_phi` takes the remaining budget as a **one-hot over 0..BUDGET_MAX**, with
`BUDGET_MAX = 24` and `INPUT_DIM = 4*EMBED_DIM + 6 + (BUDGET_MAX + 1)`.
`build_features` RAISES outside that range, which is why every caller clamps
with `min(budget, 24)`.

So the horizon is not a configuration value. The network has no input for step
25, and running H > 24 means the first H - 24 steps all present budget = 24 to a
twist that therefore believes fewer steps remain than actually do.

Three consequences, and the asymmetry is the important one:

1. The twist is MISCALIBRATED during the extra steps. The extended arm is
   handicapped, not favoured.
2. Success and target contact remain EXACT, because the terminal boundary
   `h_0(x) = 1[x in B]` is exact and independent of the twist.
3. Therefore a POSITIVE result is conclusive -- more horizon helps despite a
   handicapped controller -- while a NULL is CONFOUNDED, since it cannot
   separate "horizon does not matter" from "h_phi cannot steer past its
   training range".

A null therefore does not license concluding World B. It licenses only
"H > 24 does not help *with this twist*", and answering the question properly
would need a budget-extended h_phi, which is a retraining decision and out of
scope here.

## Design, fixed before the data

- **Arms:** H = 24 (frozen control), H = 32, H = 40. Nothing else changes:
  same R_theta, same h_phi, same exact kernel, same N = 32, same region, same
  sources, same seeds per candidate index.
- **Panel:** the 45 marginal + hard sources. Reliable sources cannot inform this
  -- they already succeed at H = 24.
- **Candidates:** 2 per source per arm. This measures CONTACT, not coverage, so
  candidate count is a noise-reduction knob rather than the endpoint.
- **Primary readout:** fraction of runs with ANY particle entering the region,
  by stratum. Source success is secondary; contact is the reachability signal
  and success additionally depends on the terminal sampling draw.
- **Work is logged.** A longer horizon costs more transitions by construction,
  so this is a diagnostic and not an efficiency comparison; the extra work is
  the price of the answer, not a confound to be corrected away.

## Decision rule

- **Contact rises materially with H** -> World A. The lever is the horizon, and
  the next work is an adaptive-horizon controller plus a budget-extended h_phi.
- **Contact is flat in H** -> World B is SUGGESTED but not established, because
  of the clamp above. Committor-stratified weighted-ensemble resampling becomes
  the more compelling direction, with the caveat recorded.

## What this does not license

Reporting any H > 24 configuration as the controller, using it on the validation
panel or the official 800, or treating a coverage number from an extended
horizon as comparable to the banked H = 24 ladder. This is a diagnostic run on
the development panel only.
