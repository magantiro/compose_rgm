# Amendment — R_theta shortlist-retention diagnostic (MOLLEO prerequisite)

Recorded before the data. Uses ONLY the consumed 64-source QED development
panel. Nothing from the running 128-source validation is inspected or used, so
that panel stays prospective.

## The question, narrowed

MOLLEO Task 3 imposes a **<= 10,000 oracle-evaluation budget**, and COMPOSE
currently sits in the wrong query-complexity class (~2,187 oracle evaluations
per preference trajectory even in the cheapest guided arm). QED never exposed
this because QED and Tanimoto are free RDKit calls; five expensive objectives
are not.

The proposed fix is `exact legal fiber -> R_theta shortlist -> expensive
objective evaluation -> control`. This diagnostic asks the ONE question that
decides whether that primitive is viable, before any five-objective apparatus
is built:

> Does an R_theta top-K shortlist retain the actions a qualified controller
> actually needs?

R_theta is a PLAUSIBILITY model, not a purpose model -- prior goal-aware
prioritisation work already found R_theta ranking weaker than purpose-aware
ranking. So the risk being tested is specific: that plausibility pruning
deletes precisely the rare exploratory edits that make long-horizon control
work.

## THE DATA DOES NOT EXIST YET, AND WHY THAT IS FINE

The H40 development runs persisted only per-candidate summaries. `n_transitions`
is a COUNT; no (x, y) pair, chosen table or coordinate was ever stored. The
rescue trajectories therefore cannot be mined from the banked records.

They are exactly reproducible. Seeds are `sha256(PROTOCOL|arm|source|k)` with
`PROTOCOL = "hphi-horizon-v4"`, candidates are independent for `arm="restart"`,
and slice parity reproduced banked candidate 1 on 64/64 sources. Re-running a
chosen (source, candidate) yields the identical trajectory, so the diagnostic
re-runs the successful candidates with the fiber enumerated in the same pass.

A re-run that does NOT reproduce its banked candidate summary invalidates that
trajectory's contribution and is reported, not silently kept.

## Populations, fixed now

1. **All chosen transitions** -- the general picture.
2. **Successful trajectories only** -- moves on routes that actually solve QED.
3. **Target-entry transitions** -- the step entering the region. Losing one of
   these breaks the solution outright.
4. **H40 hard-source rescues** -- PRIMARY. These are the trajectories the
   horizon bought us, and the ones a shortlist must not destroy.

## Metrics, fixed now

For each executed transition, enumerate the EXACT legal successor fiber at that
state, score every canonical successor under the frozen R_theta, and record the
rank of the action the controller chose.

    Recall(K) = P[ rank_Rtheta(a_chosen | x) <= K ]   at K = 4, 8, 16, 32, 64, 128

plus the full rank CDF, median, p90, p99, and the fiber size distribution
(a rank is meaningless without knowing what it is a rank out of).

These K values are DESCRIPTIVE checkpoints for characterising one rank
distribution. This is explicitly NOT a K sweep for efficacy: no operating point
is selected by trying several and keeping the best. One shortlist size is
chosen prospectively afterwards.

**Unit of analysis.** The transition is the unit, but summaries are reported
per trajectory and per source as well. 1,280 transitions from one run are not
1,280 independent observations -- the same error that made a 96-positive AUC
look meaningful when it rested on 3 trajectories.

## The refinement that tests the right thing

Retaining the one action that happened to be sampled is weaker than retaining
what purpose-aware control WANTS. So additionally: rank the fiber by the
controller's value signal and measure what fraction of the purpose-top-m
successors survive an R_theta top-K shortlist.

Cost forces a split, and it is declared here rather than discovered later:

- `h_phi` on a full fiber is ~1 evaluation per successor, so it is computed on
  a STRATIFIED SUBSAMPLE of decision states.
- `slack = -(max(0, 0.90 - QED) + max(0, 0.40 - sim))` is free and is computed
  on ALL states. Slack earned this role empirically: it reached run-level AUC
  0.94-0.99 as a prospective signal where h_phi sat at 0.58-0.73.

Both are reported. Agreement between them on the subsample is itself evidence
about whether the cheap proxy can stand in.

## Decision rule, fixed now

- **High retention on the rescue population** (e.g. ~97% of chosen actions
  within top-32, all target-entry transitions within top-64) -> the shortlist
  primitive is supported; choose ONE operating point prospectively and proceed
  to query-efficient COMPOSE.
- **Rescues depend on deep ranks** (e.g. median 18 but p90 170, p99 450) ->
  naive top-K plausibility pruning is the WRONG primitive. Fix it before
  spending any five-objective oracle budget. This outcome is as valuable as the
  first and is not a failure of the diagnostic.

## What this does not license

Changing anything about the running 128-source validation, or selecting a
shortlist size by trying several and keeping whichever reads best.
