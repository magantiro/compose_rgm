# Lazy reference lookahead repair

2026-09-09. User-authorized narrow repair following the failed production audit
at 626c0a2. The scientific problem and COMPOSE identity remain those in
HIERARCHICAL_TASK_SEARCH.md. Outputs are executable trajectories and feasible
candidate molecules, not ring templates. The claim tested here is cheaper
reference trajectory sampling with unchanged reference support and law, followed
by evidence reaching the existing task-guided decisions. No docking improvement
or coverage of every IVG winner is claimed.

The failed audit remains frozen. Do not change its routing thresholds or label
it passing. Reuse its exact saved parent and archive for a bounded cost check.
The baseline is eager execution of all products under the same reference.
The existing adaptive importance-weighted planner remains available unchanged.

## Declared change

Add an opt-in `lazy_reference` planning policy. Planning draws only from the
original hierarchy reference, so suffix importance ratios are exactly one.
This replaces adaptive planning proposals, not the reference law or committed
KL-controlled decisions. It is a different finite-compute estimator, not a claim
of seedwise equivalence to the adaptive planner. Retain complete rows when
making committed decisions. Preserve all augmented state identities and cache
only exact physical products and option-specific validity outcomes.

For WHAT, determine whether each option has at least one clean product without
materializing every product. Keep generic permanently active and use the same
purpose-balanced prior and exploration floor. For HOW, sample a mixture
component first, then reject invalid products within that component. Selecting
the component again after each rejection would change the inherited mixture
weights and is forbidden. Construction branches remain uniform over the
product-applicable branches, not weighted by their number of legal marks.
Exhaustion is an interrupted rollout, never a zero terminal observation.

No changes to R_theta, executor, primitive support/caps, macro temperature or
floor, region prior, option menu, guidance radius, task predictor, feasibility,
or the full 16-edit terminal stopping policy. No winner SMILES, motif counts,
or winner similarities enter rewards, priors, or selection. Connected,
charge-preserving, at-most-40-atom support and stereochemistry limitations stay
fixed. Five/six-member C/N/O construction channels are not universal ring support.

## Minimum verification and launch boundary

Use small fixtures to check conditional-mixture law parity with invalid marks,
uniform applicable construction branches, exact state/lineage continuity,
empty support and honest interruption, and delayed value reaching all three
decision levels. Compare lazy/eager executor work on the same production-executor
fixture. Run the affected dependency tests, lint, format and clean-source checks.
No unrelated full-suite rerun is a development gate.

A production cost check must be separately recorded with exact source and code
hashes, one saved parent, the existing 512-call planning limit, zero oracle calls,
and no automatic retry. Do not launch another eight-parent audit or docking
comparison merely because unit tests pass. Any terminal stopping-policy change
or wider ring menu is a separate scientific decision.

The frozen production probe is configs/t4_lazy_reference_probe.json: the same
first-ranked parent from the 51-call archive, 16 primitive edits, 2,500 total
public executor calls including at most 512 for planning. One CPU, 8 GiB,
one worker, no retries, 900-second administrative timeout. Expected 3-10 minutes
including initialization, not a measured runtime for the repair; maximum
15 CPU/8-GiB container-minutes, dollar rate unverified. Persist the parent and
lock, replay selected edits separately, and report interrupted attempts. Record
public executor calls inside marked-law enumeration separately from option
product materialization: lazy option draws cannot eliminate an upstream
enumerator's own validation cost. A one-parent probe does not evaluate or waive
the frozen multi-parent audit gate.
