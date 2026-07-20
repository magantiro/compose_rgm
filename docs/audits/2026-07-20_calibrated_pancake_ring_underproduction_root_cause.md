# Calibrated pancake ring-underproduction audit — 2026-07-20

## Decision

The calibrated pancake underproduces cycle rank and fused rings for two
multiplicative reasons: it commits modestly fewer whole-ring events than the
cached teacher, and each committed ring event produces substantially less
cycle rank.  The strongest common cause is the compiler's nearly deterministic
late ring schedule interacting with state-dependent legal support and
within-ring template selection.  The evidence does **not** support rebuilding
the ring catalog, globally boosting the ring family, disabling Graft, or
installing every ring as aromatic.

The smallest plausible ground fix is to reuse the existing exact
endpoint-preserving commutation scheduler so that whole-ring teacher steps are
shown earlier whenever they commute.  Before any training, a 32-state
read-only support/score comparison can cheaply falsify that hypothesis.

## Frozen evidence

This audit made no model or artifact mutation and launched no training.  It
uses:

- `/private/tmp/compose_v4_stage3_validation_paths.pt` (4,000 frozen records);
- `/private/tmp/pancake_checkpoint/checkpoint.recovery.pt` (selected step
  6,250);
- `diagnostics/pancake_step6250_eval2000_event_audit.json`;
- `diagnostics/pancake6250_calibration_eval600_metrics.json`;
- `diagnostics/ring_calibration/step2500_exact_support_audit_12.json`; and
- `diagnostics/canonical_successor_analytic_rollout_smoke20.json`.

A read-only aggregation over all 4,000 cached paths and all 4,096 checkpoint
ring templates supplied the path/catalog counts below.  The endpoint fused
test is the same graph criterion used by the rollout taxonomy: two perceived
rings share at least one bond.

## Attribution by layer

### 1. Target path and teacher ring frequency/timing — primary cause

The 4,000 paths contain 280,686 teacher steps.  Whole-ring grow contributes
9,946 steps (3.543%), or 2.4865 steps per path; 98.65% of paths contain a ring
step.  Thus the teacher does not omit rings.  It does, however, teach them
almost exclusively at the end:

- mean first ring position: 0.97464;
- mean position of every ring step: 0.98436; and
- 39.67% of ring steps are literally the last path step.

The independent 2,000-rollout checkpoint audit agrees: mean first-ring
position is 0.97837, 99.31% of sampled ring events lie in the final event
decile, and ring grow is the final rule in 94.3% of trajectories.  This is the
signature of the fixed compiler linearization, not a time-coherent molecular
rate law.

The cached teacher topology mix is also healthy enough to reject the claim
that fused systems are absent from supervision: 72.49% of ring steps are
`single_ring`, 26.20% are `bridged_or_fused`, 1.287% are `spiro`, and 0.020%
are macrocycles.  Cached targets have mean cycle rank 3.454 and fused
prevalence 60.45%.

### 2. Legal ring support/availability — important state-conditioned cause

The exact 12-state support audit is from the related step-2,500 lane rather
than the step-6,250 pancake, so it is supporting rather than fully matched
evidence.  Its executor/family semantics are the same.  Broad states retain
113--1,098 legal templates and assign only about 1.7--4.6% production mass to
small rings.  Late decorated states can instead retain only 2, 2, or 5 legal
templates, all of them small-ring topologies.  Once the Boolean hierarchical
family head selects ring grow, that residual support is renormalized to 100%.

This proves that support quality can collapse late even though the global
catalog is broad.  It directly explains the observed excess of small rings
and gives a plausible mechanism for losing common fused options before ring
commitment.  A matched pancake topology-by-support audit is still needed to
quantify the fused component.

### 3. Family prediction versus mark/topology prediction — mark side is the
larger uncertainty

The selected pancake checkpoint reports 84.44% overall family accuracy but
only 6.54% mean exact-teacher-mark probability.  These are aggregate metrics,
not ring-specific ones, so they cannot by themselves convict the ring mark
head.  Still, the large gap says family recognition is much stronger than
exact operand/template recognition.

Rollout counts reinforce that distinction.  The calibrated 600-sample arm
commits 1,274 ring-grow events, or 2.123 per molecule.  That is only 14.6%
below the teacher's 2.4865 ring steps per path.  In contrast, cycle rank per
ring event falls from approximately 1.389 in cached targets
(`3.454 / 2.4865`) to 1.158 in calibrated endpoints
(`2.4583 / 2.1233`).  Ring-event count and per-event cycle yield are therefore
both about 15--17% low; multiplied, they reproduce the approximately 29%
endpoint cycle-rank deficit.

The missing matched quantity is the frozen checkpoint's conditional
probability of the teacher topology/template on ring-teacher rows.  It belongs
in the cheap falsification test below, not in a training run.

### 4. Sampler and base-rate calibration — not the root topology cause

The incumbent calibration multiplies atom-delete intensity by `exp(-0.5)` and
only 3/4-member ring-action intensity by `exp(-1.5)`.  Exact CTMC thinning
does not renormalize ordinary or fused ring rates and never enables or removes
their support.  On 100 matched samples it increased mean cycle rank from 2.42
to 2.56 while improving size, bond-order, cycle-rank TV, and small-ring
prevalence.  It therefore did not create the fused/cycle deficit.

The lossless analytic quotient smoke further preserved the incumbent's event
mix (family TV 0.0455), size (delta -1.70 atoms), cycle rank (delta -0.20), and
event count (ratio 1.012) while removing committed backtracks.  Canonical
execution is not hiding an additional ring-rate collapse.

### 5. Whole-ring template coverage — adequate; do not rebuild

The step-6,250 catalog is full at 4,096 templates.  By template count it
contains 3,561 `bridged_or_fused` templates (86.94%), 298 spiro templates
(7.28%), 209 single-ring templates (5.10%), and 28 macrocycles (0.68%).  By
empirical prior mass the same classes receive 22.64%, 0.72%, 76.58%, and
0.063%, respectively.  The cached teacher's 26.20% bridged/fused event share
is close to the catalog's 22.64% prior mass.  Small rings are only 2.85% of
catalog prior mass.

Coverage is therefore not the binding failure.  The problem is which portion
of the catalog remains legal and is selected at the late state.  Rebuilding or
enlarging the catalog would add cost without addressing that conditioning.

### 6. Competition with Graft/delete — secondary at most

Cached teacher paths are themselves 78.36% Graft and only 3.54% ring steps.
The calibrated generator is 49.46% committed Graft and 7.06% committed ring
grow, so ring grow is *less* crowded in relative committed-event terms than in
the teacher.  Exact thinning also prevents presentation aliases from stealing
other transition intensities.  Graft is a throughput concern, not the best
explanation for fused underproduction.

Delete calibration improves size, which should generally preserve or expand
ring opportunity.  Only one ring-delete event occurs across the 600-sample
calibrated audit, so rings are not being generated and then destroyed.

### 7. Endpoint nonlinear effects — amplifies the upstream error

Fused prevalence is a thresholded molecule-level statistic: a molecule needs
at least one appropriate overlapping/multicycle installation.  A modest
reduction in both ring-event count and multicycle yield compounds into a much
larger prevalence change.  This is consistent with 2.458 versus 3.353 mean
cycle rank but 28.5% versus 58.0% fused prevalence.  It is an amplification of
the rate/topology error, not an independent executor failure.

## Ranked causal assessment

1. **High confidence:** fixed late teacher ordering; it is directly measured
   on both cached paths and generated trajectories.
2. **Medium-high confidence:** late-state legal-support collapse and
   within-ring topology/mark selection; exact support examples and the
   per-event cycle-yield deficit support it, but a matched pancake audit is
   still missing.
3. **Medium confidence:** total productive ring-family intensity; absolute
   ring events are 14.6% low, so it contributes but cannot alone explain fused
   prevalence.
4. **Low confidence:** missing catalog coverage, Graft/delete competition, or
   inference calibration.  Existing evidence argues against each.

## Smallest plausible fix and cheap falsification

Reuse the already implemented adjacent-commutation scheduler; do not add an
operator, catalog, aromaticity heuristic, or new network.  For a deterministic
32-row sample of cached paths containing `bridged_or_fused` or spiro teacher
steps, compare each original pre-ring state with the earliest exactly
commuting placement already certified to reach the identical endpoint.

For both states, record:

1. exact legal templates and topology groups;
2. empirical-prior and frozen-model mass on single, bridged/fused, spiro,
   macrocycle, and small-ring groups;
3. ring-family probability versus conditional teacher-template probability;
4. whether the teacher action remains executable; and
5. support construction wall time.

The timing hypothesis passes this cheap gate only if all 32 rescheduled paths
replay to their exact endpoints and the earlier state materially improves
ordinary multicycle opportunity—for example, at least a 2x median increase in
legal bridged/fused prior mass or a 50% reduction in tiny/degenerate
(`<=5`-template) support—without increasing small-ring-only support.  If those
support changes do not occur, stop: scheduling is falsified as the immediate
ring fix.  The next smallest diagnostic would then be a frozen ring-teacher
family-versus-template score decomposition, still without training.

Only after a positive support gate should a bounded resume from pancake
weights be considered.  Nothing in this audit authorizes training.
