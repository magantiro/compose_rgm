# T4 matched cold-start development pilot

## Identity and scope

COMPOSE learns an executable molecular rewrite process, not a ring-only generator.
Its marked rate law induces complete supported molecular successors through the
production executor. This pilot tests whether the frozen committor improves the
usefulness of proposals from the combined local-to-global, region-option controller.
It does not test a new learned docking value, Monte Carlo lookahead, generalization,
or superiority over IVG. PARP1 seed0 has already been inspected and is development
data, not a sealed evaluation source.

The user authorized this bounded comparison after the verified fused-path audit.
No editing training milestone, new corpus, broader benchmark, or weight tuning is
authorized here. Ring and non-ring options remain available. The fused C6 program
is opt-in and retains its separately documented bounded support analysis in
FUSED_OPTION_INTEGRATION.md; it does not expand executor or model support.

## Frozen comparison

- One cold-start round per arm on PARP1 seed0, similarity threshold 0.4.
- Arms: frozen committor guidance versus its option-conditioned reference law.
  Both use the same generator and the same post-hoc candidate selection.
- Identical seed 1000, eight initial lineages, three region draws per lineage,
  applicability-aware balanced option prior, generic permanently available.
- One particle per draw, at most eight frontier states and three emitted
  representatives per bundle. This smaller development compute budget applies
  equally to both arms; horizons and molecular support are unchanged.
- Generic horizon 16, existing BUILD_RING_SYSTEM horizon 11, opt-in fused horizon 5,
  ordinary options one step. Incomplete compound programs are not oracle candidates.
- Frozen R_theta, committor, Q(M), kappa=1, primitive exploration 0.1,
  region floor 0.2, option floor 0.1, macro temperature 2 and exploration 0.15.
  The reference arm samples R directly; it does not change kappa or retrain anything.
- No docking labels are available during proposal or allocation. The existing
  cold-start feasibility/QED ranking is used within bundles in both arms.
- Canonical deduplication within each arm; at most 20 docking attempts per arm.
  Both complete candidate locks and identical outer draws must be verified before
  either arm invokes docking. No second round, backfill, or outcome-based retry.

## Resources, stop and recovery

One CPU proposal worker per arm, sequential arms in one container to reuse model
initialization. One CPU docking worker, no GPU, 8 GiB memory. Common ceiling:
20,000 public executor applications per arm (including calls inside enumeration).
Whole-job timeout 3600 seconds, zero automatic retries. A cap or failure stops
the comparison without spending docking on an unmatched proposal pair. Expected
preparation is approximately 10–30 minutes total, uncertain until measured; docking
adds approximately 1–3 minutes. Maximum allocation is one CPU-hour and 8 GiB-hour;
actual provider billing rates are not assumed. Completed parent batches and arm
locks are durable units; a rerun must reuse valid locks, not regenerate them.

## Acceptance and interpretation

Verify input and serialized-code hashes, unchanged runtime gates, identical outer
plans, complete valid program endpoints, and lock-before-oracle ordering. Record
all attempts including failures; proposal time, executor/model work, oracle calls,
feasible scores, canonical diversity, selected options, intended region release,
realized structural displacement, cycle rank and ring-system deltas. Save exact
configuration, RNG seed, software, hardware, input identities, and timestamps.

A better score from one paired source is preliminary development evidence only.
A null result, no selected fused bundle, unequal realized work, or candidate
shortfall must be reported without resampling or retuning. This pilot cannot show
online docking adaptation or that the committor predicts docking affinity.

## Evidence status

Proposed and authorized, not yet run. Existing fused reference evidence proves one
valid construction path, not guidance superiority. Repository-wide historical
failures remain disclosed; focused checks of this comparison must not be presented
as a green full repository suite.

Implementation checks before launch: 61 focused tests passed (paired pilot,
option selector, fused option, fused reference audit, T4 reporting). The synthetic
integration fixture uses the real T4 preparation body and production executor,
with explicitly model-free marks and forced region/option draws. It is not a
learned-generator chemistry result. New module, launcher and tests pass lint and
format checks; the legacy Modal app retains its 17 existing lint diagnostics.

The integration also corrects exact-state cache identity to include charge and
implicit hydrogen channels, and shared-particle identity to include lineage
next-ID and fused program progress. These changes protect persistent-slot
semantics; no executor, vocabulary, region prior, or primitive support is altered.

## Completed development decision

The paired round completed from `219c1cd`, with 20 docking attempts per arm and
no docking failures. Both arms found the same best molecule; its unseeded docking
scores were -8.3 reference and -8.2 guided, not evidence of a guidance effect.
All three construction programs in each arm stopped before closure. Saved-state
analysis verified that the SMILES-index-based append-system contract falsely
rejects two valid six-membered pendant closures in one bundle in both arms.
The other two bundles' recorded closures miss the existing size requirement.
No contract repair or additional docking run has been performed.

The complete evidence, limitations, and next repair target are in
`diagnostics/t4_matched_pilot/attempt_1/README.md`. Preserve the general
region-option controller; repair atom-correspondence checks before interpreting
these construction failures as a limitation of its search space.
