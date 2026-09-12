# Option-space controlled particle policy iteration, implementation milestone

Status: prospective implementation contract, recorded 2026-09-12 before model
fitting or controller evaluation.

## Identity

- Scientific problem: efficiently discover high-utility molecules through long,
  valid, reversible COMPOSE rewrite paths when useful structural transformations
  have low primitive probability and delayed reward.
- Primary output: at each option boundary, a probability distribution over an
  applicable `(region, structural option, option parameters)` decision, plus a
  distribution over improvement reachable within the remaining option budget.
- Central claim under test: temporal abstraction, policy-congruent continuation
  value, conservative proposal learning and persistent controlled particles can
  improve search efficiency without changing the frozen molecular reference or
  executor support.
- Initial setting: retrospective development evidence from exact stored PMO and
  T4 trajectories. This is not a matched benchmark evaluation.
- Primary baselines: balanced option reference, current-score ranking, the
  rejected achieved-route head, independent option trajectories, and post-hoc
  acquisition on the same completed candidates.
- Declared support: complete connected molecular graphs inside the current
  production editing contract, at most 40 heavy atoms, charge-preserving broad
  organic vocabulary, exact persistent-slot identity, existing executor and
  canonicalization. `generic` remains available. Options compile to ordinary
  primitives and never commit an endpoint directly.

## Authorized change

Implement the reusable offline controller core:

1. censored best-improvement targets with explicit behavior-policy identity;
2. a monotone distributional improvement model over remaining option budget and
   improvement threshold;
3. an advantage-weighted option actor mixed with an applicability-aware base
   distribution and explicit exploration floor;
4. persistent option-boundary particles with exact proposal/reference log-ratio
   correction, deterministic resampling and round-trip serialization;
5. locked-candidate batch acquisition from joint posterior samples;
6. an offline retrospective evaluator that either reports source-grouped
   evidence or abstains when the stored trajectories do not support the claim.

Use existing paid labels and stored exact trajectories only. New oracle calls,
Modal deployment, accelerator training, reference training, executor changes and
live benchmark runs are outside this milestone.

## Frozen decisions

- The option clock, not a nominal primitive maximum, defines remaining budget.
- Future labels come from continuations produced by the declared behavior-policy
  snapshot. A mismatched snapshot is rejected unless an explicit importance
  weight is supplied.
- Distributional value predicts whether the best utility seen by a horizon
  exceeds the incumbent by a declared threshold. Endpoint prediction remains a
  separate object.
- Proposal learning changes the option controller only. It does not change
  `R_theta`, `Q(M)`, executor support, or the qualified region controller.
- The learned proposal retains every applicable base option with a positive
  floor; `generic` is mandatory.
- SMC records both reference and proposal probabilities. Learned proposal
  probability is not silently treated as the reference target.
- Acquisition operates only on complete, canonical-deduplicated, locked
  candidates and is separate from the continuation value.
- Compound-option infrastructure is allowed. The existing 11-step
  `BUILD_RING_SYSTEM` program remains disabled by default pending the inherited
  user decision.

## Acceptance criteria

The milestone candidate must satisfy all of the following with local runnable
commands:

1. Target construction handles success, failure and right-censoring without
   inventing future labels, and rejects mixed undeclared policy snapshots.
2. Predicted exceedance probability is nondecreasing in remaining option budget
   and nonincreasing in improvement threshold by parameterization.
3. Actor distributions normalize on the supplied applicable set, preserve a
   positive per-option base floor, require `generic`, and obey the declared
   option-level KL ceiling.
4. Persistent particle weights implement
   `log w += log B - log q + log h_next - log h_previous`; `q=B` reproduces the
   existing update. Extinction, nonfinite inputs and zero proposal support fail
   explicitly.
5. Serialization round-trips exact encoded state, histories, weights, potential,
   RNG identity and controller snapshot. Deterministic resumption reproduces the
   same resampling decision and descendants.
6. Joint-posterior batch acquisition returns unique candidate identities and
   reproduces exact probability-of-optimality frequencies on a bounded fixture.
7. An exactly enumerable toy SMDP verifies proposal correction and the exact
   option-boundary twist. Focused existing option and twisted-SMC tests remain
   green.
8. Retrospective fitting uses a source-grouped split fixed before features or
   fitting. It reports calibration, ranking, improvement precision and coverage,
   or publishes an explicit insufficient-coverage abstention.
9. Touched-code formatting, lint and focused tests pass. Repository-wide
   verification is run once only after the milestone candidate is frozen.

## Decision after the retrospective gate

- Positive: freeze a bounded paired live recipe against the strongest actual
  controller, with a separate user-authorized oracle and compute budget.
- Negative: do not scale particles or guidance. Repair option coverage or the
  continuation target named by the failure.
- Inconclusive: retain the implementation, record the missing contrast, and
  request only the smallest locked label or rollout set that resolves it.
