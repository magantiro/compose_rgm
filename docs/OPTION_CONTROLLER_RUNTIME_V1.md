# Option-controller runtime V1

Status: implemented mechanical integration, 2026-09-12. This is not a trained
guidance result or a benchmark result.

## Purpose and support

The runtime composes the controller selected in
`CONTROLLER_LITERATURE_DECISION_2026-09-12.md` over the existing production
molecular process:

\[
Q(M\mid x,z)\;\longrightarrow\;Q(o\mid x,M,z)\;\longrightarrow\;
q(\omega\mid x,M,o).
\]

The generated object remains a complete molecular graph represented by its
exact persistent-slot state. WHERE uses the qualified region law, WHAT uses the
applicability-aware balanced option law plus a conservative learned actor, and
HOW uses the existing option continuation kernel and executor. Options compile
to supported primitive rewrites. They never commit a target endpoint directly.
`generic` is mandatory. Parameterized pendant and fused ring options remain
available when applicable. The inherited 11-step `BUILD_RING_SYSTEM` option is
disabled unless a later contract explicitly activates it.

## Frozen runtime contract

- Actor and distributional-value tensors are copied, made inference-only and
  checked against their content identities before a controller is constructed.
- Value features are computed before WHERE and contain molecular descriptors,
  oriented current utility, path incumbent and the remaining option clock.
- WHAT additionally receives numeric region descriptors. This prevents a value
  model from depending on a region that has not been selected yet.
- Utility is supplied by a versioned callback and must already be oriented to
  `[0,1]`, with larger values better. The runtime owns no task oracle.
- A persistent population binds exact molecular states, histories, RNG streams,
  actor/value/HOW identities, initial incumbent and option horizon. Changing the
  controller or incumbent during a run fails closed.
- Best utility is pathwise state. Once a particle improves, later structural
  deterioration cannot erase that improvement from the exact terminal
  best-of-run potential.
- Every completed option records the full reference and proposal log
  probabilities for WHERE, WHAT and HOW. Particle updates use
  `log B - log q + log h_next - log h_previous`.
- Wall time and cumulative kernel-work counters are operational receipts, not
  persistent stochastic history, so a serialized resume is deterministic.
- Only complete option-boundary molecules can be locked. Locking uses canonical
  molecular identity, retains exact state and ancestry, and occurs before joint
  posterior batch acquisition.

## Implementation

- Runtime: `src/compose_v4/control/option_controller_runtime.py`
- Boundary actor/value: `src/compose_v4/control/option_controller.py`
- Features: `src/compose_v4/control/option_features.py`
- HOW trace accounting: `src/compose_v4/control/option_continuation.py`
- Persistent SMC: `src/compose_v4/control/persistent_option_smc.py`
- Acquisition: `src/compose_v4/control/batch_acquisition.py`

The model-free production-executor fixture runs a full WHERE/WHAT/HOW option,
persists and resumes the population exactly, locks canonical candidates, and
applies joint probability-of-optimality acquisition. It uses no oracle labels,
Modal job or accelerator.

Focused verification on the implementation candidate passed 57 tests covering
the runtime, boundary controller, option continuation, value targets, actor,
persistent SMC, acquisition, option selection and exact twisted-SMC fixture.
Touched Python files pass Ruff lint and format checks, and `git diff --check`
passes. Repository-wide collection remains blocked by the previously recorded,
unrelated Editing-V2 registry/live process-identity mismatch; it was not repinned
or retried for this runtime change.

## Claim boundary and next gate

The execution engine is now present. Learned control is not yet scientifically
enabled because the retrospective bank contains no positive policy-congruent
improvement cells. The next experiment must first obtain a frozen continuation
bank with both improving and non-improving outcomes, then validate the value and
actor by source group. Only a positive held-out gate can authorize a matched live
controller comparison.
