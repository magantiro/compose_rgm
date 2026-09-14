# T4 route-proposal quality gate

Status: proposed standalone zero-oracle development benchmark. It does not alter
the running route fit, Dynamic COMPOSE controller, or any scored launch.

## Problem and claim boundary

The output under test is a ranked pool of complete, exactly executed molecular
endpoints proposed from a benchmark source. The prospective claim is that a
source-conditioned policy ranks transformations capable of recovering held-out
teacher endpoints or narrowly equivalent local transformations. Module-label
classification, executor validity, and architecture alone do not establish that
claim.

The current `RouteDistilledProgramPolicy` is one baseline. Its recorded audit has
77 routes from 15 source groups and 1,372 recognized decisions, but 1,370 decisions
are primitive fallbacks. Its current proposal-rank metric is evaluated on the
fitted rows. Its aggregate binding prototype has no explicit WHERE label, prefix
dependency decision, parameter law, or learned stopping decision. These facts
make its high primitive-label top-1 accuracy non-equivalent to route recovery.

## Frozen data organization

All routes from one source molecule remain in one split. For the 15-cell T4
development corpus, use a predeclared three-fold grouped evaluation with one seed
per protein in each test fold. Target labels may stratify this offline split but
must not be serialized into a runtime checkpoint. Hyperparameters are selected
only on grouped training/calibration sources; the test fold is evaluated once.

Training weight for one decision is

```text
1 / number_of_sources
  / routes_for_this_source
  / decisions_in_this_route
```

This prevents a protein, source, duplicate route representation, or long primitive
trace from dominating. Duplicate exact endpoints and equivalent transformation
classes are collapsed within a source for recall denominators. Report all folds
and per-source rows rather than treating route decisions as independent samples.

## Metrics

For each policy, generate a bounded ranked pool from each held-out exact source
without an oracle. Report at ranks 1, 5, 10, 32, and 128:

- exact canonical-endpoint recall, any-hit rate, reciprocal rank, emitted precision,
  and fixed-rank precision;
- transformation-equivalent recall, any-hit rate, reciprocal rank, and precision;
- proposal attempts, exact-execution precision, unique-endpoint yield, shortfall,
  wall time, attempts per second, and unique endpoints per second.

All aggregates give equal total weight to each source. Exact endpoint identity is
the supported canonical molecular graph. Transformation equivalence uses exact
colored graph isomorphism over before/after atoms and bonds in the radius-two
union-graph change neighborhood, including typed boundary stubs. Weisfeiler-Lehman
hashing is only a prefilter. Atom/bond edit counts, fingerprints, module labels,
and descriptor distance alone are prohibited as equivalence tests because they
collapse chemically distinct changes.

The benchmark rejects self-events, duplicate ranked endpoints, overlapping source
splits, nonzero oracle calls, and rejected attempts carrying fabricated endpoints.
Stereochemistry is outside the current graph representation and must remain an
explicit limitation.

## Policy comparison

Use one common compiler/executor support and matched attempt, primitive, block,
binding-visit, candidate, and wall-clock ceilings.

1. `generic_marginal`: source-balanced marginal module and stop priors, generic
   bounded binding enumeration, and score-blind ordering. This measures corpus
   frequency without molecular context.
2. `context_module_prototype`: the current molecule-to-option actor, manual
   option-to-module mapping, module-count marginal, and aggregate stage prototypes.
   Preserve it unchanged as the current baseline.
3. `hybrid_autoregressive`: a small graph-conditioned policy over a bounded exact
   candidate panel. Factor each decision as
   `module -> WHERE -> HOW -> dependency -> stop`, conditioning every factor on the
   current exact prefix and remaining work. Reuse current global Morgan/topology
   features, invariant local attachment environments, generic module parameter
   descriptors, exact dependency handles, and v1 bounded binding enumerators.
   A practical first implementation can contrastively rank enumerated legal module
   instances instead of introducing a new graph decoder.

The hybrid checkpoint may contain shared weights, feature registries, source-balanced
marginals, and normalization fitted on training sources. It may not contain target
names, route IDs, teacher endpoints, exact programs, absolute persistent-slot IDs,
source SMILES, or target-to-policy maps. Runtime candidates are rebound against the
current graph. Retain a fixed generic exploration floor and an explicit stop action.

Evaluate policy ranking on a shared candidate panel as a causal ranking comparison,
and separately evaluate each policy's own end-to-end generated pool. Do not claim a
generation improvement from a shared-pool reranking result alone.

## Artifact contract

The standalone runner accepts `route_proposal_quality_input_v1` with disjoint
`train_sources`, `calibration_sources`, and `test_sources`, one declared evaluation
role, exact teacher endpoint states, and ranked policy attempts. `oracle_calls` must
equal zero. The runner writes an immutable, hashed report with input and code hashes.
It neither trains a policy nor launches a scored experiment.
