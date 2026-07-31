# Editing T1 prospective optimization and capacity thresholds

Status: frozen before any V4 T1 result was generated.

T1 asks whether the scratch Active8 architecture can represent the required
canonical molecular-successor laws on bounded, exact-address development
panels. It is a capacity diagnostic, not evidence of held-out generalization
and not authority for a long training run.

The optimization ladder is frozen in
`configs/editing_t1_optimization_v1.json`. Every applicable panel is evaluated
at 500 updates under the three declared parameter scopes. The learning rate,
weight decay and report points match the already implemented bounded ladder.

The numeric thresholds are frozen in
`configs/editing_t1_thresholds_v1.json`:

- unique-state teacher-successor top-1 recall must be at least 0.95;
- mean teacher-successor probability must be at least 0.80;
- mean teacher-successor negative log likelihood must be at most
  0.22314355131420976 nats, equivalent to a geometric-mean probability of
  at least 0.80;
- the global repeated-state successor law must have excess negative log
  likelihood over empirical entropy of at most 0.05 nats per observation.

These thresholds intentionally demand substantially more than a weak
above-uniform signal. A failure is retained as a negative result and blocks
the affected architecture/scope from supporting P50. Passing T1 does not
authorize P50 by itself; the exact Active8 inventory, Gate 0 structural
evidence, sealed T1 results and a separate decision artifact remain required.
