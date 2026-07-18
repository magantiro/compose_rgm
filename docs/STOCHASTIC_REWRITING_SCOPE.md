# Stochastic rewriting scope for the base model

Stochastic rewriting is the foundation, not an obligation to adopt every
rewriting formalism or rule type.

## Adopted now

- Typed rule schemas with explicit application conditions.
- Concrete graph matches and structured operands as jump marks.
- A deterministic executor for every enabled marked rule.
- A CTMC generator formed by non-negative firing rates over enabled rules.
- State-space restriction: enabled rules must preserve the admissible molecular
  state space.
- Exact inverse rules where the state contains enough information.
- Canonical aggregation when several marks produce the same molecule.

## Validity-closed meaning

The visible process contains only:

- the formal null source state; or
- complete molecular graphs satisfying local valence and RDKit sanitization.

For the default de novo process, connectedness is an additional state-space
condition. Scaffold preservation and other hard requirements are further
condition-specific restrictions. A proposal that would leave this state space
has zero generator support; it is not committed and repaired afterward.

## Deferred

- Concurrent rule firing and rule-algebra superposition.
- Open-graph or port-graph intermediate objects.
- Transaction buffers and atomic macro retirement.
- Learned or mined rewrite-rule discovery.
- Recursive rules, instruction caches, and compiler optimization.
- Physical mass-action interpretations of rates.

These are useful extensions only if the single-event validity-closed base model
shows a measured expressivity, mixing, or path-length limitation.

## Practical value being tested

The rewriting layer is not included only to rename a generic graph model. It
provides executable benefits:

- invalid and fragmenting actions have exactly zero rate through rule guards;
- local operator/operand structure factorizes an otherwise unstructured next-
  graph prediction problem;
- inverse rules support exact round trips and bidirectional editing;
- hard conditions alter the enabled state graph rather than repairing outputs;
- CTMC propensities define a normalized ancestral sampling procedure;
- verified macro rules can shorten event horizons without changing chemistry
  semantics.

The tiny learned gate demonstrates the first, fourth, and fifth benefits with
100% valid and connected committed rollouts. It does not yet demonstrate that
rewriting improves distribution quality or compute at realistic scale; those
require matched ablations.
