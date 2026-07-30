# COMPOSE ring-operator decision contract

## Current decision

`primitive_ringcore` remains the primary model until a paired comparison proves that a hybrid is better.
This is a fail-closed default, not a conclusion that a ring-growth macro cannot help.

The only admissible hybrid in this decision is:

```text
cycle_insert (primitive close)
+ cycle_attach (primitive open)
+ ring_system_grow (optional accelerator)
```

A macro-only regime is not scientifically adequate. `ring_system_grow` cannot open a ring, support
topology reversal after an objective switch, or replace the compositional support story.

The machine-readable authority is
`configs/ring_operator_decision_v1.json`; validation and launch blockers live in
`src/compose_v4/experiments/ring_operator_decision.py`.

## Why the comparison is currently blocked

The current production model rejects simultaneous cycle primitives and `ring_system_grow`, and the current
production successor evaluator fails closed when the macro is enabled. Macro alias aggregation, packed
successor training, atomic protected-mask filtering, and a paired corpus also remain to be implemented.

Those are engineering facts. They prevent an accidental hybrid launch but do not prejudge the result.

## What “support preserving” can and cannot mean

Adding a macro necessarily adds one-step edges to the embedded jump chain. The two regimes therefore do
not have identical one-step support or identical event-budget geometry.

An **optional accelerator** claim is allowed only if every macro successor has a certified path under the
primitive operator system, with valid connected committed states and the declared state-space bounds. On a
hard-constrained task, the corresponding primitive path must also satisfy the registered constraints.

If this audit fails, the hybrid is an **extended-support model**, not an accelerator. It would require an
explicit scientific reframing before it could replace primitive RingCore as the primary model.

## Fair cost accounting

Both cost views are mandatory:

1. **Committed-event budget.** One macro is one event. This measures controller horizon and intervention
   opportunities, but mechanically favors a macro.
2. **Primitive-equivalent chemistry cost.** Every action costs the number of steps in its certified
   production-primitive lowering. Primitive cycle close/open each cost one. This prevents a many-edit macro
   from being called a one-unit chemical change.

Oracle calls, model forward passes, canonical successors scored, accepted edits, wall time, and GPU-hours
are reported separately. A committed-event reduction alone cannot select the hybrid.

## Editing tasks that govern the decision

The comparison is not a ring-reconstruction benchmark in isolation. It covers:

- source-conditioned analogue transport;
- coupled atom-count and ring adaptation;
- cycle creation and opening;
- ring-size and nontrivial fused/spiro/bridged changes where reachable;
- topology simplification;
- ring addition followed by reversal after a preference switch;
- Pareto archive coverage and a shared-prefix Pareto fan;
- protected scaffold, pharmacophore, atom/bond mapping, charge, alert, and optional similarity constraints.

This matters because a shortcut can improve endpoint recovery while reducing intervention points, harming
dynamic correction, Pareto branching, or constraint-feasible routes.

## Successor-level and constraint semantics

All slot, symmetry, template, and operand aliases leading to one canonical molecule are summed before
training metrics or control. If a macro and another one-step mark share a successor, they belong to the same
successor fiber. Mark-level power, top-k, or nucleus decisions are not valid evidence.

Hard constraints apply to the complete action before successor aggregation and controller renormalization.
If any touched protected atom or bond violates the mask, or the committed successor violates a path
constraint, the entire macro is excluded. There is no partial macro execution or post-hoc repair.

## Selection rule

The hybrid is eligible only when:

- the implementation and quotient tests pass;
- every macro endpoint has the required primitive reachability certificate;
- primitive closing and opening remain non-inferior at successor level;
- held-out topology and scaffold generalization passes;
- dynamic reversal, Pareto exploration, and constrained paths do not materially regress;
- at least one preregistered non-tautological primary metric improves under paired uncertainty analysis.

The non-tautological primary metrics use primitive-equivalent cost or oracle calls: topology success,
hypervolume AUC, dynamic adaptation regret, and constraint-feasible endpoint yield.

Until the numeric margins, task artifacts, and bounded implementation gates are frozen, paired training and
sealed-test evaluation remain unauthorized.
