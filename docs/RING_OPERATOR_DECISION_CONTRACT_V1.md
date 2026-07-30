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

## Exact implementation blocker map

Audit date: 2026-07-30. Symbol names below are the durable interface references; line numbers describe the
audited revision and are secondary.

### 1. Hybrid construction is intentionally impossible

`FactorizedTraceletRateModel.__init__` in
`src/compose_v4/model/factorized_tracelet_rate_model.py` stores the two capability flags and raises when
both are true (approximately lines 1714--1737). This is a valuable guard, not a defect to delete.

The implementation already has distinct capacity for the hybrid:

- `cycle_close_head` and `cycle_open_head` are created when cycle operations are enabled;
- `ring_system_grow_head` and the electronic decoder parameters are retained even when the macro is
  disabled;
- `_action_tables` assigns cycle close/open and ring growth to separate family slots.

The smallest safe change, after the enumerability proof below, is a new explicit hybrid authorization/profile
whose default is false. The existing two-boolean combination must continue to fail unless that third,
versioned profile is present. The trainer, production loader, checkpoint metadata, and identity hashes must
all carry the same profile.

This is not currently true:

- `scripts/train_tracelet_cnof_gate.py` can pass both old flags, but construction fails;
- `scripts/evaluate_tracelet_rollouts.py:load_factorized_rollout_checkpoint` reconstructs only the old
  booleans;
- `scripts/ring_core_identity.py` correctly rejects any RingCore-V1 checkpoint with the grow macro enabled.

RingCore-V1 identity must remain unchanged. A hybrid needs a separate identity gate.

### 2. The executor is ready, but the complete learned macro law is not enumerable

`default_rewrite_system` already registers `ring_system_grow`.
`lower_ring_system_grow` returns a deterministic primitive program containing restates, one-neighbour atom
insertions, bond reorders, and cycle-closing bond insertions. `_execute_micro` commits those instructions
through a connectedness- and validity-checking runtime. Consequently every action admitted by
`is_valid_ring_system_grow` already has:

- a deterministic executor;
- valid, connected lowering intermediates;
- an unconstrained primitive reachability certificate;
- a deterministic primitive-equivalent cost, `len(lower_ring_system_grow(source, action))`.

The missing object is the **complete probability law over macro actions**.

`production_successor_kernel.enumerate_factorized_marked_law` currently:

- rejects every macro-enabled model;
- omits `ring_system_grow` from `_TABLE_FAMILIES`;
- sees only template logits in the common action tables.

A macro probability is not its template probability. It also contains placement and electronic-decoder
probabilities. `sample_rewrite_mark_conditioned` samples those factors and returns a complete action.
`_teacher_action_score` can score a supplied complete `RingSystemGrow` action and log-sum-exp its matching
template/placement aliases. There is no production iterator that returns **every** distinct complete action.

`enumerate_ring_system_grows` in `rewrite/ring_system_fiber.py` cannot simply be substituted without proof:
it is a catalog/template action enumerator, whereas the current semantic electronic modes use a
state-dependent prefix-completion decoder. The new iterator must enumerate every valid semantic completion
or fail closed. For every state it must prove:

```text
sum over distinct complete macro actions of exp(aggregate action score - family log normalizer) = 1
```

within tolerance.

This bounded enumerability/latency probe comes **before** a hybrid flag, corpus rebuild, or training run. If
exact enumeration is intractable under the current semantic decoder, the scientifically correct outcomes
are to keep primitive RingCore or preregister a deliberately restricted finite accelerator. A
template-head-only approximation is not admissible.

### 3. Quotient aggregation is structurally reusable once the law is complete

`canonical_successor_result` already executes every scored mark, canonicalizes the result, and uses
`segmented_successor_logprobs` to aggregate all marks reaching one molecule. The independent
`reference_successor_batch` performs the same grouping with a Python dictionary.

Neither needs new chemistry semantics. They need the complete macro marks from the prior section and tests
covering:

- persistent-slot and molecular-symmetry aliases;
- multiple templates/placements encoding the same macro action;
- different complete macro actions reaching the same molecule;
- any macro/non-macro one-step successor collision;
- productive versus canonical-self mass.

The iterator must retain a stable internal provenance address even when it coarsens multiple decoder paths
into one executable action. Otherwise E5 alias/refinement diagnostics cannot distinguish a true mass
preserving quotient from accidental deduplication.

### 4. Differentiable successor training needs a structured macro address

`factorized_successor_training.TeacherSuccessorAlias` currently stores
`(family_name, table_name, dense_coordinate)`. Its gradient path indexes one dense tensor coordinate. That
works for primitive families but cannot express a macro's template, placement, and autoregressive
electronic probability.

The model already exposes most of the correct scoring semantics:

- `_teacher_action_score` scores a complete macro;
- `ring_teacher_semantic_certificate` precomputes exact teacher-only chemistry support;
- `training_support_cache` and `training_support_compiler` can store macro template support and teacher
  certificates.

The bridge still needs a tagged macro address or encoded complete action plus certificate, a differentiable
scoring branch, and packed complete successor fibers. Its acceptance identity is direct:

```text
differentiable teacher-successor probability
==
sum of the complete production marked-law probabilities for that canonical successor.
```

### 5. The current action codec cannot build a paired hybrid corpus

`RewriteActionCodecV2` explicitly rejects `ring_system_grow`, does not allowlist `RingSystemGrow`, and is
used by `packed_trace_store`. This must not be relaxed in place because doing so would silently change a
frozen schema.

Create a new codec/schema version for the hybrid profile. The payload can remain explicit and
type-directed: `RingSystemGrow` is a frozen dataclass composed of already typed atoms, bonds, insertions,
reorders, aromatic edges, and a topology class.

`tree_transport` already constructs macro traces, but that is not the paired scientific corpus. Every macro
unit must be tied to:

- the same source/target endpoint unit used in the primitive arm;
- its exact validity-closed primitive decomposition;
- one shared sampling identifier and endpoint coefficient;
- identical source, scaffold, series, and split provenance.

Extra macro rows cannot create extra molecule or endpoint exposure.

### 6. Checkpoint/evaluator identity currently conflates or rejects the hybrid

`checkpoint_evaluator` currently:

- hard-requires `enable_ring_grow_macro=False`;
- omits that flag from `CAPABILITY_FLAGS`;
- omits it from `CheckpointProvenance`;
- labels every cycle-enabled support signature `ringcore_v1_compositional_cycle_ops`;
- constructs only `factorized_ringcore_segmented_pushforward`.

The hybrid needs a separate evaluator profile and implementation identity. `SupportSignature` should record
the macro capability and an explicit hybrid ring configuration. The primitive/hybrid comparison is a
**support ablation**, because the macro adds one-step edges; it can never pass `law_only` equality.

Do not weaken the production evaluator's RingCore-V1 check. Route hybrid checkpoints through the new
profile and require exact registry capability agreement.

### 7. Atomic hard-constraint conditioning is not implemented

`RewriteSystem.apply` has a useful complete-action constraint hook:

```text
constraint(source, action, successor) -> bool
```

However, there are no production implementations for protected atom/bond mappings, pharmacophore
retention, atom-count corridors, structural alerts, or similarity corridors. Passing a constrained
`RewriteSystem` to `canonical_successor_result` is also incorrect today: a rejected mark is caught as an
evaluator inconsistency rather than removed as controlled-support pruning.

Keep the unconstrained base molecular kernel unchanged. Add a controller-side semantic transition filter:

1. enumerate the complete base marked law;
2. execute the complete action atomically;
3. evaluate the protected mapping and path-constraint state;
4. reject the whole action before aggregation if forbidden;
5. deterministically update constraint state;
6. aggregate surviving mass;
7. renormalize after pruning.

If two aliases reach the same molecule but produce different protected/path state, the controlled state is
not just the molecule. Group by:

```text
(canonical molecular successor, canonical updated constraint state)
```

or prove that the constraint transition is constant over that molecular-successor fiber. Merging first and
checking later would lose exactly the action semantics hard protection needs and could break
action-refinement invariance.

The macro remains one committed event. Its internal lowering states must be valid and connected, which the
executor already proves, but they are not intervention points. A constraint-conditioned “accelerator”
claim additionally requires a constraint-feasible primitive path. If the macro jumps over a forbidden
primitive intermediate, it expands constrained reachability and must be reported as such.

### 8. Cost accounting has the definition but no experiment ledger

The decision uses two deliberately different notions:

- committed-event cost: every macro costs one;
- primitive-equivalent chemistry cost: the macro costs
  `len(lower_ring_system_grow(source, action))`.

The second is deterministic, non-learned, and property independent. The ledger must apply the corresponding
registered lowering length to every other active macro/tracelet, not only ring growth. Declared primitive
families cost one.

No rollout or controller currently records this cost. `tracelet_compiler` has related
`micro_lowered_steps` bookkeeping, but its generic fallback counts an unrecognized action as one and does
not establish the ring-growth evaluation ledger.

Add one versioned action-cost function and record per-event plus cumulative:

- committed events;
- primitive-equivalent cost;
- oracle calls;
- forward passes;
- canonical successors scored;
- accepted edits;
- wall time and GPU-hours.

Tests must reconstruct every trajectory total by summing its action records.

### 9. Non-grow operator parity must be explicit

The two arms may differ only by `ring_system_grow`. The V2 corpus contract disables
`ring_system_delete` by default, while the current factorized model still constructs and scores that family
when candidates exist. Resolve that operator-basis decision before the paired comparison and use the same
setting in both arms. Do not add a grow inverse macro merely to make the arm cosmetically symmetric;
dynamic reversal must be tested through the declared primitive basis.

## Smallest scientifically valid implementation order

1. **Bounded macro enumerability probe, macro-only profile.** Implement complete scored action enumeration
   without touching the hybrid guard. Measure normalization, execution, aliases, candidate counts, and
   latency. Stop if it is not exact or bounded.
2. **Macro-only quotient proof.** Feed those marks to segmented aggregation and the independent dictionary
   oracle. Prove slot, symmetry, multi-template, and productive-mass identities.
3. **Versioned hybrid identity and codec.** Add an explicit opt-in, distinct checkpoint/evaluator profile,
   and new codec. Preserve every V1 guard and old schema.
4. **Complete hybrid kernel.** Enable the proven iterator in the hybrid marked law. Verify both primitive
   directions and macro mass, normalization, executor agreement, and the declared support-signature diff.
5. **Differentiable macro successor fibers.** Extend structured aliases and packed serialization; match
   production probability and gradients exactly; pass macro successor micro-overfit.
6. **Paired endpoint corpus.** Compile the same endpoint units into macro and primitive representations,
   with equal endpoint exposure and compute ceilings.
7. **Constraint and cost layers.** Implement semantic pre-aggregation pruning, updated constraint state,
   and the primitive-equivalent ledger before dynamic/Pareto studies.
8. **Only short gates.** Run micro-overfit, then 50-, 500-, and 2,000-step paired pilots. Stop if the macro
   only wins on mechanically favorable event count or harms opening, reversal, Pareto, or constraint
   sentinels.
9. **Validation decision, then sealed test once.** Select on frozen validation rules, freeze the regime, and
   only then use the final test.

This order is faster than building the hybrid end to end because the cheapest and most fundamental failure
mode—an intractable or incomplete macro successor law—is tested first.
