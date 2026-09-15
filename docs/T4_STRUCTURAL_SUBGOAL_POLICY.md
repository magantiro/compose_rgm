# T4 Structural-Subgoal Policy and Exact Realizer

## Problem and model output

The T4 teacher traces contain 5 to 32 primitive executor actions, but the
strict dependency audit reduces every route to one to four structural regions,
with a median of two. The grouped primitive decoder also shows a large
train-to-held-source loss in primitive-rule prediction while legal WHERE/HOW
coverage remains high.

The model output for this milestone is therefore a generated structural
subgoal:

\[
g \sim \pi_\theta(g\mid x,A),
\]

where \(g\) specifies changed source roles, the desired local target graph,
boundary attachments and dependencies. It contains no target identity, route
identity, absolute source slot or primitive teacher trace.

The conditional realizer receives a bound current state and a generated
subgoal:

\[
C(x,g) \rightarrow P=(a_1,\ldots,a_T),
\]

then the existing exact executor produces \(x'=T(x;P)\).

## Central claim under test

The current zero-oracle claim is narrow: structural-subgoal prediction plus
conditional exact realization provides better held-source proposal support
than unconditional primitive trajectory imitation. Docking improvement and
IVG comparison are not tested in this milestone.

## Experimental setting and baselines

Use the same 77 signed routes, 15 source groups and three predeclared
whole-source folds as the complete-program decoder. Split before extracting
any reusable patch vocabulary, frequencies, preprocessing state or policy
parameters.

Compare:

1. a source-balanced marginal structural-subgoal proposal;
2. a graph-conditioned structural-subgoal proposal;
3. the existing unconditional primitive decoder as the causal baseline.

The primitive decoder jobs continue unchanged and are not a dependency for
implementing the subgoal representation.

## Representation and support

A subgoal is an address-free local graph delta. Source atoms are represented
by structural roles and rebound against the current graph. Newly requested
atoms use relative output handles. The target patch records desired atom
attributes, local bonds and boundary attachments, but not the primitive
instructions used by the teacher.

Declared support remains:

- the eight existing legal rewrite families;
- 48 persistent slots and at most 40 active atoms;
- at most 32 realized primitives;
- one to four strict structural regions;
- exact executor legality and charge preservation at every primitive;
- completed-state evaluation only.

The realizer may interleave primitive work across subgoals when global capacity
or validity requires it. A macro count is not a license to weaken peak-size or
intermediate executor checks.

## Gates

Before fitting a proposal policy:

- all admitted teacher subgoals must serialize without primitive actions;
- source-role binding must be address-free;
- exact target reconstruction coverage and precision must be reported
  separately;
- conditional realization must report failures and abstentions rather than
  falling back to the teacher action sequence;
- all learned preprocessing and patch statistics must be fit on training
  sources only.

Then report held-source Recall@8/32/128 for exact subgoals, exact endpoints and
the frozen transformation-equivalence metric. Also report validity, uniqueness,
support coverage, candidate work, wall time and failures.

## Runtime direction after this gate

If the subgoal gate succeeds, the later controller will propose several
structural goals, compile and exact-execute them, keep a small diverse frontier,
and support bounded checkpoint repair or region reentry. Dynamic-v0 remains a
separate exploration and refinement lane in the shared archive.

No scored optimization or oracle call is authorized by this document.

## Live progress and ETA contract

Conditional realization can take materially longer than target construction.
Each remote source shard therefore emits and durably publishes a structured
heartbeat at least every 30 seconds while work is active. It records the
source, phase, current route, routes completed and total, elapsed time,
throughput, search expansions and attempts, failures or abstentions, and an
explicitly labelled estimate to the configured search limit. Completion and
failure receipts use the same task identity.

The estimate is operational telemetry only. It does not authorize early
stopping, alter the search limit, or turn incomplete work into a result.

## Conditional-realizer revision after attempt 1

The first conditional-realizer implementation converted each address-free
target into one fixed persistent-slot assignment. Its partial result had one
FA7 abstention, while a BRAF route had expanded thousands of states without
finishing. Offline zero-oracle diagnosis showed that teacher-equivalent
molecules could retain a nonzero slot-level distance and that correct actions
could be excluded from the fixed-slot target fiber. Attempt 1 remains immutable
and is a failed gate regardless of its remaining worker outcome.

The next revision carries compiler state as the exact molecular graph plus a
partial logical-output-role-to-slot mapping. Physical output slots use a single
canonical first-empty allocation. Logical mappings are quotiented by exact
automorphisms of the bound target patch, with bound source roles fixed and atom
and bond attributes preserved. A route succeeds only when both the canonical
molecular endpoint and its complete logical role obligations match. The
teacher primitive trace is never available to this search.

The gate reports two levels separately:

- complete-route realization coverage and exact endpoint precision over all 77
  routes;
- target coverage and precision for all 147 subgoal instances inside their
  coordinated complete-goal endpoints.

The second metric is deliberately not called isolated-subgoal execution.
Dependency regions may share retained boundary roles and jointly determine
their final atom degree, so executing one region alone can define a different
target. The aggregate also reports the 137 unique serialized subgoal IDs so
repeated instances do not masquerade as independent support.

Compiler cost is reported per route and as median, 95th percentile and maximum
wall time, expansions, action attempts and realized primitive count. The
search remains bounded to 12 retained children per expansion. Passing the
corpus gate establishes empirical coverage on the declared 77-route corpus,
not complete search support for every representable structural goal.

The role-aware bounded-search revision recovered 76 of 77 routes and 144 of
147 subgoal instances at precision 1.0 with zero teacher-action fallback. The
remaining 18-primitive route was fully inside the generic legal target-action
support but reached the unchanged 16,384-expansion limit at mismatch two.
This is preserved as a failed zero-oracle gate.

The next revision first derives the explicit bound graph delta and attempts a
deterministic legal schedule. Persistent output slots are allocated only
inside the compiler after address-free source binding. They are absent from
the serialized subgoal and runtime policy target. Every scheduled primitive
is generated from the current graph and requested target, then verified by
the unchanged exact executor. If this schedule reaches a local dead end, the
role-aware bounded search remains available within the same declared work
limit. Report the strategy used per route and include both paths in compiler
cost accounting. Do not increase the search limit or use the teacher trace.

## Parallel execution amendment, 2026-09-15

The user authorized immediate Modal parallelization of this zero-oracle work.
Run the extraction, address-free binding and exact-target gate as fifteen
independent source-group shards, each on one CPU, with no automatic retry. Each
shard publishes a durable result under its immutable call identity. Reduce only
after all expected source groups are present and compatible. This changes wall
time only. It does not change the source folds, representation, gate, support or
scientific budget.
