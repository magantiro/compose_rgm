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

## Parallel execution amendment, 2026-09-15

The user authorized immediate Modal parallelization of this zero-oracle work.
Run the extraction, address-free binding and exact-target gate as fifteen
independent source-group shards, each on one CPU, with no automatic retry. Each
shard publishes a durable result under its immutable call identity. Reduce only
after all expected source groups are present and compatible. This changes wall
time only. It does not change the source folds, representation, gate, support or
scientific budget.
