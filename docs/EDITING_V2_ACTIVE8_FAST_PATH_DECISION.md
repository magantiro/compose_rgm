# Editing V2 Active8 fast-path decision

Status: implemented locally, verified by focused tests, not yet benchmarked or
executed on the production corpus.

## Decision

The full-corpus Process-V2 Active8 pass performs exhaustive teacher-support
validation, not a complete canonical-successor quotient for every transition.
For each policy-eligible teacher it must:

1. construct the exact production model support masks;
2. count every legal Active8 mark in those masks;
3. decode and match the complete teacher ActionV4 identity;
4. execute the unique matching mark through the production rewrite system;
5. reproduce the stored persistent-slot successor exactly; and
6. establish that the canonical successor is productive and non-self.

It records the source and target state hashes, source and successor canonical
keys, raw legal-mark count, matching-mark count, and exact-successor mark count.
Whole-trace admission remains immutable: any unsupported action excludes the
entire trace.

The full canonical molecular-successor quotient, canonical-successor count,
alias multiplicity, and independent dictionary regrouping remain mandatory on
the prospectively frozen release-sentinel sample. They are also compiled where
required for the bounded T1 and P50 successor-level objectives.

## Rationale

Active8 answers whether every stored teacher is representable and executable
under the frozen eight-family process. Constructing and executing every other
legal successor for every corpus transition is not necessary to answer that
question and made corpus admission computationally disproportionate. The
bounded sentinel retains a direct check that the production quotient and its
alias aggregation agree with the slow reference construction.

This changes computation, not model support. It does not remove candidates,
change the executor, change canonical identity, weaken whole-trace admission,
or authorize training.

## Disabled families

`ring_system_grow` and `ring_system_delete` remain outside the Active8 pilot.
Any nonzero support mask for either family is a hard error. The active families
remain atom insertion, atom deletion, atom restating, bond reordering, bond
rerouting, cycle closing, cycle opening, and ring-system restating.

## Evidence and limitations

Measured locally on the implementation worktree:

- the batched teacher checker agrees with the slow full-quotient checker on one
  genuine transition from every one of the 22 registered capability cells;
- all eight declared operator families are represented in that comparison;
- reversed input order and multi-item batching preserve evidence exactly;
- nonzero disabled-family masks fail loudly;
- the combined Active8, Gate 0, and Process-V2 contract-chain suite passes
  160 tests;
- both frozen scientific process identities and the frozen V1 Gate 0 file stay
  unchanged.

These are focused implementation checks, not production-corpus evidence. A
production-shape benchmark at 40 persistent slots, hidden width 256, and the
frozen float32 runtime must pass before remote execution. The benchmark may
select batch size, CPU, memory, and worker geometry. It may not change support,
sentinel selection, scientific thresholds, or admission rules.

## Reuse boundary

A completed content-addressed Active8 result is reusable by Gate 0, T1, P50,
and later training for the exact frozen corpus and process definition. Active8
must be regenerated if any material input changes, including corpus bytes,
whole-trace overlays, legal fibers, executor semantics, ActionV4, persistent-slot
identity, canonicalization, the Active8 evidence algorithm, or its bound source
revision. Changes confined to model training weights, controllers, objectives,
or checkpoints do not require Active8 regeneration.
