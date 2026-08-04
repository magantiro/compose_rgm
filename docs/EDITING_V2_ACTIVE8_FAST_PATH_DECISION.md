# Editing V2 Active8 fast-path decision

Status: implemented locally and verified by focused tests and a production-shape
40-slot benchmark. No remote Active8 task has run yet.

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
- the final release-focused Active8, sentinel, publication, Modal-surface, and
  Gate 0 suite passes 122 tests;
- both frozen scientific process identities and the frozen V1 Gate 0 file stay
  unchanged.

These are focused implementation checks, not production-corpus evidence. The
40-slot benchmark at hidden width 256 and the frozen float32 runtime evaluated
64 genuine queries, eight from each Active8 family. The batch-8 fast path agreed
with the slow checker on every query, completed in 20.43 seconds (3.133 queries
per second), and used 443 MB peak resident memory. The slow path took 42.61
seconds. The machine-readable diagnostic is
`/private/tmp/process_v2_active8_teacher_support_benchmark.json`; it is a local
resource-sizing diagnostic, not an authority-bearing scientific artifact.

## Release execution architecture

The release launcher binds every plan to the exact clean 40-character Git
commit serialized into the Modal image. Its default invocation submits only the
prospectively pinned train-role source chunk: lane
`real_endpoint_multistep_path`, chunk 24, entries 49,152 through 51,200. The
current task identity is derived from the final plan rather than embedded in
source code because task identities intentionally include the execution-commit
binding. This chunk is uniquely heaviest among train tasks by admitted
transition count (14,889 transitions across 2,048 source rows). No validation,
controller-validation, or final-test aggregate was inspected to choose it.

After the smoke passes, the explicit full invocation uses at most five Modal
Volume-v1 writer containers. Each container owns 16 CPUs and evaluates at most
16 independent chunk tasks in spawned, single-threaded subprocesses. Children
never reload or commit the shared volume. The parent reopens and validates every
child receipt against the exact plan and run identities before issuing one
explicit commit. Individual task artifacts are content-addressed and restart
safe. The group is not described as transactionally atomic because the volume
may publish background commits; a failed group can therefore leave valid task
artifacts that a restart safely reuses.

Reduction no longer retains the corpus-wide transition inventory in memory. It
uses a temporary SQLite index to enforce global source-row uniqueness, derive
the exact sentinel selection, and retain occurrence payloads only for selected
pairs. Preparation and finalization independently repeat the same two-pass
derivation and compare the exact inventories. Both phases request 8 GiB of
ephemeral scratch disk. This changes execution resources, not corpus support,
selection semantics, admission, or any scientific threshold.

## Reuse boundary

A completed content-addressed Active8 result is reusable by Gate 0, T1, P50,
and later training for the exact frozen corpus and process definition. Active8
must be regenerated if any material input changes, including corpus bytes,
whole-trace overlays, legal fibers, executor semantics, ActionV4, persistent-slot
identity, canonicalization, the Active8 evidence algorithm, or its bound source
revision. Changes confined to model training weights, controllers, objectives,
or checkpoints do not require Active8 regeneration.
