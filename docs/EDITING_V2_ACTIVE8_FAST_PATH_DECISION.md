# Editing V2 Active8 fast-path decision

Status: the family-local teacher-admission path and restart-safe independent-task
launcher are implemented and verified. A corrective 80-task remote pilot was stopped
after eight hours because its fixed process groups produced unacceptable
straggler cost. The subsequent three-task operational canary established the
one-CPU, 4-GiB resource envelope but was censored before its two production-sized
tasks completed. A subsequent production-chunk canary completed 13,597 teacher
transitions in 61.55 seconds. A separate bounded equivalence panel matched the
exhaustive admission evidence exactly. The release map therefore uses independent,
content-addressed task invocations with a runtime concurrency bound of at most 100.
The bound was 80, matching the original five-container/sixteen-CPU geometry. It is
raised here because the training-prep fan-out is one single-threaded CPU per
container (`CPU_PER_LEAF = 1.0`, `OMP_NUM_THREADS=1`), so concurrency is a wall-clock
dial over a fixed ~1,025 core-hours rather than a change in total work or cost.
It remains a declared bound, asserted by
`test_launcher_is_the_exact_bounded_fast_path`, so raising it again is a deliberate
edit and not a silent constant change. The Volume-v1 concurrent-committer question is
separate and is validated by a bounded fan-out before any full run.

## Decision

The full-corpus Process-V2 Active8 pass performs exact family-local teacher
validation, not a complete canonical-successor quotient for every transition.
For each policy-eligible teacher it must:

1. decode the complete teacher ActionV4 identity and declared family;
2. establish that it is one legal coordinate of that exact family;
3. execute the teacher once through the production rewrite system;
4. reproduce the stored persistent-slot successor exactly; and
5. establish that the canonical successor is productive and non-self.

It records the source and target state hashes, source and successor canonical
keys, coordinate-legality flag, exact-replay flag, and productivity flag.
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
bounded sentinel still reconstructs the complete production quotient and checks
the shared teacher evidence against the family-local result. T1 separately
caches complete quotient geometry for its bounded successor-level panels.

This changes computation, not model support. It does not remove candidates,
change the executor, change canonical identity, weaken whole-trace admission,
or authorize training.

The accompanying schema migration retires corpus-wide Gate 0 raw-fiber-size
strata. Gate 0 remains a metadata-only structural gate over exact teacher
legality, replay, productivity, family and capability-cell coverage, and
provenance. Full-fiber counts remain measured by the bounded sentinel and T1;
they are not inferred from the family-local corpus pass.

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
  Gate 0 suite passes 569 tests after the remote-pilot correction;
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

A later mask-only prototype removed every neural action-score head from the
checker. On the matched 64-query panel it took 20.42 seconds, versus 20.43
seconds for the existing fast checker. Peak resident memory fell from about
443 MB to 337 MB, but wall time was unchanged within measurement resolution.
Profiling attributed 30.8 of 31.6 profiled seconds to exact chemistry support
construction, principally cycle-close, atom-restatement, and ring-restatement
enumeration. The prototype was therefore rejected and fully reverted. Keeping
duplicated legal-mask logic for no measured time improvement would add risk
without solving the operational bottleneck.

### First bounded remote-pilot result

The 80-task pilot launched from commit `37ab57d50b2ec1de3850e77e967260968ec67d94`
as detached Modal app `ap-0USx2MXUn0oTIlGWflV8wy`. Five map containers reached
the intended geometry of sixteen spawned subprocesses each. Multiple workers
then refused their first cached rows because Active8 called the cache decoder
without the complete historical V1 payload identity. The cache and rebind
artifacts correctly bind that superseded identity, and the rebind receipt already
validates and carries its complete self-consistent object. The omission was at
the Active8 read boundary, not in the corpus or rebind evidence.

The app stopped with zero authorized Active8 output. Child processes never
commit, and each failed parent therefore published no task group. The correction
passes the exact receipt-bound historical identity to both the map decoder and
the release-sentinel decoder. It does not relabel historical rows, weaken the
identity validator, change the corpus, or alter either scientific process
identity. Two regression tests require the complete receipt-bound identity at
both read boundaries.

### Corrective 80-task pilot result

The corrected pilot launched from commit `37ab57d50b2ec1de3850e77e967260968ec67d94`
as detached Modal app `ap-atikauWElSGEieY1U9NwYm`. It requested five
containers, each with 16 CPUs and 64 GiB, and assigned one fixed group of 16
tasks to each container. It was stopped after approximately eight hours. No
reduction, release sentinel, Gate 0, T1, P50, or training ran.

Exactly 34 of the fixed 80 pilot tasks published valid content-addressed task
directories. They cover 65,117 of 159,325 pilot source rows. All 34 belong to
the three earlier local/topology lanes. The pinned 2,048-row multistep canary,
which contains 14,889 admitted transitions, did not publish. The other 46
pilot tasks were all full 2,048-row chunks. Consequently this run does not
establish the runtime or success of the dominant multistep and synthetic-walk
lanes and cannot authorize the full map.

The approximately 50 USD charge is explained by the declared request: 80 CPU
cores and 320 GiB of memory remained allocated across five fixed groups while
stragglers ran. Completed children could not pull another task, and shorter
children left their reserved cores idle until the slowest child in the same
group finished. The preserved artifacts remain valid only under their exact
historical run identity. They are diagnostic evidence, not evidence for a
changed execution implementation.

### Independent-task canary result

The three-task canary launched from commit
`fbe2e0f6e9dc04641dc226657ba882b1b473cdc5` as detached Modal app
`ap-FFaH9uho8ZMe5yEjALaJj8`. It used one CPU and 4 GiB for each task. The
smallest task published successfully at 11:46 EDT with 54 classified
transitions. The two production-sized tasks remained CPU-active without an
exception and with stable resident memory when the app was stopped. At the
final observation, their process CPU times were at least 4,596.59 and 4,895.26
seconds and their resident-memory measurements were 3,547,508 and 3,733,420
kB. Neither task published a partial result, as required by atomic task
publication.

This is a censored operational result, not a scientific pass and not a runtime
estimate for the corpus. Waiting longer could not authorize a changed execution
commit, and the two unfinished artifacts could not be reused under that commit.
The canary nevertheless answered its resource question: the hard tasks ran for
more than one CPU-hour within the one-CPU, 4-GiB envelope without memory growth
or a runtime error. The old app was stopped after the measurement. No reducer,
release sentinel, Gate 0, T1, P50, or training ran.

### Family-local production-chunk canary result

The family-local canary launched from clean commit
`16961fe7d0a4e060625d5b3bf8e2640c29ae1750` as Modal app
`ap-xEF3dGzAWkx7hAOroMLo03`. On the same full 2,048-row multistep chunk used by
the earlier progress probe, it processed all 13,597 teacher transitions in
61.5527 seconds of wall time and 57.6 seconds of process CPU time. Peak resident
memory was 3,631.53 MB. Query preparation took 14.8307 seconds, and the teacher
histogram was 6,858 atom deletions and 6,739 atom insertions. The canary emitted
heartbeats and completed without publishing an Active8 task artifact, reducing,
running the sentinel, or authorizing any downstream stage.

The independent local 64-query equivalence panel compared the same
family-local admission evidence with the exhaustive quotient path. Evidence
digests matched and measured wall time improved from 44.9379 to 1.142617 seconds,
a 39.3289-fold speedup on that bounded panel. These results establish runtime
viability and bounded decision equivalence, not full-corpus correctness. The
machine-readable record is
`diagnostics/active8/process_v2_active8_family_local_canary_2026-08-04.json`.

## Release execution architecture

The release launcher binds every plan to the exact clean 40-character Git
commit serialized into the Modal image. A remote driver is spawned as the local
entry point's only remote call, so client disconnection cannot interrupt later
fan-out. The default invocation now submits three train-only operational
canaries, each as an independent Modal invocation with one CPU and 4 GiB:

1. the smallest train chunk, which measures fixed startup and publication cost;
2. `real_endpoint_multistep_path` chunk 24, entries 49,152 through 51,200; and
3. the first full 2,048-row `reversible_synthetic_walk` train chunk.

The latter two lanes contain 285 of the full plan's 328 tasks and therefore
measure the workloads that dominate the release pass. The current task
identities are derived from the final plan rather than embedded in source code
because they intentionally include the execution-commit binding. No validation,
controller-validation, or final-test aggregate was inspected to choose them.
Each result reports wall time, process CPU time, peak resident memory, source
rows, and classified teacher actions.

The three-task canary uses at most three Modal Volume-v1 writer containers.
Each invocation reopens and validates its receipt against the exact plan and
run identities before one explicit commit. There is no multi-task barrier, so a
slow canary cannot strand reserved CPUs assigned to another canary. The cohort
is selected before completed-task filtering, so a restart resumes only missing
canaries and cannot silently advance into the full map. Individual task
artifacts remain content-addressed and restart safe.

The explicit full invocation submits one plan task per Modal invocation, with
one CPU and 4 GiB per task, and accepts a runtime concurrency bound up to 80.
The bound changes scheduling only and is not part of scientific artifact
identity, so a pressure-related restart may lower concurrency while reusing
every completed task from the same exact execution commit. There is no fixed
process group and no group barrier. A completed worker releases its container,
and the platform may immediately schedule another missing task within the
bound.

The exact release workload contains 328 disjoint plan tasks and 2,327,413
admitted transitions. Tasks are submitted in decreasing admitted-transition
count so the longest tasks start first. The measured scheduler inventory joins
all 328 plan tasks exactly once, with no missing, duplicate, or out-of-order
entry, and its modeled scheduling utilization is 1.0. These are plan and
scheduler facts, not Active8 scientific results.

Task publication alone does not authorize downstream work. The map must finish,
the reducer must reconcile the exact corpus inventory, and the bounded release
sentinel must pass before Gate 0 may consume the result. If 80 concurrent
writers produce storage pressure or repeated transient failures, the same clean
commit is relaunched with a lower bound and the already published task
directories are reused.

Reduction no longer retains the corpus-wide transition inventory in memory. It
uses a temporary SQLite index to enforce global source-row uniqueness, derive
the exact sentinel selection, and retain occurrence payloads only for selected
pairs. Preparation and finalization independently repeat the same two-pass
derivation and compare the exact inventories. Both phases are budgeted for at
most 8 GiB of temporary storage within Modal's default 512 GiB per-container
disk quota, so the launcher does not request a larger `ephemeral_disk`. This
changes execution resources, not corpus support, selection semantics,
admission, or any scientific threshold.

## Reuse boundary

A completed content-addressed Active8 result is reusable by Gate 0, T1, P50,
and later training for the exact frozen corpus and process definition. Active8
must be regenerated if any material input changes, including corpus bytes,
whole-trace overlays, legal fibers, executor semantics, ActionV4, persistent-slot
identity, canonicalization, the Active8 evidence algorithm, or its bound source
revision. Changes confined to model training weights, controllers, objectives,
or checkpoints do not require Active8 regeneration.
