# Scoped scaffold-construction interface (development)

This isolated upstream module serves COMPOSE-Lipid decision 0031: a shared
chemistry-context-conditioned process for empty construction and completion of a
supplied core. It does not alter ICLR experiments, executor operators, old model
weights, training protocols, or default compiler behavior. The tracelet compiler
has an optional preserved-source extension. No commit or publication is
authorized by this work. No biology or historical BEAE data is used.

Current milestone: a source-only immutable context, exact preserved-core teacher
construction and replay tests on synthetic cases and a fixed 64-target lipid
development panel. Supplied atoms remain in persistent slots; target mapping is
teacher-only. Exact induced core bonds are preserved, with attachment hydrogen
bookkeeping delegated to existing valid primitives. Source/target Kekule-phase
disagreement fails explicitly. There is no general reachability guarantee.

The output is a teacher trace and an explicit target-slot bijection, **not a
qualified conditional generator**. Context flags and identity must later be
carried through examples, permutations, encoder input, candidate support/rates,
and sampling. A runtime rejection hook alone is not matched rate normalization.
Raw bond-insertion teachers require explicit learned-candidate qualification.
No full-corpus preprocessing or GPU run follows from this compiler milestone.

The empty-context branch delegates to the existing micro compiler for API tests;
it does not replace the lipid tracelet/N-first training recipe. The next matched
pilot must keep empty-path conventions equal across arms. Existing generic
source-to-target deletion/reconstruction remains untouched and is not advertised
as scaffold-preserving completion.

## Component verification (2026-09-08)

33 new tests pass; 89 focused compiler/progress/tracelet dependency tests pass.
The unchanged lipid 64-task panel replays 1,704 states in 21.51 CPU seconds,
with exact target/core preservation and no excluded-motif or vocabulary failures.
Packed endpoints and typed steps are saved in the lipid repository's
`audits/scaffold_construction_v1.json`, SHA256
`522d66219938ae49f6fc845556aea95377d23e5f801154d9fc509d473c980d8b`.
The lipid full suite passes 358 tests, including replay on its unchanged old pin.

Nine raw chord-insertion teachers are not exposed by that model configuration.
The compositional-cycle flag replaces ring-family semantics; it must not be
flipped while retaining incompatible old empty-path teachers. A consistent
empty/completion teacher and learned-candidate inventory remains to be qualified,
alongside context-aware rate prediction and sampling. No conditional-model or
generation capability is claimed by these compiler tests.

This is an unfinished interface increment, not a completed upstream milestone.
The upstream repository-wide suite has not been run; it remains required at the
completed-interface/release boundary. New module/tests pass scoped Ruff checks;
`git diff --check` passes. There was no commit, production-pin change or GPU job.

## Ring-compatible compiler increment (2026-09-08)

`compile_scaffold_to_target_tracelets` starts the existing block-aware compiler
from a connected ring-closed supplied source. Existing default empty paths are
unchanged; core atoms/bonds are never rebuilt. No compositional-cycle flags or
new operator vocabulary are needed. Partial cyclic blocks are an explicit input
failure, not silently filtered tasks or a generative ring-position restriction.

Both existing carbon-carrier and typed-payload modes complete every task in the
fixed lipid panel: 64/64 empty and 64/64 completion. The nine raw chord teachers
from the first version are replaced with existing pendant-ring attachments.
The shared development catalogs give finite untrained-model log-probability to
56/56 default and 20/20 typed ring-birth/restatement teachers across both modes.
The typed flag also changes acyclic bond timing; shorter paths do not establish
better learned quality. The first task-mixture pilot retains the default ordering.

Exact replay evidence and source/runtime identities are saved in the lipid repo:
`audits/scaffold_tracelet_compatibility_v1.json`, SHA256
`605ebb947ba33eedc2d6dc20da9347812ba62b0171addeb0fc0b4a4c90ac5985`.
This CPU check took 86.61 seconds and performed no optimization or generation.
Both saved completion versions also replay on the old pinned executor.

100 focused upstream tests pass; the lipid suite passes 364 tests. New module
and tests pass Ruff; the existing tracelet compiler retains two pre-existing
UP037/SIM102 findings, confirmed against HEAD. No full upstream suite or release
qualification is claimed while the complete interface remains unfinished.

Next: source-only per-node conditioning and identical protected candidate law,
normalization, hazard and sampling, including nested ring placements. Bind the
condition into permutations and cache identities. A supported teacher is not
evidence that those still-unimplemented conditional model/sampler invariants hold.
No ICLR configuration, model weights, production pin, or training job was changed.

## CPU-conditioned model increment (2026-09-08)

The optional `scaffold_conditioning=True` model now consumes source-only core
and permitted-attachment flags, propagated by `FactorizedMarkExample`, the
collator and batch split/join/device interfaces. Primitive masks and ring
placement/electronic choices are conditioned before normalization and hazard
evaluation. Sampling uses that same law; state caches bind the condition.
Default unconditioned initialization and empty-context outputs are preserved.

The new model tests pass for hierarchical and superposed rates. They include
independent executor-filter masks, neural gradients, protected-teacher zero
probability, sampler/scorer hazard agreement, context-cache isolation, nested
ring normalization, catalog-exact ring choices and slot permutation. There are
93 passing focused scaffold/compiler/decoder/property/model tests (6.29 s).
No new operator support is claimed: the existing model's ring placements still
use carbon carriers, while decoder-level tests separately cover typed carriers.

On the lipid repository's unchanged 64-target development panel, 203/203 fixed
teacher/sampler checks pass (39.94 s CPU). Source completion traces were reused,
and the reconstructed catalog matches the earlier saved catalog fingerprint.
Artifact: `compose_lipid/audits/scaffold_conditioned_law_v1.json`, SHA256
`63fb2121b9cf70565b4e7f64ec156f6420af09efc21f95a8457ffff1a239a5f8`.
This checks selected productive rows and one-step untrained samples at the same
time, not full-schedule support, timed generation quality, or training.

**Still unfinished:** bound CPU-prepared conditional support/certificates,
GPU batch execution, training stream and trained output quality. The optional
conditional model explicitly refuses GPU execution until that pipeline exists.
Do not reuse unconditioned ring certificates for protected contexts. The richer
chemical-context feature proposal is also not implemented by these two flags.

New files pass scoped Ruff; touched legacy model/trainer retain 17 lint findings
also present at HEAD. Two new loop-local closure warnings were repaired by
explicit default bindings after the numeric probe, with focused tests passing;
the artifact retains its actual original producer hashes. Full upstream-suite
qualification remains pending at the completed-interface/release boundary.
The lipid suite passes 368 tests against its unchanged production pin. No GPU
job, commit, push, ICLR configuration change or production pin change occurred.

## Prepared conditional batches and optional features (2026-09-08, continuation)

The later lipid report `docs/CPU_PREPARED_CONTEXT_2026-09-08.md` supersedes the
unfinished-status bullets above for this CPU increment only. `NodeContextProvider`
is a generic, optional current-state-only interface; its CPU features travel
through collation, splitting, joining and device movement into an optional node
projection. The lipid repository owns the descriptor recipe. No lipid rules or
reaction-family whitelist were added upstream. Default weights/support and a
zero-projection baseline remain identical. Providers can require the aromatic
bond view to fail closed on a raw-Kekule training/sampling mismatch.

`prepare_scaffold_mark_batch` stores conditional ring support, bound teacher
certificates, placement groups and semantic decoders. Forward reuses these with
no executor/enumeration/DP discovery. Preparation reuses the validation checks
without running a neural encoder. The prepared rows bind exact state, context,
teacher and model support/catalog. They serialize and stream losslessly. The
opt-in dataset carries record-aligned contexts without changing any RNG draw,
clock, progress, rate, importance weight or resume order, and rejects legacy
state-only ring caches for those records.

CPU evidence in `compose_lipid/audits/scaffold_prepared_cpu_v2.json`: 408/408
feature-off/on scoring comparisons pass in 61.09 s, including 26 exact backward
comparisons for the saved ring teachers and NULL row. Discovery is patched to
raise during prepared forward. Artifact SHA256:
`e62785498c9cdd8b7ba439a39b9ee44a864915087f49864ae8de051b4d98a429`.
The saved panel has pendant-ring and restatement teachers, not ring-system-grow
teachers; focused tests separately cover nested ring-growth probability and
gradient parity, unsupported teachers, and stale context/teacher/catalog errors.
Catalog and 408 prepared diagnostic batches are saved in the lipid artifact
directory; they are not a timed training stream.

Conditional GPU forward requires prepared rows, but no GPU numerical or training
qualification is claimed. A fixed timed stream, model/checkpoint/schema binding,
whole-suite release checks, and CPU/GPU numeric parity remain before a lipid
training launch. Existing ICLR configurations and all non-opt-in semantics remain
unchanged. The production lipid pin is unchanged; no commit, push or GPU launch
was performed by this increment.
