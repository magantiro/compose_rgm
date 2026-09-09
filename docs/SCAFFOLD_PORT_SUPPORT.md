# Supplied-site hydrogen reserve (development, 2026-09-09)

Problem: protected scaffold ports currently preserve atom identity and induced
bonds but can consume all source hydrogens. Some supplied reaction programs
require a remaining NH or CH bond. A runtime-only predicate would disagree with
the learned candidate normalization and exit hazard.

Authorized output: an optional, generic per-protected-site minimum hydrogen
count, implemented consistently in executor constraints, primitive masks, ring
decoders, prepared support, sampling and teacher compilation. This is a shared
core interface for the lipid development lane, not a change to default COMPOSE
editing support, chemistry vocabulary, or ICLR experiments. The implementation
increment did not authorize a commit, remote publication, training or benchmark
launch; the subsequent local-commit authorization is recorded below.

The application supplies the reserve from declared reaction semantics, never
from a desired endpoint's observed substituent count. Unused port capacity is
allowed during construction. Exact terminal completion and synthetic feasibility
remain separate checks. This mechanism does not specify head/tail lengths,
ester placement, heteroatom locations or other learned regional chemistry.

Verification: compare bounded conditional candidate masks against independent
executor filtering; test forbidden teachers, legal teachers, score/sample hazard
agreement, prepared/live values and gradients, state/context cache separation,
serialization, slot permutations, and nested ring normalization. Empty reserves
must preserve legacy context identities and unconditioned behavior. Old prepared
batches without the new reserve tensor require re-preparation, not silent reuse.

Implementation is isolated in a new worktree from lipid commit 57cc913. Existing
pilot snapshot and the active dirty upstream worktree are untouched. Focused
tests establish only the covered interface invariants, not generation quality
or completed repository-wide release qualification.

## Bounded results

139 focused scaffold/context/compiler/sampler tests pass (7.48 s), including
independent conditional-mask executor oracles, forbidden primitive and ring
teachers, reserve permutations, prepared/live gradients, cache separation,
legacy identity and sampling hazards. Primitive teachers outside masks have
zero probability; absent dynamic ring teachers retain the existing explicit
unsupported-teacher error.

The lipid-side audit `artifacts/reaction_port_support_v3/summary.json` projects
all 4,352 existing programs and accepts 28,516 saved source/endpoint/checkpoint
states, without compiling paths or enumerating a full training schedule. Its
fixed 37 teacher/sampler rows pass exact prepared/live value and gradient parity
in a 21.58-second CPU check. The new lipid regional model has not been trained.

162 focused lipid tests pass on the new core. Legacy prepared artifacts remain
readable and checked on their original core, and are explicitly rejected by
field fingerprints on the new schema. New prepared batches must be produced.
New/scaffold files pass Ruff; the 25 findings in touched legacy files match the
old snapshot exactly by file/code/message. `git diff --check` passes.

No full-repository suite, release qualification, commit, push, production-pin
change, GPU job or generation-quality improvement is claimed. Clean source and
the actual new prepared stream remain prerequisites to a later launch.

## Subsequent local-commit authorization

The user approved committing only this isolated core fix, its tests and this
document, without pushing. The implementation bytes still match the recorded
CPU qualification. This creates a clean development dependency revision; it
does not authorize a release, merge into the active upstream branch, lipid pin
change, remote publication or training launch. The active upstream worktree
and the lipid repository changes remain untouched.
