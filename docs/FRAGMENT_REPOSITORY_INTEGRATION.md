# Fragment-suite repository integration

Status: saved-table and isolated local-generation interfaces implemented;
whole-repository certification and public asset distribution remain incomplete
(2026-09-27). Nothing is authorized for publication by this task.

## Scope and scientific identity

COMPOSE learns executable molecular rewrites. The fragment suite tests constrained
construction with retained molecular structure, learned edit selection in
superstructure generation, and selection among saved executable completions in
motif extension, scaffold decoration, and linker design. Scaffold morphing reuses
linker outputs; it is not a fifth independent generation campaign. The comparison
laws and their sampling support remain those of the frozen experiment contracts.

This task organizes existing evidence and interfaces. It does not change the
model's representation, chemical support, task constraints, checkpoint, training
split, executor, selectors, evaluator, seeds, or experimental conclusions.

## First integration boundary

Use the normal repository layout: importable experiment logic under
`src/compose_v4/experiments/fragments/`, usage documentation and an evidence
manifest under `experiments/fragments/`, immutable small results in
`diagnostics/`, and focused tests under `tests/`. Do not introduce a separate
reviewer package. Keep historical script paths working.

The first boundary is portable verification and reduction of the completed
fragment evidence. It must run without private worktrees, credentials, network
access, model weights, or molecule generation. Keep the distinction between
recomputing tables from saved per-prompt metrics and rerunning generation explicit.

Generation integration follows only after the complete historical dependency
closure has been reconciled. Inspection found differing shared model and rewrite
files between the paper-producing worktrees and the primary checkout, as well as
historical launchers bound to absolute interpreter paths. Do not overwrite these
core files or re-pin an authorization to make an integrated launcher pass.

## Acceptance criteria

1. Record the exact source identities, task populations, seeds, available inputs,
   and missing external assets in one task-specific manifest.
2. Preserve imported result files byte-for-byte and verify their SHA-256 hashes.
3. Recompute benchmark means and between-seed sample standard deviations,
   superstructure learned/uniform comparisons, saved-panel selection comparisons,
   and attempted-offer prefix tables from saved per-prompt rows. Validate counts,
   duplicate identities, finite values, and agreement with existing reductions.
4. Provide one documented module entry point that checks inputs and writes
   complete reports to a new output directory. Refuse overwrites and publish
   output atomically. Record input and implementation hashes, revision, software,
   aggregation rules, and scientific limitations.
5. Test in an isolated source tree without `.worktrees` and without chemistry or
   ML dependencies. Run focused tests and lint, then the repository-wide checks
   once at the milestone boundary. Report pre-existing or environment failures.
6. Preserve every pre-existing user file and every frozen contract. Do not prune
   worktrees, delete diagnostics, alter manuscripts, launch jobs, or push commits.

## Evidence boundaries

- Saved motif results give 99.9667% validity, while the submitted Table 1 prints
  100%. Preserve and report the measured value; do not alter data to match that
  rounding/error in the manuscript.
- Sample SD across three generation seeds is distinct from a prompt-bootstrap
  interval and from a population SD stored in some historical summaries.
- Linker selection changes reference weighting and a fourfold novelty preference
  together. Prefix analyses are retrospective, not runtime speedups.
- Metric-row reduction alone does not verify molecule generation, the external
  evaluator, raw trajectories, or full end-to-end reproducibility.

## Preservation policy

All worktrees and untracked data stay in place. Git protects tracked history, not
untracked catalogs, checkpoints, or run outputs. A future move requires an
inventory, a verified durable copy, and a dependency/pin audit first. No remote
backup or publication is implied by local integration.

## Verification and handoff

Measured on the local candidate:

- The focused fragment tests passed: **33 passed**, including the existing
  selection tests and the new portable reduction tests. The source-export test
  runs without `.git`, `.worktrees`, or Python site packages and checks identical
  table bytes across two runs.
- Touched-code Ruff checks passed. `git diff --check` passed.
- The standard-library `verify` command passed all seven input hashes and the
  complete metric-row populations. The optional NumPy 1.26.4 bootstrap reproduced
  every saved interval within the declared absolute tolerance.
- The generated JSON, Markdown, and provenance were inspected at
  `diagnostics/fragment_repository_integration_v1/`. Only row reduction and
  bootstrap reproduction were verified, not the molecular evaluator or generator.
- One full-suite invocation in the pinned Python 3.11 chemistry environment
  stopped during collection in `tests/test_process_v2_structural_protocol.py:270`:
  `StructuralDecisionIndex.__protocol_attrs__` does not exist in that environment.
  That pre-existing test and its implementation were not changed. The suite did
  **not** pass; it was not repeatedly retried in other environments.
- Repository-wide `ruff check src/ --statistics` reported **513 existing
  findings**. These are outside the new package. No bulk formatter/fixer was run.
- The non-strict local preflight reported the new uncommitted package as drift;
  its exit status is not a clean-launch authorization. The cloud prelaunch gate
  was not run because this task authorizes no cloud interaction or campaign.

The next integration step is the exact generator/asset dependency closure, not
another benchmark campaign. Preserve and hash the training-derived catalogs,
priors, checkpoint, development-selection records, raw attempts, and historical
code before introducing a portable generation interface. Then require focused
parity against the frozen implementations. Until then, do not claim that a fresh
checkout can reproduce the whole fragment experiment end to end.

## Second integration boundary: local generation runtime

Authorized by the user's subsequent "do it": consolidate the fragment generators
and their required assets behind the same task entry point. Keep shared library
code and every historical worktree unchanged. Preserve historical Python source
as hash-checked, task-specific source archives, executed in an isolated subprocess
so incompatible core revisions cannot mix. These are source snapshots, not new
models or replacements for the original launch authorizations.

Before publishing a local result, require the exact checkpoint, training-derived
catalogs/priors, prompt file, source archive, CPU float32 configuration, and pinned
Python/chemistry environment. Asset installation must copy, never move, and verify
both source and destination hashes. Missing assets must fail with their identities;
no automatic network download, replacement checkpoint, or catalog regeneration.

Acceptance for this boundary:

1. Provide portable asset verification and local generation commands for motif,
   decoration, linker, and learned/uniform superstructure. Separate source
   preservation, orchestration, task dispatch, and artifact publication.
2. Preserve empty output slots, the eight-offer selectors, novelty history, RNG
   derivation, exact primitive replay, and structural guards. Do not silently use
   a newer constructor, source-coupled decoration policy, or canonical-uniform law.
3. Verify the first saved BARICITINIB attempt at the first declared seed for each
   task, chosen by identity rather than quality. Also check the first two linker
   attempts to exercise per-cell novelty history. These bounded local parity
   checks are implementation verification, not new benchmark campaigns.
4. Record exact source/asset hashes, comparison scope, and any mismatch. Verify
   that the integrated path runs without the historical worktrees on its import
   path. Do not claim whole-panel regeneration from a bounded parity check.
5. Test wrong hashes, missing assets, path traversal, environment drift,
   output collisions, and deterministic source capture. Keep the existing table
   reduction independent of optional chemistry dependencies.

The original multi-gigabyte run directories remain in place. Disk availability
was 12 GiB at the start of this boundary, so do not duplicate all raw runs or
delete old copies to make room. Preservation on one disk is not an off-machine
backup. No cloud jobs, oracle calls, training, manuscript edits, or pushes.

### Second-boundary verification

- **62 focused tests passed**, including five source-export tests with no Git
  metadata or private worktree on the import path. The preserved first attempts
  match exactly for motif, decoration, linker (two attempts), and learned/uniform
  superstructure. This is six attempted slots, not a full benchmark regeneration.
- The normal generation/evaluation command also completed on the first saved
  motif slot. Its single-slot metric is only a pipeline smoke check, not a new
  quality estimate. The result remains under `runs/fragments/evaluator_smoke_v1`.
- The original 503 source-file identities across the five dependency closures
  were rechecked against their worktrees and remained unchanged. The twelve
  installed local assets passed their registered SHA-256 checks. No original
  file was moved or deleted; the copies consume about 76 MB before filesystem
  overhead. Full multi-gigabyte proposal panels were not duplicated.
- An initial motif archive omitted its caller-supplied prior module. The import
  check failed; adding that unchanged dependency repaired the package. A source
  export inside another Git checkout also exposed erroneous enclosing-repository
  attribution. The adapter now requires its own Git marker before recording a
  revision. Negative diagnostics and regression tests are retained.
- Touched-code Ruff and formatting checks passed. The saved-table verifier still
  reproduces all intervals. The full suite was attempted once at this boundary
  and again stopped during collection at the pre-existing
  `StructuralDecisionIndex.__protocol_attrs__` failure; no full-suite pass is
  claimed. Repository-wide Ruff still reports 513 existing findings.
- Exact test receipts, parity results, implementation hashes, and scope are at
  `diagnostics/fragment_runtime_integration_v1/`. The final local preflight is
  `python3 tools/preflight.py --strict`: passed at `e8a4f3e76036`, with zero
  mounted `src`/`configs` drift. This is a local check, not launch authorization;
  no cloud gate was invoked.

Remaining work is not hidden: checkpoints/catalogs still need an authorized
distribution route, full raw-panel selector replay is not made portable here,
and end-to-end whole-panel regeneration and other platforms are unverified.
The original source/runtime separation is intentional; merging historical model
and executor revisions into shared core would be a different, higher-risk task.
