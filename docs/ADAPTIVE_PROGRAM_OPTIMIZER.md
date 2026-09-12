# Adaptive coordinated-program optimizer

## Implementation contract

Problem: turn a productive but narrow coordinated-program neighborhood into
variable executable transformations that adapt to measured endpoint outcomes.
Output: a completed molecule, exact program/attachment/trace, proposal ancestry,
and a locked endpoint record. This is an optimizer over the frozen executable
process, not a new primitive kernel or an exact Doob sampling guarantee.

Current scope is implementation and zero-oracle local/cache-only development.
No new learned docking predictor or future-value model is qualified. The proposed
108-call static/adaptive campaign is not launched under the older 66-call cap.

Keep all production graph, charge, valence, persistent-slot and 40-atom guards.
Retain the broad WHERE/WHAT/HOW channel; program dispatch precedes its connected
region choice. Development allocation is 70% coordinated mutation, 20% branch
recombination and 10% broad. These are fixed starting settings, not fitted optima.
Use configurable 32-primitive/eight-block ceilings for the new development
configuration, not the former diagnostic's 24/five ceiling. Complete a chosen
transformation and stop; unused budget does not require additional edits.

## Required behavior

1. Context-match attachments, preserving unaffected assignments when mutating one
   or two sites. Parameter changes must rebind created handles and retain dependent
   fusion/refinement operations. Revalidate the full affected construction.
2. Allow supported created-atom type changes and one-neighbor chain extension or
   contraction. Reject transformations that invalidate dependent references. This
   variable program space must not become a new finite whole-ring template catalog.
3. Replace dependency-closed branches with context-compatible donor branches;
   preserve the other branches and validate the entire joint result. Do not infer
   task independence from structural independence.
4. Keep exact source/program/attachment/endpoint records in the archive. Use only
   identified measured outcomes for rank-based adaptation. Balance endpoint mass
   across multiple representations and retain exploration. No surrogate veto.
5. Bound attempts, candidate count, binding work and wall time; report failures,
   duplicates, cache hits, completion, eligibility and new endpoints separately.
6. Snapshot configuration, RNG state, archive, pending locked batch, counters and
   protocol identity. Resume must preserve the next proposal stream. Score updates
   must refer to the locked batch and correct oracle domain.

## Checks and evidence

Use focused executor, mutation, branch-dependency, dispatch, accounting and resume
tests. Preserve a static control with the same operators and initial information;
only measured-score adaptation differs. Local replay of already paid labels is
an integration check, not prospective improvement or transfer evidence.

The current 54-representation/27-endpoint bank has only one inspected source
group. Keep the known-winner information explicit and do not split its derived
programs across claimed training/test groups. Source-disjoint task validation and
any richer neural decoder remain later work. Record the runnable development
command and measured outcome here after implementation.

## Implemented interface

- `control/program_mutation.py`: typed attachment changes, created-atom class
  mutation, one-neighbor segment extension/contraction, dependency-closed branch
  extraction and replacement. Created references are renamed through dependent
  operations, then the full graph program is replayed. No endpoint template is
  inserted and no multineighbor birth is introduced.
- `control/adaptive_program_optimizer.py`: measured program archive, endpoint-
  balanced rank allocation, bounded duplicate-exhaustion adjustment, branch
  recombination, candidate locking, score receipt ingestion and exact snapshots.
  Static control freezes initial score preferences; its newly admitted programs
  inherit their parent's preference. Adaptive control uses actual measured scores.
  Both have the same mutation support, exploration and duplicate handling.
- `control/molecular_task_search.py`: channel selection before WHERE and complete
  broad reference-option execution with exact primitive witnesses. It does not
  replace the old population runtime or independently implement molecular
  transition probabilities.
- `tools/adaptive_program_search.py`: local `prepare → locked outcomes → resume`
  workflow. It verifies implementation, software, replay and oracle identities
  before resuming. Production use must inject the qualified broad hierarchy and
  bind the same runtime identities in its run manifest; this local tool does not
  fabricate a learned law when the remote assets are absent.

The optimizer intentionally has no surrogate veto, no fabricated proposal/reference
likelihood ratio, and no future-value label derived from winner reconstruction.
Missing or failed oracle evaluations do not become low-score observations.

## First local result, 2026-09-12

Authoritative artifact: `diagnostics/adaptive_program_optimizer/attempt_1/result.json`.
All 54 paid representations replayed, binding 27 distinct observations. The first
batch used 75 attempts: 16 new eligible endpoints, 18 duplicate returns, 13
ineligible endpoints and 28 explicit failures. Ten failures were unavailable
broad-reference draws in the declared local mode. Program mutation contributed
13 locked candidates and branch recombination three. Ten locked candidates
changed two sites; six changed one. Nothing was docked or assigned a new score.

Proposal work took 4.23 seconds (773 executor calls). Archive preparation took
4.95 seconds (756 executor calls); the whole local invocation took 9.39 seconds.
One arm64 CPU worker, approximately 474 MB process peak RSS, pinned RDKit
2024.03.5. This is neither a matched speed comparison with the remote broad
controller nor evidence that adaptive preferences improve docking.

Decision: the fixed bank is no longer the only reachable program neighborhood.
Use the locked candidates to test measured quality next, rather than growing an
unchanged particle population. Do not promote the adaptive policy until actual
outcomes outperform the same-support static control.

Run in a pinned chemistry environment, choosing a fresh output directory:

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 python tools/adaptive_program_search.py \
  --local-program-development \
  --output diagnostics/adaptive_program_optimizer/new_batch
```

The recorded local invocation prepended the existing RDKit overlay
`/private/tmp/compose-t4-chemistry.hizM8Y` to `PYTHONPATH` and used `.venv/bin/python`.
That machine-specific overlay is not a portable installation requirement; install
the pinned RDKit version in a compatible environment on another machine.

To consume actual results and prepare the next batch:

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 python tools/adaptive_program_search.py \
  --local-program-development \
  --snapshot diagnostics/adaptive_program_optimizer/new_batch/snapshot.json \
  --outcomes path/to/identified_outcomes.json \
  --output diagnostics/adaptive_program_optimizer/following_batch
```

The outcome file contains `batch_id` and exactly one record per candidate in
`outcomes`: `candidate_id`, `receipt_id`, `oracle_protocol`, and finite `score`.
Use `score: null` plus `failure: oracle_failed` or
`failure: cache_miss_not_evaluated` when appropriate. The receipt must identify a
real observation or explicit non-evaluation, never a fabricated numerical target.
The local entry point never invokes docking. `--static` applies only at initial
construction; resume uses its frozen configuration.

Focused checks: 25 tests passed across program mutation, optimizer feedback/resume,
edit-program execution, dependency graphs and the existing option runtime. Ruff
format/check and whitespace checks passed. The repository-wide suite was not
rerun for this local development unit; remote deployment and the full controller
milestone are not declared complete. Existing paid best remains -13.3, versus the
supplied -13.6 winner. The suggested 108-call campaign is not launched.
