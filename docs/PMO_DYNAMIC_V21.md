# Matched PMO Dynamic-v0 versus Dynamic-v2.1 four-task development

## Scientific question and claim boundary

This bounded development diagnostic asks whether Dynamic-v2.1's structured
post-filter competition channel improves over the unchanged pure Dynamic-v0
shallow proposal path. Both arms start without task-specific complete routes or
a task- or panel-informed program curriculum. The primary outputs are
arm-specific complete exact-replay programs, charged native higher-is-better
PMO rewards and matched best/top-ten trajectories.

The run covers exactly `celecoxib_rediscovery`, `perindopril_mpo`, `gsk3b`
and `isomers_c7h8n2o2`. Each task uses the predeclared search seed 20260914,
at most 1,000 charged unique-molecule calls per arm. There are eight task-arm
units, at most four concurrent single-CPU workers and 8,000 calls in total.
There is no automatic retry, GPU use, fifth task or T4 change.
Units are submitted task-major so each v0/v2.1 pair shares a worker wave.

The tasks and prior PMO molecules are answer-known development material in the
project history. The new runtime does not receive prior task outcomes, public
panel candidates, target routes, endpoints or target-to-program rules. A
positive result is therefore a four-task development diagnostic, not held-out
PMO evidence, a full-suite benchmark or a general superiority claim.

## Matched controllers and PMO adapter

The `dynamic_v0` arm uses the unchanged `DynamicProgramOptimizer` and
`initial_dynamic_program_batch` shallow path. The `dynamic_v21` arm uses the
unchanged `DynamicV21ProgramOptimizer`, preserving that v0 shallow behavior and
adding the generic Dynamic-v1 structured proposer, independent shallow,
structured and arbitration streams, exact execution before filtering,
cross-channel deduplication and post-filter pooled arbitration. Each arm has its
own initially empty measured-route archive.

PMO changes only the task boundary:

- all 16 structures in the existing objective-blind initialization lock are
  scored and charged separately for every task and arm;
- both arms use identical initialization order, task seed, configuration,
  native adapter and ledger semantics;
- each arm's initial complete-route archive and source library contain zero rows;
- endpoint eligibility is only RDKit molecule validity after exact executor
  support, without T4 similarity, QED or synthetic-accessibility gates;
- the native `PyTDC.Oracle(name=task)` returns a reward in [0, 1], and larger
  values are better;
- each task-arm unit records per-query best and top-ten curves, the 1,000-call
  development AUC and the flat-tail 10,000-query AUC. Dynamic-v2.1 additionally
  records channel proposals, rejections, duplicates and allocations.

The declared generated-object support remains the existing complete molecular
graph representation with at most 40 active atoms, 32 primitive edits and
eight blocks. Stereochemistry and formal-charge changes remain outside the
editing support. Training density and task oracle coverage do not broaden this
support declaration.

## Oracle and comparator isolation

Scoring uses the already qualified IVG-core environment: PyTDC 1.1.15, RDKit
2023.09.6 and the exact dependency versions and PyTDC source hashes recorded in
the contract. GSK3B additionally requires the previously downloaded current
PyTDC model with SHA-256
`d3a20701b80e5179c88c3ad4dc3483dd7ab35c50dc055c6773a7f5b63e89b6d5`.
The preflight verifies this environment and asset, then constructs all four
native oracle adapters without evaluating a molecule or observing a score.

Matched Dynamic-v2.1-minus-v0 arithmetic is computed only after both runtime
arms end. The existing parity-scored COMPOSE development results and IVG values are
copied into `configs/pmo_dynamic_v21_offline_comparators_v1.json`. That file is
not a runtime input. Only the post-run aggregate reads it, after every
controller decision has ended. Arithmetic against IVG uses the same flat-tail
10,000-query AUC convention. The 1,000-call development AUC is also retained
and is not silently described as a 10,000-query experiment.

## Durable accounting and failure policy

Every task-arm unit receives a separate durable ledger manifest. Each query is reserved before
evaluation, then receives a result receipt. Initialization and generated
candidates share the same 1,000-call ceiling. Repeated canonical endpoints are
cache hits rather than new charged calls. A failed or unresolved reservation
remains charged and blocks that task. It is never retried or assigned a
fabricated score.

Each completed round retains the complete proposal pool, attempt statuses,
allocation, query lock and resumable controller snapshot. The task result
records candidate shortfalls, proposal failure counts, oracle time, proposal
time, wall time, CPU time, peak resident memory and software versions. Unit and
progress receipts include arm, task, seed, configuration, contract and code
identity. The read-only `status` action derives actual charged counts and
current best/top-ten curves directly from query-indexed receipts and reports
the latest resumable snapshot without advancing RNG state or calling an oracle.
A task-arm unit may stop below 1,000 calls because proposal yield is exhausted. Such a
shortfall is a result, not permission to change the proposer or add candidates.

## Zero-oracle preparation and launch

Create the self-hashed contract and post-run-only comparator snapshot:

```bash
.venv/bin/python tools/pmo_dynamic_v21.py prepare
```

Run the structural and pinned-environment preflight with zero oracle calls:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
UV_CACHE_DIR=.uv-cache uv run --isolated --no-project --python 3.11 \
  --with-editable . \
  --with 'rdkit==2023.9.6' \
  --with 'PyTDC==1.1.15' \
  --with 'numpy==1.26.4' \
  --with 'scikit-learn==1.2.2' \
  --with 'pandas==2.1.4' \
  --with 'scipy==1.15.0' \
  --with 'seaborn==0.13.2' \
  --with 'requests==2.32.4' \
  --with 'setuptools==75.6.0' \
  --with fuzzywuzzy \
  --with huggingface-hub \
  python tools/pmo_dynamic_v21.py preflight
```

Before scoring, commit the contract, implementation and sealed preflight, then
verify that the worktree is clean. The launch command intentionally fails if
those conditions, the file hashes, oracle environment, asset identity or
zero-oracle preflight do not match:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
UV_CACHE_DIR=.uv-cache uv run --isolated --no-project --python 3.11 \
  --with-editable . \
  --with 'rdkit==2023.9.6' \
  --with 'PyTDC==1.1.15' \
  --with 'numpy==1.26.4' \
  --with 'scikit-learn==1.2.2' \
  --with 'pandas==2.1.4' \
  --with 'scipy==1.15.0' \
  --with 'seaborn==0.13.2' \
  --with 'requests==2.32.4' \
  --with 'setuptools==75.6.0' \
  --with fuzzywuzzy \
  --with huggingface-hub \
  python tools/pmo_dynamic_v21.py launch --workers 4
```

The user authorized the bounded launch, but the prepared code still requires
the requested implementation review, intentional commit and clean-worktree
validation before it can start scoring.

## Acceptance criteria

1. The self-hashed contract fixes two matched arms, exactly four tasks, one
   search seed, 1,000 charged calls per task and arm, 8,000 calls total, four
   concurrent single-CPU workers and zero retries.
2. Both arms receive the identical task-independent charged initialization and
   an empty route library, but no comparator result, winner route or panel artifact.
3. The zero-oracle preflight validates deterministic pure-v0 and pooled-v2.1
   preparation, exact v0 shallow-prefix parity, PMO-only validity, maximize
   orientation, independent v2.1 streams, both empty-channel fallbacks and exact
   execution from both v2.1 channels.
4. Each task-arm retains exact query receipts, full proposal and arbitration locks,
   best and top-ten trajectories, both declared AUC values, candidate
   shortfalls, compute and failures.
5. The aggregate never exceeds 8,000 charged calls, reports matched
   v2.1-minus-v0 deltas and loads comparator values only after runtime search ends.
6. Focused tests, lint, formatting and diff checks are reported exactly as run.
   The unrelated repository-wide suite is not an iteration gate for this
   bounded development implementation.
