# PMO Dynamic-v2.1 bounded recovery

## Scope

The matched PMO launch-v2 run
`97bbbf9a3e3d8ac36965e66ec5c9341b3441622b93f446ee68e414d03cfd0901`
completed all four 1,000-call Dynamic-v0 units. Each Dynamic-v2.1 unit stopped
after 33 completed charged observations with no unresolved reservation. Round
0 is complete and round 1 is durably pending in every failed unit. The first
round-1 candidate was evaluated, but its score was not admitted because the
unchanged Dynamic-v2.1 optimizer rejected a second physical bootstrap pool.

This one-time recovery may resume only those four Dynamic-v2.1 units. It keeps
the 132 completed observations, original candidate order, task seeds, query
indices, oracle protocols, controller configuration and every launch-v1,
launch-v2, failure, pending-round and incomplete-result artifact. At most 967
new calls may be charged per task, 3,868 total. There is no automatic retry,
fifth task, GPU, task-specific route, controller tuning, comparator access at
runtime or T4 change.

## Generic continuity repair

The original pooled controller's mismatch check remains unchanged. A PMO-only
campaign-admission adapter distinguishes two identities:

- `generation_pool_id` records each physical cold-start pool exactly;
- `continuity_pool_id` remains the first pool identity already stored in the
  durable controller snapshot.

When the all-scored-parent recipe creates a later physical pool, the adapter
merges its score-blind proposal counters once, advances the shallow,
structured and arbitration streams to the pool's locked post-proposal states,
and presents the stable continuity identity to the core. The core still raises
if the adapter supplies inconsistent continuity. Endpoint order, scores and
oracle selection are not changed.

For the existing round-1 lock, query index 32 must be recovered from the ledger
cache and admitted before candidate index 1 can invoke the oracle. A durable
consumption receipt records this zero-call step. The remaining candidates keep
the original round-1 batch ID and order.

## Artifacts

- `diagnostics/pmo_dynamic_v21/recovery_failure_census_v1.json` hashes every
  pre-recovery run file and records the four contiguous 33-call failures.
- `configs/pmo_dynamic_v21_recovery_v1.json` is the self-hashed recovery
  contract.
- `diagnostics/pmo_dynamic_v21/recovery_preflight_v1.json` is the zero-oracle
  audit.
- `diagnostics/pmo_dynamic_v21/launch_v3.json` is created once at launch and
  binds the source run, census, contract and preflight.
- `diagnostics/pmo_dynamic_v21/result_v3.json` is the distinct recovered
  aggregate. The original incomplete `result.json` is never overwritten.

## Preparation and launch

Prepare and inspect the failure census, contract and zero-call audit:

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
  python tools/pmo_dynamic_v21_recovery.py prepare
```

After focused checks, inspect the diff and commit the compact source launch,
incomplete result, failure census, recovery contract, preflight, code, tests
and documentation. The preserved source run tree contains 8,828 files and is
approximately 4.3 GB, so it may remain outside Git. The launch guard permits
that one untracked tree only when its exact path set and every file hash match
the committed census; all tracked files must be clean and no other untracked
path is admitted. The explicitly authorized one-time recovery command is:

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
  python tools/pmo_dynamic_v21_recovery.py launch-v3 --workers 4
```

Read-only status is available through:

```bash
.venv/bin/python tools/pmo_dynamic_v21_recovery.py status
```
