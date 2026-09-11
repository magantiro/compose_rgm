# Locked-batch PMO persistence repair

The completed 272-query development ledger replays with **544 serial oracle
barriers versus 8 batch barriers**, with identical canonical molecule order,
observed values and budget accounting. The four existing locked groups contain
199, 2, 26 and 45 new queries. Zero new PMO oracle evaluations were performed.

`result.json` binds the original observation artifact, implementation hashes,
base commit, uncommitted-source status, environment and replay configuration.
This measures the number of forced synchronous barriers in the scoring path,
not total Modal commits or end-to-end remote speed. Background heartbeats,
candidate preparation and result publication are outside that count. The local
durations exclude network synchronization and are not a Modal speedup estimate.

The implementation adds `DurableScores.score_many` and uses it only where the
trajectory experiment already locks an entire candidate list before reading
scores. Serial adaptive scoring is unchanged. No model, proposal law, oracle,
score transform, queried molecule, query order or scientific selection rule was
changed. The previous negative optimization result is not rerun or relabeled.

Before calling the oracle, the entire batch must fit the remaining budget and
all novel canonical identities are reserved durably. Results are persisted
before values return to the controller. A failed or ambiguous reservation
prevents automatic continuation or retry. Batch interruption can leave more
ambiguous reservations than serial interruption; this is an explicit recovery
tradeoff, not permission to assume unobserved calls were free.

Verification: 13 focused tests passed in 2.49 seconds (batch and existing serial
ledger behavior, exception/interruption recovery, budget preflight, trajectory
decision invariants). The unrelated JNK3 oracle fixture was not run. Ruff checks
passed on the four touched Python files. No repository-wide qualification or
remote performance improvement is claimed.

Reproduce from the isolated development checkout with the pinned chemistry path:

```sh
env PYTHONPATH=/private/tmp/compose-t4-chemistry.hizM8Y:src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Users/rmaganti/compose_rgm_git/.venv/bin/python tools/pmo_batch_io_profile.py --input /private/tmp/compose-pmo-trajectory-value-runs/97b2acd334d2a8dcdda5f0dbede4f033a66813da3dc7da67b82dad53faf4fca0/result.json --output diagnostics/pmo_locked_batch_io/result.json
```

The latest user authorization permits up to 30 Modal containers. This is a
concurrency ceiling, not a reason to launch redundant work or expand a scientific
budget. The pending support-repair audit still has five source workers and one
driver. The user subsequently explicitly approved the source-hashed development
exception. No denied launch was retried before that approval. Source changes
since the last deployment require a refreshed, matching deployment before launch.
