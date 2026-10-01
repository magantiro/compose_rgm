# Similarity-constrained QED editing

QED editing uses a frozen molecular reference to propose legal transitions. A
separate finite-horizon value model guides those transitions toward QED at least
0.9 while retaining source similarity of at least 0.4.

## Check the shared reference

The transition adapter loads the same hash-verified checkpoint used by fragment
generation. In the Python 3.11 environment from `requirements/core.txt`, run:

```bash
PYTHONPATH=src python tools/check_qed_source_support.py \
  --expected-sha256 704103777e8050eb59f4d15d9997ca6070ba05a6b878b8e18738bfb1e706a090
PYTHONPATH=src python tools/verify_qed_assets.py
PYTHONPATH=src python examples/qed_shared_reference.py --time 0.5
```

The first command fails if a source falls outside the graph representation or
if conversion changes its QED or benchmark Morgan fingerprint. It prints the
input and code hashes with the source counts. The representation does not store
stereochemistry. Keep the original source SMILES for benchmark scoring and
source identity. The check confirms metric equivalence, not stereochemical
equivalence.

The source roles are fixed in `shared_sources.json`. Load them through
`compose_v4.experiments.qed_shared_sources.load_qed_source_roles`. It verifies
all three input hashes and groups by nonisomeric molecular graph. One training
source shares that graph identity with a test source and is explicitly excluded
from head training. The resulting roles contain 1,023 training sources, 128
validation sources, and 800 test sources. Run
`PYTHONPATH=src python -m pytest -q tests/test_qed_shared_sources.py` to check
the split without fitting a model.

The asset check verifies the shared checkpoint and source-role hashes. It
reports `reference_only` until a matched value head is supplied with `--value`.
The final command draws one canonical successor from `CCO`. It checks the model
hash, scores the complete legal successor set, and samples from the normalized
distribution. It does not use a value head or run the benchmark.

The adapter is `compose_v4.experiments.qed_shared_reference.QEDSharedReference`.
It requires 48 persistent slots and no more than 40 active atoms. An empty legal
fiber returns `None`. At the 40-atom boundary, successors outside that support
are removed and the remaining canonical probabilities are renormalized. The
model stays in evaluation mode and is never fitted by the adapter.

## Finite-horizon control

The value head must be trained on continuations from this exact reference
checkpoint. The adapter checks the head metadata against the checkpoint hash
and rejects a mismatch. Training and validation rollouts can be generated one
source at a time from the fixed source roles. To run the complete source-disjoint
rollout, feature and fit workflow with bounded local workers, use:

```bash
PYTHONPATH=src python tools/run_qed_shared_value_pipeline.py \
  --workspace runs/qed_shared --workers 4
```

Use `--resume` after an interruption. Existing source outputs are kept. The
feature builder checks each rollout, and the fit stage checks the complete
feature corpus. The stages can also be
run separately with `--stage rollouts`, `--stage features`, or `--stage fit`.
For an individual source, the corresponding commands are:

```bash
PYTHONPATH=src python tools/run_qed_shared_rollout.py \
  --role train --index 0 --horizon 24 --replicates 8 --time 0.5 \
  --output runs/qed_shared/train_0000.json
PYTHONPATH=src python tools/build_qed_shared_features.py \
  --rollout runs/qed_shared/train_0000.json --role train --index 0 \
  --budget-max 24 --time 0.5 \
  --output runs/qed_shared/features/train_0000.npz
```

The runner stores the original source and its input row, exact reference
identity, derived seeds, every visited molecular state, QED, source similarity,
and code and input hashes. It makes no oracle call and refuses to overwrite a
prior rollout. It cannot select a test source. The feature command verifies the
source role and reference identity before building finite-horizon hitting
labels. It excludes states already in the goal region, whose value is fixed by
the exact boundary condition.

After producing one feature shard for every training and validation source,
fit the head with:

```bash
PYTHONPATH=src python tools/fit_qed_shared_value.py \
  --shards runs/qed_shared/features --time 0.5 \
  --output runs/qed_shared/value
```

The trainer refuses an incomplete or mixed corpus. It fits normalization only
on training sources and uses binary cross-entropy with a 0.3-weighted Bellman
consistency term. The Bellman targets use the observed next state, with the
goal boundary fixed exactly. Each training source contributes one optimizer
update per epoch. The trainer selects a checkpoint by source-level validation
loss and also checks the
benchmark goal, QED 0.9 and similarity 0.4, separately. The evaluator refuses
a head unless both training and validation examples include hits and misses for
that goal and its source-mean validation Brier score beats a training-only
constant predictor. A failed check is recorded, not hidden. CUDA training is optional
with `--device cuda` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before launch.
The fitted reference remains unchanged. No fitted head or 800-source result for
this reference identity is bundled here.

`QEDSharedValueHead` loads a fitted head only when its file hashes, complete
reference configuration, feature schema, and source split match the supplied
identities. Its source-bound evaluator applies the exact target boundary at
budget zero and for molecules already in the goal region.

The generic sequential Monte Carlo operations are in
`compose_v4.experiments.hphi_smc`. The source-level controller is
`compose_v4.experiments.qed_shared_smc.run_source`. It returns one terminal
candidate per independent particle run. If all particles lose target support,
it records `EXTINCT_NO_HIT` and fills that failed output slot with the original
source. Source-level success is evaluated on the 800 molecules in
`data/jin/qed_test.txt`. Returned candidates must satisfy the QED and
Morgan-fingerprint similarity thresholds above.

After fitting a head, run each test source into a separate file. For example:

```bash
PYTHONPATH=src python tools/run_qed_shared_source.py \
  --index 0 --time 0.5 --value runs/qed_shared/value \
  --output runs/qed_shared/test/source_0000.json
```

Repeat this command for indices 0 through 799 with matching output names. Then
verify and reduce the complete panel:

```bash
PYTHONPATH=src python tools/reduce_qed_shared.py \
  --results runs/qed_shared/test \
  --output runs/qed_shared/reduction.json
```

The reducer refuses missing or mixed sources, model identities and code
identities. It recomputes QED and source similarity for every returned slot and
reports the source-level success fraction and a 95% Wilson interval. These
commands define the complete-panel reduction for the fitted head named in each
source result.
