# Local fragment generation

This interface runs the preserved fragment samplers, not the latest experimental
PMO/T4 controllers. All commands are local and offline. The original paper runs,
contracts, model parameters, and chemical support remain unchanged.

## Environment and assets

Use a separate Python 3.11 environment. A newer RDKit is not interchangeable.

```bash
python3.11 -m venv .venv-fragments
.venv-fragments/bin/python -m pip install -r experiments/fragments/requirements-generation.txt
```

The required assets and SHA-256 hashes are in [assets.json](assets.json). The
checkpoint and derived catalogs are not distributed in this checkout, and a
public download has not been verified. Do not substitute another checkpoint or
rebuild catalogs using evaluation prompts. With authorized copies available:

```bash
PYTHONPATH=src python3.11 -m compose_v4.experiments.fragments assets --assets local_assets/fragments --install checkpoint --source /path/to/ringcore_a7546e2_best.pt
PYTHONPATH=src python3.11 -m compose_v4.experiments.fragments assets --assets local_assets/fragments
```

Repeat `--install NAME --source FILE` for `prompts`, `region_catalog`,
`joint_prior`, `pendant_catalog`, `mass_prior`, and the six `evaluator_*` entries.
Verification lists every missing file and expected hash. Installation copies,
verifies both ends, and refuses to replace different existing bytes. Originals
are never moved. `--task TASK` restricts verification to that task's inputs.
The local cache is ignored by Git. Preserve the evaluator's distinct license.
No network download or checkpoint/catalog regeneration occurs implicitly.

## Historical parity

```bash
PYTHONPATH=src .venv-fragments/bin/python -m compose_v4.experiments.fragments parity --task motif_extension --assets local_assets/fragments --output runs/fragments/motif-parity
```

Task names: `motif_extension`, `scaffold_decoration`, `linker_design`,
`superstructure_generation`, and `superstructure_uniform`. Parity uses the first
BARICITINIB attempt at the first declared seed, or the first **two** linker
attempts to exercise per-cell novelty history. Program panels compare exactly,
including proposals, scores, selection, and RNG state. Superstructure compares
accepted actions, event counts, and endpoints. These fixtures were selected by
identity, not quality. They do not establish full-panel equivalence or performance.

On macOS environments with duplicate OpenMP libraries, the existing environment
may require the `KMP_DUPLICATE_LIB_OK=TRUE` environment prefix. This is recorded
in the local verification notes, not silently applied to every environment.

## Generate and evaluate

Start with one explicit slot:

```bash
PYTHONPATH=src .venv-fragments/bin/python -m compose_v4.experiments.fragments generate --task motif_extension --assets local_assets/fragments --seed 2 --prompt BARICITINIB --attempts 1 --output runs/fragments/motif-example
```

To request the task's complete declared panel, omit `--seed` and `--prompt` and
explicitly pass `--attempts 100`: ten prompts and three frozen generation seeds.
This can take substantial CPU time. No complete panel is launched during this
repository integration. Repeat flags to select multiple seeds/prompts.
`--no-metrics` saves molecules and traces without evaluating benchmark metrics.
Scaffold morphing shares linker outputs; do not run it as another experiment.

Execution uses CPU float32, one Torch/inter-op thread, explicit BLAS limits, and
the historical BLAKE2b/NumPy seed derivation. The RingCore fingerprint is checked.
Each task runs in a subprocess with only its frozen source snapshot on the
project import path. Main-checkout core code and private worktrees are not used
by the sampler. No original launch authorization is reused or rewritten.

A completed directory contains `result.json`, `provenance.json`, `worker.log`,
and a README; parity adds `parity.json`. Each attempted slot is retained, including
empty panels. The official metric population includes valid chemical commits
even if they miss the prompt; fidelity remains distinct. A new local run never
replaces the original paper result. Existing outputs cannot be overwritten.

Failure preserves a named `.pending-*` directory with logs and failure provenance,
not a completed result. Automatic resume is not implemented. Inspect a failure
and choose a new output path deliberately; do not combine partial runs into a
nominal complete panel. A hard interruption may also leave the output lock;
check that its process is terminal before any manual recovery.

## Limits

Exact versions and bounded parity were checked locally, not on every CPU or OS.
The full original proposal panels remain in their historical directories; the
fixtures here are not a replacement for those multi-gigabyte records. Local
copies on the same disk are not an off-machine backup. Fresh-clone generation
requires obtaining the exact external assets; table reduction does not.
