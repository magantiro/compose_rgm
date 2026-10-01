# Fragment generation

Generate molecules with retained fragments using the frozen reference model and
task-specific structural constraints. Execution is local, CPU-only and offline.

## Install

Use Python 3.11 and the pinned chemistry environment. RDKit versions can change
canonical identities, legal edits and evaluation results.

```bash
python3.11 -m venv .venv-fragments
.venv-fragments/bin/python -m pip install -r experiments/fragments/requirements-generation.txt
```

## Assets

[assets.json](assets.json) lists exact input identities. Git LFS holds the
reference checkpoint and path-neutral region and pendant catalogs. The joint
completion and decoration mass priors are tracked directly in the repository.

```bash
git lfs install
git lfs pull --include="local_assets/fragments/r_theta_nll.pt,local_assets/fragments/region_catalog.json,local_assets/fragments/pendant_catalog.json"
```

Fetch the six evaluator files and the ten-prompt table from the pinned
InVirtuoGen commit with:

```bash
python tools/fetch_fragment_evaluator.py --download
python tools/fetch_fragment_evaluator.py
```

The downloader verifies every upstream file against the manifest before
publishing it under `local_assets/fragments/evaluator/pkg/`. It appends one
terminal newline to the upstream `references/fragments.csv` to produce the
hash-pinned `prompts.csv`. The external files are not vendored here and retain
the [upstream CC BY-NC-SA 4.0 terms](https://github.com/invirtuolabs/InVirtuoGen_results/blob/b50bb3ae2bdcb9df581f0b219d79cf14b05d0fbb/LICENSE).

Verify the inputs for a task before generating:

```bash
PYTHONPATH=src .venv-fragments/bin/python -m compose_v4.experiments.fragments assets --assets local_assets/fragments --task motif_extension
```

The region and pendant catalogs come from Git LFS. Missing files and incorrect
hashes cause an error. The evaluator download command obtains pinned external
code but does not execute it. No command rebuilds a catalog from evaluation
data.

## Run

Start with one output slot:

```bash
PYTHONPATH=src .venv-fragments/bin/python -m compose_v4.experiments.fragments generate --task motif_extension --assets local_assets/fragments --seed 2 --prompt BARICITINIB --attempts 1 --output runs/fragments/motif-example
```

Available tasks are `motif_extension`, `scaffold_decoration`, `linker_design`,
`superstructure_generation` and `superstructure_uniform`. Scaffold morphing uses
the linker outputs and is not a separate sampling task.

[generation.json](generation.json) defines the prompt panel, default seeds and
superstructure settings. Repeat `--seed` or `--prompt` to select several cells.
Omit both flags and pass `--attempts 100` to run the full task panel. This can take
substantial CPU time. Use `--no-metrics` to generate molecules and traces without
the external evaluator.

The worker uses the source tree, CPU float32 and one Torch thread. It checks the
NLL-trained reference checkpoint and its frozen catalog before sampling. Each
prompt and seed has its own BLAKE2b-derived NumPy random stream. All attempted slots remain
in the output, including empty panels. Prompt fidelity and chemical validity are
separate checks.

For motif extension, scaffold decoration and linker design, the saved offer
panels can be reused for a matched selector comparison. Run
`selection-ablation` on a completed `result.json` as shown in the
[ablation guide](../ABLATIONS.md#fragment-selection-on-fixed-panels). This
compares selection laws on identical attempted offers. It does not rerun
proposal construction.

## Outputs and interruption

A completed output directory contains:

- `result.json`: molecules, executable traces, selection records and metrics
- `provenance.json`: configuration, input and source hashes, and code revision
- `worker.log`: progress and diagnostics

Existing outputs cannot be overwritten. Failed runs retain a `.pending-*`
directory with logs but do not publish a completed result. Automatic resume is
not implemented. Check that the original process has stopped before recovering
a stale lock, then choose a new output path. Do not combine partial runs into a
nominal complete panel.

## Test

```bash
COMPOSE_FRAGMENT_ASSETS=local_assets/fragments PYTHONPATH=src .venv-fragments/bin/python -m pytest -q tests/test_fragment_generation_local.py
```

This checks a one-attempt NLL-reference generation from an isolated source
export. It verifies the checkpoint identity, output schema and provenance
without running the benchmark evaluator.

The optional local-asset tests use `COMPOSE_FRAGMENT_ASSETS` to locate assets:

```bash
COMPOSE_FRAGMENT_ASSETS=local_assets/fragments PYTHONPATH=src .venv-fragments/bin/python -m pytest -q tests/test_fragment_evaluator_loader.py
```

Exact versions and bounded regression checks do not guarantee identical output
on every CPU or operating system.
