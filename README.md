# COMPOSE

Molecular generation and optimization through executable graph transformations.
COMPOSE separates a validity-preserving rewrite system, learned molecular-edit
preferences, and task-specific controllers for structural constraints and objective
feedback.

The [method map](docs/METHODS.md) connects the molecular transition equations to
the four task runners and states where each controller uses the frozen model.
The [ablation guide](experiments/ABLATIONS.md) lists the runnable interventions
and their claim boundaries.

## Installation

Use Python 3.11 and the pinned core environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/core.txt -e .
```

PMO oracle evaluation uses a separate RDKit/PyTDC environment. Do not upgrade or
combine chemistry kernels when comparing runs. See the task guides below.

## Quick check

Run a local, zero-oracle example without downloading model weights:

```bash
python examples/reference_guidance.py --mode off --output runs/reference_guidance/example.json
```

This constructs two executable molecular edits and records a selection. To score
those edits using the frozen reference, obtain the verified checkpoint described
in the [asset guide](experiments/fragments/GENERATION.md#assets):

```bash
python examples/reference_guidance.py \
  --mode active --strength 0.25 \
  --checkpoint local_assets/fragments/r_theta_nll.pt \
  --output runs/reference_guidance/guided-example.json
```

For scored comparisons, set the guidance strength in the task configuration
before running either arm. Example outputs include exact programs, asset/code
identities, configuration and selection probabilities. Existing outputs are
never overwritten.

## Tasks

| Task | Guide | Current local interface |
| --- | --- | --- |
| Fragment-constrained generation | [Generate and evaluate](experiments/fragments/GENERATION.md) | Motif extension, scaffold decoration, linker design and superstructure generation |
| Frozen-reference selection | [Reference guidance](docs/reference_guidance.md) | Off, shadow and active modes with exact program replay and coverage checks |
| Similarity-constrained QED editing | [QED editing](experiments/qed/README.md) | Shared-reference rollouts, value-head training, source evaluation and complete-panel reduction. A fitted head is not bundled |
| PMO optimization | [PMO](experiments/pmo/README.md) | Local campaign runner, 22-objective panel preparer, reference-aware exploration and durable receipts |
| T4 lead optimization | [T4](experiments/t4/README.md) | Local generation, reference-aware selection, docking adapter and explicit resume |

The QED transition adapter uses the shared NLL-trained reference checkpoint. Its
finite-horizon value head is fitted under that same checkpoint using the
documented source split. The task guides give the inputs, configurations and
commands for generating and reducing fresh results under each controller.

## Weights and data

The [asset guide](experiments/ASSETS.md) lists the exact Git LFS assets and
what must still be obtained separately. The [fragment asset manifest](experiments/fragments/assets.json)
pins the shared checkpoint, catalogs, prompts and evaluator files by SHA-256.
The task asset command validates local copies and reports missing or incompatible
files. See the asset guide for distribution and access details. Do not
substitute similarly named weights.

Task-specific controllers remain separate from the frozen molecular reference.
Active reference selection changes probabilities. Shadow mode only records scores.

## Library layout

```text
src/compose_v4/
    chem/           molecular representation and chemistry utilities
    rewrite/        executable edits and validity checks
    model/          learned reference models and checkpoint loading
    control/        program construction and selection
    experiments/    task adapters and evaluation
examples/           small runnable examples
experiments/        task guides, inputs and configurations
requirements/       pinned environments
tests/              offline behavior and scientific-invariant checks
```

## Verification

See [verification](docs/DEVELOPMENT.md) for test environments,
external-asset requirements and full-suite commands. A focused example test is:

```bash
python -m pip install -e '.[dev]'
OMP_NUM_THREADS=1 python -m pytest -q tests/test_reference_guidance.py tests/test_reference_selection.py
```

## License

Project code is covered by [LICENSE](LICENSE). External datasets, evaluator
implementations and model assets retain their respective terms.
