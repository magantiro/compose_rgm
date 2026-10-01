# Local verification

Use Python 3.11 for the core chemistry and model code. Create a fresh environment
and install the pinned dependencies from the repository root:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements/test.txt -e .
.venv/bin/python -m pip check
```

The core environment uses RDKit 2024.3.5. The PMO oracle requires a separate
RDKit 2023.9.6 and PyTDC 1.1.15 environment. Do not combine the two or assume
that matching SMILES strings prove chemistry-kernel parity. The PMO task guide
states the current oracle setup boundary.

Fragment generation uses pandas 2.3.3, which is pinned in
`requirements/test.txt`. An otherwise compatible Python environment without
that package fails before generation. Use the complete test environment or the
fragment environment from `experiments/fragments/requirements-generation.txt`.

## Task checks

These checks use local fixtures and do not make oracle or docking calls:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_fragment_generation_local.py \
  tests/test_fragment_asset_cli.py \
  tests/test_fragment_selection_ablation.py \
  tests/test_fragment_evaluator_loader.py \
  tests/test_fragment_evaluator_fetch.py \
  tests/test_fragment_asset_export.py \
  tests/test_reference_guidance.py \
  tests/test_reference_selection.py \
  tests/test_qed_shared_training.py \
  tests/test_qed_shared_fit.py \
  tests/test_qed_shared_pipeline.py \
  tests/test_qed_shared_value.py \
  tests/test_qed_shared_reduction.py \
  tests/test_qed_shared_reference.py \
  tests/test_qed_shared_rollouts.py \
  tests/test_qed_shared_smc.py \
  tests/test_qed_shared_sources.py \
  tests/test_t4_local_campaign.py \
  tests/test_t4_quickvina.py \
  tests/test_t4_panel.py \
  tests/test_t4_cli.py \
  tests/test_t4_reference_panel_example.py \
  tests/test_t4_docking_asset_fetch.py \
  tests/test_pmo_campaign_config.py \
  tests/test_pmo_panel.py \
  tests/test_pmo_oracle_preflight.py \
  tests/test_pmo_oracle_asset_fetch.py \
  tests/test_environment_contract.py \
  tests/test_pmo_reference_controller.py \
  tests/test_pmo_binding_intervention.py \
  tests/test_shared_reference_identity.py \
  tests/test_nll_reference_checkpoint.py \
  tests/test_release_guides.py
```

The [fragment](../experiments/fragments/GENERATION.md),
[QED](../experiments/qed/README.md), [PMO](../experiments/pmo/README.md), and
[T4](../experiments/t4/README.md) guides give their own asset checks and run
commands. Tests that need an external asset must name that asset and skip only
when it has not been installed. An installed asset with the wrong hash is an
error.

Before a release, run the full suite and record its actual summary:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests/ -ra --durations=20
.venv/bin/python -m ruff check tests tools examples \
  src/compose_v4/experiments/fragments \
  src/compose_v4/experiments/pmo \
  src/compose_v4/experiments/pmo_panel.py \
  src/compose_v4/experiments/qed_shared* \
  src/compose_v4/experiments/t4 \
  src/compose_v4/model/reference_checkpoint.py \
  src/compose_v4/control/reference_guidance.py \
  src/compose_v4/control/reference_programs.py
.venv/bin/python -m ruff format --check tests tools examples \
  src/compose_v4/experiments/fragments \
  src/compose_v4/experiments/pmo \
  src/compose_v4/experiments/pmo_panel.py \
  src/compose_v4/experiments/qed_shared* \
  src/compose_v4/experiments/t4 \
  src/compose_v4/model/reference_checkpoint.py \
  src/compose_v4/control/reference_guidance.py \
  src/compose_v4/control/reference_programs.py
git diff --check
```

The [automated check](../.github/workflows/verify.yml) runs these commands on
Linux after downloading the Git LFS weights. It verifies the reference, QED
source, and QED asset identities, then fetches the six hash-pinned public
fragment evaluator files. Tests run offline with the checkpoint and evaluator
installed. The workflow does not install PMO oracle assets, fit a QED value
head, or run docking. Lint covers the task interfaces and tools named above.
The full source tree has additional lint findings and is not presented as
repository-wide Ruff clean.

Do not call a partial or asset-skipped run a complete reproduction. On macOS,
subprocess tests may need a terminal with working OpenMP shared memory and
semaphore access. A sandbox denial is not a model or chemistry failure. Run
the exact affected test once in that environment and report both outcomes.

Formatting is limited to touched files during ordinary changes. Source files
bound by an existing hash contract must not be reformatted merely to make a
lint command green. Such a change needs a new contract and a new run identity.

## Fragment catalog provenance

The region and pendant catalogs in Git LFS have portable provenance labels.
Their manifest entries record both the source and distributed SHA-256 hashes.
To audit or reproduce a path-neutral export from the exact source assets, run:

```bash
python tools/export_fragment_assets.py \
  --source-dir local_assets/fragments \
  --output-dir /path/to/new/fragment-export
```

The command checks all four source hashes, changes only provenance path labels,
and publishes a complete output directory with both input and output hashes.
It refuses to overwrite a different export. The released catalogs and priors
are already path neutral, so exporting them again leaves their bytes unchanged.
