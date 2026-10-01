# Model and evaluator assets

The source repository contains code, task configurations and small input
registries. The fixed NLL-trained reference checkpoint and two training-derived catalogs
are tracked through Git LFS on the release branch. Other assets remain
separate. Every task checks declared SHA-256 identities before using them.
Do not replace a missing asset with another checkpoint or rebuild a catalog
from evaluation examples.

| Asset | Used by | Location | Identity and source |
| --- | --- | --- | --- |
| Fixed molecular reference checkpoint | Fragment, QED, PMO, T4 | `local_assets/fragments/r_theta_nll.pt` | [Reference manifest](reference/model.json). Tracked through Git LFS and verified by SHA-256. |
| Frozen model catalog | Fragment, QED, PMO, T4 | `local_assets/fragments/catalog.json` | Hash-pinned by the reference manifest and used directly by the fragment runner. |
| Fragment prompts and training-derived catalogs | Fragment | `local_assets/fragments/` | [Fragment asset manifest](fragments/assets.json). The prompt table is fetched from pinned upstream bytes. Path-neutral region and pendant catalogs are tracked through Git LFS. The joint completion and mass priors are tracked directly. |
| Fragment benchmark evaluator | Fragment | `local_assets/fragments/evaluator/` | The manifest pins files from a specific InVirtuoGen revision. Use the [hash-checked fetch command](fragments/GENERATION.md#assets). The files are not vendored here. |
| Finite-horizon value head | QED | task-selected output path | Fit from reference rollouts using the [QED workflow](qed/README.md). It is distinct from the fixed molecular reference. |
| PyTDC oracle environment and model files | PMO | `local_assets/pmo/` | [PMO asset manifest](pmo/assets.json) and [verified fetch command](pmo/README.md#complete-local-campaign). PyTDC uses a separate RDKit version. |
| Program template priors | T4 | `experiments/t4/assets/` | Tracked here and verified by the [T4 asset manifest](t4/assets.json). They are controller inputs, distinct from the fixed molecular reference. |
| QuickVina2, Open Babel and receptors | T4 | `local_assets/t4/` | Public MOOD scoring files are identified in the [T4 guide](t4/README.md). The local executable and receptor must pass the configured hashes. |

## Fetch the LFS assets

On a normal clone of the release branch, install Git LFS and fetch the exact
tracked assets:

```bash
git lfs install
git lfs pull --include="local_assets/fragments/r_theta_nll.pt,local_assets/fragments/region_catalog.json,local_assets/fragments/pendant_catalog.json"
PYTHONPATH=src python -m compose_v4.experiments.fragments assets \
  --assets local_assets/fragments --task superstructure_generation --no-evaluator
```

The last command verifies the checkpoint and reports the other inputs still
needed for that task. A Git LFS pointer is not a checkpoint. Verify the file
hash after fetching it.

## Install remaining assets

Fetch the public fragment evaluator and prompt inputs with the pinned command
below. Then verify the task assets:

```bash
python tools/fetch_fragment_evaluator.py --download
PYTHONPATH=src python -m compose_v4.experiments.fragments assets \
  --assets local_assets/fragments --task motif_extension
```

PMO and T4 configurations point at this same checkpoint identity. The
[fragment guide](fragments/GENERATION.md) lists task-specific asset keys and the
pinned environment.

PMO's three external model files can be fetched with
`python tools/fetch_pmo_oracle_assets.py --download`. The command uses the file
IDs in PyTDC 1.1.15 and publishes only bytes that match the PMO manifest. It
does not evaluate any molecule.

The external evaluator files retain the upstream CC BY-NC-SA 4.0 terms. The
prompt table differs from the upstream reference file only by a terminal
newline. All resulting bytes are checked against the fragment manifest.

T4's QuickVina2 binary and receptors can be fetched with
`python tools/fetch_t4_docking_assets.py --download`. The binary is for Linux
x86-64. Install Open Babel separately and record the exact local executable
hash in the T4 configuration.

Task-specific external assets must still be fetched and hash-verified for
independent full-task runs.
The offline tests and the zero-weight example can run without these assets.
See the [ablation recipes](ABLATIONS.md) for the exact arm changes and their
run commands.
