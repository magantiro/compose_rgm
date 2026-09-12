# Fully scored multi-site program pool

Measure docking quality of the corrected proposal pools, including attachment
variants. This is winner-informed PARP1 seed0 delta=0.4 development, not held-out
learning or autonomous comparison against IVG. It is a smaller diagnostic within
the user's approved 66-call ceiling, not the originally suggested three-context,
eight-candidates-per-arm pilot.

## Locked allocation

Use the original-seed pools from attempt 2 and post-linker pools from attempt 3.
Exclude attempt 1 and attempt 2's post-linker serial comparison. All arms retain
the original benchmark seed for endpoint similarity. Proposal caps match within
each context; actual work differs and is reported.

The corrected pools contain 27 unique eligible molecules. Reuse the three paid
winner controls and dock all 26 others once, at QuickVina seed 1701. Repeat the
lowest successful first-evaluation candidate twice, at seeds 1702 and 1703; ties
use canonical SMILES. At most 28 NEW calls, including failures. Ambiguous started
calls are not retried. This remains separate from the nineteen-call refinement.

`configs/t4_program_pool_lock.json` seals exact replay traces, original-seed
properties, input-pool hashes, arm/context ancestry, controls and call limits.
All 26 candidates were locked before these dockings, without a surrogate or
score-based selection. The program banks use the known winning route; removing
the target from inference does not remove that development information.

Keep the existing Open Babel preparation, receptor, box, QuickVina2 binary,
exhaustiveness 1, modes 10 and CPU 1. Keep QED >=0.6, SA <=4, radius-two 2048-bit
similarity >=0.4 and the existing endpoint gate. Save ligand/pose hashes.
Maximum eight workers plus one driver, workers <=480 seconds, driver <=3600
seconds, no automatic retries, CPU only, reserved cost <=$20. Expected wall time:
60–900 seconds. Use a separate deployment so the running refinement is unchanged.

## Decisions

Report every score and failure. Attribute shared molecules to all corresponding
arms but charge one evaluation per unique candidate. Separate selection scores
from the two fresh repeats; report winner controls as reused, not discovered.
If joint proposals concentrate on good molecules but lose diversity, improve
parameter/attachment diversity and scored replay. If the varied independent pool
contains better molecules, retain that channel and learn from those combinations.
A null repeat-supported improvement is a null, not permission to scale the same
bank unchanged. No benchmark superiority follows from this inspected pool.

## Execution

Prepare with pinned RDKit 2024.03.5:

```sh
PYTHONPATH=src:. python tools/t4_program_pool.py prepare
```

A completed lock is reused, not regenerated. After focused checks, from clean
committed source:

```sh
python -m modal deploy modal_apps/t4_program_pool_app.py
PYTHONPATH=src:. python tools/t4_program_pool.py launch
```

Results persist under `compose-v4-artifacts/t4_program_pool/<run_id>`.
Docking these verified endpoints requires no reference-model initialization.
