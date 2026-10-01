# PMO molecular optimization

PMO optimizes molecular objectives using only the scores collected during search.
COMPOSE constructs executable programs, checks and deduplicates their endpoints,
then allocates objective evaluations with an online controller.

## Local controller example

From an installed checkout, exercise the actual PMO batch selector without an
oracle, network access or a cloud account:

```bash
python examples/reference_guidance.py --selector pmo --mode off \
  --output runs/reference_guidance/pmo-off.json

python examples/reference_guidance.py --selector pmo --mode active --strength 0.25 \
  --checkpoint local_assets/fragments/r_theta_nll.pt \
  --output runs/reference_guidance/pmo-active.json
```

These examples select from two fixed executable programs. They validate the
integration, not a PMO optimization result. Active mode needs the verified
[reference asset](../../docs/reference_guidance.md#run-the-offline-example).

## Complete local campaign

The local campaign runner uses the structured PMO proposer, a task-independent
16-molecule initialization bank, a generic jump-plan checkpoint, the explicit
region-replacement option and durable oracle receipts. It can use the same fixed molecular reference checkpoint as
fragment generation. The reference scores exact executed programs in the
remaining exploration panel. Off mode keeps that slot uniform, shadow mode
records reference scores without changing the draw, and active mode reweights
that slot. The other allocation decisions still use the PMO online controller.

The core process requires Python 3.11 and [the pinned core environment](../../requirements/core.txt).
PyTDC scoring runs in a second Python 3.11 process with RDKit 2023.9.6 and
PyTDC 1.1.15. Keep these environments separate. From the repository root,
install the oracle environment and check it without constructing or calling an
oracle:

```bash
python3.11 -m venv local_assets/pmo/oracle-env
local_assets/pmo/oracle-env/bin/python -m pip install -r requirements/pmo-oracle.txt
local_assets/pmo/oracle-env/bin/python -m pip install --no-deps PyTDC==1.1.15
PYTHONPATH=src local_assets/pmo/oracle-env/bin/python \
  -m compose_v4.experiments.pmo_oracle_worker --check-env
```

The preflight checks the installed versions and imports PyTDC with the RDKit
compatibility shim. It makes no score request. To fetch the three asset-backed
oracle pickles from the Harvard Dataverse file IDs in PyTDC 1.1.15, run:

```bash
python tools/fetch_pmo_oracle_assets.py --download
```

The command verifies every file against [assets.json](assets.json) before
placing it under `local_assets/pmo/oracle-assets/oracle/`. Without `--download`,
it checks local files and makes no network request. The pickles are not
redistributed here. Follow the terms of their Dataverse records. The install
recipe has not yet been tested on another machine.
The [positive-control fixture](assets/oracle_reference_panel.json) contains only
the 28 reference molecules used by the test. It records the SHA-256 of the
original 400-molecule panel. It is not a replacement for the oracle pickles.

The included [structured configuration](example.json) runs albuterol similarity.
[Uniform-chain](uniform_chain.json) and
[created-atom-rebinding](created_atom_rebinding.json) configurations change only
the proposal arm. All three pin the same reference checkpoint, generic PMO
jump plans and initialization bank. The proposal settings specify the eligible pool limit,
region-replacement offer rate and realization cap. They are written into the
run manifest rather than patched into imported modules. The reference strength
is illustrative. It is not a
selected benchmark setting. Set the oracle Python path to your installed
environment, then verify the configuration and input hashes without an oracle call.
The oracle process checks its package versions when a scored run starts:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m compose_v4.experiments.pmo validate \
  --config experiments/pmo/example.json
```

`run` starts a real scored PMO campaign and can consume 1,008 charged oracle
calls. Run it only when that evaluation is intended:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m compose_v4.experiments.pmo run \
  --config experiments/pmo/example.json --output runs/pmo/albuterol-seed-20261925
```

For another objective or replicate, use a distinct configuration and output
directory. The uniform-chain and rebinding arms are interventions, not
alternate names for the learned reference. The budget is 1,008 charged calls and the
Top-10 metrics use the first 1,000 resolved receipts. If fewer than 1,000
calls resolve, `result.json` marks the run incomplete and leaves both metrics
null. It does not publish a partial score as a completed result.

To prepare all 22 objectives, three seeds per objective and matched proposal
arms, use the fixed [task and seed registry](panel.json):

```bash
PYTHONPATH=src python tools/prepare_pmo_panel.py \
  --output runs/pmo/prepared \
  --oracle-python local_assets/pmo/oracle-env/bin/python \
  --oracle-assets local_assets/pmo/oracle-assets \
  --base-seed 20260923 --replicates 3 --strength 0.25 \
  --arms structured uniform_chain created_atom_rebinding
```

This writes 198 configuration files and a hash manifest without querying an
oracle. The 22-objective set excludes valsartan SMARTS. The `--strength` value
is illustrative and must be fixed before scoring. Use `--tasks` for a declared
subset or `--guidance-modes off shadow active` for a matched reference-selection
comparison. The panel manifest pins one checkpoint identity. Non-off configs
load that checkpoint, and matched cells use the same task-derived seed. The
file paths are local to the machine that prepared the
panel. Validate a prepared configuration, then run it with a unique output
directory. The preparer never starts campaigns or reports benchmark results.

Each campaign writes its configuration and implementation identity, query
reservations and results, candidate locks, round snapshots and a result file.
Resume uses the same configuration and code. An unresolved charged query or a
pending round requires explicit receipt-based reconciliation. It is never
retried automatically.

Offline interface tests cover this runner. The existing PMO table comes from a
different frozen configuration and is not attributed to this shared-reference
configuration. Benchmark comparisons require matched, complete scored runs at
a fixed operating point.

## Components

- `RewardAdaptiveProgramController` learns from charged endpoint-minus-parent
  rewards, restores the parent offset for cross-parent comparisons, and retains
  an exploration allocation.
- `ProgramPanelGuidance` joins candidate IDs to exact executed programs and applies
  frozen-reference preferences within the existing exploration allocation.
- `pmo_program_input` checks the exact source state and trace. It does not infer a
  program by reparsing an endpoint or inspect the unknown objective value.
- `PmoReferenceController` reserves one query from the existing batch for
  uniform or reference-weighted exploration. It does not add an oracle call.
- `PmoOracleClient` keeps the PyTDC chemistry kernel separate, verifies
  asset-backed models and records ordered score requests.

Active weights, coverage fallbacks and selected candidate identities are
recorded in a receipt. See [the API and policy](../../docs/reference_guidance.md).
Broad molecular and trajectory parity across the two chemistry kernels has
not been established. The two-program example is not a benchmark result.
