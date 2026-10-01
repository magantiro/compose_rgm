# T4 similarity-constrained lead optimization

T4 starts from a supplied lead and optimizes a docking objective subject to
endpoint constraints: QED at least 0.6, SA at most 4 and Morgan-fingerprint
Tanimoto similarity at least the configured threshold. Candidate endpoints are
docked only after they pass the configured gate. The supplied lead is a charged
initialization call even if it fails that gate. Its score informs parent selection,
but an ineligible lead cannot become the reported best feasible molecule.

## Local controller example

Exercise the actual T4 batch selector without docking or a cloud account:

```bash
python examples/reference_guidance.py --selector t4 --mode off \
  --output runs/reference_guidance/t4-off.json

python examples/reference_guidance.py --selector t4 --mode active --strength 0.25 \
  --checkpoint local_assets/fragments/r_theta_nll.pt \
  --output runs/reference_guidance/t4-active.json
```

The examples use two fixed executable programs and one cold-start selection slot.
They are implementation checks, not docking campaigns. Active mode requires the
verified [reference asset](../../docs/reference_guidance.md#run-the-offline-example).

## Proposal generation and selection

`compose_v4.experiments.t4_fiber_campaign` provides the endpoint gate and proposal
expansion. The default gate additionally applies COMPOSE's structural screening
rules. This gate is not identical to the benchmark thresholds alone.
`expand(..., include_realized_actions=True)` retains exact source states and
compiles accepted structural goals into replayable primitive programs. A compiler
abstention remains an unscored candidate, not a silently removed output.

`compose_v4.experiments.t4_integrated_route_fiber.select_batch` reserves early
template-prior and scale allocations. In the local program-only runner, the
remaining eligible candidates are sampled without a fitted program-value model.
Optional frozen-reference guidance changes that draw. The program template prior
constructs candidates, and the fixed molecular reference can guide selection.
See [reference selection](../../docs/reference_guidance.md)
for coverage, probability and receipt semantics.

`compose_v4.experiments.t4_route_proposals` provides the route-template lane. Load
a program template prior with an explicit file hash, construct complete-region
programs, and retain their exact primitive traces with `include_realized_actions=True`.
`examples/t4_reference_panel.py` runs either the shallow or route lane through
the reference-aware selector. See the [generated-panel commands](../../docs/reference_guidance.md#generated-t4-panels).
This checks generation and selection without docking. The program template
prior and neural weights must both be available locally. The five target-specific
priors are included in [`assets/`](assets/) with their source revision and SHA-256
identities in [`assets.json`](assets.json). They are leave-one-target-out controller inputs,
not the neural molecular reference.

Check their physical and internal identities with:

```bash
python tools/verify_t4_assets.py
```

## Local optimization

The local runner supports the shallow, anchored-replacement and route-template
lanes listed in its configuration. It samples archive parents without score
ranking and reduces the weight of parents whose recent proposals yielded no
fresh executable endpoint. It does not fit a program-value model. Charged scores
update the archive and the reported incumbent. The frozen reference reweights
the non-floor selection slots. This runner does not include the separate
state-routing and support-escalation policies.
The example strength is illustrative. No benchmark operating point has been
selected for this new integration.

Use Python 3.11 and [the pinned core environment](../../requirements/core.txt).
Install [Open Babel](https://openbabel.org/docs/Installation/install.html) and
obtain a compatible QuickVina2 executable and prepared receptor. The executable
and PARP1 receptor identities in [example.json](example.json) come from the
MOOD scoring assets used by the existing adapter. The source location is
[MOOD/scorer](https://github.com/SeulLee05/MOOD/tree/main/scorer).
The runner never downloads tools or accepts a hash mismatch. Third-party tools
and receptor data retain their own terms.

On Linux x86-64, fetch the exact MOOD QuickVina2 binary and five receptors
used by the T4 configurations with:

```bash
python tools/fetch_t4_docking_assets.py --download
python tools/fetch_t4_docking_assets.py
```

The fetcher pins a MOOD source commit and verifies every SHA-256 before
publishing a file. It does not install Open Babel. The pinned QuickVina2 file
is a Linux x86-64 executable and cannot run natively on macOS. Use a matching
Linux environment for that binary. A different docking binary needs a new
recorded evaluator identity.

Supply the shared NLL reference weights, both executables and the receptor. The
target-specific program template prior is included in this task directory. The
neural reference is tracked through Git LFS on the release branch. Check that its
bytes, rather than an LFS pointer, are available in your checkout. The example
uses the verified PARP1 prior.

For one cell, edit the example to match your local paths and record the Open
Babel executable's SHA-256. Relative paths are resolved from the configuration
file's directory. Use binaries built for the host platform. The run lock
requires POSIX file locking. The command below checks the environment,
configuration and file hashes. It does not load neural weights or execute
either docking tool:

```bash
python -m compose_v4.experiments.t4 validate --config experiments/t4/example.json
```

The following command performs real docking and consumes the configured budget:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m compose_v4.experiments.t4 run \
  --config experiments/t4/example.json --output runs/t4/parp1-seed0
```

For the complete panel, [`targets.json`](targets.json) identifies all 15
supplied leads, five target-specific program template priors, receptor hashes
and docking boxes. Prepare one configuration per lead, threshold and replicate
with:

```bash
PYTHONPATH=src python tools/prepare_t4_panel.py \
  --output runs/t4/prepared \
  --obabel /path/to/obabel --qvina /path/to/qvina02 \
  --receptors /path/to/receptors \
  --checkpoint local_assets/fragments/r_theta_nll.pt \
  --base-seed 20260918 --replicates 3 --strength 0.25
```

The preparer hashes every input, validates each generated configuration, writes
a manifest and refuses to overwrite an existing panel. It makes no docking
calls. The strength above is illustrative, not a selected benchmark setting.
Choose and freeze it before a scored comparison. Run each configuration with
`python -m compose_v4.experiments.t4 run --config <cell.json> --output
<distinct-run-directory>`. The preparer does not launch the campaigns.
For a matched reference-selection ablation, add
`--guidance-modes off shadow active`. The preparer then writes three configs
per cell with the same lead, program template prior, search seed, docking seed
and budget.
Off removes only the neural reference, shadow records its scores, and active
uses them for non-floor selection. See the [ablation guide](../ABLATIONS.md#t4-reference-selection).
Do not reuse one target's prior as another target's asset without declaring
that separate transfer setting. The same neural reference is pinned in every
generated configuration.

## Accounting and resume

The initial lead counts against the evaluation budget. Its molecular graph must
be representable, but it need not satisfy the endpoint QED or SA thresholds.
Candidate endpoints are checked against the original lead, then deduplicated
before selection. Each generated candidate must have a nonempty primitive
program that replays from its measured parent. Missing or failed compilations
remain in `construction_abstentions` and
are not docked. This execution gate is identical in off, shadow and active modes.
It is separate from native reference-score coverage. An executed program without
a supported neural score remains eligible under the declared coverage policy.
Completed query
receipts are cached by canonical molecule. Every attempted evaluator call gets a
durable reservation before execution. A failed or interrupted attempt remains
charged and blocks automatic retry.

Resume with the same inputs and environment:

```bash
python -m compose_v4.experiments.t4 resume \
  --config experiments/t4/example.json --output runs/t4/parp1-seed0
```

Resolved receipts from an interrupted round are reused. Its locked panel is not
generated again. Changes to code, environment, reference settings, assets,
evaluator or budget are refused. A reservation without a resolved result requires
manual reconciliation. Never remove its lock or start the same run again to
conceal an uncertain call.

Each directory contains:

- `manifest.json`: complete configuration, source hashes, software and asset identities.
- `initialization.json`: the initial lead and endpoint checks.
- `round_*/lock.json`: executable proposals, construction abstentions, reference receipts, selected queries and RNG state.
- `round_*/complete.json`: the resolved receipt identities for that round.
- `oracle/query_*`: durable reservations and evaluation results.
- `docking/query-*`: inputs, command logs, ligand files and poses.
- `result.json`: status, charged count, archive, feasible-archive count and best
  feasible score. The best score is null if no eligible molecule was docked.

`budget_exhausted`, `candidate_exhausted` and `round_limit` are distinct outcomes.
The latter two do not assert that the evaluation budget was completed.

Open Babel conformer generation is unseeded. The recorded QuickVina seed does
not make the whole scoring pipeline deterministic. Binary hashes also do not
identify every dynamically linked library or Open Babel data file. Keep the
system environment fixed when comparing runs. Offline command and resume tests
check the local driver, not numerical parity with the frozen cloud docking runs.

## Offline tests

```bash
OMP_NUM_THREADS=1 python -m pytest -q \
  tests/test_t4_local_campaign.py tests/test_t4_quickvina.py \
  tests/test_t4_cli.py tests/test_t4_panel.py
```

These tests use synthetic score fixtures or mocked subprocesses, not docking.
Set `COMPOSE_REFERENCE_CHECKPOINT` to the verified local weight path to include
the real-reference loop check.
