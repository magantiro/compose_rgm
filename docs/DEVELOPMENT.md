# Local development and verification

Use Python 3.11 for the validated core environment. Create a separate environment
for PMO oracle reproduction: its RDKit 2023.9.6 kernel is not interchangeable
with the core RDKit 2024.3.5 kernel. Installing the broad package dependencies
alone does not pin a chemistry environment.

From the repository root:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements/test.txt -e .
.venv/bin/python -m pip check
```

The test requirements include the cloud SDK because offline tests inspect its
entrypoints. Unit tests do not require a cloud login or authorize a launch.
They use fixture data and mocked expensive evaluators. Real campaigns require
their separately verified assets, experiment configuration, and authorization.

## Test environment and evidence tiers

The Python 3.11 environment above is the **core test kernel**, not every
experiment's runtime. The PMO scored oracle image uses RDKit 2023.9.6 and
PyTDC 1.1.15; older PMO property-program and winner-program contracts instead
pin PyTDC 0.3.6 with RDKit 2024.03.5. The serialized pan-lung model has its own
exact NumPy/Pandas/RDKit/scikit-learn/XGBoost/Joblib versions in its bundle
manifest. Do not force these incompatible stacks into one environment or treat
a skip in the core kernel as an experiment passing.

`external_artifact` tests need historical data that is not in a source export:
the original PMO Dynamic-v2.1 pre-recovery campaign tree or the hash-bound
InVirtuoGen source checkout. Tests skip only when their prerequisite file is
absent; present inputs still go through the experiment's hash and state checks.
To make missing input an error during an input-complete verification, set
`COMPOSE_REQUIRE_EXTERNAL_ASSETS=1`. Tests marked `alternate_kernel` similarly
report a missing/mismatched software stack; set
`COMPOSE_REQUIRE_ALTERNATE_KERNELS=1` in the appropriate separate environment
to require it. These flags expose missing prerequisites; they do not authorize
network downloads or oracle calls.

The frozen RingCore and T4 qualification receipts belong to earlier source
revisions. Current-checkout tests check that those receipts are **refused**
after source changes. The recorded RingCore operator revision is `a7546e2`;
the panel sampler freeze is `65e0feca`. Never update an old receipt's hashes
just to make the current checkout qualify.

## Checks

Run relevant tests during a repair. For example, the package/registry and fragment
interfaces can be checked with:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_process_v2_structural_protocol.py \
  tests/test_experiment_registry.py \
  tests/test_fragment_evidence.py
.venv/bin/python -m compose_v4.experiments.fragments verify
.venv/bin/python -m compose_v4.experiments.fragments --help
```

Before declaring a milestone complete, run the full suite and record its actual
exit status and summary:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests/ -ra --durations=20
git diff --check
```

The complete-suite result and known unresolved gates are in
[`REPOSITORY_READINESS_20260929.md`](REPOSITORY_READINESS_20260929.md).
The final clean-clone core suite at `4ca47afa` passed with 5,357 tests passed,
18 prerequisite/platform skips, and one strict expected failure for a known
Process-V2 import boundary. This is not full paper-result reproduction. The
PMO v2.1 slot-safety scan was a false positive after auditing its
occupied-topology semantics and scar-free producer domain. The ignored local
recovery directory is still not the original frozen pre-recovery snapshot;
do not silently treat its later state as that input.

On macOS installations with duplicate OpenMP runtimes, the documented local
invocation additionally sets `KMP_DUPLICATE_LIB_OK=TRUE`. Keep the one-thread
limits; suppressing the duplicate-library check alone does not establish native
runtime safety. Multiprocessing tests need a normal terminal environment with
working process and semaphore facilities.

Run Ruff on touched Python files. Do not automatically reformat frozen,
hash-pinned source files: a formatting-only change still changes their physical
identity. Preserve historical contracts and results. Tests should demonstrate
that incompatible historical artifacts are refused, not re-seal those artifacts
to make them appear current.

A historical failure count, a partial run, or a green focused subset is not a
green full suite.
