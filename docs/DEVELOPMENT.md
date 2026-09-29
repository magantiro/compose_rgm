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

The ongoing readiness repair and its measured results are recorded in
[`REPOSITORY_READINESS_20260929.md`](REPOSITORY_READINESS_20260929.md). A historical
failure count, a partial run, or a green focused subset is not a green full suite.
