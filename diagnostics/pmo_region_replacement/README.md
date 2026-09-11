# Region replacement: implemented, not remotely evaluated

The opt-in region-replacement channel, exact-state codec, hierarchy applicability
and explicit primitive clock are implemented in the isolated development tree.
The existing WHERE selector and generic channel are retained. See
`docs/PMO_REGION_REPLACEMENT.md` for the proposal semantics and prospective audit.

Local engineering verification on 2026-09-11: 31 tests passed in 8.61 seconds
using the pinned RDKit overlay, including real-executor ring construction at
40 atoms, protected context, slot reuse, lazy/eager agreement, resume and oracle
candidate locking. Command:

```
env PYTHONPATH=/private/tmp/compose-t4-chemistry.hizM8Y:src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Users/rmaganti/compose_rgm_git/.venv/bin/python -m pytest -q tests/test_region_replacement.py tests/test_region_replacement_probe.py tests/test_option_particle_driver.py tests/test_pmo_archive_pilot.py tests/test_option_continuation.py tests/test_ring_program.py
```

An earlier broader focused invocation returned 35 passed, one failed:
`tests/test_branch_policy.py::test_launch_checks_frozen_process_before_remote_allocation`
raised `EditingV2ProcessIdentityError` against its historical bound process
identity. That gate was not weakened or rebound. Local neural qualification is
not claimed; the production runtime has its separate remote qualification.
The remaining selected tests were not rerun as a repository-wide suite.

Ruff checks/formatting for touched code and `git diff --check` passed. Preflight
reported the uncommitted serialized-source changes. Deployment completed in
80.779 seconds to `compose-pmo-region-replacement`, with 20 workers maximum and
one driver, but no worker or driver was launched.

The subsequent launch command was rejected by auto-review because the newly
provided AGENTS.md requires clean committed code. Earlier approval and
`docs/PMO_ROUTE_SUPPORT_DEV_SNAPSHOT.md` permit uncommitted bounded snapshots,
but that conflicts with the newer instructions. Do not reroute or retry the
launch to bypass the rejection. Resolve with the user, preferably by approving
an intentional checkpoint commit and deploying that clean exact revision.

No replacement-probe PMO score, remote timing, run ID, or result exists yet.
The 20-call audit contract is prepared, not satisfied. The report producer
`tools/pmo_replacement_report.py` is ready but has not run on scientific output.
The overall IVG/PMO goal remains unmet.

The user subsequently answered "yes" to the explicit checkpoint-and-audit
request. The replacement launcher now enforces a clean exact commit using the
existing source-revision validator. This resolves the authority conflict without
waiving the clean-source gate. Commit/deployment/run receipts will identify the
actual launched revision; the earlier rejected invocation remains unlaunched.
