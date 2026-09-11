# Replacement executes large rewrites; overall best unchanged

Run `1efb450599f9999cedbb79d65b2e9c2ee296a4c84dd0e6d29975171f0269b032`
completed from clean commit `354677e96d35e5085534859b4ec555f59560484b`.
Twenty normal hierarchical proposal draws selected replacement five times;
four completed. Sixteen of all twenty options completed, producing sixteen
distinct candidates and consuming sixteen new deterministic PMO evaluations.
The best retained score remains **0.522233**. This is a warm development
proposal audit, not an optimizer comparison or PMO AUC benchmark.

| Replacement | Primitive edits | Atoms deleted/inserted | Heavy atoms | Parent → child score |
| --- | ---: | ---: | ---: | ---: |
| Six-membered C/O ring on incumbent | 20 | 12 / 6 | 40 → 34 | 0.522233 → 0.171525 |
| Five-membered aromatic C/N ring | 25 | 17 / 5 | 24 → 12 | 0.002181 → 0.097129 |
| Six-membered aromatic C/N ring | 22 | 13 / 6 | 24 → 17 | 0.002181 → 0.184900 |
| Six-membered C/O ring on fourth original root | 10 | 2 / 6 | 24 → 28 | 0.084082 → 0.366508 |

Three of four completed replacements improved their parent, but none beat the
incumbent. The channel can therefore perform substantial multi-edit replacement
under the ordinary balanced prior. This does not yet demonstrate a competitive
search controller. Generic was selected four times and remained active.

Intended release fractions for these four replacements were 0.300, 0.708, 0.542
and 0.083. Realized largest changed-component fractions were 0.175, 0.250,
0.250 and 0.292. That existing realized metric counts changed surviving and
inserted atoms, not removed atoms; the deletion counts above are essential.
Ring-system deltas were 0, -1, -1, 0 and cycle-rank deltas 0, -2, -2, 0.
A zero net ring delta does not mean no ring chemistry: these are replacements.
Mean pairwise Morgan distance over sixteen distinct products was 0.820252.

The failed replacement stopped in pruning after seventeen deletions and two
cycle openings, before construction. Its three remaining positive-probability
deletion marks all pass the executor but produce sulfur bearing hydrogens with
three heavy neighbors, rejected by the unchanged pathwise sulfur gate. The
nineteen-step trace was independently replayed and the saved production row
rechecked locally, without a new model/oracle call. This is a concrete
pruning-recipe/domain limitation, not evidence that every possible supported
path is impossible. The other failures were two `build_ring_system` programs
and one fused construction. No gate was relaxed and no failure was retried.

The driver recorded 239.59 seconds before final result publication; clean
deployment took another 81.26 seconds. The sixteen oracle calculations took
0.021 seconds. Eleven driver commits accumulated 234.52 seconds including
overlap with proposal work and heartbeat activity, so these times must not be
added as disjoint wall-clock components. Persistence is substantial overhead.
Individual replacement proposals took 23.28–102.07 seconds when completed.

`report.json` binds the source archive, input hashes, exact endpoint geometry,
actual score ledger and failed-trace diagnosis. Full result and source archive
are under `/private/tmp/compose-pmo-region-replacement-runs/<run>/`; remote
artifacts are `pmo_region_replacement/<run>/` on `compose-v4-artifacts`.

Next decision: retain the optional replacement channel and test repeated
score-informed continuation from its completed products, with an ordinary
proposal baseline. Do not call the old achieved-route head qualified foresight:
the additional saved-data audit is in
`diagnostics/pmo_option_particles/GUIDE_AUDIT.md`. No larger run has been launched.

## Implementation and launch history

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

At the time of the rejection no replacement-probe score, remote timing, run ID
or result existed. The overall IVG/PMO goal remains unmet after the completed
audit above. The report producer has now verified the actual output. Its first
attempt preceded local result download and correctly failed without publishing
a report; it was rerun only after the existing call completed.

The user subsequently answered "yes" to the explicit checkpoint-and-audit
request. The replacement launcher now enforces a clean exact commit using the
existing source-revision validator. This resolves the authority conflict without
waiving the clean-source gate. Commit/deployment/run receipts will identify the
actual launched revision; the earlier rejected invocation remains unlaunched.
