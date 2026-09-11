# Replacement persists across rounds; best-score null

Run `72e39c6079e3c06b10122a57661e8f0e529f876277a83fe16e45ec82a3962bdb`
completed from clean commit `35af8bd9adca` using 47 new PMO oracle evaluations
and 53 unique workers. Driver elapsed time was 417.83 seconds before final
publication; deployment took another 79.338 seconds. The initial and final best
score is **0.522233 in both arms**. No IVG improvement has been established.

| Arm | Best | Top-ten mean | Queried archive | Completed / attempted options |
| --- | ---: | ---: | ---: | ---: |
| Reference | 0.522233 | 0.423474 | 39 | 31 / 33 |
| Actual-score SMC | 0.522233 | 0.497514 | 42 | 34 / 35 |

These are warm-development archive summaries, not PMO AUC. Both arms have nine
initial particles and four option opportunities per particle. Failures reduce
subsequent attempts, so realized work and archive sizes differ. Shared exact
tasks are computed once; the 47 calls are the unique union across arms, not 47
calls per arm. Initial molecules and historical labels are exposed development
assets. This single run supports no significance or matched IVG claim.

## Chemistry and survival

Each arm completed five replacement programs. The score-guided arm also
completed three other ring-construction programs, including two fused rings.
Replacement therefore remains executable after score-based resampling, unlike
the earlier size-saturated trajectory behavior. Its longest single replacement
used 31 primitive edits, deleting 24 atoms and inserting five; every transition
was executed and replayed under unchanged support. Maximum new primitive depth
from a starting particle is 36 in both arms, with medians seven and four for
reference and score-guided candidates respectively.

The first shared fused-ring construction improved a replacement product from
0.097129 to 0.268462, giving `c1ccc(Cc2ccn3c2CCCC3)cc1`. This lineage survived
initial score resampling. The guided arm retained four starting ancestries after
round one and three at the end, rather than collapsing to one start. Six of its
eight final live particles nevertheless descend from the incumbent. After round
one, 16 of 25 completed guided options worsened their parent; four improved and
five tied. Scores are compared exactly in that diagnostic; tiny floating
differences are not claimed as meaningful improvements.

The incumbent-descended 31-step replacement reduced score from 0.513701 to
0.362738. It is constructive chemistry, not an optimization success. Only one
of five guided replacements improved its immediate parent, versus three of
five reference replacements. No replacement exceeded the incumbent.

Across completed candidates, intended release fractions span 0.025–0.889 in
both arms. Median realized coherent change is 0.111 for reference and 0.075 for
guided, with maxima 0.636 and 0.500. As before, this metric excludes deletions,
which remain separately recorded. Guided cycle-rank deltas are -2:1, -1:6,
0:23, +1:4; ring-system deltas are -2:1, -1:4, 0:28, +1:1. Mean pairwise Morgan
distance over each queried archive is 0.8361 and 0.7856 respectively.

Three distinct worker attempts failed: one generic draw, one legacy
`build_fused_ring`, and one legacy `build_ring_system`. The first two belong to
the reference arm, the last to the guided arm. They were not retried, and later
empty particle slots are not counted as additional failures.

## Compute and verification

Actual oracle arithmetic took 0.0805 seconds. Proposal wall times by round were
38.47, 110.08, 97.29 and 125.17 seconds (371.01 total). Worker elapsed times sum
to 1,013.13 seconds; reference-law computation sums to 740.56 seconds across
workers. Driver persistence records 30 commits totaling 63.37 seconds, partly
overlapping proposals. Do not add these as disjoint wall-clock costs. This is
not a matched before/after speed comparison: the prior runs differed in work
and commit latency. The duplicate-barrier repair has its focused fixture, but
the full change in remote timing cannot be attributed to that repair alone.

The report verifies serialized source bytes, contract and prepared-input hashes,
actual oracle receipts, query uniqueness and budget, complete queried archives,
and particle arithmetic/ancestry. Numerical particle replay error is zero here.
The producer records exact primitive replay; local analysis does not requalify
the neural model. The survival audit checks each actual parent score and chain.

The first report exposed an analysis defect: selecting a historical depth by
common chain prefix could choose a different starting particle and produce a
negative depth. New depth now accumulates exact per-option counts along saved
resampling ancestry. Two focused regression tests passed in 1.54 seconds;
Ruff and diff checks passed. No proposal law, oracle value or trajectory changed.
`report_initial_depth_error.json` and `survival_audit_initial_report.json` retain
the superseded analysis and must not support depth claims. `report.json` and
`survival_audit.json` are the corrected reports. Reanalysis of the older SMC run
is saved as `../pmo_option_particles/report_depth_corrected.json`; its original
headline depth maxima of 19/14/19 are unchanged.

The user explicitly approved the previously blocked upload/run with "go".
Deployment and durable call `fc-01M28FBKTR577P69M8HQY1AWT3` then completed. Full
raw result, snapshots and spawn receipt are under
`/private/tmp/compose-pmo-replacement-continuation-runs/<run>/`; remote artifacts
are `pmo_replacement_continuation/<run>/` on `compose-v4-artifacts`.

Decision: replacement capability and continued multi-edit execution are now
demonstrated, but fixed proposals plus actual-score resampling have not improved
the incumbent. Do not treat this as justification to launch a larger identical
campaign. The next controller intervention should test state-specific proposal
improvement and protection of useful exploratory continuations against this
same-generator baseline. Its benefit is proposed, not established here. No new
training or broader benchmark run was launched.

## Prelaunch record

Prospective recipe: `docs/PMO_REPLACEMENT_CONTINUATION.md` and the self-hashed
`configs/pmo_replacement_continuation.json`. The prepared artifact contains all
five preceding warm roots and all four completed replacements, with exact
slot-addressed state and historical label provenance. No new oracle calls were
used for preparation.

Local engineering verification: 21 focused tests passed in 8.16 seconds on
2026-09-11, pinned RDKit overlay, local Python 3.12. These cover old/new particle
arms, shared-task identity, archive completeness, resume, oracle candidate locks
and charging, replacement execution, and duplicate/concurrent persistence.
Command:

```
env PYTHONPATH=/private/tmp/compose-t4-chemistry.hizM8Y:src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Users/rmaganti/compose_rgm_git/.venv/bin/python -m pytest -q tests/test_pmo_store.py tests/test_option_particle_driver.py tests/test_pmo_archive_pilot.py tests/test_pmo_locked_batch.py tests/test_region_replacement_probe.py tests/test_region_replacement.py
```

Ruff and `git diff --check` passed for the touched code. No repository-wide
suite or new local neural qualification is claimed. The existing historical
branch-policy process-identity test failure remains recorded in the preceding
replacement report; its gate was not changed. The qualified production runtime
will enforce its own exact input and numerical identities remotely.

The persistence fixture confirms one barrier instead of two for an overdue
immutable save, and no second periodic commit after a waiting heartbeat observes
a completed barrier. This is an engineering count, not a remote timing result.
Both mandatory oracle barriers remain unconditional. The deployment and run
were pending when this prelaunch record was written; the completed result is
documented above.
