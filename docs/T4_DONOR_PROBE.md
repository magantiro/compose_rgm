# Parallel T4 transfer probe, 2026-09-11

Current question: does the unchanged donor/reference mixture that improved PMO
produce useful T4 proposals, compared with the broad reference option law from
identical parents? This is a single warm development batch, not a full optimizer
or a comparison to the workshop's multi-cell table.

Use PARP1 seed0, delta=0.4, QED>=0.6 and SA<=4. Benchmark feasibility and the
existing medicinal-chemistry screen apply at oracle allocation only. Every
primitive intermediate passes the unchanged valid-state executor, not the
endpoint similarity constraint. Reference model weights remain frozen.

Source: the completed broader controller's 134-call archive, containing 135
exact-state rows including the undocked seed. Its best observed score was -11.0;
that molecule separately redocked at -9.8. Do not import public-winner route
endpoints or their diagnostic labels. Both arms start with the original seed
and seven highest-scoring distinct archive molecules, each represented twice.
Donors are the first 100 distinct scored archive molecules, ordered by observed
score and canonical identity. This is an exposed warm-start regime, not a
fresh-budget benchmark. Preserve and disclose all 134 historical attempts.

Proposal recipe: 16 slots per arm, one whole option per slot, 64-primitive cap.
Baseline uses the current broad reference options including generic, carbonyl
and region replacement. Hybrid is the same 50/50 reference/donor mixture tested
on PMO, including its uniform donors/cuts and compiler limits. Failed programs
are recorded without replacement. No new training, learned guide, target-winner
similarity or docking predictor influences proposal or endpoint selection.

Lock all 32 proposal attempts before docking. Among eligible, novel canonical
endpoints, take at most eight per arm in slot order. The first eight slots cover
all eight parents before any second slot, so raw frontier size creates no extra
outer allocation probability. Deduplicate docking across arms and retain both
origins. Add two independent redockings each of the fixed seed and source
incumbent. Maximum 20 new docking attempts including those four controls.
Failed docking calls remain charged; no implicit repeat. An insufficient eligible
pool is a result, not a reason to sample more proposals after observing outcomes.

Dock through the existing production Open Babel/QuickVina2 implementation,
unchanged receptor/grid/binary, exhaustiveness=1, ten modes, one CPU. Bind binary
and receptor hashes before spending calls. Random seeds are not explicitly set
in this inherited pipeline, so report repeat spread and do not treat a small
best-score difference as resolved superiority.

Compute: at most 14 shared workers plus one driver, which together with the PMO
replication's 15-container cap respects the user's total 30-container ceiling.
One CPU per container, no GPU, no retries; 540-second worker and 1200-second
driver limits. At most 52 tasks including all proposal and docking slots;
worst-case CPU reservation 8.14 hours, $5 reservation cap. Expected 3–10 minutes,
with explicit timeout/failure reporting. Reuse compatible exact-state neural laws
from completed PMO work through input and implementation dependency checks.

Decision: a clear feasible score gain beyond the observed control spread, with
reasonable proposal time, earns a short round-synchronous continuation test.
A smaller difference is inconclusive and requires matched repeat evidence before
claiming improvement. No new feasible endpoints, poor scores, or dominant
compiler failure stops this transfer recipe; diagnose that measured bottleneck
instead of launching a full panel. Compare coverage, best/median score,
canonical diversity, intended/realized change, ring deltas and proposal time.
This test does not establish an exact Doob law, learned future value, universal
chemistry, or external SOTA.
