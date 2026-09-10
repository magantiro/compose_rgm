# Winner-first docking diagnostic

Authorized 2026-09-10: first dock the exact answer-known IVG endpoint from the
saved 21-edit witness, then examine the saved route. This is diagnostic oracle
validation, not model training, autonomous generation or a new optimizer result.
No executor, R_theta, option, region, task constraint or surrogate is modified.

Lock six exact saved molecules before docking: target first, four preceding
completed route stages, then the saved -11.0 incumbent. Recompute QED>=0.6,
SA<=4 and seed similarity>=0.4 with the production endpoint screen. Fail rather
than substitute any molecule. Exact persistent states validate SMILES identity;
there is no reconstruction of replay states from SMILES.

The target is evaluated first with the unchanged production PARP1 receptor,
box, Open Babel preparation, QuickVina2 exhaustiveness 1, ten modes and one CPU.
If its score is null or not below the saved incumbent score -11.0, stop after
one call and report that result. Otherwise complete the five remaining calls.
This operational screen is not a significance test or replication of the exact
published score. The one-call incumbent redock is a same-setup control, not an
estimate of replicate variability. Random seeds remain unset as in production.

All six input identities, the docking binary and receptor are hash-bound.
Reuse the existing candidate lock, per-row started/result barriers and remote
heartbeat machinery. Preserve prepared ligand and returned pose files. Never
redock a started row implicitly. No diagnostic label enters an optimizer archive
or surrogate. All failed and stopped outcomes are reported; no automatic retry.

One CPU container, 4 GiB, 1200-second ceiling, at most six new calls. Recent
production calls cost about six seconds each; allow roughly one minute docking
plus deployment/container startup. Worst-case ceiling is 1200 CPU-seconds and
4800 GiB-seconds, no GPU. Restart unit is one already-locked molecule.
Prior compatible source scores are retained as context; the target and incumbent
are explicitly redocked to answer the user's current same-pipeline question.

Use a clean committed worktree, preflight, deployed app and tools/t4_launch.py
--winner-route-docking. Only focused lock, order, stop, accounting and launcher
checks are required for this bounded development diagnostic. No broad suite.
