# Bounded production audit of hierarchical task search

2026-09-08. Authorized by the user's "great do it" after the three-step
proposal-audit and matched-comparison plan. This is a new development episode
on the inspected PARP1 seed0 d=0.4 cell, not a resumed optimizer trajectory.
The scientific identity, support and method are those in
HIERARCHICAL_TASK_SEARCH.md. No generator, executor, primitive support,
predictor recipe, exploration floor or KL budget changes belong to this audit.

The question is whether the implemented terminal-value planner supplies useful
evidence to real WHERE/WHAT/HOW decisions within its declared compute budget.
Production outputs are exact executable graph paths and completed-option
candidates. Ring count is a diagnostic, never an objective or admission rule.
The saved chronological predictor check is reused without refitting its recipe.

Run one prepare-only audit from the exact 51-call archive. Freeze eight parents,
16 primitive edits per parent, 2,500 public executor calls per parent, at most
512 of those calls for planning, and the existing row/rollout/terminal bounds.
No docking callback is available to this run. A separate verifier replays at
most 128 selected primitive steps, with its own 2,048-public-call ceiling. This
verification work is reported separately from search work. The verifier checks
exact path continuity, option boundaries, probabilities/floors/KL, valid states,
lineage, completion witnesses, candidate identity and endpoint feasibility.
It does not re-enumerate every reference row or claim an independent proof of
the learned probability law.

Every completed parent is an immutable restart unit, including candidates,
exact trace, executor attempts, objective identity and RNG state. Reusing it
must consume no new generator or executor work. An unfinished started parent
blocks automatic replay because its cost may be unknown after a hard kill.
Heartbeat publication continues during preparation; no automatic retries.

Before observing this run, fix a minimal engineering routing rule: paths and
budgets must verify, at least one planning rollout must complete, each of WHERE,
WHAT and HOW must contain a non-reference committed decision (total variation
greater than 1e-8), and at least two feasible new candidates from distinct
parent lineages must exist. Passing only permits a small matched comparison;
it is not a performance claim. Report counts and denominators, including failures,
all-reference decisions and unrealized large regions. Never weaken this rule
after inspecting the audit. Failure routes a narrow bottleneck diagnosis.

If it passes, freeze a same-generator post-hoc comparison before its preparation
or docking. Match primitive and oracle allowances, report executor and wall-time
differences, and lock both arms before fresh labels. The immediate-task-value
ablation remains part of later causal analysis, not an extra prerequisite to
this first bounded production audit. At most 20 docking slots per arm may be
proposed; this preparation contract itself authorizes zero.

Operational estimate: one CPU, 8 GiB, one sequential worker, approximately
10-30 minutes based on the prior 679.5-second proposal round, not a measured
timing for this planner. Hard timeout is 3,600 seconds. No accelerator allocation.
Expected cost is at most one CPU/8-GiB container-hour; a current dollar rate has
not been verified. The first production audit itself is the reusable benchmark.
