# Metered lookahead without executor/state cutoffs

2026-09-09. The user explicitly authorized removal of the self-imposed planning
compute cutoff and testing whether completed lookahead produces useful edits.
This prospectively amends the compute policy in HIERARCHICAL_TASK_SEARCH.md and
T4_LAZY_REFERENCE_REPAIR.md only for the new run below. Their earlier capped
contracts, negative results and scientific acceptance thresholds remain frozen.

The problem, model output and molecular support remain those in
HIERARCHICAL_TASK_SEARCH.md: executable molecular trajectories with task-value
guidance across WHERE, WHAT and HOW. The narrow question is whether removing
administrative search cutoffs permits full-horizon returns, non-reference
decisions and feasible structural edits. This is not evidence of better docking
or superiority to another generator. The immediate baseline is the saved
same-parent lazy-reference probe at a914195, with zero completed returns.
A same-generator post-hoc comparison remains required for a causal performance claim.

## Changed compute policy

The opt-in `metered_uncapped_v1` policy sets planning executor calls, total
executor calls per parent (including the option kernel), and cached search-state
limits to `null`. Every public executor attempt is still counted and recorded.
The legacy bounded configuration and serialized identities remain compatible.
Do not replace 512 with another small hidden ceiling.

Keep the 16-primitive terminal horizon, 32 attempted rollouts per parent,
initial 8 rollouts and 2 per subsequent decision, all priors, exploration floors,
kappa=1, R_theta, executor, support, option menu, predictor and feasibility fixed.
The 256 terminal-evaluation allocation is above the 32-rollout allocation.
These finite horizon/sample allocations define the algorithm being compared;
they are distinct from the removed validation-call/state cutoffs and from
expensive docking calls. Interrupted simulations are never terminal observations.
No winner-derived target, motif reward, template or selection is introduced.

## Useful first execution

Use the same first-ranked parent from the existing exact 51-call archive and
frozen predictor check, recorded in configs/t4_uncapped_lookahead_probe.json.
One CPU, 8 GiB, one worker, no retries, no new docking calls. This is one complete
parent preparation, not a new eight-parent audit or optimization campaign.
The prior probe spent 99.33 seconds on 14 completed law enumerations plus one
interrupted enumeration. At up to 32 x 16 sampled primitive states, a rough
planning estimate is 15-90 minutes including shared work and committed decisions;
this is an extrapolation, not demonstrated tractability. Use a two-hour
administrative timeout to prevent an unattended runaway, record a timeout as
incomplete, and do not turn it into a terminal reward. Maximum resource exposure
is two CPU-hours and sixteen GiB-hours; current dollar rate is unverified.

Reuse compatible completed parent artifacts. A new policy changes the scientific
preparation identity, so the old capped parent is a baseline, not a resume point.
Persist law/rollout progress, actual executor counts, completed returns, terminal
values, decisions, exact paths, candidates and elapsed time. Completed parents
are the restart unit; an interrupted started parent is not silently rerun.
Replay only selected edits independently. Do not repeat deterministic molecular
enumeration to generate a second report.

Report completed versus interrupted rollouts, nonzero terminal values, value
contrast and total-variation changes at each hierarchy level, chosen options,
intended versus realized structural change, cycle-rank and ring-system changes,
feasibility, canonical diversity and measured time. One parent cannot establish
the existing multi-parent gate; that gate is neither passed nor waived here.
No automatic docking follows. Fresh docking of locked candidates is the next
separate question if this produces useful candidates.

## Minimal verification and launch

Check uncapped metering above the old call ceiling, search past the old state
ceiling, legacy cap enforcement/config identities, unchanged reference laws,
exact replay, zero-oracle accounting and completed-parent reuse. Run focused
dependency tests, lint, formatting and preflight in a clean committed worktree.

`modal deploy modal_apps/genmol_t4_opt_app.py`, followed by
`python3 tools/t4_launch.py --uncapped-lookahead-probe`, is the only new launch.
No training or broader milestone is authorized by this compute amendment.
