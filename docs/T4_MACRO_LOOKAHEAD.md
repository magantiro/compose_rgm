# Bounded macro-continuation feedback test

Authorized by the user's request to implement the saved-run diagnosis and test
for improvement. This is a new inspected-cell development episode, not a change
to the completed feedback episode or the editing-model training ladder.

Problem: allocate search to useful executable multi-option continuations rather
than repeatedly selecting novel but unproductive states. Output: complete valid
molecules with exact primitive witnesses and parent/region/option ancestry.
Hypothesis: short completed-option lookahead plus endpoint-aware parent allocation
improves eligible proposal yield and best docking. Baseline: the completed
`t4_macro_feedback` episode, -10.6 after 27 new calls on 67 prior calls. This
episode starts with all 94 prior calls and the saved pool, so its outcome is a
warm continuation, not a matched causal comparison. The old post-hoc worker
remains runnable; no superiority or exact Doob claim is licensed by one episode.

Preserve the executor, exact persistent states, 40-atom bound, charge policy,
achiral vocabulary, R_theta, kappa=1, WHERE prior, applicability-balanced WHAT
prior, generic channel and existing ring/carbonyl programs. No winner, target
path, winner-derived fragments, new whole-ring vocabulary or Q(o) training.
All task gates remain endpoint-only: QED>=0.6, SA<=4, seed similarity>=0.4,
and the inherited endpoint medchem screen. Valid but infeasible intermediates
remain supported. No reference-model training or support changes.

## Frozen development recipe

Four synchronous rounds. Eight parent roles, two independent workers per parent,
16 proposal containers. Each worker executes at most six complete options:

1. Sample two root options from the unchanged WHERE/WHAT/HOW reference.
2. Give each completed first option one further option, without task pruning.
3. Group each first option with its completed continuation. Value the branch by
   its best observed completed endpoint under the round-frozen task surrogate.
   Ineligible states have zero terminal value; if neither branch has an eligible
   state, use exp(-minimum endpoint deficit/0.1) as an explicitly heuristic repair
   fallback. This is not a predicted recovery probability.
4. Select a branch with the existing KL=1 tilt and a 0.1 uniform exploration
   mixture over sampled branches. Continue from its best-valued completed state;
   with probability 0.1 continue from a uniformly selected state in the branch
   instead, including ineligible states. Execute two further options there.

All complete first/second/third-stage states enter the cumulative canonical pool.
No primitive intermediate is forced to satisfy endpoint constraints. Branch
values and selection are recorded before the final two attempts, which provides
an auditable in-loop task-guidance boundary. Values are maxima over sampled
completed states with a stopping option, not unbiased expectations or exact
finite-horizon h values. R_theta draws are unchanged; the finite sampled search
allocation is a different controller, not the original molecular reference law.

Use all strictly prior docking labels. For already measured molecules, planning
uses their observed score; otherwise it uses the unchanged DockingValue recipe.
This is an uncalibrated ranking heuristic. No pessimistic/optimistic confidence
bound is claimed. No score from the current docking batch enters its proposals.

Next-parent roles: original seed, best observed, two quality-ranked feasible
parents, two structurally diverse feasible parents, one nearest-feasible repair
parent, one uniform unrestricted parent. Previously unexpanded feasible parents
are preferred within diversity roles, counting expansions in this new episode.
This encourages coverage but does not guarantee that every feasible branch gets
a continuation in four rounds. Repair prioritizes fewer previous
expansions then lower endpoint deficit. There is no hard intermediate mask;
an unrestricted draw retains broad exploration. Missing role pools fall back
to uniform selection from remaining exact states. Every role is audited.

At most ten new dockings per round, forty total. Alternate predicted-quality
and canonical structural diversity, reserving every fifth slot for uniform
exploration. Never reject an eligible candidate by predicted docking threshold;
carry pending candidates between rounds. Count all attempted calls, including
failures, and never automatically redock a started row. All labels update only
after the locked batch finishes. Stop after four rounds, no automatic extension.

## Cost, persistence and acceptance

At most 384 macro attempts, versus 160 in the previous episode. The previous
4402 worker-seconds / 157 completions suggests about three CPU-hours and 24
GiB-hours of proposal work before cache reuse. With 16 workers, initial estimate is 25-50 minutes
elapsed including startup, stragglers and docking; cost is reported in actual
container-seconds, not assumed from wall time. Driver ceiling 7200 seconds,
worker ceiling 1800 seconds, no retries. No GPU. Existing exact-law caches are
reused only when model/executor/codec dependencies and frozen input hashes match.

Reuse completed subsearches on restart. Persist sampled primitive steps, RNG,
branch decisions, round locks, prior-value snapshots, worker costs and heartbeat.
Initial descendants keep their original remaining horizon, never reset to 110.
Report attempts, completion, eligible yield, recovery yield, represented options,
intended/realized changes, oracle utilization, score and time per round.

Focused checks: delayed-value branch selection with an ineligible intermediate,
unrestricted exploration, exact ancestry/budget, once-only candidate accounting,
prior-only model snapshots, cache containment and clean launch bounds. No
unrelated full suite is a launch gate for this bounded T4 development test.
Publish negative results as returned; improvement is not an acceptance gate.

Verification before launch: ten focused tests passed in 7.86 seconds, plus the
new launcher and existing once-only docking-row check passed in 2.74 seconds.
No repository-wide suite was run. Touched modules, tests and launcher pass Ruff;
the legacy app has 17 existing lint findings, verified unchanged by code/message/
source-line comparison against the source commit. New app functions and CLI
argument formatting were checked without reformatting unrelated legacy code.
