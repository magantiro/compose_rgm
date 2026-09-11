# Direct winner-trajectory imitation: reconstruction only

User authorization (2026-09-11): directly learn from verified winner trajectories
and attempt reconstruction in a short development loop. This trains a separate
controller proposal ranker, never R_theta. No docking, PMO calls, Modal jobs or
production-controller deployment are authorized by this protocol.

Problem: can a learned state-conditioned edit ranker execute the demonstrated
long routes without being fed the next action? Output: scores over valid,
goal-directed primitive proposals and closed-loop molecular trajectories.
Claim tested: training reconstruction capacity on five answer-known development
routes. All five routes are explicitly training/reconstruction data; there is
no held-out result and no generalization claim. Compiler-generated paths are
synthetic demonstrations, not IVG's actual optimization trajectories.

Reuse all five saved exact-slot witnesses and their stored source-to-goal
correspondence. The goal and its slot alignment are privileged inputs. Existing
difference-directed proposal generation supplies all its valid alternatives;
do not inject a missing teacher candidate. Deduplicate identical augmented
successors. Train against the witnessed next augmented successor. Input consists
of exact graph coordinates, goal residual, proposed edit displacement and pending
retirements. A two-layer 128-wide MLP learns the rank, with Adam at 0.003.
Sample a demonstration family uniformly, then a route uniformly among that
family's routes, then a row. This balances the large delete/insert majority.

Use seed 20260919, float32 CPU, one Torch/BLAS thread. Prepare and hash the
candidate panels before fitting. Fit at most 600 updates of 16 rows, checking
training successor ranks every 50 updates; stop at the first 100% panel recovery
or 120 seconds of fitting. Store model, optimizer, RNG, configuration and input
identities. This is a training-capacity stopping rule, not validation selection.

Run one greedy, closed-loop reconstruction per root with at most 64 primitive
steps, alongside its initial untrained network on the identical proposal generator.
No next-action or route-index lookup, teacher forcing, planner fallback, endpoint
injection, or checkpoint reselection after rollouts. Count failures and cycles.
Replay every successful complete trajectory through the unchanged executor.
Report candidate coverage, family-specific ranking and all five endpoints.

Support remains the frozen 1..40 active / 48-slot, charge-preserving, achiral
grammar. This proposal channel is restricted by a supplied goal correspondence;
it is not the full R_theta law, a new exact Doob transform, or a change to kappa.
Production generic/local-global channels are untouched. Integration as a learned
option proposal and target-free task optimization are subsequent milestones.
The diagnostic's coordinate-based representation carries no permutation-invariance
claim. Passing it does not mean that the controller can discover an unknown winner.

Run only focused tests and artifact checks during this bounded experiment. Full
repository verification remains required for final controller qualification.
