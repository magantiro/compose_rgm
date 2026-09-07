# Continuation controller: bounded implementation contract

Date: 2026-09-07. Authority: the user's request to implement the hierarchical
controller rigorously and identify the cheapest informative evaluation.

## Identity and scope

COMPOSE's output remains an executable marked molecular rewrite process. This
milestone implements a budget-indexed continuation controller over that process,
not a new generator. The testable mechanism is whether downstream terminal
value can change an earlier transition across a delayed-reward plateau while
preserving legal support, option phase, context, and the frozen KL constraint.

Production support remains the declared broad-organic vocabulary, at most 40
active atoms, charge-preserving connected complete molecular states, and no
stereochemical claim. Small test graphs are explicitly restricted engineering
fixtures, not evidence for full molecular support or the learned generator.

Frozen: R_theta, Q(M), balanced untrained Q(o), generic availability, kappa=1,
the executor, macro contracts and programs, and all benchmark thresholds.
The existing audited T4 path remains the baseline. No MCTS, new Q(o) model,
checkpoint training, winner-template library, or paid docking launch belongs
to this implementation gate. A new continuation policy is a declared ablation,
not a purported decision-equivalent speedup of structural guidance.

## Algorithm

Use the smallest exact finite-horizon backup first. For the option-conditioned
reference R, h_0(s)=G(s) and h_b(s)=sum R(s'|s) h_(b-1)(s'). R is supplied by
the existing production marked-law evaluator and option support machinery;
this module does not infer molecular successor probabilities from mark counts.
G is a caller-supplied bounded terminal weight, never a winner-similarity score.
All downstream branches in a completed backup are included. Cache by exact
augmented state and remaining budget. Different program phases or atom lineage
must not merge. Empty support is explicit unsuccessful termination.

Apply the existing KL tilt to h, then the frozen primitive exploration mixture.
Report the actual KL of the mixture. Keep the exact Doob transform separate
from this approximate capped, floor-regularized controller. All-zero exact
terminal mass and an incomplete computational budget are distinct statuses;
either returns the declared baseline for a sampled decision, not an invented
successful continuation or a partial-tree ranking. Report all failed work.

This exact bounded solver is an executable specification and small-problem
reference, not a claim that full molecule-scale tree enumeration is efficient.
It provides a target against which a later sparse/Monte Carlo estimator can be
qualified, without first choosing MCTS or training another value model.

## Acceptance criteria before inspecting results

1. Match independent direct path summation on a finite declared reference graph;
   preserve terminal weighting and remaining-budget dependence within 1e-10.
2. Demonstrate a delayed-reward decision that immediate endpoint guidance cannot
   distinguish. This is an algebra/implementation test, not a docking result.
3. Preserve zero-probability reference support, normalized probabilities, and
   per-step KL <= 1 + 1e-10 after exploration mixing.
4. Reject malformed distributions, nonfinite values, invalid horizons, and
   state-identity omissions at typed boundaries.
5. Distinguish zero terminal mass, exhausted compute, and no admissible action.
   A budget-limited decision must equal its baseline, never a partial ranking.
6. Cached and uncached results must agree. Explicitly test slot permutations,
   formal charges/hydrogens, option phase, lineage, and remaining horizon.
7. Execute a small chemistry fixture through the production executor and existing
   macro restrictions. Verify every committed state and its conditional path
   probability. Do not reconstruct intermediate states from SMILES.
8. Record reference expansions, executed candidates, terminal evaluations,
   cache hits, wall time, and all rejection/abstention statuses.
9. Produce a reproducible local receipt with input/code hashes and no external
   oracles. Run focused checks while iterating and the full suite once at the
   frozen implementation boundary; retain non-green results honestly.

## Evaluation ladder and use of winners

First run the local mathematical and executor integration gates (zero docking,
no learned-model performance claim). Then profile one reusable production
bundle with the actual frozen model and snapshot its complete input identities.
Do not extrapolate runtime from the tiny fixtures.

Before a chemistry-performance experiment, freeze a separately sourced,
scaffold-separated development panel and balanced broad capability tasks.
Compare the same-generator post-hoc baseline, immediate-value guidance, current
structural guidance, and continuation guidance. For the compute-efficiency
comparison use one declared binding resource (executor applications); report
wall time and all other calls. A safety timeout yields an incomplete result,
not a resource-matched score. For a later oracle-efficiency comparison match
oracle calls and explicitly report unmatched proposal compute.

The previous 20-call T4 audit is not a matched comparator for a different
algorithm/budget. A single first round also cannot establish online docking
learning when the task surrogate has no labels yet. Before testing task-aware
lookahead, freeze a shared, counted warm-start batch or independently qualified
development-only value snapshot and its calibration test. No silent historical
winner labels or uncounted prescreen labels are allowed.

IVG winners may be used for endpoint description, oracle-pipeline calibration,
or a separately labeled target-given executor diagnostic. They must not enter
G, reusable fragments/templates, option weights, checkpoint training, or the
independent discovery panel. This inspected T4 cell is development evidence.
Final generalization needs an untouched panel. No benchmark superiority follows
from the local gate, path existence, or architectural validity.

## 2026-09-07 repair and sampled-estimator milestone

Authority: the user's request to fix budget handling and test a principled,
efficient controller. This extends the engineering lane, not training authority
or permission to weaken any production gate. The platform identity and declared
support above remain unchanged. No T4 replacement is authorized by fixture tests.

1. Repair the diagnostic's cancellation boundary. A computational stop must
   escape legacy chemical-rejection handlers without changing executor code.
   Count and persist every entered public executor call, including interrupted
   enclosing calls. Reproduce this through actual ring-restate lowering.
2. Add a fixed-sample Monte Carlo reference-continuation estimator. For every
   positive-probability root successor, average K complete reference suffixes
   to the original terminal horizon. Sample transitions from full reference
   rows, without top-k, depth truncation, or winner-derived weights. Cache exact
   rows, not sampled outcomes. This is linear in K, root support, and horizon
   in row visits; molecular enumeration within each row can still be expensive.
3. Freeze K and a simultaneous Hoeffding confidence level before each comparison.
   Apply guidance only when complete sampling separates at least two supported
   continuation values. Use conservative lower bounds as tilt weights. A
   completed all-zero sample is not proof of unreachability. Overlapping bounds,
   insufficient budget, and empty support return distinct baseline/abstention
   statuses. Never use an interrupted partial comparison to guide a decision.
4. Preserve the existing KL tilt (kappa=1), exploration, option support, complete
   molecular executor states, and outer bundle identity. Monte Carlo randomness
   and committed-path randomness must use separate recorded streams. Conditional
   decision probabilities are not marginal path probabilities after integrating
   out the estimator's randomness.
5. Require focused tests of nested cancellation, honest receipts, fixed sample
   counts, analytic bounded-reference agreement, confidence coverage and null
   abstention, reference support, KL, determinism, and actual executor integration.
   Numerical fixture results must carry input hashes and explicit synthetic roles.

Existing production profile outputs must be inventoried and preserved before a
new cloud run. Do not repeat the exhausted exact tree as an efficiency test.
A full learned-model chemistry comparison still requires the separate frozen
development panel and matched baselines specified above. This milestone can
finish its code repairs while leaving those scientific outcomes unqualified.

For m distinct supported exact root states, the sampled estimator uses
`h_hat_i = mean(G(S_terminal))` over K independent reference suffixes and
`r = sqrt(log(2m/alpha)/(2K))`. The clipped intervals
`[max(0,h_hat_i-r), min(1,h_hat_i+r)]` cover all m expectations simultaneously
with probability at least `1-alpha` under a deterministic reference/objective.
This is a per-comparison Monte Carlo guarantee, not across-round coverage or
calibration of a learned value model. Guidance requires a disjoint pair of
intervals and uses their lower bounds in the existing KL tilt. The policy is
an explicitly conservative approximation, not an optimality guarantee.

Remaining practical risks: full row construction still enumerates and executes
many marks; K samples per supported root state can itself be too expensive on
wide support; a rare terminal event can cause honest repeated abstention; and
spending the entire executor budget on lookahead leaves no budget for later
committed steps. These require production cost/completion evidence before
promoting this estimator. The engineering K=32 is not a selected T4 setting.
