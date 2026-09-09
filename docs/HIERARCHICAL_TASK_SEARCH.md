# Hierarchical task search: implementation contract

2026-09-08. The user approved the cross-option, task-aware design after reviewing
the limitations of structural-only guidance. This explicitly reopens task-dependent
region and option allocation in a separate opt-in controller. Frozen reference
priors, R_theta, executor, kappa=1, generic availability and T4 thresholds stay
unchanged. The abandoned narrow proposal in T4_TASK_CONTINUATION.md is not the
governing recipe for this implementation.

COMPOSE's scientific identity remains an executable stochastic molecular rewrite
process, whose non-null outputs are complete supported graphs. This milestone
tests task-value propagation across WHERE, WHAT, HOW and option boundaries. It
does not assert docking improvement, new generator support or benchmark superiority.
Support is charge-preserving broad-organic graphs, at most 40 real atoms, with
no stereochemistry claim. No new operators, winner-derived templates, R_theta
training, or automatic paid docking campaign is authorized here.

## Reference and planner

Use augmented states (exact molecule, atom lineage, remaining primitive budget,
region, option, program phase). Region and option selection consume no primitive
budget; every executed edit consumes one. Completed options return to WHERE.
Generic executes at most three steps before returning to WHERE, an explicit new
option-duration policy in this controller, not a change to generic primitive
support. Other option durations remain registered and must fit the budget.
Use the existing region prior and balanced product-applicable option prior.
The existing OptionContinuationKernel supplies all legal primitive rows.

At every node retain the full reference row. Sparse planning samples trajectories
through those rows, never renormalizes over a top-k or only visited actions.
Successor estimates of terminal feasible desirability guide the existing KL tilt.
Region/option exploration uses their original scale/option floors; HOW retains
the primitive reference floor. By joint convexity, mixing a KL<=1 tilted core
with the same fixed floor keeps KL to the mixed reference <=1.

Repeated rollouts share exact rows and return statistics. Adaptive rollouts
record their actual proposal probability and use suffix reference/proposal
importance ratios, accumulated in log space, in self-normalized return estimates.
Two reference-equivalent pseudo-observations shrink each child to the parent
return estimate (one before any evidence). This is a finite-sample biased planning
approximation, not an exact Doob law or calibrated uncertainty. Full reference
support remains selectable; unvisited actions are not failures. Only completed
rollouts contribute return statistics. Interrupted paths never get a fabricated
terminal reward. Previously completed rollouts remain evidence after a stop.

Cache keys retain persistent slots, charge/H channels, region, phase, lineage
and remaining budget. Canonical identities are for property caching and oracle
dedup only. A new objective snapshot requires new return statistics. Distinguish
planning trajectories from committed outer draws in every receipt. Primitive
validity is never replaced by a surrogate; endpoint feasibility never prunes an
enabling intermediate.

## Task value and fixed development check

Fit one deterministic small-data kernel-ridge model to prior completed docking
rounds only: ridge=1, centered scores, kernel equal to the average of Tanimoto
kernels on radius-2 Morgan and topological atom-pair 2048-bit fingerprints.
These are chemistry features, not rewards for adding rings. Centering supplies
the training mean as the prior prediction for an unrelated molecule. No neural
policy, R_theta or learned Q(o) parameters are fitted. No uncertainty estimates
are used for action selection; exploration is through fixed reference floors.

Terminal desirability is unchanged T4 feasibility times
exp(-max(0, predicted_score - best_training_score) / max(1, training_score_std)).
It is a bounded heuristic, not a binding probability. Beyond-best predictions
saturate. Count every observed docking attempt; exclude missing labels from
fitting with recorded reasons. Reject nonfinite labels, canonical duplicates,
current/future-round training records and foreign schema inputs.

Freeze the first predictor check before running it: rolling chronological
validation of the saved 51-call archive, training on rounds < the scored round
with at least 16 labels. Report all scored rows, failed-oracle exclusions,
MAE versus the training-mean predictor, and within-round concordance excluding
observed ties. At least 20 scored observations, lower MAE and concordance >0.5
are prerequisites for proposing an oracle experiment, not generalization proof.
Do not change the recipe after reading this check without a new development
decision and fresh prospective evidence.

## Bounded implementation and acceptance

Implement a reusable planner, the molecular hierarchy adapter, frozen task-value
snapshots and a prepare-only T4 entry point. Preserve the legacy controller.
Default preparation budget: 8 parent lineages, 16 primitive edits/lineage,
2,500 public executor calls/parent (20,000 total), 128 cached rows/parent,
256 terminal evaluations/parent, 32 attempted planning rollouts/parent.
Planning may consume at most 512 of each parent's executor calls; the remainder
is reserved for completing committed draws. Completed earlier rollout returns
remain usable when this planning allowance stops a later rollout.
The chosen path has a separate RNG stream. Fit once per round. No docking is
performed by preparation; persist candidates, exact execution paths, intended
and realized scale, decision probabilities, value estimates, planner work,
failure reasons and snapshot/input identities. A prepared graph path is not
silently passed to the old single-bundle archive verifier: schemas are distinct.

Focused acceptance tests: delayed terminal value changes WHERE/WHAT/HOW and
reverses when only utility reverses; trajectories cross actual option boundaries;
reference normalization, original floors and KL hold; budget/empty-support
failures are honest; repeated rows reuse work; remaining budget/lineage keys
do not collide; predictors are serialization-invariant and round-clean. Include
an actual production-executor fixture and chronological saved-label check.
This is a bounded development implementation, not a release milestone; run
focused tests, lint, diff checks and preflight. Preserve unrelated dirty work.

Next scientific comparison: same generator/support with structural post-hoc,
immediate task value and cross-option continuation. Match the named binding
resource and report all other costs. Newly adaptive outer draws are part of the
treatment; a fixed-bundle ablation separately isolates HOW. Fresh docking,
clean-source Modal deployment and the comparison contract are a separate launch
boundary. No automatic expansion of this implementation task into training or
a multi-round campaign.

## Inspiration, not inherited performance guarantees

- Diffusion Controller (ICML 2026): https://arxiv.org/abs/2603.06981
- Diffusion Tree Sampling (NeurIPS 2025): https://arxiv.org/abs/2506.20701
- Twisted SMC (ICML 2024): https://proceedings.mlr.press/v235/zhao24c.html
- Sampled MuZero (ICML 2021): https://proceedings.mlr.press/v139/hubert21a.html

The adaptive importance-weighted planner here is a declared COMPOSE development
approximation. Those papers' exactness/asymptotic or empirical claims are not
claims about this implementation.
