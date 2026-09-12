# A search-and-learning controller for COMPOSE

## Current decision, 2026-09-12

The authorized paired T4 development round is complete. From the identical
51-call PARP1 seed0, delta=0.4 archive (best -9.7), post-hoc control produced four
eligible docked endpoints and reached -9.9; short in-loop control produced eight
and reached -10.0. Both best molecules were completed six-member pendant-ring
options, not decorative primitive edits. The exact result, paths, hashes, scale
diagnostics, option census, and proposal work are recorded in
[the paired comparison artifact](../diagnostics/t4_frontier_compare/README.md).

This changes the immediate diagnosis. Broad option support and the executor can
construct useful feasible ring chemistry on this cell. The endpoint predictor
ranked the observed winner last in each arm, so current value assignment is now
the clearest measured bottleneck. In-loop search increased eligible coverage but
used 58,135 executor calls versus 21,722 post-hoc and affected only eight HOW
decisions; WHERE and WHAT had no task-value contrast. One unreplicated 0.1
kcal/mol arm difference is not a reproducibility claim.

This is not an IVG-level result. The verified public IVG records for the same
cell are -13.5, -13.6, and -13.6; GenMol reports -10.6. The current run is also
below the workshop COMPOSE value of -10.7 at 500 calls, but has consumed only 59
total calls in the in-loop history. These are reference points at unmatched
budgets and do not establish a matched ranking.

The current T4 champion under this exact warm-start development protocol is
-10.0 after 59 total calls in the in-loop arm. The complete-plan PMO ranker
remains a banked null for policy improvement: predictive fit did not translate
into robust selected-return improvement. There is no active scientific run and
no automatic next round. The next discriminating controller change is reusable
option-complete delayed value plus explicit exploratory oracle allocation,
evaluated against the same-generator post-hoc arm. It must earn its additional
proposal cost before scale-up. No further T4 or PMO oracle work is authorized by
the completed contract.

## Recommendation

Develop an option-aware, receding-horizon graph-search controller with a persistent
frontier, reusable continuation values, and an outer oracle-allocation loop.
The algorithmic family is approximate policy iteration with an exact transition
model. The molecular executor supplies that model; it does not need to be learned
again. Retain the frozen molecular reference and adapt the controller.

This is a proposed development direction, not an implemented algorithm or a claim
of superior performance. Literature and repository evidence were reviewed on
2026-09-09, against repository HEAD
`0594b0e512773ce5e68fd9eaf4e8eb70224d348f`. Existing uncommitted work is not treated
as selected scientific code. The review launches no experiment or training and
does not amend a frozen run contract.

The immediate research question is whether useful continuation guidance improves
actual editing outcomes per oracle call and per unit of proposal computation
over the same generator with endpoint-only selection. A particular IVG molecule
is a diagnostic destination, not the definition of success.

## Evidence ledger

The following are existing development measurements, not new computations.
Their linked artifacts record producer revisions, input hashes and verification.

| Evidence | Status and interpretation |
| --- | --- |
| [Ring-program round](../diagnostics/t4_ring_program_round/README.md): 7/7 selected new construction programs completed; 1/7 was T4-feasible. | Measured construction capability on one inspected cell. Most proposed ring endpoints failed source similarity. This is not a general success-rate estimate or evidence that ring construction improves docking. |
| Same round: best observed score became -9.7 after 51 counted calls through a local edit; the eligible new pendant ring scored -8.2. | Measured docking outcomes. The 0.1 best-score change can include unseeded docking noise. Ring count is not a validated reward. |
| [Chronological predictor check](../diagnostics/t4_task_search/README.md): MAE 0.555 versus 0.823 for the training-mean baseline, on 31 next-round observations. | Computed retrospective predictive signal in the inspected PARP1 cell. This does not validate uncertainty, a continuation critic, or extrapolation to new scaffolds. |
| [Task-search audit](../diagnostics/t4_task_search_audit/README.md): 0/8 planning rollouts completed; 0/100 primitive decisions were task-dependent. | Measured failure to deliver guidance under that compute allocation. Exact execution of all 100 selected edits is a separate positive integrity result. |
| [Cancelled recovery attempt](../diagnostics/t4_target_recovery/README.md): last heartbeat recorded 821.98 seconds, three completed planning rollouts and zero committed molecular edits. | Incomplete attempt. WHERE did change, so guidance was not entirely inactive. The heartbeat is not a final runtime or billing receipt. |
| [Winner paths](../diagnostics/ivg_winner_paths/README.md): 82/91 distinct source–winner pairs had exact local 2D witness replay; conditional correctness was 82/82. | Target-informed executable-path evidence under the recorded local runtime. Not autonomous discovery, learned-reference support verification, or evidence that every remaining endpoint is unreachable. |
| [Saved-path values](../diagnostics/t4_target_path_values/README.md): all three inspected PARP1 seed0 witnesses contain immediate-similarity decreases and long endpoint-feasibility excursions. | Computed on saved states with pinned chemistry. These paths show a risk for myopic pruning, not that every path requires a valley or that every stochastic greedy policy fails. |

The inference is specific: capability, endpoint prediction, affordable value
propagation, and final oracle allocation are different problems. The current
evidence is weakest for effective value propagation and autonomous multiround
search, not for the existence of any ring-closing operator.

## What the literature contributes

The sources below suggest components, not a package whose empirical guarantees
transfer to COMPOSE. Modernity alone is not a selection criterion.

| Primary source | Relevant mechanism | COMPOSE decision |
| --- | --- | --- |
| [Value Matching, ICLR 2026](https://iclr.cc/virtual/2026/poster/10011272) | Online, on-policy value learning for reward adaptation, separated from base-model complexity. | Learn a small controller-side value estimator on the states the search actually visits. Do not retrain the molecular reference merely to add task feedback. |
| [TD-MPC2, ICLR 2024](https://arxiv.org/html/2310.16828v2) | Short trajectory planning with a learned terminal value that estimates returns beyond the explicit horizon. | Bootstrap beyond short executable continuations. Use the exact molecular executor, not TD-MPC2's learned latent dynamics or continuous-action optimizer. |
| [Sampled MuZero, ICML 2021](https://proceedings.mlr.press/v139/hubert21a.html) | Policy evaluation and improvement over sampled action subsets. | Allocate search to a small, expanding set of alternatives. This does not by itself make our current full-row reference evaluator cheap. |
| [Diffusion Tree Sampling, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/d6484394c4cb5e1f4ecad8d90b912025-Abstract-Conference.html) | Reuse search information and distinguish reward-aligned sampling from reward-seeking search. | Retain reusable search structure. Its full terminal rollouts should not be copied into the bottleneck we already measured. |
| [Diffusion Controller, 2026](https://arxiv.org/html/2603.06981v1) | Formulate adaptation as control of transition kernels with divergence regularization. | Keep the state-level law and control constraints explicit. A formulation is not a substitute for a cheap, accurate value estimator. |
| [Short-Term Graph Memory, July 2026 preprint](https://arxiv.org/html/2607.28437v1) | Fit an online graph surrogate on observed oracle labels and screen proposals before spending further calls. | Preserve a strong same-generator, post-hoc selector as both a useful component and a causal baseline. |

Two additional lines inform narrower questions. [Twisted SMC, ICML
2024](https://proceedings.mlr.press/v235/zhao24c.html) learns future-value twists
to allocate particles for a declared target distribution. It remains relevant
when faithful sampling is the goal; proposal tilting and particle weighting
must not double-count guidance. [Multistep quasimetric goal-conditioned RL,
ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/eddc0fab5a42f8d8de6eb5566cd9f1d3-Abstract-Conference.html)
connects learned goal-reaching distances with multistep experience. That is a
possible later replacement for fingerprint similarity in answer-known recovery,
not a docking-value model and not a prerequisite to the next T4 comparison.

The InVirtuoGen comparison matters at the optimizer level. Its latest public
revision explicitly uses GA plus PPO-based adaptation for property optimization
and carries that method into lead optimization with soft constraint penalties.
It is not just a menu of ring additions. The August 2026 revision also separates
an additional oracle-pretraining regime from its main budgeted results.
Its preprocessing, learning and query budgets must be matched explicitly before
claiming relative efficiency. [InVirtuoGen v2, sections 3.3–3.4 and appendix
B.3.2](https://arxiv.org/html/2509.26405v2)

There is a relevant source discrepancy: Short-Term Graph Memory describes its
InVirtuoGen backbone as having fixed operators and population-based feedback,
whereas the original IVG paper describes PPO updates. Its adapter results are
therefore not automatically evidence about an exact reproduction of the full
IVG optimization stack. This does not invalidate the general selector idea.
[Memory implementation description](https://arxiv.org/html/2607.28437v1),
[original optimization method](https://arxiv.org/html/2509.26405v2)

## First-principles formulation

### The object being optimized

COMPOSE supplies a discrete, state-dependent graph of supported molecular edits.
It does not supply physical reaction kinetics, a smooth differentiable molecular
manifold, or a perfect binding-energy simulator. Its useful properties here are
executable transitions, complete intermediate molecules, exact continuation from
stored states, and variable-scale editing. Other graph editors share some of
these properties; none is individually a novelty claim.

For T4 the scientific objective is an adaptive discovery problem: maximize the
quality of the best feasible molecule actually evaluated within the oracle
budget. For PMO the relevant objective is top-k quality over the query sequence;
for multiobjective editing it may be improvement of a feasible Pareto archive.
Those archive functionals are not interchangeable with sampling one terminal
molecule from a fixed reference tilt.

Let `D_t` contain actual oracle observations and `F_t` the undocked search
frontier. The outer state includes `(D_t, F_t, remaining oracle budget)`. The
exact Bayes-adaptive problem would additionally track uncertainty about the
oracle. The proposed implementation approximates this problem with a frozen
within-round predictor and explicit exploration. It does not claim to solve
Bayes-optimal experimental design.

### Three quantities that must stay distinct

1. `f_hat_t(x)`: predicted current endpoint utility, fitted to true oracle labels.
2. `h_b^R(s)`: expected terminal desirability after following a specified reference
   process for a specified horizon.
3. `V_b^pi(s)`: expected future utility from continuing under the current search
   policy, with explicit stopping and feasibility semantics.

In the current [planner](../src/compose_v4/control/task_search.py), completed
rollout returns and suffix importance weights estimate reference continuation
values. An interrupted rollout supplies no return. There is no reusable learned
leaf critic to evaluate an unfinished continuation. The
[docking predictor](../src/compose_v4/control/docking_value.py) estimates endpoints;
calling it a continuation value would not make it one.

Doob control and optimal control are not opposing mathematical ideas. For fixed
horizon, terminal utility `u`, reference path law `P_R`, and a KL penalty with
coefficient `tau > 0`, the standard variational problem

\[
\sup_{P\ll P_R}\left\{\mathbb E_P[u(X_H)]
 -\tau\,\mathrm{KL}(P\Vert P_R)\right\}
\]

has optimizer proportional to `P_R exp(u(X_H)/tau)` when the normalizer is finite.
Its backward values give a Doob transform. Thus an accurate, efficiently
learned Doob value could be useful. The distinction is between the declared
objective and its computational approximation, not an alleged defect in Doob
theory. [Soft value and controlled-kernel equations](https://arxiv.org/html/2506.20701v1)

Our per-row KL ceiling, explicit exploration floors, option-conditioned
references, anytime candidate harvesting, changing oracle model, and archive
selection do not automatically inherit that fixed-terminal theorem. The next
implementation must declare its new value target rather than silently remove
importance correction while still claiming reference-value estimation.

### A coherent inner planning problem

Use exact augmented state `s = (x, source, lineage, region, option, phase, b)`.
Remaining primitive budget `b` decreases on executed edits, not WHERE or WHAT
choices. Macro duration is the actual number of primitive transitions; options
are not single molecular jumps. Let `C(s)` be the declared class of controlled
rows, including the existing support, fixed floors and KL bound.

For a bounded within-round surrogate utility, retain a running best eligible
candidate utility `m`. This makes an anytime best-candidate objective Markov.
A conceptual finite-budget Bellman recursion is

\[
V(s,m)=\sup_{K\in\mathcal C(s)}
 \sum_{s'}K(s'\mid s)\,V\bigl(s',m'(s',m)\bigr),
\]

where `m'` updates only at eligible candidate states, the remaining edit budget
is contained in `s'`, and `V(s,m)=m` at termination. WHERE and WHAT form acyclic
administrative stages between budget-consuming edits. Empty support and failed
programs require declared termination/continuation rules. The initial value for
no eligible candidate must be fixed, not inferred from a convenient result.

This is an inner surrogate-search objective, not the true unknown docking
objective. A temporarily endpoint-infeasible state can have high continuation
value even though it contributes no eligible candidate now. Endpoint feasibility
is enforced when selecting a molecule for docking, not imposed on every path
state. Executor validity remains mandatory everywhere.

Short planning uses actual executed transitions for a small number of steps or
completed options and estimates the remaining return with a leaf value. Replay
and multistep backups update those values; subsequent policy improvement changes
WHERE, WHAT and HOW within their declared constraints. Learned approximation,
sampled search and finite memory remove any claim of exact optimality.

## Proposed implementation

### Keep the three-level interface

Preserve WHERE / WHAT / HOW, the region scale floor, permanently available
generic editing, and parameterized primitive-executed options. Keep the balanced,
applicability-aware option distribution as a reference prior. A task-dependent
value can guide selections without pretending that this reference prior is a
trained option model.

Construction programs should provide temporal commitment through enabling
steps. They must compose with restating, deletion, rerouting and subsequent
construction. A completed ring is a possible waypoint, not necessarily the
end of a useful molecular modification. Do not reward ring count, reproduce
winner fragments, or equate a larger permitted region with a larger realized edit.

### Separate three memories

- **Oracle archive:** canonical molecules with actual scores, attempt accounting,
  feasibility and provenance. Only actual labels fit the endpoint predictor.
- **Search frontier:** exact valid molecular states, including temporarily
  endpoint-infeasible states and interrupted programs, with continuation context.
  These may survive across option and oracle-round boundaries without being
  docked or represented as feasible results.
- **Transition/value replay:** executed transitions and their context, plus
  explicitly tagged observed, surrogate and bootstrapped targets.

Molecular property caches may use canonical identity. Continuation caches must
retain persistent slots, lineage, region, option phase, remaining budget and
objective snapshot. Two equal canonical SMILES need not be equivalent search
states. Reuse molecular work across rounds but invalidate or recompute values
when the task predictor changes. Do not invent canonical-to-slot replay mappings.

### Start small, then amortize useful search

First implement resumable short expansions, completed-option value backups and
the persistent frontier using existing code. Reuse saved compatible rows for
mechanical checks. This delivers a runnable search baseline without waiting for
a large learned controller to exist.

Then fit a small regularized continuation head from the replay collected by that
search. Use completed multistep returns as anchors and explicitly tagged
bootstrapped targets for shorter segments. Frozen molecular features plus budget,
source margins and option context are a reasonable first parameterization, not
a selected architecture. Train/calibrate by source and trajectory, not shuffled
prefixes from the same paths.

The head approximates the declared current-policy/within-round search value,
not an unexplained mixture of reference and optimized returns. Training on
surrogate-labeled continuations does not create new docking knowledge. Predictions
and bootstrap targets must never be added to the true docking-label archive.
Fifty-one docking observations do not establish a generally reliable neural
binding model or a universal continuation critic.

At cold start, use reference exploration and short completed continuations.
Retain a no-critic fallback if the head does not improve decisions on fresh
development continuations. Distilling improved search into a lightweight policy
is a later efficiency step, contingent on measured search benefit. It is not
necessary to train a new `Q(o)` policy or change `R_theta` in the first comparison.

### Bound work without destroying useful depth

Use a sparse shared search graph and progressively allocate additional expansion
to useful alternatives. Reuse continuations when moving the root. Maintain a
diverse population; a strict width-one beam would discard possible enabling paths.
Allocate outer effort to selected bundles, not in proportion to the number of
intermediate states each bundle happens to produce.

A planning quantum limits work before returning a usable frontier and decision;
it must not erase the unresolved suffix. Small per-decision planning horizons
and long cumulative trajectories can coexist. Report actual continuation depth
and interruptions, rather than labeling a truncated program completed.

Batch learned-law and value evaluation and cache deterministic chemistry using
compatible dependency identities. Parallel parents can reduce elapsed time but
not total work. A sampled action planner does not eliminate reference-law
enumeration automatically: require a measured profile and a bounded law-parity
check before changing the sampling implementation. Do not rename top-k
renormalization as an exact speed optimization.

### Put oracle feedback in both loops

Freeze endpoint and continuation models during an oracle round. Generate and
compress a diverse candidate pool, lock identities, and dock feasible selected
candidates concurrently. Update models only after the round closes.

Use both predicted quality and explicit structural/bundle diversity for
allocation. Keep exploratory slots so the endpoint model is not evaluated only
on molecules it already favors. Do not interpret ensemble disagreement as
calibrated uncertainty without a relevant validation check. Begin with the
existing small-data predictor; compare a stronger chemistry representation only
if new observations show a ranking failure under the encountered shift.

Learning that one ring program happened to score well is not evidence that
all extra rings help. The desired signal is context-dependent: attachment site,
source preservation, heteroatom pattern, downstream edit opportunities and
predicted task value. Atom/bond and non-ring edits remain eligible competitors.

## Efficient development and evaluation

### The next comparison

Use one small, explicitly development-only, multiround comparison. The arms are:

1. Same frozen COMPOSE generator and options, with task scoring only for endpoint
   selection and population update.
2. The same system with short in-loop task guidance and persistent continuations.
3. The same system with the learned continuation head, when it has earned use.

Start with arms 1 and 2; arm 3 is an incremental addition, not a prerequisite
to observing real optimization. Preserve immediate-value guidance as the cheap
leaf baseline so continuation learning must demonstrate additional value.
Do not wait for exact recovery of an IVG winner before this comparison.

A practical proposed first T4 slice is three rounds of at most 20 new docking
attempts per arm from the identical 51-call development archive. This is at most
120 new attempts for the first two arms, not a launch authorization and not a
claim that 60 calls establish superiority. Count the shared 51-call prefix in
each arm's complete history; a warm-start pilot is not a fresh-seed benchmark.
Use paired random seeds and the same oracle protocol. If an arm cannot fill its
allowance with feasible candidates, report that failure and actual spend.

Match the primary oracle budget and report proposal time, law calls, executor
work, surrogate evaluations and parallel resources separately. The pilot should
show time to first committed edit, completed useful continuations, persistence
through endpoint-feasibility excursions, candidate diversity, intended versus
realized structural change, and best feasible docking score versus calls.
No score-gain claim rests on one small unreplicated difference.

The implementation needs focused checks for value propagation, exact valid
execution, reference/floor/KL accounting, and interrupted-state reuse. Reuse
existing tests for unchanged invariants. A sequence of unrelated test suites
or single-ring audits is not the development loop.

### Diagnose the failing layer

| Observation | Next action |
| --- | --- |
| Legal desired edits do not enter an applicable row | Inspect executor-to-option support and context masks; do not compensate by tuning value. |
| Realistic constructive paths appear, then disappear before becoming feasible | Inspect frontier survival, phase continuation and feasibility handling. |
| Valuable completed candidates exist but are not queried | Inspect endpoint prediction and oracle allocation. |
| Predictions improve but measured docking does not | Investigate extrapolation and oracle noise; do not treat surrogate gain as task gain. |
| Continuation guidance is more expensive without better outcomes | Keep the cheaper controller. Complexity must earn its cost. |

### IVG winners and broader tasks

Use the already inspected IVG winners to localize support defects and explain
failures after runs. Maintain exact-state witness replay and canonical endpoint
identity as separate from fingerprint proximity. Do not derive a reusable ring
catalog, weights, reward bonuses or continuation training set from those winners
and then call their recovery held-out performance.

If target-informed imitation is studied, give it its own explicitly supervised
development arm and evaluate on independent targets. Training data from a known
winner can be useful for that question but changes the experiment. Ordinary
SMILES string distance is not a molecular edit distance or a docking objective.

For paper breadth, select tasks before measuring the candidate controller:

- A source-constrained editing task with cheap objective evaluations for rapid
  comparison of immediate and multistep guidance.
- T4 for constrained, expensive-oracle optimization with matched per-run budgets.
- A declared PMO evaluation lane for broader oracle functions, respecting its
  initialization protocol rather than silently converting it into T4-style editing.
- A controlled retargeting or pathwise-constraint experiment to isolate reuse
  of complete molecular intermediates. Multiobjective tasks need their own
  frozen archive functional and relevant same-generator evolutionary baseline.

Select the controller on development evidence, then freeze it for independent
evaluation. Already inspected benchmark winners and cells are not newly sealed
test data. Report infeasibility and unsupported cases over the declared panel;
do not select only the tasks that benefit from additional rings.

## Claim boundaries and decision

The proposed contribution is a useful realization of adaptive, multiscale control
on COMPOSE's executable trans-dimensional process, supported by causal baselines
and transfer across objectives. It is not the invention of policy iteration,
MCTS, Doob transforms, model-predictive control or online surrogate optimization.

The current molecular support remains connected charge-preserving graphs with
at most 40 active atoms under the frozen vocabulary, valence, executor and
canonicalization rules. Stereochemical generation is not established. Retain
`generic`, the existing reference parameters, kappa=1 and endpoint thresholds
for the initial comparisons. New value-learning targets and persistent-frontier
semantics require their own explicit development contract before execution.

No reviewed method proves that this implementation will find IVG's winners or
outperform every task baseline. The best justified next direction is to make
search experience reusable and make task feedback affect affordable decisions,
then retain only the added machinery that improves measured optimization.

## Sources

1. Jensen, Schaufelberger, De Santi, Jorner and Krause. [Value Matching: Scalable
   and Gradient-Free Reward-Guided Flow Adaptation](https://iclr.cc/virtual/2026/poster/10011272).
   ICLR 2026. The official conference abstract supports the online value-learning
   mechanism cited here; this memo does not reproduce its empirical figures.
2. Hansen, Su and Wang. [TD-MPC2: Scalable, Robust World Models for Continuous
   Control](https://arxiv.org/html/2310.16828v2). ICLR 2024, sections 2–3.
3. Hubert et al. [Learning and Planning in Complex Action
   Spaces](https://proceedings.mlr.press/v139/hubert21a.html). ICML 2021.
4. Jain, Sareen, Pedramfar and Ravanbakhsh. [Diffusion Tree Sampling: Scalable
   inference-time alignment of diffusion models](https://proceedings.neurips.cc/paper_files/paper/2025/hash/d6484394c4cb5e1f4ecad8d90b912025-Abstract-Conference.html).
   NeurIPS 2025. [Full-text equations and algorithm](https://arxiv.org/html/2506.20701v1).
5. Yang et al. [Diffusion Controller: Framework, Algorithms and
   Parameterization](https://arxiv.org/html/2603.06981v1). 2026, section 3.
   ICML 2026 listing corroborated by the [authors' institutional conference
   page](https://research.google/conferences-and-events/google-at-icml-2026/).
6. Yang, Thost, Ling and Ma. [Oracle-Budgeted Molecular Optimization with
   Short-Term Graph Memory](https://arxiv.org/html/2607.28437v1). July 2026
   preprint, sections 2–3. Not presented as an accepted conference result.
7. Zhao, Brekelmans, Makhzani and Grosse. [Probabilistic Inference in Language
   Models via Twisted Sequential Monte Carlo](https://proceedings.mlr.press/v235/zhao24c.html).
   ICML 2024.
8. Zheng, Myers, Eysenbach and Levine. [Scaling Goal-conditioned Reinforcement
   Learning with Multistep Quasimetric Distances](https://proceedings.iclr.cc/paper_files/paper/2026/hash/eddc0fab5a42f8d8de6eb5566cd9f1d3-Abstract-Conference.html).
   ICLR 2026. The abstract supports the limited conceptual comparison here.
9. Kaech, Wyss, Borgwardt and Grasso. [Refine Drugs, Don't Complete Them:
   Uniform-Source Discrete Flows for Fragment-Based Drug
   Discovery](https://arxiv.org/html/2509.26405v2). ICLR 2026; public revision
   dated 2026-08-03. That revision identifies appendix B.3.2 as a post-review
   addition, not part of the peer-reviewed camera-ready.

The supplied COMPOSE workshop LaTeX is contextual manuscript material, not an
independently verified result source. Repository evidence is linked in the
ledger above; historical documents retain their own point-in-time limitations.
