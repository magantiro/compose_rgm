# Controller decision memo: temporal abstraction is necessary, but semi-Markov control is not sufficient

Date: 2026-09-12

Repository revision reviewed: `cea34c8a52becc0537d50e1b7097fbb95d7c9afa`

Audience: COMPOSE research and implementation team

Status: literature synthesis and proposed development decision; no new benchmark or oracle result

Evidence labels used below:

- **Measured:** observed in the versioned COMPOSE artifacts named in this memo.
- **Reported:** stated by the cited primary paper or official results repository.
- **Inferred:** the memo's interpretation of measured and reported evidence.
- **Proposed:** an untested controller or experimental decision.

## Executive decision

**Decision, inferred and proposed.** COMPOSE should use a semi-Markov decision process (SMDP) at option boundaries, but it should not use "semi-Markov" as the name for, or substitute for, the complete optimizer. Temporal abstraction is the correct answer to one measured failure: useful molecular transformations require many valid primitive rewrites, while primitive-by-primitive selection makes their path probability and credit assignment collapse. The options framework gives a mathematically appropriate state transition model when actions take variable numbers of primitive steps.[^1] It does not, by itself, make useful options likely, infer their delayed value, allocate expensive oracle calls, or learn from successive rounds.

The recommended controller is therefore a hybrid:

1. **Compositional structural options** reduce the effective horizon. An option describes a graph-change objective and an executable low-level policy, not a fixed molecular endpoint or a finite ring-template catalog.
2. **A policy-congruent distributional improvement value** estimates the probability or distribution of best improvement reachable within the remaining option budget under the policy that will actually be run.
3. **A learned task-adaptive option proposal** moves probability toward successful region, option, and structural-parameter choices using off-policy, trust-region policy improvement, while retaining an explicit reference and generic exploration floor.
4. **Persistent option-boundary sequential Monte Carlo (SMC), optionally stratified as weighted ensemble**, allocates internal computation across distinct promising molecular histories and corrects for the learned proposal when a formal tilted target is claimed.
5. **An archive and calibrated batch acquisition layer** decides which completed molecules receive PMO or docking calls. This is distinct from the continuation value used inside a molecular trajectory.

In shorthand:

\[
\boxed{
  \text{structural options}
  + \text{policy-congruent return distribution}
  + \text{learned option proposal}
  + \text{persistent SMC/WE}
  + \text{calibrated oracle acquisition}
}
\]

This is not a recommendation to replace COMPOSE with PPO, a genetic algorithm, beam search, Monte Carlo tree search (MCTS), or a GFlowNet. It uses the part each line of work addresses well while preserving COMPOSE's distinctive substrate: exact executable rewrites between complete supported molecular states under a frozen reference process.

## 1. Project identity and claim boundary

### Scientific problem

The problem is goal-directed optimization over a vast, reversible, trans-dimensional graph of complete molecular states when:

- the executor provides broad valid support but the useful route has very low probability under the frozen reference law;
- a useful structural change may require a variable number of primitive rewrites;
- task rewards are sparse, delayed, noisy or expensive;
- the objective is usually best-of-budget or top-k quality, not the value of an arbitrary terminal sample;
- local decoration is easy and constructive topology change is hard;
- search must remain diverse enough to avoid irreversible population collapse.

### Primary controller output

At an option boundary, the controller should output a distribution over:

\[
(M,o,\xi),
\]

where $M$ is the selected molecular region and intended scale, $o$ is a structural option family, and $\xi$ contains option-specific structural parameters or subgoals. The low-level option policy emits a variable-length sequence of ordinary executable COMPOSE primitives and terminates in another complete valid molecule, failure state, or declared timeout.

### Central claim to test

The intended claim is not that options are novel, or that exact Doob mathematics alone produces a strong optimizer. It is:

> COMPOSE's executable molecular process can be controlled efficiently at multiple spatial and temporal scales by learning task-adaptive probability over compositional structural options and their reachable future improvement, while preserving broad molecular support and exact valid-state execution.

### Validation setting

The first validation setting is exposed development evidence on PMO Perindopril MPO and T4 PARP1 seed 0 at $\delta=0.4$. The eventual claim requires frozen, matched, multi-seed evaluation on PMO and T4, including all policy-training oracle calls, proposal compute, failures, and task constraints. Public winners may be used for development diagnostics, but not as blind evaluation.

### Required baselines and causal ablations

- the strongest actual COMPOSE controller in the same protocol;
- the same broad generator with post-hoc ranking only;
- primitive-boundary versus option-boundary control;
- current-score versus policy-congruent continuation value;
- fixed option prior versus learned option proposal;
- independent trajectories versus persistent SMC/weighted ensemble;
- single representation versus calibrated multi-representation acquisition;
- winner-informed reconstruction versus autonomous discovery.

### Support boundary

The generated object remains a complete connected molecular graph within COMPOSE's declared element, valence, size, persistent-slot, executor, and canonicalization support. Options must compile to existing supported primitive rewrites. `generic` remains active. A structural option may describe a large family of endpoints, but it must never teleport to one.

## 2. What the existing COMPOSE evidence already resolves

This decision should begin from measured repository evidence, not architectural preference.

### 2.1 Primitive horizon is a real failure mode

The exact T4 InVirtuoGen-winner reconstruction requires 23 COMPOSE primitives, including deletions, insertions, ring opening, three ring closures, bond reordering, and ring-system restating. Intermediate molecules can temporarily violate endpoint QED, SA, or seed-similarity gates. The current 16-primitive T4 horizon therefore excludes the full witnessed route. Existing option programs could reduce that route to roughly six to eight high-level decisions while still executing every primitive.

This supports temporal abstraction. It does not prove the autonomous controller will choose the right options.

### 2.2 Complete-option SMC was already tried, and it did not improve the best score

The bounded PMO experiment used eight particles, six complete-option opportunities, and options as long as eleven primitives. Reference, immediate-score SMC, and an existing future-head SMC all reached the same best score, 0.522233. The immediate arm improved top-ten mean modestly, while the future arm tied the reference. Actual executed depths were only 14 to 19 primitives despite a 66-primitive theoretical allowance. The immediate arm also saturated molecular size. See [`PMO_OPTION_PARTICLES.md`](PMO_OPTION_PARTICLES.md) and the stored run audit.

This is direct negative evidence against the proposition that adding option boundaries and SMC alone solves optimization.

### 2.3 The existing future head is not a future value for the deployed policy

At the first option boundary, the head's highest-ranked candidate had current score 0.002166 and predicted achieved value 0.780725, but its sampled suffix reached only 0.079737. Across eight trajectories, first-boundary Spearman correlation with eventual terminal score was -0.238 for the head and +0.524 for current score. The head was worse at all five nonterminal boundaries. It was conditioned on a nominal 55-primitive budget when actual remaining option trajectories used 7 to 17 primitives. It had been trained on best witnessed answer-known teacher paths rather than the expected or distributional outcome under the stochastic option policy. See [`GUIDE_AUDIT.md`](../diagnostics/pmo_option_particles/GUIDE_AUDIT.md).

This result does not refute future-aware control. It refutes using a policy-mismatched endpoint or achieved-route predictor as the twist.

### 2.4 Proposal probability, not only endpoint expressibility, is the bottleneck

COMPOSE can execute constructive ring changes and exact winner-informed routes. Yet the useful complete paths appear rarely under the current behavior policy, current-scale and future-head rankings can be wrong, and repeated sparse endpoint policy updates have not robustly improved the champion. The current PMO decision record reports a one-seed best of 0.683530 whose advantage did not replicate, while later plan-policy and edit-chooser interventions were null or negative. See [`CONTROLLER_LIVE.md`](CONTROLLER_LIVE.md).

The evidence therefore separates three requirements:

- **reachability:** a valid path exists in the declared support;
- **proposal:** the path receives enough probability to be sampled;
- **credit and allocation:** the system recognizes useful partial paths and spends oracle calls on the right completed molecules.

COMPOSE has substantial reachability. The current controller is weak on proposal and credit.

## 3. What semi-Markov options solve

The classical options framework defines an option by an initiation set, an internal policy, and a termination rule. A set of options over an underlying Markov process induces an SMDP at option boundaries because the next state and duration depend on the current boundary state and selected option.[^1]

For COMPOSE, define a boundary state

\[
s=(x,u_\star,b,\mathcal A,\mathcal D,\nu),
\]

where $x$ is the current exact molecule, $u_\star$ is the best task utility currently known, $b$ is the remaining option or oracle budget, $\mathcal A$ is an archive summary, $\mathcal D$ is the scored-data summary, and $\nu$ identifies the current controller snapshot. If all terms that influence future control are included, this augmented state restores the Markov property at round or option boundaries.

Let the base option decision be $a=(M,o,\xi)$. An option execution produces a primitive trace $\tau$, duration $\ell$, and next state $s'$:

\[
B(ds',d\tau,d\ell\mid s,a)
=Q_0(dM\mid x)\,\rho_0(do,d\xi\mid x,M)\,
q_0(d\tau,d\ell\mid x,M,o,\xi).
\]

The low-level distribution $q_0$ is supported only on ordinary executor-valid primitive transitions. This factorization preserves the desired WHERE, WHAT, HOW interpretation while making the duration explicit.

The benefits are real:

- effective high-level horizon can fall from tens of primitive choices to a handful of structural choices;
- delayed value can be assigned to the complete structural operation rather than an unattractive intermediate;
- compute can be allocated by intended transformation rather than by raw primitive frontier multiplicity;
- the controller can reason separately about region, topology operation, and execution details;
- failed or terminated programs have explicit durations and outcomes rather than disappearing.

Theory and empirical work support temporal abstraction as a route to lower regret or more efficient exploration in some problem classes.[^2] Data-efficient option-learning work also shows that action abstraction, temporal abstraction, off-policy reuse, and trust regions are distinct contributors, not interchangeable labels.[^3]

## 4. What semi-Markov options do not solve

### 4.1 They do not raise the probability of the right option

If the balanced option prior samples the right structural program rarely, or its structural parameters are nearly uniform over thousands of attachments and atom identities, the route remains effectively absent. Six option decisions are only better than 23 primitive decisions if the desired option-level route has meaningful probability.

### 4.2 They do not define the correct optimization objective

The PMO statistic is top-10 area under the curve, and practical lead optimization often reports the best feasible molecule within an oracle budget. A value that predicts the terminal score of one random continuation is not aligned with the probability of finding a better archive member. A winner-path maximum is also not an expectation under the deployed policy.

### 4.3 They do not supply a calibrated future value

A Doob transform is exact only with the backward value of the declared base process and terminal potential. Controlled SMC similarly depends on a twist that approximates the relevant future potential.[^4] An option-aware clock fixes one mismatch, but the target must also match the active policy and utility.

### 4.4 They do not allocate expensive oracle calls

Within-trajectory value asks whether a partial molecular path can produce improvement. Batch acquisition asks which completed, feasible, diverse candidates should be measured next. Mixing these quantities is a category error, particularly in T4 where docking is noisy and only a small batch is evaluated.

### 4.5 They do not learn across rounds

InVirtuoGen's strong optimizer updates its population every 50 scored samples, mutates elite candidates, updates its generative policy after every 100 scored SMILES, retains replay in the no-prescreen regime, and adaptively selects sequence length.[^5] Its ablations attribute material performance to mutation and PPO separately, and removing replay in the no-prescreen setting further degrades results.[^6] Whatever one thinks of its representation, this is evidence that a static option prior plus one-shot resampling is not a matched-strength optimizer.

## 5. Recommended controller

### 5.1 Layer 1: compositional structural options

The option library should be defined by structural outcome predicates and low-level policies, not endpoint templates. A useful option parameterization is:

\[
o=(\text{intent},\Delta n,\Delta r,\text{ring relation},\text{attachment mode},
\text{composition class},\text{termination contract}),
\]

where $\Delta n$ is heavy-atom change, $\Delta r$ is graph cycle-rank change, and ring relation includes pendant, fused/annulated, expanded, opened, or restated. Parameters may be partially unspecified and resolved by the low-level policy.

Examples:

- construct one five- or six-member pendant ring attached through a selected region;
- annulate one ring onto an existing ring system;
- replace a side chain while preserving a protected core;
- open and rebuild a ring system;
- grow a scaffold until a closure-compatible pair becomes available, then close and restate;
- generic rewrite with no semantic restriction.

These are program families over atom and bond resolution. They are not fragments in a fixed vocabulary. New ring composition, substitution, fusion site, and primitive realization remain open-ended within executor support.

The existing macro inventory is a suitable seed, not a finished option system. The 11-step `BUILD_RING_SYSTEM` program should be treated as a compound option only when its endpoint contract, failure semantics, duration, and probability under the low-level policy are explicit. Its three stages may also serve as interruptible sub-options. Classical option theory permits interrupting options when replanning is beneficial.[^1]

### 5.2 Layer 2: policy-congruent distributional improvement value

The value should match best-of-budget optimization. Let $U(x)$ be the task utility, oriented so larger is better, and $u_\star$ the current archive best. Define the first-passage improvement probability

\[
H_\phi(s,b,\delta)
\approx
\Pr_{\pi_\nu}\!\left[
\max_{0\le j\le b} U(X_j) \ge u_\star+\delta
\mid S_0=s
\right].
\]

For PMO, a distributional head can predict several improvement thresholds $\delta$ or quantiles of

\[
Y_b=\max_{0\le j\le b} U(X_j)-u_\star.
\]

For T4, $U$ should incorporate endpoint feasibility exactly and model the distribution of docking improvement, not only its mean. A survival-style representation of time to improvement is attractive because failed and truncated continuations can be handled as censored observations rather than assigned fabricated outcomes. Recent work formalizes goal-conditioned control through time-to-goal survival distributions, although its direct applicability to molecular optimization remains proposed rather than established.[^7]

The training contract is critical:

- rollouts come from the actual versioned option proposal and low-level executor policy;
- the remaining budget is counted in option decisions and, if needed, oracle opportunities, not a nominal maximum primitive count;
- every prefix of a completed trajectory supplies a return-distribution target;
- incomplete runs are censored or marked as explicit failure states;
- replay records the behavior-policy probability and controller snapshot;
- calibration is evaluated by molecular source, scaffold, task, and option family;
- winner-known paths can supervise proposal skills, but cannot be used as Monte Carlo expectations under the behavior policy.

Distributional reinforcement learning provides established methods for predicting return distributions rather than only expected return.[^8] Here, the reason is operational: two states with the same mean can differ sharply in their probability of rare large improvement, which is the quantity a best-of-budget search needs.

### 5.3 Layer 3: learned option proposal with a reference floor

Define a task-adaptive proposal

\[
q_\psi(a\mid s)
=(1-\epsilon)\,\tilde q_\psi(a\mid s)+\epsilon\,B_A(a\mid s),
\]

where $B_A$ is the applicability-aware base distribution over region, option, and parameters, and $\epsilon>0$ preserves explicit exploration. `generic` receives permanent nonzero mass. The learned component should be trained from scored and rollout-labeled option trajectories with conservative policy improvement.

A practical first objective is advantage-weighted regression:

\[
\max_\psi
\mathbb E_{(s,a)\sim\mathcal R}
\left[
  \omega(s,a)\log q_\psi(a\mid s)
\right],
\qquad
\omega=\operatorname{clip}\!\left(
\exp\left(\frac{\hat A(s,a)}{\tau}\right),0,w_{\max}
\right).
\]

The advantage should be derived from the distributional improvement target, not raw endpoint score alone. The update is constrained by a KL or total-variation trust region to the previous option policy, and training batches are balanced by task, parent/source, option family, and improvement status. Hindsight off-policy option learning and advantage-weighted methods support this general strategy of learning reusable option behavior from off-policy trajectories while controlling update size.[^3][^9]

This updates the controller, not the frozen $R_\theta$. It therefore preserves the scientific separation between goal-independent molecular plausibility and goal-specific purpose.

### 5.4 Layer 4: persistent option-boundary SMC or weighted ensemble

Let a completed option trace $\tau_t$ be proposed from $q_\psi(\tau_t\mid s_{t-1})$, while $B(\tau_t\mid s_{t-1})$ denotes the declared option-augmented reference. If the target path law is

\[
P^*(d\Gamma)\propto P_B(d\Gamma)G(\Gamma),
\]

with best-improvement potential

\[
G(\Gamma)=\exp\left\{\beta
\left(\max_t U(X_t)-u_\star\right)_+\right\},
\]

then a learned proposal must be importance-corrected. A generic incremental weight is

\[
w_t=w_{t-1}
\frac{B(\tau_t\mid s_{t-1})}{q_\psi(\tau_t\mid s_{t-1})}
\frac{\hat h_{b-t}(s_t)}{\hat h_{b-t+1}(s_{t-1})},
\]

with the exact terminal potential replacing the last twist. Twisted-SMC work makes the target/proposal distinction explicit: learned proposals can reduce variance, but they do not define correct intermediate targets automatically.[^10] Controlled SMC similarly treats approximate dynamic programming and policy refinement as a way to approach the optimal twist, with effective sample size (ESS) and weight behavior serving as diagnostics.[^4]

The system should persist exact particle states, histories, RNG states, option durations, proposal/reference log probabilities, weights, and archives across option and oracle rounds. It should not restart from only the current top molecules after every docking batch.

Weighted-ensemble stratification is a useful optional addition. Particles can be binned by a small scientific progress coordinate such as:

- feasibility distance;
- heavy-atom and cycle-rank delta;
- ring-system relation;
- intended versus realized edit scale;
- predicted probability of exceeding the incumbent;
- scaffold or fingerprint diversity cell.

Resampling within strata preserves multiple route families while conserving weights. Weighted ensemble is statistically exact when resampling conserves probability, but efficiency depends on the progress coordinate. Rare-event literature consistently warns that a poor reaction coordinate can produce severe variance; the committor is the ideal coordinate but must be approximated.[^11][^12]

The practical role of SMC/weighted ensemble here is internal compute allocation, not a promise that a small particle set will discover an astronomically rare route.

### 5.5 Layer 5: separate batch oracle acquisition

Completed, endpoint-feasible molecules enter a globally canonical-deduplicated candidate pool. The acquisition model should predict the oracle outcome distribution with calibrated uncertainty from all authorized scored molecules. It should use multiple molecular representations because recent PMO evidence shows that the useful fingerprint can vary by task, optimization phase, and seed; multi-fingerprint Gaussian-process ensembles improved low-budget robustness in MolLIBRA.[^13]

For PMO, oracle evaluation is inexpensive but counted, so acquisition should optimize expected top-k improvement under the remaining budget. For T4, docking is expensive and noisy, so a batch policy such as posterior Thompson sampling, q-probability-of-optimality, or a diversity-aware expected-improvement approximation should be compared on frozen retrospective labels before deployment. Recent batch Bayesian optimization work shows that correlation-aware probability-of-optimality can select diverse chemical batches without an ad hoc diversity penalty, but the best policy is dataset-dependent.[^14]

The acquisition model is not the continuation value. The former ranks completed molecules for measurement; the latter guides partially executed option histories.

## 6. How the recommended controller relates to Doob control

The clean mathematical target remains a KL-regularized path tilt. For a declared reference path measure $P_B$ and nonnegative terminal or path desirability $G$,

\[
P^*(d\Gamma)=\frac{P_B(d\Gamma)G(\Gamma)}{
\mathbb E_{P_B}[G]}.
\]

The exact backward function at option boundary $t$ is

\[
h_t(s)=\mathbb E_{P_B}[G(\Gamma)\mid S_t=s].
\]

An exact option-level Doob transform is therefore well-defined. The SMDP only changes the transition kernel and clock. This is the rigorous backbone.

The implementation should distinguish three modes:

1. **Exact target sampling:** proposal ratios and twists are recorded, the terminal potential is exact, and SMC approximates the declared tilted reference law.
2. **Approximate controlled sampling:** a learned, policy-congruent twist is used, with approximation and calibration reported.
3. **Optimization heuristic:** learned proposal, resampling, archive selection, or evolutionary operations are used for search without claiming exact sampling from a tilted law.

This distinction prevents elegant mathematics from masking a weak optimizer. Value Matching is relevant because it trains a value network on-policy for reward-guided adaptation of a frozen generative process, directly addressing train-inference mismatch.[^15] Its continuous flow-control derivation cannot simply be copied onto COMPOSE, but its lesson is important: a value used for control should be learned from the controlled distribution it is meant to improve, not frozen from unrelated teacher maxima.

## 7. Comparison with leading alternatives

| Approach | Strength | Failure in COMPOSE if used alone | Decision |
|---|---|---|---|
| Primitive Doob/twisted SMC | Clean target law and support preservation | Useful option routes have negligible primitive path mass; bad twist already failed | Retain mathematics, move control to option boundaries and learn a policy-congruent twist |
| Semi-Markov options alone | Correct variable-duration abstraction; shorter effective horizon | Does not learn option probability, future value, or oracle allocation | Necessary representation layer, not sufficient optimizer |
| PPO on the whole reference model | Strong online adaptation; InVirtuoGen ablation supports policy updating | Data hungry, can collapse support, changes frozen reference, difficult probability accounting | Do not fine-tune $R_\theta$; use conservative updates on the smaller option proposal first |
| Genetic or evolutionary search | Strong population reuse and crossover; robust molecular baseline | Hand-designed mutation can dominate platform identity and provide weak path-level credit | Keep as a same-generator baseline and optional donor proposal channel |
| GFlowNet | Targets diverse high-reward endpoints and handles multiple paths | Standard constructive state graph is acyclic; COMPOSE is reversible and cyclic; reward labels remain sparse | Borrow prioritized replay and flow-matching diagnostics; do not replace the controller initially |
| MCTS | Explicit multi-step lookahead and option-level planning | Primitive branching and molecular materialization are enormous; weak leaf value makes search expensive | Consider only option-level MCTS after a useful prior and value exist |
| Beam search | Simple deterministic top-k continuation | Collapses diversity and is brittle to temporarily bad intermediates | Diagnostic only, not the main optimizer |
| Bayesian optimization | Excellent oracle allocation under calibrated local uncertainty | Does not construct multi-step molecular routes | Use only as the completed-candidate acquisition layer |
| Fixed fragment vocabulary | Efficient long jumps and strong empirical priors | Narrows support and risks copying benchmark chemistry | Reject as the central representation; allow observed components only as an explicitly labeled proposal channel |
| End-to-end GFlowNet over time-augmented COMPOSE paths | Principled diverse terminal distribution despite multiple paths | Major new training system, cyclic-to-DAG augmentation, large off-policy dataset, and uncertain near-term advantage | Long-term research branch, not the smallest next experiment |

GFlowNet research remains informative. It was designed to sample diverse endpoints in proportion to reward and address multiple construction paths.[^16] More recent work shows that prioritized replay and guided trajectory balance improve substructure credit assignment,[^17] and genetic-guided GFlowNets distill strong genetic search into a generative policy.[^18] However, COMPOSE already has the stronger valid-state transition substrate. The immediate need is not a new global formalism, but effective option-level proposal and value learning on that substrate.

## 8. What InVirtuoGen teaches, and what it does not

InVirtuoGen is not strong merely because it can alter many tokens in one flow step. Its target-property optimizer combines rank-based population sampling, fragment-level crossover, mutation of elites, PPO updates, replay in the no-prescreen regime, and adaptive sequence-length selection.[^5] Its published PMO numbers are top-10 AUC over 10,000 oracle calls and three runs, not directly comparable to COMPOSE's current warm developmental best molecule.[^19]

The transferable lessons are:

- update the task policy repeatedly from scored outcomes;
- preserve an archive and elite mutation channel;
- learn the scale or duration of search rather than fixing it;
- use replay so sparse improvements affect more than one update;
- measure the full optimizer, not only generator validity;
- count policy-training oracle calls in the matched regime.

The non-transferable aspects are:

- InVirtuoGen can begin from invalid or globally inconsistent token sequences and refine them; COMPOSE commits only complete executor-valid molecular states;
- its fragmented-SMILES representation and billion-molecule pretraining are not COMPOSE's reference process;
- direct full-model PPO would erase the clean frozen-reference separation unless declared as a new scientific contract;
- its prescreen results depend on roughly 250,000 task labels and belong to a distinct benchmark regime.

The goal is therefore not to imitate InVirtuoGen's syntax. It is to match its optimizer strength using COMPOSE's different advantage: reusable exact valid-state trajectories that support closed-loop control, pathwise constraints, reversible edits, and explicit structural options.

## 9. Proper use of public winners

Public winners should be used aggressively in development, with roles separated in every artifact.

### Legitimate uses

- compute exact graph differences and minimal or near-minimal executable routes;
- map primitive routes into structural option sequences;
- estimate reference and option-proposal log probability along those sequences;
- identify the first boundary where the route loses mass or is misranked;
- train or test low-level option completion on route prefixes;
- create hard positive examples for transferable graph-change intents;
- compare autonomous candidates to winner-required capabilities such as pendant ring addition, annulation, heteroatom restating, or scaffold replacement.

### Illegitimate uses

- supplying the winner or its similarity as a task input in an autonomous benchmark;
- deriving a ring or fragment catalog from held-out winners and claiming blind recovery;
- choosing hyperparameters on the same winner-informed cell and reporting it as prospective evidence;
- calling exact reconstruction benchmark success.

Each trajectory should carry one of `autonomous`, `winner_informed`, or `answer_known_reconstruction`. Only the first can support autonomous benchmark claims.

## 10. Smallest decisive development sequence

The next step should not be a long PMO or T4 run. It should isolate whether a correct option-level future target and learned option proposal can recover probability on improving plans before any broad scale-up.

### Phase A: zero-new-oracle policy-congruence audit

Use existing exact option-boundary trajectories and cached scores.

1. Reconstruct boundary states with actual remaining option counts and behavior-policy identifiers.
2. For every boundary with multiple sampled continuations, derive realized best-improvement returns and censoring status.
3. Fit only on training groups, with molecular-source or parent grouping.
4. Compare current score, old future head, endpoint model, and a distributional improvement head on held-out groups.
5. Report rank correlation, Brier score or calibration error for improvement thresholds, top-1 realized improvement, and coverage by option family.

**Pass observable:** the new head improves held-out top-1 realized continuation value and calibration over current score, with improvement not confined to one parent or one option family.

**If positive:** integrate it as an SMC twist and option-proposal advantage target.

**If negative:** do not launch SMC. The available rollouts lack adequate counterfactual coverage; collect more internal option completions or change the option parameterization before buying oracle labels.

### Phase B: zero- or minimal-oracle option-proposal test

On fixed scored parents, generate a locked candidate pool under the same compute allowance from:

- balanced base option prior;
- learned option proposal with exploration floor;
- learned proposal plus persistent option-boundary resampling.

Use cached outcomes where available. Before requesting new calls, measure:

- completed and unique endpoints per proposal second;
- reference and proposal mass assigned to known improving routes;
- option and topology diversity;
- fraction of trajectories reaching intended structural predicates;
- realized duration and failure modes;
- effective sample size and ancestry collapse.

If too few candidates have labels, lock the entire assay before requesting a bounded oracle batch. A positive result requires enrichment of actual improvements, not only more rings or different ancestry.

### Phase C: one paired autonomous round

Only after A or B is positive, compare the strongest existing controller with the full hybrid using identical roots, RNG coupling where valid, candidate and proposal-compute caps, and a frozen batch acquisition policy.

For PMO, report best, top-10 mean, and top-10 AUC from the start of the paired run. For T4, report best feasible docking score, repeat-docking uncertainty, distinct bundle/option coverage, structural deltas, diversity, and proposal time. One positive exposed-cell result authorizes replication, not a benchmark claim.

### Phase D: replication and matched benchmark

- replicate on a new seed before promoting a controller;
- freeze policy, option inventory, acquisition, and budgets;
- evaluate a small task panel chosen without inspecting its outcomes;
- only then run the full PMO or T4 benchmark;
- include all prescreen, policy-training, docking, and proposal costs in the declared comparison.

## 11. Computational design

The recommended implementation can be efficient if the expensive unit is a completed option, not primitive enumeration across a huge shared frontier.

- Batch the graph encoder across all particle boundary states.
- Cache legal primitive fibers by exact state and executor identity.
- Cache option applicability and structural features.
- Lazily sample primitive successors within the selected option support; do not enumerate every raw successor for every particle.
- Compile and execute different particle/options independently across the approved worker pool.
- Deduplicate canonical completed endpoints globally before surrogate or oracle evaluation.
- Keep only a small bounded number of persistent particles per structural stratum.
- Train the small option actor and distributional critic locally or on one accelerator; keep $R_\theta$ frozen.
- Record proposal seconds, executor seconds, model seconds, acquisition seconds, and oracle seconds separately.

The internal rollout budget can exceed the oracle-call budget, as it does in other competitive optimizers, but must remain bounded and reported. Increasing particles without improving the proposal or twist should stop when ESS, improvement enrichment, and top-k candidate quality plateau.

## 12. Scientific interpretation and likely paper contribution

If the hybrid succeeds, the strongest contribution is not "we invented options" or "we applied SMC to molecules." It is the integration of:

- a frozen learned stochastic process over exact executable molecular rewrites;
- local-to-global spatial control through $M$;
- compositional temporal abstraction through structural options;
- policy-congruent future-improvement control on a reversible, trans-dimensional valid-state graph;
- online proposal adaptation and persistent particle search;
- matched oracle-aware acquisition.

The technical claim must remain evidence-dependent. Exact option-level Doob control can be proved for exact backward values on the declared SMDP. Scalable learned control is approximate. Competitive optimization is empirical. Broad support follows from the executor and `generic` channel, not from universal reachability.

## 13. Bottom line

**Is semi-Markov the best option?** Yes as the high-level mathematical state/action abstraction. No as the complete search algorithm.

The most defensible and promising next controller is an option-space, persistent controlled particle policy-iteration system. Its critical new component is not another macro. It is a policy-congruent distributional estimate of future improvement, trained from the exact option policy and used both to improve the option proposal and to twist persistent particles, with correct proposal accounting. A separate calibrated batch acquisition layer spends oracle calls.

The existing negative results are useful because they rule out three shortcuts:

- options without proposal learning;
- SMC with an endpoint-like, policy-mismatched head;
- sparse complete-plan updates without delayed continuation credit.

That narrows the next experiment substantially. Build and validate the continuation target first. If it cannot beat current score on held-out realized suffixes, no amount of particle scaling will rescue it. If it can, test whether it enriches improving complete options under a locked compute budget, then spend a small matched oracle batch.

## Sources

[^1]: Richard S. Sutton, Doina Precup, and Satinder Singh, ["Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning"](https://doi.org/10.1016/S0004-3702(99)00052-1), *Artificial Intelligence* 112, 1999.

[^2]: Ronan Fruit and Alessandro Lazaric, ["Exploration-Exploitation in MDPs with Options"](https://proceedings.mlr.press/v54/fruit17a.html), AISTATS 2017.

[^3]: Markus Wulfmeier et al., ["Data-efficient Hindsight Off-policy Option Learning"](https://proceedings.mlr.press/v139/wulfmeier21a.html), ICML 2021.

[^4]: Jeremy Heng, Adrian N. Bishop, George Deligiannidis, and Arnaud Doucet, ["Controlled Sequential Monte Carlo"](https://arxiv.org/abs/1708.08396), 2019 revision.

[^5]: Benno Kaech, Luis Wyss, Karsten Borgwardt, and Gianvito Grasso, ["Refine Drugs, Don't Complete Them: Uniform-Source Discrete Flows for Fragment-Based Drug Discovery"](https://arxiv.org/html/2509.26405v2), ICLR 2026 paper and 2026 extended appendix, especially Sections 3.3 and A.3.

[^6]: InVirtuoGen authors, [official results and ablation repository](https://github.com/invirtuolabs/InVirtuoGen_results), accessed 2026-09-12.

[^7]: ["Goal-Conditioned Reinforcement Learning as Survival Learning"](https://arxiv.org/abs/2604.17551), 2026 preprint. This is cited as a recent modeling analogy, not as validated evidence for COMPOSE.

[^8]: Marc G. Bellemare, Will Dabney, and Rémi Munos, ["A Distributional Perspective on Reinforcement Learning"](https://arxiv.org/abs/1707.06887), ICML 2017; Will Dabney et al., ["Distributional Reinforcement Learning with Quantile Regression"](https://arxiv.org/abs/1710.10044), AAAI 2018.

[^9]: Eric Mitchell et al., ["Offline Meta-Reinforcement Learning with Advantage Weighting"](https://proceedings.mlr.press/v139/mitchell21a.html), ICML 2021. The memo uses its advantage-weighted regression principle, not its meta-RL problem setting.

[^10]: Stephen Zhao, Rob Brekelmans, Alireza Makhzani, and Roger Grosse, ["Probabilistic Inference in Language Models via Twisted Sequential Monte Carlo"](https://arxiv.org/abs/2404.17546), ICML 2024.

[^11]: Daniel M. Zuckerman and Lillian T. Chong, ["Weighted Ensemble Simulation: Review of Methodology, Applications, and Software"](https://pmc.ncbi.nlm.nih.gov/articles/PMC5896317/), *Annual Review of Biophysics* 2017.

[^12]: Charles-Edouard Bréhier et al., ["Analysis of Adaptive Multilevel Splitting Algorithms in an Idealized Case"](https://arxiv.org/abs/1412.3362), 2015 revision. The reaction-coordinate conclusion is used qualitatively; its theorem is not asserted for COMPOSE.

[^13]: Masashi Okada et al., ["MolLIBRA: Genetic Molecular Optimization with Multi-Fingerprint Surrogates and Text-Molecule Aligned Critic"](https://arxiv.org/abs/2602.07002), 2026 preprint.

[^14]: Logan Ward et al., ["Batched Bayesian Optimization by Maximizing the Probability of Including the Optimum"](https://pmc.ncbi.nlm.nih.gov/articles/PMC12573218/), *Digital Discovery* 2025.

[^15]: Cristian Perez Jensen et al., ["Value Matching: Scalable and Gradient-Free Reward-Guided Flow Adaptation"](https://openreview.net/pdf?id=7iXt44Actj), ICLR 2026.

[^16]: Emmanuel Bengio et al., ["Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation"](https://arxiv.org/abs/2106.04399), NeurIPS 2021.

[^17]: Max W. Shen et al., ["Towards Understanding and Improving GFlowNet Training"](https://proceedings.mlr.press/v202/shen23a.html), ICML 2023.

[^18]: Hyeonah Kim et al., ["Genetic-guided GFlowNets for Sample Efficient Molecular Optimization"](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4b25c000967af9036fb9b207b198a626-Abstract-Conference.html), NeurIPS 2024.

[^19]: Wenhao Gao et al., ["Sample Efficiency Matters: A Benchmark for Practical Molecular Optimization"](https://arxiv.org/abs/2206.12411), NeurIPS 2022 Datasets and Benchmarks.

Additional primary sources consulted:

- Jeff Guo and Philippe Schwaller, ["Saturn: Sample-efficient Generative Molecular Design using Memory Manipulation"](https://arxiv.org/abs/2405.17066), 2024.
- Hyeonah Kim et al., ["Neural Genetic Search in Discrete Spaces"](https://proceedings.mlr.press/v267/kim25b.html), ICML 2025.
- Tong Chen et al., ["pCoMole: Pareto-Constrained Molecule Editing with Discrete Flows"](https://openreview.net/pdf?id=1mCS10EFRq), ICLR 2026 LMRL workshop.
- Maxime Deleu et al., ["Discrete Probabilistic Inference as Control in Multi-path Environments"](https://proceedings.mlr.press/v244/deleu24a.html), UAI 2024.

## Local evidence identities

- `docs/PMO_OPTION_PARTICLES.md`, SHA-256 `4034866dcd3578efab5bb413f53daca52d91fa75a926cd261507d9b14cf50bf6`
- `diagnostics/pmo_option_particles/GUIDE_AUDIT.md`, SHA-256 `5c2cb56b6d711cc624492c77a28d3bb1d346270f989ac638ac5cd0c675f9a631`
- `docs/CONTROLLER_LIVE.md`, SHA-256 `e1a414320e4ee60d53c4e8d8440c393fb27945e842d308626e9e70f32d0ce6ec`
- `docs/MACRO_INVENTORY.md`, SHA-256 `259227764acf04d2bde5f150d314d91a505264ebca6c053add5bc12adc3f484a`
