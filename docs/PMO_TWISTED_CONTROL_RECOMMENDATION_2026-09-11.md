# Option-level particle control for COMPOSE

## Recommendation

Develop a controller that learns from successful executable continuations and
uses twisted sequential Monte Carlo (SMC) to preserve several promising futures
through multi-edit transformations. Keep the learned molecular reference frozen.
Adapt controller-side continuation values and, when separately admitted, its
proposal policy. Preserve local-to-global region allocation and generic editing.

This is a proposed direction, not a demonstrated PMO improvement. The immediate
implementation is narrower: repair the discrepancy between executor-valid
development routes and the deployed proposal support. The existing SMC machinery
from the QED experiment is reusable, but its QED value head and output-accounting
rules cannot simply be transplanted into PMO.

The September 9 recommendation already identified twisted SMC, reusable values,
and policy iteration. The change here is to make particle survival and learned
multi-edit proposals the next explicit mechanisms to test, instead of treating
another endpoint reranker as a test of future-aware control.

## Evidence and failure localization

The completed ordinary-proposal experiment at source revision
`07b59ead03939210f6121bb9d9b1c923d5284f13` did not improve the best retained
perindopril-MPO score, 0.5222329678670935. It used 272 new oracle calls and took
1218.66 seconds. Its future and immediate arms retained identical scores from
the five original starts. This is a negative development result for that
small achieved-return head and two-option comparison, not a test of twisted SMC.
The result artifact is `pmo_trajectory_value/97b2acd334d2a8dcdda5f0dbede4f033a66813da3dc7da67b82dad53faf4fca0/result.json`
on `compose-v4-artifacts`; the downloaded plain payload has SHA-256
`647aeee64661a743a3c0fa5f35613da132a790f6ca2d2a7e542dc18122caeabf`.

The same artifact found the desired next canonical successor in only seven of
ten inspected production rows. Three initial decisions were missing, despite
successful unrestricted executor replay. Code inspection identifies aromatic
non-leaf atom deletion and a partial ring-restatement pattern as the relevant
exclusions. Cyclic bond reordering is also absent from the deployed primitive
enumerator. Therefore, decomposing an unavailable ring restatement into cyclic
bond reorders would not repair the problem. First-free-slot atom insertion adds
a separate coordinate-correspondence issue, not necessarily missing chemistry.

The target routes are answer-known development examples. Their existence does
not establish shortest routes, ordinary proposal probabilities, autonomous
discovery, or molecular reachability outside the declared support. The
[support-repair protocol](PMO_ROUTE_SUPPORT_REPAIR.md) records the bounded
production-law compilation and its unresolved outcomes separately.

| Layer | Question | Appropriate evidence |
| --- | --- | --- |
| Molecular support | Can the executor represent and execute the change? | Exact stored-state forward replay |
| Proposal support | Does the deployed law offer the transition or a supported alternative path? | Positive production probabilities and exact replay |
| Proposal frequency | Does ordinary sampling generate the useful option and attachment? | Unconditioned draw frequencies, not target-filtered enumeration |
| Continuation survival | Do particles retain enabling edits until useful endpoints appear? | Genealogies and completed options under matched work |
| Task feedback | Are useful observed outcomes changing later decisions? | Chronological paired optimization, not fitted prediction accuracy |

All five questions matter. A sampler cannot create a zero-support transition;
positive support alone can still imply an impractically rare useful trajectory.

## Literature comparison

The source survey covers established control methods and primary publications
available through September 11, 2026. External performance is reported evidence
in each paper's setting, not a guarantee for COMPOSE.

**Controlled SMC.** Heng and colleagues formulate proposal adaptation as control
and approximate backward recursions with dynamic programming. This supports
learning how to allocate particle computation, but its approximation guarantees
require assumptions and sufficient function approximation and sampling quality.
It does not imply that a small learned head is accurate. [1]

**Twisted SMC for discrete generation.** Zhao and colleagues use future-potential
estimates to shape intermediate targets and proposals. Their general importance
weight contains both a reference/proposal ratio and the twist ratio. The target
normalizer can be estimated without bias under the stated sampling conditions;
finite-particle normalized samples are still approximate. Learned twists help
efficiency, not support expansion. This is the closest computational starting
point for COMPOSE. [2]

**Feynman-Kac steering.** Singhal and colleagues separate the proposal generator,
intermediate potentials, and resampling schedule. Their diffusion experiments
support testing delayed resampling and comparing better guidance against more
particles. Neither the diffusion endpoint estimator nor every practical formula
should be copied mechanically: COMPOSE can score complete intermediate graphs,
and its option state and termination conventions differ. [3]

**InVirtuoGen.** Its optimization stack combines genetic exploration, policy
updates, and adaptive length selection. The important comparison is therefore
not fixed proposal sampling versus a more elaborate selector. Its Table 2
separates no-prescreen PMO from the prescreened regime in Table 1, and Appendix
B.3.2 adds a separate unlimited-oracle-pretraining regime. The August 2026
revision explicitly identifies that appendix as a post-review addition. [4]

**Genetic-guided GFlowNets.** This method combines exploratory molecular search
with replay-based policy learning. The reusable lesson is to turn independently
found improvements into future proposal probability. Its string-construction
state graph is not COMPOSE's cyclic, variable-size edit graph. A backward policy
or trajectory-balance equation must account for that distinction; assuming every
molecular deletion has a one-step inverse would be incorrect. [5]

**Retrospective Backward Synthesis.** This goal-conditioned GFlowNet work creates
successful training trajectories from specified goals and decomposes difficult
goals. It motivates using development winners to learn executable continuation
skills. For COMPOSE, every reversed or reconstructed route must be verified in
the forward deployed law. Training on a winner makes its later recovery a
supervised development result, not a blind benchmark success. [6]

**Discrete Adjoint Matching.** This ICLR 2026 method derives optimal discrete
rate control and adjoint estimators. Its scalable realization exploits masked
diffusion structure. COMPOSE's fixed-jump, irregular molecular successor graph
does not supply those simplifications automatically. It is a useful mathematical
relative, but replacing the present controller with its full machinery would
add a substantial derivation and implementation dependency before testing the
current proposal defect. [7]

**Recent molecular adaptation studies.** A July 2026 preprint studies acquisition,
reward shaping, debiasing, replay and validity in an online loop. An August 2026
preprint finds useful gains from elite-supervised updates, but also reports that
training on the generator's own elites alone can fail to discover improvements.
These are not settled universal conclusions. Together they motivate a practical
comparison: search that discovers new improvements plus controller learning,
against the same search without learning. Neither establishes that an elaborate
RL loss is necessary, or that SMC alone is sufficient. [8,9]

## Mathematical specification

### A declared path law

Let `s` include the exact molecule, source context, selected region, option,
option phase, atom provenance needed by the option, remaining primitive budget,
and remaining option decisions. Include a stopping flag. This augmentation is
necessary: the same molecule midway through two different programs need not
have the same admissible next actions.

Define a normalized reference `B(s'|s)` on this augmented state. Region selection
uses the existing WHERE law. WHAT initially uses the balanced applicability-aware
prior with generic permanently active. HOW uses the option-restricted production
law, aggregated over equivalent successors before nonlinear state guidance.
Administrative choices and executed primitive transitions are recorded separately.

This is an **option-augmented reference**, not automatically the original raw
`R_theta` path law. Conditioning on macro support changes probabilities even when
model weights stay frozen. A compound option is one planning decision but
executes multiple valid primitive transitions; it is never an endpoint teleport.

For fixed horizon `H`, choose nonnegative terminal desirability `g_z(s_H)` and
target

\[
\pi_z(s_{0:H}) \propto P_B(s_{0:H})g_z(s_H).
\]

Stopping and failed programs must have explicit boundary rules. A completed
molecule is not automatically a completed option. PMO's reward and T4's
feasibility predicates belong to their declared task contracts; T4 endpoint
constraints must not become intermediate molecular support restrictions.

The ideal future potential is

\[
h_b(s)=\mathbb E_B[g_z(S_b)\mid S_0=s],\qquad h_0=g_z.
\]

Its normalized row is `B(s'|s) h_{b-1}(s') / h_b(s)`. The recursion concerns the
chosen option-augmented reference, not maximum observed reward along a teacher
path. The latter is useful supervised information but estimates a different
quantity. [1,2]

### Twisting without double-counting

For any normalized proposal `q(s'|s)` covering the target support and a strictly
positive intermediate potential `psi_b`, use incremental weight

\[
w \leftarrow w\,
\frac{B(s'\mid s)}{q(s'\mid s)}
\frac{\psi_{b-1}(s')}{\psi_b(s)},\qquad \psi_0=g_z.
\]

With `q=B`, only the twist ratio remains. With a proposal already tilted by the
potential, the `B/q` correction is essential. Exact path-density accounting is
needed for parameterized programs; an attachment score or a single primitive's
probability is not a macro probability. [2,3]

Retain the existing systematic resampling and low-ESS trigger where applicable.
Compute ESS within the correct particle population and preserve weights between
resampling events. Canonical duplicates can share computation, but their
multiplicity and weight cannot disappear. A single terminal weighted draw and
the best member of an oracle archive are different outputs with different claims.

The existing kappa=1 bound can constrain an individual proposal row. Subsequent
particle weighting/resampling changes the effective selected path law, so it
does **not** automatically preserve that bound for the effective controlled
transition. An SMC development contract must state whether KL limits proposal
computation or the final controller. This is not a silent reinterpretation of
the frozen region controller.

### Learning and temporal commitment

Use complete options as synchronization points for outer particle competition.
Keep primitives and option phase explicit internally. This permits a ring
construction or ring retyping program to pass through temporarily unattractive
states without asking an immediate endpoint score to endorse every primitive.
Program completion still does not guarantee a good task score.

Estimate continuation value from completed continuations, with the data-generating
policy and horizon recorded. A reference expectation requires reference rollouts
or valid importance correction; unweighted optimized trajectories are not such
targets. Known-target paths can instead supervise an explicitly goal-conditioned
proposal, using graph correspondence and option phase. That separate supervised
skill must be evaluated beyond its training routes before it guides blind tasks.

First adapt a small controller-side scorer over canonical successors and
completed options. An additional distilled proposal is justified only if measured
sampling frequency or enumeration cost is the bottleneck. Do not retrain the
reference, add a winner-derived template vocabulary, or train a new region/option
model merely because those actions are technically possible.

## Local-to-global search and chemistry

Keep region bundles as the outer allocation unit. A long global trajectory must
not acquire extra WHERE probability merely by yielding more intermediate states.
Record intended release scale separately from realized graph change. Resample
particles within bundles first; allocate oracle slots across diverse viable
bundles with a declared rule.

The relevant chemistry is broader than ring addition: ring opening and reclosure,
aromaticity changes, heteroatom restating, bond-order changes, pendant relocation,
atom deletion for size recovery, functional-group editing, and repeated
construction all matter. Existing macro families provide proposal channels, not
a finite molecular vocabulary. Every claimed capability needs an actual
supported example under the 40-active-atom, charge-preserving, non-stereochemical
production scope. No claim of arbitrary ring or molecule reachability follows.

The earlier developmental paths suggest that delayed credit and proposal gaps
both matter. They do not establish that the original long routes are chemically
minimal. It may be much easier to reach the same endpoint using a different
sequence of options than to preserve every intermediate chosen by a compiler.

## Efficient experimental sequence

1. Finish the current zero-oracle support repair. Record where each original
   route first fails, not just a pass fraction. If exact intermediate rejoining
   prevents repair, test a supported alternative between wider structural
   waypoints. Do not confuse failure of one compiler with unreachable endpoints.
2. On cached development neighborhoods, compare unguided particles, immediate
   guidance, and future-guided particles. Do not restrict ordinary proposals to
   known winning edges. Use known routes only to measure where useful probability
   or ancestry disappears. This is a mechanism diagnostic, not PMO performance.
3. Run a small, matched, target-free PMO comparison once proposal availability
   and continuation behavior are understood. Lock the new budget and stopping
   rule before outcomes. Count every new oracle evaluation, including training
   labels, even when the objective is cheap. Preserve the same-generator,
   endpoint-only baseline and a no-learning particle baseline.
4. Retain only improvements in actual queried score, top-ten archive quality and
   time. A changed ranking or a completed ring alone is not success. Follow a
   useful development result with multiple seeds and objectives fixed beforehand.
5. Freeze the selected controller for the matched no-prescreen PMO evaluation.
   Previously inspected winners/tasks remain development-exposed and must be
   disclosed. Aggregate benchmark comparison is distinct from independent
   task-generalization evidence.

The optimization target is the official query-indexed top-ten AUC, not exact
rediscovery of a figure molecule. Warm-start probes are not from-scratch
benchmarks. Model pretraining, oracle pretraining, initialization screening,
online queries, and internal computation require separate accounting. [4]

No method reviewed guarantees discovery of an IVG winner in a finite budget.
The defensible ambition is a controller that repeatedly converts executable
search and real feedback into improved proposal distributions, with enough
particle diversity to survive the intermediate edits that make those
improvements possible.

## Sources

1. Heng, Bishop, Deligiannidis and Doucet. [Controlled Sequential Monte Carlo](https://www.stats.ox.ac.uk/~doucet/HengBishopDeligiannidisDoucet_controlledSMC.pdf). Annals of Statistics, 2020; sections 3–5.
2. Zhao, Brekelmans, Makhzani and Grosse. [Probabilistic Inference in Language Models via Twisted Sequential Monte Carlo](https://arxiv.org/pdf/2404.17546). ICML 2024; sections 2–4, especially equations 9–12.
3. Singhal et al. [A General Framework for Inference-time Scaling and Steering of Diffusion Models](https://arxiv.org/html/2501.06848v2). 2025; section 3 and appendix C. The displayed appendix C.3 inequality for skipping resampling appears inconsistent with the usual low-ESS trigger; do not copy it in place of COMPOSE's tested trigger.
4. Kaech, Wyss, Borgwardt and Grasso. [Refine Drugs, Don't Complete Them: Uniform-Source Discrete Flows for Fragment-Based Drug Discovery](https://arxiv.org/html/2509.26405v2). ICLR 2026; August 3, 2026 revision, sections 3.3–3.4 and appendix B.3.
5. Kim, Kim, Choi and Park. [Genetic-guided GFlowNets for Sample Efficient Molecular Optimization](https://arxiv.org/html/2402.05961v4). NeurIPS 2024; December 30, 2024 revision, sections 2–3.
6. He, Chang, Xu and Pan. [Looking Backward: Retrospective Backward Synthesis for Goal-Conditioned GFlowNets](https://arxiv.org/html/2406.01150v2). ICLR 2025; section 3.
7. So, Karrer, Fan, Chen and Liu. [Discrete Adjoint Matching](https://arxiv.org/html/2602.07132v1). ICLR 2026; sections 2–3, including the masked-process specialization.
8. Chen et al. [On the Design Space of Discrete Diffusion Online Adaptation for Molecular Optimization](https://arxiv.org/html/2607.02834v1). July 3, 2026 preprint, submitted to NeurIPS; sections 2–4 and appendix D. Not treated as an accepted conference result.
9. Wa et al. [Elite-Weighted Supervised Fine-tuning for Goal-Directed Molecular Optimization](https://arxiv.org/html/2609.00189v1). August 31, 2026 preprint; sections 3–4. Not treated as an accepted conference result.

The supplied workshop manuscript and the local September 9 research memo are
context, not independent verification of manuscript performance claims. This
recommendation changes neither their historical evidence nor any frozen run.
