# COMPOSE author guide

## 1. The paper’s central thesis

The paper should be evaluated around one compact claim:

> COMPOSE is a reversible, trans-dimensional stochastic process defined intrinsically on a bounded edit graph of connected, valence-admissible molecular graphs. A Metropolis-corrected legal forward CTMC converges toward an explicit high-entropy prior, and a hierarchical operator scheduler learns the reverse process for unconditional de novo generation. A user-supplied molecule is an optional initialization for conditional editing, not a requirement of the model.

The strongest contribution is the **conjunction** of:

1. variable atom count, composition, bond order, and cycle topology;
2. pathwise valence and connectivity support under a declared grammar;
3. reversible atomic edit support;
4. an explicit, target-independent de novo prior;
5. operator-family scheduling;
6. exact single-jump CTMC semantics, with multiple issue treated separately as an approximation.

Do not reduce the paper to “a molecule ISA” or “MH in the forward process.” Those are mechanisms serving the larger probabilistic claim.

## 2. Claim hierarchy

### Claim A — pathwise manifold restriction

Every **committed** state lies in the declared state space

\[
\mathcal M_\Gamma = \{G: G \text{ is connected and valence-admissible under } \Gamma\}.
\]

Evidence:

- operator contracts and legality predicates;
- Proposition 1;
- exhaustive or property-based implementation tests;
- all-step internal validity, connectivity, and independent RDKit audit.

Use “connected and valence-admissible under `Γ`,” not the unqualified phrase “chemically valid.”

### Claim B — legitimate unconditional de novo generation

At inference,

\[
G_T \sim \pi_0, \qquad G_T \perp G_{\text{requested}},
\]

and no source molecule is supplied. The initial graph is structured noise, not a scaffold.

Evidence:

- direct prior sampling or independently equilibrated prior samples;
- terminal corruption indistinguishable from independent prior samples;
- low source-atom survival, MCS retention, scaffold retention, and property correlation;
- output quality insensitive to source strata after sufficient corruption.

### Claim C — the chosen prior is controlled rather than implicit

The MH correction prevents the terminal distribution from silently becoming the legal-action-degree distribution of an uncorrected random walk.

Evidence:

- detailed-balance derivation;
- prior-statistic convergence;
- uncorrected-versus-MH terminal discrepancy;
- action-degree, size, and cycle-rank bias plots;
- acceptance rate by operation family and time.

### Claim D — operator scheduling is useful

The normalized clock/operator/operand factorization gives an exact exit rate `λθ` and separates semantically different edit families.

Evidence:

- flat-action-head versus hierarchical-scheduler ablation;
- operation utilization over time;
- calibration of event clock and family rates;
- quality/compute comparison.

### Claim E — multiple issue is a useful approximation

COMPOSE-MI is not the defining stochastic process. It is a conflict-aware, serializable acceleration layer.

Evidence:

- issue width and NFE reduction;
- wall-clock speedup;
- all-step validity under bundled commits;
- two-sample distribution gap from exact single issue;
- quality–cost Pareto curve.

## 3. Why the computer-architecture framing helps

The framing is valuable only where it induces a real object or guarantee:

| Architecture term | COMPOSE meaning |
|---|---|
| Architectural state | Current molecular graph |
| ISA | Typed edit schemas with legality and inverse contracts |
| Instruction | Operator family, operands, and parameters |
| Decoder | Candidate family/operand/parameter heads |
| Scheduler | State- and time-dependent event and operator rates |
| Hazard check | Operand conflict, valence capacity, articulation, and connectivity checks |
| Execution unit | Deterministic graph transition implementation |
| Commit | Atomic exposure of a valid output state |
| Macro-operation | Canonically compiled primitive sequence or atomic compound jump |
| Multiple issue | Serializable bundle of jointly legal instructions |

This framing hurts if it is only vocabulary. Reviewers will reject an “ISA” that is merely a list of actions or a “scheduler” that is merely an ordinary categorical head. The paper therefore leads with the molecular/probabilistic construction and uses architecture terminology as a precise secondary interpretation.

Prefer “multiple issue,” “conflict-aware issue,” or “serializable bundle scheduler.” Use “maximum independent set” only if that optimization is literally solved and pairwise conflicts are sufficient. Molecular valence and connectivity can create capacity and higher-order conflicts, so a hypergraph or integer-program view may be more faithful.

## 4. Main-text versus appendix mathematics

### Main text

The main paper should contain enough mathematics that a reviewer can evaluate the method without opening the appendix:

- exact state space and molecular grammar scope;
- operator/inverse/legality definitions;
- explicit prior energy;
- alias-aware legal proposal and Metropolis-corrected forward generator;
- normalized reverse clock/operator/operand factorization;
- state-transition alias aggregation;
- reverse-path likelihood;
- theorem statements for closure, invariance, reachability, and convergence;
- the endpoint-error decomposition;
- the exact-versus-multiple-issue distinction.

### Appendix

The appendix contains material needed for verification and reproduction rather than the first understanding:

- full grammar and atom/bond tables;
- complete operator contracts and edge cases;
- proofs;
- reachability assumptions and constructive path;
- action-alias accounting;
- prior construction and calibration;
- forward/reverse simulation algorithms;
- time-integral estimator;
- architecture and masking details;
- serializability and conflict constraints;
- all hyperparameters and evaluation definitions;
- expanded diagnostics and conditional-editing experiments.

Because ICLR reviewers are not required to read the appendix, no indispensable empirical conclusion or unstated assumption may live only there.

## 5. The prior to implement

Use an explicit, evaluable high-entropy distribution

\[
\pi_0(G) \propto \mathbf 1\{G\in\mathcal M_\Gamma\}\exp[-U_\eta(f(G))],
\]

where `f(G)` contains only coarse statistics such as:

- heavy-atom count;
- element/charge counts;
- total bond-order budget;
- cycle rank;
- coarse valence-state histogram.

The prior may match broad dataset support but should not encode fingerprints, scaffold identities, exact ring-system frequencies, pharmacophores, nearest-neighbor retrieval, or a learned drug-likeness score. The target learning problem should be to recover high-order chemical organization from low-order valid structured noise.

Practical sampling:

1. draw coarse statistics from smoothed, overdispersed marginals;
2. construct a feasible connected graph, initially via a capacity-respecting tree;
3. add legal closure edges and bond orders;
4. run reversible legal-edit MCMC to reduce constructor bias;
5. maintain a bank of independently equilibrated samples if direct sampling is expensive.

Do not make the all-tree constructor itself the claimed prior. The stationary law, not the initializer, is the prior.

## 6. Forward MH: what must be shown

Forward MH is defensible because MH corrects a proposal toward a chosen invariant distribution; it is not intrinsically a reverse-time algorithm. The paper should call it a **Metropolis-corrected forward reference process**, not “MH diffusion.”

For instruction proposals `a ~ r_t(a|G)`, aggregate all aliases producing the same transition when required:

\[
r_t(G'\mid G)=\sum_{a:T_a(G)=G'}r_t(a\mid G).
\]

The continuous-time Metropolis rate must satisfy detailed balance with `π0`. The implementation must preserve inverse proposal support at each time. For example, scheduling deletion with zero insertion support makes the corresponding corrected deletion rate vanish.

The decisive experiment is not merely endpoint quality. It is whether MH gives a controlled terminal law compared with the uncorrected legal walk.

## 7. Final experimental program

### 7.1 Primary matched benchmark: GuacaMol

Use one frozen preprocessing and evaluator across all matched reruns. Report at minimum:

- validity;
- uniqueness;
- exact train-set novelty;
- raw FCD;
- GuacaMol KL score or clearly named distribution statistic;
- all-step internal validity;
- samples/second and NFE/sample.

Current high-priority comparison set in the draft:

- BWFlow;
- GraphBSI;
- DiGress;
- GruM;
- DISCO;
- Cometh;
- DeFoG;
- ConStruct;
- CoCoGraph;
- GrIDDD;
- COMPOSE.

A row should be removed rather than populated with a non-comparable published number. If a method cannot emit a trajectory, use `--` for all-step validity and explain why.

### 7.2 Secondary distribution benchmarks

Use MOSES and QM9 as secondary checks rather than multiplying headline claims. Include recent molecular comparators where code and protocol permit, particularly FragFM and MolHIT on MOSES. Keep dataset-native metrics, but use one evaluator per table.

### 7.3 Novelty and coverage

Exact novelty is weak. Also report:

- Bemis–Murcko scaffold novelty;
- nearest-training-neighbor fingerprint similarity;
- maximum-common-subgraph ratio;
- internal diversity;
- duplicate rate;
- property and scaffold coverage.

State fingerprint radius, bit length, chirality treatment, MCS timeout, and timeout policy.

### 7.4 Path validity

For every committed forward and reverse state, record:

- internal grammar validity;
- single-component connectivity;
- independent RDKit sanitization;
- failure category;
- operator proposal, legality, acceptance, and execution counts;
- atom-count and cycle-rank excursions;
- path length.

An internal validity rate of 100% is expected by construction; the independent audit is still necessary to detect a mismatch between `Γ` and the chemistry toolkit.

### 7.5 Mixing and source erasure

Compare terminal corruptions `X_T | X_0` with independent `G ~ π0` using:

- two-sample classifier AUC;
- kernel MMD over coarse and fingerprint features;
- source atom survival;
- initial–terminal MCS ratio;
- scaffold retention;
- initial–terminal property correlations;
- convergence from distinct source strata.

The model cannot credibly claim unconditional generation merely because `G_T` is sampled from a graph distribution. It must demonstrate that the forward process erases a particular source and that the independently sampled prior matches the terminal law.

### 7.6 Core ablations

The main ablation table should include:

- single-atom prior;
- random-tree prior;
- coarse Gibbs prior without MH;
- coarse Gibbs prior with MH and flat action head;
- COMPOSE single issue;
- COMPOSE-MI.

Appendix ablations can additionally vary:

- prior feature set and regularization strength;
- operation family removal;
- schedule pairing and time parameterization;
- alias handling;
- constructor equilibration length;
- maximum atom count;
- event-clock parameterization;
- model size and data scale.

### 7.7 Scheduler analysis

Plot proposal, legality, forward acceptance, forward execution, and learned reverse execution by family versus time. Report net atom-count and cycle-rank changes. Nominally supporting insertion/deletion is not meaningful if their realized rates vanish.

### 7.8 Multiple issue

Compare exact single issue and COMPOSE-MI on:

- NFE/sample;
- accepted edits per NFE;
- average and distribution of issue width;
- wall-clock throughput;
- all-step validity;
- FCD/novelty;
- two-sample discrepancy between generated distributions.

Report the approximation as an accelerator, not as exact CTMC simulation unless a compound-jump process is formally modeled.

### 7.9 Conditional editing

Treat source-conditioned editing as a secondary capability after unconditional generation succeeds. Candidate tasks:

- scaffold-preserving decoration;
- property optimization under a similarity constraint;
- formula-changing optimization;
- ring-system modification.

Report success, similarity, edit distance, path validity, and source retention. Do not let this section become the evidence for the de novo claim.

### 7.10 Statistics and compute

- at least three independent training seeds for learned models;
- mean and standard deviation;
- paired bootstrap confidence intervals on a common generated set where possible;
- fixed checkpoint rule selected before test evaluation;
- identical hardware for throughput claims;
- warm-up, batch size, NFE, accepted edits, and wall-clock time reported separately;
- training GPU-hours, peak memory, and parameter count.

## 8. Treatment of preliminary experiments

Pilot work can play one of four roles:

| Pilot outcome | Final disposition |
|---|---|
| Motivates a design choice central to a main claim | Rerun under the final pipeline and include as a main ablation |
| Diagnoses mixing, operator use, or prior behavior | Rerun and place in the appendix |
| Uses obsolete preprocessing/model/evaluator but suggests a hypothesis | Mention no number; test the hypothesis afresh |
| Cannot be reproduced or compared fairly | Omit |

Never label an old pilot value as “preliminary” in a final results table. A submission should read as though all reported evidence was planned and executed under the final evaluation contract.

## 9. Figure plan

A compelling final paper should contain four or five information-dense figures rather than many decorative molecule grids.

1. **Method overview:** data molecule → legal MH corruption → valid prior; independent prior → learned reverse scheduler → generated molecule. Include ISA/scheduler/hazard/commit mapping.
2. **Source erasure and mixing:** terminal classifier AUC, MMD, atom survival, and MCS versus corruption time, comparing MH and uncorrected walk.
3. **Operator schedule:** family-wise forward proposal/acceptance and reverse execution heat maps over time.
4. **Quality–cost Pareto:** FCD or benchmark score versus NFE and wall-clock time; single issue versus multi-issue and current baselines.
5. **Novelty–fidelity analysis:** nearest-neighbor similarity/scaffold novelty versus FCD, plus representative samples selected by a prespecified rule.

The current LaTeX overview is a clean placeholder; a final vector figure with actual molecular states and operator icons would improve reviewer comprehension.

## 10. Baseline discipline

- Separate **matched reruns** from **reported literature values**.
- Record repository, commit, environment, checkpoint, training set, evaluator, sample count, and hardware.
- Do not compare FCD values from different preprocessing or ChemNet implementations without a warning.
- Do not assign all-step validity to methods that do not expose comparable trajectories.
- Give CoCoGraph full credit for valid intermediate states within a fixed formula/degree-sequence fiber.
- Give Morph, GrIDDD, and Edit Flows full credit for variable-size/edit-process components.
- Avoid “first” unless the final literature audit supports the exact narrowed claim.

## 11. Theory/implementation audit before submission

The code must match every theorem assumption:

- every declared primitive is implemented;
- every legal primitive remains in `MΓ`;
- every primitive has the claimed inverse support;
- deletion never removes the sole atom unless an explicit empty state is in the grammar;
- connectivity checks correctly handle articulation vertices and bundled deletions;
- bond-order operations use the exact same valence accounting as the theorem;
- action aliases are aggregated consistently in proposal and likelihood code;
- scheduler masking leaves a normalized distribution;
- the modeled exit rate equals the learned clock under the chosen factorization;
- ring macro semantics expose only states covered by the guarantee;
- multiple-issue bundles have a canonical legal serialization.

Property-based tests should randomly generate valid states, enumerate or sample legal actions, execute/invert them, and compare graph canonicalizations.

## 12. What would make this a spotlight-level submission

A strong result is not merely slightly better endpoint validity. The paper becomes memorable if it demonstrates all of the following:

1. competitive or leading distribution quality against current methods;
2. essentially perfect path validity under both internal and independent audits;
3. convincing source erasure and convergence to the declared prior;
4. a clear empirical reason for MH rather than an ornamental correction;
5. meaningful operator-scheduler interpretability or efficiency;
6. a multi-issue speedup with a quantified approximation gap;
7. theory that accurately describes the released implementation;
8. restrained novelty language and unusually thorough evaluation provenance.

If endpoint quality is modest, the paper can still be strong if the hard-valid trans-dimensional process enables guidance, editing, or efficiency that ambient-space methods cannot reproduce. That advantage must be measured, not asserted.
