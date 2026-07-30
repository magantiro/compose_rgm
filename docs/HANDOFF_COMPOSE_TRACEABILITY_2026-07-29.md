# COMPOSE authoritative handoff traceability — 2026-07-29

## Purpose and precedence

This note records a claim-by-claim analysis of the authoritative COMPOSE handoff pasted on
2026-07-29. It is an interpretation and traceability record, not a replacement for the handoff.

Precedence used here:

1. The pasted 2026-07-29 authoritative handoff controls scientific framing, current run state,
   claim boundaries, and task order.
2. `docs/HANDOFF_RINGCORE_V1_POSTRUN.md` supplies additional completed-run diagnostics that do not
   conflict with the pasted handoff, especially the cycle-opening micro-overfit and the
   family-competition diagnosis.
3. `docs/EXPERIMENT_INFRASTRUCTURE_PLAN.md` and `configs/experiment_registry.yaml` control the frozen
   experimental protocols.
4. Current production code controls what is actually implemented.
5. The current paper is useful framing, but stale empirical or mathematical statements in it do not
   override the handoffs or code.

Status labels:

- **CONFIRMED** — present in current code, a frozen contract, or a committed artifact.
- **HANDOFF-AUTHORITATIVE** — accepted as the current scientific decision; external artifact
  verification may still be pending.
- **PENDING** — required work not yet completed.
- **STALE/CONFLICT** — an older repo statement or current paper sentence that must not guide
  implementation without correction.

No paper or existing handoff file was modified while producing this analysis.

---

## 1. Executive summary

- **CONFIRMED:** The one-sentence thesis, causal chain, and title match the current
  `docs/PAPER1_FRAMING_AUTHORITATIVE.md`, `docs/PAPER_MASTER_PLAN.md`, and the current
  `paper_iclr_stochastic_rewriting/main.tex`.
- **HANDOFF-AUTHORITATIVE:** The fundamental object is a learned stochastic program over executable
  graph-rewrite marks. Editing, Pareto control, Doob control, and RingCore are consequences or
  instantiations, not the paper's definition.
- **HANDOFF-AUTHORITATIVE:** The tagline must be qualified as “Every non-null step is a molecule”
  because the null source is deliberately part of the state space.
- **Implementation consequence:** Training and evaluation work must be judged by whether it
  establishes the complete causal chain, not merely whether a molecular editor emits valid endpoints.

## 2. What is scientifically new

- **CONFIRMED:** Existing framing documents already reject novelty claims for individual atom edits,
  valid editing, Generator Matching, Doob transforms, graph grammars, or Pareto optimization.
- **HANDOFF-AUTHORITATIVE:** The defensible novelty is the formal composition of:
  state-dependent executable marks, Generator Matching, trans-dimensional molecular semantics,
  support closure, canonical-successor pushforward, and exact/dynamic successor-level control.
- **Implementation consequence:** Operator unit tests alone cannot establish the main contribution.
  The experimental program must measure learned transport, trans-dimensional/topological adaptation,
  quotient invariance, and control.

## 3. Formal modeling object

### 3.1 State space

- **CONFIRMED:** The implementation uses padded persistent slots as coordinates while molecular
  occupancy is defined by the active element predicate. `NULL` and `SCAR` cannot be counted as atoms
  by a generic nonzero or `!= NULL` test.
- **CONFIRMED:** The enumerable RingCore implementation explicitly includes a distinguished null
  state and production root insertions from it; tests cover null reachability and null cycle rank.
- **HANDOFF-AUTHORITATIVE:** The semantic state space is
  `X = disjoint_union_{n=0}^N X_n`, with `X_0 = {empty}` and `X_n` for `n >= 1` containing connected,
  supported molecules.
- **HANDOFF-AUTHORITATIVE:** The null graph is not a molecule. It is a reversible source
  `empty <-> one-atom molecules`.
- **STALE/CONFLICT:** The current paper defines the main state space only for `n >= 1`, says every
  state is a molecule, and omits the distinguished null state in its formal development. The paper's
  title/framing is current, but this formal detail is not.
- **Implementation consequence:** Exact-state builders, statistics, plots, and claims must separate
  null from molecular chemistry. “Every state is a molecule” must become “every non-null committed
  state is a molecule.”

### 3.2 Rewrite system

- **CONFIRMED:** Current code implements a state-dependent legal fiber, typed executable actions,
  deterministic executors, and teacher-in-candidate checks.
- **HANDOFF-AUTHORITATIVE:** A mark is the complete operation—rule, concrete match/address, and typed
  payload—not an unconstrained tensor delta repaired after prediction.
- **Implementation consequence:** A teacher outside the exact dynamic candidate set is a contract
  failure, not an ignorable training example.

### 3.3 Marked stochastic process

- **CONFIRMED:** `FactorizedTraceletRateModel` predicts a total hazard and a normalized conditional
  mark law. The family law is masked by the state-dependent executable support.
- **HANDOFF-AUTHORITATIVE:** Hazard governs event timing; the mark law governs event identity.
  Editing consumes the embedded jump law and therefore discards hazard magnitude.

### 3.4 Training objective

- **CONFIRMED:** `factorized_mark_bregman_loss` is
  `Lambda - r * (log Lambda + selected_mark_log_probability)`, importance weighted and averaged.
- **CONFIRMED:** The implemented objective is mark-level for all families except graft, whose teacher
  score already aggregates `graft_successor_groups`.
- **HANDOFF-AUTHORITATIVE:** The molecular-successor process is a pushforward of the trained mark
  process. We must not claim that the completed run directly optimized canonical-successor likelihood.
- **STALE/CONFLICT:** The current paper writes its main RGM objective directly over aggregated
  successor rates and says each minibatch aggregates aliases for the teacher successor. That is not the
  completed production training objective except for graft.
- **Implementation consequence:** The checkpoint leaderboard may use canonical-successor NLL, but its
  report must explicitly distinguish raw mark NLL, successor NLL, and their alias gap.

## 4. Canonical molecular-successor quotient

- **CONFIRMED:** The scientific contract in `experiments/successor_kernel.py` defines the molecular law
  as execution plus canonical aggregation and explicitly records the mark-level-except-graft boundary.
- **CONFIRMED:** `model/segmented_successor.py` is an efficient segmented aggregation implementation,
  while `experiments/reference_successor_kernel.py` is a slow dictionary oracle restricted to tests.
- **CONFIRMED:** Production consumers are required to depend on the shared
  `CanonicalSuccessorKernel` interface rather than reconstructing probabilities independently.
- **HANDOFF-AUTHORITATIVE:** Persistent-slot aliases, symmetries, operand orderings, and future
  primitive/macro refinements must be grouped by canonical molecular successor.
- **HANDOFF-AUTHORITATIVE:** The exact embedded-jump convention conditions on productive,
  non-self/virtual successors. This convention must come from the shared evaluator.
- **CONFIRMED:** Successor-preserving, rate-preserving re-encoding invariance applies only within a
  canonical-successor fiber.
- **HANDOFF-AUTHORITATIVE:** Successor-level controllers are invariant; mark-level power, top-k,
  nucleus, and per-family tilts generally are not.
- **Implementation consequence:** Sentinel and capability reports may include mark-level metrics for
  diagnosis, but checkpoint adequacy and controller claims require successor-level metrics.

## 5. Production molecular implementation

### 5.1 Vocabulary

- **CONFIRMED:** The editing model uses the 15-class broad-organic element–valence vocabulary and a
  40-active-atom cap.
- **CONFIRMED:** Semantic compatible initialization copies shared vocabulary rows by label and leaves
  genuinely new rows fresh.
- **HANDOFF-AUTHORITATIVE:** Base B is an initialization/plumbing artifact, not the final scientific
  model.
- **Implementation consequence:** Initialization parity must compare transferred shared rows and
  inherited capabilities separately from fresh broad-organic rows and fresh cycle heads.

### 5.2 Operators and naming

- **CONFIRMED:** The dense model registry has ten historical family slots; nine are active in RingCore
  and `ring_system_grow` is disabled.
- **CONFIRMED:** Executor names map `bond_insert -> cycle_insert` and
  `bond_delete -> cycle_attach`.
- **HANDOFF-AUTHORITATIVE:** Public semantics are cycle close and cycle open. The internal name
  `cycle_attach` must never be misread as a ring-attachment operator.
- **Implementation consequence:** Every audit and plot must carry an explicit public/internal/executor
  naming map.

### 5.3 Birth/death scope

- **CONFIRMED:** The current atom-insert factorization supports a root insertion and a connected
  insertion with one existing neighbor.
- **HANDOFF-AUTHORITATIVE:** Simultaneous multi-neighbor insertion is out of scope. This removes the
  one-step inverse for some degree-two deletions but does not invalidate Generator Matching.
- **HANDOFF-AUTHORITATIVE:** Exactly three such traces were excluded through the frozen
  representability overlay.
- **PENDING:** Verify those three overlay rows and hashes against the live unified artifact during the
  run provenance audit.
- **Implementation consequence:** Do not “fix” this by silently broadening the operator support during
  checkpoint evaluation or repaired training.

### 5.4 RingCore

- **CONFIRMED:** Ring closing inserts a legal bond; ring opening deletes a legal non-bridge ring edge.
- **CONFIRMED:** The legacy whole-ring growth macro is disabled in the completed run.
- **HANDOFF-AUTHORITATIVE:** Do not restore `ring_system_grow` based on the completed learning
  asymmetry. A future macro is an efficiency proposal, not the definition of topology support.

## 6. Two distinct generative regimes

- **HANDOFF-AUTHORITATIVE:** The completed 16k model is a source-conditioned editing prior sampled as
  a fixed-budget embedded jump chain.
- **HANDOFF-AUTHORITATIVE:** The editing prior is source-agnostic at the network level; the lead is an
  initial condition, while source, goal, and protected mask belong in the controller/value model.
- **HANDOFF-AUTHORITATIVE:** Unconditional de novo generation requires a distinct broad-organic
  checkpoint, nonzero de novo training mass, a calibrated hazard, timed Gillespie sampling, and a
  separate benchmark.
- **STALE/CONFLICT:** The current paper sometimes says the same fitted model is consumed in both
  regimes or that one learned process drives both. The shared framework/operator basis is correct; one
  completed checkpoint serving both regimes is not.
- **Implementation consequence:** Never use the editing checkpoint for E1 or timed-CTMC claims.

## 7. Why every valid intermediate matters

- **CONFIRMED:** The executor/support design makes every non-null committed state a complete,
  supported molecular graph.
- **HANDOFF-AUTHORITATIVE:** Intermediate validity is valuable because it creates intervention,
  scoring, branching, continuation, and pathwise-constraint points.
- **Implementation consequence:** Trajectory-level experiments must record every committed state,
  not only endpoints, and pathwise constraints must be evaluated before committing/scoring a
  successor.

## 8. Exact and approximate stochastic control

- **HANDOFF-AUTHORITATIVE:** Editing control is finite-horizon and remaining-budget indexed:
  `h_b(x) = E[g(X_b) | X_0=x]` and
  `P_b^g(y|x) = P(y|x) h_{b-1}(y) / h_b(x)`.
- **HANDOFF-AUTHORITATIVE:** Controlled support is always a subset of base support. Equality requires
  strictly positive relevant values.
- **HANDOFF-AUTHORITATIVE:** `h_b(x)=0` means unreachable/undefined and must be reported, not patched
  numerically.
- **HANDOFF-AUTHORITATIVE:** Exactness is claimed only on an enumerable finite slice. Learned
  full-scale values are approximations and require calibration/residual reporting.
- **PENDING:** Exact current-registry kernels and solver work follow A2.2b; they are independent of
  editing-checkpoint selection.

## 9. Data lineage and production corpus

- **HANDOFF-AUTHORITATIVE:** The earlier ~6,922-record runs are data-starved engineering baselines,
  not candidates for final scientific claims.
- **HANDOFF-AUTHORITATIVE:** The full corpus has three explicit layers:
  general corruption, cycle operations, and MMP analogues.
- **HANDOFF-AUTHORITATIVE:** Before overlay exclusion the totals are 725,671 traces and 3,370,821
  states; effective training count is 661,105 after excluding three unsupported training traces.
- **HANDOFF-AUTHORITATIVE:** Fixed layer draw weights are `0.40 / 0.25 / 0.35`.
- **CONFIRMED:** The production sampler is trace-first:
  `layer -> trace -> one progress state -> jump or terminal`.
- **CONFIRMED:** Existing progress-family stratification is importance corrected back to the original
  progress law. It changes estimator variance/exposure, not the population family weighting.
- **HANDOFF-AUTHORITATIVE:** Train/validation/test partition by source and scaffold before trajectory
  generation; cross-partition MMP pairs are dropped.
- **CONFIRMED:** Packed storage retains exact slot-addressed states/actions; canonical SMILES is
  identity metadata and cannot reconstruct action coordinates.
- **Implementation consequence:** Any repaired family-aware objective must be explicit about changing
  the population objective; it cannot be described as the existing importance-correct
  stratification.

## 10. Completed scientific training run

- **HANDOFF-AUTHORITATIVE:** Run
  `compose-v4-ringcore-v1-scientific-a7546e2-v1`, commit `a7546e2`, completed 16,000 steps with
  batch size 64, 500-step warmup, and 16,000-step schedule.
- **HANDOFF-AUTHORITATIVE:** Thirty-two snapshots exist at every 500 steps, with no missing,
  duplicate, or unexpected steps; final checkpoint, metrics, and training manifest exist.
- **HANDOFF-AUTHORITATIVE:** Final-relaunch throughput was about 0.726 seconds/step with no further
  preemption/resume.
- **PENDING:** Independently list the live run, hash all 32 snapshots, verify step metadata and
  provenance, and freeze an inventory without modifying the run directory.
- **Implementation consequence:** No trainer changes or new run can substitute for freezing the
  completed baseline.

## 11. Completed-run metrics

- **HANDOFF-AUTHORITATIVE:** Initial validation loss 3.9712; trainer-best raw GM loss 2.7919 at step
  8500; inspected final test loss 2.9708.
- **HANDOFF-AUTHORITATIVE:** Aggregate family metrics improve, but only nine of ten families are
  represented because disabled `ring_system_grow` correctly has no teachers.
- **HANDOFF-AUTHORITATIVE:** The final family table is diagnostic, mark-level, count-imbalanced, and
  not a checkpoint-selection result.
- **HANDOFF-AUTHORITATIVE:** The trainer's internal `selected_step=8500` is not the preregistered
  selection.
- **Implementation consequence:** The existing trainer-selected model must not be silently relabelled
  “best” in the frozen leaderboard.

## 12. Major findings

### Ring closing vs opening

- **HANDOFF-AUTHORITATIVE:** Final mark top-1 is about 0.502 for close and 0.015 for open despite
  similar held-out counts.
- **CONFIRMED BY ADDITIVE POST-RUN HANDOFF:** A cycle-opening micro-overfit reaches family
  probability 1.0 in 50 steps, so the operator, mask, and basic gradient path are not broken.
- **PENDING:** Determine whether the full-run failure is family competition, alias/candidate
  difficulty, labeling/order, representation, or a combination at the successor level.
- **Implementation consequence:** Do not add architecture before the remaining support,
  successor-level, and micro-overfit evidence demands it.

### Graft forgetting

- **HANDOFF-AUTHORITATIVE:** Graft top-3 fell from 0.821 at initialization to 0.000 at the inspected
  final point.
- **CONFIRMED BY ADDITIVE POST-RUN HANDOFF:** Graft received roughly 2.9% of supervision while cycle
  operations dominated; no replay, distillation, differential LR, or frozen-body phase protected it.
- **PENDING:** Trace base B -> initialized RingCore -> all 32 snapshots using one frozen graft
  development probe and successor-level likelihood.

## 13. Current scientific decision

- **HANDOFF-AUTHORITATIVE:** Do not select any checkpoint yet.
- **HANDOFF-AUTHORITATIVE:** Hash/inventory, provenance, full validation leaderboard, family
  trajectories, and ring/graft diagnosis precede the select-versus-retrain decision.
- **HANDOFF-AUTHORITATIVE:** Never select step 8500, step 16000, minimum raw GM loss, or maximum family
  accuracy merely by that label/statistic.
- **HANDOFF-AUTHORITATIVE:** Do not revert to `ring_system_grow`.
- **Implementation consequence:** The next mutations should build missing evaluation/forensic
  infrastructure only after the read-only artifact freeze.

## 14. Immediate forensic program

- **PENDING:** Full 32-snapshot validation leaderboard:
  production-weighted canonical-successor NLL, balanced-family successor NLL,
  held-source/scaffold/topology safeguards, calibration, and hard-gate status.
- **PENDING:** Family curves for close, open, graft, reorder, and optionally ring restate/delete:
  mark NLL, successor NLL, top-1/top-3, MRR, raw/canonical candidates, alias multiplicity, effective
  family weight, and retention from initialization.
- **PENDING:** Cycle-opening support audit over every validation teacher:
  support presence/uniqueness, undirected endpoint normalization, executor/stored-successor equality,
  successor grouping, aliases, candidate counts, random baselines, and close/open matched comparisons.
- **PARTLY COMPLETE:** Cycle-opening micro-overfit is already decisive at mark-family level; it still
  needs the successor-level reporting required by the authoritative protocol.
- **PENDING:** Graft parity/retention audit across base, initialization, and all snapshots.
- **Implementation consequence:** Expensive controller suites do not run per checkpoint.

## 15. Future repaired-training protocol

- **CONDITIONAL:** This section applies only if no completed checkpoint satisfies the frozen
  successor-level and capability safeguards.
- **PENDING:** Gate 0 initialization parity, transferred-row verification, fresh-head identification,
  and active-family teacher/candidate coverage.
- **PENDING:** Gate 1 successor-level tiny-set overfit for every load-bearing family. Cycle opening has
  a strong partial result; graft and remaining families still need it.
- **PENDING:** Gate 2 500-step pilot with automated checks for capability loss, zero gradients, absent
  teachers, non-learning, support explosion, and successor degradation.
- **PENDING:** Gate 3 2,000-step pilot requiring new-family successor gains and acceptable legacy
  retention.
- **PENDING:** Gate 4 full run only after all earlier gates pass.
- **HANDOFF-AUTHORITATIVE:** Candidate repair components are replay, legacy-logit distillation,
  family-aware sampling/objective weighting, differential LRs, frozen-body warmup, and only
  evidence-driven architecture/auxiliary successor losses.
- **HANDOFF-AUTHORITATIVE:** The already inspected final test cannot tune or seal a repaired model; a
  new sealed holdout or external test is required.
- **Implementation consequence:** A top-k-only sentinel is insufficient. It must include
  successor-level retention/learnability and support/teacher/gradient gates.

## 16. Experimental infrastructure plan

- **CONFIRMED:** `docs/EXPERIMENT_INFRASTRUCTURE_PLAN.md` and
  `configs/experiment_registry.yaml` are the controlling plan/registry.
- **CONFIRMED:** The registry protocol hash is `b4cd905a640ab32a`.
- **HANDOFF-AUTHORITATIVE:** One shared canonical evaluator must own checkpoint reconstruction,
  capabilities, persistent slots, legal enumeration, and successor aggregation.
- **CONFIRMED:** The plan explicitly forbids reusing old exact-control scripts/JSONs as evidence and
  distinguishes the editing and de novo lanes.

### E1–E7 implications

- **E1:** Separate broad-organic timed-CTMC checkpoint; editing checkpoint is inadmissible evidence.
- **E2:** Compare learned and uniform distributions over canonical successors on identical support.
- **E3:** Test both growth and shrinkage, including dynamic reversal and fixed-cardinality-impossible
  tasks.
- **E4:** Test cycle creation/opening and topology changes; no-cycle is the key ablation.
- **E5:** Verify quotient invariance and deliberately demonstrate mark-level controller divergence.
- **E6:** Exact finite-state control on the frozen `carbon_6_slots` benchmark.
- **E7:** Same-base controller study with frozen normalization, reference points, budgets, and dynamic
  protocols.

## 17. Exact benchmark status

- **CONFIRMED:** The selected `carbon_6_slots` benchmark has 967 states and 14,432 canonical directed
  edges, with the recorded structural and semantics hashes.
- **CONFIRMED:** This is carbon-only, six-slot, and algebraically useful; it is not representative
  broad-organic chemistry.
- **PENDING:** A2.2b deterministic indexing/action classification/boundary audit/alias compression
  and exhaustive production-vs-dictionary verification.
- **PENDING:** Budget-indexed kernels, exact Doob solver, path multiplicity, target tilts, and later
  learned-checkpoint projection.
- **Implementation consequence:** Exact-control work can continue independently but cannot be used to
  bypass editing-checkpoint selection.

## 18. Cycle-rank and slot-state semantics

- **CONFIRMED:** The authoritative cycle-rank definition is
  `beta_1 = |E| - |V| + components`; null is reported separately.
- **HANDOFF-AUTHORITATIVE:** RDKit ring counts, ring-system counts, and graph cycle rank are distinct
  metrics.
- **CONFIRMED:** Slot-safety tests enforce authoritative element/occupancy predicates and guard
  against `[:n_real_atoms]`, nonzero counts, and `!= NULL` misuse.
- **Implementation consequence:** Every new evaluator/forensic script must use the same predicates and
  cycle-rank function.

## 19. Paper framing

- **CONFIRMED:** The current stochastic-rewriting paper has the correct title, framework-first causal
  chain, trans-dimensional emphasis, canonical quotient, and exact/dynamic control framing.
- **STALE/CONFLICT:** The paper still says all production runs are data-starved. The completed
  large-data 16k run supersedes that empirical-status sentence, though it is still only a diagnostic
  until checkpoint selection.
- **STALE/CONFLICT:** The paper's formal objective is successor-level, while production training was
  mark-level except for graft.
- **STALE/CONFLICT:** The paper omits the distinguished null state from its main state space and uses
  unqualified “every state is a molecule.”
- **STALE/CONFLICT:** Some paper sentences imply one fitted checkpoint serves editing and de novo;
  the current decision requires separate checkpoints/regimes.
- **HANDOFF-AUTHORITATIVE:** These are synchronization items for a later paper pass, not reasons to
  alter the implementation truth or the live user-edited manuscript now.

## 20. Claims to make and avoid

- **MAKE:** learned executable mark process; successor pushforward; semantic trans-dimensionality;
  non-null all-step validity; quotient-level controller invariance; bounded-slice exactness; separate
  editing and de novo regimes.
- **AVOID:** novelty for individual edits; literal molecule status for null; full-scale exactness;
  exact inverse closure for every deletion; unconditional claims from editing; production
  representativeness of the carbon slice; final-model status for base B; mark-level refinement
  invariance; automatic selection of step 8500.
- **Implementation consequence:** Reports and checkpoint metadata should encode the regime and
  capability contract so these claim boundaries cannot be blurred downstream.

## 21. Non-negotiable implementation boundary

- **CONFIRMED:** Reuse production checkpoint loading, executor, legal enumerators, persistent-slot
  semantics, canonical identity, capabilities, and segmented successor aggregation.
- **HANDOFF-AUTHORITATIVE:** Write experimental registry/task/solver/controller/metric/statistical logic
  independently; do not copy old exactness evidence.
- **Implementation consequence:** The slow dictionary kernel is a test oracle only, never a production
  result path.

## 22. Immediate task order

The controlling order is:

1. freeze/hash/provenance-check the completed run;
2. validation-only successor leaderboard and family forensics;
3. cycle-open/graft diagnosis;
4. select an existing checkpoint only if all frozen safeguards pass;
5. otherwise run gated repaired-training pilots;
6. continue A2.2b/exact control independently;
7. run E2/E3/E4/E7 only after editing checkpoint freeze;
8. run the separate de novo lane, including the deferred de novo `bond_reorder` repair.

This order supersedes an earlier implementation impulse to build the repair trainer before proving that
all 32 completed checkpoints are inadequate.

## 23. Explicit prohibitions

All of the following are active implementation constraints:

- no automatic step-8500 selection;
- no tuning on the inspected test set;
- no whole-ring-growth restoration before/against the forensic evidence;
- no family top-k as the sole capability metric;
- no architecture change before relevant micro-overfit/support tests;
- no independent reimplementation of the molecular kernel in experiments;
- no unconditional result from the editing checkpoint;
- no representative-chemistry claim for the carbon slice;
- no null-is-a-molecule language;
- no post-output changes to objectives, normalization, HV references, or task thresholds;
- no controller suite over every snapshot;
- no dismissal of the 16k run as wasted.

## 24. Directive distilled into a decision rule

The completed run is a valid large-data diagnostic baseline, not yet a selected model.

Freeze an existing snapshot **only if**:

1. it wins or is admissible under the frozen validation canonical-successor rule;
2. ring opening is acceptable at successor level or its mark/successor gap is fully understood;
3. graft retention is adequate;
4. held-source/scaffold/topology safeguards pass;
5. learned transport beats uniform legal rewriting at matched support and budget.

Otherwise:

1. preregister a repaired objective and gate thresholds on development data;
2. establish initialization parity and per-family successor overfit;
3. run 500 and 2,000 steps with live abort gates;
4. train a long run only after both pilots pass;
5. evaluate on a fresh sealed holdout.

The main thesis remains invariant under either branch:

`executable rewrites -> Generator Matching -> trans-dimensional molecular process -> canonical successor
kernel -> exact and dynamic control`.
