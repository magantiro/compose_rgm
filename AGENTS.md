# AGENTS.md: Production Generative Machine Learning Research

Read this file completely before doing anything. It is the working contract for this repository.
Project-specific plans, acceptance criteria, data contracts, and decision logs may narrow this
contract, but they may not silently weaken its scientific-validity rules.

## Project identity

Before implementation begins, record:

- the scientific or product problem;
- the primary model output;
- the central claim that the project is intended to test;
- the experimental or operational setting in which that claim will be validated;
- the primary baselines and causal ablations;
- the declared support of the model, including data, representation, size, modality, and domain limits.

Keep the platform identity distinct from any one experimental instantiation. A comparison or ablation
may explain why a method works without becoming the identity of the project.

## Authorized scope

The current milestone, and only the current milestone, defines authorized implementation scope.

- Read the applicable task and its acceptance criteria before editing.
- Do not start later-stage model training, data generation, deployment, or experiments merely because
  prerequisite code exists.
- If a task appears to require a meaningful scope expansion, stop and report the dependency.
- Do not weaken a scientific gate, alter a split after inspecting test results, or broaden a claim to
  make a milestone appear complete.
- Prefer the smallest complete change that satisfies the contract.
- Avoid speculative abstractions, framework adoption, and unrelated cleanup.

Record project-specific prohibitions here, such as:

- training runs that require explicit authorization;
- datasets or endpoints that cannot be used for selection;
- upstream repositories that are read-only;
- claims that require prospective evidence;
- frozen transforms, registries, schemas, or evaluation protocols.

## Scientific-validity rules

These rules take precedence over green tests and attractive metrics.

1. Never relax a preregistered or frozen gate to make it pass. Fail loudly and report the negative
   result.
2. Split data before extracting reusable components, templates, fragments, motifs, statistics, or
   learned preprocessing state. Deriving these objects from held-out examples and then measuring
   recovery is leakage.
3. Fit scalers, imputers, feature selectors, target transforms, vocabularies, calibration models, and
   other learned preprocessing on training data only.
4. Do not select architectures or hyperparameters using outer-test results. Use training and
   calibration evidence, then evaluate the frozen choice once on the outer test.
5. Preserve random splits only as diagnostics when the intended use involves scaffold, component,
   family, source, temporal, or other structured distribution shifts.
6. Audit class, source, family, and template imbalance before sampling. Do not sample raw records when
   a dominant family would collapse the learned distribution. Use an explicit, justified weighting or
   balancing policy.
7. Distinguish model support from training density. An architecture may represent objects outside an
   enumerated corpus even when the training mixture concentrates probability near that corpus.
8. Never claim universal expressibility. State that examples are representable only within the
   declared vocabulary, topology, size, conditioning, and representation support.
9. Never silently reduce support to make an implementation fit in memory or run faster. Report the
   failure and change the parameterization, batching strategy, or scientific scope explicitly.
10. Identify degenerate metrics or rules that appear strong because they match nearly everything.
    Exclude them from headline conclusions and preserve the exclusion rationale.
11. Report both coverage and precision when evaluating recovery, decomposition, retrieval, routing,
    or constraint satisfaction. Coverage alone does not establish correctness.
12. Distinguish observed, inferred, enumerated, generated, verified, and prospectively validated
    evidence. Do not promote one category into another.
13. Treat robustness stress tests as secondary unless the scientific claim explicitly makes them a
    hard requirement.
14. Report failed, null, blocked, and abstaining outcomes with the same discipline as positive
    results. Negative results are first-class deliverables.
15. If labels are unavailable, use the data only for its legitimate role, such as representation
    learning, applicability analysis, or distance analysis. Do not invent targets or interpret
    proximity as predictive validation.
16. Keep applicability domains explicit. A model may be selected as the best tested model while still
    being unauthorized for guidance or deployment in unsupported domains.
17. When uncertainty is used for decision-making, validate its calibration under the shifts relevant
    to intended use. Do not maximize raw predicted means in regions where uncertainty is unearned.
18. Distinguish predictive association from causal or mechanistic evidence.
19. Do not fabricate, synthesize, or silently substitute missing source data. Report the exact missing
    asset and expected identity or hash.
20. Keep upstream projects or source repositories immutable when the project consumes them as
    versioned inputs. Vendor or reference exact revisions rather than modifying upstream history.

## Generative-model design rules

- Define the generated object and its representation precisely. State whether generation is atom,
  token, graph, sequence, image, route, program, component, or whole-object generation.
- Keep generated-object support separate from downstream evaluator support. A narrow oracle, route
  corpus, reward model, or verifier must not silently redefine the support of a broader generator.
- If supervision is dense for one level and sparse for another, use a data-aligned factorization.
  Jointly model densely paired variables and use conditional, hierarchical, retrieval-based, or
  hybrid reasoning for sparsely supervised variables.
- A modular or hierarchical model is not post-hoc merely because its modules are separate. Guidance
  is integrated when downstream value changes transition probabilities before the candidate is
  locked.
- Preserve a same-generator post-hoc baseline. Hold the base generator fixed and apply assessment only
  after sampling to isolate the causal effect of in-loop guidance.
- Match compute and candidate budgets where possible. Report generator calls, evaluator calls,
  verifier calls, wall-clock time, accelerator time, and final experimental budget.
- Sweep guidance strength. Evaluate objective value together with validity, diversity, novelty,
  support coverage, calibration, and collapse toward common training components.
- Choose the operating point by maximizing the target objective subject to frozen floors on diversity,
  realism, broad-distribution coverage, and novelty.
- Separate whole-object novelty from component, motif, scaffold, or combination novelty.
- Demonstrate open-endedness on actual generated and locked candidates, not only through an
  architectural argument.
- Define abstention and failure states explicitly. A missing-knowledge failure is different from an
  impossible, incompatible, or out-of-scope candidate.
- Use a frozen candidate lock before prospective evaluation. Report every attempted candidate and all
  deviations from the planned workflow.

## Data contracts and provenance

- Every source asset must have a documented origin, license or access basis, version, and SHA-256 hash.
- Keep observed data, programmatic enumeration, inferred labels, synthetic negatives, and prospective
  outcomes in distinguishable fields or datasets.
- Never describe a structure corpus as a reaction corpus unless records contain the required reaction
  objects.
- Preserve source identifiers and row-level provenance through cleaning, deduplication, reconciliation,
  and splitting.
- Deduplicate using the representation appropriate to the scientific question. Preserve excluded or
  merged identities in a reconciliation ledger.
- Freeze train, calibration, and test assignments before model fitting. Validate zero group leakage
  under every grouping variable used by the evaluation.
- Record exclusions with reason codes. Do not delete inconvenient examples without an auditable
  ledger.
- Version schemas. Document intentional schema changes and provide migration logic or an explicit
  incompatibility error.
- Sort mappings and collections in serialized artifacts when order is not meaningful.
- Use deterministic serialization where possible. Generated artifacts should be byte-stable across
  reruns in the supported environment.

## Code organization

- Keep domain logic in typed, testable functions or small dataclasses.
- Keep command-line parsing, orchestration, network access, and filesystem I/O thin.
- Separate:
  - configuration and policy;
  - data loading and validation;
  - pure transformations;
  - model definitions;
  - training loops;
  - evaluation and calibration;
  - artifact serialization;
  - command-line entry points.
- Read settled domain definitions, transforms, roles, vocabularies, and policies from versioned
  registries or configuration. Do not duplicate them in code or retype them from memory.
- Validate inputs at boundaries. Error messages must identify the bad file, field, value, or invariant.
- Avoid broad exception handling. Do not hide errors or silently coerce invalid records.
- Avoid partial outputs. Compute into a temporary location, validate, then publish the complete
  artifact atomically or through an equivalent safe handoff.
- Preserve backward compatibility unless the task explicitly changes a contract.
- Add dependencies only when the standard library and existing dependencies are insufficient. Place
  optional tooling in the appropriate dependency group.
- Do not introduce a framework when plain functions and the existing stack are sufficient.
- Keep public interfaces small and explicit. Avoid hidden global state.
- Make device, dtype, precision, worker count, seed, and determinism settings explicit.
- Keep inference and training configuration serializable and hashable.
- Do not commit credentials, tokens, private keys, personally identifying information, or
  machine-specific secrets.

## Compute efficiency and portability

- Use available CPU and GPU hardware efficiently.
- Batch and vectorize hot paths.
- Avoid repeated parsing, Python loops over tensors, unnecessary host-device transfers, and implicit
  synchronization.
- Keep memory bounded through streaming, chunking, memory mapping where appropriate, and configurable
  batch sizes.
- Profile representative inputs before non-trivial optimization.
- Record a representative before-and-after benchmark for performance changes.
- Never gain speed by dropping records, shrinking scientific support, changing labels, or weakening
  gates without an explicit scientific decision.
- Mixed precision and nondeterministic kernels are opt-in. Require a numerical-equivalence or
  decision-equivalence check for affected outputs.
- Make resume and checkpoint behavior explicit for long-running work.
- Checkpoints must include model state, optimizer and scheduler state when applicable, epoch or step,
  random-number generator states, configuration, data identity, and software version.
- A resumed run must not silently change data order, objective, schedule, or random state.
- Design expensive analyses for independent partitions when valid, but ensure deterministic reduction
  and bounded concurrency.
- Report throughput, peak memory, hardware, precision, batch size, and representative input scale.

## Testing

Add tests in proportion to risk:

- a focused happy-path test for public behavior;
- a regression test for every fixed defect or protected scientific invariant;
- boundary and malformed-input tests when failure might otherwise be silent;
- leakage tests for splits, preprocessing, and reused components;
- determinism and provenance assertions for scripts that produce numbers;
- schema assertions for persisted JSON, CSV, checkpoint, or database artifacts;
- round-trip tests for encoders, decoders, transformations, or serialization;
- numerical tests with justified tolerances;
- CPU tests for core logic, plus targeted GPU tests where device behavior materially differs;
- resume and interruption tests for long-running jobs;
- tests that gates fail when their prerequisites are not met.

Testing rules:

- Prefer small fixtures and behavior-level assertions.
- Do not copy production logic into tests.
- Tests must not require network access.
- Mark tests that require vendored data and skip only when the documented asset is absent.
- Never skip or weaken a test because its result is inconvenient.
- Run the narrowest relevant test while iterating.
- Do not use the repository-wide suite as an iteration gate. Freeze the milestone candidate first,
  run focused dependency-level tests after each repair, then run the full suite once at the milestone or
  scientific-launch boundary. Rerun the full suite only when later code changes can affect its result.
- When a known multiprocessing or native-runtime test is denied by sandbox policy, do not retry the
  suite or repeatedly retry inside the sandbox. Run only the exact affected node once in the documented
  environment that provides the required semaphore, subprocess, filesystem, or accelerator capability.
- Run formatting and lint checks for touched code.
- Run the repository-wide verification and test suite before declaring a milestone complete.
- Inspect generated artifacts and the final diff.
- Never describe an unrun check as passing.

## Reproducible numerical artifacts

Every script that produces a scientific number must record:

- exact input paths or immutable identifiers;
- SHA-256 hashes of all material inputs;
- code revision;
- complete configuration;
- deterministic seeds and seed derivation;
- software and relevant library versions;
- hardware and precision where they can affect results;
- sample counts and exclusions;
- split identities;
- output schema version;
- timestamps when operational state matters.

Write numeric findings to a stable task-specific result directory. Pair machine-readable artifacts with
a concise human-readable interpretation. A result without provenance is not an authoritative result.

## Definition of done

A task is complete only when all of the following hold:

1. The written acceptance criteria are met by a runnable command.
2. Focused tests, formatting, lint, repository verification, and the required full test suite pass.
3. Numeric findings are stored in versioned machine-readable artifacts with input hashes.
4. The decision log records what was measured or decided, including negative results.
5. Documentation identifies limitations, residual risk, and unresolved human decisions.
6. Generated artifacts and the final diff have been inspected.
7. No unrelated user work was overwritten.
8. Changes are committed in intentional, reviewable units.
9. The branch is pushed only after verification and only when authorized.

## Git and change discipline

- Treat existing changes as user-owned unless their origin is known.
- Do not overwrite or revert unrelated work.
- Keep commits scoped to one coherent concern.
- Separate data corrections, implementation, tests, generated artifacts, and documentation when that
  improves reviewability.
- Use descriptive commit messages that state the outcome.
- Do not combine unrelated completed work into one large commit.
- Do not use destructive history or worktree commands without explicit authorization.
- Inspect `git status`, the staged diff, and `git diff --check` before committing.
- Report the commits, verification performed, and any unpushed state.

## Scientific writing

### Build an evidence ledger first

Classify every material claim:

- **Measured:** directly observed in an experiment or primary source data.
- **Computed:** produced by a versioned and reproducible analysis.
- **Reported:** taken from a verified external primary source.
- **Inferred:** an explicitly labeled interpretation of measured, computed, or reported evidence.
- **Proposed:** future work, design choice, hypothesis, or untested mechanism.

Use computed results only when their artifacts record input hashes and required verification passed.
If evidence is missing, state the gap or omit the claim. Never invent a citation, count, yield,
uncertainty, protocol outcome, benchmark, or causal explanation.

### Preserve claim boundaries

- State denominators, split definitions, units, uncertainty, and evaluation conditions alongside
  headline numbers.
- Distinguish retrospective computation from prospective validation.
- Report negative and null results with the same prominence as positive outcomes.
- Separate predictive association from causal or mechanistic claims.
- Identify model and data applicability limits, including support, modality, size, source,
  experimental setting, and endpoint.
- Treat human review and PI decisions as unresolved until recorded.
- Prefer primary literature and primary data.
- Keep quotations minimal and verify them exactly.
- Do not turn architecture, intent, feasibility, or planned experiments into completed results.
- Use settled terminology consistently.

### Match prose to the section

- **Results:** lead with the finding, then give quantitative evidence, robustness checks, and the
  limitation.
- **Methods:** provide enough information to reproduce the work. Name frozen inputs, splits,
  parameters, software, seeds, exclusions, and decision rules.
- **Discussion:** interpret without repeating every result. Distinguish evidence, inference, and
  proposal.
- **Captions:** make each figure or table interpretable without the main text. Define cohorts, panels,
  statistics, units, and error bars.
- **Abstract and title:** claim no more than the completed evidence supports. Use cautious framing until
  prospective evidence exists.
- **Reviewer response:** answer directly, point to changed evidence or text, and acknowledge valid
  limitations without rhetorical defensiveness.

### Style

- Use precise, compact scientific prose.
- Prefer concrete subjects and strong verbs.
- Keep terminology consistent.
- Organize each short paragraph around one claim.
- Lead with the scientific or practical problem and outcome before implementation detail.
- Put mathematical and architectural rigor where technical reviewers can inspect it without making the
  abstract inaccessible to the field's primary audience.
- Avoid promotional adjectives, vague novelty claims, anthropomorphism, and claims of universality.
- Avoid comparison-first framing when the platform contribution is broader than one ablation.
- Do not use em dashes in manuscript prose. Use commas, parentheses, colons, semicolons, or separate
  sentences.
- Define every acronym on first use.
- Verify every number against its source artifact.
- Verify every citation against the cited source.
- Flag every sentence that depends on unresolved human judgment.

## Reporting work to collaborators

Lead with the outcome. Report:

- what changed;
- the exact checks that ran;
- the authoritative artifact or file;
- the scientific interpretation;
- negative findings and abstentions;
- remaining risks or decisions;
- the next safe action.

Do not present ongoing, blocked, or unverified work as complete.

## COMPOSE project-specific contract

This section instantiates the general contract for this repository. It does not replace the frozen
registries, run contracts, decision artifacts, or task-specific acceptance criteria. When a current,
self-hashed contract is stricter than this prose, the contract governs. When documents disagree, stop,
identify the conflicting revisions, and resolve the lineage instead of choosing the convenient version.

### Objective and primary claim

- **Scientific problem:** learn an executable stochastic molecular editing process whose committed
  non-null states are complete supported molecules and whose transitions can change atom identity,
  cardinality, connectivity, and ring topology.
- **Primary model output:** a state- and time-dependent marked rate law over complete legal rewrite
  actions. Executing and canonicalizing the marks induces a molecular-successor kernel.
- **Central claim under test:** COMPOSE learns an executable, trans-dimensional stochastic rewrite
  process over molecular graphs and turns its induced canonical molecular-successor process into a
  substrate for exact and dynamic molecular design.
- **Platform identity:** Rewrite Generator Matching and COMPOSE are the framework. Lead optimization,
  Pareto exploration, dynamic retargeting, exact Doob control, and topology or size adaptation are
  experimental consequences, not the identity of the framework.
- **Primary validation setting:** source-conditioned molecular editing under a fixed embedded-jump
  budget, followed by exact finite-horizon verification on a bounded persistent-slot graph and dynamic
  multiobjective design experiments. Unconditional de novo generation is a separate timed-CTMC lane
  with its own checkpoint, source distribution, hazard requirements, and gates.

### Current authorized milestone

**Scoped option-controller continuation-bank preparation, 2026-09-12:** after
reviewing the small `+0.0112784407` plan-pool gain, the user agreed to stop
scaling that unchanged complete-plan generator and proceed to the smallest
decisive test of the actual persistent WHERE/WHAT/HOW controller. This
authorizes a zero-oracle local preparation and implementation of a locked,
balanced-reference option-boundary continuation-bank experiment on existing
exact Perindopril development parents. It may reuse the already qualified
inference package and existing paid labels for prediction-only diagnostics.
No new oracle call, Modal deployment, model fitting, reference or executor
change, or benchmark claim is authorized by this preparation scope. The remote
proposal run and any later scoring require a separately recorded exact compute
and call authorization. `BUILD_RING_SYSTEM` remains disabled.

**Scoped PMO plan-pool assay authorization, 2026-09-12:** the user explicitly
authorized scoring the already locked Perindopril MPO plan-pool prevalence
assay. This permits exactly 95 new PMO oracle calls, one for every candidate in
the immutable compiled lock with file SHA-256
`4f43b5a0b51942ae12e72ef0393eddfaec10f75fdf5c75ec6f7b7037963aeeb7`.
Run the existing restart-safe local scorer once and publish the complete result
and per-candidate receipts. No candidate regeneration, relocking, adaptive
selection, additional call, Modal deployment, docking call, controller fitting,
or broader benchmark run is authorized. An ambiguous started receipt must fail
closed rather than be retried.

**Scoped option-controller runtime integration, 2026-09-12:** after reviewing
the completed offline core, the user explicitly requested proceeding with the
missing end-to-end implementation. This authorizes production-state feature
extraction, WHERE/WHAT/HOW composition through the existing hierarchy and
option continuation kernel, exact persistent population advancement, and
locked-candidate acquisition wiring. It also authorizes local model-free
fixtures and cache-only dry runs. It does not authorize new oracle calls, Modal
deployment, accelerator fitting, reference/executor changes, or a benchmark
claim. Learned guidance must remain fail-closed unless supplied with immutable,
validated actor and value snapshots. The inherited `BUILD_RING_SYSTEM`
compound-option decision remains unresolved, so this integration must consume
the caller's frozen applicable option set without activating that program.

**Scoped option-controller construction authorization, 2026-09-12:** the user
explicitly requested implementation of the full controller architecture selected
in `docs/CONTROLLER_LITERATURE_DECISION_2026-09-12.md`. The immediate milestone
is the offline, reusable controller core and its retrospective gate described in
`docs/OPTION_CONTROLLER_V1.md`: compositional option decisions, a
policy-congruent distributional improvement target/model, a conservative learned
option proposal with explicit reference probability, persistent option-boundary
SMC state with proposal correction, and batch acquisition over locked completed
candidates. Local CPU fitting may use existing paid development outcomes and
exact stored trajectories. This authorization permits zero new oracle calls, no
Modal deployment, no accelerator training, no reference-model or executor
change, and no benchmark claim. A later live paired run still requires its own
prospective recipe and budget authorization. `BUILD_RING_SYSTEM` support may be
implemented but remains disabled by default until the inherited compound-option
decision is recorded.

**Scoped learned-plan comparison authorization, 2026-09-11:** the user approved
trying the proposed learning search policy and then requested continued work.
Authorize one warm Perindopril development comparison: four rounds, sixteen
parents per arm, frozen versus online-updated complete-plan policy, at most 128
new oracle calls total, CPU only, at most 29 workers plus one driver and $10.
Initialize the controller from existing paid labels, preserve the frozen reference
and executor, and retain the broad reference channel. No public winner enters
training or the population. `docs/PMO_PLAN_POLICY.md` records the exact recipe;
this supersedes the completed offline-only scope below, not the editing-V2 gates.

**Scoped offline edit-chooser authorization, 2026-09-11:** the user's latest
approval authorizes implementing and fitting a context-sensitive complete-edit
controller on existing paid development outcomes, including damaging edits and
separately recorded compilation failures. The immediate scope is local CPU data
preparation, source-grouped/chronological retrospective evaluation, and focused
checks, with zero new task-oracle calls and no remote training or deployment.
`docs/PMO_EDIT_CHOOSER.md` records this bounded recipe. No reference-model,
executor, benchmark-gate, or editing-V2 training-contract change is authorized.
The completed replication below is not authority for another online run.

**Scoped PMO replication authorization, 2026-09-11:** the user approved the
requested unchanged fresh-seed replication after the first local-guidance
comparison passed its best-score criterion. Authorize one paired Perindopril
MPO replication, seed 20261009, four rounds, at most 128 new oracle calls across
both arms, 29 CPU workers plus one driver, no GPU, and $10 reserved cost.
Preserve the original sixteen starts, 116 donors, fitted endpoint model and
proposal mixture. Reuse previous paid scores only when requested, separately
from the original proposal exclusions. Record a distinct replication contract
and preserve the first run. Focused checks, clean committed source, strict
preflight, deployment then durable spawn, and complete accounting remain
required. This does not authorize another task, model training, additional
replications, or a no-prescreen benchmark.

**Scoped PMO authorization, 2026-09-11:** after the explicit request to permit the
prepared four-round comparison at 128 new oracle calls, 30 containers and $10,
the user replied "ok well lets just go ahead" and "what are you waiting for".
This authorizes recording this amendment, deploying
`modal_apps/pmo_local_guidance_app.py`, and durably spawning the single paired
Perindopril MPO development comparison in `configs/pmo_local_guidance.json`
(contract `0682c9efc6dea9051ef8bee34a6d0d3dca4f95b84eba8cf8285e6a4381bb07aa`).
The limit is 128 new oracle calls across both arms, at most 29 workers plus one
driver, no GPU, and $10 total reserved cost. Reuse the prepared inputs, frozen
endpoint model, compatible caches and completed focused checks; retain strict
preflight, clean committed source, input hashes, executor replay, candidate locks
and full oracle accounting. This is a bounded development run, not a declaration
of a completed milestone or a waiver of release/full-suite verification. It does
not authorize another task, extra rounds, fresh model training or a broader
benchmark. The frozen process and editing-V2 training ladder remain unchanged.

**Completed scoped T4 comparison, 2026-09-12:** the authorized warm-start
paired frontier comparison on PARP1 seed0, delta=0.4 is complete. It used 12 of
the allowed 40 new docking calls. The in-loop arm produced eight eligible
endpoints and reached -10.0; post-hoc produced four and reached -9.9. The
machine-readable review is `diagnostics/t4_frontier_compare/result.json`.
This result does not authorize another round, reference-model training, or a
broader benchmark.

As of the current editing-V2 rebuild, the authorized sequence is:

1. finish and freeze the content-addressed editing corpus derivatives;
2. build and validate whole-trace Active8 admission;
3. rebuild Gate 0 structural evidence on the exact admitted corpus;
4. run the T1 successor-level capacity and micro-overfit gate;
5. run the bounded P50 pilot only after its semantic prerequisites authorize it;
6. advance to P500 and P2000 only after their preceding decision artifacts pass;
7. authorize a long run only after the staged retention and learnability gates pass.

The current existence of trainer, launcher, or checkpoint code does not authorize a later stage. A stale,
null-threshold, `NO_GO`, mismatched, or merely hash-valid prerequisite is not training authority.

### Scoped evidence reuse and minimal reruns

- Gate 0, T1, P50, P500, and P2000 are scientific decision points. Implement each with the minimum
  sufficient computation and artifact chain that answers its declared question while preserving every
  authorization, leakage, provenance, and acceptance requirement in this contract.
- Record scientific semantics, data or panel contents, optimization or evaluation recipe, and operational
  execution code as separate identities when they differ. A launcher, logging, scheduling, retry, or
  publication-only change must not invalidate a scientific result unless it changes that result's declared
  computational dependency closure.
- Before launching any materialization or fan-out, inventory existing local and remote artifacts and test
  whether they satisfy the requested scientific dependency closure. Reuse compatible complete units and
  rerun only missing, corrupt, or scientifically changed units. A new Git commit, launcher revision, schema
  wrapper, or output directory is not by itself a reason to repeat molecular computation.
- Address deterministic scientific caches by their narrow semantic and implementation dependency closure,
  not by the whole repository revision. Record the exact Git commit as provenance, but do not put unrelated
  documentation, orchestration, logging, or test changes into a cache key.
- When a prior artifact contains all information required by a new compact schema, prefer a deterministic,
  validated conversion over re-enumeration. The conversion must bind the old artifact hash, selection or
  stream identity, process identity, executor and canonicalization identities, target schema, and converter
  implementation. It must prove the converted payload equals direct computation on a bounded oracle sample.
- Active8 admission, Gate 0 reductions, exact fibers, prepared panels, and collated tensors are reusable
  when their scientific identities and dependency closures remain unchanged. A downstream model-scoring
  change does not authorize recomputing those unaffected inputs.
- Publish T1 evidence independently by family and training scope. Re-evaluate only families whose forward
  computation, trainable parameter route, objective, panel, thresholds, initialization semantics, or
  scientific dependencies changed. Do not replay a passing family solely because an unrelated
  family-specific parameter or operational file moved a global hash.
- Reuse a family result across model revisions only through a machine-checkable containment or equivalence
  artifact that binds both revisions and proves that the relevant initialization, forward law, trainable
  parameter route, objective, panel, and thresholds are unchanged. A human assertion, matching tensor
  shape, or zero-initialized parameter alone is insufficient. When this proof is not cheaper and clearer
  than a bounded rerun, rerun only the affected family and scope.
- Prefer typed content-addressed manifests of compatible per-family receipts over hard-coded serial chains
  of predecessor result hashes. A failed unit may route the next scientific action without invalidating
  unaffected passing units.
- Cache expensive deterministic intermediates, publish restart-safe progress, and retry only missing,
  failed, corrupted, or scientifically affected units. Remote work that could otherwise conceal progress
  or force material recomputation must emit heartbeats and cost-relevant progress receipts.
- Split expensive workflows into explicit `prepare-only` and `train-only` commands. CPU preparation must
  publish and validate an immutable reusable artifact before any accelerator is allocated. GPU training must
  consume that artifact directly, perform no corpus scan or molecular re-enumeration, and start only through
  a separate explicit launch after preparation is reviewed. A combined convenience driver may orchestrate
  both only when it preserves this boundary and never leaves an accelerator idle during CPU work.
- Before remote execution, record the task census, concurrency, representative task time, expected wall
  time, expected cost, restart unit, and stop limit. Prefer one reusable production task as the benchmark;
  do not run disposable canaries when an exact bounded equivalence benchmark and reusable task execution
  already answer the risk.
- During a narrow repair, run focused tests for the affected dependency and launch surface. Do not rerun a
  repository-wide suite for an unchanged exact commit, and do not repeatedly run it while iterating on a
  later commit. Run the required repository-wide suite once on the frozen milestone candidate, and rerun it
  only when subsequent code changes can affect its result.
- Do not delay a bounded scientific run for unrelated cleanup, generic framework construction, additional
  hardening, or mutation testing beyond the current named threat and acceptance criteria. Record such work
  as follow-up unless it blocks correctness, reproducibility, leakage control, or safe recovery of the
  current run.

### Project-specific forbidden actions

- Do not launch P50, P500, P2000, a long editing run, or a de novo run without the exact current
  prerequisite and authorization artifacts.
- Do not launch a Modal scientific job from a dirty serialized-code tree. Use a clean committed
  worktree or clone, bind the exact commit, use the required detached or spawned execution mode, and
  preserve the content-addressed output namespace.
- Do not mention an AI assistant in commit messages or pull-request text. Use the repository convention
  `<module>: <one-line imperative, lowercase, no trailing period>` when it applies.
- Do not select a checkpoint by raw Generator-Matching loss, mark-level family accuracy, final step, or
  a trainer-internal `selected_step`. Apply the frozen validation-only canonical-successor rule.
- Do not use the already inspected legacy final-test aggregates to select or repair a model. A repaired
  model requires a new sealed final holdout or external final evaluation.
- Do not use an editing checkpoint as evidence for unconditional generation.
- Do not re-enable `ring_system_grow` or add a finite whole-ring template catalog merely to improve one
  metric. Any macro proposal is an optional efficiency ablation and requires its own support analysis.
- Do not enable `ring_system_delete` in the current Active8 pilot without a recorded support and task
  decision. It is outside the current eight-family pilot.
- Do not silently add multi-neighbor atom insertion. Current supported birth is root insertion or
  insertion with exactly one existing neighbor. If a task requires bond subdivision or another
  multi-attachment birth, record the support expansion and redesign the factorization explicitly.
- Do not alter executor semantics, legal fibers, persistent-slot identity, canonicalization, operator
  mappings, charge policy, or representability rules to make a gate pass.
- Do not reconstruct exact slot-addressed states from canonical SMILES for training or replay.
- Do not independently reimplement molecular-successor probabilities in experiment scripts. Use the
  one production evaluator and the independent dictionary implementation only as a bounded test oracle.
- Do not reuse old exact-Doob scripts, old exactness JSONs, stale pre-RingCore figures, data-starved
  checkpoints, or old task selections as scientific evidence.
- Do not apply mark-level top-k, nucleus, power, or per-family control and describe it as
  representation-invariant molecular control.
- Do not call the null state a molecule, claim exactness at full molecular scale, claim universal
  expressibility, or claim every deletion has a one-step insertion inverse.
- Do not change objectives, normalization, hypervolume references, task thresholds, budgets, or dynamic
  protocols after inspecting corresponding evaluation outputs.

### Frozen terminology and process semantics

- The semantic state space is a disjoint union over active atom cardinality and includes a distinguished
  reversible null source. The padded persistent-slot tensor is a coordinate representation, not the
  semantic dimension.
- The model parameterizes a stochastic process over complete executable rewrite **marks**. The completed
  16,000-step RingCore-V1 diagnostic run used mark-level Generator Matching. The editing-V2 bounded
  training lane instead declares a canonical-successor objective for the productive embedded jump chain,
  with any hazard term separate and explicit. Bind objective identity in contracts and checkpoints. Do not
  describe the historical run as successor-trained or silently resume a mark-objective checkpoint under
  the successor objective.
- The molecular kernel is the pushforward of the mark law through the production executor and canonical
  molecular identity, with all marks in a successor fiber summed.
- Successor-level control is invariant to refinements that preserve aggregate mass within one canonical
  successor fiber. Arbitrary mark-level controllers are not.
- `cycle_insert` is the model-family name for executor `bond_insert` and means ring closing.
  `cycle_attach` is the model-family name for executor `bond_delete` and means ring opening. The inherited
  name `cycle_attach` does not mean attaching a ring.
- Use graph cycle rank `|E| - |V| + c`. Keep it distinct from SSSR ring count and ring-system count.
- Determine real atoms with the authoritative element predicate. SCAR is occupied but is not an element.

### Declared bounded editing support

- **Object:** connected, supported molecular graphs represented by exact persistent-slot states.
- **Vocabulary:** the declared broad-organic 15-class element-valence vocabulary, including represented
  sulfur, phosphorus, and halogen classes. The legacy CNOF four-class model is not the production
  scientific model.
- **Size:** at most 40 active atoms in the current production editing contract.
- **Charge and valence:** the frozen production charge, aromaticity, bond, and valence policies.
- **Stereochemistry:** out of scope in the current editing-V2 corpus contract. Do not interpret the 2D
  molecular graph law as stereochemical generation or preservation evidence.
- **Formal-charge changes:** out of scope. The current editing contract is charge-preserving only.
- **Current Gate0, T1, and P50 Active8 development families:** `atom_insert`, `atom_delete`,
  `atom_restate`, `bond_reorder`, `bond_reroute`, `cycle_insert`, `cycle_attach`, and
  `ring_system_restate`.
- **Disabled in the bounded pilot:** `ring_system_delete` and `ring_system_grow`.
- Active8 is a development-only operator freeze for Gate0, T1, and P50. It does not authorize final
  production support. Final operator support remains undecided until the registered reachability,
  capacity, and downstream pilot evidence is recorded.
- **Birth factorization:** zero-neighbor root birth and exactly one-neighbor connected birth.
- Every non-null committed state must be valid, connected, and within declared support. The null source is
  reversible but is not a molecule.
- This support declaration does not imply uniform training density, one-step inverse closure, reachability
  of every broad-organic molecule, or representative drug-like chemistry in the exact carbon benchmark.

### Data sources and immutable boundaries

- Use exact slot-addressed source states, semantic actions, exact successors, and canonical molecular keys
  only as identity metadata. Never make SMILES reparsing the source of truth.
- Partition by source and scaffold before trajectory generation. For molecular matched-pair records, both
  endpoints must belong to the same partition; record and drop cross-partition pairs.
- The editing-V2 corpus contract declares five evidence lanes:
  `observed_local_analogue`, `operator_aware_real_endpoint`,
  `linker_positional_topology_analogue`, `real_endpoint_multistep_path`, and
  `reversible_synthetic_walk`. The multistep lane name is deliberate: genuine observed-series provenance
  is not currently available, so compiled paths between real endpoints must not be labelled as observed
  series paths.
- Preserve endpoint, pair-relationship, path, intermediate, and action-sequence evidence as separate
  component-level fields bound by a versioned evidence profile. Do not collapse them into one scalar
  `evidence_class`; real endpoints do not make compiler-generated actions or intermediates observed.
- Preserve the legacy corruption, cycle-operation, and matched-pair inputs as distinguishable provenance
  sources where they contribute records, but do not substitute the legacy three-layer mixture for the
  editing-V2 five-lane capability contract.
- Sample through the declared hierarchy `data_lane` to `series_scaffold_or_source_group` to
  `semantic_capability_cell` to `endpoint_pair_or_path` to `progress_state`. Raw pair-uniform sampling is
  forbidden. The corpus contract remains `DESIGN_NOT_TRAINING_AUTHORIZED`; freeze exact lane and cell
  weights, stream identity, effective teacher coefficients, and path-position coefficients before
  optimization. Report all importance corrections.
- Apply representability, charge, and Active8 decisions as immutable whole-trace admission where the
  contract requires it. A disallowed middle action excludes every neighboring progress row from that trace.
- Bind packed shards, overlays, source-corpus cache, unified manifest, operator capability, support
  contract, sampler, scheduler, codec, and implementation identities by physical and semantic hashes.
- Preserve train, validation, controller-validation, and final-test roles. Do not use test aggregates for
  threshold setting, recipe selection, architecture choice, or early stopping.
- Treat the completed 16,000-step RingCore-V1 run as a scientifically valid diagnostic baseline, not as a
  selected production model.

### Training and early-abort gates

- Gate 0 must verify exact architecture and capability identity, active teachers and candidates, inherited
  initialization parity where applicable, and corpus or support provenance.
- T1 must measure canonical-successor capacity for every load-bearing family on bounded development
  panels. A family that cannot overfit its small successor-level panel blocks a larger run.
- Bounded T1 preparation requires one exact-commit successor-partition materialization, immutable
  internal validation, the completed Active8 sentinel, and bounded independent dictionary-oracle
  tests. Re-enumerating the same prepared fibers with the identical compiler is a diagnostic, not a
  prerequisite or source of training authority. Do not represent same-compiler repetition as
  independent algorithmic evidence.
- The pre-P50 capacity gate uses the prospectively frozen unique-state, single-target canonical-successor
  panel. Repeated-state empirical-law fitting is a separate conditional calibration gate. It may be run
  only when every counted transition has an authoritative independent-observation multiplicity receipt.
  Raw record duplication and mark-alias multiplicity are not empirical transition counts. Absence of such
  receipts blocks empirical-law claims, but does not turn fabricated frequencies into a P50 capacity gate.
- A scratch model is allowed. Warm-starting is not a scientific requirement. If used, inherited
  capabilities require explicit retention probes, replay or distillation where justified, and drift gates.
- Monitor family-resolved canonical-successor NLL, teacher-successor probability and rank, raw mark count,
  canonical successor count, alias multiplicity, effective update mass, gradient exposure, and retention
  relative to initialization. Aggregate loss is insufficient.
- The first bounded pilot must stop on unsupported teachers, missing candidates, zero or nonfinite
  gradients, corpus or stream identity drift, inherited-capability collapse, new-family non-learning,
  exploding support, nonfinite loss, or validation divergence.
- Freeze exact address and training streams before optimization. A resumed run must reconstruct the same
  stream and all optimizer, scheduler, RNG, data, and implementation identities.
- Run P50 before P500, P500 before P2000, and P2000 before a full schedule. Do not continue merely because
  the process remains numerically stable.

### Primary baselines and causal ablations

- This list is a summary. `configs/comparator_registry_v1.json` is the complete comparator authority. Its
  `PREIMPLEMENTATION` status declares required planned arms, not completed or runnable baselines.
- **E2 transport:** learned canonical-successor transport versus both uniform over canonical molecular
  successors and the state-independent empirical-family law, with the learned unguided prior as the
  reference arm, at matched support, seeds, source set, and edit budget.
- **Editing control:** same frozen base prior with unguided sampling, endpoint reranking, greedy one-step
  reward, local Boltzmann and static scalarized guidance, a declared MOG-DFM-style comparator where
  faithfully runnable, sequential Monte Carlo or Feynman-Kac control, and learned Doob or value control.
- **Post-hoc causal baseline:** keep the generator fixed and evaluate or rerank only after sampling.
- **E3 cardinality ablations:** full COMPOSE, no insertion, no deletion, and fixed cardinality.
- **E4 topology ablations:** full RingCore, no cycle operations, and a finite-catalog comparator only when
  a legitimate, support-honest implementation is available.
- **E3 and E4 attachment ablation:** include the registered no-`bond_reroute` support ablation.
- **E7 Pareto search:** include NSGA-II over COMPOSE successors as a required same-base comparator, with
  MOEA/D and the AReUReDi-style adaptation only under their registered conditional gates.
- **External small-molecule baselines:** apply the registry's adapter and task-compatibility gates to
  InVirtuoGen, GenMol, GraphGA, MARS, RetMol, and HN-GFN. Do not claim equal support or substitute
  published numbers when task, oracle, constraint, budget, and selection contracts differ.
- **E5 quotient ablations:** slot relabeling, within-fiber refinement, aggregate successor-mass checks,
  sampled successor frequencies, controlled-law invariance, and the negative mark-level controller
  counterexample.
- Match edit, oracle, successor-scoring, particle, wall-clock, and accelerator budgets where required.
  Report unmatched compute explicitly.

### Primary endpoints and experimental gates

- Checkpoint selection uses production-weighted validation canonical-successor NLL as the primary
  criterion and balanced-semantic-cell canonical-successor NLL as secondary. Family-balanced metrics are
  diagnostics only. Apply the frozen held-source, scaffold, topology, calibration, and
  productive-rollout safeguards.
- E1 de novo generation has its own broad-organic timed-CTMC checkpoint. Primary endpoints are FCD,
  held-out chemical-space recall, and ring or topology distribution distance. Endpoint validity and
  all-step validity are separate hard gates, together with connectedness, uniqueness, and sanitization.
- E2 primary endpoints are held-out canonical-successor NLL, analogue or target recovery at matched
  budget, and path overhead.
- E3 primary endpoints are atom-count target success, edits or oracle calls to success, and retained
  source similarity.
- E4 primary endpoints are topology target success, valid-path rate, and edit efficiency.
- E5 primary endpoint is quotient-level successor-mass and controlled-law invariance, accompanied by a
  demonstrated mark-level negative result.
- E6 establishes exact finite-horizon identities only on the frozen bounded verification graph. Report
  unreachable states where the backward value is zero instead of numerically patching them.
- E7 primary endpoints are normalized hypervolume and hypervolume area under the curve versus all oracle
  calls. Report IGD+, feasible nondominated count, source similarity, diversity, held-out evaluation,
  dynamic adaptation, and prefix reuse as declared secondary metrics.
- Dynamic experiments include preference switching, Pareto fan or branching, and pathwise constraints.
  Remaining-budget values, not original-budget values, govern continuation after a switch.

### Authoritative files and artifact locations

- Local traceability for the 2026-07-29 authoritative pasted handoff:
  `docs/HANDOFF_COMPOSE_TRACEABILITY_2026-07-29.md`. It records precedence and claim boundaries but is not
  a replacement for the pasted handoff.
- Historical RingCore-V1 diagnostic handoff: `docs/HANDOFF_RINGCORE_V1_POSTRUN.md`. Use its completed-run
  measurements only when they do not conflict with the newer handoff, current code, or current self-hashed
  editing-V2 contracts. Its branch, live-run, context-file, and historical suite instructions are stale.
- Experimental build plan: `docs/EXPERIMENT_INFRASTRUCTURE_PLAN.md`.
- Frozen experiment registry: `configs/experiment_registry.yaml`, loaded through its validating API rather
  than hand-parsed in scientific code.
- Paper framing and title: `docs/PAPER1_FRAMING_AUTHORITATIVE.md` and the applicable portions of
  `docs/PAPER_MASTER_PLAN.md`. Stale mathematical or empirical paper text does not override the handoff,
  current code, or self-hashed contracts.
- Evidence status: `docs/CLAIM_LEDGER.md` plus authoritative machine-readable artifacts. Revalidate its
  entries against current evidence before using them in manuscript prose.
- Training authorization and recipe contracts: current files under `configs/`, plus their physically
  hashed prerequisite artifacts. Filenames alone do not establish currency.
- Decision-log policy: prospective scientific decisions live in task-specific, self-hashed contracts under
  `configs/`, such as `editing_corpus_v2_contract.json`, `editing_training_v2_gate.json`, and
  `ring_operator_decision_v1.json`. Measured gate decisions and deviations live in task-specific,
  provenance-bound artifacts under `diagnostics/coherence/` or the registry-declared `results/` path and
  must name the governing contract hash. `docs/DEVIATION_REGISTER.md` is a historical register at its
  declared source revision, not a current global decision log unless revalidated.
- Machine-readable diagnostics belong under task-specific `diagnostics/` paths; final E1-E7 experimental
  outputs belong under their registry-declared `results/` paths.
- Remote data and checkpoint artifacts live in content-addressed namespaces on the declared Modal volume.
  Preserve exact workspace, volume, path, commit, and SHA-256 identities.

### Repository-specific verification

- User-authorized T4 development policy (2026-09-08): bounded controller development
  runs use focused tests for changed dependencies and scientific invariants, plus
  clean-source, input-hash, candidate-lock, and budget checks. Do not make an
  unrelated repository-wide suite a launch blocker or rerun unchanged expensive
  tests merely for ceremony. Full-suite verification remains a milestone/release
  requirement, not a prerequisite to every bounded T4 development experiment.
  Preserve failures and incomplete checks honestly; never waive scientific gates,
  leakage controls, executor validity, or oracle accounting under this policy.
- While iterating, run the narrowest focused tests for every touched invariant.
- For the current Python environment, use `.venv/bin/python -m pytest` and
  `.venv/bin/python -m ruff check` so the repository-pinned toolchain is used.
- On macOS, set `KMP_DUPLICATE_LIB_OK=TRUE` and `OMP_NUM_THREADS=1` for the full scientific suite when the
  native dependency stack requires the documented OpenMP guard.
- Run `git diff --check`, inspect generated artifacts, and inspect the exact staged diff before committing.
- Run repository-wide verification before declaring a milestone complete. Record the actual pass, fail, and
  skip counts. Do not reuse historical suite counts as current evidence.
- Scientific jobs require the applicable prelaunch or release-gate command, a clean exact source revision,
  and verification of every physical prerequisite. A green unit-test suite alone does not authorize
  training.

### Decision and reporting discipline

- Record operator, corpus, architecture, recipe, threshold, checkpoint, and experimental-protocol decisions
  before inspecting the outputs they govern.
- Record negative findings such as ring-opening collapse, graft forgetting, failed gates, unreachable
  control states, unsupported tasks, or noncompetitive baselines without hiding them.
- Keep the evidence ledger synchronized with manuscript prose. Results placeholders remain placeholders
  until authoritative artifacts exist.
- The current core thesis is fixed unless an explicit scientific revision is recorded:

  > COMPOSE learns an executable, trans-dimensional stochastic rewrite process, pushes its mark law to a
  > canonical molecular-successor kernel, and uses that kernel for exact and dynamic molecular design.

## Portability note

Before adopting this file in another repository, replace the COMPOSE project-specific section with:

- project objective and primary claim;
- current authorized milestone;
- forbidden actions;
- data sources and immutable boundaries;
- frozen terminology;
- model support declaration;
- primary endpoint;
- required splits and baselines;
- acceptance gates;
- verification commands;
- artifact locations;
- decision-log location;
- repository-specific formatting and testing commands.

Do not copy project-specific scientific claims from another repository unless they are genuinely valid
for the new project.
