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

**Scoped zero-oracle T4 compositional structural-subgoal generator,
2026-09-15:** after the grouped-source component audit found zero whole-patch
overlap but high low-level graph, atom, bond and dependency reuse, the user
explicitly authorized the smallest genuine compositional generator milestone.
Freeze the complete prospective contract before fitting. On the same three
whole-source folds, model
`pi(R|x) pi(H,alpha,D|x,R)`: first select an address-free source region, then
generate the complete target patch, attachment edges and construction
dependencies jointly through one graph-conditioned autoregressive local-patch
law. Its support must be generic executor-supported atom, bond and relative-role
events, not whole-patch identifiers. Do not replace this generator with a larger
retrieval pool, a complete-candidate reranker or independent marginal topology,
attribute, attachment and dependency classifiers.

Derive every learned vocabulary, feature moment and density after the frozen
split. Give explicit equalized mass to training domains, source or task-family
lineages, routes, components and decisions. Auxiliary Full-146 applications may
enter only from T4 training sources. PMO may enter only through its hash-bound
training-only artifact and only for routes exactly reconstructed by the existing
immutable conversion path. Do not race the unfinished delta-0.6 exporter or
repair the 99 PMO extractor abstentions inside this milestone. Keep the exact
structural realizer and extractor byte-unchanged.

Lock autonomous candidates before loading held teachers. Compare the learned
joint generator with a uniform scorer over the identical grammar and work
budget. At K=8, 32 and 128 report exact endpoint and patch recall,
radius-two transformation-equivalent recall, unique and novel patch yield,
held-component coverage, legal compile coverage, exact-realization precision,
diversity, abstentions and work. Exact teacher recovery is diagnostic, not a
utility objective. The engineering gate requires exact execution for every
accepted candidate and nonzero novel generated support on every held source.
The scientific gate requires a strict learned improvement in held component
coverage or transformation-equivalent recall at K=32 or K=128 without lower
exact-realization precision. Preserve failed gates as negative results. This is
single-CPU and zero-oracle. It may not call an oracle or docking function,
launch Modal, access a live run, alter prior artifacts or authorize a scored
pilot. The self-hashed contract is
`configs/t4_compositional_structural_subgoal_generator_v1.json`.

**Scoped zero-oracle T4 structural-subgoal component-novelty audit,
2026-09-15:** after the grouped whole-template proposal gate recovered zero
held-source transformations and every held complete template identifier lay
outside its training-fold vocabulary, the user authorized one split-first
component audit before choosing a compositional generator architecture. Preserve
the failed whole-template gate and the sealed exact structural realizer without
modification. On the same three frozen whole-source folds, measure held-source
coverage and novelty separately for source-region motifs, target topology, atom
attributes, bond attributes, attachment patterns, dependency motifs and smaller
target-patch fragments. Derive every training vocabulary after the fold split,
use exact address-free graph isomorphism rather than role order or a hash alone,
and report both instance-weighted and source-balanced coverage, unique coverage,
train-vocabulary transfer precision, whole-patch overlap, and the fraction of
held patches that are novel combinations of individually familiar components.

Represent the scientific object as `g=(R,H,alpha,D)`, where `R` is a selected
source region, `H` is a target patch graph, `alpha` binds it to surviving source
context and `D` records dependency structure. This audit does not authorize
independent marginal heads for those coupled variables, a fitted generator, a
new template retriever, a larger proposal pool or reranking revision. Audit
whether the remaining 69 non-complete-route programs in Full-146 and the sealed
PMO dependency-region training corpus can be converted to the same structural
representation. These are auxiliary-data compatibility checks, not permission
to merge corpora or train a joint checkpoint. Publish one deterministic
training-only result under the self-hashed contract in
`configs/t4_structural_subgoal_component_novelty_v1.json`. Use zero oracle and
docking calls, do not launch Modal, and do not access or alter any live run. A
compositional patch generator, PMO/T4 joint training, score-blind candidate lock,
Dynamic-v0 integration or scored experiment requires a subsequent scoped
contract.

**Scoped T4 structural-subgoal policy and exact realizer, 2026-09-14:** after
the 77-route dependency audit showed that every route contains at most four
strict dependency regions (median two), the user explicitly authorized moving
the primary development direction beyond primitive beam search. Preserve the
already running complete-program beam jobs as immutable diagnostics, but do
not wait for them or widen their beam. Define an address-free structural
subgoal over changed source roles, desired local graph state, boundary
attachments and inter-region dependencies. First prove that the subgoal corpus
is derived after the grouped source split, contains no primitive teacher trace
at runtime, and exactly reconstructs the teacher endpoints within declared
Active8, 48-slot, 40-active-atom and 32-primitive support. Reuse the legal
autoregressive machinery only as a conditional realizer of a generated
subgoal. Compare a generic marginal subgoal proposal with a graph-conditioned
proposal on the existing three whole-source folds, then measure held-source
subgoal and endpoint recall from teacher-free runtime checkpoints. Use zero
oracle calls. On 2026-09-15 the user explicitly authorized Modal fan-out to
remove local serial delay. The zero-oracle extraction, binding and exact-target
gate may use at most fifteen concurrent single-CPU source shards; later fitting
may use the same ceiling only after its split-first inputs and focused tests are
sealed. Every shard must have a durable call identity, independent result and
deterministic reduction, with zero automatic retry. A scored T4 launch, PMO
training, broader controller integration and Dynamic-v0 optimization wave
remain unauthorized until this zero-oracle representation, realization and
proposal gate is sealed.

Every conditional-realizer shard that can run longer than 30 seconds must
emit a structured heartbeat at least every 30 seconds. The heartbeat must
identify the source, phase and current route; report completed and total
routes, elapsed time, recent or overall throughput, search expansions and
attempts, failure or abstention counts, and an explicitly labelled estimated
time to the configured search limit. Publish the same state as a durable
progress receipt so live status does not depend on an attached terminal. The
estimate is operational telemetry, not a scientific stopping rule, and may
not change any search bound, source order, gate or evidence classification.

The first 15-source launch failed in every worker before loading a teacher
trace because the Modal image omitted the transitive
`tools/t4_program_vocabulary_audit.py` import. Preserve its launch receipt and
all 15 durable zero-oracle failure calls. The user explicitly authorized an
immediate bounded operational repair. Add only that missing module to the
material-file manifest, publish under `attempt_2`, and relaunch the same 15
source groups once with unchanged scientific inputs, support and gates.

**Scoped T4 dependency-region horizon audit, 2026-09-14:** while the three
zero-oracle complete-program decoder fits finish unchanged, the user authorized
one parallel diagnostic over the same 77 signed T4 teacher traces. Decompose
each exact route into dependency-connected structural regions using action
footprint overlap, created-handle dependencies, cycle open/close dependencies
and the lifetime molecular bond graph. Preserve primitive order and verify
exact replay. Report both the PMO-comparable lifetime-neighbor definition and a
stricter shared-handle sensitivity analysis so horizon reduction is not
overstated by a permissive connectivity rule. Report route coverage, replay
precision, component and component-run distributions, reentries, abstentions,
input hashes, software identity and per-source results. This audit authorizes
zero oracle calls, no model training, no decoder change, no beam change and no
live-run mutation. A structural-subgoal controller remains future work until
this diagnostic and the current decoder comparison are observed.

**Scoped T4 route-distilled complete-program decoder, 2026-09-14:** after the
grouped route-policy comparison showed that the fitted context actor ranks
teacher decisions well but autonomously recovers zero teacher-equivalent
transformations, the user explicitly authorized one focused generative
revision. Preserve the sealed route-distillation attempts, the source-sharded
comparison, Full-146, Dynamic-v0/v1/v2.1/v2.2, the live full-v0 benchmark and
all candidate-utility locks without modification. Train on the same 77 exact
complete T4 traces using the already frozen one-source-per-protein grouped
folds. Replace the three-module coarse generator with a target-free,
dependency-aware autoregressive decoder over exact legal canonical rewrite
actions, an explicit STOP action and at most 32 primitives/eight dependency
regions/40 active atoms. Later decisions may use relative handles created by
earlier decisions. Runtime artifacts may contain numeric preprocessing,
generic action support, model parameters and hashes, but no target/protein,
seed, route ID, endpoint, SMILES, absolute source address, target map or
executable teacher route.

Compare the learned decoder with the frozen source-balanced marginal decoder
under identical legal-action support, beams, proposal counts and compute
accounting. Generate and seal teacher-free candidate locks before a separate
evaluator loads teacher identities. Report teacher-forced action likelihood,
exact execution precision, complete-program and endpoint coverage/precision,
exact and radius-2 transformation recall/rank at 10, 100 and 1,000 generated
programs, unique yield, abstentions, expansions, wall time and memory. Preserve
the existing complete-candidate contrastive ranker as an optional downstream
reranker and preserve an unchanged Dynamic-v0 exploration lane; neither may
inject teacher candidates. The sealed prior zero-recovery result remains a
negative result and its gate may not be retroactively relaxed.

The engineering gate requires exact-execution precision 1.0, deterministic
locks, complete provenance, enforced support and at least one novel completed
endpoint per held source. Autonomous transformation recovery is a primary
diagnostic, not the sole authorization for molecular utility measurement. A
separate score-blind candidate-utility pilot may proceed only from a
prospectively frozen candidate lock with unchanged T4 endpoint filters and its
own explicit docking ledger; it cannot replace or alter candidates after
scores are observed. This milestone is otherwise zero-oracle and CPU-only. It
does not authorize a five-cell optimization campaign, a full T4 run, delta
0.6, PMO changes, new edit families, altered executor semantics or changes to
any live run. The detailed contract is
`docs/T4_COMPLETE_PROGRAM_DECODER.md`.

**Scoped route-distilled T4 qualification and conditional dual-threshold benchmark,
2026-09-14:** after explicitly rejecting the stored Full-146 controller as the
paper controller, the user authorized the newest generic route-distilled
Dynamic COMPOSE line for a staged T4 launch. The scientific problem is whether
one target-free policy learned from the 77 successful T4 trajectories can
propose complete executable programs without loading those trajectories at
runtime. The primary model output is a distribution over generic module
sequences, relative structural bindings, parameters and stop decisions. The
prospective claim is trained-on-T4 optimization performance, not held-out-route
generalization. Model support remains the declared neutral, charge-preserving,
stereochemistry-free persistent-slot graph representation with at most 40
active atoms, three high-level modules, 32 primitives and eight blocks.

Preserve the already running three held-source zero-oracle policy-comparison
folds exactly. Do not relaunch, repair, relax or select a policy before all
three immutable folds are durable and provenance-compatible. Select the
proposal factorization using the predeclared source-balanced autonomous
transformation-recovery evidence, with exact recovery secondary and execution
precision, unique endpoint yield and compute reported as safeguards. The
shared-panel result is reranking evidence only and cannot substitute for
autonomous generation. Fit the selected factorization once on all 77 training
routes, freeze its numeric target-free checkpoint, and audit that runtime
contains no protein/target name, source or seed identity, route identity,
endpoint molecule, absolute atom address, target-to-program map or executable
teacher route. Preserve an unchanged shallow generic exploration lane and an
empty task-specific complete-route archive at search start.

Before any benchmark-wide launch, run exactly one scored qualification unit on
5HT1B seed 0, BRAF seed 1, JAK2 seed 1, PARP1 seed 0 and FA7 seed 0 at delta
0.4 using the replicate-0 controller seed 20260913 and docking seed 1701. Each
unit must run its full 1,000-call allowance unless proposal yield is exhausted;
there is no score-plateau termination, automatic retry, confirmation call or
GPU. Use at most five concurrent single-CPU workers and at most 5,000 charged
docking calls. Launch requires a clean committed revision, a deterministic
zero-oracle runtime preflight, exact input and checkpoint hashes, isolated RNG
streams, exact replay, strict endpoint constraints and a complete candidate and
query ledger. Qualification passes only if all five units are durable, every
cell produces at least one eligible scored endpoint, the runtime provenance
audit passes, and at least four of five champions meet or improve the exact
cell-specific IVG mean frozen in the existing T4 comparator registry. This
gate is fixed before scores are observed and must not be weakened after a
negative result.

Only if that qualification passes, freeze the controller and launch the full
T4 benchmark at both delta 0.4 and delta 0.6. Use the same 15 benchmark source
molecules at both thresholds and the same three paired controller/docking seed
assignments per cell: (20260913, 1701), (20260914, 1702), and (20260915, 1703).
The only scientific adapter change between thresholds is the similarity gate.
Keep QED greater than 0.6, SA less than 4, the 40-atom support, docking protocol,
program support, policy checkpoint, parent/archive logic and every search
hyperparameter fixed. Run all 90 units to their 1,000-call ceilings unless
proposal yield is exhausted. Do not use plateau stopping, result-dependent
replacement, automatic retry or confirmation calls in this search wave. The
wave may use at most 15 concurrent single-CPU Modal workers, no GPU and at most
90,000 new docking calls. Use a new dedicated artifact volume rather than the
nearly inode-exhausted shared volume.

Record best eligible score at every charged call, matched-call summaries at 50,
100, 200, 400, 600, 800 and 1,000 calls, all failed charged calls, candidate
shortfalls, proposal/executor/filter outcomes, complete champion lineages,
wall-clock and compute, hashes, configuration, software and hardware. Preserve
the five qualification results separately even if their identical unit
identities later appear in the full wave; do not silently count or reuse them
as full-wave results. Full-146, Dynamic-v0/v1/v2.1/v2.2 and IVG values are
read-only comparators and may not enter runtime selection. If qualification
fails, stop before the 90-unit wave, report the negative result and require a
new scoped controller revision. The full dual-threshold result is a
trained-on-public-T4 evaluation of one generic route-distilled controller.

**Scoped PMO route-distillation export and cross-benchmark transfer preparation,
2026-09-14:** the user explicitly authorized a bounded, zero-oracle revision that
exports generic route supervision from the locked PMO development artifacts and
prepares a later balanced T4+PMO controller comparison. Admit exactly the 181
complete programs bound by the authoritative target, property, panel, formula and
Perindopril curriculum results plus the five saved Perindopril winner witnesses.
Freeze every inclusion, exclusion, source hash, task-family label and shared-base
lineage before fitting. Preserve exact primitive ancestry and distinguish complete
routes within the 32-primitive runtime support from longer routes that can supply
only local decision supervision. Use the same state features, structural option
vocabulary and 18-field stage descriptor as
`src/compose_v4/control/route_distilled_program_policy.py`, while exporting
task-independent action roles, relative created-handle provenance and dependency
labels without silently treating primitive fallbacks as learned high-level modules.

This revision may publish a deterministic training-only dataset and zero-oracle
audit. It must not train or advertise the current coarse actor as a final PMO
policy because the frozen recognizer finds no compound PMO stages, and it must not
make an oracle call, launch a scored experiment, access or change a live run, or
alter a T4 artifact. A future PMO-only checkpoint and a future balanced T4+PMO
checkpoint must remain distinct and use explicit domain, task-family, lineage and
decision weights. Runtime checkpoints may contain no task name, route identifier,
endpoint, SMILES, absolute atom address, assignment, source graph,
target-to-program map or executable teacher route. The detailed contract is
`docs/PMO_ROUTE_DISTILLED_TRANSFER.md`. Training either checkpoint, changing the
generic stage recognizer or running a scored PMO or T4 pilot requires a subsequent
scoped contract; any scored call also requires separate launch authorization.

**Scoped T4 route-distilled Dynamic COMPOSE development, 2026-09-14:** after
Dynamic-v2.2 recovered the Full-146 BRAF seed-1 score early but remained far
behind on JAK2 and candidate-starved on FA7, the user explicitly authorized
supervised proposal learning from all 77 complete T4 routes. Preserve Full-146
as a read-only teacher/reference and preserve every v0, v1, v2.1 and v2.2
artifact. Build one shared target-free proposal policy over complete generic
programs, structural site roles, relative bindings, parameters, dependencies
and stop decisions. Training may consume all 77 answer-known complete-route
traces. The fitted runtime artifact must contain no target or protein name,
seed identity, route identifier, endpoint molecule, absolute source atom
address, target-to-program map or executable stored route. This is
trained-on-T4 route distillation, not winner-independent or held-out evidence.

Before docking, publish four zero-oracle gates under the contract in
`docs/T4_ROUTE_DISTILLED_DYNAMIC.md`: exact teacher ingestion/replay coverage
and precision; high-level compression with explicit primitive fallbacks and
abstentions; proposal probability/rank against fixed generic and marginal
baselines; and a runtime-input/provenance audit. Preserve one unchanged
shallow generic exploration lane and the empty-at-start run-local route
archive. Failed, ineligible, duplicate and unqueried products are not task
labels. Do not relax a failed gate. If the sealed gates pass, freeze one
distilled controller and run only 5HT1B seed 0, BRAF seed 1, JAK2 seed 1,
PARP1 seed 0 and FA7 seed 0 at delta 0.4, one predeclared replicate-0 search
seed each, unchanged endpoint constraints, 40-heavy-atom support,
32-primitive/eight-block proposal limits and at most 1,000 charged calls per
cell. Use at most five single-CPU workers, no GPU, confirmation calls,
automatic retry or comparator information at runtime. Report matched-call
curves against Full-146 and Dynamic v0/v1/v2.1/v2.2 read-only artifacts. A
wider T4 run, delta 0.6, PMO transfer, another replicate or post-result repair
is a separate revision. Literal route replay remains a labeled ceiling and
cannot be reported as autonomous distilled recovery.

**Scoped parallel execution repair for the T4 route-policy gate, 2026-09-14:**
after the unchanged zero-oracle comparison remained healthy but serialized
three independent predeclared folds on one local CPU, the user explicitly
authorized a faster execution-equivalent copy. Preserve the original local
process and its eventual artifact. Add no scientific policy, feature, support,
seed, split, attempt-budget or acceptance-gate change. Run exactly one complete
fold per Modal worker, with three single-CPU workers total, and merge only after
verifying the complete fold census, exact source assignments, configuration,
input hashes, code revision and zero oracle/docking counts. Preserve the
original canonical within-fold source order and seed formulas, and retain the
same candidate pools for marginal and hybrid scoring. Combine source-balanced
metrics with the existing source-weighted reduction, not an unweighted mean of
fold summaries. Publish immutable shard, launch and merge receipts. If the
original local run finishes first, preserve it and use the parallel copy only
as an execution-equivalence check. This repair may use Modal for the three
zero-oracle fold workers. It may not launch the scored five-cell pilot, change
PMO, weaken a gate, retry automatically or exceed three concurrent CPUs.

The original local process subsequently failed after 11,800.67 seconds during
metric evaluation because an executed canonical self-event had been retained as
a complete proposal. Preserve that negative result and the live three-fold
parallel launch unchanged. A bounded follow-up repair may classify an executed
endpoint identical to its source as a rejected `canonical_self_event` in both
the marginal and context-actor proposal paths before metric evaluation. It may
not change candidate generation, random streams, policies, folds, attempt
budgets, metrics or gates. After the live folds terminate, retain every durable
completed fold and relaunch only folds that fail from this exact defect, under a
new immutable revision and output namespace. No automatic retry, oracle call,
PMO change or concurrent execution above the existing three-CPU ceiling is
authorized.

The live three-fold copy remained opaque and exposed the same terminal
self-event risk, so the user subsequently authorized the fastest
execution-equivalent durable replacement and up to the previously declared
Modal capacity. Preserve the failed local run and all three live fold calls
untouched. Under a new revision and namespace, partition the same predeclared
test census into exactly 15 one-held-source tasks, one per T4 source. Every task
must refit from the identical ten-source training fold, preserve the original
within-fold source index and seed formulas, generate the unchanged 32 training
negatives per training source and 128 marginal plus 128 actor proposals for its
single test source, and use the unchanged models, metrics and selection gate.
This is an operational shard only: no policy, split, feature, support, attempt
budget, acceptance rule or oracle access changes. Use at most 15 new concurrent
single-CPU workers, zero oracle calls, zero automatic retries, a dedicated
artifact volume, live phase/attempt progress, and an atomic result/model pair
per source. The recovery unit is one source. Merge only after verifying all 15
unique source artifacts, identical model checkpoints within each fold, exact
input/revision identities and source-balanced reduction. This later explicit
scope supersedes only the earlier three-CPU execution ceiling; it does not
authorize the scored qualification before the frozen zero-oracle gate passes.

**Scoped Dynamic COMPOSE v2.2 five-unit quick pilot, 2026-09-14:** after the
completed PMO Dynamic-v2.1 comparison showed that global channel allocation
underperformed Dynamic-v0 on three of four tasks, and the live T4 v2.1 run
showed an absorbing zero-query FA7 bootstrap plus coarse candidate allocation
on the active cells, the user explicitly authorized an aggressive but bounded
follow-up revision. Implement Dynamic-v2.2 as the existing route-free v2.1
shallow and structured proposal support plus exactly two generic repairs: a
persistent proposal-only T4 feasibility frontier and a chronological
candidate-specific parent-edit allocator. Preserve an auditable shallow
opportunity spine, independent proposal random streams, the empty initial
task-specific route archive, exact executor, 3-module/32-primitive/8-block
support and post-filter pooled competition. Fit the existing small regularized
parent-edit model only on charged outcomes from the current unit, predict
completed absolute utility/archive gain, and retain score-blind structural
exploration. Ineligible, failed, duplicate and unqueried candidates are never
task-utility labels.

After a deterministic zero-oracle preflight and focused resume, leakage,
support and query-accounting tests pass on clean committed source, run exactly
T4 FA7 seed 0, BRAF seed 1 and JAK2 seed 1 plus PMO GSK3B and
`isomers_c7h8n2o2`, one predeclared search seed each and at most 256 charged
calls per unit (1,280 new calls total). T4 keeps the unchanged delta-0.4,
QED, SA, 40-heavy-atom and docking protocol; PMO uses only its native validity,
higher-is-better oracle and top-ten trajectory adapter. Use at most five
concurrent single-CPU workers, no GPU, confirmation calls or automatic retry.
Full-146, v0, v1, v2.1, IVG and task-informed outcomes are offline comparators
only and may not enter runtime inputs. Preserve the completed PMO v2.1 result
and all live/frozen T4 runs without modification. This is a five-unit
answer-known development pilot, not a held-out or benchmark-wide result. A
wider run, another seed, changed support, altered endpoint gate or post-result
v2.2 repair remains a separate revision and is not automatically authorized.

The first five-unit launch preserved its zero-oracle preflight, but all three
T4 calls then failed before `run_t4_pilot_campaign` because the clean-input
guard searched provenance-only source filenames for forbidden comparator and
winner substrings. The default local PMO environment likewise failed before an
oracle call because it lacked PyTDC; the two PMO units were relaunched once in
the already qualified PyTDC 1.1.15 and RDKit 2023.9.6 environment and completed
all 256 calls. Preserve both zero-query failure records and the completed PMO
artifacts. The user explicitly authorized a bounded operational T4 repair and
relaunch on 2026-09-14. Exclude only `image_revision` and `files_sha256`
provenance fields from the runtime-payload substring scan while retaining the
guard on actual controller inputs. Add read-only status handling for a missing
worker artifact and seal the exact original three-call, zero-query failure
census. Relaunch only FA7 seed 0, BRAF seed 1 and JAK2 seed 1 after a new remote
zero-oracle preflight, preserving every scientific field, seed, budget and
controller input from the original launch. Do not relaunch PMO, overwrite the
original receipt, retry automatically, or change the scientific controller.

**Scoped PMO Dynamic COMPOSE v2.1 four-task development, 2026-09-14:** while
the five-cell T4 Dynamic-v2.1 launch proceeds independently, the user
explicitly authorized a parallel PMO development worker to adapt the same
pooled shallow/structured controller core through a PMO task adapter. Use
exactly `celecoxib_rediscovery`, `perindopril_mpo`, `gsk3b` and
`isomers_c7h8n2o2`, one predeclared search seed each and at most 1,000 charged
oracle calls per task (4,000 total). Start with zero task-specific complete
routes and no task/panel-informed program curriculum. Preserve the generic
v0 shallow proposer, generic v1 structured proposer, independent RNG streams,
post-filter pooled arbitration, common run-local route archive and exact
executor. PMO changes only initialization, validity, higher-is-better reward,
native oracle and top-ten trajectory accounting. Do not inherit T4 similarity,
QED or SA filters. All initialization scores count. Compare with existing
locked PMO development results and IVG values offline only; neither may enter
runtime selection. Record best and top-ten score curves, official development
AUC arithmetic, exact query ledgers, candidate shortfalls, compute and
failures. Use at most four single-CPU workers, no automatic retry, T4 change,
fifth PMO task or wider PMO claim. Seal a task-specific zero-oracle preflight
and a clean committed contract before scoring. This is an answer-known PMO
development diagnostic, not held-out or full-suite PMO evidence.

**Scoped T4 Dynamic COMPOSE v2.1 pooled-channel development, 2026-09-14:**
after the frozen Dynamic-v2 revision failed its sealed zero-oracle preflight,
the user authorized a new Dynamic-v2.1 implementation and five-cell scored
development run. Preserve Dynamic-v2 and all of its failure artifacts as a
closed negative revision; do not relax or bypass its 32-attempt gate. Preserve
the live Dynamic-v1 workers and every v0, 69-only and Full-146 artifact without
modification. Dynamic-v2.1 tests whether independently generated shallow-v0
and structured-v1 eligible proposal pools can compete only after exact
execution, strict endpoint filtering and cross-channel deduplication, so an
empty channel consumes no docking allocation. Use independent shallow,
structured and arbitration random streams. The shallow synthesizer remains
unchanged, the structured channel retains only the generic protected
ring-path, substituted-ring and contextual-binding capabilities, and both use
one empty-at-start run-local route archive and the same exact executor.

Seal a new zero-oracle preflight before scoring. It verifies zero initial
routes and absent comparator outcomes at runtime, unchanged shallow behavior,
independent random streams, representative exact execution through both
channels, deterministic combined pooling and recorded strict-filter yield on
all five development cells, clean fallback when either channel is empty, and
complete provenance. It must
not require either isolated stochastic expert to produce a strict-eligible
endpoint within an arbitrary draw count. After focused checks and the pinned
preflight pass, run 5HT1B seed 0, BRAF seed 1, JAK2 seed 1, PARP1 seed 0 and
FA7 seed 0 with the same replicate-0 controller/docking seeds, delta-0.4 gates,
40-heavy-atom support,
32-primitive/eight-block work limits, competitive-plateau rule and 1,000-call
per-cell ceiling as v0/v1. Add PARP1 seed 0 and FA7 seed 0 so the development
wave contains one predeclared cell from each T4 protein. At most five
concurrent single-CPU workers, 5,000 new docking calls, no GPU, confirmation
calls or automatic retry are
authorized. Record all generated pools, attempts, filters, arbitration,
channel-awarded calls, outcomes, ancestry, compute and score curves. Runtime
may not load Full-146, v0 or v1 outcomes, known routes, endpoints, scores,
target names or target-to-program rules. Remaining cells, delta 0.6, PMO and
post-result controller changes remain outside this milestone. The detailed
contract is `docs/T4_DYNAMIC_V21.md`.

**Scoped T4 Dynamic COMPOSE v2 implementation, 2026-09-14:** while the
three-cell Dynamic-v1 docking study remains immutable and live, the user
authorized a separate zero-oracle implementation of Dynamic-v2. The scientific
problem is whether a generic parent-specific planner gate can preserve the
efficient route-free Dynamic-v0 proposal law while invoking Dynamic-v1's
protected structured planner only when molecular opportunity or within-run
search evidence warrants it. The primary output is a planner-mode choice
(`shallow_program_channel` or `structured_program_channel`) followed by one
complete executable program under the existing representation and exact
executor. The prospective claim is that selective structured planning can
recover v0's early efficiency and improve harder transformation regimes; it is
not established by implementation or offline tests.

Implement one shared, interpretable, bounded gate with a 0.75 shallow / 0.25
structured prior and a 0.10 to 0.50 structured-probability floor/ceiling. The
gate may use only generic current-molecule features and statistics from charged
observations within the same run. Preserve Dynamic-v0 behavior as the shallow
channel, preserve Dynamic-v1 ring-path, substituted-ring and contextual-binding
capabilities as the structured channel, start with an empty task-specific route
archive, and retain the 40-atom, 32-primitive, eight-block support. Record
channel decisions, probabilities, features, proposal/execution/eligibility and
scored credit so state is exactly resumable. Do not load Full-146, the 146-bank,
winner routes, endpoints, scores, target names or target-to-program rules at
runtime.

The user subsequently authorized launching the frozen v2 implementation in its
own three concurrent single-CPU containers after focused checks and a
zero-oracle pinned-environment preflight pass. Run exactly the same three cells,
replicate-0 controller and docking seeds, delta-0.4 gates, 40-heavy-atom support,
32-primitive/eight-block work limits, competitive-plateau rule and 1,000-call
per-cell ceiling as v0/v1. At most 3,000 new docking calls, no GPU, no
confirmation calls and no automatic retry are authorized. The live v1 launch
may be referenced read-only by its sealed receipt, but v2 must not consume v1
outcomes or modify, restart, cancel or otherwise affect any v0/v1/69-only/
Full-146 artifact. Preserve every v2 proposal, gate decision, rejection, query,
failure and score curve. Remaining T4 cells, delta 0.6, PMO, neural policy and
any post-result controller change remain outside this milestone. Use focused
checks during this bounded development launch; an unrelated repository-wide
suite is not an iteration gate. The detailed contract is
`docs/T4_DYNAMIC_V2.md`.

**Scoped T4 Dynamic COMPOSE v1 development study, 2026-09-13:** after the
route-free diagnostic separated a successful shallow BRAF case from a JAK2
representation/protected-horizon failure and a 5HT1B binding-conditioning
failure, the user authorized one focused Dynamic-v1 repair. Preserve the
Full-146 controller, Dynamic-v0 implementation, completed BRAF/JAK2 outcomes
and still-running Dynamic-v0 5HT1B unit byte-for-byte. First run a zero-oracle,
answer-known teacher-forced reachability audit on 5HT1B seed 0, BRAF seed 1 and
JAK2 seed 1. Then implement exactly three generic capabilities in a separate
v1 path: bounded protected ring-path remodeling, jointly parameterized
substituted-ring construction, and bounded context-conditioned binding
enumeration using no task-oracle value. The reachability audit must report
exact executor replay, coverage, precision or abstention, and the minimum v1
module count needed for each inspected Full-146 transformation. It must not
promote an inspected route or endpoint into the deployed grammar, sampler or
initial archive.

After the live Dynamic-v0 5HT1B unit is durable and only if the zero-oracle
gate and launch-boundary tests pass, freeze one shared Dynamic-v1 controller
and run it once on the same three development cells with the same replicate-0
controller and docking seeds, delta-0.4 endpoint gates, 40-heavy-atom support,
32-primitive/eight-block work limits, competitive-plateau rule and 1,000-call
per-cell ceiling. At most 3,000 new docking calls, three concurrent single-CPU
workers, no GPU, no confirmation calls and no automatic retry are authorized.
Record every proposal, rejection, candidate shortfall, query, failure, module
choice, binding alternative, complete score-versus-call curve and provenance.
Dynamic-v0 and Full-146 are read-only comparators and must not be redocked.
This is winner-informed development on three declared cells, not held-out or
full-benchmark evidence. Evaluation on the remaining twelve T4 cells, another
replicate, a delta-0.6 run, PMO work, neural proposal learning and any fourth
controller capability remain outside this milestone. The detailed contract is
`docs/T4_DYNAMIC_V1.md`.

**Scoped T4 complete-route and dynamic-synthesis diagnostic, 2026-09-13:** the
user explicitly authorized a matched three-cell study of whether COMPOSE can
recover the frozen program controller's performance without initializing from
77 stored whole-route entries. Preserve the live 45-unit T4 run and its
146-program controller without modification. Arm A reuses the completed
full-146 replicate-0 outcomes read-only. Arm B mechanically derives one
69-entry library by removing only programs whose sole block is labelled
`compiled_complete_transformation`. Arm C loads zero rows from the 146-entry
file, starts with an empty complete-route archive, and constructs protected
complete proposals at runtime by composing one to three generic parameterized
modules. Only routes discovered and scored within Arm C may subsequently enter
its mutation/recombination archive. Run Arms B and C only on 5HT1B seed 0,
BRAF seed 1 and JAK2 seed 1 with the matched replicate-0 controller and docking
seeds, unchanged endpoint gates, 1,000-call unit ceilings and the frozen
competitive-plateau rule. Do not redock or relaunch Arm A. The two new arms may
use at most six concurrent single-CPU workers, 6,000 new docking calls total,
no GPU, no confirmation calls and no automatic retry. Preserve per-round score
curves, candidate shortfalls, failures, exact query ledgers, dynamic module
choices and all provenance. Report each arm at the same call count as Arm A as
well as final scores. Report the lower-is-better fraction of full-controller
improvement recovered only against a separately identified measured source
score, and label protocol-identity limitations. This is a winner-informed
three-cell development diagnostic, not a held-out benchmark. Another replicate,
a wider T4 wave, deeper dynamic composition and any subsequent controller
change remain outside this bounded milestone.

**Scoped frozen T4 interrupted-unit rescue, 2026-09-13:** the original frozen
full-suite run sealed a partial aggregate after 22 workers stopped with one
reserved query lacking a durable result, while a separately restarted JAK2
seed-2 replicate subsequently completed. The user explicitly requested that the
unfinished units resume from their durable points with the same seeds and exact
oracle accounting. Implement and run the bounded recovery in
`docs/T4_FROZEN_PROGRAM_RESCUE.md`. Preserve the original controller, source
states, program library, endpoint gates, search and docking seeds, query order,
checkpoints, round locks, 40-atom support and deployed scientific image. For
each interrupted unit, convert the one already reserved query into an explicit
charged failure with no score, never redock or relabel that molecule, then
resume only the remaining budget. Preserve the stale aggregate, its 24 early
confirmation calls and every failure artifact. Reaggregate only after all 45
unit outcomes are durable. Lock final champions and obtain only the fresh
confirmation calls needed by the final lock, reusing an earlier receipt only
when its complete query identity is identical. At most 16,943 new search calls
and 30 new confirmations are authorized, using at most 22 concurrent one-CPU
workers and no GPU or automatic retry. The cumulative run must remain below the
original 45,030-call and $20 ceilings. No controller, T4 protocol, PMO, training,
candidate replacement or result-dependent scientific change is authorized.

The first rescue relaunch then failed closed in non-bootstrap units after their
locked batches finished: the original optimizer did not recognize the precise
`ambiguous_charged_query_unobserved` tombstone status. Preserve the relaunch
receipt and every unit failure. The bounded repair may accept only that exact
status as a charged missing observation, without adding a score, blacklisting
the endpoint, changing proposal state, or relabeling it as an oracle failure or
cache miss. Audit that each failed unit has a complete contiguous query ledger
before another invocation. Leave still-running bootstrap invocations untouched;
total concurrent rescue workers remain at most 22. No other controller or
protocol change follows.

The sealed repaired relaunch subsequently stopped all 19 affected invocations
before `run_unit` or any oracle call. The image-level revision check accepted
the repaired source, but the original frozen benchmark contract correctly
rejected the changed byte identity of
`src/compose_v4/control/adaptive_program_optimizer.py`. Preserve the v2 launch
receipt and record all 19 remote exceptions as one zero-query prelaunch
failure. A second bounded repair may override contract-input verification only
for that exact file, only from the frozen digest recorded by the original
contract to the repaired digest sealed by the new rescue lock, and only for the
already authorized tombstone-status compatibility change. Every other original
contract input remains byte-exact. Reaudit unchanged query ledgers before
relaunching the same 19 units; leave the three bootstrap invocations untouched,
use zero automatic retries, and make no controller or protocol change.

The v3 launch verified the optimizer override and then stopped before any
oracle call on the next frozen-input mismatch,
`src/compose_v4/control/program_transfer.py`. A complete manifest comparison
identified exactly three changed contract inputs: the optimizer status repair;
the previously recorded direct-retrieval implementation whose frozen
`cold_start_retrieval_candidates=0` setting disables that path; and the local
benchmark CLI's read-only status-loader change, which is not worker runtime.
Preserve the v3 receipt and seal its zero-query failure census. The bounded v4
repair may verify exactly those three paths against their sealed current
digests and declared decision-equivalence reasons. It must reject any other
manifest mismatch, path, digest or reason. Reaudit all 19 unchanged query
ledgers before relaunch; all other rescue and scientific constraints remain
unchanged.

The v4 launch passed the complete repository-input manifest and entered
`run_unit`, then stopped before docking while verifying `/opt/dock/qvina02`:
the rescue-only input wrapper incorrectly required every verified path to be
relative to `/root/compose`. Preserve the v4 receipt and seal the 19 identical
zero-query runtime-preflight failures. The bounded v5 repair may delegate paths
outside the repository root unchanged to the original verifier. It may not
override the QuickVina or receptor hashes, add another repository override, or
change any controller/runtime setting. Reaudit unchanged ledgers before the
same 19-unit relaunch.

**Scoped frozen T4 program-vocabulary audit, 2026-09-13:** the user explicitly
requested an immediate audit of whether the exact frozen 146-program T4 bank is
a general context-bound transformation vocabulary or retains task-, seed- or
winner-specific information. Implement and run the bounded zero-oracle audit in
`docs/T4_PROGRAM_VOCABULARY_AUDIT.md`. Inspect the frozen bank without changing
it; map its opaque source groups against the public 15-cell registry; scan for
literal target, seed, score and endpoint fields; verify typed address-free atom
handles and runtime context rebinding; distinguish complete winner-route
programs from other program structures; and measure bounded structural-binding
and exact-execution applicability on all 15 frozen source states. Reconcile the
existing answer-known direct-retrieval winner recovery and trace the best four
measured development candidates back to their library program origins. Report
coverage and execution precision separately, preserve negative results, input
hashes and source provenance, and make no oracle call, Modal call, library,
controller, live-run, PMO or T4 scientific change. This is a retrospective
specificity and applicability audit, not held-out generalization or autonomous
optimization evidence.

**Scoped PMO Median1 closing query, 2026-09-13:** after the exact IVG-core
oracle audit exceeded IVG no-prescreen on all eleven completed tasks and IVG
prescreen on ten, the user requested aggressive continuation across PMO. The
sole completed-task miss is Median1 by 0.001550419204571274 AUC. Implement and
run the bounded one-query closeout in `docs/PMO_MEDIAN1_CLOSE.md`. The single
candidate is an iodine analogue selected before scoring by extending the
public-panel O/S/N/F/Cl/Br substituent series. Compile it from the same generic
Median1 root into one complete exact-replay COMPOSE program, freeze its query
identity, and score it once under the already qualified IVG-core PyTDC 1.1.15
and RDKit 2023.9.6 environment. Append that observation to the immutable prior
eleven-call Median1 chronology and report the official 10,000-query top-ten AUC.
No retry, replacement, reselection, additional candidate, training, Modal use,
T4 change, or remaining-task query follows automatically. A positive outcome
is prospective panel-series extrapolation inside an answer-known,
winner-informed development regime, not held-out or autonomous PMO evidence.

**Scoped PMO IVG-oracle parity audit, 2026-09-13:** after the user requested
continued aggressive PMO work, inspection of the official IVG repository found
that its Dockerfile installs PyTDC 1.1.15 while its requirements pin RDKit
2023.9.6. The eleven completed COMPOSE PMO development tasks were scored under
PyTDC 0.3.6 and RDKit 2024.03.5, so their arithmetic margins are not yet
protocol-matched. Implement and run the bounded audit in
`docs/PMO_IVG_ORACLE_PARITY.md`: rescore the exact 150 already locked molecules,
without candidate replacement or reselection, under the IVG-pinned oracle
environment. Preserve source query order, source-result hashes, both original
scores and parity scores, the official 10,000-query top-ten AUC, and both IVG
comparison regimes. Count and preserve the one precontract C7 isomer parity
probe separately, for 151 new parity-environment oracle calls in total. Verify
PyTDC/RDKit and all material dependency, source-module and downloaded
GSK3B/JNK3 pickle identities before authoritative scoring. No automatic retry,
new molecule generation, training, Modal use, T4 change, remaining-task wave or
general PMO claim follows automatically. Positive results remain answer-known,
winner/panel-informed development evidence; this audit changes the evaluator,
not the locked candidates.

The first parity launch stopped before any authoritative query because its
asset gate correctly rejected legacy HN-GFN `gsk3b.pkl`/`jnk3.pkl` hashes as
identities for PyTDC's distinct `*_current.pkl` downloads. Preserve the failed
lock and `diagnostics/pmo_ivg_oracle_parity/prequery_failure_0001.json`. The
bounded repair may freeze the actual PyTDC 1.1.15 current assets served under
Dataverse file IDs 6413412 and 6413420, update only those identities, and seal
`query_lock_v2.json` from the unchanged 150 source observations. No task score
was observed, so the molecule set, order, comparators and call ceiling remain
unchanged.

The repaired v2 launch also stopped before any authoritative query. Its hashes
and Dataverse IDs were correct, but the contract had copied byte counts from
the legacy pickle manifest instead of the downloaded current files. Preserve
`query_lock_v2.json` and
`diagnostics/pmo_ivg_oracle_parity/prequery_failure_0002.json`. Correct only
the two measured byte counts and seal `query_lock_v3.json` from the unchanged
150 observations. The call ceiling and all scientific inputs remain unchanged.

**Scoped PMO formula/median panel wave, 2026-09-13:** after the panel-informed
QED/GSK3B/JNK3 refinement beat both IVG regimes on all three tasks, the user
requested continued aggressive PMO development. Implement and run the bounded
follow-on in `docs/PMO_FORMULA_MEDIAN_PANEL_WAVE.md`: `isomers_c7h8n2o2` and
`median1`, ten predeclared neutral structures per task plus one predeclared
task-specific generic neutral root. The isomer set must contain ten canonical-unique
molecules whose RDKit molecular formula is exactly C7H8N2O2; the median set
must retain candidate-level public-panel transcription/derivation roles.
Compile every endpoint into a complete COMPOSE program and exact-replay all 20
before any new task score is observed. Score root and endpoints exactly once,
at most 22 new local CPU PyTDC calls total, with no retry or replacement.
Preserve the official 10,000-query top-ten AUC, 40-active/48-slot neutral
support, exact upstream hashes and both IVG comparison regimes. This is
answer-known white-box development, not held-out discovery, autonomous search
or general PMO superiority. No Modal use, T4 change, reference training or
automatic later wave follows.

The prelock attempt to reuse the earlier unrelated root failed the zero-oracle
structural gate for the C7 task and is preserved in
`diagnostics/pmo_formula_median_panel_wave/prelock_failure_0001.json`. It
observed no task score and produced no committed or sealed curriculum. The
task-specific roots above therefore replace that inappropriate root before the
candidate lock; they do not alter the candidates, oracle, metric, support or
call ceiling.

The first sealed curriculum then exposed a specialized-adapter dispatch defect
before starting any query. Preserve that lock and
`diagnostics/pmo_formula_median_panel_wave/prequery_failure_0002.json` as an
invalidated zero-query attempt. Repair only the adapter by using the existing
general pinned-PyTDC dispatcher, seal a new curriculum from the changed
implementation, and retain the same candidates, roots, metric and 22-call
ceiling.

**Scoped PMO panel-informed property refinement, 2026-09-13:** after the
single-anchor learned-property wave reproduced excellent anchors but diluted
top-ten performance with generic one-edit variants, the user requested
aggressive continued PMO development. Implement and run the bounded follow-on
in `docs/PMO_PROPERTY_PANEL_REFINEMENT.md`: QED, GSK3B and JNK3, ten predeclared
neutral panel-transcribed or explicitly panel-derived structures per task, plus
the same previously charged unrelated exact root. Compile every endpoint into a
complete COMPOSE program and exact-replay all 30 before any new task score is
observed. Score each root and endpoint exactly once, at most 33 new local CPU
oracle calls total, with no retry or replacement. Reuse the already qualified
QED and byte-matched frozen GSK3B/JNK3 oracle implementations, official
10,000-query top-ten AUC, 40-active/48-slot support, canonical deduplication and
both IVG comparison regimes. Preserve candidate-level source roles and do not
describe derived positional or family variants as exact public panel records.
This is answer-known development and a test of panel-informed program support,
not held-out discovery, general PMO superiority or autonomous optimization.
DRD2 remains excluded pending current-oracle parity. No frozen-reference
training, Modal use, T4 change or automatic later wave follows.

**Scoped PMO learned-property program wave, 2026-09-13:** after the first
five-task target-program wave beat both recorded IVG comparators on every task,
the user requested aggressive continuation across PMO. Implement and run the
next bounded wave in `docs/PMO_PROPERTY_PROGRAM_WAVE.md`: QED, GSK3B and JNK3.
Use one high-scoring molecule transcribed from each official IVG top-molecule
panel as disclosed answer-known development supervision, then compile and replay
fifteen unique complete anchor-neighborhood programs per task from the same
previously charged unrelated exact root before any task score is observed. Score
the root and locked endpoints exactly once per task, at most 48 new local CPU
oracle calls total, with no retry or replacement. Use the official PyTDC QED
implementation. For GSK3B/JNK3, use the existing validated flat-tree evaluators
only after recording that the newly downloaded official TDC pickle hashes match
the frozen extraction manifests byte-for-byte. Preserve 40-active/48-slot
neutral support, canonical deduplication, the official 10,000-query top-ten AUC,
exact IVG file hashes and both no-prescreen and prescreen comparisons. DRD2 is
excluded from this wave because its current TDC pickle does not byte-match the
local frozen SVM and has not passed a current parity gate. This is a
winner-informed development wave, not held-out discovery or general PMO
superiority. No learned-reference training, Modal use, T4 change or automatic
later wave follows.

**Scoped PMO exact-target program wave, 2026-09-13:** after the measured
Perindopril curriculum exceeded both recorded IVG comparators, the user
explicitly requested aggressive application of the same winner-informed
program strategy across PMO. Implement and run the first bounded five-task
wave in `docs/PMO_TARGET_PROGRAM_WAVE.md`: albuterol and mestranol similarity,
plus celecoxib, troglitazone and thiothixene rediscovery. Use the exact public
task targets only as disclosed answer-known development supervision. From the
same previously charged unrelated exact root, compile and replay fifteen unique
complete target-neighborhood programs per task before scoring. Score the root
and locked endpoints exactly once per task, at most 80 new local CPU oracle
calls total, with no retry or replacement. Preserve 40-active/48-slot neutral
support, canonical deduplication, the official 10,000-query top-ten AUC, exact
IVG file hashes and both no-prescreen and prescreen comparisons. This is a
winner-informed development wave, not held-out discovery or general PMO
superiority. Later task waves, broader support and autonomous 10,000-query runs
remain separate scoped milestones.

**Scoped editing-V2 registry-lineage repair, 2026-09-13:** the user explicitly
authorized resolving the repository-wide verification blocker so the locked
Perindopril curriculum can be scored. The read-only Process-V2 verifier currently
crashes while treating an arbitrary long JSON prose value as a filesystem path;
after that defect is fixed, it must enumerate the complete stale pointer chain.
Repair only machine-checkable hash and identity pointers whose targets are
unchanged current files, in dependency order and to a fixed point. Do not change
any process semantics, operator support, registry cells, thresholds, splits,
authority flags, scientific status or prior evidence. Preserve superseded
identities as lineage where explicitly declared. Acceptance requires a focused
regression for the long-string scanner, the verifier reporting `AGREES`, the
previously failing registry loader importing successfully, and the full suite
passing. Record every changed pointer and prove all non-pointer JSON/Python
content is unchanged. This repair confers no Gate-0, T1, P50 or training
authority. Contract: `docs/EDITING_V2_REGISTRY_REPIN_2026-09-13.md`.

**Scoped winner-informed Perindopril program curriculum, 2026-09-13:** after the
completed fast PMO run exposed poor transfer from the generic T4 program library,
the user explicitly requested aggressive use of published IVG molecules and
approval to pursue a Perindopril result above IVG. Implement one answer-known,
task-informed program curriculum from the already verified public 0.8088297766
endpoint and its five exact COMPOSE recovery witnesses. Compile, replay and lock
a bounded family of supported endpoint variants before scoring. The run may make
at most 16 deterministic Perindopril-MPO calls: four declared exact roots, the
published endpoint and at most eleven program-derived variants. Compute the
official counted-only top-ten AUC with `finish=True` and a 10,000-query
denominator. Compare against the repository-reported IVG no-prescreen value
0.645 and prescreen value 0.753, but label any positive result as
winner-informed Perindopril development, not held-out PMO or general controller
superiority. Candidate molecules must be outputs of replay-verified complete
programs from the charged roots, never injected as an unaccounted initial
archive. Preserve 40-active/48-slot support, exact executor semantics, the
pinned PyTDC/RDKit oracle, unique canonical accounting and every failed program.
The already observed element-restatement plateau is a disclosed retrospective
development finding, not prospective evidence. T4 runs and their frozen
configuration remain unchanged. Contract and acceptance:
`docs/PMO_WINNER_PROGRAM_CURRICULUM.md`.

**Scoped T4 context-ranked program-retrieval diagnosis, 2026-09-13:** while the
frozen full-suite run remains immutable, the user requested continuing the
parallel controller-development track. The first completed live unit exhausted
proposal yield on 5HT1B seed2 after fourteen calls even though the shared library
contains verified programs from that source. Implement an optional bounded cold-
start channel that executes high-probability context-matched programs without
first forcing a parameter or attachment mutation. Defaults remain unchanged.
Compare the frozen cold-start proposal with an eight-candidate direct-retrieval
slice on all fifteen exact T4 delta=0.4 source states, with the same shared
146-program library, strict endpoint gates, work limits and random remainder.
This is an answer-known, winner-informed, zero-oracle structural diagnostic, not
an autonomous benchmark result. Record exact public-winner overlap separately
from general eligible yield. No docking, Modal deployment, live-run change,
reference/executor modification or successor benchmark launch is authorized by
this local diagnosis. Contract and acceptance: `docs/T4_PROGRAM_RETRIEVAL.md`.

**Frozen full T4 delta=0.4 program-controller benchmark, 2026-09-12:** the
user requested a paper-result track in parallel with controller development and
authorized Modal use up to 30 total containers. Freeze one shared fast
coordinated-program recipe before observing the full results, then evaluate all
15 official T4 starting cells at delta=0.4 over three predeclared search seeds.
Each cell/run may use at most 1,000 distinct candidate docking evaluations; the
published seed score is not charged, matching the official IVG result files.
After all 45 runs complete, lock one overall first-score champion per cell and
obtain two fresh confirmation dockings, for a hard ceiling of 45,030 new calls.
Use one driver plus at most 29 single-CPU workers, no GPU, no automatic oracle
retry, and a $20 reserved compute ceiling. The frozen controller is the
program-only, score-blind, cache128 recipe with exact-current-state continuation,
verified decomposition, current-state editing, shared 146-program library,
random mutation, no learned selector, no reference inference and no protected
composition. Preserve 40-atom support and apply the official strict endpoint
filters: similarity >0.4, QED >0.6 and SA <4. Freeze hashes for the source
registry/states, shared library, upstream protocol evidence, receptors, docking
binary, code/config, seed derivation, candidate/query receipts and failure
treatment. Persist 30-second live heartbeats and full score-versus-call curves.
After at least 500 calls, stop a run only if it is already at or better than the
reported cell-level IVG mean and has improved by less than 0.3 docking units over
the preceding 250 calls. Weak runs retain the full ceiling. Public winner-derived
programs are disclosed development inputs to the single frozen controller; no
cell-specific controller or result-dependent repair is allowed. Contract and
acceptance criteria:
`docs/T4_FROZEN_PROGRAM_BENCHMARK.md`. Launch only from clean committed source
after the zero-oracle 15-cell preflight and focused/full launch-boundary checks.

**Scoped T4 protected program-composition experiment, 2026-09-12:** after
reviewing the latest T4 evidence, the user requested testing whether protected
program continuation helps and explicitly kept PMO out of the loop. Implement a
general bounded 2..K exact program-composition channel, but freeze K=2 for the
first comparison. Every later component binds on the actual exact successor;
only the completed composition receives the unchanged T4 endpoint eligibility
check or docking evaluation. Preserve the fast program-only control, shared
winner-informed development library, score-blind parent law, 40-atom support,
original-seed delta=0.4 gates, exact executor, ancestry, total 32-primitive and
eight-block limits, and zero reference inference. First run the frozen local
structural comparison in `docs/T4_PROGRAM_COMPOSITION.md`. A positive structural
gate authorizes one locked BRAF seed1/JAK2 seed1 docking pilot of at most 20 new
calls, eight single-CPU workers plus one driver, no GPU or automatic retry, and
$10 reserved maximum under the user's standing Modal approval. Freeze exact
candidate, oracle and compute manifests and use clean committed source before
launch. PMO, K>2 scoring, task-specific endpoint templates, reference/executor
changes, model fitting, and any full-suite or benchmark-win claim are excluded.

**Scoped fast PMO completion run, 2026-09-12:** the user replied "great lets od
it" to one short direct-scoring PMO run after the measured throughput repair.
Authorize one albuterol-similarity development run, seed 20260921, at most 128
new queries including the sixteen prelocked initialization structures, seven
rounds of at most sixteen candidates, one CPU container, no GPU or automatic
retry, 600-second worker timeout and $1 total reserved compute. Freeze
`configs/fast_pmo_run.json` before launch from clean committed source. Use the
program-only random-mutation recipe, cache128, all-scored initial-parent pool,
direct scoring, and the existing exact executor and oracle. No predictor or
reference fitting, new docking, other task, grid or recovery of the previous
ambiguous PMO reservation follows. Record completed rounds, shortfalls, actual
query/compute curves and termination reason. This is a bounded development
run, not a full-suite benchmark or milestone/release sign-off.

**Scoped fast program-search repair, 2026-09-12:** the user requested restoring
productive program-first search after the mixed learning-cycle campaign. The
immediate unit is local, zero-oracle implementation and bounded throughput
measurement under `docs/FAST_PROGRAM_SEARCH.md`: explicit program-only dispatch,
optional selection only with excess candidates, exact attempted-proposal reuse,
and PMO initialization/dispatch/failure-isolation repairs. Preserve old recipes,
all results and incumbents, reference/executor semantics, 40-atom T4 support and
charged-query guards. This changes the optimization proposal, not the frozen
reference law. No new paid campaign, reference training or automatic recovery
of unresolved queries follows. Later proposal learning requires the throughput
decision first; the previous paid campaign must not be relaunched unchanged.

**Scoped parent-edit learning-cycle approval, 2026-09-12:** the user replied
"sure" to a ceiling of 108 new docking calls across BRAF/JAK2, 24,000 PMO
queries, $20 total reserved compute and at most 30 simultaneous containers.
Implement and run the bounded paired development comparison in
`docs/PARENT_EDIT_LEARNING_CYCLES.md`. Both arms use the same repaired broad
program engine; learned selection alone differs. Preserve exact execution,
40-atom T4 support, original-seed gates, all historical information declarations,
query locks, receipts and failures. CPU-only modest predictor updates are part
of this authorization; frozen-reference training is not. Freeze input/recipe,
oracle and compute manifests before paid execution, use clean committed source,
no automatic oracle retries, and report completed batches promptly. This
supersedes the preceding local-only restriction for this bounded experiment,
not the editing-V2 training ladder or any frozen scientific gate.

**Scoped parent-edit controller implementation, 2026-09-12:** the user requested
learning complete parent/edit choices and bringing PMO into the shared program
engine. The immediate unit is local, zero-oracle implementation: inspect and
repair opaque program branches, ancestry-limited proposal horizons and access to
existing-atom edits; add explicit T4/PMO task semantics and a modest completed-
program utility predictor. Local CPU fitting may reuse measured development
outcomes under a recorded chronological diagnostic, not claim prospective
validation. Preserve the score-blind recipe, exact executor/reference, 40-atom
T4 support, original-seed constraints, ancestry and all work accounting. New
remote jobs, oracle calls, the suggested 24-run PMO comparison and further T4
campaigns need separate bounded launch contracts. Acceptance and decisions:
`docs/PARENT_EDIT_CONTROLLER.md`. Earlier frozen artifacts remain unchanged.

**Scoped second-generation docking approval, 2026-09-12:** after the request
for 73 calls and clarification that $20 is a ceiling rather than expected spend,
the user approved proceeding. Score all 49 distinct locked attempt_2 queries,
then obtain two fresh evaluations of each arm/cell champion and each of the four
first-generation incumbents, sharing identical confirmation queries. At most
73 new calls, eight single-CPU workers plus one driver, no GPU or automatic
oracle retries, and $20 maximum reserved cost. Reuse the exact candidate pools,
original-seed gates and target-specific docking protocol; preserve the FA7
ranked shortfall and separate arm label access. Launch from a clean committed
source and publish restart-safe receipts and measured archives. No regeneration,
further paid generation, reference training or benchmark-win claim follows.
Contract: `docs/T4_SECOND_GENERATION.md` and its frozen scoring manifest.

**Scoped measured second-generation preparation, 2026-09-12:** following the
completed four-target round, the user requested program-level exploitation of
the 32 measured outcomes and whole-program size features, without changing the
architecture or 40-atom support. Implement source/peak/final size accounting,
retain measured-score parent allocation and an explicit exploration floor, and
prepare one bounded local second-generation comparison on the same four cells.
Use score-ranked versus score-blind parents from the identical paid archive,
with the same mutation/recombination/broad channels and no surrogate veto.
Do not assign a shrinkage reward or transfer numeric scores across targets.
This preparation makes zero new oracle calls. A second paid generation exceeds
the two remaining calls of the previous allocation and requires a new bounded
call/cost approval. No additional model training or executor/reference change.
Recipe and acceptance: `docs/T4_SECOND_GENERATION.md`.

**Scoped four-cell program curriculum, 2026-09-12:** the user explicitly requested
continuing onto other docking targets, using their released winners aggressively
as shared development material, and letting the controller run. Authorize one
bounded cold-start scoring round: JAK2 seed1, FA7 seed0, BRAF seed1 and 5HT1B
seed0 at delta=0.4, eight already locked program candidates plus one seed control
per cell, exactly 36 new calls maximum. The prior follow-on used 28 calls;
cumulative maximum is 64, within the existing 66-call/$20 follow-on ceiling.
Use at most eight CPU workers plus one driver, no automatic retry, the same
shared library and mutation recipe across cells, and the existing docking
pipeline with individually hash-verified target receptors. Freeze
`configs/t4_program_curriculum_lock.json` before scoring. This is the initial
paid archive for a development curriculum, not a full-suite benchmark or a
matched adaptive/static superiority result. Preserve failures, exact traces,
source exposure and old results. No subsequent paid round follows automatically.

**Scoped shared-controller transfer, 2026-09-12:** the user requested retaining
the measured -13.3 PARP1 result and making coordinated, context-bound complete
programs the default architecture, then freezing one recipe across T4. Prepare
the same shared program library and cold-start adapter on JAK2 seed1, FA7 seed0,
BRAF seed1 and 5HT1B seed0, using existing exact source states. These form the
user-requested small development curriculum, not the frozen final comparison.
Use one shared library, including the newly authorized public winner-derived
programs from all tasks, not separate target-specific scripts. Do not transfer
PARP1 scores as new-target labels. Preserve the broad
reference channel and measured-feedback implementation. The initial local
preparation uses zero new oracle calls; a full-suite paid sweep is not authorized
by the existence of these prepared candidates. Freeze a bounded launch contract
within the recorded call/cost authority before any remote execution. No change
to the reference, executor, frozen evaluation gates or editing-V2 training lane.

**Scoped adaptive program-optimizer integration, 2026-09-12:** following the
completed 28-call pool, the user explicitly requested promoting coordinated
programs into the main implementation, variable attachments/parameters, branch
replacement/recombination, and score-adaptive archive search. Authorize this
implementation and local cache-only development using existing paid records.
The next development recipe uses top-level 70/20/10 mutation/recombination/broad
dispatch before legacy WHERE, configurable primitive/block caps, exact serial
execution, endpoint-only gates, durable state and measured-outcome updates.
This is a new optimization proposal, not an exact reference law or Doob claim.
No neural model training, new oracle evaluation or remote deployment follows
from this implementation unit. The suggested 108-call campaign exceeds the
previous 66-call ceiling and remains a separate budget decision. Existing
reference/executor semantics, source roles, completed locks and training gates
remain unchanged. Acceptance and local commands: `docs/ADAPTIVE_PROGRAM_OPTIMIZER.md`.

**Scoped multi-site program extension and bounded docking approval, 2026-09-12:**
the user explicitly requested adding jointly planned edits across disconnected
mutable sites, dependency/conflict separation, and serial executor revalidation.
Implement an optional two-to-four-site program channel, preserving the existing
single-region and generic channels. Reuse the attachment-program backend and
exact traces, and compare uninterrupted serial, independently chosen multi-site,
and jointly conditioned multi-site proposals without intermediate task pruning.
The user's subsequent full Modal approval authorizes a bounded follow-on docking
pilot, not unlimited compute: retain at most 66 new calls, 30 total containers,
CPU only, and a $20 reserved ceiling. Freeze a task-specific recipe, contexts,
candidate locks, repeat allocation and work limits before launch; adding the
multi-site arms must fit that ceiling or remain a separate zero-oracle assay.
Use a clean committed source, existing production docking protocol and durable
receipts. No reference/executor change, hidden-oracle screening, new outer-test
selection or benchmark-superiority claim follows from this development scope.
This supersedes the offline-only restriction immediately below, not the
editing-V2 training ladder or the separate nineteen-call refinement recipe.

The next concrete use of that approval is the smaller locked-pool diagnostic in
`docs/T4_PROGRAM_POOL.md`: score all 26 eligible non-winner molecules from the
corrected, already-locked two-context/three-arm pools, then repeat the best twice.
At most 28 new calls, eight workers plus one driver, CPU only, and $20 reserved.
Reuse the three completed winner controls. It is NOT the proposed three-context,
eight-per-arm pilot and makes no matched benchmark claim. Preserve the confounded
earlier diagnostics and their exclusion notices.

**Scoped attachment-specific program development, 2026-09-12:** the user approved
trying the supplied program-proposal recommendation. The immediate implementation
unit is a local, zero-oracle typed program/replay adapter and context-conditioned
attachment proposal, reusing exact saved demonstrations and immutable public scored
candidate pools. Preserve existing source roles, label external scores separately,
and keep answer-known reconstruction distinct from transfer and discovery. This
optional executor-supported proposal does not modify R_theta, executor semantics,
the existing WHERE law, or its KL contracts. Any eventual departure from the old
controller is a separately identified proposal arm, not an exact Doob claim.
The proposed 66-call comparison remains a separate unapproved experiment. No
remote deployment, new oracle calls, or large training/data run follows from this
local implementation scope. Preserve the previous bounded refinement recipe.

**Scoped winner-initialized T4 refinement, 2026-09-12:** the user approved the
proposed small experiment starting from a published IVG winner. Implement and run
the recipe in `docs/T4_WINNER_REFINEMENT.md` and
`configs/t4_winner_refinement.json`: one exact supported PARP1 seed0 delta=0.4
winner, a broad primitive census and sixteen short existing-controller streams,
at most nineteen new docking calls including winner and best-candidate repeats,
CPU only, at most eight workers plus one driver, and $10 reserved. The numeric
caps are the bounded implementation of that approval. No model fitting,
reference/executor change, automatic follow-on round, or matched benchmark
superiority claim is authorized. Keep original-seed endpoint constraints,
immutable candidate locks, exact trace replay and full call accounting.

The first launch spent three of those nineteen calls on completed winner
controls, then failed at a stale training-chain identity. The bounded repair may
reuse the already-qualified, dependency-matching inference package and those
exact controls. Preserve the failed receipt and gate, verify source/tensor/law
parity, and spend at most sixteen additional calls under the unchanged recipe.
No new replicate, changed proposal law, or new control selection is authorized.

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
