# COMPOSE Parallel Experimental Workstreams and Handoff Protocol

**Purpose.** Advance the remaining ICLR experiments in parallel while the main workstream continues debugging and completing same-prefix dynamic retargeting. Each parallel lane must answer a different paper claim, use frozen COMPOSE components, and stop before claim-bearing held-out evaluation unless the main workstream explicitly authorizes it.

## Paper-level thesis

COMPOSE should read as one causal chain, not a bag of capabilities:

1. **Legality — exact executor.** The rewrite kernel defines the executable successor graph over complete molecules.
2. **Plausibility — learned reference process.** `R_theta(y|x)` learns which legal continuations are molecularly supported; it is not merely a global operator-frequency model.
3. **Purpose — finite-horizon control.** Goal information reshapes the reference process without retraining the molecular dynamics.
4. **Intervention — trajectory-level design.** Realized histories can be retargeted and constrained because every intermediate state is itself a molecule.
5. **Adaptive computation.** Cheap local control is used when sufficient; verified future-aware computation is focused where local decisions destroy downstream reachability.

The completed exact-target branch already supports the third and fifth layers: myopic control left known-reachable targets unrecovered, verified remaining-budget rollout improved recovery, and goal-aware shortlisting reduced continuation evaluations while preserving the greedy-safe option. Parallel work should broaden the paper, not reopen that branch.

---

# Workstream B — Claim 2: What molecular transport did `R_theta` learn?

## Mission

Characterize the frozen reference process as a stochastic molecular transport law. Show whether its multi-step trajectories are coherent, mobile, chemically distribution-preserving, and non-pathological relative to support-matched unlearned laws.

This is **not** another likelihood experiment. Experiment 1 already established that `R_theta` beats uniform legal rewriting and the empirical-family law. This workstream asks what that learned preference produces when iterated.

## Scientific claim

> The executor defines what can happen; `R_theta` organizes those legal edits into coherent molecular trajectories that achieve nontrivial structural movement without the pathological drift, cycling, or operator collapse induced by unlearned same-support transport laws.

Do not claim physical dynamics or synthetic route plausibility. These are **design dynamics** over executable edits.

## Frozen inputs

- Frozen Process-V2 chemistry kernel and process identity.
- Frozen selected `R_theta` development checkpoint.
- Frozen matched reserve and source exclusions.
- Existing support-band definitions.
- No retraining, objective changes, corpus edits, or operator changes.

## Panel construction

### Development panel

Choose a deterministic held-in development panel for metric debugging only.

### Confirmatory panel

After metrics and horizon are frozen, carve a fresh matched-reserve panel disjoint from:

- exact-target development and sealed panels;
- retargeting sources;
- pathwise-constraint development sources;
- global final test.

Stratify by:

- scaffold support band `0 / 1–4 / 5–24 / 25+`;
- molecular heavy-atom size band;
- ring-containing versus acyclic sources.

The source molecule is the independent statistical unit. Multiple rollout seeds from one source are repeated measures, not independent examples.

## Arms

Use the exact same legal canonical-successor support for every arm.

1. **Full learned reference:** `R_theta`.
2. **Uniform canonical successor:** uniform over distinct productive canonical successors after alias collapse.
3. **State-independent empirical-family law:** frozen realized family frequencies, renormalized over legal families at each state, uniform over distinct successors within family, summing cross-family aliases.

Optional fourth arm only if already implemented and cheap:

4. **Empirical family + learned identity:** isolates the within-family scorer that Experiment 1 showed carries most of the learned signal.

Do not add external generators here; they do not share the executable support and cannot isolate the reference-process claim.

## Rollout protocol

- Productive-edit horizon: start with `H=6`; compare `H=8` only on held-in smoke if `H=6` is too short to expose structural movement.
- Use the same horizon for every arm.
- Use several fixed stochastic seeds per source.
- Commit every accepted canonical successor; do not use controller objectives.
- Count kernel calls and wall time, but performance is not the scientific endpoint.

Before scale, run an 8-source smoke to measure actual enumeration cost and verify all three arms genuinely produce different transition laws.

## Primary metric family: mobility–fidelity frontier

No single metric should be allowed to define “good trajectories.” The main result should be a frontier.

### A. Structural mobility

At each step and at the endpoint report:

- ECFP4 Tanimoto distance from the source;
- heavy-atom-count change;
- ring-count and ring-system change;
- bond-edit and atom-edit cumulative counts;
- fraction of trajectories using at least two operator families;
- endpoint uniqueness across repeated rollouts from the same source.

### B. Trajectory health

- unique-state fraction;
- immediate reversal rate;
- two-cycle rate;
- any-state revisit rate;
- no-op/self-transition rate if such events exist;
- early dead-end rate;
- productive trajectory completion rate;
- operator-family entropy and capability-cell coverage.

Do not assume fewer reversals is always better; report it and compare. Some reversible behavior may be part of the reference law. The paper claim should focus on absence of pathological domination.

### C. Chemical-envelope retention

Calibrate a descriptor envelope on held-in molecules before looking at confirmatory trajectories. Use simple reproducible descriptors:

- molecular weight;
- cLogP;
- QED;
- formal charge;
- H-bond donors/acceptors;
- ring count;
- fraction sp3;
- heavy-atom count.

Report:

- fraction of all intermediate states inside the pre-frozen multivariate/coordinate-wise envelope;
- standardized descriptor drift versus edit count;
- endpoint distribution distance to the held-out reference population using a frozen descriptor-space metric.

This is a distributional sanity check, not proof of medicinal plausibility.

### D. Support generalization

Report the above metrics separately across scaffold-support bands. The desired story is not “support does not matter”; it is an honest characterization of how mobility and fidelity change as training support weakens.

## Main hypothesis

The strongest preregistered hypothesis should be:

> At matched edit horizon and identical executable support, `R_theta` occupies a better mobility–fidelity frontier than uniform canonical and empirical-family transport: it achieves nontrivial structural displacement while reducing pathological revisits and descriptor drift, without collapsing operator coverage.

This avoids requiring every individual metric to move in a predetermined direction.

## Figure design

**Panel A:** three representative source-conditioned trajectory fans for `R_theta`, uniform, empirical-family.

**Panel B:** mobility versus descriptor-envelope retention at steps 1–6.

**Panel C:** revisit/two-cycle/operator-entropy table.

**Panel D:** support-band stratification.

Choose qualitative examples only after quantitative results are frozen; examples illustrate, not establish, the claim.

## Stop rules

Stop and report rather than retune if:

- all arms are nearly identical because the horizon is too short after the single held-in horizon check;
- `R_theta` collapses to one family or cycles pathologically;
- descriptor metrics reveal broad off-distribution drift;
- the metric suite cannot distinguish mobility from trivial growth/shrinkage.

Do not retrain `R_theta` in this lane.

## Deliverables

Suggested repository paths:

- `docs/workstreams/claim2_trajectory/PROTOCOL.md`
- `configs/claim2_trajectory_protocol_v1.json`
- `modal_apps/claim2_trajectory_characterization_app.py`
- `scripts/analyse_claim2_trajectories.py`
- `diagnostics/claim2_trajectory_smoke.json`
- `docs/workstreams/claim2_trajectory/HANDOFF.md`

The parallel agent may take this workstream through held-in smoke and, if the protocol is frozen cleanly, through matched-reserve results. It must not touch the global final test.

---

# Workstream C — Pathwise constraints

## Mission

Show that COMPOSE controls **which molecular states may appear throughout a trajectory**, not merely whether the final molecule passes a filter.

## Scientific claim

> Because every COMPOSE state is a complete molecule and every transition is executable, structural requirements can be imposed on the support of the entire controlled process. This prevents forbidden intermediate molecules and lets future-aware control optimize within the constrained reachable set.

This is distinct from endpoint generation plus post-hoc filtering.

## Constraint choice

Use a protected labeled subgraph, preferably a ring-containing core or pharmacophore-like motif automatically derived from each source.

A source is eligible when:

- the protected motif occupies a registered fraction of the molecule, e.g. neither nearly the whole molecule nor a trivial one-atom pattern;
- there are enough atoms outside the motif to permit meaningful edits;
- the frozen kernel offers both motif-preserving and motif-destroying legal successors;
- the terminal objective has measurable headroom.

The pathwise constraint is:

> At every committed state, at least one exact atom/bond-labeled subgraph embedding of the protected motif must exist.

Do not define preservation by fingerprint similarity. Use explicit subgraph matching under frozen atom/bond semantics.

## Goal

Use a target-free property objective with headroom, chosen from an already frozen goal language if available. The constraint experiment should not invent another bespoke oracle.

The main result does not require future-aware control to beat greedy. It requires:

1. endpoint-only handling to allow forbidden intermediates;
2. support masking to guarantee zero path violations;
3. reasonable terminal utility under the guarantee;
4. ideally, future-aware control to recover some utility lost by the mask.

## Held-in viability gate

Before any confirmatory panel:

- run unconstrained trajectories on held-in sources;
- measure the fraction that ever violate the protected motif while ending with a motif-valid endpoint;
- measure the fraction of candidate successors removed by the pathwise mask;
- verify that feasible trajectories can still improve the terminal objective.

A useful regime has:

- nonzero endpoint-valid/path-invalid trajectories, proving endpoint-only filtering is insufficient;
- neither a vacuous mask nor one that removes nearly all support;
- feasible objective headroom.

If the constraint is vacuous or impossible under the frozen rule, stop. Do not keep changing motifs until one makes the desired arm win.

## Arms

All internal arms use the same frozen `R_theta`, source set, terminal objective, and oracle budget.

1. **Unconstrained greedy or verified control.** Context only.
2. **Endpoint-only generation/filtering.** Generate under the unconstrained policy; at the end select the best motif-valid endpoint under the same total trajectory/oracle budget.
3. **Pathwise mask + greedy.** Remove every action whose successor violates the protected motif.
4. **Pathwise mask + verified remaining-budget control.** Optimize within the constrained support.
5. **Mask-only sampling from `R_theta`** if useful to separate feasibility from objective control.

Every pathwise arm must have zero committed violations by construction. Do not present a p-value for zero violations.

## Primary metrics

- any-intermediate violation rate;
- endpoint validity under the motif;
- final objective utility/success among feasible trajectories;
- feasible-trajectory completion rate;
- fraction of legal support removed by the mask at each step;
- terminal-utility cost of the guarantee versus unconstrained control;
- utility recovered by future-aware control versus mask+greedy;
- kernel/oracle cost.

## Main comparison

The central causal contrast is:


a. **endpoint-only:** can finish valid after traversing forbidden states;

b. **pathwise mask:** forbidden states are unreachable by construction;

c. **pathwise mask + future-aware control:** searches effectively within the reduced reachable set.

## External baseline

GraphXForm is the first baseline to qualify because it supports graph editing from an existing molecule and action masking/structural constraints. The baseline agent should verify the exact semantics before any claim-bearing comparison.

Do not force static generators without path-state semantics into a pathwise benchmark. They can be endpoint-only context baselines.

## Figure design

**Panel A:** source molecule with highlighted protected motif.

**Panel B:** endpoint-only trajectory that temporarily destroys the motif and later restores it.

**Panel C:** pathwise COMPOSE trajectory preserving the motif at every state.

**Panel D:** violation–utility tradeoff and support removed per step.

## Stop rules

- protected motif almost never violated by unconstrained control → constraint is vacuous;
- mask removes nearly all candidates → task is infeasible under this support;
- endpoint-only and pathwise trajectories are indistinguishable → no trajectory-level claim;
- future-aware adds nothing under the mask → retain the pathwise guarantee claim but drop the planning-under-constraint subclaim.

## Scope of parallel agent

The agent should complete:

- exact motif-preservation predicate;
- held-in panel eligibility;
- all internal arm implementations;
- unit tests and 4–8 source smoke;
- cost estimate;
- protocol document and stop-rule verdict.

It should **stop before any held-out claim-bearing run** and hand the frozen design back to the main workstream.

## Deliverables

- `docs/workstreams/pathwise_constraints/PROTOCOL.md`
- `configs/pathwise_constraints_protocol_v1.json`
- `modal_apps/pathwise_constraints_app.py`
- `tests/test_pathwise_constraint.py`
- `diagnostics/pathwise_constraints_smoke.json`
- `docs/workstreams/pathwise_constraints/HANDOFF.md`

---

# Workstream D — External-baseline qualification and adapter readiness

## Mission

Prepare faithful external comparisons without delaying or contaminating the internal causal experiments. This lane does **not** decide winners and does not run final sweeps.

## Core rule

A baseline is included only where its native scientific object matches the COMPOSE claim. “N/A” is preferable to a distorted adaptation.

## Comparator map

### MARS

Best suited for:

- static property optimization;
- source-initialized iterative graph editing;
- possibly goal-switch reoptimization from the realized switch molecule.

Qualification questions:

- Can it start from an exact supplied molecule?
- Can its objective change without retraining its molecular proposal?
- Are adaptive proposal statistics reset or carried across a switch?
- How are rejected proposals and oracle calls counted?

### GraphXForm

Best suited for:

- source-conditioned graph editing;
- static optimization;
- structural/action constraints;
- pathwise-constraint comparison if its mask semantics truly apply at each step.

Qualification questions:

- Does the model require objective-specific fine-tuning?
- Can it preserve a specified subgraph exactly?
- Does it expose the full intermediate trajectory?

### DDSBM

Best suited for:

- source-to-target or distribution-to-distribution graph transformation;
- source-conditioned molecular editing.

Do not force it into unexpected same-prefix retargeting unless the official method natively supports mid-trajectory goal replacement.

### GraphGA / REINVENT / PMO baselines

Best suited for:

- conventional static property optimization;
- oracle-budget comparisons.

They are not pathwise or same-prefix baselines unless their native implementations support those semantics.

### HN-GFN / preference-conditioned multiobjective methods

Best suited for:

- target-free multiobjective optimization;
- preference generalization;
- Pareto coverage.

They are related work for flexible goals, but not automatically same-prefix intervention baselines.

## Fairness contract

The adapter registry must record for each method:

- native task and claim addressed;
- source conditioning support;
- whether objective-specific retraining is required;
- operator/action support;
- pathwise-state availability;
- terminal oracle definition;
- edit/trajectory budget;
- all oracle calls, including rejected proposals;
- number of generated molecules;
- wall time and hardware;
- code commit, model checkpoint, license;
- deviations from the official protocol;
- status: `MUST_RUN`, `CONDITIONAL`, or `CONTEXT_ONLY`.

Do not compare raw wall time across incompatible hardware as the primary metric. Oracle calls and achieved objective under fixed budget are primary for optimization.

## Priority order

1. MARS adapter qualification.
2. GraphXForm adapter qualification.
3. GraphGA/REINVENT static wrappers.
4. DDSBM native transformation protocol.
5. One preference-conditioned multiobjective method.

## Agent scope

The agent may:

- inspect official papers/repos;
- create isolated environments;
- implement adapters;
- run held-in smoke tests on 3–5 sources;
- record failures and incompatibilities.

The agent may **not**:

- alter COMPOSE protocols to accommodate a baseline;
- run final held-out sweeps;
- exclude a baseline because preliminary results are strong;
- report literature numbers as directly comparable without protocol matching.

## Deliverables

- `configs/comparator_registry_v3.json`
- `docs/baselines/COMPARATOR_MATRIX.md`
- `docs/baselines/FAIRNESS_CONTRACT.md`
- `baselines/<method>/README.md`
- `baselines/<method>/environment.lock`
- `diagnostics/baselines/<method>_smoke.json`
- `docs/workstreams/baselines/HANDOFF.md`

---

# What should not be parallelized now

Do not start:

- another exact-target `h_phi` project;
- new teacher-data generation;
- new corpus mining or operator design;
- fresh paper-bearing `R_theta` seeds;
- global final-test evaluation;
- a new target-free control benchmark separate from the current retargeting work;
- architecture sweeps.

Those either reopen answered questions or depend on protocols still being frozen.

---

# Shared handoff machinery

## Branch/worktree isolation

Each agent gets a separate worktree and branch:

- `codex/compose-claim2-trajectory`
- `codex/compose-pathwise-constraints`
- `codex/compose-baseline-qualification`

No agent commits onto the main retargeting branch.

## Required workstream directory

Each lane must maintain:

```text
docs/workstreams/<workstream>/
├── STATUS.md
├── PROTOCOL.md
├── DECISION_LOG.md
├── HANDOFF.md
└── handoff.json
```

### `STATUS.md`

One screen only:

- current status;
- current commit;
- what is running;
- last completed gate;
- next action;
- whether any held-out data have been opened.

### `PROTOCOL.md`

Scientific contract:

- claim;
- non-claim;
- frozen inputs;
- panel construction;
- arms;
- primary and secondary metrics;
- stop rules;
- allowed calibration;
- forbidden adaptations.

### `DECISION_LOG.md`

Every material decision with:

- date/time;
- decision;
- evidence available before decision;
- alternatives rejected;
- whether it changes a frozen object;
- commit containing the decision.

### `HANDOFF.md`

Human-readable final packet using the template in the companion file.

### `handoff.json`

Machine-readable manifest containing hashes, commands, and artifact paths.

## Artifact status vocabulary

Every result artifact must declare exactly one:

- `DESIGN_ONLY`
- `SMOKE_HELD_IN`
- `DEVELOPMENT`
- `CONFIRMATORY_HELD_OUT`
- `SUPERSEDED`
- `INVALID_INSTRUMENT`

A main-session agent must never accidentally cite a smoke result as claim-bearing.

## Durability requirements

- No load-bearing output may exist only in `/private/tmp`, a scratchpad, or a local untracked file.
- Every expensive job writes per-task durable shards before returning.
- The aggregate must be reconstructible from shards.
- Every frozen model file must be tracked or stored on a durable volume and hash-bound in a committed manifest.
- All commands must use durable paths or explicit required arguments; no session-specific hardcoded paths.

## Handoff acceptance gate

The main workstream should accept a handoff only if:

1. branch is pushed and working tree clean;
2. `handoff.json` validates;
3. all frozen-input hashes match the main branch;
4. held-out-open status is explicit;
5. exact reproduction commands are present;
6. known defects and invalid runs are listed;
7. recommended next action is bounded and does not silently expand scope.

## Main-session pickup procedure

1. Read `STATUS.md` and `HANDOFF.md`.
2. Verify `handoff.json` and referenced hashes.
3. Inspect the protocol diff before result files.
4. Confirm whether results are smoke, development, or confirmatory.
5. Cherry-pick only protocol/infrastructure commits first.
6. Reproduce one small smoke command.
7. Decide explicitly whether to authorize held-out evaluation or merge results.

## Progress updates

Parallel agents should report only when:

- a preregistered gate resolves;
- a bug invalidates an instrument;
- cost/runtime changes materially;
- a held-out boundary is about to be crossed;
- the workstream is ready for handoff.

Do not stream every shard count into the main conversation.

---

# Recommended immediate assignment

## Agent 1 — Claim 2

May proceed through a complete held-in smoke and metric freeze immediately. It may run the matched-reserve characterization only after the protocol and panel hashes are committed and the main workstream has reviewed the smoke.

## Agent 2 — Pathwise constraints

May implement and smoke-test all internal arms on held-in sources. It must stop before held-out panel selection/evaluation.

## Agent 3 — Baselines

May fully qualify adapters and environments, but only held-in smoke runs are allowed. No final benchmark sweep.

## Main workstream

Continues same-prefix retargeting. It owns all decisions about:

- retargeting held-out confirmation;
- opening new held-out panels;
- merging parallel branches;
- final experiment ordering;
- paper-bearing seeds and final test.

