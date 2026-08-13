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


---

# Main-agent handoff evaluation order (binding)

When a lane's packet arrives, do NOT merge the branch or ask the agent for more
experiments. Process in this order, and note that **results come fourth**:

1. **`PROTOCOL.md` — before any result.** Does the scientific question still
   match the paper story?
2. **`DECISION_LOG.md`.** Were any choices made *after* seeing outcomes that
   should have been frozen earlier?
3. **`handoff.json`.** Base commit, frozen model/data hashes, held-out status,
   artifact locations.
4. **Then the results.**

Then exactly one of three decisions — no fourth option:

- **MERGE** — the lane answered its question cleanly.
- **AUTHORIZE NEXT GATE** — e.g. permit a held-out run.
- **STOP / REVISE** — the instrument or the scientific premise failed.

The ordering exists to stop us being seduced by a good number before checking
whether the experiment that produced it was the one we intended.

# Project-wide rule: could this statistic have gone the other way?

**Before reporting any statistic, ask what value it could take if the hypothesis
were false. If the answer is "none", it is not a measurement.**

Adopted after three defects in a single day, all the same shape — a quantity
whose sign was fixed before any data existed:

1. **sacrifice-to-win** — the action was chosen as `argmax V_G` and then scored
   by `V_G`. 50 higher, 5 tied, 0 lower out of 55; zero losses was definitional.
2. **a "verified" arm that was secretly greedy** — `verified_land` was a plain
   greedy continuation and the decision loop committed greedy's action, so
   headroom was 0 by construction and two runs' CEILING verdicts were void.
3. **a sign test against a false null** — policy improvement guarantees
   `verified >= greedy`, so testing against 0.5 tested something already known.

Related trap, same family: a constraint arm that *cannot* violate its constraint
by construction has not demonstrated anything by not violating it. The
measurement is what the *unconstrained* arms do, and what the constraint costs.

This rule applies to every lane and to the main lane equally.

---

# Project-wide rule: MMP proxies are diagnostics, not authorities

**Do not use one-cut matched-pair proxies for load-bearing reachability or
control-geometry conclusions when exact successor fibers are available.**

Two measured cases, both of which changed the qualitative picture:

1. **Reachability.** A matched-pair movability census suggested a DRD2 potency
   threshold was out of reach in four edits — median favourable edit +0.38
   log-odds against a required climb of ~5.5. The real fiber then climbed
   **+4.42 log-odds in three greedy edits**. The proxy sees a median of one
   neighbour per molecule; the fiber is ~500–600 wide.
2. **Control geometry.** The Pareto census ran both instruments on the same
   objective pair. The MMP proxy gave 1.66 mean distinct preference selections
   and 41% unanimous states; the exact fibers gave **2.34 and 10%**. The proxy
   reads borderline where the executable support is comfortable.

That is not a calibration offset — it changes how much choice the controller
appears to have.

**Consequence.** MMP mining stays useful as a cheap diagnostic and for
nominating real chemical relationships. It is not evidence about executable
support. Any manuscript claim resting materially on MMP-based reachability or
support estimates must be re-evidenced with exact-fiber measurement where
feasible.

---

# Project-wide rule: provenance must resolve

**Every load-bearing internal claim must resolve to a committed artifact or
manifest hash. Every external capability claim must resolve to an actual
primary-source citation. If it cannot be traced, it stays `UNVERIFIED`.**

No prose-only provenance. No "according to sub-agent X." No citation to a
report that has not arrived.

Adopted after two fabrications in one day, both self-caught, neither of which
contaminated a measurement:

1. A **fabricated SHA-256 tail** — real 16-character prefix, invented
   remainder — in a lane's `handoff.json`.
2. A **fabricated attribution** — a committed artifact stated that a check had
   "REFUTED a delegated claim" from a sub-agent whose report had not yet
   arrived. The measurement was real; the citation was not.

The pattern is specific and worth naming: **agents fabricate provenance far
more readily than they fabricate numbers.** A hash tail or a citation feels
like bookkeeping while writing and becomes load-bearing when read. And the
failure is asymmetric — *a wrong number gets re-measured; a fake citation gets
trusted.*

Both were caught because the artifacts are machine-checkable. Keep them that
way: recompute digests by test, and require every capability cell to name a
paper section, a file and line, or a commit.

This is the whole rule. Do not build further process around it.

---

# Project-wide rule: every efficiency claim must name its resource axis

**A reduction in complete molecular trajectories does not imply a reduction in
oracle evaluations or in kernel calls. Never write "sample efficient" without
saying which sample.**

Three distinct axes, never substituted:

| axis | quantity | scope |
|---|---|---|
| **trajectory efficiency** | HV vs complete molecular trajectories | COMPOSE-internal ONLY |
| **process-compute efficiency** | HV vs kernel / reference-process calls | COMPOSE-internal |
| **oracle efficiency** | HV vs `oracle_requests` and unique valid evaluations | **the common currency for external comparison** |

Trajectories are not a shared object — HN-GFN, GraphXForm and GraphGA do not
have one — so external methods are never plotted on that axis.

**Why this rule exists.** In the Pareto smoke, `greedy_pref` records
**6,400–15,620** oracle requests at **13–23** kernel calls, while kernel-matched
`gen_rank@greedy` records **1–4**.

> **THAT RATIO IS NOT YET INTERPRETABLE AND MUST NOT BE QUOTED AS A COST
> COMPARISON.** It is pending an accounting audit. At matched *kernel* budget,
> preference control is charged for interrogating the ~600-wide legal successor
> fiber at every decision, while generate-and-rank is charged only for the
> terminal molecules it produced. Those are different acts, so the ratio
> currently mixes "property evaluations per unit of kernel work" with
> "cost of optimizing a molecule". Four things must be separated before any
> number is quoted: whether a request is a distinct evaluator invocation or a
> row in a vectorised batch; expensive DRD2 evaluations versus cheap
> deterministic QED/cLogP descriptors; requests versus post-cache evaluator
> calls; and marked actions versus distinct canonical successors after alias
> collapse.

Whatever the audit finds, both of these can be simultaneously true:

> COMPOSE covers the Pareto front with far fewer molecular trajectories.
> COMPOSE covers the Pareto front with *more* oracle evaluations.

The internal result table must therefore carry all of it per arm — final HV,
HV-AUC, trajectories, kernel calls, oracle requests — so a universal efficiency
win cannot be claimed by selecting an axis.

"Explicit control is trajectory-efficient but oracle-intensive" is a legitimate
and informative outcome, not a failure. It would also identify the single
follow-up worth doing — goal-aware allocation of expensive computation, which
is the role the exact-target work already established for prioritisation. That
follow-up is justified ONLY if the completed experiment shows the preference
capability is real, exhaustive scoring is materially oracle-inefficient, and
reducing it is necessary for the claim.

# Project-wide rule: every new metric needs an adversarial fixture

**Knowing the anti-tautology rule is not sufficient.** Seven statistics with
signs fixed by construction have now been caught here, and the seventh appeared
in *new* code written by an agent who had already internalised the rule — a
potency-watch statistic using `argmin`, which always returns something and so
could never report "neither axis dominates".

So: every new analysis metric ships with a fixture in which its intended
conclusion is **false**, and a test asserting the metric says so. This is the
whole rule; do not build more process around it.

---

# Project-wide rule: no hyperparameter search for a favourable operating point

**Do not sweep a knob and select the attractive value.** Where an efficiency or
capacity parameter must be chosen, pre-register ONE operating point from a
stated requirement — a target reduction in oracle demand, a fixed query budget,
a structural argument — and report the outcome at that point.

The prohibited pattern is running `K ∈ {4, 8, 16, 32, 64, 128}` and reporting
whichever K looks best. That is choosing the result and then naming a
configuration for it.

Where a curve is genuinely the object of study, say so in advance and report the
whole curve, not its maximum.

The same applies to comparisons defined by a resource: report **fraction of
quality retained against fraction of resource spent**, which is a curve with a
meaning, rather than a single ratio at a chosen point.

---

# Interpretation rule: pointwise improvement is not set-level improvement

**A guarantee on each element of a set does not transfer to a property of the
set.** Check which one your metric measures before deciding whether its sign is
fixed.

Concretely, and the reason this rule exists:

- In **exact-target recovery**, verified control's advantage over greedy IS
  sign-fixed. Greedy's action is always in the candidate set and strict
  improvement never commits a lower `V_G`, so the landing value cannot be worse.
  That comparison was correctly barred as a claim.
- In **Pareto control**, verified control guarantees the scalarized continuation
  value *for each requested preference* — but hypervolume is a property of the
  **endpoint set**. Five individually better points can enclose less dominated
  area than five worse but better-spread ones. So `HV_verified − HV_greedy` has
  a genuine falsifying range and **is** a valid comparison.

Measured: source 000 returned verified HV **below** greedy (1.0060 vs 1.0074)
while per-preference values improved.

**Do not bar a comparison by analogy.** The two cases look identical and are
not. Report the two questions separately — per-preference scalarized improvement,
and set-level frontier quality — and treat disagreement between them as a
finding rather than an error: it distinguishes optimising trajectories
individually from constructing a complementary set.

# Interpretation rule: a normalizer is not a bound

A scale frozen from data (a p99, an IQR, a held-in quantile) **normalizes**; it
does not **cap**. Values may legitimately exceed it.

Held-in-scaled hypervolume can exceed 1 because `z*` is the held-in p99, not an
attainable utopia. The fix when this is discovered is to **correct the prose**,
never to clip the metric — clipping suppresses exactly the signal the excess
carries and hides ceiling effects. Report exceedance descriptively; do not
introduce a robust or clipped variant after seeing a result.

Ratios built on the same normalizer are unaffected, since the constant cancels.

# Meta-rule: a sign guarantee belongs to an estimand, not to an arm pair

Every `GUARANTEED_SIGN` declaration must carry **both**:

1. **a written mathematical reason** the sign is guaranteed, naming the estimand
   it applies to; and
2. **an adversarial fixture** showing that a NEIGHBOURING metric on the SAME
   arm pair, which lacks that guarantee, can go the opposite direction.

A green test must establish the **scope** of a guarantee, not merely perpetuate
the declaration.

**Why.** A registry keyed on the contrast alone marked every metric on
`verified_pref` vs `greedy_pref` as sign-guaranteed — hypervolume included —
and suppressed the comparison. A passing test asserted exactly that, so the
error was held in place by the machinery meant to prevent it. Only the
per-preference scalarized continuation value that verified control explicitly
optimises carries the guarantee; hypervolume is a set-level object and moves
either way.

**Permanent regression case:** Pareto smoke source 000, where verified improved
the preference-conditioned scalarized values while `HV_verified` (1.0060) fell
*below* `HV_greedy` (1.0074). If a future refactor starts suppressing that
comparison again, this must fail loudly.

Note the shape of the two late failures this rule addresses: in both, the
**guard** rather than the metric was the sign-fixed object. A wrong metric gets
re-examined; a test enforcing a wrong metric makes re-examination look like a
regression. The anti-tautology rule must point at its own guards.

---

# Pending fix: Stage B has no per-arm checkpointing

`modal_apps/pathwise_stage_b_app.py` writes a durable shard per source but no
per-arm partial, unlike the retargeting and Pareto apps. A preemption there
costs the **whole task** rather than one arm, and intra-task progress is
invisible — which is why Stage B reports 0 partials while the other lanes
report many.

Completed per-source shards are safe and the resumable driver skips them, so
this is a cost and visibility gap rather than a correctness risk.

**Fix it before any Stage B rerun or held-out confirmation, and not before the
current run finishes** — touching a live app to add durability would create more
risk than it removes.
