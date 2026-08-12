# COMPOSE — Canonical Experiment Plan

**STATUS: CANONICAL. This supersedes every prior experimental plan, including the
manuscript's existing experimental section and the A/B/C framing used earlier in
development. Do not execute or cite the old plan.** The old draft describes a
materially different model: a Metropolis-corrected forward CTMC toward a
prescribed high-entropy prior with a time-dependent learned reverse scheduler,
learned event clock and trajectory likelihood, centered on unconditional
GuacaMol-style distribution learning. That is not the system we have.

### Documents this supersedes — do not read, cite or execute

Found in the repo at the time this plan was written. Several are titled as
though authoritative; none of them is. **This file is the only current plan.**

```
paper/PAPER_STATUS.md
paper/manuscript.pdf
paper_arxiv/COMPLETION_PLAN.md
docs/PAPER_MASTER_PLAN.md
docs/PAPER1_FRAMING_AUTHORITATIVE.md      <- name claims authority; superseded
docs/PAPER_POSITIONING_EXACT_CONTROL.md
docs/PAPER_REWRITE_BRIEF.md
docs/PAPER_REFRAME_CONTROL_SUBSTRATE.md
docs/DEVELOPMENT_PLAN.md
docs/EXPERIMENT_INFRASTRUCTURE_PLAN.md
docs/BASE_ELEMENT_EXPANSION_PLAN.md
docs/EDITING_V2_CYCLE_OPEN_MIGRATION_PLAN.md
```

They are left on disk unmodified — deleting or rewriting them was not
authorized. If a stale plan being merely *present* is a risk, stamping a
supersession banner on each is a one-line change per file, on request.

## What the paper is

> COMPOSE learns a source-agnostic stochastic reference kernel over executable
> molecular rewrites, and specializes that same kernel using finite-horizon
> bridge control for constrained, dynamic molecular editing.

Working titles:

- *COMPOSE: Learning and Controlling Executable Molecular Rewrite Processes*
- *COMPOSE: Bridge-Controlled Molecular Editing on an Executable Rewrite Graph*

**Do not put "Generator Matching" in the title** unless a real time/path
construction and trained hazard are added. The current `R_theta` is a legitimate
learned stochastic jump process, but naming it Generator Matching reopens an
avoidable terminology dispute.

## What the paper is NOT

It must not try to be simultaneously a GuacaMol SOTA unconditional generator, a
new Generator Matching estimator, a molecular editor, a Schrödinger bridge
paper, and a parallel-sampling paper. That becomes diffuse and impossible to
prove rigorously.

The focused submission is four things:

1. A learned reference process on the graph of complete valid molecules.
2. Exact bridge control of that process on enumerable spaces.
3. Scalable approximate bridge control on full molecules.
4. Clear advantages for source-conditioned, path-constrained, dynamically
   retargetable molecular optimization.

---

### Dataset provenance — say this, not "we trained on GuacaMol"

> The editing corpus was derived from a fixed **500,000-molecule GuacaMol source
> pool** by compiling executable local transitions, real-endpoint analogue
> relations, and reversible synthetic walks. COMPOSE was **not** trained as an
> unconditional GuacaMol distribution model: the learned object is
> `R_theta(y|x)`, a transition law over source–successor edits, not `p_theta(x)`.

Source asset `guacamol-subset-500000-seed0`, sha
`70526d92f1f08d8e292cb31218f81b6924a2182f772c43348015110669d47791`. Full record
and the open links in `diagnostics/editing_v2_corpus_provenance_binding.json`.

**Disclose, do not hide, the shared provenance with PMO's GuacaMol MPO
objectives.** The defence is measured, not asserted: against a training universe
of 96,094 sources, the matched reserve and the external 16-shard cohort both have
**zero exact molecule overlap** and **zero near-duplicates** at ECFP4 Tanimoto
≥ 0.95. `final_test` is sealed and **not yet audited** — audit it before opening
that role.

GuacaMol is harder than ZINC250k for *distribution matching*, which is not the
object here; for a broad executable-edit reference kernel the extra chemical
diversity is an asset. **Do not retrain on ZINC because ZINC is easier.** A
matched-scale `R_theta^{GuacaMol}` vs `R_theta^{ZINC}` ablation is worth having
only if reviewers make dataset dependence central — it is not on the critical
path. Making at least one **ZINC-derived source-conditioned panel** a major
Experiment 5 Track B task is the cleaner answer, since it tests cross-dataset
transfer directly.

## The four claims

Every experiment must map to one of these.

| # | Claim | Evidence required |
|---|---|---|
| 1 | **Learning matters** | `R_theta` > uniform legal rewriting, and > simple empirical transition laws. If the learned reference does not beat strong nonlearned proposals, the controller runs on expensive infrastructure the network did not improve. |
| 2 | **The reference process produces healthy executable trajectories** | Every committed state a molecule; uses the operator basis; no cycles or degenerate families; genuinely changes size and topology. |
| 3 | **Bridge control is genuinely future-aware** | On a small exact space the Doob controller reproduces the analytically tilted path law; learned control approximates it; greedy/immediate-reward guidance does not. *This is the mathematical centerpiece.* |
| 4 | **The architecture buys capabilities static optimizers lack** | Same frozen `R_theta` supports multiple objectives, mid-trajectory preference change, pathwise constraints, shared-prefix branching, flexible-size and topology-changing edits. *This is the empirical headline.* |

The unique advantage to argue:

> Other methods may produce excellent endpoints, but COMPOSE exposes a learned,
> executable path distribution that can be constrained, interrupted, branched
> and retargeted without retraining.

Validity is the substrate. **Reusable stochastic control is the contribution.**

---

### The three-layer story — and a de-emphasis Experiment 1 forces

> **legality → plausibility → purpose**
>
> The rewrite system defines *what is executable*. The learned successor model
> learns *which concrete molecular continuation is plausible*. Finite-horizon
> control learns *which plausible continuation is useful for the future
> objective*.
>
> executor → legality · `R_theta` → plausible molecular transport ·
> `h_phi` → future-aware task control

Experiment 1 established the middle box **developmentally** — a sampled
evaluation on the development checkpoint, not yet paper-established, though
scientifically there is no ambiguity left about whether learning matters.

**De-emphasize hierarchical operator scheduling as a source of novelty.** The
measured decomposition is `identity learning >> family scheduling`: learning the
family head buys 1.08 nats over uniform, learning within-family identity buys
2.15 and lands within 0.19 of the full model. Scheduling matters, but the
evidence says the learned chemical knowledge lives in choosing the specific
molecular successor within an edit family. That is a cleaner story anyway, and
it retrospectively explains the training behaviour: family probability could
move without representing capability loss, because the hard learning problem was
the conditional molecular decision inside a family.

**Do not run more C1 variants now.** No local-feature MLP, no full external
cohort, no 20k-source breakdown, no architecture ablations. Bank them as
possible later confirmatory work.

## Phase 0 — Finish and freeze `R_theta`

Epoch 3 is running. Let it finish. Use the **epoch-level** result, not adjacent
500-step points: measured effective resolution of the validation metric is
~0.02–0.03 nats.

**Stopping rule** (preregistered in `diagnostics/editing_v2_epoch3_preregistration.json`):

- freeze after epoch 3 if best eligible reference-law NLL improves **< ~0.05 nats** over epoch 2;
- consider epoch 4 only if epoch 3 gains clearly **more than 0.05–0.10 nats** AND every identity-capability gate is healthy;
- no automatic continuation beyond that.

**For the final paper: three independent fresh seeds**, under the same frozen
data, initialization protocol, maximum epoch budget and checkpoint-selection
rule. The present run is **development only**. The paper reports mean, standard
deviation and paired confidence intervals across the three selected checkpoints.

**Every checkpoint must bind:**

```
(code commit, library hash, split hash, sampling-law hash,
 manifest hash, store hash, initialization hash)
```

---

## Experiment 1 — Does `R_theta` learn a useful molecular transition law?

### 1A. Primary likelihood evaluation (frozen matched reserve)

- reference-law-weighted canonical-successor NLL *(primary)*
- panel-native NLL *(secondary, descriptive)*
- within-family identity NLL
- family-selection NLL
- mean probability assigned to the observed canonical successor
- calibration error / reliability plots
- by all 8 operator families
- by all 19 capability cells
- support bands 0 / 1–4 / 5–24 / 25+
- the original 16-shard cohort as an **external-cohort** result
- zero-mass family–lane strata **reported separately, never folded into selection**

Top-1 teacher agreement may appear in the appendix; it is not primary.

### 1B. Required internal baselines

Scientifically more important than a long external-generator list, because they
isolate what the model learned.

| Baseline | What it proves |
|---|---|
| Uniform over **distinct productive canonical successors** | Whether learning beats the executable kernel alone |
| Empirical family prior + uniform within family | Whether learning adds beyond family frequencies |
| Learned family head + uniform within family | Isolates family scheduling |
| Frozen empirical family law + learned within-family scorer | Isolates exact successor selection |
| Untrained same architecture | Improvement from optimization vs architectural bias |
| Local-feature MLP | Whether a full GNN is necessary |
| Flat raw-action head | Tests hierarchical factorization |
| No alias aggregation | Tests whether canonical successor aggregation matters |

**Local-feature MLP** uses the same legal support and objective but only simple
descriptors: atom element, degree, valence, charge, aromaticity, ring
membership; bond type and local endpoint features; molecule size, charge, cycle
rank; family and table identity. Strong and inexpensive — if it matches the GNN,
reviewers will reasonably conclude the graph encoder was unnecessary.

### 1C. What counts as a strong result

- \> 0.5 nat improvement over the empirical-family baseline overall
- positive improvement in ≥ 6 of 8 families
- no important capability cell abandoned
- consistent gains over the local-feature MLP
- matched-domain improvement without material external-cohort degradation

The earlier development model already showed ~1 nat over the empirical-family
law, so these targets are realistic.

---

## Experiment 2 — Multi-step behavior of the learned reference

500–1,000 held-out sources stratified by family support, scaffold-support band,
molecule size, and real vs synthetic provenance. ~8 stochastic trajectories per
source at horizons **8, 16, 32** edits.

Compare: learned `R_theta` · uniform canonical rewriting · empirical-family
proposal · optionally the local-feature baseline.

Report: internal grammar validity at every step; independent RDKit sanitization
at every step; single-component connectivity; stalls / empty-support states;
revisit-cycling rate; unique states per trajectory; operator-family usage and
entropy; change in atom count; change in cycle rank; source-to-state Tanimoto
over time; endpoint diversity; fraction of trajectories using all major
operation classes; wall-clock and neural evaluations per accepted edit.

The key figure is **not** "100% valid" — the kernel makes that largely
structural. The useful comparison is whether the learned law explores the valid
state graph *more plausibly and less degenerately* than uniform legal rewriting.

A compact free-running generation panel from a fixed, target-independent bank of
small valid seeds can establish this is a generative stochastic process. **Do not
claim GuacaMol distribution-learning SOTA** without the prior/reverse machinery.

---

## Experiment 3 — Exact finite-horizon control *(mathematical anchor)*

Production editing kernel on a bounded enumerable closure — ideally the
966-state closure or larger if inexpensive — with an explicit cemetery state.

30–50 goals covering: broad terminal sets; intermediate-probability sets; rare
but multistate sets; smooth Boltzmann goals; property boxes; conjunctions;
budgets 2, 4, 6, 8 edits.

| Arm | Purpose |
|---|---|
| Exact Doob transform | Gold standard |
| Greedy immediate reward | Myopic baseline |
| Local Boltzmann tilt | Soft myopic baseline |
| Hard feasibility masking | Constraints without future value |
| Learned `h_phi` | Amortized controller |
| Untwisted SMC | Particle-search baseline |
| Learned-`h_phi` twisted SMC | Approximate correction |
| Exact-`h` twisted SMC | Monte Carlo oracle / floor |

Metrics: terminal TV to the exact controlled law; per-row transition TV; KL to
the exact bridge; target probability; ESS; support violations; backward-equation
residual; unreachable-goal behavior; retargeting after a prefix; error
stratified by goal rarity and conjunction depth.

Must prove: (1) exact control agrees with the analytic tilted law to numerical
precision; (2) greedy and local Boltzmann can reach high reward while sampling
the *wrong distribution*; (3) learned `h_phi` closely approximates the exact
bridge on broad/moderate goals; (4) dynamic retargeting from a realized
intermediate state is mathematically coherent.

Strong main-text result: exact TV below numerical tolerance; learned-`h_phi`
terminal TV below ~0.1 on broad/interpolative goals; materially lower error than
greedy and local Boltzmann; zero support violations. Rare-conjunction failures
reported honestly in the appendix, not dominating the main claim.

---

## Experiment 4 — Build the scalable full-molecule controller

Largest remaining method component. **Do not jump from the enumerable controller
straight to large benchmark claims.**

Train a universal value model `h_phi(x, z, b)` — `x` current molecule, `z`
semantic objective specification, `b` remaining edit budget. **`R_theta` stays
frozen.**

**Goal language** — structured continuous goals, not named one-hot tasks:
property increase/decrease directions; target intervals or boxes; random
multiobjective weight vectors; similarity floors; protected scaffold flags;
charge/size/cycle constraints; edit budgets; moderate conjunctions. Begin with
broad, smooth goals — exact-space work showed rare conjunctions are the weakest
regime and they must not define the first scalable controller.

**Training data:** sample sources → roll out frozen `R_theta` → sample goals and
budgets → compute terminal rewards or soft desirabilities → train with terminal-
value regression, reachability classification, backward/Bellman consistency,
controlled-row KL where targets exist, optional Monte Carlo return regression.
For selected states, **enumerate the legal `R_theta` successor row and compute
the Bellman expectation directly** rather than using a loose one-sample target.

**Pre-benchmark controller gates:** held-out goal interpolation; held-out goal
composition; new source generalization; Bellman residual; ranking correlation
with Monte Carlo estimates; controller success vs greedy and hard mask; SMC ESS;
no support violation; stability across budgets. **Do not proceed to expensive
external sweeps until the controller clearly beats the future-blind internal
baselines.**

### 4A. Exact-target recovery — DONE, SEALED, AND CLOSED

The first goal language run was the narrowest one: `z` is a specific target
molecule and the objective is exact recovery within a declared edit budget.
Development panel 24 pairs, then a one-shot sealed panel committed at 571ec9d
before any `h_phi` work existed. Result:
`diagnostics/editing_v2_sealed67_result.json`; protocol
`diagnostics/editing_v2_sealed67_preregistration.json` and its pre-outcome
amendment.

**Primary — 65 endpoint-clean pairs:**

| arm | recovered | gain retention | rescue-set overlap | continuations |
|---|---|---|---|---|
| greedy | 26/65 (40%) | — | — | 0% |
| verified rollout, full candidate set | 40/65 (62%) | 100% | 14/14 | 100% |
| `h_phi` top-1 + verified rollout | 40/65 (62%) | 100% | 13/14 | 35% |
| similarity top-2 + verified rollout | 37/65 (57%) | 79% | 11/14 | 52% |
| similarity top-1 + verified rollout | 34/65 (52%) | 57% | 8/14 | 35% |
| `R_theta` top-1 + verified rollout | 33/65 (51%) | 50% | 7/14 | 35% |

Paired difference greedy → verified rollout **+21.5 pp [12.3, 33.5]**, exact,
rescuing 35.9% [21.2, 52.8] of greedy's 39 failures. Reported as an effect size,
not a McNemar p-value: the controller keeps greedy's action in every shortlist
and overrides only on strict improvement, so greedy-only wins are impossible by
construction and the one-sided discordance is a design property. That same
property is what makes the safety result meaningful — no arm lost a pair greedy
recovers, on any panel.

**What this licenses:**

> Myopic control leaves many known-reachable molecular targets unrecovered.
> Verified remaining-budget control substantially improves exact recovery, and
> goal-aware prioritization can preserve most or all of that gain while
> evaluating far fewer future continuations.

**What it does not license:**

- **Not** "`h_phi` beats similarity." Not preregistered, not established. Exact
  target recovery privileges Tanimoto-to-the-answer; a similarity win here would
  be a fact about the test bed. `h_phi` is a **secondary efficiency** result.
- **Not** "`h_phi` reproduces full rollout." It matches the **count**, not the
  **set** — it misses one of full's rescues and finds one full misses. The
  rollout controller re-plans after each committed action, so it is not globally
  optimal over the rewrite graph and a smaller shortlist can steer into a
  different successful basin. Magnitude and set are not interchangeable.
- **Not** a claim about the `h_phi` of this plan's goal language. This `h_phi`
  was trained for one target-molecule objective, not for property directions,
  boxes or conjunctions.

The separation the three-layer story predicts is now measured: `R_theta` top-1 is
the **weakest** prioritiser of the three (50% gain retention, 7/14 rescues).
Plausibility is not purpose.

**This branch is closed.** Do not tune `h_phi`, sweep more `K`, rank all ~500
successors, build a pairwise `h_phi`, or manufacture another target-recovery
panel. Development and sealed confirmation both exist; further percentage points
here are worth less than any of 4B–4D.

### 4B–4D — where the remaining Claim 4 value is

Exact target recovery is the one goal with a privileged non-learned heuristic. The
capabilities a static optimizer actually lacks are the ones where no such
heuristic exists:

1. **Dynamic retargeting** from a realized intermediate molecule (Experiment 6).
2. **Pathwise constraints** — a motif or scaffold valid throughout the
   trajectory, not merely at the endpoint (Experiment 7).
3. **A goal with no known target molecule**, where learned or estimated
   desirability has a real reason to exist.

---

## Experiment 5 — Static molecular optimization (two tracks)

### Track A — Oracle-efficiency benchmark (PMO)

PMO standardizes oracle accounting across 23 tasks; primary metric is **AUC of
top-10 performance vs oracle calls** under a max 10,000-query budget. It exists
because methods look strong when oracle budget is ignored.

Preregistered subset: **DRD2, GSK3β, JNK3**, plus three official GuacaMol
multi-property objectives spanning similarity and MPO behavior. QED is a smoke
test, not a headline task.

Budgets **100 / 250 / 500 / 1,000** for main efficiency curves; optionally the
official 10,000-query endpoint for direct PMO comparability.

Also report: best score; top-10 mean; diversity among top candidates; unique
valid oracle evaluations; wall-clock **separately** from oracle calls.

### Track B — Source-conditioned lead optimization *(carries more weight)*

30–50 fixed held-out sources per task, shared by every method. Tasks:
similarity-constrained single-property; two- and three-objective; an
atom-count-changing task; a ring/topology-changing task; scaffold-preserving
decoration. Edit budgets 4, 8, 12.

Primary metrics: hypervolume AUC vs oracle calls; success under all constraints;
property gain; endpoint Tanimoto; feasible nondominated count; endpoint
diversity; accepted edit count.

---

## External baselines

### Must-run

| Baseline | Why | Use for |
|---|---|---|
| **MARS** (ICLR 2021 Spotlight, public code) | Closest established iterative molecular editing baseline — adaptive GNN proposals inside annealed MCMC for multiobjective design | static multiobjective; oracle-efficiency curves; similarity-constrained editing |
| **DDSBM** (ICLR 2025) | Closest modern bridge comparison — CTMCs solving a discrete Schrödinger bridge for graph transformation, with minimal graph change | (1) reproduce its **native** protocol and add COMPOSE under the same data/evaluator; (2) adapt to the common source-conditioned panel **only if** the released code makes that faithful. Do not force it into dynamic/pathwise tasks it was not designed for. |
| **GraphXForm** | Iteratively modifies valid molecular graphs, can initialize from existing structures, supports structural constraints | source-conditioned static optimization; structural-constraint tasks; endpoint quality and oracle efficiency |
| **GraphGA** and **REINVENT** (official PMO impl.) | PMO found simple established methods frequently outperform newer ones under controlled oracle budgets | important sanity checks, not token old baselines |

### Optional

- **GrIDDD** — variable-size static optimization, if code and representation align.
- **A current PMO-leading method** (SEISMO, MolLIBRA) if code, model/API costs and oracle accounting reproduce fairly. SEISMO is a trajectory-aware LLM agent — a modern optimization comparator, **not** a process-model ablation.
- **Morph** — contextual related work, or a native 3D task only. Adapting a geometric 3D generator to a 2D rewrite setting is not automatically fair.
- **Edit Flows** — related work; its demonstrated domain is variable-length sequence editing, not this molecular graph process.

### Do NOT run

The old draft's unconditional-generator matrix — JTVAE, GDSS, DiGress, GruM,
DISCO, Cometh, DeFoG, ConStruct, CoCoGraph, GrIDDD — existed because the paper
was centered on GuacaMol distribution learning. **Do not spend months rerunning
it.** Use DeFoG/DiGress/Cometh only if a serious unconditional
distribution-learning section is retained; otherwise they are contextual related
work, not baselines that causally test the actual claims.

---

## Internal control baselines *(the most important ones)*

External algorithms tell reviewers whether COMPOSE is competitive. Internal
baselines tell them **why it works**. For every full-scale control task, compare
the **same frozen `R_theta`** under:

| Arm | Information used |
|---|---|
| Unguided `R_theta` | No objective |
| `R_theta` + hard mask | Feasibility only |
| Greedy one-step objective | Immediate reward |
| Local Boltzmann tilt | Soft immediate reward |
| Beam / best-first search | Explicit finite search |
| Untwisted SMC | Generic particle search |
| Immediate-potential SMC | Future-blind Feynman–Kac |
| Learned bridge controller | Future-aware value |
| Monte Carlo value / MPC | Expensive planning baseline, small subset |

The decisive causal comparison:

> Does future-aware bridge control outperform ordinary generation plus
> ranking/search under the same legal kernel, reference model, oracle budget and
> source molecules?

Without it, a reviewer can attribute gains to the rewrite kernel or the oracle
rather than the bridge.

---

## Experiment 6 — Dynamic preference switching *(signature experiment)*

Per source: optimize objective **A** for `k` edits → switch to conflicting
objective **B** → continue **from the realized intermediate molecule** for
another `k` edits. Objective pairs chosen **before** seeing results, preferably
empirically conflicting in the candidate space.

Arms: COMPOSE bridge replanning from the current state; greedy replanning from
the current state; static compromise objective (A+B) from the start; restart from
the original source after the switch; beam-search replanning; MARS or GraphXForm
restarted at the switch state where technically possible.

Metrics: final **B** reward; retained **A** reward; post-switch regret; oracle
calls after the switch; additional edits to reach a target; path constraint
satisfaction; diversity across repeated retargets; fraction of original prefix
reused.

A strong result is **not** merely a higher endpoint score. It is comparable or
better final quality with **substantially lower post-switch oracle cost and no
restart**.

---

## Experiment 7 — Pathwise constraints

Constraints that must hold **at every intermediate state**: protected Murcko
scaffold; minimum source similarity; forbidden substructures; atom-count
interval; formal-charge interval; maximum cycle rank; allowed-element set.

Arms: bridge controller with pathwise masks; `R_theta` with masks but no value;
greedy with masks; endpoint-only reranking; endpoint-only filtering; beam search
under endpoint constraints.

External methods get the same **endpoint** constraints, but all-step metrics are
marked **N/A** when they do not expose comparable molecular trajectories.

Metrics: all-step constraint satisfaction; final success; oracle calls wasted on
infeasible endpoints; acceptance rate; final property/hypervolume; endpoint
diversity; average edit length.

Expected: 100% pathwise feasibility from the constrained COMPOSE support; better
endpoint quality than hard masking alone; much less wasted oracle budget than
endpoint-only filtering.

---

## Experiment 8 — Pareto fan / shared-prefix branching

Visually strong but less essential than dynamic switching; can be combined with
the switching figure rather than forming its own section.

From one source: neutral/broad controlled prefix for 2–3 edits → branch into
8–16 preference vectors → continue each under its own controller. Compare against
fully independent runs from the source.

Report: Pareto hypervolume; nondominated count; endpoint diversity; shared
computation/oracle savings; prefix reuse; path validity.

---

## Fair comparison rules *(freeze before final evaluation)*

**Same problem definition.** All methods receive identical source molecules,
oracle implementation, allowed elements and size bounds where configurable,
similarity/scaffold constraints, oracle-call budget, timeout and failure policy.

**Count the right costs.** Report separately: oracle calls; neural function
evaluations; accepted molecular edits; wall-clock; GPU/CPU hours. Do not trade
oracle calls for undisclosed hidden computation.

**Treat invalid output honestly.** Invalid or unscorable molecules count as
failed oracle proposals. Deduplicate before oracle evaluation. Cache repeated
oracle calls across all methods. A method that cannot support a pathwise or
dynamic metric receives **N/A**, not an invented adaptation.

**Two comparison tracks.** (1) *Native reproduction* — run DDSBM, MARS,
GraphXForm on the protocols they were designed for and add COMPOSE. (2) *Matched
adaptation* — common sources/oracles/budgets only where adaptation is faithful.
**Only matched results support direct superiority claims.**

---

## Statistical design

- 3 independently trained `R_theta` seeds
- same source/task panels for all methods
- 3–5 controller/sampling seeds per source where stochastic
- paired bootstrap confidence intervals over sources
- mean and standard deviation across model seeds
- primary metric declared **before** running the final test
- final test evaluated **once**, after all choices are frozen

For optimization tasks the **source molecule** — not the individual generated
endpoint — is the primary independent unit for paired statistics.

---

## Main-paper structure

- **Figure 1 — Method.** `valid molecule → legal rewrite graph → R_theta → h_z → controlled trajectory`. **Do not show the obsolete MH-forward/reverse-time construction.**
- **Figure 2 — Learned reference process.** Reference-law NLL vs internal baselines; family decomposition; scaffold-support generalization; rollout validity / operator utilization.
- **Figure 3 — Exact and learned control.** Enumerable state graph; exact terminal tilt; TV/ESS curves; greedy/Boltzmann vs learned `h_phi`; retargeting.
- **Figure 4 — Full molecular control.** Static Pareto or oracle AUC; dynamic objective switch; pathwise scaffold constraint; representative trajectories.
- **Table 1** — Static optimization and oracle efficiency vs MARS, DDSBM, GraphXForm, GraphGA, REINVENT.
- **Table 2** — Reference-model baselines and ablations.

Everything else — full PMO suite, all 19 cells, all trajectories, zero-mass
stress strata, detailed compute — goes to the appendix.

## What makes this a strong submission

The paper does **not** need to win every PMO task. It needs this pattern:

1. the learned reference clearly beats legal/random/empirical proposals;
2. exact bridge control is mathematically and numerically correct;
3. learned control beats myopic guidance and ordinary search;
4. static optimization is competitive — ideally top-two or close to the strongest baselines;
5. COMPOSE is decisively better on dynamic retargeting and pathwise constraints;
6. all comparisons are oracle-budget matched and reproducible.

---

## Go / no-go gates

**After epoch 3.** Freeze `R_theta` unless the gain clearly exceeds the
predeclared threshold and capabilities remain healthy.

**Before external baseline sweeps.** The full-molecule bridge must beat greedy,
hard mask, local Boltzmann and beam search on a **10-source development panel**.
If it cannot, do not spend on large benchmarks yet.

**Before final test.** Three trained seeds; all main baselines runnable; fixed
task list and budgets; controller frozen; no use of final-test outcomes in
tuning.

**Reopen goal-marginal training only if** the frozen ordinary `R_theta`
systematically assigns too little support to modes held-out controllers need —
evidenced by low bridge ESS, high control KL, repeated particle collapse, or
failure the controller cannot fix. It is **no longer a critical-path repair**.

---

## Immediate execution order

**While epoch 3 runs:**

1. Replace the manuscript's old claim/RQ list with the four-claim hierarchy.
2. Build the reference-model baseline evaluator.
3. Freeze the exact-control confirmatory goal suite.
4. Implement the full-molecule `h_phi(x, z, b)` training pipeline.
5. Wrap greedy, hard-mask, Boltzmann, beam and SMC baselines behind one interface.
6. Verify MARS, DDSBM, GraphXForm, GraphGA and REINVENT environments *without* running full experiments.

**After epoch 3:**

1. Freeze the best eligible `R_theta`.
2. Train the remaining two model seeds.
3. Run Experiment 1 and the rollout study.
4. Run the exact-control confirmatory suite.
5. Pilot the scalable controller on 10 development sources.
6. Freeze the task suite and oracle budgets.
7. Run static optimization.
8. Run dynamic switching and pathwise constraints.
9. Run the final test exactly once.
10. Rewrite the results section around what the experiments actually establish.

> **Do not fill in the TBDs in the old paper.** Replace the old experiment
> architecture with one built around the actual learned reference kernel and
> bridge-controlled editing system now in hand.

---

## Current state against this plan

Development run `run_v2_01`, identity chain: reserve `b580fdef6486` / law
`b0cc66f168f1` / manifest `e27494a46250` / store `b232a6fa069f`, freeze gate
FROZEN 9/9.

- **Phase 0** — done. `R_theta` frozen under the preregistered epoch-level rule.
- **Experiment 1** — done developmentally. Learning matters: **1.46 nats** over
  uniform legal rewriting on the frozen matched reserve. Decomposition is
  identity ≫ family scheduling, which is why this plan de-emphasizes hierarchical
  scheduling as a novelty source.
- **Experiment 4A, exact-target recovery** — **done, sealed, and closed** (§4A).
  Greedy 40% → verified rollout 62% on 65 endpoint-clean held-out pairs;
  learned top-1 prioritization matches that recovery count at ~35% of the
  continuation evaluations. Claim 4 is no longer speculative for this goal.
- **Experiments 2, 3, 5–8** — not started. §4B–4D names the three that now carry
  the most Claim 4 value.

Superseded en route, recorded so they are not re-run: the C0 planning-signal
probe on DRD2 was **negative** (`diagnostics/` C0 artifacts) and the
corpus-trace reconstruction of multi-step histories was **refuted**; the
evaluation panel was rebuilt fresh from held-out molecules instead. A direct
`h_phi` **override** controller was also negative — the working controller uses
`h_phi` only to propose, with a verified rollout making every commit.

### Known gaps against this plan's requirements

- **Checkpoints do not bind the code commit.** `RunIdentity` currently carries initialization seed, initial model state, library, split, sampling law, manifest, packed store and eval panel hashes — seven of the eight required bindings. The code commit is missing and must be added before any paper-bearing run.
- **Three fresh seeds are required for the paper**; the present run is development only. This is a budget item, not just a scheduling one — three seeds at ~3 epochs each is several times the current cap.
