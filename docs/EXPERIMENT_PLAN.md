# COMPOSE — Canonical Experiment Plan

**STATUS: CANONICAL. This supersedes every prior experimental plan, including the
manuscript's existing experimental section and the A/B/C framing used earlier in
development. Do not execute or cite the old plan.** The old draft describes a
materially different model: a Metropolis-corrected forward CTMC toward a
prescribed high-entropy prior with a time-dependent learned reverse scheduler,
learned event clock and trajectory likelihood, centered on unconditional
GuacaMol-style distribution learning. That is not the system we have.

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

- **Phase 0** — epoch 3 in flight (8,502 → 12,753). Epoch-2 selected step 8,500 at reference-law NLL 2.9238, from an initialization value of 5.4603. Within-family identity improved in all 8 families. Stopping rule preregistered.
- Everything else in this plan is **not started**.

### Known gaps against this plan's requirements

- **Checkpoints do not bind the code commit.** `RunIdentity` currently carries initialization seed, initial model state, library, split, sampling law, manifest, packed store and eval panel hashes — seven of the eight required bindings. The code commit is missing and must be added before any paper-bearing run.
- **Three fresh seeds are required for the paper**; the present run is development only. This is a budget item, not just a scheduling one — three seeds at ~3 epochs each is several times the current cap.
