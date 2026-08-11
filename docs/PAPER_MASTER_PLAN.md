> **ARCHIVED — DO NOT USE.**
> Superseded by [`docs/EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md), which is the only
> current plan. This document describes an earlier framing of the project and its
> experiment list must not be executed or cited. Kept on disk because other files
> still link to it; read it as history, not as instruction.

# Paper master plan — source-conditioned molecular optimization via valid-path generator matching

> ## ⚠️ SUPERSEDED THESIS — 2026-07-28
> **The framing/positioning in this document is SUPERSEDED by
> [`PAPER1_FRAMING_AUTHORITATIVE.md`](PAPER1_FRAMING_AUTHORITATIVE.md)** (RGM-first, trans-dimensional).
> Its thesis (*source-conditioned molecular optimization via valid-path generator matching*) put lead optimization at the centre; lead optimization is now the FLAGSHIP APPLICATION, not the definition. Its de-novo demotion, figure order, and A/B/C source-conditioned ablation are also superseded.
>
> **STILL VALID and deliberately preserved here — do not delete, this is the granular record:**
> the baseline set and head-to-head rationale · the critical ablations (§11) · the metric hierarchy (§12) · the Doob/Pareto goal-conditioned formulation (§8) · the four-level claim hierarchy (§6) · MOG-DFM orthogonality (§5) · the reviewer-facing argument for why the carbon-tree model is not the main control model (§2).
>
> When this document and the authoritative framing conflict on thesis, positioning, title, contribution
> order, or novelty claims, **the authoritative framing wins**. Operational detail below remains usable.



> **AUTHORITATIVE. Read this before any positioning, training-recipe, experiment, baseline, or
> terminology decision.** It supersedes/extends `PAPER_REFRAME_CONTROL_SUBSTRATE.md` (same thesis; this
> is the detailed execution blueprint) and the baseline-map addendum therein.
>
> - **Received:** 2026-07-26 (advisor recommendation), preserved faithfully. Math notation was
>   line-broken in transit and has been normalized to canonical inline form; prose is preserved.
> - **One-line thesis:** *A general stochastic-control framework for source-conditioned molecular
>   optimization, demonstrated through multi-objective lead optimization.*

---

## 0. Architecture decision — LOCKED (2026-07-26)

**Universal editing-trained edit prior + explicit source-conditioned Doob controller.** Do NOT choose
"universal" vs "source-conditioned" at every layer — the base is universal, the control is source-aware.

- **Base generator = universal, source-AGNOSTIC valid-edit prior `Q_θ(y | x, t)`** — acts only on the
  current molecule; the immutable source `x_src` is NOT fed into the base at every step. Trained on
  real-molecule→real-molecule EDITING trajectories (MMP / analogue-series / shared-scaffold / reversible
  perturbations / longer compositions) — NOT the carbon-tree de-novo prior, NOT a fully source-conditioned
  translator. Call it "universal source-initialized edit prior" / "source-agnostic edit prior with
  source-conditioned control" — never "unconditional generator." *(Our existing B-edit model already IS
  this: `Q_θ(y|x,t)`, no `x_src` input — so the ARCHITECTURE is right; only the TRAINING DATA broadens.)*
- **Source-conditioning lives in the CONTROLLER** `h_φ(t, x; x_src, z, b, m)` — source lead `x_src`,
  preference/target-region `z`, remaining edit budget `b`, protected scaffold/pharmacophore mask `m`.
  Controlled generator: `Q^{z,x_src}_t(x,y) = Q_θ,t(x,y) · h_φ(t,y; x_src,z,b−1,m) / h_φ(t,x; x_src,z,b,m)`.
  Trajectories are INITIALIZED at the lead (`X_0 = x_src`); source-relative objectives, protected
  substructures, and budgets enter through the desirability
  `g(x; x_src, z) = exp[β·U_z(x) − λ·d(x,x_src)]` (soft) or `1{f(x)∈B_z, sim(x,x_src)≥τ}` (hard) — NOT the
  reference dynamics.
- **Augmented state where memory is needed:** `X_t = (X_t, x_src, b_t, m)`. The universal prior acts on
  `X_t`; the controller uses the augmented info (source-sim, budget, protected atoms). Decomposition:
  `Q^ctrl_t(x,y | x_src,z) = Q^edit_θ,t(x,y) · C_φ,t(x,y | x_src,z)`, control multiplier = the Doob ratio.
- **Why universal base:** reusable molecular dynamics (not a translator); cleaner Doob (one universal
  process initialized at different states, source = boundary condition); ideal dynamic steering (swap z,
  prior unchanged); modularity (train once, many controllers); experimental isolation (hold `Q_θ` fixed,
  vary controller). Explicit source in the base entangles plausibility with campaign conservatism, weakens
  steering (source stays an attractor), complicates exactness, risks shortcut/copy learning, and narrows
  the identity toward supervised translation.
- **Mandatory A/B/C ablation:** A = universal `Q_θ(y|x,t)`; B = source-conditioned base `Q_θ(y|x,t,x_src)`;
  C = universal + source-conditioned controller. Expect C strongest (A drifts too far; B conservative).
  Evaluate HV, source-sim, scaffold retention, property displacement, diversity, preference-switch
  adaptation, unseen-scaffold generalization, edit-budget sensitivity.

---

## 0b. Phase-A data recipe + A2 edit-path compiler — LOCKED (2026-07-26)

**Corpus: GuacaMol (broad) + recovered ChEMBL metadata for analogue mining. NO ZINC switch now.** GuacaMol
is standardized ChEMBL; recover the ChEMBL document/assay/target context the GuacaMol SMILES discard. Run
**full-scale MMP mining on the 500k (ideally the complete GuacaMol split)** — the 1,500-mol result meant
"random minibatch too sparse for pair discovery," NOT "GuacaMol too diverse" (MMP pairs scale
**super-linearly** once cores recur: `N_MMP = Σ_c C(n_c,2)`; ref: ChEMBL 800,714 drug-like → **2.63M MMPs**
via mmpdb @ max-variable-ratio 0.33). ZINC = later, external purchasable-space generalization test only.

**Three-layer training mixture (proportions are an ABLATION, not fixed):**
- **~50% synthetic corruption** (existing GuacaMol corruption — operator coverage, executable, controllable
  lengths, full-support; stops the prior being limited to observed transformations).
- **~35% global MMP traces** from full-scale GuacaMol/ChEMBL. Filters: one-cut first, max variable ratio
  ≤0.33, large constant core, no unsupported elements / stereo changes, bounded compiled length. Train
  **both directions A→B and B→A** (no database-ordering bias).
- **~15% longer scaffold-series paths** (compositional curriculum).
- **Layer-3 ChEMBL-series-aware subset:** map standardized GuacaMol → ChEMBL records; congeneric pairs
  sharing document / target (assay only when using measured property diffs) + common core/MMP.

**Murcko-scaffold pairs = a SPARSE analogue graph, NOT all-pairs cliques** (1,363 from 38 groups is
misleading — big groups dominate via `C(n,2)`). Per group: connect each molecule to its k nearest analogue
neighbours (operator-distance / MCS-diff); cap examples per source / scaffold / transformation; stratify
short/medium/long; sample **groups** ~uniformly (not raw pairs). Murcko = longer-range curriculum, not bulk.

**A2 edit-path compiler — staged; for MMP use mmpdb's constant fragment + labelled attachment mapping, NOT
an unconstrained MCS** (MCS picks awkward correspondences under symmetry / aromatic / repeated heterocycles):
- **A2.1 one-cut MMP compiler (build FIRST):** one attachment, no ring opened across the variable region,
  small variable fragment, bounded atom-count diff. Preserve the constant core → **delete** the source
  variable fragment leaf-to-root → retype / bond-order as needed → **insert** the target fragment
  attachment-outward → **replay through the REAL executor** → check graph isomorphism vs target. Both
  directions.
- **A2.2 linker + ring-system replacements:** two-cut linker exchange, heteroaryl substitution, fused/pendant
  ring replacement via **ring macros** — do NOT force a heteroaryl swap through long atom-by-atom
  destruction/reconstruction when a ring-replace macro exists (recreates the identity-bias failure).
- **A2.3 general scaffold-pair compiler:** only after MMP is reliable; MCS + bounded search; require
  ring-to-ring matching, complete-ring preservation, deterministic tie-breaking, max compiled length. Defer
  distant/ugly pairs.

**Compiler verification (STRONGER than endpoint replay), per pair:** every-intermediate validity +
connectedness; exact endpoint isomorphism; forward AND inverse replay; sequence length; repeated /
immediately-reversed edits; max source-similarity departure; protected core unchanged; operator-family
usage; shorter-compilation existence. **Round-trip `A →π B →π⁻¹ A`.** REJECT a path that reaches the right
canonical SMILES but breaks the attachment mapping or needs gratuitous cycles.

**"De-risked" scoping:** close-pair compilation success de-risks only the **trace compiler**. Phase A is
de-risked only after a small model trained on those traces shows low identity collapse, low edit cycling,
held-out-scaffold generalization, reasonable operator coverage, useful local exploration from unseen sources.

**Data splitting (mandatory):** source/scaffold-held-out split (unseen series) + transformation-held-out
split (compose beyond memorized templates) + existing GuacaMol val/test exclusions. With ChEMBL metadata,
split whole documents / analogue series together (never same-publication compounds in train + eval).

**GuacaMol is the PRIMARY corpus** (the official training split IS the main-model corpus). ChEMBL-metadata
recovery is LATER/optional and must NOT block the compiler or initial training. **Curriculum bins are set
FROM MEASURED statistics** (compiled path length, ring complexity, #attachment sites, operator families) —
never preset easy/medium/hard thresholds. **Never dump unrestricted GuacaMol pairs into training** —
compiler success + path complexity gate what enters each stage.

**Execution order (LOCKED):** (1) finish + validate the A2 compiler → **its diagnostic report is the next
deliverable** (success by pair type, path-length distribution, failure categories, operator coverage,
inverse-round-trip rate, success/failure examples); (2) A1 mining at GuacaMol scale (500k → full split);
(3) build the filtered curriculum from the measured stats; (4) train the first universal edit prior;
(5) verify it edits UNSEEN sources without identity collapse / excessive cycling; (6) same-base controller
comparisons; (7) exact-Doob correctness on enumerable spaces; (8) dynamic steering + initial Pareto;
(9) external baselines LAST.

---

## 1. Positioning — four layers of ONE paper (do not treat as competing labels)

| Layer | Language |
|---|---|
| Scientific task | Source-conditioned multi-objective molecular optimization |
| Flagship application | Similarity-constrained lead optimization |
| Generative mechanism | Valid-path molecular editing with generator matching |
| Control capability | Exact, dynamic, multi-objective property steering |
| Secondary demonstration | De novo generation from the carbon-tree prior |

Paper-level: **"We introduce a valid-path generative framework for source-conditioned, multi-objective
molecular optimization, and demonstrate it primarily in lead optimization."**

- **Do NOT** position as "multi-objective guided de-novo generation" — it obscures the distinctive,
  drifts toward MOG-DFM, and forces pathwise control near the carbon-tree source.
- "Property-guided generation" = too broad. "Molecular editing" = how it moves. "Lead optimization" =
  the application. "Controllable multi-objective molecular optimization" = the field category.

## 2. Why the carbon-tree (de-novo) model is NOT the main control model

Carbon-tree early states are graph-valid but not necessarily drug-like, within property-predictor
training domains, pharmacologically meaningful, close to an analogue series, or states a med-chemist
would revise the objective at. That doesn't make the de-novo generator bad — it's just not the clean
setting to show why every intermediate being valid matters.

Source-conditioned **editing** starts from an actual molecule, so every intermediate is an interpretable
analogue of the source; properties are evaluable throughout (no severe early-time distribution shift);
scaffold/pharmacophore/similarity/charge/alert constraints can be imposed at every step; preferences can
change after seeing an intermediate; a trajectory can branch into several directions; the process
resembles a real lead-optimization campaign. **That is where the state-space construction becomes
scientifically consequential.**

## 3. Train a source-conditioned editing REFERENCE PROCESS (same architecture + operator basis)

Change only the **source–target coupling**. Practical training mixture:

1. **Data-backed analogue pairs.** Build a molecular graph over a large corpus using matched molecular
   pairs (MMP), shared Bemis–Murcko scaffolds, high max-common-subgraph similarity, or known chemical
   series. Sample source→target pairs at several graph distances (1-step MMP = local med-chem
   transformations; longer paths = compositional editing).
2. **Synthetic legal perturbation pairs.** Take a real molecule, apply a short sequence of reversible
   legal operators, train **both directions**. **Do NOT make the whole problem "corrupt a molecule and
   reconstruct the identical molecule"** — that risks a strong **identity bias**. Mix reconstruction-style
   corruption with **distinct** source→target analogue pairs.
3. **Edit-distance curriculum.** 1–2 edits (precise local), 3–6 (ordinary lead-opt), longer (scaffold
   modification / large shifts). **Condition on an edit budget** → exposes the structural-preservation vs
   property-improvement trade-off directly.
4. **Source + protected-subgraph conditioning.** Model receives: the source molecule, a protected
   scaffold/pharmacophore mask, an allowed edit budget, optionally a source-type token. **Base model stays
   property-agnostic** — the controller supplies objectives at inference (plug-and-play).
5. **Full-support legal component.** `Q_ref = (1−ε)·Q_θ + ε·Q_legal`, where `Q_legal` has support over all
   currently-applicable legal operators. Learned = chemical plausibility; small legal component =
   reachability + prevents the controller being trapped by an imperfect base. **Ablate ε.**

You do **not** need one universal checkpoint. A de-novo checkpoint and an editing checkpoint in the same
framework is coherent: *the framework is general; the source process is chosen for the design regime.*

## 4. The strongest conceptual position

**Not** "our intermediates are valid" — MARS (MCMC fragment-graph editing), SMER-Opt
(feasibility-filtered local edit planning), and MolWorld (reachability over valid local transformations)
already edit valid molecules; GenMol does de-novo/fragment/hit/lead-opt but via **masked** representations,
not a process where every intermediate generative state is a complete molecule.

**Real intersection:** *a learned generator-matched stochastic process over an operator-closed molecular
graph state space, combining the distributional modeling of generative flows with the intervention
semantics of valid molecular editing.*

- vs masked/latent generators → every state directly evaluable + editable.
- vs heuristic graph-edit search → a learned reference path distribution.
- vs local property scoring → a future-aware Doob value.
- vs a fixed conditioned model → change the objective during sampling.
- vs endpoint-only constraints → control the path support itself.

**TERMINOLOGY (reviewer-proofing):**
- Avoid **"valid manifold"** (molecular graphs are a discrete combinatorial space, not a smooth manifold).
  Use **"valid molecular graph state space" / "operator-closed molecular state space" / "valid-path
  molecular space" / "valid molecular graph complex."**
- Avoid **"synthesizable"** unless operators correspond to reactions/MMPs. Use **"graph-valid" /
  "chemically admissible."**

## 5. Relationship to MOG-DFM (orthogonal, not a competitor)

MOG-DFM addresses **how to choose among candidate transitions** under competing objectives (rank-directional
scoring + adaptive hypercone filter over a pretrained discrete-FM base; demonstrated on peptide/enhancer-DNA).
Our work **defines the chemistry-native transition system** (every candidate transition and intermediate is
a valid molecule) and supplies **exact path-space control**. The methods are orthogonal — implement a
MOG-DFM-style controller **over our exact same operator candidates + base rates** and put it in the
controller table; MOG-DFM pairs with the framework rather than competing with it.

## 6. What validity truly gives — the logical chain + the four-level claim hierarchy

Chain: operators preserve state-space invariants → every reachable state is a complete molecule → rewards,
uncertainty, constraints are defined at every state → control can be applied *during* the trajectory → a
Doob transform reweights *existing* rates, preserving legal support → exact terminal conditioning (with
exact h) stays inside the valid state space → **better Pareto HV/oracle-efficiency is an EMPIRICAL result.**

Repeated wording: **"Validity enables control; the Doob transform makes that control distributionally
principled; the experiments show the resulting control improves optimization."** Do **not** claim validity
*guarantees* better hypervolume.

| Claim | Status |
|---|---|
| Every committed state is connected + valence-valid | by construction / proof |
| Doob steering preserves legal transition support | theoretical consequence |
| Terminal tilted distribution is exact | only with exact h + exact simulation |
| Better Pareto HV / oracle efficiency | empirical (via ablations) |
| Synthetic accessibility | **NOT** implied by graph validity |

## 7. Mathematical centerpiece — exact Doob control

Reference generator `Q_t(x,y)` on valid molecular graphs; positive terminal desirability `g(x)` (soft:
`g(x)=exp{β·u(x)}`; conditioning: `g(x)=1{f(x)∈A}`). Define `h_t(x)=E[g(X_T) | X_t=x]`. Controlled
off-diagonal rates:

```
Q^g_t(x,y) = Q_t(x,y) · h_t(y)/h_t(x),   x≠y,   diagonal re-summed so each row sums to 0.
```

- **Support preservation:** illegal edit ⇒ `Q_t(x,y)=0` ⇒ `Q^g_t(x,y)=0`. Guidance cannot leave the valid
  operator graph.
- **Exact terminal reweighting:** with exact h, `p^g_T(x) ∝ p_T(x)·g(x)`; indicator g ⇒ exact conditioning
  on the target event (if reachable).
- **Dynamic continuation:** at time τ replace g→g′, continue from the current molecule via `h^{g′}` — exact
  controlled continuation (with exact h′). NOT generally identical to using g′ from t=0 → **comparison vs
  restarting is scientifically interesting.**
- Terminal conditioning = ordinary Doob. **Cumulative/path-dependent** rewards need a Feynman–Kac potential,
  a killed/restricted generator, or an augmented state recording the path statistic. (Discrete FK correctors
  + SMC test-time alignment = comparison points.)
- **Honesty:** on real spaces a learned `h_φ` is approximate. **Demonstrate exactness on enumerable spaces,
  then demonstrate calibration + control quality at scale.**

## 8. Pareto formulation — goal-conditioned Doob (not just linear scalarization)

Linear scalarization `u_w(x)=wᵀf(x)` undersamples non-convex regions and slides to extremes. Learn a
**goal-conditioned** `h_φ(t,x,z)` where z = preference vector / objective-space target point / box /
lower-upper constraints / Chebyshev reference direction. For a target region `B_z`:
`g_z(x)=1{f(x)∈B_z}` ⇒ `h_t(x,z)=Pr(f(X_T)∈B_z | X_t=x)`. **Pareto-exploration algorithm:** maintain a
nondominated archive → identify under-covered / high-HV-contribution regions → select target region z →
Doob-steer/branch toward z → update archive → repeat. (A direct HV story, not "we picked good weights.")

## 9. Experimental program (8)

1. **Operator closure, reachability, medicinal-edit coverage.** Closure under validity; inverse existence +
   correctness; connectivity/irreducibility within a bounded chemical class; MMP-transformation coverage;
   held-out source→target analogue reconstruction; path efficiency vs shortest/known paths. Metrics:
   all-state trajectory validity, connectedness/sanitization, MMP coverage, target reconstruction, median
   operator distance, path-length ratio, inverse consistency, operator-family utilization, backtracking/cycle
   rate, runtime/edit. Ablations: primitives-only; no graft; no ring-system ops; no delete/inverse; learned
   rates without the full-support legal mixture; unconstrained-propose-then-repair.
2. **Exact Doob benchmark (most mathematically decisive).** *Setting A:* fully-enumerable state space (small
   vocab, ≤5–7 heavy atoms, bounded charge/bonds/horizon) → compute the reference law + exact `h_t` (backward
   equation / matrix exponentiation); test conditioning on a rare substructure, a property interval, two
   simultaneous constraints, a soft Boltzmann tilt, an objective-space Pareto cell. *Setting B:* real
   drug-like seeds with restricted horizon/library so local valid neighborhoods enumerate. Compare: exact
   Doob / learned `h_φ` / one-step greedy Δf / local Boltzmann Δf / classifier-value guidance / endpoint
   reranking / MOG-style / SMC-FK. Metrics: TV to exact endpoint law, KL/JS, target-event-prob error, h
   calibration, backward-equation residual, partition-function error, ESS, path-occupancy error, support
   violations. **Headline:** exact Doob matches the analytical target to MC error while local guidance
   attains high reward but the WRONG distribution.
3. **True multi-objective similarity-constrained optimization (main application).** Reproduce the GenMol
   setup for comparability (docking with QED≥0.6, SA≤4, Morgan-sim 0.4/0.6 over PARP1/FA7/5HT1B/BRAF/JAK2;
   vs GraphGA, RetMol) — **but not as the only result** (it collapses properties to hard constraints +
   best-docking). Add a **vector-valued** version retaining the objective vector (docking, QED, SA/complexity,
   logS, hERG/safety, source-sim/scaffold-preservation): 2-obj (exact/visual), 3-obj (front plots), 4–5
   (stress). Include a **selectivity** problem (↑target affinity, ↓off-target, keep QED/solubility, preserve
   scaffold). MARS multi-property tasks = another comparison. Metrics: normalized HV, HV-AUC vs oracle calls,
   IGD+, additive-ε, expected utility over held-out preferences, worst-preference regret, #feasible
   nondominated, coverage/spacing + molecular (success, source-sim, scaffold/pharmacophore/MCS retention,
   internal diversity, unique scaffolds, edit cost, path length, all-hard-constraint fraction). **HV hygiene:**
   fix objective bounds before evaluation, publish the reference point, cap archive size per method, count
   ALL oracle-evaluated intermediates+proposals, report HV vs oracle calls (not just final), pair HV with
   IGD+/coverage.
4. **Dynamic steering (main-paper figure).** *Preference switch:* potency+QED → solubility+safety →
   structural preservation, switching at 25/50/75%. Compare vs static-compromise / continue-unchanged /
   restart-from-lead / restart-from-current-with-conventional-optimizer / greedy / MOG / endpoint-rerank.
   Report steps-to-new-region, adaptation regret, retention of prior gains, final HV, target success, path
   violations, compute-vs-restart. *Pareto fan:* one lead → 8–16 preferences → shared prefix + fan of valid
   trajectories. *Constraint injection* midway (frozen scaffold / MW ceiling / prohibited toxophore /
   charge range / protected pharmacophore) — expect far fewer wasted oracle calls vs endpoint filtering.
5. **Pathwise (not endpoint-only) control.** `c(X_t)=1` ∀t (never alter a protected scaffold; never create a
   forbidden alert; stay in a charge/size range; preserve a pharmacophore throughout; stay above a min
   source-sim). Compare hard operator-support masking / FK running penalties / terminal penalties / endpoint
   rejection. Report completely-feasible-trajectory %, integrated + max violation, final quality, feasible HV,
   oracle calls wasted on rejected trajectories.
6. **Anytime interruption, continuation, reversal.** Interrupt after any #edits → valid candidate; resume
   later under a new objective; reverse a subset of edits; branch from a saved intermediate; change the
   remaining budget. Preference cycle A→B→A (hysteresis / irreversible-drift test).
7. **Oracle-exploitation robustness.** Optimize one oracle, evaluate with ≥1 INDEPENDENT (separate predictor
   / ensemble / alt-docking / consensus / higher-fidelity). Report guidance-oracle vs held-out-evaluator gap,
   uncertainty, OOD distance, alert/SA statistics. (A flexible legal editor is very good at finding
   adversarial predictor regions.)
8. **De-novo generation (secondary; after the editing experiments or appendix).** Validity/uniqueness/novelty
   /FCD/property-divergence/scaffold-diversity/size/trajectory-length. Optional late-onset guidance. The paper
   must NOT depend on arguing carbon-tree intermediates are all useful lead-opt states.

## 10. Baselines (small set of logically-appropriate head-to-heads, not a leaderboard)

- **Same-base control comparisons — ESSENTIAL** (hold the molecular process fixed, vary only the controller):
  unguided GM / endpoint+rerank / greedy local Δf / local Boltzmann rate reweighting / MOG-DFM-style
  rank-hypercone / learned Doob `h_φ` / exact Doob / SMC-FK. *This table matters more than beating unrelated
  generators — it separates what comes from the state space vs the steering rule.*
- **Molecular editing / lead-opt:** GenMol (modern diffusion hit/lead-opt), GraphGA + RetMol (GenMol's
  benchmark), MARS (classic multi-obj graph-edit MCMC), SMER-Opt (learned multi-step feasible edits),
  MolWorld (reachability-aware analogue-series), InVirtuoGen (discrete flow + GA/RL; PMO + lead-opt). Include
  MolDQN/Modof/graph-to-graph only if they match the exact source-conditioned benchmark.
- **Pareto:** NSGA-II/III, MOEA/D, SMS-EMOA, goal-conditioned/multi-obj GFlowNets, MOG-DFM, AReUReDi
  (Tchebycheff + locally-balanced MH + SMILES). **Fairest:** let NSGA-II/MOEA/D use OUR legal operator
  proposals (then improvement isn't just a better edit vocabulary).
- **PMO / sample efficiency:** GraphGA, GP-BO, REINVENT/Augmented Memory, Genetic GFN, + a current
  sample-efficient method (SEISMO/SEGO); report several budgets (50–100, 1000, 10000). PMO = breadth +
  efficiency, NOT the main Pareto evidence.

## 11. Critical ablations (support the causal "valid-path matters" claim)

Editing-trained vs carbon-prior; valid-operator vs propose-and-repair; exact-intermediates vs
endpoint-guidance; Doob vs local; full-support-mixture vs learned-alone (sweep ε); operator-family ablations;
hard-online vs endpoint-penalty constraints; preference-conditioning scheme (linear / Chebyshev / region /
adaptive-HV); exact-vs-learned h (control error vs h-error + backward residual); preference
interpolation/extrapolation (hold out simplex regions, test generalization).

## 12. Metric hierarchy (designate BEFORE final runs)

- **Primary:** normalized HV; HV-AUC vs oracle calls; exact-target TV (enumerable Doob); dynamic-adaptation
  regret; completely-feasible-trajectory rate.
- **Secondary Pareto:** IGD+; additive-ε; expected utility over held-out preferences; coverage/spacing;
  #feasible nondominated.
- **Chemical/preservation:** seed sim; scaffold/pharmacophore retention; diversity; edit distance;
  SA/complexity; held-out evaluator.
- **Computational:** oracle evals; model evals; legal candidates scored; accepted transitions; wall-clock;
  memory; cost per feasible-nondominated molecule.
- **Stats:** paired across identical source molecules; ≥5 stochastic repetitions; bootstrap CIs over seeds +
  sources.

## 13. Figure sequence

- **F1 Framework** — valid state space + reversible operators + GM rates + Doob transform. Message: *every
  state is a molecule, every edge a legal edit, every point an intervention point.*
- **F2 Exactness** — enumerated chemical graph: unconditioned vs desired-tilted vs exact-Doob vs
  local-guidance samples + TV error.
- **F3 Pareto lead-opt** — 2D + 3D fronts + HV-AUC + aggregate over the realistic benchmark.
- **F4 Dynamic Pareto fan** — one lead, shared prefix, several preference-conditioned valid branches.
- **F5 Pathwise constraints + ablations** — online hard constraints vs endpoint filtering; operator +
  controller ablations. Tables support, not carry.

## 14. Central contribution / abstract / titles

Abstract formulation: *"We formulate source-conditioned molecular optimization as stochastic control of a
generator-matched jump process on an operator-closed space of connected, valence-valid molecular graphs.
Because every state is a complete molecule and every transition is a legal molecular edit, rewards,
constraints, and user preferences can be applied throughout generation. We derive Doob-transformed
transition rates that preserve the legal operator support and exactly realize specified terminal
reweightings when the desirability function is exact. This enables dynamic preference changes, pathwise
constraints, and branching exploration of multi-objective trade-offs without restarting generation."* Then
the empirical claim: *"On similarity-constrained lead optimization, the resulting controller improves Pareto
hypervolume and oracle efficiency while preserving source structure, and accurately realizes target
distributions on enumerable chemical state spaces."*

Titles (preferred → technical → multi-objective):
- **Every Step Is a Molecule: Valid-Path Generator Matching for Controllable Molecular Optimization**
- Doob-Controlled Generator Matching on Valid Molecular Graphs
- Valid-Path Molecular Optimization with Exact and Dynamic Multi-Objective Steering

Use "lead optimization" in the subtitle/abstract/primary-experiment section, **not** the main title.

## Bottom line

Position as *a general stochastic-control framework for source-conditioned molecular optimization,
demonstrated through multi-objective lead optimization.* The carbon-tree de-novo generator is evidence the
operator system is a genuine generative framework; the **editing-trained process is where the valid-path
property becomes transformative.** The paper is large when it establishes all three: (1) a new molecular path
space (expressive, reversible, valid operators with demonstrable reachability); (2) a new control result
(exact Doob conditioning on that space, with carefully-scoped approximation at scale); (3) a capability gap
(dynamic preference switching, branching Pareto exploration, hard pathwise constraints — hard for masked
generation, not distributionally principled in heuristic graph-edit search).

---

## DIGEST — how this maps onto the current build (ours, not the advisor's)

**Confirms our direction:** the **editing (B-edit) model is the MAIN model**; carbon-tree de-novo is
secondary. Our whole B-edit investment is validated as the flagship.

**What this plan CHANGES vs. what we built / were about to train:**
1. **Training data must broaden.** Our corrupted-source-prior is only mixture item (2) — and the plan
   explicitly warns pure corrupt-and-reconstruct risks **identity bias**. We must ADD (1) **data-backed
   analogue pairs** (MMP / shared Bemis–Murcko scaffold / high-MCS / chemical series) as distinct
   source→target pairs, (3) an **edit-budget curriculum**, and (5) the **ε full-support legal mixture**
   `Q_ref=(1−ε)Q_θ+εQ_legal`. ⇒ **The held A100 fine-tune (corruption-only) does NOT match this recipe.**
2. **Explicit source-conditioning.** The plan leans toward `Q_θ(G,H | G_s, protected-mask, budget)` — an
   explicit source-conditioned model, not our current "universal edit prior" where the source is only the
   initial state. This is a model-input change.
3. **Doob becomes the centerpiece + needs the enumerable exact benchmark** (Setting A/B) and a
   **goal-conditioned `h_φ(t,x,z)`** for Pareto (region-conditioned, not linear scalarization). Our
   value-guided SMC (V0) is a controller, not the exact-Doob story yet.
4. **Terminology:** stop using **"valid manifold"** (I used it repeatedly) → "valid molecular graph state
   space." Never "synthesizable" → "graph-valid / chemically admissible."
5. **Baselines expand** beyond the earlier lit-search set: add GenMol, MARS, SMER-Opt, MolWorld,
   InVirtuoGen, NSGA-II/MOEA/D-over-our-operators, PMO set. The **same-base controller table** is the
   scientifically load-bearing comparison.

**Still valid from prior work:** anytime Pareto archive, dynamic steering, pathwise constraints, the
per-step-valid differentiation, cnof_leads→Jin-800 swap. **New load-bearing asks:** MMP/analogue-pair data,
ε mixture, source/budget conditioning, enumerable exact-Doob benchmark, goal-conditioned Doob, held-out
oracle, GenMol-comparable benchmark.
