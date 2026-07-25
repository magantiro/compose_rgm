# Conditional claims → controls → metrics (the experiment matrix)

**Authority.** This operationalizes `PAPER_POSITIONING_EXACT_CONTROL.md` (the *corrected*
positioning that supersedes earlier over-claimed drafts). Where this doc and ad-hoc framing
(`COMPOSE_paradigm_advantages_talking_points.md`, the "barrier spotlight") conflict, the
positioning doc wins. Nothing gets run that is not a row in the matrix below.

---

## Two kinds of control answer two different questions

- **External baseline** — *is the every-step-valid-molecule paradigm better at this task than the
  alternative (latent guide-then-decode, or a molecular search heuristic)?* This is the ICLR
  headline: the field wants to see we beat/match real methods on the axes we claim.
- **Internal ablation** — *within our method, is the component we credit (the learned generator
  `Q_θ`, or the controller) actually responsible — or would a trivial version (uniform-over-the-
  legal-fiber, or no controller) do just as well?* This is attribution/honesty. A reviewer runs it
  in their head for **any performance claim**: "is this just the valid substrate?"

### When is an internal ablation actually required?

**Not for everything.** The rule:

- **By-construction guarantee / capability claim** → *external only.* The internal ablation is
  **uninformative** because the trivial version *also* satisfies the guarantee. (Constraints are the
  clean case: uniform-legal is 100% too — it proves nothing.)
- **Performance / quality claim** (we do X *better / more efficiently / more realistically*) →
  *internal ablation is mandatory defense*, external is the headline. Without it a reviewer assumes
  the gain came from the substrate + controller, not from the learned generator, and we have no
  answer.

The internal ablation is not there to impress reviewers; it is there to **stop us from making a
claim it would demolish in review**, and to tell us *which* claim is even true. (It already did its
job: the **de-novo** HV comparison turned out **gameable** — uniform "ties" only by producing trivial
alkanes — so we **retired it as the wrong regime**. The informative version is the uniform ablation in
the **editing** regime, where uniform's random thrashing loses to sensible edits. We do **not** claim
"we optimize better"; we claim what survives: realism / anytime / constraints / exactness.) The right
internal control also **differs per claim** — `uniform-legal` is only one of {uniform-legal,
no-controller, unconditioned-base, continuation-vs-restart}, is **run in the editing regime (never
de-novo)**, and is actively the *wrong* one for the constraint claim.

> **Retired experiment (recorded):** the de-novo composition experiment (`composition_learned_vs_uniform.py`)
> is retired as structurally uninformative — the intermediates of a de-novo trajectory are valid but
> not *useful* molecules, so archive/anytime/steering can't be tested there, and the endpoint metric is
> gameable. Claims 1 & 2 live only in the editing regime.

---

## The matrix

| # | Honest claim (per positioning doc) | Internal ablation — needed? | External baseline (the paradigm test) | Metric | Do NOT claim |
|---|---|---|---|---|---|
| **3** Exact conditioning | Rule-closed structural constraints exact by construction; property conditioning verified **exact only on a tractable exhaustive toy** (true h-transform). Real molecular guidance is **approximate** (finite-particle SMC / learned twist). | **unconditioned base + analytic target — YES** (it *is* the demonstration). uniform-legal N/A. | None head-to-head — capability contrast only (latent guidance is a noisy-state surrogate). | \|guided − analytic conditional\| → 0 on the toy. | exact conditional generation on *real* molecules. |
| **4** Hard constraints | Rule-closed constraints (fixed scaffold, required/forbidden substructure, connectivity, size) enforced exactly by construction, native in one process; conjunctions don't collapse. | **uniform-legal is ALSO 100% (it's in the fiber) → NOT informative → SKIP.** The clean case where an internal baseline adds nothing. | generate-then-filter; direct conditional (GrIDDD/FreeGress); constrained diffusion (ConStruct/PRODIGY/CoCoGraph). | satisfaction rate under 1..k conjoined constraints; usable-oracle rate. | "diffusion can't do hard constraints" — ConStruct/PRODIGY/CoCoGraph guarantee topology/validity/absence classes. Our scoped edge = **required-substructure** class + **evaluable** intermediates + the **conjunction**. |
| **2** Frontier / HV + realism | For a fixed oracle budget the trajectory yields an **anytime, source-relative Pareto set of realistic molecules**. | **uniform-legal + no-controller — YES, in the EDITING regime.** (De-novo HV is gameable — uniform ties via trivial alkanes; the informative test is editing a real lead, where uniform thrashes and learned-SE edits sensibly.) | latent multi-objective (MOG-DFM / AReUReDi / training-free MO-diffusion); molecular search (GraphGA / MARS / GFlowNet). | feasible HV / oracle-call; **FCD / distance-to-data (realism)**; anytime best-so-far. | optimization SOTA (GraphGA is strong — Gao 2022, PMO). |
| **1** Dynamic steering | Change ω mid-trajectory and warm-start from a molecule; no re-noise / re-decode. | **continuation-vs-restart — YES** (quantifies the warm-start benefit *within* our method). uniform-legal N/A. | latent guidance (must re-noise / restart per ω). | unique oracle-calls to cover a frontier under an ω-sweep: continuation vs restart vs latent-restart. | — |

**Read-out for the "is internal needed?" question:** needed for **3, 2, 1** (each attributes something
to the learned generator/controller/warm-start); **not** needed for **4** (by-construction guarantee).
And the headline for the field is the **external** column throughout — internal ablations are the
defensive flank, not the pitch.

---

## The SE (editing) model design — LOCKED

**Decision.** The SE model is a **fine-tune of the de-novo base B with an objective-agnostic,
lightly-corrupted-molecule source prior** — a general "realistic edits from a molecule" prior carrying
**no property information**. The controller supplies the objective at inference. This is the
reconstruction/bridge setup (I2SB-style: *corrupted data as the source instead of noise*), **NOT a
property bridge**.

**Why objective-agnostic — the load-bearing reason.** Our headline capabilities *require* it:
- **Dynamic steering (claim 1)** — rotating ω mid-trajectory is impossible if the objective is baked
  into the model.
- **Pareto exploration (claim 2)** — covering a frontier needs an objective-agnostic prior the
  controller steers in many directions.
A property-baked model (a low→high-QED bridge) can do *neither* — it is committed to one objective.
This is the control-closed-substrate thesis: **fixed objective-agnostic base + arbitrary controllers.**

**Closest prior work — DDSBM (must-cite; baseline; differentiate).** DDSBM (Discrete Diffusion
Schrödinger Bridge Matching, 2024, ZINC250K/Polymer) does molecular optimization as a **property
bridge** between two real-molecule datasets (low→high property), minimal graph-edit transport. It
**bakes the objective into the model** → no dynamic steering, no Pareto, no per-step validity
guarantee. Our differentiation: (a) **objective-agnostic prior + arbitrary controllers** (steerable,
dynamic, Pareto); (b) **validity-closed executable-rewrite CTMC** (every step a valid molecule by
construction); (c) **exact control composition** (Doob/SMC). Borrow their one good idea:
**graph-edit-distance as the coupling cost** = the minimal-edit "short/near" coupling.

**The corruption that makes a good prior (design spec).**
1. **Objective-agnostic / property-free** — no labels; the controller steers.
2. **Short/near** — minimal edits (graph-edit-distance coupling); stay near the lead (similarity
   constraint). Inference *chains* single-step edits for reach, so training need not be deep.
3. **Operator-diverse** — peripheral (atom/bond insert-delete) + **bioisostere (`atom_restate`)** +
   **occasional ring (`ring_system_*`, ~15%)** + graft, so the model can edit in *many directions*
   (Pareto coverage). *[The current corruption is too narrow — atom/bond insert only — and must be
   broadened.]*
4. **Stochastic/high-entropy** — random operator + location → a *diverse* edit prior, not a narrow
   undo-map (matches the entropic Schrödinger bridge).
5. **Realism-preserving** — drug-like intermediates → the anytime archive is usable.
6. **Both directions** — real→simpler (real source; teaches *trim*; makes real leads in-distribution,
   stops premature termination) and simpler→real (teaches *grow*).
7. **Mixed with de-novo** carbon-tree trajectories — retain de-novo (the exactness/constraint spine).

**Training.** Warm-start fine-tune from B, **standard mode** (not the Modal-failing `superposed`
ring-fix), corrupted-prior + carbon-tree mix. **Verify before committing GPU:** the realized
edit-operator histogram, ring-change fraction, source realism (Tanimoto-to-target, QED), edit-length
spread — confirm they match this spec (no silent under-coverage).

**Literature grounding (we're on established ground).** Corrupted-data-as-source = I2SB (ICLR 2023);
source-distribution choice is a studied FM design axis; edit-operation flows = Edit Flows / discrete
FM; molecular bridges = DDSBM / SynBridge. Generator Matching subsumes all. **Our contribution is the
objective-agnostic, steerable version on a validity-closed rewrite CTMC** — which is exactly what a
property bridge (DDSBM) cannot be.

---

## Reconciliation — where recent framing drifted from the positioning doc

The positioning doc (§"What we DO NOT claim") already retracted several claims that the talking-points
/ barrier framing re-introduced. Corrections (talking-points doc edited to match):

1. **Exactness overclaim.** Talking points said the Doob h-transform "reproduces the exact Bayesian
   conditional to machine precision **on the real rewrite process**." → Positioning §36: exact
   h-transform needs the true (intractable) harmonic function; real guidance is approximate. **Fix:**
   scope exactness to the **tractable toy**; real molecular guidance is honestly approximate.

2. **"Diffusion can't do constraints" overclaim.** Talking points said structured diffusion
   "**provably cannot even represent** a required scaffold/SMARTS." → Positioning §40: ConStruct/
   PRODIGY/CoCoGraph *can* enforce topology/validity constraints. **Fix:** state a **guarantee gap**,
   not an impossibility — their by-construction guarantees cover topology/validity/absence classes; a
   *required substructure that must hold at every step* isn't something their process guarantees,
   whereas our legal-event set does. Do **not** claim they can't be *guided* toward it.

3. **Barrier mis-positioned.** The barrier was framed as a molecular-vs-latent advantage ("no meaning
   in a latent process"). It is **greedy-vs-non-greedy**, not molecular-vs-latent — a non-greedy
   latent optimizer crosses its own barriers. The positioning doc does **not** feature it. **Fix:**
   the barrier is a **mechanistic/science result** (why non-myopic valid-path control helps), *not* a
   load-bearing external comparison. Demote it; keep grow/shrink (flexible-size) reachability as a
   **capability**, not an optimization-superiority claim.

4. **Exclusivity → conjunction.** Reframe the headline from "problems latent fundamentally cannot do"
   to the positioning doc's actual thesis: the **conjunction** {valid **and evaluable** intermediates
   + flexible non-monotone size + executable-rewrite editing + rule-closed constraints} is native to
   **one** process. Each piece may exist elsewhere; the combination does not.

**The trap to keep flagged:** methods that *also* keep valid molecular intermediates — GraphGA, MARS,
GFlowNet — are **not** differentiated from us on validity. Versus *them*, our edge is the **learned
generative prior + exact control composition**, not raw optimization. The valid-intermediate
paradigm contrast is specifically against **latent** methods.

---

## Net honest positioning (from `PAPER_POSITIONING_EXACT_CONTROL.md`)

Framework + native-conjunction capability + **rule-closed exact constraints (provable, scoped)** +
**competitive** (not SOTA) optimization at matched budgets + **honestly-approximate** learned guidance,
verified exact only on a tractable toy. No claim of exact conditional generation on real molecules; no
claim that constraints are impossible elsewhere.

---

## Execution plan, generator decision, and the ring defect

### The plan splits into two tiers by *generator dependency*

| # | Experiment | Generator | Retrain? | Prerequisite |
|---|---|---|---|---|
| **3** exactness | E0 toy + Doob ground-truth | toy — generator-agnostic | none | fixed machinery ✓ |
| **4** hard constraints | scaffold/SMARTS 100% + conjunction-collapse vs generate-then-filter / ConStruct | de-novo B | none | fixed machinery ✓ |
| **2** anytime archive + realism | source-relative Pareto archive on real leads | **SE-edit** | **SE fine-tune (from B)** | corrupted-prior data-gen + working Modal |
| **1** dynamic steering | continuation-vs-restart on a real lead | **SE-edit** | **SE fine-tune (from B)** | corrupted-prior data-gen + working Modal |

- **Tier 1 (spine: 3, 4)** — de-novo B, **no retrain, ring defect irrelevant** (toy + fiber-guarantee do the work). Bankable now, zero Modal.
- **Tier 2 (headline: 2, 1)** — SE-edit generator. The memo's hero (anytime archive) and the med-chem-relevant lead-optimization story.

### Generator decision (forced by the logic of the claims)

**SE-edit for the headline; de-novo B for the spine.**
- 3 & 4 don't depend on base quality → de-novo B suffices, no retrain.
- Claim 2's honest axis is **realism** (learned ties uniform on HV), and realism is exactly where de-novo B is worst (~52% small rings) → de-novo B is the wrong tool for it.
- The only regime where the learned generator cleanly beats uniform-legal is **editing from a real lead** (uniform-legal from a lead = random thrashing) → SE.

### Ring-defect verdict: NO dedicated retrain

- Spine (3, 4): ring defect irrelevant. No retrain.
- Headline via **SE-editing**: you start from a real, realistic lead and edit locally → inherit its good ring systems; ring-*adding* is rare in lead-opt → the defect is **largely sidestepped** (measure ring-quality of edited outputs to confirm).
- The only route that *needs* the ring fix is **de-novo unconditional realism (E1/FCD)** — not a headline (per the memo), and Modal-blocked anyway. We don't take it.
- Do **not** fold the Modal-failing `superposed` mode into the SE retrain — do SE in **standard mode** to avoid inheriting the 3× failure; add the ring fix later only if editing outputs show ring problems.

### The real gate is Modal, not rings

Both Tier-2 routes need an at-scale retrain, and Modal training has **failed 3× recently** (that's what blocks the ring fix). So the true first prerequisite for the headline is **establishing that Modal training works at all** — above even the SE data-gen fix.

### Recommended sequencing
1. **Now (local, bankable, no Modal):** Tier-1 spine + build the objective-agnostic corruption prior
   (broaden operators; verify the edit-skill / ring / realism distribution).
2. **Gate:** de-risk Modal — get any small fine-tune to complete.
3. **Then (Modal):** SE **fine-tune from B** (standard mode, corrupted-prior + carbon-tree mix) →
   Tier-2 editing headline. Include the **corrupted-prior-vs-de-novo** ablation (under the same
   controller) — it is under-reported for GM/rewrite editing and is ours to own, not assume.

---

## Baselines for the editing headline (claims 1 & 2)

Grounded in `CONDITIONAL_CONTROLLER_DESIGN.md` §Tasks/Baselines (already vetted). The internal
ablation (uniform-legal, no-controller) is **not** the headline for 1 & 2 — it is the "is it just
the substrate?" flank. The headline is the comparison against **other paradigms**, on a **standard
benchmark** so most baselines come from published tables (avoids the crippled-baseline criticism).

### Benchmark
- **Headline: ZINC constrained QED optimization at Tanimoto ≥ 0.4** to the lead (Jin et al. 2018) —
  this *is* lead-editing (improve QED while staying similar to a real lead). Secondary: penalized-logP
  (constrained), DRD2.
- Metrics: success rate (≥ target at the similarity floor), **diversity, novelty**, **FCD/realism**,
  **oracle-budget curves**. Always per-sample, matched oracle budget — never selected-top-k.
- **Endpoint QED is a *defensive* number, not the contribution.** We use the ZINC task for its **hard
  constraint** (similarity ≥ 0.4) + anytime/realism/steering, **not** for QED optimization. Report our
  QED improvement only as a **"no-tax" check** (the guarantees come free); if it isn't competitive,
  **drop it** and lead purely with the differentiated axes. The task is the *stage*; the constraint is
  the point.

### Tier 1 — cited method baselines (published numbers on this exact task)
GrIDDD (45.1% @ sim≥0.4), **JT-VAE (latent BO — the benchmark's origin)**, GCPN, MARS, GraphAF,
GFlowNet-TB. Report **competitive, not SOTA** (GraphGA/PMO strong; Gao 2022). Cite, don't
re-implement — keeps the comparison fair and un-crippled.

### Tier 2 — the ONE latent foil we RUN ourselves (the paradigm contrast)
Published tables report endpoint success; they do **not** measure the axes we're differentiated on.
So we run one latent guide-then-decode method on our leads to make the intermediate-step gap concrete.
- **Recommended foil: JT-VAE** (latent Bayesian optimization; released code; the task's origin) — or
  a modern latent-diffusion (DRAKES / GeoLDM + guidance) for a current comparison. Its optimization
  happens in latent space and it decodes once, so it **cannot** hold a hard constraint at every step,
  score the real oracle per step, or yield an anytime archive.
- Paradigm-specific axes (where we win, published tables silent): (a) hard-constraint-under-conjunction
  satisfaction — ours 100% by construction vs latent+filter collapse; (b) anytime archive — our
  feasible Pareto set from one trajectory vs its single endpoint at matched budget; (c) per-step
  validity/realism of intermediates.

### Per-claim external comparison
- **Claim 2 (frontier/archive):** feasible HV/oracle-call **+ FCD** vs {latent foil, GraphGA, MARS,
  GFlowNet}; internal ablation (uniform-legal, no-controller) underneath.
- **Claim 1 (dynamic steering):** continuation-vs-restart (internal) **and** vs the latent foil
  (which must re-noise / restart per preference).

### Why FCD/realism is non-negotiable (empirical, this session)
Unguided **uniform-legal produces trivial branched alkanes** (`CCC(C)(CC)CO`, QED 0.60; mean QED 0.37
over n=8, no rings) that score deceptively high on QED×SA — which is *why* it ties/beats learned on
that metric. The learned model produces realistic ring-containing drug-like molecules (lower QED,
real). So HV-on-gameable-scalars rewards triviality; **FCD / distance-to-data is the metric that
separates the paradigms honestly**, and it is where the learned model — and editing a real lead —
wins. Do not report a bare QED×SA HV number without a realism axis beside it.
