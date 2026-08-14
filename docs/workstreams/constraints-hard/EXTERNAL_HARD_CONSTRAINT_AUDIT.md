# External hard-constraint benchmark audit

**Status: `DESIGN_ONLY`.** Nothing was run, installed, or launched. Every
capability cell cites a paper section or a repo file; anything not confirmed
against a primary source is marked **UNVERIFIED** rather than filled in.

**Scope:** the reframed question —

> Given a pretrained generative process, how should hard constraints be imposed
> during generation?

COMPOSE's answer is `F_C(x) = {y ∈ F(x) : C(y) = 1}`: restrict the exact
executable support, then run the same frozen control inside it. No
constraint-specific training, no soft penalty, no generate-then-reject.

**Audited:** CDD, PRODIGY, ConStruct.

---

## 0. Verdict

**Can ONE published constraint family be instantiated in COMPOSE exactly as
published?**

> **Yes for the constraint *predicate*; no for any of the three *mechanisms*;
> and no published numbers are head-to-head comparable.**

The instantiable object is **CDD's synthetic-accessibility predicate,
`SA(y) ≤ τ`**. It is a boolean function of a complete molecule, it was defined
independently of us, its thresholds were not chosen for us, and COMPOSE already
computes it with the canonical implementation — `src/compose_v4/eval/molecular_quality.py:13`
imports `from rdkit.Contrib.SA_Score import sascorer` and exposes
`"sa_score": lambda mol: float(sascorer.calculateScore(mol))`. **Zero new
scoring code.**

But four findings block the head-to-head:

1. **No official CDD code exists.** The authors' repo `jacobchristopher/CDD` is a
   **59-byte README** reading *"Anonymized repository. TODO: Add code
   implementaiton here."* — one commit, 2025-02-16, never touched. The NeurIPS
   checklist promises *"Code will be released following the review process"*;
   it has not been. A rerun is impossible.
2. **The task differs structurally.** CDD's molecular experiment is **de novo,
   unconditional SMILES generation on QM9**; COMPOSE's production process is
   **source-conditioned editing**. §4.
3. **The optimised predicate is a surrogate.** Gradients come from a **GPT-2
   (124M) finetuned to emit `s ∈ [0,10]`**, trained on QM9 with RDKit SA labels —
   not from `sascorer` itself, which is non-differentiable. §2.1.
4. **The reported 0.0% violation sits on a denominator that collapses.**
   Violations are reported **over valid molecules only**, and validity falls from
   **895 → 353** (≈60%) at the tightest threshold. §2.1.

So the correct classification, in the lead's own vocabulary, is:

> **An external hard-constraint *competence* benchmark — contextual,
> non-head-to-head. Not the same experiment.**

> ### ⚠️ Corrections to this lane's earlier report
>
> Two claims I made in my previous summary were **wrong** and are corrected here:
>
> - **"CDD's venue is unverified, possibly ICML 2025."** **Wrong — it is
>   NeurIPS 2025**, confirmed from the camera-ready footer (*"39th Conference on
>   Neural Information Processing Systems (NeurIPS 2025)"*), DOI
>   `10.52202/085713-0415`. My doubt came from the **v1** arXiv comment; the paper
>   was also **renamed** between versions (v1: *"Constrained Language Generation
>   with Discrete Diffusion Models"*). The charter was right and I was wrong.
> - **"CDD reports 21.3% satisfaction at τ = 3.0 and 63.9% at τ = 4.5."**
>   **Wrong — those numbers do not appear in the paper.** CDD reports **0.0%
>   violations at every τ**. The real critique is the *denominator*, not the rate;
>   see §2.1. The substance survives, the numbers do not.
>
> Both errors came from a first-pass search that a full-PDF reading overturned.
> Recorded rather than quietly edited.

---

## 1. Provenance

| method | venue / year | id | official code | checkpoints | license |
|---|---|---|---|---|---|
| **CDD** | **NeurIPS 2025** (poster) — **CONFIRMED** | arXiv:2503.09790; DOI `10.52202/085713-0415`; OpenReview `Es4s9dtCjR` | ⚠️ **placeholder README only** | no | none |
| **PRODIGY** | **ICML 2024**, PMLR **235:44545–44564** | **no arXiv preprint found**; OpenReview `ia0Z8d1DbY` | `github.com/prodigy-diffusion/code` | **yes, vendored** | ⚠️ **NONE** |
| **ConStruct** | **NeurIPS 2024** | arXiv:**2406.17341** | `github.com/manuelmlmadeira/ConStruct` | **none shipped** | **MIT** |

- **PRODIGY** — exact title *"Diffuse, Sample, Project: Plug-**And**-Play
  Controllable Graph Generation"*; Sharma, Kumar, Trivedi (Georgia Tech / MIT).
  PRODIGY = **PRO**jected **DI**ffusion for controlled **G**raph generation.
- **ConStruct** — exact title *"Generative Modelling of Structurally Constrained
  Graphs"*; Madeira, Vignac, Thanou, Frossard (all EPFL). "ConStruct" is the
  method name and does **not** appear in the title — cite the title, not the
  method name, or the reference will not resolve.

- **CDD** — *"Constrained Discrete Diffusion"*; Cardei\*, Christopher\*, Hartvigsen,
  Kailkhura, Fioretto (UVA / LLNL). **Cite the v2/v3 title.** v1 was
  *"Constrained Language Generation with Discrete Diffusion Models"*; stale
  citations under the old title are the same work. Base model is **UDLM** (92M);
  a companion workshop paper is noted in §2.4.

> ⚠️ **CDD's repo is a placeholder, not a release.** `jacobchristopher/CDD`
> contains a single 59-byte README: *"Anonymized repository. TODO: Add code
> implementaiton here."* One commit (2025-02-16), repo size 0, no code, no
> checkpoints, no licence, untouched since. The NeurIPS proceedings page carries
> no supplemental material and no code link. Two unofficial third-party
> reimplementations exist on GitHub (1 star and 0 stars); **neither is authored by
> the paper's authors and neither may be used as "the published method"** under
> `docs/BASELINE_IMPLEMENTATION_POLICY.md`.

> ⚠️ **PRODIGY's repo carries NO LICENSE** — GitHub API reports `license: None`
> and there is no top-level LICENSE file (only vendored ones inside
> `models/DiGress/` and `models/GraphScoreMatching/`). No licence is no grant of
> rights. This is the **same blocker class as DDSBM** and is legal before it is
> technical. The repo is also dormant — 4 stars, last push 2024-07-15 — and is
> not a standalone library: the README states the authors *"update the official
> code of each paper to incorporate our projected sampling approach"*, i.e.
> integration is a manual per-model patch.

## 2. Constraint mechanism — the structural comparison

| | **CDD** | **PRODIGY** | **ConStruct** | **COMPOSE** |
|---|---|---|---|---|
| where enforced | projection inside discrete-diffusion sampling | partial projection each step | edge-absorbing process + projector | filter on the enumerated legal successor fiber |
| **requirement on the constraint** | **differentiable / relaxable** | **each `h_i` differentiable**, for a closed-form KKT projection | **edge-deletion invariant** (Def. 3.1) | **any decidable boolean predicate on a complete molecule** |
| constraint form | `g_i(x) < τ_i` via a trained surrogate, **or** a bespoke per-task operator | `{G : h_i(G) ≤ 0}` — aggregate scalars | boolean `P`, closed under edge removal | `C(y) ∈ {0,1}` on a complete molecule |
| families implemented | SA threshold, novelty, lexical/counting; **+ 3-membered-heterocycle absence** in the companion workshop paper | **7** — edge/triangle count, degree, valency, atom count, mol. weight, dipole | **3 in code** — planar, tree, lobster | unbounded |
| hard guarantee? | **empirically 0.0%** violations, **but on valid molecules only** (895→353, ≈60% drop at τ=3.0); **no formal guarantee for non-convex `C`** | **NO — `VAL_C` 0.17–1.00** | **yes, 100.0 ±0.0** | **yes, by construction** |
| intermediate states | **simplex distributions**; the argmax-decoded string is checked, often not a valid molecule mid-trajectory | **continuous relaxation**, `A ∈ [0,1]`, not molecules | partial edge-subsets, not complete molecules | **complete valid molecules at every step** |
| training-free at inference | yes | **yes** (plug-and-play on frozen models) | **no** — needs its own edge-absorbing noise model (Appendix G.2 is the one exception) | yes |
| source-conditioned | no | no | no | **yes** |
| licence | n/a (no code) | ⚠️ **none** | **MIT** | — |

### 2.1 CDD reports 0.0% violations — on a denominator that shrinks by ~60%

**All four thresholds are confirmed verbatim** (§5.2): *"using a series of
thresholds (τ = 3.0, 3.5, 4.0, and 4.5)"*, in the direction `SA ≤ τ`. The
charter's values were exactly right.

**CDD reports 0.0% violations at every τ**, measured by the true black-box
scorer rather than the surrogate. That is a real result and should be stated as
one. Three qualifications, all from the paper:

**(a) The denominator is valid molecules only, and it collapses.** Figure 4
caption, verbatim:

> *"QED and constraint violations are reported for **only valid molecules**, and
> novel molecules must be valid and have no violation (τ ≤ 3.0)."*

At τ = 3.0 CDD yields **353 valid** molecules against **895** for its own
unconstrained UDLM base — a **~60% drop**. Invalid generations simply leave the
violation denominator. The paper does not frame this trade-off; it says only that
CDD generates *"a competitive number of valid molecules"*. Validity recovers as
the constraint loosens: 863 / 936 / 938 at τ = 3.5 / 4.0 / 4.5.

**This is the project's own denominator discipline applied to someone else's
table**, and it is the honest critique — not a claim that the rate is wrong.

**(b) There is no formal hard guarantee for a non-convex constraint.**
Theorem 4.1 is a *non-asymptotic feasibility* bound —
`D_KL(x'_s, C) ≤ (1−α_t)·D_KL(x'_t, C) + α²_{t+1}·G²`, i.e. distance to the
feasible set contracts geometrically — and it assumes `C` is β-prox-regular. §4.1
hedges explicitly: *"This ensures the denoising trajectory remains within the
allowable set **when C is convex**"*. The augmented-Lagrangian inner solve is
itself a relaxation: *"the primal optimization problem … is a lower bound of the
original projection operator (5) by weak duality"*.

So CDD's 0.0% is **empirical**, not structural. COMPOSE's 100% is structural and
therefore **not a result at all** — it is a construction check recorded without a
denominator, per the sign-guarantee rule.

**(c) The optimised predicate is a surrogate, though the measured one is not.**
Appendix B.2.1: *"We finetune GPT2 (124M) to act as this surrogate model and
directly output a score s ∈ [0,10]"*, trained on QM9 with RDKit SA labels.
Gradients flow through the surrogate and through a **Gumbel-Softmax relaxation**
of the non-differentiable `argmax` (§4.2, Eq. 6). Scoring and violation
measurement use the true black-box.

> **The admissible comparison is therefore: at matched feasible-output yield,
> what does the constraint cost in QED and novelty?** That has a free sign in
> both directions. It is also a comparison CDD's published numbers cannot
> support, because there is no code to run at a matched denominator.

**Not pinned, and it matters for reuse:** the paper cites RDKit generally
(Bento et al.) and Ertl & Schuffenhauer separately, but **never names
`sascorer.py` and pins no RDKit version, commit or hash**. Since SA values shift
with RDKit's fragment-contribution tables, a τ = 3.0 boundary is not
version-portable. If we ever instantiate `SA ≤ τ`, we pin our own version and say
so — we cannot inherit theirs.

### 2.2 The nearest thing to our predicate class is in CDD's companion workshop paper

Same five authors, NeurIPS 2025 **AI4D3 workshop**: *"Constrained Molecular
Generation with Discrete Diffusion for Drug Discovery"*. It reuses the identical
SA and novelty tables and **adds a third molecular constraint absent from the
NeurIPS paper**: prohibiting **three-membered heterocycles**, detected by **RDKit
substructure matching**.

Violations: AR 9.3% · MDLM 22.2% · UDLM 16.9% · **CDD 0.0%** (Valid & Novel:
11 / 271 / 345 / 351).

**This is the only genuinely non-differentiable, hard structural predicate on
molecules in the CDD line of work, and it is a substructure *absence*
constraint** — the dual of the presence constraint this lane audited. It is worth
citing precisely, because it is the closest any of the three methods comes to
COMPOSE's predicate class. Note how it is achieved: **not** through the ALM
projection, but through a **bespoke hand-written operator** (ring
expansion/opening/excision edits), which is exactly the pattern in §2.3.

### 2.3 None of the three can express "this labeled subgraph must be present"

This was the audit's key structural question, and the answer is **no** for all
three, for *different* and individually decisive reasons.

**CDD — only via a surrogate or a bespoke operator, never generally.** §4.3
advertises breadth (*"or even serve as a black-box function computed by an
external routine"*), but the mechanism splits in two. The ALM projection — the
paper's actual method — needs a differentiable or relaxable score, supplied by a
trained surrogate. Genuinely non-differentiable boolean predicates are handled
**not** by ALM but by **hand-written combinatorial operators, one per task**:
best-first search over token probabilities for novelty; overwriting numeric-token
probabilities from an external counter for letter-counting; a closed-form update
at an aligned index for lexical constraints; ring expansion/opening/excision for
the heterocycle constraint. **There is no general mechanism** — each predicate
costs a new operator. And the convergence theory assumes β-prox-regular `C`, with
guarantees claimed only for convex `C`.

**PRODIGY — aggregates only, and the wrong logical direction.** Its constraint
class is `C = {G : h₁(G) ≤ 0, …, h_k(G) ≤ 0}`, with Appendix C.1 adding *"where
each h_i is assumed to be differentiable to have a closed-form projection
operator"*. Three independent blockers:

1. **Direction.** All seven implemented families are **upper bounds** (`≤`).
   Presence is an existential *lower* bound. The `-Box`/`-Low` variants bound the
   same aggregate scalars, so *"at least 3 N atoms"* is expressible but is a
   **count**, not an identified substructure — it cannot require those atoms be
   bonded in a pattern.
2. **No node identity.** Subgraph presence needs a matching between the query's
   nodes and the target's. Nothing in the `h_C` family is position-aware;
   projection rescales whole matrices.
3. **No closed-form projection exists.** Appendix C.4 abandons even the far
   weaker case of a one-hidden-layer property predictor: *"Solving such a system
   of equations is hard … we do not consider non-linear approximators of graph
   properties and leave it for future works."* Subgraph matching is strictly
   harder — combinatorial, NP-hard, non-differentiable.

Its molecular *valency* constraint is a **validity** condition that COMPOSE's
executor already enforces natively in `is_valid_state`, so there is nothing to
compare there either.

**ConStruct — formally excluded by its own Definition 3.1:**

> *"P is said to be edge-deletion invariant if, for any graph G and any subset of
> edges Ẽ ⊂ E, it satisfies: P(G) = True ⟹ P(G′) = True, with G′ = (X, E \ Ẽ)."*

If `G` contains labeled subgraph `H`, deleting `H`'s edges gives `G′` without
`H` — so `P(G) = True` and `P(G′) = False`, violating the definition. **Subgraph
presence is edge-*insertion* invariant, the exact dual.** The mechanism makes
this structural rather than incidental: the forward process is edge-*deletion*,
the reverse is edge-*insertion*, and the projector only ever *discards* candidate
edges — a projector that can only remove edges can never create a required
substructure.

The authors name both missing pieces in §5 Limitations, as future work with no
implementation: inverting the framework for edge-insertion-invariant properties
(*"having at least n cycles"*), and — decisively for *labeled* cores —
*"incorporating **joint node-edge constraints** … represents an exciting future
direction."* That second gap matters because node types use a **marginal,
non-absorbing** noise model, so atom labels keep re-randomising during sampling:
even an inverted edge-side projector would not pin down a *labeled* substructure.

Also noted, §4.1: connectedness *"cannot be included as a constraining property
since it is not edge-deletion invariant"* — a constraint COMPOSE enforces
natively via `connected_successor_constraint`.

### 2.4 ConStruct Appendix G is the closest published analogue — and its planarity result is a negative

ConStruct **does** have molecular experiments, in Appendix G only. The main text
is synthetic (planar/tree/lobster) plus digital-pathology cell graphs.

- **G.1, planarity on QM9 / MOSES / GuacaMol.** Planarity 99.7 → **100.0** on
  QM9, but FCD *worsens* 0.2090 → 0.3443 and validity drops 99.0 → 98.5. The
  authors' own verdict: *"**planarity is too loose of a constraint**, since the
  atoms composing molecules typically have low degrees … This ends up **slightly
  harming the performance** of the constrained generative model without bringing
  the benefits of increased validity."* **A published negative, and directly
  relevant — it is the same failure mode as our own vacuous-constraint stop
  rule.**
- **G.2, acyclicity on QM9.** ConStruct reaches **99.8% validity** and **100.0
  ±0.0 acyclicity**, against a constrained-DiGress+ baseline at 81.3% validity.
  Crucially, G.2 *"in contrast to all the experiments in the paper, we do not
  train with only graphs that verify the property"* — the projector is applied at
  **sampling time only** on unconstrained QM9 models.

**G.2 is the single closest published analogue to COMPOSE's setup:** a frozen
pretrained generative process, a hard constraint imposed at inference, 100%
satisfaction by construction. Cite it precisely. It is still de novo, still
unlabeled-structural, and still not a shared benchmark — but it is the honest
nearest neighbour and the paper is stronger for naming it.

**Practical caveats if anyone ever tries to run it:** MIT-licensed and
maintained (19 stars, active to 2025-03), but **no checkpoints are shipped**, and
`configs/dataset/` contains exactly five files — `high_tls`, `lobster`,
`low_tls`, `planar`, `tree`. The QM9/MOSES/GuacaMol *loaders* exist in the
library, but **no molecular Hydra config is wired up**; you would author the YAML
yourself. Two further honesty notes from the paper: validity ≠ property
(connectedness is not guaranteed — tree V.U.N. is only 83.0 despite 100%
acyclicity), and Theorem 1 guarantees the GED-optimal projection is *in the set
of possible outputs*, not that a given run finds it.

## 3. The cross-cutting finding — a trilemma, and this is the methods-paper point

Set the three side by side and a pattern falls out that none of them states,
because each sees only its own corner.

**Only ConStruct delivers a *structural* guarantee** — and it buys that by
restricting to edge-deletion-invariant families, which is precisely what excludes
subgraph presence. PRODIGY keeps a broader-looking language and gives up the
guarantee outright (`VAL_C` 0.17–1.00). CDD achieves **0% violations
empirically** but has no formal guarantee off the convex case, and pays in
**validity yield** rather than in violations.

So the design space looks like a trilemma. Of

- **(a)** an arbitrary constraint class,
- **(b)** a guarantee that holds by construction,
- **(c)** sampling that avoids enumerating the successor support,

each method takes two:

| method | (a) arbitrary class | (b) guarantee by construction | (c) no enumeration | gives up |
|---|:---:|:---:|:---:|---|
| **CDD** | surrogate or bespoke operator per predicate | ✗ empirical 0%; **validity 895→353** | ✓ | **the structural guarantee, and yield** |
| **PRODIGY** | ✗ differentiable `h_i`, aggregates | ✗ `VAL_C` 0.17–1.00 | ✓ | **the guarantee** |
| **ConStruct** | ✗ edge-deletion invariant only | ✓ 100.0 ±0.0 | ✓ | **the class** |
| **COMPOSE** | ✓ any decidable predicate | ✓ by construction | ✗ ~600-wide fiber | **cheap sampling** |

**The CDD row is the instructive one.** A projection method that cannot restrict
the support can still hit 0% violations — by discarding, mid-trajectory, the
mass that would have violated. The cost surfaces as a **60% loss of valid
outputs**, not as violations. Support restriction moves that cost from yield to
enumeration. Which is preferable is an empirical question about the domain, and
saying so is stronger than claiming a win.

Because COMPOSE enumerates an **exact executable successor fiber of complete
molecules**, `C` may be any decidable predicate — a threshold on a
non-differentiable score, a substructure match, a conjunction of both — imposed
or changed at inference time on the state already reached.

**And the counterweight belongs in the same paragraph, not a later limitations
section:** that generality is bought with exact fiber enumeration, which is
expensive. The projection methods do not pay it, and they therefore scale to
graph sizes and settings where enumerating an exact successor support is
infeasible. PRODIGY runs on ZINC250k; ConStruct on GuacaMol.

> The claim is **constraint expressiveness and exactness, at the price of
> enumeration** — not "COMPOSE wins". Stated as a trilemma it is a genuine
> methodological contribution and survives review. Stated as a win, it invites
> the reviewer to find §2.1 and the ~600-wide fiber.

## 4. The task-semantics verdict, per method

The lead asked this be answered directly and not papered over.

| method | native task | same experiment as COMPOSE? | verdict |
|---|---|---|---|
| **CDD** | de novo molecular SMILES generation, QM9, token-level discrete diffusion | **no** | **contextual, non-head-to-head** |
| **PRODIGY** | de novo graph generation, QM9 + ZINC250k, aggregate constraints | **no** | **contextual**; constraint class disjoint, **and no licence** |
| **ConStruct** | de novo graph generation, structural families; molecules in Appendix G | **no** | **cite as lineage — G.2 by name; no number** |

**No shared benchmark exists.** Neither PRODIGY nor ConStruct contains any
source-conditioned molecular editing or lead-optimization task; a targeted search
of both full texts for `inpaint | scaffold | lead optim | editing | source
molecule | given molecule` returned **zero relevant hits**. QM9 is common ground
only as a *dataset*: both report FCD, validity, uniqueness and novelty, which are
**distribution-matching de novo metrics** with no ground-truth paired
source→target and therefore no definition under an editing task. Adapting either
to editing means building the evaluation protocol from scratch — which is exactly
the "distorted adaptation" the baseline policy tells us to refuse.

### Why CDD is not a head-to-head, specifically

Five independent mismatches, any one of which is disqualifying:

1. **Task.** De novo **unconditional** generation vs source-conditioned editing.
   §5.2 has no source molecule, no prompt, no scaffold — generation starts from
   the fully corrupted sequence. (§5.1's toxicity task *is* prompt-conditioned,
   so the distinction is one the authors draw themselves.) The source anchors
   COMPOSE's output distribution; a QED figure from unconditional generation and
   one from editing a given lead are not the same estimand.
2. **Process availability.** COMPOSE's production kernel *can* dispatch to a de
   novo runtime — `production_successor_kernel.py:282` falls through to
   `de_novo_rewrite_system()` — but the **frozen production `R_theta` is an
   Editing-V2 process** (`_default_rewrite_system` returns
   `editing_v2_semantic_rewrite_system()` for it). There is **no de novo
   checkpoint in the frozen chain.** Running the editing checkpoint de novo
   would be off-distribution.
3. **Dataset.** CDD uses QM9; COMPOSE's corpus is not QM9.
4. **Predicate.** CDD's operative constraint is a differentiable surrogate;
   ours would be the true RDKit `sascorer`. **Different predicates**, so
   "the same constraint" would be false.
5. **Unverifiable protocol, and not only because of the missing code.** The
   paper itself **never states** the number of molecules generated, the SA query
   budget, the number of seeds, or the train/val/test split, and reports **no
   error bars on either molecular table**. The word **"canonical" appears zero
   times** in the camera-ready — there is no stated sanitization procedure, no
   canonical-SMILES dedup, no duplicate-handling rule. Metrics are Valid, Novel,
   QED, Viol% with **no uniqueness and no diversity**, and none is given a formal
   equation. Published numbers may be **cited as reported and labelled that way**;
   they may never be presented as a rerun.

**One further trap in their table.** The CFG/CBG comparison rows in Figure 4
(right) are marked † = *"as reported by Schiff et al."* — **copied from prior
work, not re-run by the CDD authors**. Anyone quoting the headline "203.4%
increase" is quoting a cross-paper comparison, and against the *MDLM* variant
rather than CDD's own UDLM base. Do not propagate it.

### What would have to be true for it to become fair

All five, together:

1. a **de novo COMPOSE process with its own trained `R_theta`**, in the frozen
   chain — does not exist today;
2. **the same dataset and split** (QM9), which means a corpus COMPOSE is not
   trained on;
3. **the same operative predicate** — meaning we would have to obtain and run
   CDD's differentiable surrogate, which is **not released**;
4. **matched sample budget, seeds, and canonicalization**, which requires code
   that does not exist;
5. **matched satisfaction rate**, so the comparison is quality-at-fixed-feasibility
   rather than a guarantee against an empirical rate.

**Recommendation: do not pursue this.** Conditions 1 and 3 each require
substantial new work, and condition 3 depends on an artifact the authors never
released. Per `docs/BASELINE_IMPLEMENTATION_POLICY.md` this is squarely option 3
— *"do not pretend we have a faithful baseline; put it in related work and choose
a stronger executable comparator instead."*

## 5. What this lane recommends the paper say

**Related work, positioning — no numbers:**

> Recent work imposes constraints on pretrained discrete generative processes by
> projection during sampling. CDD inserts a differentiable constrained projection
> into discrete diffusion; PRODIGY projects each diffusion step onto a
> user-specified graph-constraint set of the form `{G : h_i(G) ≤ 0}` with each
> `h_i` differentiable; ConStruct guarantees structural graph properties
> throughout the trajectory via an edge-absorbing process and a projector.
> Each buys tractability by restricting the admissible constraint class — to
> differentiable, projectable, or edge-deletion-invariant constraints
> respectively — and the first two relax the guarantee itself, reporting partial
> constraint satisfaction. ConStruct attains exact satisfaction, at the cost of a
> constraint class that formally excludes substructure presence, since deleting
> an edge of a required subgraph destroys it. COMPOSE instead restricts an exact
> executable successor support over complete molecules, so the constraint may be
> any decidable predicate and may be imposed or changed at inference time on the
> state already reached; the price is exact fiber enumeration, which the
> projection methods avoid and which lets them scale to larger graphs.

**The one comparison worth drawing explicitly, because it is the nearest
neighbour:** ConStruct's Appendix G.2 applies its projector at sampling time to
an unconstrained QM9 model, without training on constraint-satisfying graphs, and
attains 100% acyclicity at 99.8% validity. That is the same *shape* as COMPOSE's
claim — frozen process, hard constraint at inference — on an unlabeled structural
property in a de novo setting.

**And cite G.1 as a caution, not just a rival:** ConStruct's planarity experiment
is a published negative of exactly our own vacuous-constraint failure mode —
*"planarity is too loose of a constraint … This ends up slightly harming the
performance."* Naming it makes our own scaffold stop legible as good practice
rather than a lane that failed.

**If any CDD number is quoted at all**, cite it **as reported** — never as a
rerun — and carry three things with it: the threshold, the **valid-molecule
denominator** (353 of them at τ = 3.0, against an unconstrained base of 895), and
the fact that gradients come from a trained surrogate rather than `sascorer`.
CDD's 0.0% is a real and creditable result; quoting it without its denominator
misrepresents the paper in the method's favour, which is the failure mode we
would most object to if it were done to us.

**Do not propagate the "203.4% increase" headline.** It is 117 → 355 against
Schiff et al.'s *MDLM* CBG variant, with the comparison row copied from that
paper rather than re-run, and against a different base model than CDD's own UDLM.

## 6. Unverified — flagged, not filled in

| item | status |
|---|---|
| ~~CDD venue~~ | **RESOLVED — NeurIPS 2025**, camera-ready footer + DOI `10.52202/085713-0415` |
| ~~CDD thresholds~~ | **RESOLVED — all four (3.0, 3.5, 4.0, 4.5)** confirmed verbatim, §5.2 |
| CDD sample budget, seeds, variance | **NOT STATED IN THE PAPER** — not merely missing code. No error bars on either molecular table |
| CDD train/val/test split | **NOT STATED** |
| CDD canonicalization / dedup convention | **NOT STATED** — "canonical" appears zero times in the camera-ready |
| CDD RDKit / `sascorer.py` version pinning | **NOT STATED** — τ boundaries are therefore not version-portable |
| whether CDD's QM9 UDLM base is Schiff et al.'s checkpoint or retrained | **UNVERIFIED** |
| CDD OpenReview reviews / rebuttals | **UNVERIFIED** — forum returns a bot challenge (403) |
| PRODIGY arXiv preprint | **none found** — high confidence, but a negative cannot be proven. The project page's own arXiv link is an unfilled placeholder (`arxiv.org/abs/2402.NNNNN`) and the official BibTeX cites OpenReview. **Cite PMLR 235:44545–44564, not arXiv.** |
| PRODIGY OpenReview reviewer discussion | **UNVERIFIED** — forum returned a CAPTCHA wall; the ID `ia0Z8d1DbY` is confirmed from the paper's own BibTeX |
| whether any of the three shares a benchmark with source-conditioned molecular editing | **none — searched both PRODIGY and ConStruct full texts; zero relevant hits** |

Now verified that previously were not: PRODIGY's licence (**none**) and
checkpoints (**yes, vendored**); ConStruct's projector list (**exactly three**,
from `projector/projector_utils.py`) and its molecular configs (**none shipped**;
`configs/dataset/` holds exactly `high_tls`, `lobster`, `low_tls`, `planar`,
`tree`).

**If any CDD number is quoted, quote its denominator with it.** "0.0% violations"
without "over valid molecules, 353 of them at τ = 3.0 against an unconstrained
base of 895" is not a faithful report of the paper.
