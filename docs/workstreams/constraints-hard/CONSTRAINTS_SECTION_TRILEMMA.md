# The constraint trilemma — organizing claim for the constraints section

**Status: `DESIGN_ONLY`.** Nothing run. Every cell is sourced to a primary
document. **Every satisfaction rate carries its denominator**, because a
satisfaction rate without one is the specific way this literature misleads.

**Governed by** `docs/COMPARATOR_ROLES_CANONICAL.md`: the constraints block is
organized around **hard-support control as the framework contribution**, with
CDD / PRODIGY / ConStruct as the methodological lineage. A scaffold-specific RL
paper does not define this section.

---

## 1. The claim

> **Imposing a hard constraint on a pretrained generative process forces a
> choice among three properties, and no published method has all three.**
>
> **(a)** an **arbitrary constraint class** — any decidable predicate, not only
> those that are differentiable, projectable, or closed under some graph
> operation;
> **(b)** a **guarantee by construction** — satisfaction that follows from the
> mechanism, not from a measurement;
> **(c)** **no support enumeration** — sampling that never has to materialize the
> set of legal next states.
>
> COMPOSE takes **(a) + (b)** and pays for it in **(c)**. Every method that keeps
> (c) gives up (a), (b), or both.

The claim is a **characterization of a design space**, not a ranking. Stated as a
trilemma it survives review. Stated as a win it invites the reviewer to look for
the denominator — and in CDD's case they will find one.

## 2. The table

| | **(a) arbitrary class** | **(b) guarantee by construction** | **(c) no enumeration** | pays in |
|---|:---:|:---:|:---:|---|
| **CDD** (NeurIPS 2025) | ✗ surrogate or bespoke operator per predicate | ✗ empirical | ✓ | **output yield** |
| **PRODIGY** (ICML 2024) | ✗ differentiable aggregate scalars | ✗ partial projection | ✓ | **the guarantee** |
| **ConStruct** (NeurIPS 2024) | ✗ edge-deletion-invariant only | ✓ | ✓ | **the class** |
| **COMPOSE** | ✓ any decidable predicate | ✓ | ✗ ~500–600-wide fiber | **enumeration** |

### Cell sources

**CDD** — *Constrained Discrete Diffusion*, Cardei\*, Christopher\*, Hartvigsen,
Kailkhura, Fioretto. NeurIPS 2025; arXiv:2503.09790; DOI `10.52202/085713-0415`.
*(v1 was titled "Constrained Language Generation with Discrete Diffusion Models";
same work.)*

- **(a) ✗** — §4.2 Eq. 6 relaxes the non-differentiable `argmax` via
  Gumbel-Softmax; Appendix B.2.1: *"We finetune GPT2 (124M) to act as this
  surrogate model and directly output a score `s ∈ [0,10]`"*, trained on QM9 with
  RDKit SA labels. Predicates that resist relaxation are handled **not** by the
  ALM projection but by **hand-written combinatorial operators, one per task** —
  best-first search for novelty, numeric-token overwriting for counting, a
  closed-form aligned-index update for lexical constraints, ring
  expansion/opening/excision for the heterocycle constraint. **There is no
  general mechanism.**
- **(b) ✗** — Theorem 4.1 is a *non-asymptotic feasibility contraction*,
  `D_KL(x'_s, C) ≤ (1−α_t)·D_KL(x'_t, C) + α²_{t+1}·G²`, assuming β-prox-regular
  `C`. §4.1 hedges: *"This ensures the denoising trajectory remains within the
  allowable set **when C is convex**"*. The ALM inner solve is itself *"a lower
  bound of the original projection operator (5) by weak duality"*.
- **(c) ✓** — projection acts on simplex distributions; no successor set is
  materialized.

**PRODIGY** — *Diffuse, Sample, Project: Plug-And-Play Controllable Graph
Generation*, Sharma, Kumar, Trivedi. ICML 2024, **PMLR 235:44545–44564**;
OpenReview `ia0Z8d1DbY`. **No arXiv preprint** — cite PMLR.

- **(a) ✗** — constraint set is `C = {G : h₁(G) ≤ 0, …, h_k(G) ≤ 0}`;
  Appendix C.1: *"where each `h_i` is assumed to be differentiable to have a
  closed-form projection operator"*. Seven implemented families — edge count,
  triangle count, degree, valency, atom count, molecular weight, dipole — all
  **aggregate scalars with upper bounds**. Appendix C.4 abandons even a
  one-hidden-layer property predictor: *"Solving such a system of equations is
  hard … we do not consider non-linear approximators of graph properties and
  leave it for future works."*
- **(b) ✗** — the step is a *partial* projection,
  `G_{t−1} ← (1−γ_t)·G̃ + γ_t·Π_C(G̃)`.
- **(c) ✓**.

**ConStruct** — *Generative Modelling of Structurally Constrained Graphs*,
Madeira, Vignac, Thanou, Frossard (EPFL). NeurIPS 2024; arXiv:2406.17341; MIT.
*("ConStruct" is the method name and is not in the title — cite the title or the
reference will not resolve.)*

- **(a) ✗** — Definition 3.1: *"P is said to be edge-deletion invariant if, for
  any graph G and any subset of edges Ẽ ⊂ E, it satisfies: P(G) = True ⟹
  P(G′) = True, with G′ = (X, E \ Ẽ)."* Subgraph **presence** is edge-*insertion*
  invariant — the exact dual — so it is formally excluded. `projector/projector_utils.py`
  implements exactly three projectors: **planar, tree, lobster**. §4.1 notes even
  connectedness *"cannot be included as a constraining property since it is not
  edge-deletion invariant"*.
- **(b) ✓** — 100.0 ±0.0 on its families, by construction.
- **(c) ✓**.

**COMPOSE**

- **(a) ✓** — `F_C(x) = {y ∈ F(x) : C(y) = 1}` evaluates `C` on a **complete,
  sanitized molecule**, so any decidable predicate qualifies. No differentiability,
  no projection operator, no invariance requirement.
- **(b) ✓** — the commit set *is* `F_C(x)`, so every committed state satisfies
  `C`. **This is a construction check recorded without a denominator, never an
  empirical result** (project sign-guarantee rule; `CONSTRAINT_SEMANTICS.md` §7).
- **(c) ✗** — the exact legal fiber is materialized at every state.
  `docs/workstreams/PARALLEL_WORKSTREAMS_AND_HANDOFF.md` measures it at
  **~500–600 wide** (a ~586-wide fiber in the Pareto scorer profile). This is the
  real price and it belongs in the same paragraph as the claim, not in a later
  limitations section.

## 3. The instructive row is CDD, and it is about the denominator

CDD reports **0.0% violations at every threshold** — τ ∈ {3.0, 3.5, 4.0, 4.5},
confirmed verbatim in §5.2, in the direction `SA ≤ τ`. That is a real and
creditable result.

**Its denominator is valid molecules only.** Figure 4 caption:

> *"QED and constraint violations are reported for **only valid molecules**, and
> novel molecules must be valid and have no violation (τ ≤ 3.0)."*

And validity is where the constraint's cost lands:

| τ | valid molecules |
|---|---:|
| unconstrained UDLM base | **895** |
| 3.0 | **353** |
| 3.5 | 863 |
| 4.0 | 936 |
| 4.5 | 938 |

**A ~60% loss of usable output at the tightest threshold**, absorbed into the
denominator rather than reported as violations. The paper does not frame this
trade-off; it says only that CDD generates *"a competitive number of valid
molecules"*.

> **This is the trilemma made concrete.** A method that cannot restrict the
> support can still reach 0% violations — by discarding, mid-trajectory, the mass
> that would have violated. The cost surfaces as **yield**. Support restriction
> moves that same cost to **enumeration**. Which is preferable is an empirical
> question about the domain, and that is the honest claim.

**Rule for our own text:** if a CDD number appears anywhere in the paper, its
denominator appears beside it. Quoting "0.0% violations" alone misrepresents the
paper *in the method's favour* — which is precisely what we would object to if it
were done to us.

**Also do not propagate CDD's "203.4% increase" headline.** It is 117 → 355
against Schiff et al.'s *MDLM* CBG variant, with the comparison row marked
† = *"as reported by Schiff et al."* — copied, not re-run, and against a
different base model than CDD's own UDLM.

## 4. What the trilemma does NOT claim

- ❌ **Not** that COMPOSE's 100% beats CDD's 0.0%-on-valid or PRODIGY's `VAL_C`.
  Comparing a construction guarantee to a measurement is barred.
- ❌ **Not** that these methods are weak. They scale to graph sizes and datasets
  where enumerating an exact successor support is infeasible — PRODIGY on
  ZINC250k, ConStruct on GuacaMol. That is the (c) column doing real work.
- ❌ **Not** a head-to-head claim of any kind. `EXTERNAL_HARD_CONSTRAINT_AUDIT.md`
  §4 establishes there is **no native common protocol**: all three are de novo,
  none is source-conditioned, and a full-text search of PRODIGY and ConStruct for
  `inpaint | scaffold | lead optim | editing | source molecule | given molecule`
  returns zero relevant hits.
- ❌ **Not** that COMPOSE invented hard-constrained generation. ConStruct got
  there first for structural graph properties.

## 5. The nearest published analogue, cited by name

**ConStruct Appendix G.2** applies its projector **at sampling time only**, to an
unconstrained QM9 model, *"in contrast to all the experiments in the paper, we do
not train with only graphs that verify the property"* — reaching **100.0 ±0.0
acyclicity at 99.8% validity**, against a constrained-DiGress+ baseline at 81.3%
validity.

That is the same *shape* as COMPOSE's claim: frozen pretrained process, hard
constraint imposed at inference, satisfaction by construction. It is still de
novo, still an unlabeled structural property, still not a shared benchmark — but
it is the honest nearest neighbour and the section is stronger for naming it than
for leaving a reviewer to find it.

## 6. Cite ConStruct G.1 as a caution, not only as a rival

ConStruct's planarity experiment is a **published negative in the same shape as
our own scaffold stop**:

> *"planarity is too loose of a constraint, since the atoms composing molecules
> typically have low degrees … This ends up **slightly harming the performance**
> of the constrained generative model without bringing the benefits of increased
> validity."*

Planarity goes 99.7 → 100.0 on QM9 while FCD *worsens* 0.2090 → 0.3443 and
validity drops 99.0 → 98.5.

Naming it makes our Bemis–Murcko feasibility stop legible as **normal practice in
this literature** — a constraint can be too loose or too tight, and measuring
which before running the experiment is the discipline, not a lane that failed.

## 7. Where the evidence comes from

Per `COMPARATOR_ROLES_CANONICAL.md`, the constraints block's evidence is
**concentric**, and the trilemma is the *framework* layer only:

| layer | who supplies it | role |
|---|---|---|
| framework novelty | CDD / PRODIGY / ConStruct — **conceptual, this document** | `FRAMEWORK_NEIGHBOR` |
| **causality** | **post-hoc vs soft guidance vs exact support** — same `R_θ`, source, objective, controller, budget | `MATCHED_CAUSAL_CONTROL` |
| competence | at most one or two native editors | `TASK_COMPETENCE` |

**The primary comparison is internal.** The trilemma establishes that exact
support restriction is a *distinct* mechanism; only the matched internal arms
establish that it *matters*. **Lane 2 owns building that** — see
`LANE2_FIVE_QUESTIONS.md`. This lane does not design or run it.

### Why the framework layer carries no numbers

Under `docs/AMENDMENT_PUBLISHED_NUMBER_FIRST.md`, **every method in this block is
tier 3 or tier 4 — none is tier 1 or tier 2.** All three framework neighbours are
tier 4: *cite and discuss, do not reconstruct.* CDD's repo is a placeholder,
PRODIGY has no licence, ConStruct ships no checkpoints and no molecular config.

So this section **cannot be carried by published numbers**, and that is a
measured finding rather than an omission. The trilemma is the framework evidence.

**One published object does transfer:** CDD's `SA(y) ≤ τ` predicate at
τ ∈ {3.0, 3.5, 4.0, 4.5}, reusable as an **externally defined task** because the
constraint is then independent of our corpus — see `BASELINE_TASK_MATRIX.md` §3a
for the three verbatim-reproduction conditions. **The task travels; the numbers
do not.**
