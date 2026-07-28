# Paper 1 — authoritative framing record

**Status: AUTHORITATIVE.** Supersedes the control-first framing in earlier drafts and the RGM-first
interim reframe. Derived from the owner's transmission of the amended three-paper PhD strategy document
(2026-07-28). The named source file `Generative_Design_PhD_Program_and_Three_Paper_Figure_Plans_
Original_Format_Latest_Content_ICLR_Framing_Updated.docx` is **not on disk** (closest present:
`~/Downloads/..._Original_Format_Latest_Content.docx`); the owner's verbatim transmission is the record.

This document exists so the framing survives without re-pasting. Read it before any positioning,
title, abstract, section-order, claim, or experiment-scope decision.

---

## 1. The one thing to remember

> COMPOSE learns an executable, trans-dimensional stochastic rewrite program over molecules and turns
> its canonical molecular process into a substrate for exact and dynamic design.

**Causal chain — every element of the paper must sit on this line:**

```
executable state-dependent rewrites
  → Generator Matching
    → a learned TRANS-DIMENSIONAL molecular process
      → a canonical molecular-successor kernel
        → exact and dynamic stochastic control
```

Rewrite Generator Matching is the **primary invention**. COMPOSE is its **molecular instantiation**.
Source-conditioned lead optimization is the **flagship application** — it demonstrates why the framework
matters; it does not define it. **Do not open the introduction with lead optimization.**

**Tagline:** *Every step is a molecule; every transition is an executable rewrite.*

**Title — SETTLED (owner, 2026-07-28):**

> **COMPOSE: Stochastic Rewrite Generator Matching for Trans-Dimensional Molecular Generation and Control**

COMPOSE is retained because it is already the recognizable name of the molecular instantiation and is used
via `\method` throughout; dropping it would make the paper read as an unnamed framework paper. "Stochastic
Rewrite Generator Matching" reads more naturally than "Generator Matching over stochastic rewrites",
preserves framework-first positioning, and carries the three things a reviewer must remember: stochastic
rewrite generator matching · trans-dimensional molecular generation · control. The tagline is a **tagline
only** and must never appear as the title.

---

## 2. Three contributions (in this order)

1. **Rewrite Generator Matching** — generative modeling over a state-dependent executable rewrite
   language: learn a total event hazard and the identity/location of domain-meaningful events, rather
   than reversing a generic tensor corruption.
2. **COMPOSE + the canonical successor quotient** — a trans-dimensional molecular process whose events
   create, delete, relabel, reconnect and cyclize structure while every committed state stays inside a
   declared molecular graph space; quotienting makes the state-level process independent of action encoding.
3. **Generation and control** — one learned process supports unconditional generation *and*
   source-conditioned editing; finite-horizon control of its molecular successor kernel gives exact event
   conditioning, dynamic retargeting, Pareto branching, and pathwise constraints.

---

## 3. What actually has to land (editorial judgment)

The manuscript's real risk is reading as a grab-bag of capabilities. Ranked by what carries the paper:

**(a) The generative object is new — lead by selling that.** Not "we optimize leads well." The
invention is that the generative process *is* a stochastic program of executable rewrites. If a reviewer
finishes the intro thinking "molecular editor with a neural scorer," the paper has failed regardless of
its numbers.

**(b) Trans-dimensionality is what makes it a generator rather than an editor.** Birth/death genuinely
moving between `X_n` spaces is the property that defeats the most likely dismissal. It must be
*measured*, not merely illustrated by a trajectory cartoon — and it must be visibly distinguished from
padded-slot bookkeeping (see §7 experiment C). This is the single most attackable claim in the paper.

**(c) The canonical quotient is the subtle, high-value technical contribution.** It converts mark-level
probabilities into a molecule-level Markov kernel, which is precisely what makes every downstream control
statement well-posed and representation-independent. Sophisticated reviewers will reward this; it was
previously buried as an implementation detail. Promote it.

**(d) Control is the consequence, not the premise — and the honest framing is stronger.** Exactness holds
on enumerable spaces with exact desirability. Framed as *"a ground truth against which approximate
controllers are measured rather than assumed correct"* it is a better contribution than an overclaimed
"we control exactly," and it is immune to the obvious attack.

**(e) Validity closure is interesting only through what it enables.** State it once, powerfully, then
*use* it: oracle evaluation at any step, interruption, branching, pathwise feasibility. Repeating
"every state is a molecule" in five sections is the current draft's biggest redundancy.

**(f) Write the experiment section around questions, not around numbers.** All results are currently
`\XXX` (see §9). Structuring §7 as "Q: does the learned process transport better than uniform legal
rewriting? → measured by …" means results slot in when they land, and the argument survives review of the
structure before the numbers exist.

---

## 4. Novelty boundaries — state these explicitly in the text

**We do NOT claim as new:** atom insertion/deletion per se · valid molecular editing per se · Generator
Matching · Doob transforms · variable-dimension generation · graph grammars.

**We DO claim** the formal combination and interface:

> a generator-matched stochastic graph-rewrite process whose state-dependent marks create, delete,
> relabel, reconnect, and cyclize molecular structure; whose legal support is closed over molecular
> states; whose equivalent action encodings are quotiented into a molecular kernel; and whose resulting
> process supports exact finite-horizon control.

Remove or qualify **every** unsupported "first". Crediting prior art strengthens the combination claim.

**Required related work:** Trans-Dimensional Generative Modeling via Jump Diffusion Models · GrIDDD /
insertion-deletion graph diffusion · MARS and valid graph editors · graph grammars · generic Generator
Matching · standard graph diffusion/flow · classical Doob transforms.
`RELATED_WORK_MATRIX.md` columns: variable cardinality · atom birth/death · topology-changing ops ·
complete molecular intermediates · learned reference path distribution · canonical successor quotient ·
exact control · dynamic retargeting · pathwise constraints.

---

## 5. Trans-dimensional formalism

`X = ⋃_{n=1}^{N} X_n`, with `X_n` the supported connected molecular graphs on `n` **active** atoms.

| event class | map | members |
|---|---|---|
| birth | `X_n → X_{n+1}` | typed atom / structured path insertion |
| death | `X_n → X_{n-1}` | remove eligible atom/substructure without disconnecting |
| same-cardinality | `X_n → X_n` | atom/valence restate · bond-order change · subgraph reroute · cycle closure · cycle opening · ring electronic restate |

**Load-bearing distinction:** the bounded persistent-slot implementation is a *coordinate system*; the
semantic state is the **active graph**. Birth/death therefore genuinely move between molecular spaces of
different cardinality. The process models size, composition, connectivity and topology *jointly*.

**Terminology ban:** never "tokens" for atoms/bonds. Use *atom birth and death*, *creation/destruction of
typed graph entities*, *variable-cardinality molecular generation*, *trans-dimensional molecular rewrites*.

---

## 6. Canonical molecular-successor quotient

`G_y(x) = {a ∈ A(x) : T(x,a) ≅ y}` and `P_θ(y|x) = Σ_{a∈G_y(x)} p_θ(a|x)`.

Removes symmetric/address multiplicity · defines the process on molecules not coordinates · makes
controllers independent of mark encoding · lets primitive and macro routes coexist without double-counting.

**Already instantiated in code:** `_teacher_action_score` returns `torch.logsumexp` over
`graft_successor_groups` for `BondReroute`, with the in-code comment *"Generator Matching supervises the
molecular successor, not one arbitrary atom-slot presentation of the same Graft."* Other families return a
single raw logit. Cite this as the concrete instance.

**Action-refinement invariance (PROPOSED, not yet claimed):** *any refinement or coarsening of internal
marks preserving aggregate successor rates induces the same state-level process and controlled molecular
law.* **Must survive an adversarial proof check before it may be stated** — attack surfaces: hazard
renormalization; merging marks with differing legality masks; composition with the Doob transform;
refinements changing applicable-set cardinality. State conditions, or omit and record why.

---

## 7. Experiment program

| id | experiment | depends on |
|---|---|---|
| A | Unconditional RingCore generation — all-step validity, FCD, precision/recall or coverage, uniqueness, novelty, scaffold novelty, atom-count + cycle-rank distributions, non-memorization | **a separate DE-NOVO checkpoint** (does not exist) |
| B | Learned transport vs uniform legal rewrites — canonical-successor NLL, analogue recovery, productive displacement, path overhead, reversal/cycle rates, endpoint fidelity | scaled edit checkpoint |
| C | Trans-dimensional adaptation — smaller/larger targets, grow-then-delete in ONE trajectory, held-out size bins, no-insert / no-delete ablations | scaled edit checkpoint |
| D | Topological adaptation — cycle creation/opening, fused/spiro/bridged restructuring, cycle-rank change, held-out topology, RingCore vs no-cycle vs finite-catalog | scaled edit checkpoint |
| E | Canonical quotient invariance — symmetric marks, aggregate successor mass, sampling frequencies, action re-encoding invariance, controlled-law invariance | any checkpoint (mechanism test) |
| F | Exact control — **rebuild the enumerable graph on the CURRENT production RingCore operator set** | none (toy) |
| G | Conditional/dynamic design — same-base controller table, Pareto/HV-AUC/IGD+, dynamic switching, Pareto fan, pathwise constraints, cardinality/topology correction, held-out oracle | frozen scaled edit checkpoint |

**Nothing from the conditional program was dropped.** Pareto, dynamic switching, the Pareto fan and
pathwise constraints remain load-bearing — they now sit under *what the new process enables*.

**Causal wording to use throughout:** valid intermediate → meaningful oracle evaluation at every step ·
atom birth/death → size adaptation · cycle ops → topology redesign · canonical kernel →
representation-independent control · remaining-budget value → exact continuation after a preference change ·
legal-support restriction → pathwise constraints · source-agnostic prior → controller swapping without
retraining · complete prefixes → interruption, branching, Pareto fans.

---

## 8. Structure, length, and a principled cut policy

**Section order:** 1 Introduction · 2 Rewrite Generator Matching · 3 COMPOSE (trans-dimensional
instantiation) · 4 Canonical successor quotient · 5 Finite-horizon control · 6 Goal-conditioned controller ·
7 Experiments (generation → transport → cardinality/topology → exactness → conditional design).

**Budget (ICLR 9 pages main text):** intro ~1 · RGM ~1–1.25 · COMPOSE ~1.5 · quotient ~0.75 · control ~1 ·
controller ~0.5 · experiments ~2.5–3 · related work ~0.5–0.75 · discussion/conclusion ~0.5.
**Abstract ~200 words** (hard ceiling 250; measured at 448 on 2026-07-28), **at most one number**.

**The cut principle:** *main text carries the causal chain; everything else supports it from the appendix.*
A sentence that would not change a reviewer's assessment belongs in the appendix or nowhere.

| default to APPENDIX | keep in MAIN TEXT |
|---|---|
| full operator ontology tables | the causal chain |
| all proofs (sketches only in main) | trans-dimensional formalism `X = ⋃ X_n` |
| compilation algorithms | quotient definition + why it matters |
| complete experimental protocol | control theorem *statements* with hypotheses inline |
| per-task Pareto fronts, ablation grids | the 2–3 ablations that carry claims |
| checkpoint lineage, scheduler config | honesty clauses (validity ≠ synthesizability; no exactness at scale) |
| failure modes, broader impact, LLM statement | contributions, novelty boundaries |

Merge candidates: "Background and Positioning" likely folds into Introduction + Related Work.

**Rigor requirements:** hypotheses stated *inline* in every theorem (finite horizon, exact desirability,
rule-closed constraints, bounded slots) · notation introduced once and never reused
(`A(x)`, `T(x,a)`, `G_y(x)`, `P_θ(y|x)`, `Λ_θ`, `p_θ(a|x,t)`, `X_n`) · every theorem also stated in words.

---

## 9. Current empirical reality (constrains every claim)

**Program state: `IMPLEMENTATION_GO / SCIENTIFIC_PRIOR_NO_GO_DATA_STARVATION`**
(`diagnostics/coherence/program_state.json`).

Every RingCore checkpoint to date trained on **6,922 unique records** (200 corruption source molecules +
5,000 MMP traces — and the MMP loader takes a *contiguous prefix*, not a sample) against 363,456 available
pool rows and 466,483 eligible corpus molecules: ~28 passes, with the textbook overfitting signature
(train −24%, held-out +8%). All are labeled **`DATA_STARVED_BASELINE`** and barred from final scientific
status.

**Consequences for the manuscript:**
- Every unconditional / transport / conditional number stays `\XXX`. Quote no RingCore result.
- Surviving, quotable qualitatively: hard rollout gates, numerical stability, resume determinism,
  operator-support completeness (14/14 topology classes).
- The editing checkpoint has `denovo_weight=0` **and** an uncalibrated hazard (~2× teacher), so it
  **cannot** carry the timed-CTMC generation claim. Experiment A needs a separate de-novo checkpoint that
  does not yet exist. Keep the edit and de-novo checkpoints conceptually distinct in the text.
- Six previously-published figures were retracted as unsourced (see the retraction block in `numbers.tex`).

---

## 10. Reviewer stress tests

| attack | response |
|---|---|
| "Generator Matching with handcrafted edits." | It is a new Markov-process design: show competitive generation, learned-vs-uniform transport, trans-dimensional/topological ablations, state-level control. |
| "Insertion/deletion already exists." | Credit Jump Diffusion and GrIDDD; claim the richer typed rewrite language, molecular closure, topology change, generator-matched event law, quotient kernel, dynamic control. |
| "Validity by construction is tautological." | Agree it is a support property; show why it matters — pathwise constraints, mid-trajectory intervention, branching, zero wasted invalid oracle calls. |
| "Doob is classical." | The contribution is the learned chemistry-native kernel on which exact, retargetable, support-preserving control becomes operationally meaningful. |
| "Too many ideas." | One causal chain; each experiment tests one link. |
| "Only molecular." | RGM is stated abstractly; broader structured-object reach is outlook only — do not dilute the molecular package. |

---

## 11. Do not change the production model for prose

This is a manuscript, experiment-plan and claim-hierarchy update. Any code change must correspond to a
genuinely missing capability or experiment and **must be reported before implementation**. Keep edit and
de-novo regimes distinct, exact and learned controllers distinct, macros optional, and the RingCore
production contract authoritative.
