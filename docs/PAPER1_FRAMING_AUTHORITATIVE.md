> **ARCHIVED — DO NOT USE.**
> Superseded by [`docs/EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md), which is the only
> current plan. This document describes an earlier framing of the project and its
> experiment list must not be executed or cited. Kept on disk because other files
> still link to it; read it as history, not as instruction.

# Paper 1 — authoritative framing record

**Status: AUTHORITATIVE.** Supersedes the control-first framing in earlier drafts and the RGM-first
interim reframe. Derived from the owner's transmission of the amended three-paper PhD strategy document
(2026-07-28). The named source file `Generative_Design_PhD_Program_and_Three_Paper_Figure_Plans_
Original_Format_Latest_Content_ICLR_Framing_Updated.docx` is **not on disk** (closest present:
`~/Downloads/..._Original_Format_Latest_Content.docx`); the owner's verbatim transmission is the record.
Reconciled on 2026-07-30 with the later authoritative COMPOSE handoff, the production-generative
research contract in `AGENTS.md`, the editing-V2 contracts, the current comparator registry, and the
evidence ledger. Those sources narrow empirical and implementation claims without changing the thesis
or settled title below.

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

**Tagline:** *Every non-null step is a molecule; every transition is an executable rewrite.*

**Title — SETTLED (owner, 2026-07-28):**

> **COMPOSE: Generator Matching over Stochastic Rewrites for Trans-Dimensional Molecular Generation and Control**

COMPOSE is retained because it is already the recognizable name of the molecular instantiation and is used
via `\method` throughout; dropping it would make the paper read as an unnamed framework paper. "Generator Matching over Stochastic Rewrites" names the framework the way the paper does
throughout, preserves framework-first positioning, and carries the three things a reviewer must remember: Generator Matching over stochastic
rewrites · trans-dimensional molecular generation · control. The tagline is a **tagline
only** and must never appear as the title.

**Naming hierarchy:** Rewrite Generator Matching (RGM) is the general framework. COMPOSE is the
molecular instantiation and expands once, at first use, as **CO**mpositional **M**olecular **P**rocess
**O**ver **S**tochastic **E**xecutable rewrites. Elsewhere, use COMPOSE as the system name without
re-expanding it. Do not call COMPOSE the generic framework, and do not change the settled title to force
the expansion into the title.

---

## 2. Three contributions (in this order)

1. **Rewrite Generator Matching** — generative modeling over a state-dependent executable rewrite
   language: learn a total event hazard and the identity/location of domain-meaningful events, rather
   than reversing a generic tensor corruption.
2. **COMPOSE + the canonical successor quotient** — a trans-dimensional molecular process whose events
   create, delete, relabel, reconnect and cyclize structure while every non-null committed state stays
   inside a declared molecular graph space; quotienting makes the state-level process independent of action
   encoding.
3. **Generation and control** — the framework supports unconditional generation *and*
   source-conditioned editing through separate checkpoints and inference contracts; finite-horizon control
   of a molecular successor kernel gives exact bounded-state conditioning, dynamic retargeting, Pareto
   branching, and pathwise constraints.

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

`X = ⨆_{n=0}^{N} X_n`, with `X_0 = {∅}` a distinguished reversible null source and, for `n ≥ 1`,
`X_n` the supported connected molecular graphs on `n` **active** atoms. The null state is not a molecule.

| event class | map | members |
|---|---|---|
| birth | `X_n → X_{n+1}` | root birth from null or one-neighbor connected typed-atom insertion |
| death | `X_n → X_{n-1}` | remove an eligible atom with a connected non-null remainder, or use the registered one-atom-to-null transition |
| same-cardinality | `X_n → X_n` | atom/valence restate · bond-order change · subgraph reroute · cycle closure · cycle opening · ring electronic restate |

**Load-bearing distinction:** the bounded persistent-slot implementation is a *coordinate system*; the
semantic state is the **active graph**. Birth/death therefore genuinely move between molecular spaces of
different cardinality. The process models size, composition, connectivity and topology *jointly*. The
current birth factorization supports zero or one existing neighbor, not arbitrary multi-attachment
insertion, and no one-step inverse is claimed for every deletion.

**Terminology ban:** never "tokens" for atoms/bonds. Use *atom birth and death*, *creation/destruction of
typed graph entities*, *variable-cardinality molecular generation*, *trans-dimensional molecular rewrites*.

---

## 6. Canonical molecular-successor quotient

`G_y(x) = {a ∈ A(x) : T(x,a) ≅ y}` and `P_θ(y|x) = Σ_{a∈G_y(x)} p_θ(a|x)`.

Removes symmetric/address multiplicity · defines the process on molecules not coordinates · makes
controllers independent of mark encoding · lets primitive and macro routes coexist without double-counting.

**Production boundary:** the molecular law is defined by enumerating production marks, executing each
mark, canonicalizing the successor, and summing the complete successor fiber. The authoritative segmented
implementation evaluates that definition efficiently; a fresh dictionary grouping is a bounded test
oracle, not an alternative scientific kernel. Do not use the historical Graft-only alias aggregation as
evidence that the completed 16k run optimized canonical-successor likelihood for every family.

**Action-refinement invariance (PROVED, with explicit conditions):** a refinement or coarsening leaves the
induced molecular process and a successor-level controlled law unchanged only when:

1. every refined mark remains inside the same canonical-successor fiber;
2. aggregate off-diagonal successor **rate**, not merely the number of marks, is preserved;
3. canonical self/virtual mass keeps the same disposition under the declared productive-jump convention;
4. legality and execution remain valid for every refined mark; and
5. multi-step quotient semantics use representative-consistent aggregate rows or a fixed deterministic
   canonical section.

Under these conditions the raw pushforward, productive molecular kernel, state-level process, backward
values, and fiber-constant successor controller are unchanged. The theorem does **not** make selected-mark
training invariant. It also excludes mark-level top-`k`, nucleus, power, family, and per-alias tilts. For
example, replacing one unit-mass mark by four aliases of mass `1/4` changes a mark-level power tilt unless
the exponent is one. Keep this negative result beside the theorem.

---

## 7. Experiment program

| manuscript / registry id | experiment and registered comparison | depends on |
|---|---|---|
| A / E1 | Unconditional RingCore generation — all-step validity, FCD, held-out chemical-space recall, ring/topology distance, uniqueness, novelty, scaffold/size distributions, and non-memorization | **a separate broad-organic timed-CTMC checkpoint** (does not exist) |
| B / E2 | Learned transport — compare the learned unguided prior with uniform canonical-successor transport **and** the state-independent empirical-family law on identical executor support, sources, seeds, and edit budget | frozen editing checkpoint |
| C / E3 | Trans-dimensional adaptation — full basis vs no insertion, no deletion, fixed cardinality, and no `bond_reroute`; include smaller/larger targets, grow-then-shrink, held-out size bins, and fixed-cardinality-impossible tasks | frozen editing checkpoint |
| D / E4 | Topological adaptation — full basis vs no cycle operations and no `bond_reroute`; the finite-catalog topology editor is `PROSPECTIVE_UNAVAILABLE` and runs only if its preregistered implementation/support gate passes; disabled `ring_system_grow` is not a substitute | frozen editing checkpoint |
| E / E5 | Canonical quotient invariance — symmetric marks, aggregate successor mass, slot relabeling, within-fiber refinement, sampled frequencies, controlled-law invariance, and the required negative mark-level counterexample | any compatible checkpoint for projection; mechanism checks use the current production registry |
| F / E6 | Exact control — rebuild and audit the enumerable graph on the **current** production executor and operator registry; verify row-stochastic kernels, exact backward values, unreachable rows, terminal tilt, and dynamic continuation | no selected molecular checkpoint required for the algebraic base test |
| G / E7 | Conditional/dynamic design — same-base unguided, endpoint reranking, greedy, local Boltzmann, static scalarization, MOG-DFM-style, SMC/Feynman--Kac, learned Doob/value, and NSGA-II-over-COMPOSE-successors arms; MOEA/D and AReUReDi-style adaptations are conditional; compare the four registered dynamic arms and retain the mechanistic switch controls | frozen editing checkpoint |

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
This page budget applies to the conference-formatted companion. The preferred `paper_arxiv/` package keeps
the rigorous long-form development and appendix, while preserving the same claim hierarchy and evidence
boundaries.

**The cut principle:** *main text carries the causal chain; everything else supports it from the appendix.*
A sentence that would not change a reviewer's assessment belongs in the appendix or nowhere.

| default to APPENDIX | keep in MAIN TEXT |
|---|---|
| full operator ontology tables | the causal chain |
| all proofs (sketches only in main) | trans-dimensional formalism `X = ⨆_{n=0}^{N} X_n`, with null qualified |
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

### Completed scientific diagnostic run

The first scientifically valid large-data editing run completed **16,000 steps** on **661,105 admitted
training traces** and wrote all **32 scheduled snapshots**. It proves that the packed large-corpus training
path is operational. It does **not** select a production model:

- the run optimized the historical mark-level Generator-Matching objective, not the editing-V2
  productive canonical-successor objective;
- ring closing learned while ring opening remained severely weak in mark-level diagnostics;
- the inherited Graft / `bond_reroute` capability collapsed during training;
- the validation-only canonical-successor leaderboard, family trajectories, support forensics, and
  capability safeguards have not selected a snapshot; and
- the already inspected final-test aggregates are barred from architecture, threshold, recipe, or
  checkpoint selection. A repaired model needs a new sealed final holdout or an external final
  evaluation.

Treat the run as a **measured diagnostic baseline**, not as wasted compute and not as a paper model. Do not
select step 8,500, the final step, the minimum raw Generator-Matching loss, or the maximum mark-level
family accuracy by default.

### Editing-V2 corpus and training status

The editing-V2 corpus contract declares five evidence lanes:

1. `observed_local_analogue`;
2. `operator_aware_real_endpoint`;
3. `linker_positional_topology_analogue`;
4. `real_endpoint_multistep_path`; and
5. `reversible_synthetic_walk`.

`real_endpoint_multistep_path` means a compiler-generated path between real endpoints with a preserved
observed or inferred pair-relationship label. There is currently **no genuine observed-series action
provenance**, so neither that lane nor its actions or intermediates may be called an observed series path.
Endpoint evidence, pair-relationship evidence, path origin, intermediate-state origin, and action-sequence
origin remain separate fields. A real endpoint does not make compiler-generated actions or intermediates
observed.

The legacy corruption, cycle-operation, and exact-state MMP packs remain distinguishable input sources.
The MMP pack is one source lane, not the complete editing-V2 capability curriculum. Sampling must follow
the frozen hierarchy `data_lane → series/scaffold/source group → semantic capability cell → endpoint pair
or path → progress state`; raw pair-uniform sampling is forbidden.

Preserve distinct training, validation, controller-validation, and final-test roles. Validation selects
the editing checkpoint under the frozen canonical-successor rule; controller-validation selects controller
settings. The sealed final test does neither.

The corpus contract remains **`DESIGN_NOT_TRAINING_AUTHORIZED`**. Long training is not authorized. The
permitted sequence is whole-trace Active8 admission, Gate 0 structural evidence, T1 successor-level
micro-overfit, P50, P500, and P2000, with each later stage blocked until the exact current decision artifact
passes. Scratch initialization is allowed. Warm-starting is optional and requires parity and retention
evidence.

### Consequences for the manuscript

- Every learned unconditional, transport, adaptation, Pareto, dynamic, and calibration result remains a
  keyed `\resultpending{...}` / rendered `XXX` until its provenance-validated artifact exists.
- The completed editing run has zero de-novo mixture weight and cannot carry experiment A / E1. No
  production de-novo checkpoint exists; E1 needs a separate broad-organic timed-CTMC checkpoint, source
  distribution, hazard contract, and evaluation.
- The bounded `carbon_6_slots` graph remains an algebraic exact-control verification slice, not production
  chemistry. Its current graph and solver artifacts must pass their own registry-bound audit before
  numeric exactness claims are filled.
- Earlier data-starved checkpoints, pre-RingCore exactness artifacts, and unsourced figures remain barred.
- Negative findings, failed gates, unavailable comparator arms, unreachable control rows, and abstentions
  are reported rather than hidden.

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

## 11. Document authority map — where the granular material lives

**This document is the single authority on thesis, positioning, title, contribution order, novelty
claims, section order, and length policy.** It is deliberately NOT a replacement for the granular
operational documents. Nothing was deleted; each superseded doc carries a banner naming exactly what in it
is superseded (thesis only) and what remains binding. Use this map to find detail rather than
re-deriving it.

| I need… | Go to | Status |
|---|---|---|
| **the preferred live long-form manuscript** | `paper_arxiv/main.tex`, its `sections/`, `SCIENTIFIC_TRACEABILITY.md`, and `COMPLETION_PLAN.md` | **preferred manuscript package**; Homological-Flows-derived arXiv format |
| **the conference-formatted companion** | `paper_iclr_stochastic_rewriting/main.tex` | maintained companion, not a separate scientific authority; semantic changes must be synchronized |
| **the current thesis handoff and precedence record** | `docs/HANDOFF_COMPOSE_TRACEABILITY_2026-07-29.md`, with `docs/HANDOFF_RINGCORE_V1_POSTRUN.md` only for non-conflicting historical measurements | current traceability plus historical diagnostic record |
| **claim status and evidence boundaries** | `docs/CLAIM_LEDGER.md` plus provenance-complete machine-readable artifacts | **current evidence authority** |
| **the full comparator plan** | `configs/comparator_registry_v1.json` | **current comparator authority**, status `PREIMPLEMENTATION`; a declared arm is not necessarily runnable |
| **experiment objectives, tasks, budgets, and outputs** | `configs/experiment_registry.yaml`, loaded through its validator | current protocol authority |
| **editing-V2 corpus roles and evidence profiles** | `configs/editing_corpus_v2_contract.json` | current design authority, status `DESIGN_NOT_TRAINING_AUTHORIZED` |
| **operator decision and bounded Active8 support** | `configs/ring_operator_decision_v1.json`, current Gate 0 and T1 contracts, and their provenance-bound diagnostics | development support only; not final production support |
| **training-stage authorization** | `configs/editing_training_v2_gate.json`, current Gate 0/T1/P50 contracts, and their exact decision artifacts | filenames alone do not authorize a run; current long training remains unauthorized |
| **exact-control implementation plan and benchmark identity** | `docs/EXPERIMENT_INFRASTRUCTURE_PLAN.md` and current registry-bound graph-audit artifacts | carbon-only bounded verification, not representative chemistry |
| **baselines, scientific rationale, and controller context** | `docs/PAPER_MASTER_PLAN.md`, `docs/RELATED_WORK_MATRIX.md`, `docs/EDITING_MODEL_FIRST_PRINCIPLES_SPEC.md` | supporting detail; current registries override stale arm lists |
| **historical coherence and deviations** | `docs/PROGRAM_COHERENCE_REPORT.md`, `docs/DEVIATION_REGISTER.md`, `docs/LEGACY_QUARANTINE.md` | historical at their declared revisions unless revalidated |
| **completed-run identity and snapshot selection readiness** | frozen 16k run manifest, `configs/ringcore_v1_successor_leaderboard_v1.json`, and `diagnostics/coherence/ringcore_v1_successor_leaderboard_readiness_2026-07-30.json` | diagnostic evidence and readiness only; no selected checkpoint |
| **broad-organic lead support census** | `diagnostics/composition/benchmark_lead_scope_coverage.json` | provenance-complete computed support census, not a performance result |

**Known traps:**

- `paper_iclr_control_substrate/main.tex` is a 2026-07-22 fork with unrelated uncommitted prose work and
  is not the preferred manuscript. Do not overwrite it or use its generated files as authority.
- `diagnostics/coherence/program_state.json` records an earlier milestone and is not the current global
  program-state authority. Current self-hashed contracts and decision artifacts govern.
- A configuration filename, green unit test, or hash-valid `NO_GO` artifact does not authorize training.

**Precedence rule:** this framing record governs thesis, positioning, and claim hierarchy. `AGENTS.md`,
current self-hashed contracts, registries, and production code govern implementation and scientific
authorization. If manuscript prose disagrees with the current system contract or evidence ledger, the prose
is a bug and must be reconciled explicitly; do not choose the convenient document silently.

---

## 12. Do not change the production model for prose

This is a manuscript, experiment-plan and claim-hierarchy update. It does not authorize model training,
data generation, or a later pilot stage. Any code change must correspond to a genuinely missing capability
or experiment and **must be reported before implementation**. Keep edit and de-novo regimes distinct,
exact and learned controllers distinct, macros optional, and the current production contracts
authoritative. Long training remains blocked until whole-trace Active8 admission, Gate 0, T1, P50, P500,
and P2000 pass in order under their exact current identities.
