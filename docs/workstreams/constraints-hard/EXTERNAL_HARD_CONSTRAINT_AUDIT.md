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

But three findings block the head-to-head, and the third is the one that matters:

1. **No official CDD code exists** (searched; none found). A rerun is impossible.
2. **The task differs structurally.** CDD's molecular experiment is **de novo
   SMILES generation on QM9**; COMPOSE's production process is
   **source-conditioned editing**. §4.
3. **CDD's operative constraint is not the published predicate.** During
   sampling it enforces a **differentiable ML surrogate** of SA, not RDKit's
   `sascorer`; the reported SA appears to be computed post hoc. Its measured
   satisfaction is **21.3% at τ = 3.0** and **63.9% at τ = 4.5** — it is not a
   hard constraint at all. §2.1.

So the correct classification, in the lead's own vocabulary, is:

> **An external hard-constraint *competence* benchmark — contextual,
> non-head-to-head. Not the same experiment.**

---

## 1. Provenance

| method | venue / year | id | official code | checkpoints | license |
|---|---|---|---|---|---|
| **CDD** | **UNVERIFIED** — arXiv comment indicates an **ICML 2025** submission; NeurIPS 2025 acceptance **could not be confirmed** | arXiv:2503.09790 | **none found** | no | n/a |
| **PRODIGY** | ICML 2024 | "Diffuse, Sample, Project: Plug-and-Play Controllable Graph Generation" | yes | UNVERIFIED | UNVERIFIED |
| **ConStruct** | NeurIPS 2024 Poster | arXiv:2406.17341 | yes (no checkpoints) | no | MIT |

> ⚠️ **The charter described CDD as "NeurIPS 2025". That could not be verified.**
> The arXiv HTML carries a comment consistent with an ICML 2025 submission. It
> may have been accepted at NeurIPS 2025 subsequently — I could not confirm
> either way. **Do not cite a venue for CDD until someone checks the
> proceedings.** A cited-but-absent source is worse than a wrong number.

## 2. Constraint mechanism — the structural comparison

| | **CDD** | **PRODIGY** | **ConStruct** | **COMPOSE** |
|---|---|---|---|---|
| where enforced | projection inside discrete-diffusion sampling | projection each diffusion step | edge-absorbing process + projector | filter on the enumerated legal successor fiber |
| **requirement on the constraint** | **differentiable / relaxable** | admits a **projection operator**; global-aggregate properties | **edge-deletion invariant** | **any decidable boolean predicate on a complete molecule** |
| hard guarantee? | **NO — 21.3%–63.9%** | approximate | **yes, 100%** on its families | **yes, by construction** |
| intermediate states | noisy token sequences, not molecules | partially noised graphs | graphs satisfying the property, not necessarily valid molecules | **complete valid molecules at every step** |
| training-free at inference | yes | yes | requires its own trained model | yes |
| source-conditioned | no | no | no | **yes** |

### 2.1 CDD's constraint is soft, and its operative predicate is a surrogate

This is the audit's most consequential finding and it must not be papered over.

- The sampler enforces a **differentiable ML surrogate** of synthetic
  accessibility, because CDD's method *requires* differentiability. RDKit's
  `sascorer` is a non-differentiable fragment-contribution score and cannot be
  used inside the projection.
- Reported SA satisfaction: **21.3% at τ = 3.0**, **63.9% at τ = 4.5**.
- The two threshold values **τ = 3.0** and **τ = 4.5** were confirmed from the
  results table. The intermediate values **τ = 3.5 and 4.0 are UNVERIFIED** —
  the charter lists all four; I could confirm only the endpoints.

**Consequence.** "COMPOSE 100% vs CDD 21.3%" is comparing a construction
guarantee against an empirical rate, and is **barred as a headline** by the
project's sign-guarantee rule — the same rule that has caught seven prior
instances. COMPOSE's 100% is definitional and is recorded as a construction
check without a denominator.

What *is* admissible and interesting: **at matched satisfaction, what does the
constraint cost in QED / novelty / diversity?** That comparison has a free sign.
It is also the comparison CDD's own numbers cannot support, because there is no
released code to run at matched satisfaction.

### 2.2 PRODIGY and ConStruct cannot express our constraint class

- **PRODIGY**'s constraints are **global / aggregate graph properties** — edge
  count, degree, triangle count, valency-type validity. It **cannot express
  "this specific labeled subgraph must be present"**. Its molecular constraint
  (valency) is a *validity* condition that COMPOSE's executor already enforces
  natively in `is_valid_state`, so there is nothing to compare.
- **ConStruct** requires constraints to be **edge-deletion invariant** (closed
  under edge removal). Subgraph *presence* is not — deleting an edge of the
  protected core destroys it. This **structurally rules out** a
  "preserve this subgraph" projector in the released design, which is why the
  shipped projectors are only `planar` / `tree` / `lobster`. It ships **no
  molecular dataset config**; molecular work appears only in an appendix.

**ConStruct is the closest methodological lineage and the furthest from a shared
number.** Cite it for the "every intermediate state remains admissible" idea; do
not attempt a numerical comparison.

## 3. The cross-cutting finding — this is the methods-paper point

All three published mechanisms constrain the **constraint class** in order to
make projection tractable:

- CDD needs **differentiability**;
- PRODIGY needs a **projection operator** and targets aggregate properties;
- ConStruct needs **edge-deletion invariance**.

COMPOSE needs none of these. Because it enumerates an **exact executable
successor fiber of complete molecules**, `C` may be *any decidable predicate* —
a threshold on a non-differentiable score, a substructure match, a conjunction
of both, changed at inference time on the state already reached.

**And the honest counterweight, which must appear in the same paragraph:** that
generality is bought with exact fiber enumeration, which is expensive — the
legal fiber is ~600-wide, and enumerating it at every state is a cost the
projection methods do not pay. They scale to graph sizes and settings where
enumerating an exact successor support is infeasible.

> The comparison is **constraint expressiveness vs enumeration cost**, not
> "COMPOSE wins". Stated that way it is a genuine methodological contribution and
> survives review; stated as a win it invites the reviewer to find §2.1.

## 4. The task-semantics verdict, per method

The lead asked this be answered directly and not papered over.

| method | native task | same experiment as COMPOSE? | verdict |
|---|---|---|---|
| **CDD** | de novo molecular SMILES generation, QM9, token-level discrete diffusion | **no** | **contextual, non-head-to-head** |
| **PRODIGY** | de novo graph generation with aggregate constraints | **no** | **contextual**; constraint class disjoint |
| **ConStruct** | de novo graph generation, structural families | **no** | **cite as lineage; no number** |

### Why CDD is not a head-to-head, specifically

Five independent mismatches, any one of which is disqualifying:

1. **Task.** De novo generation vs source-conditioned editing. The source
   anchors COMPOSE's output distribution; a QED figure from unconditional
   generation and one from editing a given lead are not the same estimand.
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
5. **Unverifiable protocol.** No code ⇒ sample budget, seeds, variance
   reporting, and the canonicalization/dedup convention cannot be checked.
   Published numbers may be **cited as reported and labelled that way**; they may
   never be presented as a rerun.

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

> Recent work imposes hard constraints on pretrained discrete generative
> processes by projection during sampling: CDD inserts a differentiable
> constrained projection into discrete diffusion, PRODIGY projects each diffusion
> step onto a user-specified graph-constraint set, and ConStruct guarantees
> structural graph properties throughout the trajectory via an edge-absorbing
> process. Each buys tractability by restricting the admissible constraint
> class — to differentiable, projectable, or edge-deletion-invariant
> constraints respectively. COMPOSE instead restricts an exact executable
> successor support over complete molecules, so the constraint may be any
> decidable predicate and may be imposed or changed at inference time on the
> state already reached; the cost is exact fiber enumeration, which the
> projection methods avoid.

**If any CDD number is quoted at all:** cite it **as reported**, name the
threshold, state that the enforced constraint is a differentiable surrogate
rather than RDKit SA, and give the satisfaction rate alongside it. A CDD SA
number quoted without its satisfaction rate misrepresents the method.

## 6. Unverified — flagged, not filled in

| item | status |
|---|---|
| CDD venue (NeurIPS 2025?) | **UNVERIFIED** — arXiv comment suggests ICML 2025 submission |
| CDD thresholds τ = 3.5 and 4.0 | **UNVERIFIED** — only 3.0 and 4.5 confirmed |
| CDD full baseline table numbers (AR, MDLM/UDLM, CFG/CBG) | **partially extracted; treat as UNVERIFIED** |
| CDD sample budget, seeds, variance reporting | **UNVERIFIED** — no code |
| CDD canonicalization / dedup convention | **UNVERIFIED** |
| PRODIGY license and checkpoints | **UNVERIFIED** |
| whether any of the three shares a benchmark with source-conditioned molecular editing | **none found** |

**Before any number from this audit is quoted head-to-head**, the CDD venue and
the full threshold set must be checked against the proceedings by someone with
the PDF in hand.
