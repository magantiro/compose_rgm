# Baseline philosophy for a methods paper, and four main-lane decisions

**Supersedes the ad-hoc baseline selection that was accumulating across lanes.**
Binds Lanes 2, 3, 5 and 6.

## The mistake this corrects

We were drifting toward *find the strongest method for each narrow downstream
task*. For a methods paper the question is:

> **What is the closest alternative way to BUILD or CONTROL the generative
> process, and what does COMPOSE's representation buy over it?**

The analogy is how a method paper positions itself against its own family rather
than against every model that happens to touch the same data — Expanding Flow
Maps motivates itself by the fixed-canvas limitation and recovers fixed-dimensional
flow maps as the special case where the expand operator is the identity. CDD's
molecular hard-constraint experiment compares against autoregressive generation,
masked/uniform discrete diffusion and guidance variants — **not** against
whichever medicinal-chemistry model happens to score highest on synthetic
accessibility.

COMPOSE's analogous statement:

> Existing molecular generators and editors can generate or optimize structures;
> constrained-generation methods can project or guide a sampler. **COMPOSE
> instead defines the design process directly on an exact executable successor
> graph, so legality and hard admissibility are enforced at the state-transition
> level while the learned dynamics remain reusable.**

## Three rings

| ring | question it answers | members |
|---|---|---|
| **1 — same methodological realm** | why COMPOSE rather than another contemporary generative/editing abstraction? | **DDSBM** (graph CTMC, source-conditioned transformation); **Edit Flows** (edit-based CTMC lineage — *related work only*); **GraphXForm**; **InVirtuoGen** |
| **2 — same hard-constraint problem** | given a pretrained process, how should hard constraints be imposed? | **CDD** (NeurIPS 2025), **PRODIGY** (ICML 2024), **ConStruct** (NeurIPS 2024); **Prompt-MolOpt^P** as an applied sanity check |
| **3 — COMPOSE causal controls** | **what does the COMPOSE factorization itself buy?** Possibly more important than either external ring | hard mask vs post-hoc; frozen `R_θ` vs empirical family; closed-loop vs generate-and-rank; continuation vs restart; endpoint vs trajectory constraint |

**Do not force every baseline into every experiment.** That was the error.

### Placements that were being got wrong

- **Edit Flows** is prominent conceptual lineage — an edit-based CTMC with
  insert/delete/substitute — but its published state space is **variable-length
  sequences**. A homemade molecular-graph port would require solving the
  support/executor problem that is *part of COMPOSE's own contribution*: we would
  be inventing a competitor for ourselves. **Related work, never a numerical
  baseline.**
- **DDSBM** belongs, but for the **general source-conditioned editing** question,
  not because it has hard constraints. Its published contribution is graph
  transport / minimal transformation. If it cannot natively say "this labeled
  core must remain present at every state", we do not invent that mechanism for
  it.
- **MolEditRL** and **Prompt-MolOpt^P** answer *"is COMPOSE a competent molecular
  editor?"* — the general editing table. **They do not define the intellectual
  baseline for exact hard-support control** and must not drive the methods
  narrative.
- **ConStruct** is methodologically beautiful context for the pathwise claim
  (constraints guaranteed throughout the trajectory), but its native tasks are
  graph-structural. Cite prominently; do not port.

---

## Decision 1 — C0 closed permanently; all claim-bearing code pinned to B=1

The mixed `margin_cache` is a real software and reproducibility defect
(`docs/ORACLE_BATCH_INVARIANCE_DEFECT.md`), but it invalidates **no live
scientific claim**: the decision-bearing comparisons index one internally
consistent batched array, C0 was negative and superseded, and no headline depends
on it.

- **Do not requalify C0.**
- **Do not reopen** exact-target, retargeting, Stage B or Pareto.
- **PINNED RULE: all claim-bearing COMPOSE decision code uses the existing legacy
  `B=1` oracle path, consistently.** No mixed batch/scalar decision semantics.
- **Do not adopt the batch-invariant reduction mid-paper** — it would re-baseline
  everything for ~7% of wall time.

## Decision 2 — redirect Lane 5 now; do not wait for P3/P4

Waiting would be *dangerous*, not merely slow: **Panel B's benchmark choice must
be independent of whether P3 makes COMPOSE look strong or weak.**

Lane 5 stops dependency archaeology and performs a **published-protocol
alignment audit**. Candidate **GSK3β / JNK3**, chosen because it is externally
motivated and shared by the relevant published literature — *not* because it
suits COMPOSE.

Freeze only if the audit confirms alignment on **all** of: oracle
definitions/hashes or faithful equivalents; budget; HV reference point; top-K and
preference conventions; validity/canonicalization; seed and reporting rules.

- **Aligned** → run **COMPOSE only**, cite external numbers **as reported**.
- **Not aligned** → label published values **contextual, non-head-to-head**.
- **Never** spend 10 V100-hours merely to say we reran HN-GFN.

### Panel A and Panel B have different jobs

| panel | task | answers |
|---|---|---|
| **A** | source-conditioned, exact realized state, fixed `R_θ`; P5/P6, repaired P3/P4, reference-law ablation | *why COMPOSE's control/process structure matters* |
| **B** | established published global benchmark, published external numbers where alignment is exact | *is COMPOSE also a credible ordinary multiobjective optimizer?* |

P3/P4 may change **how much emphasis** Panel B needs. It must not change **which
benchmark** Panel B uses. Global fronts never sit beside per-source fronts.

## Decision 3 — honor the Bemis–Murcko stop; fold the internal contrast into Lane 2

The full BM core is a **median 78.95% of heavy atoms**, leaves ~**5** editable
atoms, and only **22.03%** of sources pass the preregistered eligibility. That
triggered our own stop rule.

**No "smaller scaffold", no pharmacophore substitute, no hand-tuned protected
core.** Any smaller core would be *chosen* to leave enough room — selection on
the outcome, and obvious task shopping after learning the full scaffold is too
restrictive. **No new constraint may be invented from our own failed result.**

**Lane 6's scope narrows to auditing externally defined hard-constraint
benchmarks only** (CDD / PRODIGY / ConStruct). It designs nothing internal.

**The internal contrast folds into Lane 2's existing harness**, which already has
the right causal structure — endpoint/post-hoc versus masked support, with a
free-sign feasible-yield contrast. Its n=6 signal (endpoint-only returned nothing
on 2/6 sources, masked arms succeeded on 6/6) is **developmental only**, and the
claim it could license is *"constraining during search can improve feasible-return
yield"* — **never** *"100% scaffold preservation"*, which is true by construction.

### The three-arm design that moves to Lane 2 — recorded so it is not lost

Identical `R_θ`, controller, source panel, objective, horizon and resources:

1. **post-hoc / endpoint filtering** — normal control, filter violations after.
2. **soft constraint guidance** — the **same** controller, constraint expressed
   as a penalty or soft desirability rather than removing infeasible successors.
3. **hard-support masking** — `F(x) → F_C(x)` before control.

**Arm 2 is the sharpest and is new.** It isolates the actual methodological
distinction — *soft guidance* versus *exact feasible support* — and answers "why
support restriction rather than ordinary guidance?" It must not be dropped in the
handoff, and its penalty must be specified so it does not become a free parameter
tuned until hard-masking wins.

**Load-bearing outcomes: feasible objective improvement, feasible-return rate,
and retained support `|F_C(x)|/|F(x)|`.** Not the constraint-satisfaction rate —
hard masking gets 100% by construction and that stays barred. The support-cost
axis tells a reviewer whether the guarantee is cheap or whether we are simply
preventing the model from acting.

The claim, which can genuinely fail: *at a fixed resource budget, exact support
restriction yields more and better feasible improved molecules than post-hoc
filtering or soft guidance.*

## Decision 4 — the source-000 probe finishes exactly as frozen

No changes because it is slow. If it reaches 368, run the other eleven in
parallel under their committed targets. If `MATCHING_UNREACHABLE`, keep that
outcome and follow the frozen fallback and resource-frontier logic. **Do not
alter caching or generate-and-rank to manufacture a matched point.** Its verdict
determines matcher feasibility only — never scientific interpretation.

---

## The hard-constraint figure this licenses

**A. Method cartoon** — one frozen `F(x)`, three strategies: post-hoc, soft
guidance, **hard support**.
**B. Feasible optimization** — objective improvement against constraint
violation / feasible-output yield.
**C. Support cost** — `|F_C(x)|/|F(x)|`, so the guarantee's price is visible.
**D. True pathwise example** — the cLogP corridor: ✓ → ✗ → ✗ → ✓ against a
trajectory that never leaves.

## Three forms of thrashing this prevents

1. **Oracle re-baselining** — adopting the batch-invariant reduction mid-paper.
2. **Benchmark shopping** — choosing Panel B after seeing P3/P4.
3. **Scaffold rescue** — inventing a smaller core after the BM stop.
