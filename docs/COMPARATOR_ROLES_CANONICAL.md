# Canonical comparator policy — comparators have ROLES, not rankings

**Project-wide. Effective immediately. Supersedes the comparator-selection
guidance in `docs/BASELINE_PHILOSOPHY_AND_MAIN_LANE_DECISIONS.md` (its four
main-lane decisions and its stop rules remain in force).** Binds Lanes 2, 3, 5, 6
and the main lane.

## The rule

> **Framework-neighbor comparisons establish methodological novelty; matched
> internal controls establish causality; task-specialist baselines establish
> practical competence. Do not let task-SOTA methods substitute for the closest
> generative-framework comparison, and do not force same-lab methods into
> numerical comparisons unless they natively instantiate the same scientific
> question.**

**Comparator selection is framework-first.** A method does not become a primary
baseline merely because it optimizes the same property. An RL editor, a genetic
algorithm, a diffusion model and a controlled CTMC can all improve the same
scalar and still test entirely different scientific claims.

## Why: the concentric structure this imitates

Strong framework papers use a **concentric evidence structure**, not one
indiscriminate leaderboard:

1. **Identify a limitation of an existing generative family.** Expanding Flow
   Maps: ordinary flows and flow maps operate on a fixed-dimensional canvas.
2. **Compare first against the nearest members of that family.** It reuses the
   *same DeFoG graph-transformer denoiser* for fairness and compares against
   DeFoG and a categorical flow-map baseline at matched step budgets — it does
   not primarily ask whether it beats every autoregressive molecule generator.
3. **Ablate to show which new mechanism matters.** MOG-DFM validates its
   underlying flow generators against the relevant flow baseline, then removes
   guidance objectives and the adaptive hypercone.
4. **Add bounded task-level comparison for credibility.** MOG-DFM also compares
   against NSGA-III, SMS-EMOA, SPEA2, MOPSO — establishing useful designs, not
   conceptual novelty.

```
nearest framework comparison  +  matched mechanism ablations  +  bounded task competence
```

## What COMPOSE's contribution actually is

Not "another molecular optimizer". It is

> **an executable stochastic process over complete molecular graphs whose
> one-step support is the canonical fiber of legal chemistry rewrites, together
> with a reusable learned reference law and a separate inference-time control
> layer.**

`F(x) = {y : y is a distinct canonical molecular state reachable from x through a
legal rewrite}`, with `R_θ(y|x)` learning plausibility over that exact support
while control supplies purpose. That decomposition is what enables finite-budget
reachability, continuation from the exact realized molecule, objective change
with `R_θ` fixed, exact support restriction, statewise constraints, and
target-free preference recontrol. **Comparators are chosen on those axes.**

## The four roles

| role | what it tests |
|---|---|
| **`FRAMEWORK_NEIGHBOR`** | the closest alternative generative abstraction. **Primary external evidence for methodological novelty.** |
| **`MATCHED_CAUSAL_CONTROL`** | an internal counterfactual changing ONE COMPOSE mechanism, holding executor, state, objective and budget fixed. **Primary evidence for why each component matters — often more informative than adding a named model.** |
| **`TASK_COMPETENCE`** | strong native modern methods showing COMPOSE performs credibly on ordinary molecular design. **Does not define the novelty claim.** |
| **`CONCEPTUAL_LINEAGE_ONLY`** | scientifically close, but a faithful common numerical task would require substantial adaptation. |

**A method may hold different roles in different experiments.**

## Intended roles

| method | role |
|---|---|
| **DDSBM** | `FRAMEWORK_NEIGHBOR` — the principal numerical external process comparator. Source-conditioned stochastic graph transformation; answers *why COMPOSE's executable graph CTMC rather than another graph CTMC/bridge abstraction?* |
| **GrIDDD** | qualify as `FRAMEWORK_NEIGHBOR` — discrete graph diffusion with node insertion **and deletion**, closer to COMPOSE's variable-size dynamics than any RL optimizer. Main row only if native task and conditioning semantics align without substantial adaptation |
| **Edit Flows** | `CONCEPTUAL_LINEAGE_ONLY` — edit-based CTMC over variable-length **sequences**. Prominent in related work; **no homemade molecular-graph port**, which would require inventing the chemical-support machinery COMPOSE contributes |
| **Expanding Flow Maps** | `CONCEPTUAL_LINEAGE_ONLY` — same broader flow lineage, but variable-size *de novo* generation rather than source-conditioned legal rewriting |
| **CDD / PRODIGY / ConStruct** | hard-constrained generative-process lineage. Numerical only under a native common protocol |
| **GraphXForm / InVirtuoGen / MolEditRL** | primarily `TASK_COMPETENCE` |
| **HN-GFN / InversionGNN** | global multiobjective competence. **Never substitutes for source-conditioned P3/P5/P6** |
| **MOG-DFM / AReUReDi / pCoMole / PepTune** | same-lab lineage — see the policy below |

### Two demotions this makes explicit

**GraphXForm and MolEditRL are no longer the "core intellectual opponent."** They
ask *is COMPOSE competitive with a powerful sequential graph model / a
specialized editor?* DDSBM asks *why COMPOSE's executable graph CTMC rather than
another graph CTMC?* — and the second question is the one closer to the novelty.

## Same-lab policy — neither auto-run nor auto-avoid

> **Scientific proximity determines comparison. Not shared authorship, in either
> direction.**

Avoiding the closest prior work because it is from the same lab is *riskier* than
a careful comparison — a reviewer may reasonably suspect easier baselines were
selected. There is clear precedent for direct same-lab comparison where the task
genuinely matches, and equally clear precedent for *not* comparing where the
generative abstraction differs.

| case | action |
|---|---|
| same lab, same scientific question, faithful native implementation | **compare numerically**, framed as *what the executable molecular graph fiber contributes beyond the prior guidance mechanism* |
| same lab, close lineage, different state space or task | **compare conceptually** — feature table, formal mapping, what carries over. No invented graph version of a sequence method |
| the causal question is already covered by an internal matched arm | **use the internal control as primary evidence** and cite the prior method as lineage. Cleaner, because it holds executor, reference, sources and budget fixed |

**AReUReDi is the only same-lab method to seriously assess for direct numerical
comparison**, and only if official code runs ordinary small-molecule SMILES, the
objectives attach through a thin wrapper, no redesign of its proposal/backbone/
scalarization is needed, and the original authors validate the configuration.
Otherwise it is lineage. **Do not port pCoMole or PepTune onto the COMPOSE
executor ourselves.**

**Barred framing:** *"here is why every earlier model from the group was
inadequate."* **Required framing:** prior work developed powerful guidance and
control for discrete diffusion, flow and edit processes; COMPOSE contributes a
new executable molecular graph substrate, and matched experiments determine what
that substrate enables beyond prior guidance formulations. **Cumulative, not
adversarial.**

## Five questions, recorded BEFORE any experiment is implemented

1. **Question** — what methodological axis is being tested? (state support;
   transition dynamics; local vs future-aware control; hard support vs soft
   guidance; evolving-state feedback vs endpoint ranking.) *Not "what task?"*
2. **Framework counterfactual** — the nearest alternative generative abstraction
   that changes **that axis** while holding as much else fixed as possible.
3. **Matched causal control** — can the question be isolated more cleanly
   internally?
4. **Competence comparator** — at most one or two native task methods.
5. **Falsifier** — what outcome weakens or kills the claim?

Example falsifiers, all live: empirical-family + control matches `R_θ` on quality
*and* fidelity; generate-and-rank matches closed-loop control on every resource
frontier; soft guidance matches hard support on feasible performance; pathwise
enforcement imposes an unacceptable terminal cost.

**Do not add a comparator because it is current or high-performing. Do not omit a
close comparator because it is inconvenient or same-lab.**

## Per-block consequences

- **Exact-target recovery** — no external baseline zoo. Its job is *future
  reachability over the executable state graph matters*, and the evidence is
  internal: greedy, verified, prioritized verified, exact finite-state
  validation. Forcing DDSBM into the sealed six-edit experiment would make the
  causal result less clean.
- **Retargeting** — internal arms remain primary (continue / retarget / restart /
  clairvoyant). It tests a stateful intervention semantic, not optimization rank.
  External methods appear in a capability matrix: do they preserve an explicit
  realized state? do learned parameters update? can the objective change while
  continuing from the exact state? **Do not create fake retargeting variants of
  unrelated models.**
- **Pareto** — centered on P5/P6, repaired P3/P4, and `R_θ` vs empirical-family.
  Greedy vs verified is a secondary increment. HN-GFN/InversionGNN are a separate
  global panel.
- **Hard constraints** — organized around hard-support control as the framework
  contribution: post-hoc vs soft guidance vs exact support, with CDD / PRODIGY /
  ConStruct as the methodological lineage. **A scaffold-specific RL paper does
  not define the section.**

## Frozen decisions preserved — none of this reopens anything

C0 closed and superseded, with the mixed-cache defect recorded and no live result
requalified. Exact-target and held-out retargeting banked and closed. Lane 1
closed. P3/P4 repair continues exactly as frozen. Lane 5 pivots to
published-protocol alignment rather than expensive reproduction. **The broad
Bemis–Murcko branch stays killed by its feasibility gate — no "smaller scaffold"
rescue.** Lane 2 owns the pathwise/hard-support harness; cLogP remains the
endpoint-vs-path experiment. Larger Pareto development is the next major
unresolved capability run. Pareto continuation and plausibility shortlisting stay
`QUEUED_CONDITIONAL`.

**Do not reopen a closed claim absent an actual action- or outcome-changing
defect.**
