# Conditional / control results & scoping (Paper 1)

**Scoping decision (locked): Option A — the paper is *validity-closed structural control*.**
The contribution is exact rule-closed constraints + oracle-efficiency + anytime editing +
the exactness anchor. Property optimization is **not** a headline success-rate race; it
appears only as the *objective inside constrained tasks*. We do **not** claim to beat
GrIDDD on unconstrained QED (oracle-hungry vs oracle-light — an unfair, off-thesis
framing). The weak oracle-light conditioning is a scoped limitation / future work, not a
required win.

## Thesis
COMPOSE = Rewrite Generator Matching: a validity-closed, flexible-size molecular generator
learned via Generator Matching over a CTMC of chemical graph rewrites. Every committed
state is a complete, chemically valid, connected molecular graph; every transition is an
executable chemical rewrite. Per the plan: *"sufficiently competitive de novo, strongest
in conditional design."*

## Honesty guardrails
- Exact **only** for rule-closed *structural* constraints (scaffold, substructure,
  connectivity, size). Property conditioning is learned/guided (approximate).
- Do **not** claim diffusion can't do constraints (PRODIGY/ConStruct/CoCoGraph can); our
  edge is the *native conjunction* validity-closed + oracle-evaluable-every-step +
  flexible-size + executable-rewrite editing in one process.
- Flexible size alone isn't unique. Not unconditional-distribution SOTA; FCD non-SOTA
  acceptable by design (no FCD arms race). Say "chemically valid connected molecular
  graphs," not "real molecules."

## Committed results

| Experiment (plan map) | Result | Commit |
|---|---|---|
| **E0 — Doob h-transform exactness** (well-posedness) | tilted kernels reproduce the exact Bayes conditional to machine precision (Linf 1.1e-16) on the actual 118-state rewrite CTMC; Q^h a valid generator (off-diag negativity 0.0); first-order convergence; robust across 3 predicates | `375761e` |
| **Scaffold collapse** (constraint headline) | baseline generate-then-filter usable fraction 95.7%→79.9%→29.4%→12.0% by scaffold size (47 leads); ours flat **100%** | `f98c074` |
| **Substructure generality** | required aromatic baseline 74.4%, forbidden aldehyde 97.0%; ours **100%** (24 leads) | `a0a4f9f` |
| **Usable-oracle efficiency** | ours **100%** in-fiber vs baseline 39.8% (wastes 60%; 88% on hard scaffolds) | `43ee1f9` |
| **Anytime** | best-so-far reaches ~90% of final gain by ~97 oracle calls (12 leads) | `ecd8bf2` |
| **E1 de novo sufficiency** (unconditional foundation) | smoke: 100% valid/unique/novel, element-dist TV ~0.03; V/U/N + descriptor Wassersteins + diversity + coverage (no FCD arms race) | in progress |
| **E3 constrained optimization** (integrated headline) | QED **+0.072 on 12/12 leads** (0.755→0.827) *while* the Murcko scaffold is exactly preserved (**12/12, 100%**), at **100%** oracle-efficiency, reaching 90% of the gain by **~35 oracle calls** (12 leads) | `diagnostics/conditional_smc/scaffold_opt_panel12.json` |

## QED / property optimization — the reframe
QED is the *objective being optimized within constrained tasks*, never a standalone
benchmark line. Our value-guided SMC (~30% at ~560 oracle calls on 48 leads) is
**oracle-hungry** and is **not** a fair comparison to GrIDDD's 45.1% oracle-light
(~20 calls). If reported at all, it is a disclosed-budget appendix row vs search
baselines (MARS/GEGL) — never a headline GrIDDD beat. The integrated E3 result shows QED
climbing under an *exact* constraint that generate-then-filter cannot satisfy (collapses)
and unconstrained optimizers ignore.

## Matched-controls positioning (cite reported; our proxy)
- **GrIDDD** (arXiv 2506.15725): 2D insert/delete graph diffusion, oracle-light, same
  CNOF+N⁺/O⁻ footing, QED 45.1%. Reports ~nothing unconditional → our de novo evidence
  already exceeds theirs. Our generate-then-filter proxy stands in for the "conditional
  model + rejection" baseline on the constraint tasks.
- **Morph** (arXiv 2606.07239): 3D flexible-size editor, Edit-Flows/Generator-Matching
  family (our lineage); reports 3D stability, not 2D FCD/marginals → related-work anchor,
  not a 2D-metrics competitor.
- **Constrained diffusion** (PRODIGY/ConStruct/CoCoGraph): *can* enforce hard constraints
  → cite as matched controls; our edge is the native conjunction + valid oracle-evaluable
  intermediates, not "they can't."
