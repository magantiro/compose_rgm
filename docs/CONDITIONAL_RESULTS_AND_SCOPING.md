# Conditional / control results & scoping (Paper 1)

**Scoping decision (locked): Option A — the paper is about validity-closed structural
control.** The contribution is exact rule-closed constraints, usable-oracle efficiency,
anytime editing, and the exactness anchor. Property optimization is not a standalone
success-rate race; it is the objective inside constrained molecular-design tasks.

## Method and thesis

COMPOSE is **Rewrite Generator Matching (RGM)**: a validity-closed, flexible-size
molecular generator learned by Generator Matching over a continuous-time Markov chain
(CTMC) of chemical graph rewrites.

- **States:** complete, chemically valid, connected molecular graphs.
- **Transitions:** executable graph rewrites; the process is non-monotone and can both
  grow and shrink.
- **Sampling:** ancestral rollout from a directly sampled, degree-bounded carbon-tree
  prior until a terminal event.
- **Production rewrite vocabulary:** atom grow/shrink (insert/delete), atom retype,
  bond-order change, subtree Graft, and whole-ring-system rewrite.
- **Conditional control:** hard structural conditions define the legal event fiber;
  property oracles can evaluate every committed state and every executable successor.

The intended paper position is: **sufficiently competitive de novo generation, strongest
in controlled molecular design.** The distinctive contribution is not flexible size,
validity, editing, or Generator Matching in isolation. It is their conjunction in one
learned stochastic process over executable molecular rewrites.

## Production reference configuration

The base checkpoint for the committed conditional results is **Lineage B**:

- quotient-correct flexible Graft and whole-ring-system rewrites;
- zero canonical self-events and no delete-to-one collapse;
- `rate_factorization=hierarchical`;
- `ring_template_factorization=flat`;
- `max_atoms=40`;
- trained on GuacaMol (C/N/O/F) with charged N+/O-; constrained-design leads are held-out CNOF optimization leads from the Jin et al. set (ZINC-derived); C/N/O/F elements match GrIDDD (which trains on ZINC-250k), base corpora differ;
- seed `20260717`.

Lineage B was selected for clean, interpretable edit dynamics. Its remaining ring-marginal
defect is a limitation rather than part of the conditional headline.

## Conditional mechanisms

- **Hard constraints:** Tanimoto similarity, Murcko scaffold, required SMARTS, forbidden
  SMARTS, connectivity, and size can be enforced in the legal event fiber when the
  constraint is closed under the chosen rewrite rules.
- **Oracle ordering:** feasibility is checked before scoring. Infeasible successors are
  never accepted and never consume an oracle call.
- **Exact conditioning anchor:** when the harmonic function `h` is available, the Doob
  h-transform gives the exact conditional generator for the learned rewrite CTMC.
- **Practical property guidance:** `h` is generally intractable, so property-conditioned
  design uses approximate value guidance / finite-particle SMC.
- **Reward fine-tuning:** Relative Trajectory Balance was tested as an oracle-light route,
  but current conditioning is inconclusive and is not part of the headline result.
- **No beam search:** beam search is deliberately excluded because it collapses diversity
  and does not reflect the stochastic-process formulation.

## Committed experiment suite

| Experiment | Result | Artifact / commit |
|---|---|---|
| **E0 — Doob h-transform exactness** | On a 118-state slice of the production legal-event CTMC, tilted kernels reproduce the exact Bayes conditional to machine precision (`Linf = 1.1e-16`). `Q^h` is a valid generator (off-diagonal negativity `0.0`, row-sum error `2e-13`), and generator integration converges first-order. Verified for three predicates: contains-N, size-min, and size-max. | `375761e` |
| **E2 — Scaffold satisfaction** | On 47 leads, COMPOSE remains at **100%** satisfaction. Generate-then-filter falls with scaffold size: **95.7%** (<=8 atoms), **79.9%** (9–15), **29.4%** (16–22), and **12.0%** (>=23). For the 32 hard leads with scaffolds >=16 atoms, the proxy baseline is approximately 19% usable. | `f98c074` |
| **E2 — Substructure generality** | On 24 leads, COMPOSE is **100%** for both required-aromatic and forbidden-aldehyde constraints. The proxy baseline is **74.4%** and **97.0%**, respectively. | `a0a4f9f` |
| **E2 — Usable-oracle efficiency** | COMPOSE spends **100%** of oracle calls on structurally feasible candidates. The proxy baseline is **39.8%** overall and wastes approximately 88% on hard scaffolds. | `43ee1f9` |
| **E2 — Anytime performance** | Best-so-far reward reaches approximately **90% of final gain by 97 oracle calls** across 12 leads. Every intermediate is a usable molecule, so the process can stop and resume. | `ecd8bf2` |
| **E3 — Integrated constrained optimization** | Scaffold-enforced, value-guided SMC improves QED on **12/12 leads**, from mean **0.755 to 0.827** (`+0.072`), while preserving the Murcko scaffold on **12/12 leads** at **100% oracle efficiency**. Approximately 90% of the gain is reached by **35 oracle calls** (`~155` mean calls used). | `diagnostics/conditional_smc/scaffold_opt_panel12.json` |

## De novo foundation (E1)

The unconditional experiment establishes that the same model is a credible general
generator; it is not the paper's leaderboard headline.

Report:

- validity, uniqueness, and novelty;
- MOSES/GuacaMol-style descriptor-distribution distances, including logP, QED, MW,
  TPSA, and ring descriptors;
- internal diversity and train-support coverage;
- final evaluation at 10,000 samples across three seeds.

Current smoke results are approximately 100% valid / unique / novel, with
element-distribution TV around `0.03` and bond-order TV around `0.09` at very small `n`.
These smoke numbers are not final estimates.

Do not elevate per-marginal atom-count, cycle-rank, or ring-count total-variation metrics
to headline benchmark status. They remain useful internal diagnostics, while the public
comparison follows recognized MOSES/GuacaMol reporting conventions. FCD may be reported
if informative, but COMPOSE is not positioned around an FCD arms race.

## Property optimization framing

QED is the objective optimized **inside a structurally constrained task**, not a standalone
benchmark claim. The key integrated result is that QED rises while the scaffold remains
exactly preserved and every oracle call is usable.

The separate oracle-hungry SMC result (approximately 30% success at approximately 560
oracle calls on 48 leads) is not a fair head-to-head comparison with GrIDDD's reported
45.1% oracle-light result at approximately 20 calls. If retained, it belongs in an
appendix with disclosed budgets and search-method comparators rather than in the headline
story.

## Comparator positioning

- **GrIDDD:** primary 2D flexible-size comparison on the same CNOF + charged N+/O-
  preprocessing. Use it for matched targeting/optimization context, but do not claim an
  unmatched raw-QED win.
- **Morph:** flexible-size 3D editor in the Edit-Flows / Generator-Matching lineage. It is
  a related-work anchor rather than a direct 2D distribution-metric competitor.
- **PRODIGY, ConStruct, and CoCoGraph:** constrained-generation controls demonstrating
  that hard constraints are not exclusive to COMPOSE. The COMPOSE distinction is the
  native combination of flexible size, executable rewriting, valid oracle-evaluable
  intermediates, generation, and non-monotone editing.
- **MOSES and GuacaMol:** establish the accepted de novo reporting suite.

The generate-then-filter proxy is the current empirical control for how a conditional
model plus rejection behaves as structural difficulty increases. Direct constrained-model
comparisons should be added when implementations and protocols can be matched fairly.

## Presentation policy

Main slides should report the positive scientific result directly. They do not need a
repeated caveat box on every slide.

Recommended narrative:

1. Exactness of conditioning on the learned rewrite generator (E0).
2. Exact hard-constraint satisfaction where filtering collapses (E2).
3. Structural oracle efficiency and anytime usable intermediates (E2).
4. Integrated constrained optimization: improve QED while preserving the scaffold (E3).
5. Sufficient de novo generation as the common model foundation (E1).

Consolidate scope and open questions into one final **Limitations and next experiments**
slide. That slide should note approximate property conditioning, the Lineage B ring
trade-off, inconclusive oracle-light reward fine-tuning, and the need for direct matched
constrained-model controls.

## Claim boundaries for manuscripts and Q&A

- Exactness applies to rule-closed structural constraints and to the Doob transform of the
  learned generator when `h` is available. Learned property targets remain approximate.
- Say **chemically valid connected molecular graphs**, not "real molecules."
- Do not claim that diffusion cannot enforce constraints.
- Do not claim flexible size, validity, or Generator Matching alone as the novelty.
- Do not claim unconditional distribution-modeling SOTA or an unmatched GrIDDD QED win.

These boundaries belong in the methods, discussion, limitations slide, and responses to
questions—not as repeated disclaimers that obscure the main results.
