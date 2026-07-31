# Related-work matrix — Paper 1

**Purpose.** The novelty claim is a *conjunction*, so it needs a property-by-property ledger rather than a
prose assertion. No single column below is claimed as new; the claim is the row.

**Status of citations.** Every row currently names a bibliography key verified against a primary
paper or official proceedings page. Property-level `~` assignments listed under verification debt
still require a full primary-paper audit before they become manuscript claims.

## Columns

| column | meaning |
|---|---|
| **var-card** | cardinality changes *within one trajectory* (not "draw n, then generate") |
| **birth/death** | entities are both **created and destroyed**; not insertion-only, not deletion-only |
| **topology** | operators that change cycle structure (close/open/restructure rings) |
| **complete** | every *visible generative state* is a complete, connected, valence-valid molecule |
| **learned path** | a learned reference **path distribution**, not a search heuristic over a fixed proposal |
| **quotient** | probability defined on **canonical successors**, not on action encodings |
| **exact ctrl** | exact terminal reweighting given an exact desirability |
| **retarget** | objective may change **mid-trajectory** without restarting |
| **pathwise** | constraints enforced at **every committed state**, not at the endpoint |

## Matrix

| method / family | var-card | birth/death | topology | complete | learned path | quotient | exact ctrl | retarget | pathwise | citation |
|---|---|---|---|---|---|---|---|---|---|---|
| Graph diffusion / discrete flow (DiGress, DeFoG, D3PM, CTMC) | ✗ | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ | ~ | ✗ | `vignac2023digress`, `qin2025defog`, `austin2021d3pm`, `campbell2022ctdd` |
| Score / spectral / continuous-time graph variants | ✗ | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ | ~ | ✗ | `jo2022gdss`, `jo2023grum`, `xu2024disco`, `siraudin2024cometh` |
| **Trans-dimensional generative modeling via jump diffusion** | ✓ | ✓ | ~ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | `campbell2023jump` |
| GrIDDD (insertion/deletion graph diffusion) | ✓ | ~ (monotone) | ~ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | `ninniri2025griddd` |
| Edit Flows (variable-length insert/delete/substitute) | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | `havasi2025editflows` |
| Morph (unbalanced OT, 3D cardinality) | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | `franke2026morph` |
| DDSBM (CTMC bridges for graph transformation) | ✗ | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | `kim2025ddsbm` |
| ConStruct (constrained graph diffusion) | ✗ | ✗ | ✓ | ~ | ✓ | ✗ | ✗ | ✗ | ~ | `madeira2024construct` |
| PRODIGY (aggregate-set projection) | ✗ | ✗ | ~ | ~ | ✓ | ✗ | ✗ | ✗ | ~ | `sharma2024prodigy` |
| CoCoGraph (double-edge swaps in a fixed fiber) | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | `ruizbotella2026cocograph` |
| Molecular hypergraph grammars | ✓ | ✓ | ✓ | ~ | ✓ | ✗ | ✗ | ✗ | ~ | `kajino2019mhg`, `guo2022grammar` |
| Junction-tree / motif generation | ✓ | ✓ | ✓ | ~ | ✓ | ✗ | ✗ | ✗ | ~ | `jin2018jtvae` |
| Reaction-based GFlowNets | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ~ | ✓ | `koziarski2024rgfn`, `zhu2026spacegfn` |
| MARS (annealed MCMC, fragment edits) | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ~ | ✓ | `xie2021mars` |
| MIMOSA (iterative add/replace/delete) | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ~ | ✓ | `fu2021mimosa` |
| Graph GA / PMO search | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ~ | ✓ | `jensen2019graphga`, `gao2022pmo` |
| Stochastic graph rewriting (Kappa-style) | ✓ | ✓ | ✓ | ✓ | ✗ (fixed/thermo propensities) | ~ | ✗ | ✗ | ✓ | `danos2015thermodynamic`, `behr2021rewriting` |
| Generic Generator Matching | n/a | n/a | n/a | n/a | ✓ | ✗ | ✗ | ✗ | ✗ | `holderrieth2025generator`, `billera2026latent` |
| Classical Doob *h*-transform | n/a | n/a | n/a | n/a | n/a | n/a | ✓ | n/a | n/a | (textbook) |
| RetMol (retrieval-based controllable generation) | ✗ | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | `wang2023retmol` |
| GenMol (masked discrete diffusion, hit/lead-opt) | ✓ | ~ | ✓ | ✗ | ✓ | ✗ | ✗ | ~ | ✗ | `lee2025genmol` |
| HN-GFN (preference-conditioned multi-objective molecular optimization) | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ~ | ~ | `zhu2023hngfn` |
| InVirtuoGen (fragment discrete flow and lead optimization) | ~ | ~ | ✓ | ✗ | ✓ | ✗ | ✗ | ~ | ✗ | `kaech2026invirtuogen` |
| **COMPOSE (this work)** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | — |

Legend: ✓ yes · ~ partial / restricted class · ✗ no · n/a not applicable (the row is a principle, not a
molecular model).

## How to use this honestly

- **The row, not a column, is the claim.** Every ✓ in the COMPOSE row exists somewhere above it. Say so.
- **Attribute the closest precedents explicitly.** Jump diffusion for trans-dimensionality; GrIDDD and
  Edit Flows for learned insertion/deletion; MARS/MIMOSA/graph-GA for valid-state editing; grammars for
  validity by construction; stochastic rewriting for the rule/match/propensity ontology; Generator
  Matching for the fitting principle; Doob for the transform.
- **The two genuinely under-occupied columns are `quotient` and `exact ctrl`+`retarget` together.** These
  are where the paper's technical weight sits, and they are worth defending in detail.
- **`complete` is a support property, not a result.** The reviewer response is "yes, and here is what it
  buys": pathwise feasibility, mid-trajectory intervention, branching, no wasted oracle calls.
- **Partial (`~`) marks need a sentence, not a symbol.** E.g. ConStruct's `pathwise` is `~` because its
  projector covers only edge-deletion-invariant properties, which excludes required labeled subgraphs.

## Verification debt

1. Re-verify the `~` cells for GenMol, InVirtuoGen, and reaction-based GFlowNets. These were assigned from
   strategy-level descriptions, not from a read of the primary papers.
2. The ConStruct / PRODIGY / CoCoGraph cells **are** verified against primary sources (carried over from
   the earlier positioning audit) and may be stated as fact.
