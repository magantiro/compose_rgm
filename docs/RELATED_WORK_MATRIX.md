# Related-work matrix — Paper 1

**Purpose.** The novelty claim is a *conjunction*, so it needs a property-by-property ledger rather than a
prose assertion. No single column below is claimed as new; the claim is the row.

**Status of citations.** Every row currently names a bibliography key verified against a primary
paper or official proceedings page. Property-level `~` assignments listed under verification debt
still require a full primary-paper audit before they become manuscript claims.

**Amended 2026-08-13 by the baseline-qualification lane** (branch
`codex/compose-baseline-qualification`), with lead approval. Changes: DDSBM `var-card` and
`birth/death` corrected `✗ → ~`; new rows for GraphXForm and REINVENT 4; per-cell sentences added
below. The MARS `pathwise` cell was **not** changed — see verification debt item 4.

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
| DDSBM (CTMC bridges for graph transformation) | ~ | ~ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | `kim2025ddsbm` |
| ConStruct (constrained graph diffusion) | ✗ | ✗ | ✓ | ~ | ✓ | ✗ | ✗ | ✗ | ~ | `madeira2024construct` |
| PRODIGY (aggregate-set projection) | ✗ | ✗ | ~ | ~ | ✓ | ✗ | ✗ | ✗ | ~ | `sharma2024prodigy` |
| CoCoGraph (double-edge swaps in a fixed fiber) | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | `ruizbotella2026cocograph` |
| Molecular hypergraph grammars | ✓ | ✓ | ✓ | ~ | ✓ | ✗ | ✗ | ✗ | ~ | `kajino2019mhg`, `guo2022grammar` |
| Junction-tree / motif generation | ✓ | ✓ | ✓ | ~ | ✓ | ✗ | ✗ | ✗ | ~ | `jin2018jtvae` |
| Reaction-based GFlowNets | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ~ | ✓ | `koziarski2024rgfn`, `zhu2026spacegfn` |
| MARS (annealed MCMC, fragment edits) | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ~ | ✓ | `xie2021mars` |
| MIMOSA (iterative add/replace/delete) | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ~ | ✓ | `fu2021mimosa` |
| Graph GA / PMO search | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ~ | ✓ | `jensen2019graphga`, `gao2022pmo` |
| GraphXForm (graph transformer, constructive atom/bond addition) | ✓ | ✗ | ✓ | ~ | ✓ | ✗ | ✗ | ✗ | ~ | `pirnay2025graphxform` |
| REINVENT 4 (SMILES policy RL, staged learning) | ✓ | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ | ~ | ✗ | `loeffler2024reinvent4` |
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

### Sentences for the `~` cells added by the baseline-qualification audit

- **REINVENT 4 `retarget` is `~`, and the distinction is the claim.** REINVENT 4 ships staged
  (curriculum) learning whose stated purpose is "to allow the user to optimize a prior model
  conditioned on a calculated target profile by varying the scoring function in stages", writing a
  checkpoint after each stage that is reusable as the next stage's agent. So the objective genuinely
  changes **within one run, without restarting**. It is `~` and not `✓` because what is re-targeted is
  the **policy**, not a realized molecular trajectory: there is no in-flight molecule to continue
  from, the switch fires at a stage boundary set by `max_score`/`max_steps` rather than at an
  arbitrary chosen step, and the agent re-adapts by further RL. **Do not write "no existing method
  changes objective mid-run."** COMPOSE's claim is the conjunction — unanticipated goal, realized
  molecular history preserved, finite remaining budget, frozen reference process.
- **GraphXForm `pathwise` is `~` because two different mechanisms must not be conflated.** Its
  *online graph-validity masks* are genuinely pathwise: `update_action_mask()` is recomputed after
  every action and violating logits are set to −∞ before the softmax. But the maskable class is only
  valence, atom type, atom count, no self-bonding and no re-bonding. Its *structural* constraints —
  ring size and disallowed bonding patterns — are a **terminal** check
  (`molecule_evaluator.py::infeasible_by_special_constraints`, asserted on `mol.synthesis_done`,
  returning −∞), and there is no SMARTS or substructure matching anywhere in the released code.
  Online validity masking ≠ pathwise structural constraint.
- **GraphXForm `complete` is `~`** because a terminable, fully connected molecule exists only at
  action level 0; between levels the graph carries a dangling unbonded atom.
- **GraphXForm `birth/death` is `✗`** because the action space is strictly constructive — `AddAtom`,
  `AddBond`, `DontChange` — and the paper defers removal to future work. A seed structure therefore
  persists in every later state because nothing *can* be deleted, not because a constraint holds.
- **DDSBM `var-card` and `birth/death` are `~`, corrected from `✗`.** Every molecule is padded to the
  dataset-wide maximum node count with a first-class dummy atom type `X`, and the uniform CTMC
  transition freely flips a slot C→X (deletion) or X→C (insertion), with dummies stripped at decode.
  The node *tensor* is fixed (37 slots on their ZINC subset, mean molecule 23.7 heavy atoms); the
  *molecule* size is not. `~` rather than `✓` because cardinality varies only inside a fixed padded
  maximum. The `complete ✗` cell is confirmed correct: the sampler resamples every node and edge
  category independently at each step with no valency, connectivity or sanitisation check.

## Verification debt

1. Re-verify the `~` cells for GenMol, InVirtuoGen, and reaction-based GFlowNets. These were assigned from
   strategy-level descriptions, not from a read of the primary papers.
2. The ConStruct / PRODIGY / CoCoGraph cells **are** verified against primary sources (carried over from
   the earlier positioning audit) and may be stated as fact.
3. The DDSBM, GraphXForm and REINVENT 4 cells were verified against primary papers and released code by
   the baseline-qualification lane; per-cell evidence is in
   `docs/workstreams/baseline-qualification/comparator_registry_v3.json`.
4. **The MARS `pathwise ✓` cell is deliberately unchanged and remains ambiguous.** The released MARS
   code contains no masking, no SMARTS, no substructure matching and no atom freezing — an exhaustive
   grep returns only training-loss masks — and `break_bond` may delete a fragment constituting a motif
   we wanted preserved. Under a *released-code* reading the cell is `✗`. Under an *affordance* reading
   it is `✓`, because every MARS state is a complete molecule and a proposal filter could reject
   motif-breaking edits. The change was not made because it would flatter COMPOSE on an ambiguous
   column definition. **Settle the column's intended reading first**, then apply it uniformly to MARS,
   MIMOSA, Graph GA and Kappa-style rewriting, which all sit on the same ambiguity.
