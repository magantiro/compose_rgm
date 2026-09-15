# T4 target-conditioned utility selector

## Frozen question

The current complete-macro selector is target blind and utility blind. It
scores source, endpoint and structural-goal features plus observed compiler
work. Its frozen contract prohibits task or protein identity, and neither its
fit nor its score consumes a measured docking outcome. The exact limitation is
therefore not representational support. It is the absence of an objective-
conditioned utility signal.

This revision asks a narrower retrospective question: among immutable T4
candidates with exact endpoint graphs and measured QuickVina outcomes, does one
shared target-conditioned endpoint ranker improve held-source ordering over the
same target-blind utility ranker and a fixed target-blind structural baseline?

The generated object is unchanged. In fact, this milestone generates no
molecule and no program. It ranks an already observed exact source-to-endpoint
graph pair. The generator, compiler, complete-macro selector, candidate locks,
scored artifacts and integration code remain immutable.

## Coverage-first data gate

The prospective machine-readable contract is
`configs/t4_target_conditioned_utility_selector_v1.json`. It binds four kinds
of measured artifacts:

Its frozen payload SHA-256 is
`266a8e02b34155b946bfa71cb059272443356d4dc81eb556d47e6e9bd683c743`.

1. the 49 first-evaluation rows from the committed T4 second-generation run and
   all eight exact optimizer archives needed for a candidate, endpoint and
   observation join;
2. the charged T4 program-pool result and its exact replay join;
3. the sealed 19-request delta-0.6 prospective result and its exact candidate
   and request locks;
4. the sealed delta-0.4 prospective result, where only two successful finite
   outcomes can be labels and the two failed requests remain exclusions.

An apparent row is not admitted merely because it has a score. Every label must
join to an immutable measured receipt, exact endpoint graph, source cell,
target, delta, protocol and docking seed. Second-generation rows must join by
candidate and protocol to exactly one exact archive entry and observation.
Program-pool rows must join to the charged result and exact final replay state.
Prospective rows must join through their immutable request and candidate locks.
Any ambiguous or incomplete join abstains.

Reported InVirtuoGen values, published source scores, null docking failures,
unscored structural proximity, champion digests without exact endpoints, live
results and legacy summaries without row-level evaluator receipts are excluded.

Before graph features, normalization or score pairs are derived, records are
held out by target-local source index. Physical request, endpoint, Bemis-Murcko
scaffold, source/cell and generator/program lineage groups may not cross folds.
Any group that conflicts with the predetermined split is excluded in full.
Pairs are formed only within one target, cell, delta, protocol and docking seed.

The fit is allowed only if at least three targets and five held-source strata
remain jointly supported and every evaluated target-fold has a strict measured
training pair. If this gate fails, the authoritative output is the coverage and
abstention audit. No model is fit and no rule is relaxed.

## Fixed comparison

All arms receive identical supported candidate rows:

- a fit-free target-blind structural control, ranked by minimal source-to-
  endpoint feature displacement;
- one target-blind pairwise utility ranker;
- the identical shared pairwise utility ranker with regularized target by
  structural-feature interactions.

Target identity is conditioning input to one shared model. It is not a switch
between five controllers. A target absent from a fold's measured training pairs
is unsupported and abstains. Whole-target transfer and calibrated utility are
not claimed.

Only within-stratum measured score order is supervised. Lower docking is
better. Ties create no pair. The comparison reports pairwise accuracy, NDCG,
top-1 and top-3 regret, best-candidate recall and selection precision together
with row, stratum, source, target and fold coverage. Every abstention and
exclusion is retained.

## Scope boundary

This work is CPU-only and zero-oracle. It may not access Modal or a live run,
change an existing model or lock, launch the five-cell qualification, impute a
missing score, pair incomparable protocols, or use an outer-test result to tune
the fixed model. A positive result is retrospective within-known-target ranking
evidence. It is not prospective optimization or evidence of beating IVG.
