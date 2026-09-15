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
`configs/t4_target_conditioned_utility_selector_v3.json`. It binds four kinds
of measured artifacts:

Its frozen payload SHA-256 is
`a1fca9322ca916916c6b2b5e4c2c8bc8612ff4cd4e4fece6db731c6147bff3dd`.

1. the 49 first-evaluation rows from the committed T4 second-generation run,
   its sealed query-to-batch selection lock, all hash-bound exact candidate
   batches and all eight optimizer archives that bind the measured result;
2. the charged T4 program-pool result, exact replay join, authoritative source
   registry and exact historical winner-control receipt and protocol;
3. the sealed 19-request delta-0.6 prospective result and its exact candidate
   and request locks;
4. the sealed delta-0.4 prospective result, where only two successful finite
   outcomes can be labels and the two failed requests remain exclusions.

An apparent row is not admitted merely because it has a score. Every label must
join to an immutable measured receipt, exact endpoint graph, source cell,
target, delta, protocol and docking seed. Second-generation rows must join from
the measured query through the sealed selection membership to one exact batch
candidate and complete state trace. Program-pool rows must join to the charged
result and exact final replay state, and their input state must equal the
authoritative address-free cell source. A replay from an intermediate context
is not a row for the original cell even when its endpoint was measured.
Prospective rows must join through their immutable request and candidate locks.
Any ambiguous or incomplete join abstains.

Reported InVirtuoGen values, published source scores, null docking failures,
unscored structural proximity, champion digests without exact endpoints, live
results and legacy summaries without row-level evaluator receipts are excluded.

Before graph features, normalization or score pairs are derived, records are
held out by target-local source index. Physical request, endpoint, Bemis-Murcko
scaffold, source/cell and generator/program lineage groups may not cross folds.
Any group that conflicts with the predetermined split is excluded in full.
Connected grouping components are removed together so that a scaffold or
lineage conflict cannot survive indirectly through another retained row.
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

## Authoritative outcome

The v3 coverage gate abstained before feature fitting. The immutable inputs
contain 99 physical units: 97 finite measured labels and two failed or null
delta-0.4 requests. Exact source validation admits 84 labels before split-group
filtering. Thirteen of 27 program-pool endpoints have no representation from
the authoritative PARP1-0 source and therefore abstain; 25 non-cell program
representations are reported separately and are not label denominators.

The predetermined endpoint, scaffold, source and per-ancestry grouping graph
contains one connected component spanning source folds. Removing that complete
component excludes 37 otherwise admitted rows and leaves 47 rows. Those rows
support only two held-source strata, both for JAK2. The frozen requirements of
at least three supported targets and five supported strata therefore fail.
No target-blind or target-conditioned utility model was fit, no checkpoint or
prospective candidate lock was emitted, and no oracle, docking or Modal call
occurred. The 240 remaining within-stratum pairs are descriptive combinations,
not independent samples and not a basis for bypassing the source-level gate.

The authoritative machine-readable result is
`diagnostics/t4_target_conditioned_utility_selector/attempt_3/result.json`
(SHA-256 `1d6b056c3d724443686fe27586e63d70fe70b8af1103b942937b78519cd12627`).
Its data audit SHA-256 is
`6cb69f88f6b77ca808ea34b3869ff0b5cb26252c0a04f4e29e7214b07d860bfe`.
The safe next action is to acquire additional immutable measured candidate
cohorts across source indices and targets under a separately frozen contract,
not to relax the grouping, target-support or comparability gates.

## Repair lineage

The v1 and v2 executions were unsealed coverage previews. They stopped before
fitting but were not committed with a recoverable producer identity, so they
are not cited as immutable evidence. They exposed that second-generation query
identity differed from exact batch-candidate identity and that raw persistent
slots are not chemical graph identity. Review before the first authoritative
execution also found that the reused PARP1 winner needed its exact receipt and
that program-pool rows mixed the true cell source with an intermediate
post-linker context. V3 freezes those provenance repairs while leaving the
scientific split, model, metrics and coverage thresholds unchanged.
