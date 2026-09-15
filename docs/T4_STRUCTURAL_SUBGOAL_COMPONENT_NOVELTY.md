# T4 Structural-Subgoal Component-Novelty Audit

## Decision motivating this audit

The grouped whole-template proposal gate recovered zero held-source teacher
subgoals at Recall@8, Recall@32 and Recall@128. Its runtime support was a finite
vocabulary of complete training-fold patches. Every held serialized template
identifier lay outside that vocabulary. This rejects larger whole-template
retrieval pools and better whole-template reranking as the next controller
direction. It does not reject structural subgoals or the sealed exact realizer.

## Scientific question

The audit asks whether held target patches are new combinations of familiar
components or whether the components are themselves absent from training-fold
support. The prospective generated object is

\[
g=(R,H,\alpha,D),
\]

where `R` is an address-free selected source region, `H` is a generated target
patch graph, `alpha` attaches the patch to surviving source context and `D`
records dependency structure needed for realization. A later policy may use

\[
\pi(g\mid x)=\pi(R\mid x)\,\pi(H,\alpha,D\mid x,R),
\]

but this audit does not fit or select that policy. In particular, it does not
authorize independent marginal classifiers for topology, attributes,
attachments and dependencies. Those variables constrain one another and a
future generator must model them jointly or conditionally.

## Frozen grouped evaluation

Use the same three predeclared whole-source folds as the failed policy gate.
Derive each component vocabulary only from the ten training sources of that
fold, then measure the five held sources. Role order and hashes are not treated
as chemical identity. Graph components are bucketed by deterministic structural
signatures and compared with exact attributed graph isomorphism.

For each fold and component family, report:

- held instance coverage and novelty;
- held unique coverage and novelty;
- train-vocabulary transfer precision;
- per-subgoal all-component and any-component coverage;
- instance-weighted and source-balanced aggregate coverage.

For complete patches, distinguish:

1. exact whole-patch overlap;
2. a novel whole patch whose core component families are all familiar;
3. a novel whole patch with at least one unseen core component.

The frozen component families are source-region motif, target topology, atom
attributes, bond attributes, attachment pattern, dependency motif, radius-one
target fragments, radius-two target fragments and connected target graphlets of
one to three vertices. Strict whole-component bundles are reported alongside a
finer sensitivity analysis over source and target graphlets, individual atom and
bond attribute tokens, attachment-edge tokens and dependency tokens.

An unsealed implementation sanity preview showed that treating a whole
atom-transition multiset or whole dependency graph as one component still asks a
bundled-template question. No authoritative result or architecture selection had
been published. The contract therefore preserves those strict metrics and adds
the finer analysis before the clean audit. The report must distinguish strict
bundle familiarity from granular compositional support.

## Auxiliary-data compatibility

The audit also asks whether two read-only sources can use the same structural
representation:

- the 69 Full-146 entries that are not labelled complete compiled routes;
- the sealed PMO dependency-region training corpus.

For each, measure exact source-program replay where available, structural-goal
conversion, exact target reconstruction, supported primitive/region length,
abstentions and unique structural components. Also report held T4 component
coverage for T4-training-only, PMO-only and union vocabularies. When the 69
Full-146 programs augment a T4 fold, admit only applications whose origin is a
training source for that fold. This is a compatibility audit only. It does not
authorize merging T4 and PMO data, fitting a joint policy or
placing executable teacher routes in a runtime checkpoint.

## Acceptance and claim boundary

The output must be deterministic, provenance-bound and zero-oracle. It must
preserve the 77-route, 15-source and 147-subgoal T4 census, verify zero group
leakage and record every conversion failure or abstention. The sealed exact
realizer and all live or scored runs remain untouched.

The current sealed structural extractor compares retained atom signatures and
deleted-role nulls directly while ordering some otherwise equivalent source
roles. PMO traces that reach that branch must be reported as extractor
abstentions in this audit. They must not be classified as chemically
inexpressible, and this audit does not authorize repairing the sealed extractor.

This result can motivate a compositional structural-patch generator. It cannot
establish autonomous patch generation, molecular utility, docking improvement
or an IVG comparison. Those require a later contract, nonzero and improving
held-source generated-patch recall with diversity and exact-realization
precision, then a frozen score-blind docking panel.
