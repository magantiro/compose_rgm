# PMO zero-oracle route-policy quality comparison

## Scientific question and output

This revision asks whether the structural state features in the sealed Practical
Molecular Optimization (PMO) route export predict the next generic primitive
decision better than a balanced marginal. The primary output is held-out
likelihood and rank for a hierarchical local action policy. This is not an
autonomous molecule generator, task optimizer or scored result.

The only input is
`diagnostics/pmo_route_distillation/attempt_1/training_dataset.json.gz`, bound by
its sealed audit and byte hash. Original curricula, source graphs, endpoints,
oracles and live trajectories are not reopened.

## Frozen split and balance

Three folds hold out complete task families, which also holds out every lineage
and route in those families:

1. bioactivity and formula;
2. rediscovery and multi-property; and
3. similarity, druglikeness and multiobjective optimization.

The folds contain 2,052, 1,797 and 2,294 decisions. No held-out row contributes
to vocabulary construction, feature normalization, feature selection, class
priors or centroids. Within each training fold, the sealed family/base-lineage/
decision weights are renormalized. A base lineage is the frozen exact-source and
common-prefix grouping, so it is also the available source grouping after the
export intentionally removed source graphs.

## Compared policies

`balanced_marginal_action_prior` uses only weighted categorical frequencies. It
is the preregistered baseline.

`graph_conditioned_hierarchical_local_ranker` is a deterministic, regularized
nearest-centroid ranker. For each categorical head it fits training-only scaling,
selects at most 32 state features by weighted between-class variance and combines
distance to the class centroid with the same smoothed marginal. It factorizes one
decision as:

1. **WHAT:** primitive executor rule;
2. **WHERE:** structural region for every generic operand role;
3. **HOW:** typed generic parameter record;
4. **DEPENDENCY:** preexisting versus route-created operand and exact creation lag;
5. **CREATE:** whether the action produces a relative handle; and
6. **STOP:** whether the primitive ends its teacher route.

WHERE descriptors contain element class, charge, implicit hydrogen count, degree,
bond-class histogram and neighbor-element histogram. Persistent slots are absent.
Created handles use relative provenance only. Evaluation teacher-forces preceding
generic factors; unseen contexts or classes abstain and reduce coverage.

## Metrics and evidence boundary

For each head and the complete factorization, report held-out coverage, top-one
precision among covered decisions, top-one recall over all decisions, negative
log likelihood, and rank. Report runtime-length and long-route local-only decisions
separately. Also record CPU time, peak resident memory and checkpoint size.

The sealed export contains 6,143 primitive decisions: 2,792 belong to complete
routes of at most 32 primitives and 3,351 belong to longer routes. It contains no
exact graph/action pair from which a structural role can be rebound to an absolute
legal mark. All 6,143 stages are primitive fallbacks and zero are compound modules.
Thus a complete-program decoder and a contrastive complete-candidate panel are not
identified by this dataset. The report must preserve this negative result:

- complete-policy-decodable coverage is zero;
- no candidate is proposed, so precision is undefined rather than zero;
- exact and transformation-equivalent recall are zero;
- unique yield is zero; and
- all 106 runtime-length unique reference traces are explicit shortfall.

Long-route local decisions may improve local held-out likelihood but cannot count
toward complete-route recall.

## Runtime exclusions

Fold checkpoints contain only generic categorical labels, numeric parameters,
training-only preprocessing and a hashed training identity. A fail-closed scan
excludes task or family names, route/member/lineage identifiers, endpoint hashes,
SMILES, source paths or graphs, absolute slots, assignments and executable action
records. Offline reports may retain fold and evidence provenance but are not
runtime inputs.

## Run

```bash
PYTHONPATH=src:. .venv/bin/python tools/pmo_route_policy_quality.py \
  --output diagnostics/pmo_route_policy_quality/attempt_1
```

The command is deterministic, refuses overwrite, makes zero oracle/docking calls
and publishes self-hashed runtime-fold checkpoints and an offline result.

## Follow-on complete-program data contract

The actual complete-program objective requires a new training-only export. Unlike
`attempt_1`, it should retain each exact parent graph, executable primitive action,
successor graph and route boundary after freezing task-family and base-lineage
splits. Those teacher fields are legitimate offline supervision; the prohibition
applies to fitted runtime checkpoints, not to split training data.

A generic next revision should deterministically segment each trace at changed
connected regions and created-handle dependency closures. Each segment must replay
exactly, expose only relative entry/exit roles to the model and preserve primitive
fallback when no reusable segment is supported. For every training parent, build a
same-source negative panel by exact-compiling task-blind generic Dynamic proposals
under the same 32-primitive/eight-block support. Label teacher transformation
equivalence, validity, duplicate status and structural distance offline, without
using an oracle score. Fit a complete-candidate ranker only after this panel is
frozen. Evaluate on held-out task families/lineages with teacher-free generation,
reporting proposal recall, precision, valid and unique yield, and shortfall. The
runtime artifact must retain only generic segment/role parameters and numeric
ranker state, never source graphs, executable teachers or provenance labels.
