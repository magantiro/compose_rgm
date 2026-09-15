# T4 complete-macro selector

## Scope

This milestone tests ordering, not generation. It consumes the immutable
one- and two-construction-event candidate pools from the compositional
structural-subgoal generator and changes only the order of the 128 learned
candidates for each held T4 source. The generator, legal grammar, exact
compiler, whole-source folds and candidate identities remain unchanged.

The selector reuses `proposal_features` and `ContextSubgoalRanker`. Its input is
one exact source graph, one complete realized endpoint and the jointly
represented structural-goal templates. For selector fold `f`, positives are
the exact successful route transformations from the ten training sources.
Each source contributes the first 32 wrong learned candidates from the
immutable case in which that source was held out from generator fitting.
Exact endpoint and strict radius-two transformation matches are excluded from
the negative set. The fit gives equal mass to source, route and same-source
negative.

The selector score is the pairwise context score minus a fixed 0.05 penalty on
training-fold-normalized `log1p(primitive_count + expanded + attempted)`. Those
work values come from immutable exact-realization receipts. The lock contains
no rejected candidate rows, so a realizability classifier is unsupported and
the model abstains from fitting one. A diversity guard emits the best candidate
for every distinct exact complete-macro signature before any repeated
signature.

## Frozen data and fit

The prospective contract is
`configs/t4_complete_macro_selector_v1.json`, with payload self-hash
`2ced5ba49c9b01c026cf1320ddcbe64db4a89431902a9c4a93751962bb022214`.
The physical file hash is
`b949d23b1886506f39abd29136394f0e00ba3dbe9d6d0bbd935a929683e41a63`.

All three folds fit without abstention:

| Fold | Positive routes | Training sources | Hard negatives | Pairwise rows |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 50 | 10 | 320 | 1,600 |
| 1 | 51 | 10 | 320 | 1,632 |
| 2 | 53 | 10 | 320 | 1,696 |

Every source supplied 32 hard negatives. None of the inspected top-32
training candidates was an exact or radius-two teacher match. This is label
coverage for fitting, not held-source recovery evidence.

## Held-source result

The frozen scientific gate passed at K=8, 16 and 32. The selector substantially
improved early held-teacher granular component coverage over the raw learned
order while keeping every selected endpoint and exact complete-macro signature
unique:

| K | Raw learned coverage | Selector coverage | Raw learned exact patch recall | Selector exact patch recall | Selector novel patch yield |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.180126 | 0.320572 | 0 | 0.004444 | 7.27 |
| 16 | 0.251341 | 0.401935 | 0 | 0.009206 | 14.53 |
| 32 | 0.346326 | 0.485553 | 0.009206 | 0.009206 | 29.40 |

One held whole patch was recovered by K=8 and two by K=16. The same two were
present at K=32. Exact canonical endpoint recovery and strict radius-two
transformation-equivalent recovery remained zero at every cutoff for the
selector, raw learned order and uniform order. Thus the positive result is
better early structural support, not full-route endpoint recovery.

Generated granular component precision changed from 0.643238, 0.648389 and
0.665167 in the raw learned order to 0.635354, 0.643988 and 0.659177 in the
selector order at K=8, 16 and 32. The selector therefore trades a small amount
of generated-component precision for much higher held-component coverage. Its
mean log compiler work is also higher than the raw learned prefix at every
cutoff: 1.905 versus 1.429 at K=8, 1.923 versus 1.460 at K=16, and 1.956 versus
1.545 at K=32. The fixed cost auxiliary did not override the selector's
preference for more route-like, structurally richer candidates.

The selector inherits learned-pool legal compile coverage 0.935217 and exact
realization precision 1.0. It generated no new candidate and made zero oracle,
docking or Modal calls. Proposal throughput is unsupported because this
benchmark only reorders an already generated lock.

## Artifacts and reproducibility

Authoritative artifacts are under
`diagnostics/t4_complete_macro_selector/attempt_1`:

- selector lock: physical SHA-256
  `aab18cc4fd737d3bf5953c79a7e3332459009813626aa6ee65519516c43d77c2`,
  payload SHA-256
  `9ad8cea49864ceb286d0caee0cf714ac5034e78e4b9d7bd6557fb8721f78112f`;
- result: physical SHA-256
  `2cc879e03ead3303ebbd5134a63e6e673f19d7fb0f9b90d173d5b5140ce53b80`,
  payload SHA-256
  `03502a8661537635762d0369fd48fc250a6e440f0a3c93da59f50d028b6967b3`;
- fold runtime checkpoints: physical SHA-256 values
  `616032094f84626a98381378eb2225ca3e2831c85e7c3b8df9cfabc1dbf78920`,
  `d7099a1dcab3ed82553e06a2bf313958d36e69e02f50ca4e984c519e2cedbfbf`
  and `bcb543991bc9d9c21b2aaf6a9c6b051502e9723907459a5344836921cd97966a`.

A full independent `fit`, `rank` and `evaluate` rerun produced a byte-identical
artifact directory. Focused tests cover checkpoint sanitation, source-balanced
fit, exact-realization receipt validation, immutable pool membership, diversity
ordering, input census, zero-time throughput abstention and exact fold-quality
reduction.

## Interpretation and next gate

The computed evidence supports a small complete-macro selector for earlier
structural coverage inside the existing generated support. It does not show
exact route recovery or molecular utility. A later score-blind prospective
utility prelock may select a very small development panel from this immutable
ranking, but docking, oracle evaluation, live-run access and controller
selection require separate authorization.
