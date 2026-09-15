# T4 delta-0.6 route-reference corpus

This milestone converts the already inventoried public IVG T4 delta-0.6 endpoint
references into a deterministic, training-only COMPOSE corpus. It makes no
docking or task-oracle call.

The input census contains 39 unique source-endpoint pairs and 49 run references
over all 15 T4 cells. Exactly 32 pairs have exact executable witnesses. The
remaining seven endpoint references are retained as explicit abstentions: three
charge changes outside the declared rewrite policy, two unresolved searches and
two unsupported representations.

The evidence boundary is strict. Endpoints are reported IVG endpoint references.
Routes are **COMPOSE-compiled witnesses inferred from those endpoints**, not
observed IVG trajectories. Tied run references do not create repeated training
examples: one canonical source-endpoint pair has one unit training weight and
retains all of its delta-0.6 references.

For admitted witnesses the exporter verifies the immutable receipt, replays the
exact primitive ancestry, extracts one-to-four dependency regions, serializes
address-free structural subgoals and invokes the unchanged sealed realizer. It
reports exact replay and realization coverage and precision separately. A route
remains a valid exact witness if the generic structural realizer abstains; such
an abstention is not silently reclassified.

The declared representation support is 48 persistent slots, at most 40 active
atoms, at most 32 primitive edits and at most four structural regions. Metrics
include primitive length, dependency-region count, source edit radius, retained
source-atom fraction, cycle-rank change and exact-isomorphism diversity of target
topology, attachment and dependency components.

Run after committing the implementation and contract:

```bash
PYTHONPATH=src .venv/bin/python tools/t4_delta06_route_corpus.py \
  --output-dir diagnostics/t4_delta06_route_corpus/attempt_1 \
  --code-revision "$(git rev-parse HEAD)"
```

The output directory is new and immutable. The command refuses to overwrite an
existing result. Its compressed corpus uses a zero gzip timestamp, and every
scientific payload is self-hashed for byte-stable reruns.

This corpus may support a later generic structural-delta generator. It does not
authorize training, a route-aware scored search, endpoint harvesting, Modal, a
live-run change or a claim that COMPOSE beats IVG at delta 0.6.

## Authoritative attempt 1

The clean-revision export at commit
`f5160de53fafd99115db7eedaf69ceb5d2e3f8b9` retained 39 unique endpoint
references and all 49 tied run references. It admitted 32 exact witnesses and
preserved seven abstentions: three `unreachable_charge_change`, two
`search_unresolved` and two `unsupported_representation`.

All 32 admitted routes passed exact primitive replay, dependency-region
extraction, address-free target reconstruction and the unchanged sealed
structural realizer, each at 32/32 coverage and 32/32 precision. They contain 45
structural subgoals. Primitive length ranges from 6 to 24 (median 14.5), while
dependency-region count ranges from 1 to 3 (median 1). The retained source-atom
fraction ranges from 0.595 to 1.0 (median 0.842), and source edit radius ranges
from 0 to 11 (median 4).

Exact attributed-isomorphism counts are 36 target-topology classes, 30
attachment-pattern classes and 44 dependency-motif classes across 45 subgoals.
These are retrospective training-data measurements, not autonomous proposal or
docking results.

Authoritative files:

- `diagnostics/t4_delta06_route_corpus/attempt_1/training_corpus.json.gz`,
  SHA-256 `d96265ca0f3364a118b10037b8312263f2e627ff629651bca22eca1ae834cded`;
- `diagnostics/t4_delta06_route_corpus/attempt_1/result.json`, SHA-256
  `93ed30b94362352d1affadcbf8ddb18105931a84c3e045aa6d706abcd057bdab`.

An independent in-memory rebuild reproduced the compressed corpus byte for byte.
