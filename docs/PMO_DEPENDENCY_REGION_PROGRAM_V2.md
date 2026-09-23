# PMO dependency-region program representation v2

## Problem and claim boundary

The split-first v1 corpus replayed every derived segment exactly, but its flat
eight-segment limit represented only eight of the 106 routes within the unchanged
32-primitive runtime budget. The observed failure arose from representing every
primitive fallback as a separate top-level module. This revision tests a
macro-free hierarchical representation. It does not test a learned proposal
policy or PMO objective improvement.

The frozen v1 `attempt_2` result and training corpus are immutable inputs. Their
physical and self-hashed payload identities are sealed in the v2 contract. V2
reuses the existing 576-attempt same-source negative panel by reference because
component grouping changes neither its executed endpoints nor its exact or
radius-two transformation labels. It generates no new proposal.

## Split and training-only data

V2 preserves the v1 whole-task-family and shared-lineage folds byte-for-byte and
derives each representation independently within one route. No vocabulary,
statistic or preprocessing state is derived before the frozen split. The output
is training-only and may retain exact persistent-slot states, primitive payloads,
terminal endpoints, membership and provenance. No runtime checkpoint is emitted.

## Macro-free dependency-region program

Each atom lifetime receives a route-local structural token. An undirected
top-level component joins primitive actions only when their changed atom-lifetime
footprints overlap, touch through a bond observed in the exact route, share a
created-handle producer/consumer edge, or form a connected generic cycle-open to
cycle-close or ring-restatement pair. Component names carry no chemical or task
semantics.

The original global primitive order is preserved as an emission schedule. The
first emission for a component opens it, later emissions continue it, and its
last emission closes it. A component may remain open while another component is
emitted. Each created output has a component-local ordinal; later uses record a
most-recent-first created-handle backreference, producer-component backreference
and primitive lag, never an absolute created slot in the generic emission record.
Exact primitive payloads remain separately available only in the training corpus.

Complete runtime support remains at most 32 total primitives and at most eight
components. Routes over 32 primitives abstain explicitly. V2 does not raise the
primitive budget, truncate a route, or fabricate a semantic macro.

## Frozen gate

The representation gate requires:

- exact replay coverage and precision 1.0 over all 184 unique traces;
- complete representation of all 106 routes containing at most 32 primitives;
- nonzero complete support in every frozen held-out fold;
- zero created-handle and cycle dependency edges crossing components;
- unchanged 32-primitive and eight-component limits;
- verified reuse of the sealed v1 panel with zero new proposals; and
- no model fit, runtime checkpoint, task-score label, oracle, docking or Modal
  activity.

If the gate passes, a later separately authorized revision may compare a balanced
marginal component decoder with a graph/region/dependency-conditioned
autoregressive decoder and same-source contrastive complete-candidate ranker.
That later comparison must still split before fitting and keep teacher graphs,
routes, endpoints and executable programs out of runtime checkpoints.

## Run

```bash
PYTHONPATH=src:. .venv/bin/python tools/pmo_dependency_region_program_v2.py \
  --output diagnostics/pmo_dependency_region_program_v2/attempt_1
```

The command is deterministic, single-CPU and zero-oracle. It refuses overwrite
and atomically publishes a self-hashed training corpus and compact result.
