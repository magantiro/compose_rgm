# v3 reuse ledger

COMPOSE v4 is a clean pivot, not a wholesale fork.

## Copied now

- `MolecularGraph`, element/bond vocabularies, SMILES conversion, exact valence
  checks, and RDKit validity from v3 `src/utils/molecular_graph.py`.
- Connectivity helpers from v3 `src/utils/graph_primitives.py`.
- The design principle that validators and executors are the single semantic
  source of truth for both training and inference.

## Reimplemented cleanly

- Minimal rewrite dataclasses and deterministic executors.
- Joint atom-state restating rather than separate partially overlapping heads.
- A typed rewrite registry with condition hooks.
- Canonical aggregation of marked rates at complete successor molecules.
- The formal null graph as a valid de novo source state.

## Deferred for evidence-driven reuse

- The transactional `RingSystemPatchSpec` executor. It remains sound and can be
  ported when the micro-only model establishes a path-length or mixing failure.
- Stitch insertion/deletion operators. These are candidate macros for the first
  multiscale ablation.
- Topology features and the compact graph-transformer backbone.
- Corpus marginal estimators and evaluation utilities.

## Deliberately not copied

- Destruction-noising and inverse-oracle supervision.
- Candidate-bank, beam, and isolated patch-recall machinery.
- The old MIS scheduler as the definition of the reverse process.
- Scar restoration as a core state representation.
- Hand-coded motif vocabularies.
- Legacy paper drafts and stale claims.

`compose_v3` remains unchanged as the provenance and experimental record.
