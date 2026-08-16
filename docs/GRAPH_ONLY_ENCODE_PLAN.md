# Graph-only `h_φ` encode path — analysis, ready to implement

**Target: the 79.1% of a particle transition currently spent in "encode".**

## What the encoder actually reads

`_encode_batch` reads exactly 10 `FactorizedMarkBatch` fields (verified by AST,
every `batch.*` access enumerated): `atom_types`, `formal_charges`,
`implicit_h_counts`, `neural_bonds`, `times`, `atom_topology`,
`closure_topology`, `ring_system_topology`, `property_condition_values`,
`property_condition_mask`. **It never touches a mark-space tensor.**

## What the current path computes anyway

`_one_state_batch` → `prepare_factorized_mark_batch` (561 lines). Inside the
per-state loop, only two calls feed the encoder:

| line | call | verdict |
|---|---|---|
| 304 | `compute_topology_features(state)` | **KEEP** — atom/closure/ring-system topology |
| 372 | `ChemistryStateFeatures(...)` | **KEEP**, but only 4 of its fields are read |

Seven call sites are pure waste for an embedding:

`de_novo_rewrite_system` · `_graph_application_masks` ·
`_semantic_cycle_close_admission_mask` · `_semantic_atom_restate_admission_mask` ·
`_semantic_cycle_open_admission_mask` · `enumerate_clean_ring_system_deletes` ·
`_charge_preserving_macro_actions` (×2)

## The lever already exists

**All of it sits behind `if features is None:`** — the builder accepts a
`chemistry_feature_cache` and skips every expensive call on a hit. The encode
path just never passes one.

Two options, in increasing order of risk:

1. **Pass a `chemistry_feature_cache`.** Zero new code paths, purely additive.
   Helps only on *repeated* states — and in SMC most proposals are new, so the
   ceiling is low. Cheap to try first.
2. **Encoder-only feature construction.** Build `ChemistryStateFeatures` with
   only `atom_topology`, `closure_topology`, `ring_system_topology` and
   `neural_bonds` populated, masks left `None`, and assemble a batch with the
   21 required-but-unread fields as empty tensors. This is what actually removes
   the 79%.

## Qualification gate — non-negotiable

**Bitwise-identical embeddings AND `h_φ` values** against:
- all 736 transitions in the banked SMC reference artifact, and
- a broader sample of diverse molecular states.

Not exact → discard. Batching already failed this bar at 5.96e-07, which is why
the bar exists: an epsilon shift moves `h_φ`, which moves an acceptance decision.

## Order

Read Round 1 first. Then: encode path → re-profile → particle parallelism with
fixed per-particle RNG substreams (so serial and parallel implement the *same*
contract and compare exactly) → fast 12-16 source dev panel.
