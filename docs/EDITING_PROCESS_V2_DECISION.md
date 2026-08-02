# Editing Process V2: connected-nonleaf atom deletion

## Status and authority

This document records a **prospective support decision** and its frozen artifacts. It is Phase 1 of
`docs/HANDOFF_PROCESS_V2_CONNECTED_NONLEAF_DELETE.md`.

It carries **no** training or experiment authority. The self-hashed contract
`configs/editing_v2_semantic_process_v2.json` declares
`status = DESIGN_FROZEN_PROCESS_V2_SUPPORT_DECISION_NOT_TRAINING_OR_EXPERIMENT_AUTHORIZED`,
`training_authorized = false`, and an `authority` block whose every entry is `false`. Active8
materialization, Gate 0, T1, P50, later pilots, a long editing run, and a de novo run all remain
unauthorized. No Modal job is authorized by this decision.

The recorded prospective decision is:

> I approve COMPOSE Process V2 expanding the learned marked fiber to executor-verified,
> charge-preserving, non-aromatic, non-articulation connected-nonleaf atom deletions, invalidating V1
> downstream artifacts.

## What Process V2 changes

Process V2 changes exactly one thing: the **admission fiber of `atom_delete`**.

The V1 dense production mask excludes every cyclic atom before any executor validation. Because an
acyclic vertex of real-atom degree at least two is always a cut vertex in a connected real-atom
graph, the V1 rule (`atom_topology == 0` and not an articulation point and no neighbour implicit
hydrogen exceeding `MAX_H_COUNT`) admits only slots of real-atom degree at most one, that is roots,
singletons, and leaves. Valid non-aromatic, non-articulation deletions of higher-degree slots are
legal under the production executor but receive zero probability under the learned V1 model. That is
a support mismatch, not a chemistry limit.

Process V2 preserves the V1 root, singleton, and leaf rule unchanged and additionally admits
connected-nonleaf candidates:

```
V2 effective atom_delete mask = (unchanged V1 dense mask) UNION (connected-nonleaf mask)
```

The two operands are disjoint by the degree argument above, so the union is a disjoint union and no
V1 admission is re-decided.

### Admission semantics for the expansion

`src/compose_v4/rewrite/process_v2_atom_delete.py` is the single scientific authority. A slot enters
the expansion only when its real-atom degree is at least `CONNECTED_NONLEAF_MINIMUM_DEGREE = 2`, and
it is admitted only when all seven conditions hold:

1. the slot is a real element under the authoritative element predicate;
2. it is non-aromatic under the frozen production representation;
3. it is not a graph articulation point of the real-atom graph;
4. the unchanged production atom-delete executor accepts the operation;
5. the exact persistent-slot successor is connected;
6. the frozen charge policy is preserved by the transition;
7. the successor is within the declared support and is canonicalizable.

The executor is the legality authority. No weaker approximate valence test may be substituted for
it. Executor validity is deliberately not a connectivity predicate (`is_valid_atom_delete` returns
`True` for articulation-point deletions), so conditions 3 and 5 are independently load-bearing.

### Aromatic connected-nonleaf deletion stays excluded

Deleting one Kekule-encoded slot of a perceived aromatic system yields a representation-sensitive
open-chain successor under the current executor. Admitting it would make the learned fiber depend on
the stored Kekule phase rather than on the molecule. Aromatic connected-nonleaf deletion therefore
requires a separate future semantic decision and its own resolver, and is excluded here.

### The charge policy is not applied to preserved V1 candidates

Applying the charge policy uniformly would silently remove existing V1 leaf candidates, which the
decision declares unchanged. Measured on `C[N+](C)(C)CC(=O)[O-]`: the V1 dense mask admits slots
`[0, 2, 3, 6, 7]`, of which only slot 6 preserves the net formal charge. Uniform application would
delete four of the five preserved leaf candidates. The charge policy is therefore applied to the
connected-nonleaf expansion only, which is the set this decision introduces.

## Frozen non-goals

Process V2 does not change atom-delete executor semantics, persistent-slot identity, canonicalization,
the formal-charge policy, or any other Active8 operator. It does not admit aromatic connected-nonleaf
deletion, does not add multi-neighbour atom insertion, does not enable `ring_system_delete` or
`ring_system_grow`, and does not relabel V1 artifacts as V2. It claims no one-step inverse closure.

## Artifacts and identities

| artifact | path |
| --- | --- |
| V2 semantic process contract | `configs/editing_v2_semantic_process_v2.json` |
| V2 process identity provider | `editing_process_v2_identity()` in `src/compose_v4/rewrite/editing_v2_process_identity.py` |
| connected-nonleaf resolver | `src/compose_v4/rewrite/process_v2_atom_delete.py` |
| V1 semantic process contract (unchanged) | `configs/editing_v2_semantic_process_v1.json` |

Stable values, which do not move with implementation edits:

| value | SHA-256 |
| --- | --- |
| superseded V1 process identity | `6b98ee21ef8b853deda9fa56a2963178208ecc893a397fb4aa412629fc2414d7` |
| V1 contract semantic self-hash | `f928f6adaf22ba7520dd28839655c93bc523ce77317b32050d3dd1b02bbcf288` |
| V1 contract physical hash | `43cb26e1ba33b27a0149a8142895ef9988853b458c0ace9bc9191f7b460dedbb` |

Values as of the integrated branch head, after every bound implementation source was final:

| value | SHA-256 |
| --- | --- |
| V2 contract semantic self-hash | `30369f36373b1163114b9a8bf6849f3fbed4627b6070e4478c4a5efb6c3c6261` |
| V2 contract physical hash | `6111edce5fd7ab6214ae069804c48e6c1e902ac969463a7b279b834d16780cc9` |
| V2 process identity | `9fde14b59fc6bfb7be7aaf83564658a9a6758f479d9fd94c134206e84873319b` |
| V1 process identity, superseded | `6b98ee21ef8b853deda9fa56a2963178208ecc893a397fb4aa412629fc2414d7` |
| V1 process identity, current | `9874a69a01902dd1c4a0353c14cb9db0c657887b3e497800867629d82f1bf63e` |

The authoritative current values are always the committed contract's `contract_sha256` and
`editing_process_v2_identity()["process_identity_sha256"]`, not this table. Any table of hashes in
prose goes stale the moment a bound source changes; treat this one as a record of the integrated
head, not as an oracle.

### The V1 identity value moves; its definition does not

`editing_v2_process_identity()` hashes an implementation boundary that includes
`src/compose_v4/model/factorized_tracelet_rate_model.py` and
`src/compose_v4/rewrite/editing_v2_process_identity.py`. Editing either file moves the computed V1
identity value. That is the declared downstream-invalidation mechanism, not a defect. The V1 identity
*definition* is preserved exactly: same schema string, same schema version, same
`process_semantics = "semantic_editing_v2_v1"`, same contract path, same eighteen-file implementation
boundary, same body field set and derivation.

The re-pin blast radius is larger than the two configs that pin the identity value directly, because
those configs are themselves content-addressed by others. Measured at integration, the complete
transitive set is nine artifacts, re-pinned in dependency order:
`configs/editing_gate_zero_semantic_model_process_v1.json`,
`configs/editing_v2_semantic_capability_cells_v1.json`,
`configs/editing_v2_semantic_development_cell_roles_v1.json`,
`configs/editing_v2_semantic_active8_decision_runtime_v1.json`,
`configs/editing_v2_semantic_gate_zero_structural_v1.json`,
`configs/editing_v2_semantic_t1_panel_policy_v1.json`,
`configs/editing_v2_semantic_t1_capacity_policy_v1.json`,
`configs/editing_v2_semantic_p50_recipe_policy_v1.json`, and
`configs/editing_training_v2_gate.json`, plus four source constants in
`src/compose_v4/experiments/editing_v2_semantic_gate_zero.py`,
`src/compose_v4/experiments/editing_v2_semantic_t1_capacity_policy.py`, and
`src/compose_v4/experiments/editing_v2_semantic_t1_decision.py`. Stopping partway leaves the chain
internally inconsistent, and one link (the T1 capacity policy) compares against a source constant
rather than the live loader, so a partial re-pin there is silently stale rather than loud.

### Two identities coexist without ambiguity

| | V1 | V2 |
| --- | --- | --- |
| identity schema | `compose.editing.semantic_process_identity` | `compose.editing.semantic_process_v2_identity` |
| process semantics | `semantic_editing_v2_v1` | `semantic_editing_v2_v2` |
| contract | `configs/editing_v2_semantic_process_v1.json` | `configs/editing_v2_semantic_process_v2.json` |
| implementation boundary | eighteen sources | the same eighteen plus `process_v2_atom_delete.py` |

`editing_v2` names the lane; the trailing `_v1` or `_v2` names the semantic process version.

A V1 identity object cannot be read as a V2 object even if it is relabelled and re-self-hashed: the
V2 contract path, contract hashes, and implementation-source map all differ, so
`require_editing_process_v2_identity` rejects it, and `validate_frozen_process_identity` rejects a
relabelled object whose `contract_relative_path` does not match the schema it claims.

`validate_frozen_process_identity(payload)` validates a historical identity object for internal
self-consistency only. It reads no file and compares nothing against live source, so it deliberately
does not prove currency: a payload that passes it may still describe a superseded process. Use it to
accept a pinned superseded V1 identity carried by an immutable V1 payload, and use
`require_editing_v2_process_identity` or `require_editing_process_v2_identity` when currency is
required.

## Downstream invalidation

The V2 contract declares, and this decision accepts:

- Active8 inventory rebuild required;
- affected whole-trace replay required;
- candidate and successor cache rebuild required;
- Gate 0 and T1 rebuild required;
- no checkpoint resume across process versions;
- historical artifacts remain readable only under their original identity;
- source and scaffold splits unchanged;
- P50 not authorized by this contract;
- the V1 contract is not edited in place and no V1 artifact is relabelled as V2.

## Regenerating the contract

The contract records the SHA-256 of the sources it binds, so it must be regenerated whenever a bound
source changes. That regeneration is the invalidation signal, not a convenience. In particular the
Phase 2 candidate-mask change to `src/compose_v4/model/factorized_tracelet_rate_model.py` requires a
regeneration in the same change.

```bash
.venv/bin/python -c "from compose_v4.rewrite.editing_v2_process_identity import \
write_editing_process_v2_contract as w; print(w())"
.venv/bin/python -m pytest tests/test_editing_process_v2_identity.py -q
```

`tests/test_editing_process_v2_identity.py` fails if the committed JSON drifts from
`build_editing_process_v2_contract()`, and validates both the physical file hash and the semantic
self-hash.

## Evidence classification

- **Computed and verified.** The seven admission conditions, the disjoint-union rule, the contract
  and identity hashes, and the coexistence and anti-masquerade behaviour, all covered by
  `tests/test_editing_process_v2_identity.py`.
- **Measured, bounded development observation.** The fourteen-source disjointness panel and the
  `C[N+](C)(C)CC(=O)[O-]` charge counterexample recorded under `source_evidence`. Both carry
  `authoritative_corpus_evidence: false`. They are not corpus evidence and do not authorize anything.
- **Proposed.** Every downstream consequence: the aligned learned candidate mask, the V1-payload to
  V2-proof rebind, the rebuilt Active8 inventory, and the rebuilt Gate 0 and T1 evidence.

## Residual risk and unresolved decisions

- The Phase 2 candidate mask must expose exactly this fiber and must keep legacy behaviour
  byte-identical under `LEGACY_ATOM_DELETE_ACTION_SEMANTICS`. Until that lands and is measured, the
  effective-mask rule in the contract is a declared design, not a measured property of the model.
- Aromatic connected-nonleaf deletion remains unresolved and needs its own semantic decision and
  resolver.
- Heteroatom and higher-degree deletion changes ring topology, so the reachability consequences for
  E3 size adaptation and E4 topology adaptation are stated as motivation, not as measured results.
- Re-pinning the V1-lane binding chain to the moved V1 identity value only restores hash agreement.
  It regenerates no measured evidence: any Gate-0 or T1 result produced under the superseded identity
  stays invalid until its lane is authorized and re-run.
