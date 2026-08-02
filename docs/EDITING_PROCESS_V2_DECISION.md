# Editing Process V2: one gated admission authority over `atom_delete`

## Status and authority

This document records a **prospective support decision** and its frozen artifacts. It is Phase 1 of
`docs/HANDOFF_PROCESS_V2_CONNECTED_NONLEAF_DELETE.md`, as corrected by the decision recorded below.

It carries **no** training or experiment authority. The self-hashed contract
`configs/editing_v2_semantic_process_v2.json` declares
`status = DESIGN_FROZEN_PROCESS_V2_SUPPORT_DECISION_NOT_TRAINING_OR_EXPERIMENT_AUTHORIZED`,
`training_authorized = false`, and an `authority` block whose every entry is `false`. Active8
materialization, Gate 0, T1, P50, later pilots, a long editing run, and a de novo run all remain
unauthorized. No Modal job is authorized by this decision.

## The recorded decision, and the reading it supersedes

The original approval was:

> I approve COMPOSE Process V2 expanding the learned marked fiber to executor-verified,
> charge-preserving, non-aromatic, non-articulation connected-nonleaf atom deletions, invalidating V1
> downstream artifacts.

The first implementation read the handoff sentence "preserve existing root, singleton, and leaf
behavior" as *preserve the admission set bit for bit*, and therefore exempted inherited candidates
from the authoritative charge policy. That preserved a legacy defect. An independent review rejected
it, and the owner recorded this decision, which supersedes the earlier reading and governs here:

> Apply the authoritative charge policy to every Process V2 atom-delete candidate, including
> inherited root, singleton, and leaf candidates. Preserve the general root/singleton/leaf
> capabilities only when the unchanged executor, connectivity, charge, support, and canonicalization
> predicates all pass. Exclude newly introduced connected-nonleaf candidates incident to a SCAR
> pending a separate SCAR semantic decision. This invalidates the current candidate Process V2
> contract and identity before any downstream run.

"Preserve the capability" means root, singleton, and leaf deletion remains **reachable**. It does not
mean preserving an unfiltered admission set.

## What Process V2 changes

Process V2 changes exactly one thing: the **admission fiber of `atom_delete`**. There is now **one
admission authority** over every candidate, implemented in
`src/compose_v4/rewrite/process_v2_atom_delete.py`. It is not a disjoint union of two admission
rules, and the V1 dense mask is no longer the effective Process-V2 mask for any slot.

The two structural candidate sources remain distinguishable, but only as **diagnostic labels**: they
select which additional gates apply, and they never constitute two admission rules.

| source | definition | introduced by V2 |
| --- | --- | --- |
| `inherited_root_singleton_leaf` | real slot of real-atom degree at most one | no |
| `connected_nonleaf` | real slot of real-atom degree at least `CONNECTED_NONLEAF_MINIMUM_DEGREE = 2` | yes |

### Common exact gates, applied to both sources, in order

1. real element under the authoritative element predicate (SCAR and null are never candidates);
2. the unchanged production atom-delete executor accepts the operation (`is_valid_atom_delete`);
3. the exact persistent-slot successor is connected or null (`is_connected_or_null`);
4. the authoritative charge policy is preserved (`charge_policy_preserved`);
5. the successor is within the declared broad-organic, at-most-40-active-atom support;
6. the successor is canonicalizable (`canonical_state_key` does not raise).

The executor is the legality authority. No weaker approximate valence test may be substituted for it.

### Gates applied to `connected_nonleaf` only

7. non-aromatic under `resonance_invariant_bond_classes` (frozen exclusion, unchanged);
8. not a graph articulation point of the real-atom graph;
9. **not incident to any SCAR slot**, pending a separate SCAR semantic decision.

Gate 9 is connected-nonleaf-only for a reason that is recorded, not assumed: a SCAR-adjacent *leaf*
deletion was already reachable under V1, so excluding it would withdraw an inherited capability the
recorded decision does not withdraw. A SCAR-adjacent *ring* deletion is newly introduced by Process
V2, and `apply_atom_delete` writes implicit hydrogen onto the SCAR neighbour, which is unsettled.

### Aromatic connected-nonleaf deletion stays excluded

Deleting one Kekule-encoded slot of a perceived aromatic system yields a representation-sensitive
open-chain successor under the current executor. Admitting it would make the learned fiber depend on
the stored Kekule phase rather than on the molecule. It requires a separate future semantic decision
and its own resolver.

### The effective mask is an equality, not a union

```
batch.atom_delete_mask == process_v2_atom_delete_mask(state)
```

`process_v2_atom_delete_mask` is the complete effective mask, not an extension of the V1 dense mask.
The model forward asserts **equality** with the recorded admission array, not a subset relation: a
subset assertion cannot catch the union defect this correction removes.

## Measured motivation, and its claim boundary

**Bounded measured observation, not corpus evidence.** On 800 Jin-QED leads, **125 of 3,319**
inherited root/singleton/leaf candidates (**3.77%**), across **105 of 800** molecules, violate the
authoritative charge policy and were admitted anyway by the superseded rule. The common executor and
connectivity gates exclude **0** further, so the charge policy is the whole gap on this panel.

The contract records this under `source_evidence.inherited_candidate_charge_violation_observation`
with `authoritative_corpus_evidence: false` and
`evidence_class: bounded_measured_observation_not_a_corpus_result`. It authorizes nothing. It is the
motivation for the correction, not a corpus result and not a gate.

## Frozen non-goals

Process V2 does not change atom-delete executor semantics, persistent-slot identity, canonicalization,
the formal-charge policy, or any other Active8 operator. It does not admit aromatic or SCAR-incident
connected-nonleaf deletion, does not add multi-neighbour atom insertion, does not enable
`ring_system_delete` or `ring_system_grow`, and does not relabel V1 artifacts as V2. It claims no
one-step inverse closure.

## Frozen names

| thing | value |
| --- | --- |
| legacy atom-delete mode (unchanged) | `legacy_acyclic_atom_delete_v1` |
| Process V2 atom-delete mode | `process_v2_uniform_gated_atom_delete_v2` |
| V2 process semantics | `semantic_editing_v2_v2` |
| admission resolver / enumerator / mask | `resolve_process_v2_atom_delete` / `enumerate_process_v2_atom_deletes` / `process_v2_atom_delete_mask` |

The earlier mode name `process_v2_connected_nonleaf_atom_delete_v1` is **removed and not aliased**. It
falsely implied that inherited candidates were unfiltered. It survives only as a recorded removed name
under `atom_delete.superseded_reading` and `lineage`, and
`scripts/verify_process_v2_hash_chain.py` fails if it appears anywhere else.

The complete reason-code list is
`invalid_source`, `invalid_slot`, `not_a_real_element`, `aromatic_atom`, `articulation_point`,
`scar_incident`, `executor_rejected`, `successor_disconnected`, `charge_policy_violated`,
`successor_outside_declared_support`, `successor_not_canonicalizable`.
`outside_connected_nonleaf_expansion` is removed: it only exists when the expansion is a separate
admission rule.

### Where the mode string, the symbol names, and the reason codes are bound

The contract is the frozen authority for all three. It does not import them from the implementation,
because a contract that reads whatever the implementation currently says is not a frozen decision, and
because the builder has to stay deterministic while the resolver is being edited. The binding is
enforced in the other direction: `scripts/verify_process_v2_hash_chain.py` reads the live
`factorized_tracelet_rate_model.py` and `process_v2_atom_delete.py` sources and fails if the mode
string, the three declared symbols, or the reason-code list disagree with the contract.

## Artifacts and identities

| artifact | path |
| --- | --- |
| V2 semantic process contract | `configs/editing_v2_semantic_process_v2.json` |
| V2 process identity provider | `editing_process_v2_identity()` in `src/compose_v4/rewrite/editing_v2_process_identity.py` |
| admission authority | `src/compose_v4/rewrite/process_v2_atom_delete.py` |
| V1 semantic process contract (unchanged) | `configs/editing_v2_semantic_process_v1.json` |
| hash-chain verifier | `scripts/verify_process_v2_hash_chain.py` |

Stable values, which do not move with implementation edits:

| value | SHA-256 |
| --- | --- |
| V1 contract semantic self-hash | `f928f6adaf22ba7520dd28839655c93bc523ce77317b32050d3dd1b02bbcf288` |
| V1 contract physical hash | `43cb26e1ba33b27a0149a8142895ef9988853b458c0ace9bc9191f7b460dedbb` |

The authoritative current values are always the committed contract's `contract_sha256` and
`editing_process_v2_identity()["process_identity_sha256"]`, recomputed from live source. This document
deliberately records **no** table of current hashes: any such table goes stale the moment a bound
source changes. Run the verifier to read them:

```bash
.venv/bin/python scripts/verify_process_v2_hash_chain.py
```

## Lineage

Two historical identity values are recorded under `lineage` in the contract. They are different kinds
of record and must not be conflated.

| value | kind |
| --- | --- |
| `9fde14b59fc6bfb7be7aaf83564658a9a6758f479d9fd94c134206e84873319b` | **rejected pre-run candidate identity** |
| `6b98ee21ef8b853deda9fa56a2963178208ecc893a397fb4aa412629fc2414d7` | **superseded V1 process identity** |

`9fde14b5…` is the identity the **rejected** candidate reading computed to. It was rejected before any
downstream run and **produced no downstream artifact**: no payload, receipt, cache, checkpoint,
Gate-0, T1, or P50 object was ever built under it. It is therefore recorded as
`status: REJECTED_PRE_RUN_CANDIDATE_IDENTITY` with `was_a_superseded_production_identity: false`. It is
neither current nor superseded production, and it grants nothing.

`6b98ee21…` is a genuine superseded **production** identity: the immutable V1 migration payloads were
built under it, so a historical V1 artifact stays readable under V1 only.

### The V1 identity value moves; its definition does not

`editing_v2_process_identity()` hashes an implementation boundary that includes
`src/compose_v4/model/factorized_tracelet_rate_model.py` and
`src/compose_v4/rewrite/editing_v2_process_identity.py`. Editing either file moves the computed V1
identity value. That is the declared downstream-invalidation mechanism, not a defect. The V1 identity
*definition* is preserved exactly: same schema string, same schema version, same
`process_semantics = "semantic_editing_v2_v1"`, same contract path, same eighteen-file implementation
boundary, same body field set and derivation.

### Two identities coexist without ambiguity

| | V1 | V2 |
| --- | --- | --- |
| identity schema | `compose.editing.semantic_process_identity` | `compose.editing.semantic_process_v2_identity` |
| process semantics | `semantic_editing_v2_v1` | `semantic_editing_v2_v2` |
| contract | `configs/editing_v2_semantic_process_v1.json` | `configs/editing_v2_semantic_process_v2.json` |
| implementation boundary | eighteen sources | the same eighteen plus `process_v2_atom_delete.py` |

`editing_v2` names the lane; the trailing `_v1` or `_v2` names the semantic process version. The
contract's `schema_version` names the semantic process version too, not the body shape; the body shape
is recorded separately as `contract_revision`, currently
`process_v2_uniform_gated_admission_authority` (superseding
`process_v2_disjoint_union_preserving_v1_admission`).

A V1 identity object cannot be read as a V2 object even if it is relabelled and re-self-hashed: the V2
contract path, contract hashes, and implementation-source map all differ, so
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

## The transitive re-pin chain, and the tool that walks it

The contract records the SHA-256 of the sources it binds, so it must be regenerated whenever a bound
source changes. That regeneration is the invalidation signal, not a convenience. The blast radius is
larger than the two configs that pin the identity value directly, because those configs are themselves
content-addressed by others, and stopping partway leaves the chain internally inconsistent.

`scripts/verify_process_v2_hash_chain.py` is the read-only tool to run after every source edit and last
of all before handoff. It:

- recomputes the live V1 identity, V2 identity, and V2 contract self and physical hashes, and fails if
  the committed contract has drifted from a fresh deterministic rebuild;
- **discovers** the chain rather than enumerating it: a self-hash is identified by the equation it
  satisfies, not by a field-name list; pointer edges are found from path-and-sibling-hash structure;
  process-identity pins are found by the repository's own `process_identity_sha256` naming and checked
  against both live identities. A newly added pin therefore cannot hide from it;
- sweeps for every hash value the current edits obsoleted, meaning the base physical hash of every
  changed file plus every 64-hex literal a changed file's base version carried and its current version
  does not, and reports anything that still references one;
- asserts every V1-named config modification is hash-pointer-only, that no V1-named artifact is
  treated as Process-V2 authority, and that `configs/editing_v2_semantic_process_v1.json` still
  self-hashes to `f928f6ad…`;
- asserts the contract's mode string, resolver, enumerator, mask, and reason codes against the live
  implementation source.

It exits non-zero on any disagreement and **writes nothing**. Findings about artifacts outside the
Process-V2 chain, and findings in `docs/` ledgers that record superseded values on purpose, are
reported as warnings rather than failures.

```bash
.venv/bin/python -c "from compose_v4.rewrite.editing_v2_process_identity import \
write_editing_process_v2_contract as w; print(w())"
.venv/bin/python -m pytest tests/test_editing_process_v2_identity.py -q
.venv/bin/python scripts/verify_process_v2_hash_chain.py
```

## Evidence classification

- **Computed and verified.** The nine gates, the single-authority rule, the effective-mask equality,
  the contract and identity hashes, the lineage records, and the coexistence and anti-masquerade
  behaviour, all covered by `tests/test_editing_process_v2_identity.py`.
- **Measured, bounded observation.** The 125 / 3,319 inherited-candidate charge violation on 800
  Jin-QED leads, carrying `authoritative_corpus_evidence: false`. Not corpus evidence; authorizes
  nothing.
- **Proposed.** Every downstream consequence: the aligned learned candidate mask, the V1-payload to
  V2-proof rebind, the rebuilt Active8 inventory, and the rebuilt Gate 0 and T1 evidence.

## Residual risk and unresolved decisions

- **SCAR incidence is deferred, not resolved.** Gate 9 excludes SCAR-incident connected-nonleaf
  candidates pending a separate SCAR semantic decision. Until that decision exists, a SCAR-adjacent
  ring deletion is unreachable, and the asymmetry with SCAR-adjacent leaf deletion is a recorded
  choice rather than a chemical conclusion.
- **The candidate mask must expose exactly this fiber** and must keep legacy behaviour byte-identical
  under `LEGACY_ATOM_DELETE_ACTION_SEMANTICS`. Until that lands and is measured, the effective-mask
  equality in the contract is a declared design, not a measured property of the model.
- **Aromatic connected-nonleaf deletion remains unresolved** and needs its own semantic decision and
  resolver.
- **The 125 / 3,319 observation is bounded.** It is measured on one benchmark lead set, not on the
  editing corpus, and it does not establish how many *admitted* candidates the corrected authority
  gains or loses at corpus scale.
- **Heteroatom and higher-degree deletion changes ring topology**, so the reachability consequences for
  E3 size adaptation and E4 topology adaptation are stated as motivation, not as measured results.
- **Re-pinning restores hash agreement only.** It regenerates no measured evidence: any Gate-0 or T1
  result produced under a superseded identity stays invalid until its lane is authorized and re-run.
- **One residual chain defect is pre-existing and outside this decision's write scope.**
  `configs/editing_training_v2_gate.json` and `src/compose_v4/experiments/editing_training_gate.py`
  pin `semantic_t1_capacity_policy_sha256 = 2186dbd9…`, which is the value that policy carried before
  commit `85169b1`; the live value is `2bb8e951…`. The paired `_file_sha256` was re-pinned and the
  semantic hash was not, and because the gate compares the config against its own source constant
  rather than against the live loader, the two agree with each other and the staleness validates
  silently. The verifier reports it; repairing it belongs to the training-gate owner.
