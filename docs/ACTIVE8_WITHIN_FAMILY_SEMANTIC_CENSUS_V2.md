# Active8 within-family semantic census V2

## Purpose and scope

This diagnostic measures which row-local and graph-context labels occur in the
whole-trace-admitted Active8 training rows. Its output is a set of independent
within-family marginals. It does not construct a Cartesian semantic cell, make
a corpus pass/fail decision, measure model learnability, or authorize training.

V2 is frozen to the exact Active8 identities copied from
`configs/editing_t1_successor_gate_v8.json`: the inventory file and logical
hashes, effective source-corpus hash, support-contract hash, and unified packed
manifest hash. The policy has a canonical self-hash, while its physical file
hash is independently pinned in the validating implementation. Recomputing a
mutated policy's self-hash therefore cannot make it admissible.

The generated object remains the COMPOSE marked rate law over executable
molecular rewrites. This census is only a training-density diagnostic within
the declared charge-preserving, stereochemistry-free, at-most-40-atom Active8
editing support. The T1 successor-capacity experiment remains the learnability
gate. Candidate-count and alias-count diagnostics, when supplied, are reported
orthogonally and are never joined to semantic labels.

## Evidence boundary

Observed inputs are the immutable whole-trace admission decision, exact stored
persistent-slot source and successor states, exact stored executable action,
packed layer, and progress position. The implementation does not reconstruct a
slot-addressed state from SMILES and does not execute or reinterpret the action.

Computed fields are graph cycle rank, shortest paths, bridge context, exact
action-local transitions, independent marginal row counts, and exact distinct
source-state and successor-state counts using the authoritative persistent-slot
state hash. Declared finite domains and the sparse-value threshold are proposed
diagnostic policy. Thus:

- `absent_declared_values` is defined only for a small declared finite axis;
- `absent_declared_values: null` means an open-observed axis has no absence
  claim;
- `sparse_observed_values` means at most four training rows under V2 policy;
- neither sparse nor absent values imply corpus failure or unsupported model
  semantics.
- repeated rows remain visible as row density but cannot masquerade as unique
  state coverage, because both counts appear per family and per axis value;
- no scaffold count is claimed. The exact rows do not carry an authoritative
  scaffold identity, so `unique_scaffolds` is null.

The census is train-only. Validation, controller-validation, and final-test
roles are not selection inputs for this training-density diagnostic.

## Family-specific marginal axes

Every family receives the exact packed layer, trace-step role, source real-atom
count, and graph cycle rank. Additional axes are:

- `atom_insert`: root versus one-neighbor birth, inserted element and charge,
  attachment bond class, and anchor element.
- `atom_delete`: last-atom, isolated, leaf, or multi-neighbor context, deleted
  element and charge, source degree, and incident bond-class multiset.
- `atom_restate`: element, charge, and implicit-H changes, source degree,
  cyclic versus acyclic site context, and whether a cyclic target belongs to
  the authoritative `ORGANIC_RING_ELEMENTS` set.
- `bond_reorder`: bond-class transition, endpoint elements and degrees, and
  bridge context.
- `bond_reroute`: removed-to-added bond-class transition, removed and added
  endpoint elements, endpoint reuse, cut component sizes, attachment degrees,
  and removed-edge bridge context.
- `cycle_insert` (cycle close): closure bond class, endpoint elements and
  degrees, preclosure shortest path, and resulting shortest cycle size.
- `cycle_attach` (cycle open): opened bond class, endpoint elements and degrees,
  remaining shortest path, and opened shortest cycle size.
- `ring_system_restate`: changed-bond and affected-atom counts, transition
  multiset, affected element set, changed-edge cycle context, and whether an
  aromatic bond class participates.

These axes are deliberately not multiplied together. A report can therefore
show useful density gaps without inventing a mostly empty Cartesian support.
The `cyclic_target_support_class` marginal is a support-consistency diagnostic.
For example, cyclic C→I is labelled `cyclic_nonring_element` because iodine is
not in `ORGANIC_RING_ELEMENTS`. The census retains the row and makes no support,
mask, or data-exclusion decision.

## Reproducible command

The command below reads every exact admitted training trace, reconciles every
physical shard with the Active8 inventory, checks family row totals, binds all
material inputs and implementation sources by SHA-256, and publishes bytes
without overwrite. It also refuses to produce scientific numbers from a dirty
tree, including untracked inputs. The clean-tree check runs before corpus access. A temporary SQLite
store provides disk-bounded exact distinct-state counting and is removed after
the final report is assembled.

```bash
PYTHONPATH=src python scripts/build_editing_active8_semantic_census_v2.py \
  --active8-inventory /path/to/ACTIVE8_TRACE_INVENTORY.json \
  --unified-manifest /path/to/unified_packed_corpus_manifest.json \
  --audit-root /path/to/audit_layers \
  --mmp-root /path/to/mmp_layer \
  --output results/editing_active8_semantic_census_v2.json
```

No corpus-scale result is checked in by this change because the immutable
source shards are not present in this workspace. Missing execution is reported
as missing evidence, not replaced with synthetic counts.
