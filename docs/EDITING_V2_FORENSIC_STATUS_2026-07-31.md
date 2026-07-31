# Editing V2 forensic status, 2026-07-31

Status: development evidence only. This document does not authorize P50 or a
long training run.

## Current outcome

The Active8 editing architecture is not yet eligible for mixed training.
Atom-insert now passes its prospective optimization-only successor-capacity
follow-up, and the validation-only aromatic-opening audit has localized the
production semantic defect. The remaining work is localized rather than a
reason to discard the corpus or framework:

- `atom_restate` contains parameter-independent neural-orbit collisions that
  require a train-only corpus census before choosing an architecture or support
  policy;
- aromatic `cycle_attach` currently depends on the stored Kekule phase at the
  executor level. The semantic prototype preserves source-level topology
  reachability on nearly every relevant validation source, but it changes many
  historical canonical successors and therefore requires an explicit process
  revision rather than a scorer-only repair;
- the Stage-A capability objective now has an isolated balanced semantic-cell,
  canonical-successor implementation, but it is not yet wired to an authorized
  deterministic training stream.

No result in this document weakens a frozen threshold. A failed check remains a
failed check.

## Atom-insert V2 result

The immutable Modal result is:

```text
run_label:
  editing-t1-uniform-v2-atom-insert-464642c-v1

source_commit:
  464642c4ea58edc55e5e11ec15d8fea163f0e899

result_relative_path:
  _editing_t1_uniform_successor_capacity_v2/
  editing-t1-uniform-v2-atom-insert-464642c-v1/
  unique_state.atom_insert.all.json

result_sha256:
  ef165330fcb674f56a49c71ce1e671fa5b3563e228a16b07d0c8e29faeca88c6

uniform_v2_contract_sha256:
  bd96dc3ac19e37daf0ec249e9470a8e12d6d3f699f9ec392dc9b57313d0ff60a

Active8_inventory_file_sha256:
  bbf44483c4d14bfd6c4bde7cbd2dc4fc708c46a14acff5eadb0b6f82902832ab
```

The independently validated selected state is update 495:

- mean canonical-successor NLL: `0.017092175781726837`;
- minimum teacher-successor probability: `0.5790815223223192`;
- teacher successor top-1: `64/64`;
- examples below the frozen `0.8` probability floor: `2/64`.

The terminal update 500 is not an acceptable substitute for the selected
state. Its mean NLL is `2.94991397857666` and its minimum teacher probability is
`1.1982774772434368e-08`. The update 496 through 500 collapse demonstrates
schedule instability. It does not convert the selected state into a pass.

Interpretation: insertion is broadly learnable under the current factorization,
but the frozen optimization recipe did not satisfy the per-example capacity
gate. A prospective optimization-only follow-up may change the schedule; it may
not relabel this result.

## Atom-insert V3 result

The prospective schedule-only follow-up replayed the first 495 updates under
the exact V2 implementation identity and optimizer state, then lowered the
learning rate from `1e-3` to `1e-4`. It retained the panel, objective, unit
weights, seed, hazard exclusion, and every frozen threshold. The immutable
result is:

```text
run_label:
  editing-t1-atom-insert-v3-ed3a91b-v1

source_commit:
  ed3a91bb7238359e7956e801f6315dc815471cbb

result_relative_path:
  _editing_t1_atom_insert_schedule_v3/
  editing-t1-atom-insert-v3-ed3a91b-v1/
  unique_state.atom_insert.all.json

result_file_sha256:
  4dc752e8c5e2bd225094f59ac3015641cf86f9e9fcc247bd146363ef5703a4dc

result_logical_sha256:
  c9d3a59a691dce5e134f584bf47e071a947afae6c38443b2e8ef60de726310e6
```

The independently hash-verified terminal and selected state is update 519:

- mean canonical-successor NLL: `0.0074065253138542175`;
- mean teacher-successor probability: `0.9932336211204529`;
- minimum teacher-successor probability: `0.8104637897749241`;
- maximum per-example teacher-successor NLL: `0.21014861520797484`;
- teacher successor top-1: `64/64`;
- optimizer steps with nonzero gradient: `519`;
- required declared components without a nonzero gradient: none.

All frozen checks passed, including the every-example probability floor of
`0.8` and corresponding NLL ceiling of `-log(0.8)`. The early-stop condition
proved at update 519, only 24 low-rate updates after the V2 selected state.
Update 518 still failed with a minimum probability of `0.7913575172438468`, so
the bottleneck margin is narrow and no post-pass plateau has been demonstrated.
Hazard remained excluded. This establishes bounded-panel atom-insert capacity
under the prospective schedule; it is not evidence of held-out generalization,
mixed-family behavior, or production calibration and does not authorize P50.
The artifact is also bound to the isolated exact-V2 replay commit, not to the
eventual integrated semantic-cycle-open process, so it is capacity evidence
rather than a candidate production checkpoint.

## Atom-restatement boundary

Commit `f921443` adds a train-only, parameter-independent neural-orbit audit.
It mirrors the current six additive message-passing rounds, including the loss
of neighbor-to-edge pairing, executes the current broad-organic restatement
fiber, and computes exact family-conditional successor-probability ceilings.

The committed diagnostic does not narrow restatement support. In particular,
`ORGANIC_RING_ELEMENTS` is not treated as the authoritative generic
atom-restatement vocabulary. Any cyclic-site restriction remains hypothetical
until the exact admitted Active8 train inventory has been audited.

## Training and artifact safeguards

Commit `118f0e9` adds the isolated Stage-A capability objective. It requires a
frozen semantic-cell identifier on every nonterminal row, ignores legacy
importance coefficients by design, averages successor NLL within each cell and
then equally across represented cells, and has no hazard gradient. Existing
production-weighted objectives remain unchanged. This is the correct objective
primitive, but deterministic cell-balanced scheduling, coefficient/gradient
exposure accounting, trainer integration, and runtime authorization remain
separate gates.

Commit `ac06281` advances the Active8 map/reduce lineage schemas and binds each
future plan, task, receipt, inventory, and completion artifact to a clean Git
commit and tree plus a transitive 72-source implementation snapshot. The
closure includes the factorized model, legal fiber, executor, operator, kernel,
codec, canonicalization, and cache semantics. Dirty or off-revision workers and
legacy narrow-lineage receipts now fail closed. No inventory was rebuilt by
this change.

## Cycle-opening boundary

Commit `2be830a` adds a fail-closed alternate-Kekule and persistent-slot
invariance gate. It establishes a model-independent no-go on asymmetric
aromatic fixtures: two exact Kekule encodings with the same canonical molecule
and the same resonance-invariant neural view can induce disjoint canonical
cycle-open successor support under raw `BondDelete` execution.

Commit `bd5b0d4` adds an optional contextual exact-order scorer as a diagnostic
capacity probe. The legacy default and state-dict layout remain unchanged. The
probe is not a production proposal: raw exact order can distinguish alternate
representations of one molecule, so the invariance gate blocks its promotion.

Commit `ed8a061` adds a non-authorizing prototype of the smallest principled
repair under investigation, an edge-anchored aromatic opening resolver:

1. identify a semantically aromatic selected edge;
2. enumerate charge-, hydrogen-, atom-, connectivity-, and source-identity-
   preserving Kekule aliases without silent truncation;
3. retain aliases where the selected edge is single;
4. delete that single edge and require all valid products to collapse to one
   canonical successor;
5. reject unsupported or ambiguous edges explicitly;
6. use single-bond insertion as the corresponding inverse.

The frozen validation-only impact audit is now complete and independently
hash-verified:

```text
result_relative_path:
  _frozen_corpus_audits/
  active8_aromatic_cycle_open_impact_v1/
  result.5ae18096d7fcb0cf0e09e9040d6153ee264e3da2ae4a958aa83a74d37f5c2df4.json

result_logical_sha256:
  cc1893991b75a4e592d344b592f6f12cf1eef3428781d6b85bddeb4ad75f3cf2
```

Among 4,331 perceived-aromatic validation teachers, the prototype admits
4,023 (`92.89%`), rejects 304 because the selected edge has no preserving
forced-single alias (`7.02%`), and reports four bounded-enumeration overflows
(`0.09%`). Among admitted teachers, 1,700 of 4,023 (`42.26%`) produce a
different canonical molecule from the historical raw executor. This confirms
that the repair changes scientific process semantics and that legacy and
repaired artifacts must not be mixed.

At the source level, 1,427 unique sources contain semantic aromatic cycle
edges. The prototype retains at least one admitted aromatic opening for 1,426
of them (`99.93%`). Across all 19,311 semantic aromatic cycle edges on the
1,707 audited cycle-open teacher sources, 17,966 are admitted (`93.03%`). Of
614 sources with at least one rejected edge, 613 retain another admitted edge.
Rejection is concentrated in heteroaromatic contexts: 1,343 of 14,145
heteroaromatic edges are rejected or overflow (`9.50%`), versus 2 of 5,166
nonheteroaromatic edges (`0.04%`). The sole relevant source with no admitted
edge is an overflow at the explicit 256-alias audit cap, not a proved semantic
impossibility.

The audit excludes train, controller-validation, and test, rebuilds no corpus,
and authorizes no training. The remaining production decision is therefore
narrow: eliminate the whole-molecule enumeration overflow without silently
truncating support, then freeze explicit semantic `CycleOpenEdge` admission
and rejection behavior. The high source-level coverage supports the
compositional cycle-open operator; it does not support reverting to a finite
whole-ring-growth catalog.

The current preferred integration boundary is prospective, not frozen: add an
explicit semantic edge action and executor rule such as
`CycleOpenEdge(a,b)`/`cycle_open`, while retaining raw `BondDelete` only as a
legacy or internal micro primitive. This prevents old action-codec records from
being silently reinterpreted and avoids changing unrelated exact compilers.

## Required sequence before P50

1. Remove the whole-molecule alias-overflow boundary from the semantic
   aromatic-opening design, then freeze explicit admission and rejection
   semantics.
2. Map every executor, enumerator, mask, inverse, compiler, cache, and packed
   derivative affected by the semantic change.
3. Implement the semantic action, executor, legal fiber, inverse, and complete
   transitive process identity with invariance and boundary tests.
4. Recompile every affected open, its paired close, and every downstream
   multistep trace. Preserve unaffected trace records as immutable source
   evidence, but publish a new process-bound manifest namespace.
5. Rebuild and verify the validation Active8 inventory under the new process
   identity before launching the larger train derivative.
6. Build the admitted-train Active8 inventory and run the train-only
   restatement orbit census.
7. Freeze the resulting support identity and semantic-cell registry.
8. Rebuild the full successor-fiber cache namespace, candidate headers,
   semantic sidecars, and remaining process-bound derivatives; legal fibers
   change even for rows whose teacher family is not cycle opening.
9. Re-run Gate 0 and family-local T1 from clean committed code.
10. Wire the committed balanced semantic-cell canonical-successor Stage-A
   objective into the deterministic capability pilot, with hazard frozen.
11. Authorize P50 only if every required cell passes representability, capacity,
   effective exposure, and retention gates.

The molecular sources, raw endpoints, mined pairs, and unaffected trace records
are not broadly invalidated. A semantic cycle-opening change will, however,
require exact replay and reserialization of every affected whole trace because
stored slot-addressed successor states are part of the training contract. It
also changes the process identity globally, so process-bound caches and gate
artifacts must be rebuilt rather than mixed with legacy evidence.
