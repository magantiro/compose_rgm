# Editing V2 semantic cycle-open migration plan

Status: implementation plan. This document does not authorize corpus
publication, P50, or a long training run. The production process identity must
remain unchanged until the overflow-free resolver and its invariance gates are
frozen.

## Decision boundary

The historical `cycle_attach` family executes raw `BondDelete(a, b)`. For a
semantically aromatic edge, the resulting molecule can depend on the stored
Kekule phase. That behavior cannot define the production molecular-successor
kernel.

The prospective process uses an explicit semantic edge action,
`CycleOpenEdge(a, b)`. Its executor must:

1. identify the selected edge using resonance-invariant aromaticity;
2. preserve atoms, formal charges, hydrogen counts, and all unselected
   connectivity;
3. admit only edges with a valid selected-edge-single electronic assignment;
4. produce one canonical molecular successor independent of the input Kekule
   phase and persistent-slot relabeling;
5. reject unsupported or ambiguous edges with typed reason codes;
6. expose only admitted edges in the legal action fiber; and
7. define and verify the corresponding single-bond close inverse.

Raw `BondDelete` remains a legacy or internal micro-operation. Old serialized
actions must never be silently decoded as `CycleOpenEdge`.

## Frozen evidence so far

The validation-only impact artifact has logical SHA-256
`cc1893991b75a4e592d344b592f6f12cf1eef3428781d6b85bddeb4ad75f3cf2`.
It reports:

- 4,023 of 4,331 aromatic teachers admitted by the bounded prototype;
- 304 teachers with no charge/H-preserving forced-single assignment;
- four teacher occurrences censored by the 256-alias audit bound;
- 1,700 of 4,023 admitted teachers with a canonical successor different from
  the historical raw executor;
- at least one admitted opening on 1,426 of 1,427 relevant source molecules;
- all 613 sources containing a no-forced-single edge retain another admitted
  opening.

The no-forced-single outcome is a candidate support boundary. The alias
overflow is not. All overflow edges came from one molecule with multiple
aromatic systems, so irrelevant whole-molecule resonance products must be
removed before production semantics are frozen.

## Minimal scientific rematerialization unit

Primitive cycle-open teachers occur only in the `cycle_ops` layer. General
corruption explicitly excludes primitive bond insertion/deletion, and the MMP
compiler accepts only atom insert/delete/restate, bond reorder, and bond
reroute.

The migration unit is one historical open record and its paired close:

1. decode the immutable exact source state without replaying the old action;
2. verify the legacy source identity, endpoints, and non-bridge condition;
3. construct and execute `CycleOpenEdge(a, b)`;
4. if admitted, serialize the new exact and canonical successor;
5. derive the close action from the semantic result and verify that executing
   it returns the original source molecule canonically;
6. publish an explicit pair identifier and legacy shard/entry lineage; and
7. reconcile every historical open and close exactly once.

If an edge is rejected, exclude both the open and its paired close with one
typed ledger entry. Do not select a replacement edge merely to preserve the
old count. A replacement would be new data generation rather than faithful
rematerialization.

The old packed precursor is derivative evidence and must not define the new
semantic successor.

## Artifact boundary

Preserve as immutable inputs:

- raw molecular sources and licenses;
- source/scaffold partitions;
- frozen per-layer source manifests;
- mined MMP endpoints and raw MMP pool;
- historical traces, audits, checkpoints, and results.

Physically rebuild:

- all `cycle_ops` raw trace and audit shards;
- the cycle pair-migration and typed-rejection ledger;
- all packed `cycle_ops` exact-state shards and manifests.

Replay-certify into the new process namespace without chemical regeneration:

- general corruption raw and packed trace bytes;
- MMP raw and packed trace bytes.

The adoption certificate must bind old and new byte hashes, source manifests,
partitions, trace counts, trace identities, exact states, actions, and the
proof that the layer contains no changed action family. Without that
certificate, existing tooling must rebuild the layer because its global codec
and operator identities are stale.

Rebuild globally after publishing the new unified process manifest:

- representability and Active8 inventories;
- candidate headers, semantic sidecars, and training-lane materializations;
- every successor-fiber cache, including atom and MMP teachers;
- Gate 0 contracts and evidence;
- T1 panels and cache receipts;
- deterministic Stage-A streams and all later pilot authorities.

Global successor-cache rebuilding is mandatory because `CycleOpenEdge`
changes the legal molecular neighborhood of any state, even when that row's
teacher belongs to another family.

## Ordered gates

1. Finish the overflow-free semantic resolver prototype.
2. Prove equivalence to the exhaustive oracle wherever the oracle completes.
3. Pass alternate-Kekule, slot-relabeling, fused, heteroaromatic,
   no-forced-single, inverse, and bounded multistep-law tests.
4. Freeze the semantic admission/rejection and exact-representative policies.
5. Add the explicit action, executor, codec, fiber, model mask, sampler,
   teacher router, and canonical-successor evaluator integration.
6. Bind the complete transitive implementation and clean Git revision into the
   process identity.
7. Run a read-only validation migration census and prove exact open/close
   reconciliation.
8. Rematerialize validation `cycle_ops` and replay-certify unchanged validation
   layers into a new namespace.
9. Rebuild the validation unified manifest and Active8 inventory.
10. Repeat the deterministic migration for train only after validation passes.
11. Run the admitted-train restatement and semantic-cell censuses.
12. Build sealed partitions without using their aggregates for design or
    selection.
13. Rebuild all process-bound caches, then rerun Gate 0 and family-local T1.
14. Authorize a Stage-A pilot only after representability, capacity, exact
    exposure, and retention gates pass.

## Failure rules

- Never reinterpret a legacy action code under the new semantic executor.
- Never preserve a rejected trace by silently choosing another edge.
- Never use the 256-alias overflow as evidence of chemical impossibility.
- Never mix old and new process-bound caches or manifests.
- Never use test or controller-validation aggregates to choose the resolver,
  support boundary, semantic cells, or checkpoint.
- Report unreachable controlled tasks explicitly when their desired transition
  is outside the frozen legal successor support.
