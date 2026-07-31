# Editing V2 forensic status, 2026-07-31

Status: development evidence only. This document does not authorize P50 or a
long training run.

## Current outcome

The Active8 editing architecture is not yet fully eligible for mixed training.
Five family-local successor-capacity diagnostics pass. The remaining work is
localized rather than a reason to discard the corpus or framework:

- `atom_insert` learns all 64 bounded teachers as top-1, but misses the frozen
  every-example probability floor and becomes unstable under the frozen
  constant-learning-rate schedule;
- `atom_restate` contains parameter-independent neural-orbit collisions that
  require a train-only corpus census before choosing an architecture or support
  policy;
- aromatic `cycle_attach` currently depends on the stored Kekule phase at the
  executor level, so a scorer-only repair cannot establish a canonical
  molecular-successor kernel.

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

## Atom-restatement boundary

Commit `f921443` adds a train-only, parameter-independent neural-orbit audit.
It mirrors the current six additive message-passing rounds, including the loss
of neighbor-to-edge pairing, executes the current broad-organic restatement
fiber, and computes exact family-conditional successor-probability ceilings.

The committed diagnostic does not narrow restatement support. In particular,
`ORGANIC_RING_ELEMENTS` is not treated as the authoritative generic
atom-restatement vocabulary. Any cyclic-site restriction remains hypothetical
until the exact admitted Active8 train inventory has been audited.

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

This is not yet production semantics. Fused and heteroaromatic coverage, exact
representative determinism, two-step successor-law invariance, and affected
trace counts must pass before integration.

The current preferred integration boundary is prospective, not frozen: add an
explicit semantic edge action and executor rule such as
`CycleOpenEdge(a,b)`/`cycle_open`, while retaining raw `BondDelete` only as a
legacy or internal micro primitive. This prevents old action-codec records from
being silently reinterpreted and avoids changing unrelated exact compilers.

## Required sequence before P50

1. Finish and test the non-authorizing aromatic-opening semantic prototype.
2. Map every executor, enumerator, mask, inverse, compiler, cache, and packed
   derivative affected by the semantic change.
3. Run the frozen Active8 train-only restatement orbit census.
4. Count affected aromatic cycle-open teachers and measure semantic resolver
   coverage without rebuilding unrelated corpus lanes.
5. Freeze the production semantic decision and resulting support identity.
6. Recompile every affected open, its paired close, and every downstream
   multistep trace. Preserve unaffected trace records as immutable source
   evidence, but publish a new process-bound manifest namespace.
7. Rebuild the full successor-fiber cache namespace, candidate headers,
   semantic sidecars, Active8 inventory, and other process-bound derivatives;
   legal fibers change even for rows whose teacher family is not cycle opening.
8. Re-run Gate 0 and family-local T1 from clean committed code.
9. Run a prospective atom-insert optimization follow-up without weakening the
   original negative result.
10. Authorize P50 only if every required cell passes representability, capacity,
   effective exposure, and retention gates.

The molecular sources, raw endpoints, mined pairs, and unaffected trace records
are not broadly invalidated. A semantic cycle-opening change will, however,
require exact replay and reserialization of every affected whole trace because
stored slot-addressed successor states are part of the training contract. It
also changes the process identity globally, so process-bound caches and gate
artifacts must be rebuilt rather than mixed with legacy evidence.
