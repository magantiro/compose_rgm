# Successor-fiber training integration design

Date: 2026-07-30

Status: address/cache/dataset bridge implemented and focused-tested; production
optimizer unchanged; no compute launched; full-corpus cache build blocked on a
measured storage-backend decision

## Decision

Canonical-successor fibers must be joined **after** the existing deterministic
layer/record/time/progress draw. The cache is a derivative lookup table, not a
sampler. It must not flatten progress rows into a new training urn.

The minimal safe data path is:

```text
absolute training-stream index
  -> existing seeded FactorizedMarkDataset draw
       layer/record -> time -> progress
  -> ordinary FactorizedMarkExample plus record_index/progress_index
  -> O(1) lookup in the successor cache for that exact packed trace/progress
  -> existing FactorizedMarkCollator
  -> FactorizedSuccessorBatch(
       mark_batch,
       teacher_fibers,
       state_supports,
       cache_addresses,
       optional semantic_cell_ids,
     )
  -> forward_teacher_successor_batch
  -> successor identity loss and separately reported/weighted hazard loss
```

This preserves:

- the configured layer weights;
- hierarchical record/bin selection;
- trace-first sampling;
- the existing time law;
- progress stratification and its importance weight;
- index-deterministic resume behavior;
- the exact `FactorizedMarkBatch` model-input contract.

## What exists now

### Implementation checkpoint

The first four data-path stages are now implemented:

- `PackedTraceAddress` preserves the exact packed-shard SHA-256, immutable
  entry index and trace envelope through `PathRecord`;
- `FactorizedMarkExample` exposes the already-selected record and progress
  indices without adding an RNG call;
- `successor_fiber_cache.py` schema v2 binds exact persistent-slot source and
  target SHA-256 identities and provides an O(1) primary lookup keyed by
  `(packed_shard_sha256, entry_index, progress_index)`;
- `sharded_successor_fiber_cache.py` provides a self-hashed, provenance-bound,
  immutable inventory and lazy bounded JSON LRU for development panels;
- `factorized_successor_data.py` performs a fail-closed post-draw join and
  returns aligned `FactorizedSuccessorBatch` objects;
- `successor_fiber_cache_builder.py` compiles complete addressed development
  traces for correctness tests.

The canonical-JSON backend is explicitly
`BOUNDED_DEVELOPMENT_ONLY`. It cannot authorize a full-corpus run. The 63-shard
build waits for measured JSON/columnar-mmap/SQLite storage results and a frozen
complete-corpus inventory.

### Sampling and batching

`FactorizedMarkDataset.__getitem__` in
`src/compose_v4/experiments/factorized_mark_conditional.py` is the scientific
owner of the streamed draw. For absolute index `i`, it:

1. constructs a seeded RNG from `(seed, i)`;
2. selects a record, using `record_index_sampler.draw(rng)` when configured;
3. samples time;
4. calls `_sample_tracelet_progress`;
5. obtains the exact persistent-slot state and teacher mark;
6. returns `FactorizedMarkExample`.

`FactorizedMarkCollator` then calls `prepare_factorized_mark_batch`. The
resulting `FactorizedMarkBatch` contains tensors, exact states, teacher actions,
teacher rules, rates and importance weights. It has aligned `subbatch`, `to`
and `pin_memory` methods.

The example now retains `record_index` and `progress_index`. Contract tests
verify resume stability and that the successor wrapper returns the same
scientific mark-example fields while making no second sampling call.

### Packed corpus

The packed reader now emits an immutable `PackedTraceAddress`; production
corruption, cycle and MMP loaders retain it in `PathRecord.corpus_address`.
Successor training rejects raw/replayed `PathRecord` objects whose address is
`None`.

Read-only inspection of the connected immutable artifacts found:

| Packed shard | Entries | States | Trace address observed |
| --- | ---: | ---: | --- |
| MMP train shard 0000 | 20,687 | 142,482 | `mmp-0`, layer `mmp_analogue`, partition `train` |
| corruption train shard 0000 | 2,087 | 8,187 | `corruption-train-s0000-000000`, layer `corruption` |
| cycle train shard 0000 | 85,000 | 170,000 | `cycle_ops-train-s0000-000000`, layer `cycle_ops` |

Each inspected record also carries exact `source_key`, `target_key`,
`path_length`, step successor keys, and exact slot states.

The connected corpus has 63 packed shards and 3,370,821 progress states. The
current unified-manifest checksum covers shard names/counts and the
representability identity, but not every packed shard byte hash. Each valid
provenance overlay does carry the exact packed-shard SHA-256.

### Successor bridge and cache

`factorized_successor_training.py` already provides:

- `TeacherSuccessorAlias`;
- `TeacherSuccessorFiber`;
- `StateProductiveSupport`;
- `forward_teacher_successor_batch`;
- canonical-successor Bregman and identity losses.

`successor_fiber_cache.py` now provides deterministic, bounded, fail-closed
serialization of complete trace-progress records. It stores no model weights,
rates, logits or probabilities. It builds an exact O(1) primary index over
packed shard SHA, entry index and progress. Canonical molecular keys remain in
the artifact as scientific quotient identities; hot-path joins use the exact
persistent-slot digest and therefore do not invoke RDKit.

## Exact interface changes required

### 1. Preserve immutable packed trace addresses

File: `src/compose_v4/data/packed_trace_store.py`

Implemented as a dependency-light dataclass:

```python
@dataclass(frozen=True)
class PackedTraceAddress:
    packed_shard_content_sha256: str
    packed_shard_name: str
    entry_index: int
    trace_id: str
    layer: str
    partition: str
    source_key: str
    target_key: str
    path_length: int
```

Add `read_addressed_packed_shard(...) -> Iterator[AddressedPackedTrace]`.
Keep `read_packed_shard` backward compatible by projecting addressed rows back
to its current `(trace, path)` pairs. Both readers must share one internal
decoder; no second packed parser is allowed.

The addressed reader verifies the complete envelope and exact packed-shard
identity. To avoid an RDKit pass over every one of 3.37 million startup states,
endpoint recanonicalization is performed on the reader's frozen
`verify_fraction` audit sample; the exact packed-shard SHA binds the stored
endpoint keys for all other rows.

It verifies:

- all address strings are present;
- stored path length equals the decoded path;
- stored source/target keys equal the exact decoded endpoint keys;
- entry indices are contiguous within the shard;
- trace IDs are unique within the shard;
- the exact packed-shard SHA agrees with its validated provenance overlay.

### 2. Carry the address with each `PathRecord`

File: `src/compose_v4/experiments/cnof_conditional.py`

Backward-compatible extension:

```python
@dataclass(frozen=True)
class PathRecord:
    target_key: str
    path: TraceProgressCTMC
    corpus_address: PackedTraceAddress | None = None
```

Files:

- `src/compose_v4/data/production_edit_corpus.py`
- associated loader tests

Use `read_addressed_packed_shard` and populate `corpus_address`. Existing
mark-level callers may accept `None`; successor training must fail unless every
record is packed and addressed. Do not infer addresses from tuple position.

The sampler's production layer (`general_corruption`, `cycle_operations`,
`mmp_analogue`) and the immutable trace-envelope layer (`corruption`,
`cycle_ops`, `mmp_analogue`) are different namespaces. The cache must use the
immutable envelope layer; the hierarchical sampler keeps its production layer.

### 3. Expose the selected record and progress without changing RNG

File: `src/compose_v4/experiments/factorized_mark_conditional.py`

Append optional audit fields to `FactorizedMarkExample`:

```python
record_index: int | None = None
progress_index: int | None = None
```

In `FactorizedMarkDataset.__getitem__`, assign the selected integer to a local
`record_index`, then return it and the already-sampled `progress`. This
refactor must make **no additional RNG calls** and must leave all existing
example fields byte-identical.

Add a contract test comparing the complete old/new sampled scientific fields
for many absolute indices and verifying resume at `start_step * batch_size`.

### 4. Add an O(1), lazy sharded cache index — bounded implementation complete

New file: `src/compose_v4/data/sharded_successor_fiber_cache.py`

One successor cache artifact should derive from one packed shard. Add a frozen
inventory mapping:

```text
packed_shard_content_sha256
  -> source shard name
  -> successor-cache path
  -> frozen successor-cache content SHA-256
  -> record/progress/alias counts
```

`ShardedSuccessorFiberCache.require(address, progress)` now:

- select the artifact by packed-shard SHA, never by a permissive directory glob;
- validate the expected cache hash and complete provenance;
- lazily load a bounded number of cache shards per worker;
- uses the cache's exact dictionary keyed by
  `(packed_shard_sha256, entry_index, progress_index)`;
- verifies the returned exact persistent-slot source digest against the sampled
  state;
- fail on missing, duplicate or extra trace-progress rows.

The inventory additionally binds the complete active/excluded entry census.
This prevents an internally complete subset of traces from masquerading as a
complete shard derivative. Raw cache-file SHA-256 and byte count are checked
before JSON decoding.

Do not key this cache by absolute training-stream index. Absolute-index caches
duplicate repeated trace/progress draws and become invalid when seed, horizon
or batch size changes.

### 5. Keep fibers beside, not inside, `FactorizedMarkBatch` — implemented

File: `src/compose_v4/experiments/factorized_successor_data.py`

Define:

```python
@dataclass(frozen=True)
class FactorizedSuccessorExample:
    mark_example: FactorizedMarkExample
    cache_record: SuccessorFiberCacheRecord
    semantic_cell_id: str | None = None

@dataclass(frozen=True)
class FactorizedSuccessorBatch:
    mark_batch: FactorizedMarkBatch
    fibers: tuple[TeacherSuccessorFiber | None, ...]
    state_supports: tuple[StateProductiveSupport, ...]
    cache_addresses: tuple[SuccessorFiberCacheAddress, ...]
    semantic_cell_ids: tuple[str | None, ...]
```

The batch must implement aligned `batch_size`, `subbatch`, `to` and
`pin_memory`. Only `mark_batch` tensors move to the accelerator. Fiber
coordinates and source keys remain immutable CPU/Python metadata.

`FactorizedSuccessorDataset` wraps the existing `FactorizedMarkDataset`. It
uses the returned `record_index` and `progress_index` for a cache lookup after
the draw. It performs no sampling.

`FactorizedSuccessorCollator` delegates tensorization to the existing
`FactorizedMarkCollator`, then aligns cache metadata. It must assert:

- nonterminal rate iff a teacher fiber is present;
- terminal rate iff only state support is present;
- sampled persistent-slot source digest equals the cached exact-state digest;
- the cached target equals the next exact trace state;
- explicit support equals `fiber.state_support`.

### 6. Avoid an import cycle

Required dependency direction:

```text
factorized_mark_conditional
  -> model/rewrite modules

factorized_successor_training
  -> model + production successor kernel

successor_fiber_cache
  -> factorized_successor_training types

factorized_successor_data
  -> factorized_mark_conditional + successor_fiber_cache

successor trainer
  -> factorized_successor_data + factorized_successor_training
```

`factorized_mark_conditional.py` must not import
`factorized_successor_data.py`. The successor orchestration belongs in a new
module or behind a dependency-neutral objective protocol.

### 7. Share the optimizer loop without duplicating the sampler

Preferred target:

- define a small dependency-neutral training-objective protocol;
- make the existing loop operate on an objective-provided batch adapter,
  forward/loss function, metrics function and checkpoint-selection key;
- retain the current mark objective as the default;
- implement the successor objective in a module that imports the protocol.

Do not copy the current multi-hundred-line optimizer/resume/checkpoint loop into
a second trainer. Duplicate loops will drift on resume, scheduling, checkpoint
state and early stopping.

The successor objective must call:

```python
prediction = forward_teacher_successor_batch(
    model,
    successor_batch.mark_batch,
    successor_batch.fibers,
    state_supports=successor_batch.state_supports,
)
```

The bounded design comparison must support:

- productive canonical-successor identity NLL;
- canonical-successor generator Bregman;
- separately weighted hazard Bregman.

Hazard and identity metrics must be separate. The editing checkpoint selector
must be `production_weighted_canonical_successor_nll`, with balanced semantic
cell NLL secondary. It must not inherit the current
`factorized_gm_loss` early-stopping rule.

### 8. Build successor-aware validation artifacts

The current shared evaluation artifact stores only `FactorizedMarkBatch`.
Successor validation additionally needs aligned fibers, terminal supports,
cache provenance and semantic-cell IDs.

Create a versioned `FactorizedSuccessorBatch` validation artifact, bound to:

- the exact base validation-batch hash;
- corpus/partition identity;
- successor-cache inventory hash;
- operator/capability/canonicalizer/compiler identities.

Cheap checkpoint-time metrics can compute canonical-successor NLL directly
from cached teacher fibers. Exact successor rank/MRR/top-k requires the full
canonical successor distribution and should remain in the shared production
leaderboard on scheduled checkpoints or a shortlisted set; the teacher-only
cache cannot derive rank by itself.

## Provenance that must be frozen before a cache build

The existing cache provenance is necessary but not yet a complete launch
identity. The inventory must additionally freeze:

- full SHA-256 of `UNIFIED_PACKED_MANIFEST.json`;
- full SHA-256 of `REPRESENTABILITY_OVERLAY.json`;
- each packed shard's exact byte SHA-256;
- each packed manifest and provenance-overlay SHA-256;
- successor cache artifact SHA-256;
- operator registry hash;
- capability/support-signature hash, including vocabulary and ring catalog;
- canonicalizer contract hash, including relevant source hashes and RDKit
  version;
- executor/action-table/compiler implementation hash;
- semantic-cell definition hash;
- cache inventory content hash.

The existing unified `manifest_checksum` is not sufficient because its builder
hashes shard names and entry counts, not packed-shard bytes.

A checkpoint hash is intentionally unnecessary for a support-only cache.
However, the current compiler obtains coordinates through a scored marked-law
enumerator. Before the full cache build, split support enumeration from
probability scoring or prove by test that aliases and virtual support are
identical across time and independently initialized finite models with the
same support contract. Cache contents must not depend on learned weights.

## Unresolved blockers

1. **Production storage backend:** deterministic JSON is the correctness oracle,
   but loading large shards materializes Python object graphs per worker. Measure
   one representative corruption, cycle and large-MMP shard before choosing a
   dense columnar mmap backend or SQLite.
2. **No frozen complete-corpus inventory yet:** the inventory schema now binds
   each packed shard byte hash, cache byte hash, manifest/overlay identities,
   active trace census and declared exclusions, but it has not been built for
   all 63 shards.
3. **Support/scoring coupling:** the current compiler traverses a scored law.
   A one-state invariance regression exists; a stratified multi-family gate or
   support-only enumerator is required before the complete build.
4. **Semantic cells:** balanced-cell checkpoint selection requires a frozen
   per-example cell sidecar not currently present in `FactorizedMarkBatch`.
5. **Validation artifact:** the existing 535.4 MiB evaluation batch lacks
   successor fibers and terminal supports.
6. **Successor optimizer integration:** the shared optimizer loop now accepts a
   typed objective/loader adapter.  The canonical-successor adapter trains the
   productive embedded successor identity (with an explicitly separate
   optional hazard term), evaluates production-weighted and balanced-semantic
   successor NLL, and selects checkpoints by the registered successor metric.
   The historical mark objective remains the default and resume artifacts bind
   the objective identity and selector, so a mark checkpoint cannot silently
   resume as successor training.
7. **Build cost:** the corpus contains 3,370,821 progress states. Measure
   compilation throughput and exact-state/fiber deduplication on a bounded
   sample before launching a 63-shard CPU build.

## Implementation and verification order

1. ~~Preserve packed addresses and add address round-trip/uniqueness tests.~~
2. ~~Add `record_index`/`progress_index` without changing any sampled field or
   RNG stream.~~
3. ~~Add the lazy O(1) sharded development cache inventory/store.~~
4. ~~Add successor dataset/collator/batch alignment tests, including
   multiworker, pinning, subbatching and resume.~~
5. Prove support invariance across a stratified set of times, initializations
   and active families.
6. Build tiny caches from real corruption, cycle-open/close and MMP traces;
   compare every loaded fiber to fresh production compilation.
7. Benchmark representative JSON shards, then freeze the measured production
   storage backend and complete-corpus inventory.
8. ~~Add successor metrics and successor-level checkpoint selection to the
   shared optimizer loop.~~
9. Run a CPU dry launch with one real batch and zero optimizer steps.
10. Freeze numeric sentinel thresholds from development-only panels and run the
   exactly-50-step gradient/collapse pilot.
11. Only after the fail-closed gates pass, authorize 500 and 2,000 steps.

No long training or full cache build is authorized by this design.

The optimizer integration is covered by deterministic fixture tests for a
zero-update dry launch, a real backward/update, checkpoint metadata, and exact
resume.  These tests do not replace step 9: the required artifact must still
use one frozen production-addressed batch.  The P50 implementation now rejects
nonfinite loss/gradients, missing family-specific gradient routes, required
family NLL collapse, a non-contiguous stream, and anything other than exactly
50 updates.  It remains deliberately unrunnable as a scientific PASS until
development-only thresholds and the exact production exposure plan are frozen.
