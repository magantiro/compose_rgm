# Successor-fiber training integration design

Date: 2026-07-30

Status: design complete; production trainer unchanged; no compute launched

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

The current example does **not** retain the selected record index or progress
index. Once returned, there is no safe way to join a corpus-addressed fiber.
Re-running the RNG in a wrapper would duplicate the sampling implementation and
create a drift risk.

### Packed corpus

The packed corpus stores the required address fields in every trace envelope,
but `decode_packed_trace` currently drops them. `PathRecord` retains only
`target_key` and `path`.

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
rates, logits or probabilities. Its current `record_at` lookup is linear and
must not be used as-is in a training hot path.

## Exact interface changes required

### 1. Preserve immutable packed trace addresses

File: `src/compose_v4/data/packed_trace_store.py`

Add a dependency-light dataclass:

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

The addressed reader must verify:

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

### 4. Add an O(1), lazy sharded cache index

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

`ShardedSuccessorFiberCache.require(address, progress)` must:

- select the artifact by packed-shard SHA, never by a permissive directory glob;
- validate the expected cache hash and complete provenance;
- lazily load a bounded number of cache shards per worker;
- build an in-memory dictionary keyed by
  `(layer, partition, trace_id, progress_index)`;
- verify the returned cache source key against the sampled state;
- fail on missing, duplicate or extra trace-progress rows.

The current linear `SuccessorFiberCache.record_at` can either be replaced by a
validated internal dictionary or wrapped by this indexed store. A linear scan
over up to 142,482 MMP progress rows per example is not acceptable.

Do not key this cache by absolute training-stream index. Absolute-index caches
duplicate repeated trace/progress draws and become invalid when seed, horizon
or batch size changes.

### 5. Keep fibers beside, not inside, `FactorizedMarkBatch`

New file: `src/compose_v4/experiments/factorized_successor_data.py`

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
- sampled canonical source key equals cached source key;
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

1. **Address loss:** packed trace IDs are present on disk but discarded by the
   current loader.
2. **No whole-corpus byte identity:** the unified short checksum does not bind
   every packed shard byte.
3. **Cache hot lookup:** current lookup is linear.
4. **Support/scoring coupling:** the current compiler traverses a scored law;
   support invariance to weights/time is expected but not yet formally tested.
5. **Canonicalizer environment:** current short source hashes do not explicitly
   bind the RDKit version.
6. **Semantic cells:** balanced-cell checkpoint selection requires a frozen
   per-example cell sidecar not currently present in `FactorizedMarkBatch`.
7. **Validation artifact:** the existing 535.4 MiB evaluation batch lacks
   successor fibers and terminal supports.
8. **Build cost:** the corpus contains 3,370,821 progress states. Measure
   compilation throughput and exact-state/fiber deduplication on a bounded
   sample before launching a 63-shard CPU build.

## Implementation and verification order

1. Preserve packed addresses and add address round-trip/uniqueness tests.
2. Add `record_index`/`progress_index` without changing any sampled field or RNG
   stream.
3. Add the lazy O(1) sharded cache inventory/store.
4. Add successor dataset/collator/batch alignment tests, including multiworker,
   pinning, subbatching and resume.
5. Prove support invariance across time/model initialization.
6. Build a tiny cache from real corruption, cycle-open/close and MMP traces;
   compare every loaded fiber to fresh production compilation.
7. Add successor metrics and successor-level checkpoint selection.
8. Run a CPU dry launch with one real batch and zero optimizer steps.
9. Benchmark a bounded cache compile and one 50-step pilot.
10. Only after the fail-closed gates pass, authorize 500 and 2,000 steps.

No long training or full cache build is authorized by this design.
