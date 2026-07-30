# Successor-fiber training integration design

Date: 2026-07-30

Status: address/cache/dataset bridge, indexed backend, parent-validated
DataLoader worker receipts, shared successor objective, and bounded
state-centric compiler implemented and focused-tested; no training compute
launched; full-corpus cache build remains blocked

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

The bounded core data path is now implemented:

- `PackedTraceAddress` preserves the exact packed-shard SHA-256, immutable
  entry index and trace envelope through `PathRecord`;
- `FactorizedMarkExample` exposes the already-selected record and progress
  indices without adding an RNG call;
- `successor_fiber_cache.py` schema v3 binds the complete packed source/target
  envelope plus exact persistent-slot source and target SHA-256 identities and
  provides an O(1) primary lookup keyed by
  `(packed_shard_sha256, entry_index, progress_index)`;
- `sharded_successor_fiber_cache.py` provides a self-hashed, provenance-bound,
  immutable inventory and selects either the JSON oracle or indexed SQLite
  backend for development panels;
- `factorized_successor_data.py` performs a fail-closed post-draw join and
  returns aligned `FactorizedSuccessorBatch` objects;
- `successor_fiber_cache_builder.py` groups complete addressed shard
  occurrences by exact persistent-slot source state, compiles the full static
  successor map once per unique state, replays every stored teacher step
  against the production executor, emits only the requested fibers/support,
  and discards the transient map.

The inventory is explicitly `BOUNDED_DEVELOPMENT_ONLY`, regardless of storage
backend. It cannot authorize a full-corpus run. The 63-shard build waits for a
production-sized real-inventory DataLoader benchmark, a resumable parallel
build orchestrator, stratified support invariance, and a frozen complete-corpus
inventory.

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

Before worker creation,
`prepare_indexed_worker_open_receipts()` can fully validate every indexed shard
once in the trusted parent. Its same-launch receipt binds the resolved path,
device/inode/size/mtime/ctime, expected hashes and provenance, exact validated
metadata, resource limits, and packed-entry census. Forked or spawned workers
must then reopen query-only and recheck file identity, application ID, schema,
metadata, and census. They do not repeat file hashing, `quick_check`, or the
semantic row scan. Once receipt mode is prepared, a missing or mismatched
receipt is an error; there is no silent fallback.

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

1. **Production storage qualification:** deterministic JSON remains the
   correctness oracle, and an immutable indexed SQLite backend now passes
   bounded single-process and direct fork/spawn process-isolation,
   throughput, RSS and mutation checks without materializing shard-sized
   Python object graphs. The actual two-worker spawn `DataLoader` path also
   passes with full-scan functions deliberately disabled in workers, proving
   validation reuse on a bounded fixture. Full qualification still requires a
   production-sized sharded DataLoader startup/throughput/RSS measurement and
   one complete real-shard census.
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
7. **Build cost:** the corpus contains 3,370,821 progress states. The bounded
   state-centric compiler now realizes the measured exact-state reuse and
   proves output equality against record-by-record compilation. A resumable
   parallel build orchestrator and its failure/restart inventory have not yet
   been implemented. No 63-shard build is authorized.

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
7. ~~Benchmark the indexed backend on a bounded 170,000-row artifact, qualify
   direct fork/spawn lookup isolation, and prove on the actual sharded-cache
   `DataLoader` path that workers reuse parent validation.~~ Measure
   production-sized real-inventory DataLoader startup, throughput and RSS,
   then freeze the storage backend and complete-corpus inventory.
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

## First real-shard smoke finding

A bounded read-only smoke used the frozen `shard_0000` artifacts from all three
training lanes:

- corruption SHA-256
  `d5dce3f31d46ea18f82838a91ceb143f852f8d3bc83248ff45c7028931739915`;
- cycle-operation SHA-256
  `482afb090817b333813ef9689bfa71f49b61c8cff4e51fa41c2207303a887fcc`;
- MMP SHA-256
  `02040c6ab6208971c3d21730c0d467280e5b6bb6c45439dde25fe5ccdd0b3248`.

The first pass found that two of the first eight corruption traces reached the
null state and failed fresh support compilation.  The broad-organic
`grow_root` mask had admitted neutral hypervalent classes whose implied
zero-heavy-bond hydrogen counts were five or six, above the representation's
`MAX_H_COUNT=4`.  The executor correctly rejected those marks.  The mask now
requires the isolated atom's implied hydrogen count to lie within the same
bound; those valence classes remain available for connected insertion and
restatement when heavy bonds reduce the implied hydrogen count.  The identical
eight-trace sample then compiled all 36 progress states without failure, and a
null-source regression executes every marked root action through the production
executor.

This changes only null-state root support, but it means any checkpoint used for
the separate de novo lane must be evaluated under the corrected mask.  It is
also direct evidence that the full cache census and zero-step support gate must
precede training.

Preliminary single-core compilation on eight traces per lane was only about
0.7--1.6 progress states/second, while canonical JSON occupied roughly
0.62--0.66 KiB per compiled progress row.  These are bounded development
measurements, not a production throughput claim.  They rule out a naive serial
3.37-million-state build and reinforce the requirement to benchmark parallel
compilation plus indexed, non-materializing storage before authorizing all 63
shards.

## Indexed storage and exact-state reuse measurements

The indexed schema stores one immutable SQLite artifact per packed shard and
resolves rows by `(entry_index, progress_index)`. It now independently verifies
the byte SHA-256, exact SQLite schema, resource bounds, recomputed semantic
content hash, derived census, complete trace chains, full packed source/target
envelope, and persistent-slot state identity. Read handles are process-local,
pickle-safe and bounded by a decoded-row LRU. File identity is checked before
and after every lookup, including device, inode, size, modification time and
change time. The sharded inventory remains
`BOUNDED_DEVELOPMENT_ONLY`; this implementation does not itself authorize
training.

A 170,000-row storage stress artifact was constructed by repeating complete
records from the corrected 36-state real-corruption smoke under new immutable
addresses. It is a storage/index stress test, not a representative chemistry or
alias-distribution sample. Under indexed schema v2:

- file bytes: 45,674,496;
- bytes per progress row: 268.67;
- deterministic write: 3.339 seconds, 50,913 rows/second;
- full byte/schema/semantic/census verification on open: 7.181 seconds;
- 100,000 forced uncached single-process lookups: 16,734/second;
- semantic content SHA-256:
  `477474d2e8bd375cd41b939e000b27aa00d8657ec4e9db352a8d5882008623ed`;
- file SHA-256:
  `105fcd28ab6fb32a8b43039543ab9b564d71fb4d0206afa789fa8f6358ec489d`.

The lookup margin is sufficient for a bounded pilot. The 7.2-second open is an
intentional full-audit cost; before full-corpus use, parent-process
verification must be structured so each worker does not repeat every semantic
scan.

Three-repeat process qualification then partitioned the same fixed 170,000
exact lookups across direct, fork and spawn workers. Throughput is computed
from the slowest worker's synchronized lookup interval:

| Start mode | Workers | Median lookups/s | Range | Speedup | Ready time | Full lifecycle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| direct | 0 | 19,028 | 18,932--19,064 | 1.00x | -- | -- |
| fork | 4 | 67,527 | 67,396--67,561 | 3.55x | 0.015 s | 2.541 s |
| fork | 8 | 119,832 | 118,727--121,156 | 6.30x | 0.020 s | 1.447 s |
| spawn | 4 | 68,473 | 68,427--69,033 | 3.60x | 4.464 s | 7.542 s |
| spawn | 8 | 121,129 | 121,112--122,635 | 6.37x | 8.556 s | 10.535 s |

All 72 workers exited successfully. Forked handles replaced the inherited
connection on a guaranteed first miss; spawned and explicitly pickled handles
started without a connection or decoded LRU; every reopened connection was
owned by the worker PID; every requested address matched; synthetic
device/inode/size/mtime/ctime changes failed closed; and the source artifact's
bytes and identity were unchanged after all trials.

Those trials are process/backend qualification rather than a
production-DataLoader result: they passed an already fully validated handle,
used a warm page cache, and measured a synthetic at-most-two-alias workload on
macOS.

The sharded wrapper now closes the validation-reuse gap. A fully validated
parent creates immutable same-launch receipts, and fork/spawn process-transfer
tests plus an actual two-worker spawn `torch.utils.data.DataLoader` test
deliberately poison file hashing and semantic row-audit functions inside each
worker. All 512 requested fixture rows resolve exactly, showing that workers do
not repeat the full scans. File drift and lost receipts fail without fallback.
Across the affected cache/data suite, 102 tests pass.

One warm-cache diagnostic on the same 170,000-row artifact measured a
21.64-second full verified parent open under the then-current host load and a
1.15-millisecond median receipt open over 20 trials. This comparison isolates
open-path reuse; it was not run through DataLoader and is not a production
throughput result. A production-sized sharded DataLoader startup,
throughput/RSS measurement on the frozen real inventory therefore remains
blocking.

A separate read-only census over the three representative real shards used
only `persistent_slot_state_sha256`; it did not invoke RDKit, canonicalization
or executor replay:

| Lane | Progress states | Unique exact states | Exact support calls saved | Jumps | Unique exact directed pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| corruption | 8,187 | 4,180 | 48.94% | 6,100 | 6,032 |
| cycle operations | 170,000 | 53,126 | 68.75% | 85,000 | 85,000 |
| MMP analogues | 142,482 | 129,963 | 8.79% | 121,795 | 116,032 |
| combined | 320,669 | 185,924 | 42.02% | 212,895 | 207,027 |

Thus a state-centric compiler can reduce 320,669 support enumerations to
185,924 on these shards. Caching only exact directed teacher pairs is much less
valuable (2.76% combined reuse), while cross-lane sharing adds only 1,345 state
hits. These are exact call-count reductions, not claimed wall-clock speedups.
The safe implementation is to enumerate and group all canonical successors
once per exact source state, emit only the requested teacher fibers for that
state's trace occurrences, then discard the full successor map. The bounded
builder now follows that design, preserves record-by-record output equality,
and additionally proves that every stored teacher action itself executes to
the stored exact next state **and** that its full canonical action identity is
present in the enumerated marked support paired with that exact output. Action
identities remain transient and are not written to cache rows. The remaining
engineering work is resumable, parallel shard orchestration rather than
another successor-cache semantics change.
