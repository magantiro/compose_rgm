# Atom-restatement neural-orbit audit V1

This is a prospective, train-only, non-authorizing diagnostic. It asks whether
the current six-round additive node encoder can distinguish the legal generic
atom-restatement marks that lead to different canonical molecular successors.
It does not choose or apply a support policy.

The current behavior remains broad-organic generic restatement at every real
site, including cyclic sites, subject to the existing executable mask. Two
explicitly hypothetical views are reported for comparison: restricting cyclic
targets to the declared `ORGANIC_RING_ELEMENTS` set, and retaining acyclic
sites only. That declared set is not authoritative for generic atom
restatement, and it must not be confused with the narrower currently wired
macro ring head.

## Exact certificate

The initial node label is:

```text
(atom type, formal charge, implicit H count, atom topology)
```

Each round refines it with:

```text
(self label,
 multiset of neighbor labels,
 multiset of incident neural edge classes)
```

The neighbor-to-edge pairing is intentionally absent. This matches the current
encoder, which separately sums transformed neighbor states and edge embeddings.
After six rounds, the candidate certificate adds the target atom-valence class.
Equal certificates therefore guarantee equal atom-restatement logits for every
parameter setting of the current information path. They do not assert that all
unequal certificates are distinguishable.

For each certificate class, the audit executes every mark with the production
rewrite system, canonicalizes every successor, and records successor
multiplicity. If a teacher successor occupies `t` of `m` marks in a tied class,
that class contributes a family-conditional ceiling of `t/m`. The overall
ceiling is the maximum such fraction among classes containing the teacher
successor.

## Implementation boundary

The tested core and prospective contract are:

```text
src/compose_v4/experiments/atom_restate_neural_orbit_audit.py
configs/editing_atom_restate_neural_orbit_audit_v1.json
tests/test_atom_restate_neural_orbit_audit.py
```

The core requires an immutable train packed-trace progress address, exact source
and stored-successor states, and an executable teacher action. Row aggregation
retains every address whose current-policy ceiling is below one.

The full-corpus execution layer is:

```text
src/compose_v4/experiments/atom_restate_neural_orbit_full_corpus.py
scripts/run_atom_restate_neural_orbit_full_corpus.py
modal_apps/run_atom_restate_neural_orbit_full_corpus.py
configs/editing_atom_restate_neural_orbit_full_corpus_v1.json
tests/test_atom_restate_neural_orbit_full_corpus.py
```

## Frozen-input blocker

The previously supplied Active8 identity was checked directly against the
existing `compose-v4-artifacts` volume. The immutable object is:

```text
/artifacts/active8_trace_inventory_legacy3_validation_1ef8941/
  objects/manifests/
  bbf44483c4d14bfd6c4bde7cbd2dc4fc708c46a14acff5eadb0b6f82902832ab.json
```

Its physical SHA-256 is `bbf44483...`, its logical inventory SHA-256 is
`ce1a1b0...`, and its internal source, support, and unified-manifest identities
match the previously supplied values. However, it contains exactly four
`validation` shards, 31,815 admitted validation traces, and zero train shards.
It is not a train inventory and cannot be relabelled as one.

The production full-corpus contract therefore has status
`BLOCKED_MISSING_FROZEN_ACTIVE8_TRAIN_PARENT`, contains no frozen train parent,
and authorizes neither audit execution nor training. The executable driver is
implemented and tested against exact tiny fixtures, but the production entry
point fails before dispatch until a separate prospective contract binds a real
train inventory.

Once that parent exists, the driver resolves every train shard from the exact
unified manifest and requires an exact matching entry in the Active8
inventory. It verifies each packed shard, sidecar manifest, provenance overlay,
admission decision, entry count, and admitted atom-restatement teacher count.
It refuses validation, controller-validation, test, arbitrary, or legacy shard
sets.

Rows are streamed to deterministic gzip JSON Lines. Aggregates retain bounded
address examples rather than all rows in memory. The row artifact and summary
are written in a temporary directory and published together by one atomic
directory rename. The summary binds every material input hash, every resolved
train shard identity, the clean Git revision, the complete implementation-file
hash set, software versions, row evidence hash, and its own logical hash.

The local command shape after the blocker is resolved is:

```bash
PYTHONPATH=src .venv/bin/python \
  scripts/run_atom_restate_neural_orbit_full_corpus.py \
  --active8-inventory <exact-active8-inventory.json> \
  --unified-manifest <exact-unified-packed-manifest.json> \
  --audit-root <resolved-general-and-cycle-root> \
  --mmp-root <resolved-mmp-root> \
  --output-dir <new-content-addressed-result-directory>
```

The worktree must be clean and contain core revision
`f921443193038be423383074a96f834de72f5123`. The output directory must not
exist. This audit has not been run over the full corpus by this implementation
change. Its contract explicitly does not authorize T1, P50, support selection,
or training.

## Resumable Modal execution

The Modal wrapper mounts the existing `compose-v4-artifacts` volume without
permission to create a replacement volume. The existing source defaults are:

```text
/artifacts/UNIFIED_PACKED_MANIFEST.json
/artifacts/edit_packed_v1
/artifacts/mmp_packed_v1
```

There is intentionally no default Active8 train inventory path. It must be
provided explicitly after the exact train inventory is frozen.

Each CPU map task owns exactly one frozen train shard. Its task identity is a
hash over the location-independent shard identity, full source revision,
implementation-file hashes, software versions, both audit contracts, and the
five-part Active8 parent. A successful task publishes one deterministic row
stream and one self-hashed receipt under:

```text
/artifacts/_frozen_corpus_audits/atom_restate_neural_orbit_full_corpus_v1/
  runs/<run-identity>/map/<task-identity>/
```

Retries skip only receipts whose directory contents, receipt self-hash, task
binding, row hash, and row count all verify. A corrupt receipt fails loudly
instead of being overwritten. The reducer refuses missing or unexpected task
objects, revalidates every row address against its source shard, and atomically
publishes the final evidence under `runs/<run-identity>/final/`.

The eventual launch command, to be used only after the contract is unblocked
and from a clean committed tree, is:

```bash
MODAL_PROFILE=nitya modal run --detach \
  modal_apps/run_atom_restate_neural_orbit_full_corpus.py \
  --active8-inventory-path <exact-train-inventory-object-path>
```

The wrapper performs no training and has not been launched by this change.
Before dispatch, it binds the clean local Git revision and implementation hash;
each remote task recomputes the copied implementation hash and refuses a
snapshot mismatch.

## Smallest remediation path

The repository already contains the required production admission builder:

```text
modal_apps/build_active8_trace_inventory_app.py
src/compose_v4/data/active8_inventory_mapreduce.py
src/compose_v4/data/active8_trace_inventory.py
```

It uses the exact production candidate checker and whole-trace admission. Its
map tasks are immutable half-open ranges of physical source shards, persist
content-addressed decision objects and receipts to the artifact volume, and
reuse only verified receipts. A retry therefore does not repeat completed CPU
scans. Its reducer refuses missing, overlapping, or unexpected tasks.

The frozen unified manifest contains 55 train shards: 36 general-corruption,
3 cycle-operation, and 16 matched-pair shards. Keep the tested 500-entry range
size rather than collapsing each physical shard into one task. Each range is
independently content-addressed and resumable, while the reducer reconstructs
one exact logical decision stream per physical shard. This bounded granularity
limits retry loss and reduces timeout exposure for the most expensive candidate
enumeration ranges. The post-cycle-open validation build must confirm the range
runtime envelope before the 55-shard train dispatch.

The prospective train-only command shape is:

```bash
MODAL_PROFILE=nitya COMPOSE_ACTIVE8_MAX_MAP_CONTAINERS=64 \
  modal run --detach modal_apps/build_active8_trace_inventory_app.py \
  --unified-manifest-path /artifacts/UNIFIED_PACKED_MANIFEST.json \
  --audit-root /artifacts/edit_packed_v1 \
  --mmp-root /artifacts/mmp_packed_v1 \
  --support-contract-path /root/compose/configs/<post-cycle-open-frozen-runtime-contract>.json \
  --partitions train \
  --target-entries-per-range 500 \
  --output-root <new-content-addressed-train-inventory-root>
```

This is deliberately not an executable command until the cycle-opening
process, its complete transitive implementation identity, and the resulting
runtime contract are frozen. The current runtime-v2 contract still binds the
legacy cycle-opening process and must not be used for the 55-shard build. This
command has not been launched here. After the post-freeze strict reducer
succeeds, record the resulting manifest-file hash, logical inventory hash,
effective source-corpus cache hash, support-contract hash, and unified-manifest
hash. Only then may a prospective contract replace the null train parent and
authorize the diagnostic run.
