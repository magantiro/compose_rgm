"""The development cache store is exact, immutable, and fail-closed."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import compose_v4.data.indexed_successor_fiber_cache as indexed_cache_module
from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.chem.persistent_state_identity import (
    persistent_slot_state_sha256,
)
from compose_v4.data.indexed_successor_fiber_cache import (
    write_indexed_successor_fiber_cache,
)
from compose_v4.data.charge_policy import CHARGE_POLICY_VERSION
from compose_v4.data.packed_trace_store import PackedTraceAddress
from compose_v4.data.sharded_successor_fiber_cache import (
    SUCCESSOR_FIBER_INDEXED_STORAGE_BACKEND,
    ShardedSuccessorFiberCache,
    ShardedSuccessorFiberCacheError,
    SuccessorFiberCacheCompatibility,
    SuccessorFiberShardSpec,
    assert_store_is_pickle_safe,
    deserialize_successor_fiber_cache_inventory,
    read_successor_fiber_cache_inventory,
    seal_successor_fiber_cache_inventory,
    summarize_successor_fiber_cache,
    write_successor_fiber_cache_inventory,
)
from compose_v4.data.successor_fiber_cache import (
    SUCCESSOR_FIBER_CACHE_SCHEMA,
    SUCCESSOR_FIBER_CACHE_SCHEMA_VERSION,
    SuccessorFiberCacheAddress,
    SuccessorFiberCacheProvenance,
    SuccessorFiberCacheRecord,
    write_successor_fiber_cache,
)
from compose_v4.experiments.factorized_successor_training import (
    StateProductiveSupport,
    TeacherSuccessorAlias,
    TeacherSuccessorFiber,
)

_UNIFIED_SHA256 = "7" * 64
_OVERLAY_SHA256 = "8" * 64
_MANIFEST_SHA256 = "5" * 64
_PROVENANCE_OVERLAY_SHA256 = "6" * 64


def _forbid_full_indexed_scan(*args, **kwargs):
    raise AssertionError("worker attempted a full indexed-cache scan")


def _worker_lookup_with_full_scans_forbidden(
    store,
    address,
    source_state,
    sender,
) -> None:
    indexed_cache_module._sha256_file = _forbid_full_indexed_scan
    indexed_cache_module._audit_indexed_records = _forbid_full_indexed_scan
    try:
        record = store.require(
            address,
            progress_index=0,
            source_state=source_state,
        )
        sender.send(
            (
                "ok",
                record.source_state_sha256,
                store.worker_open_receipt_digests,
            )
        )
    except Exception as error:  # noqa: BLE001 - relay worker failure
        sender.send(("error", type(error).__name__, str(error)))
    finally:
        sender.close()


class _PreparedReceiptLookupDataset:
    """Small spawn-pickled stand-in for the production training Dataset."""

    def __init__(self, store, address, length: int) -> None:
        self.store = store
        self.address = address
        self.length = length
        self.states = (_state(), _state(target=True))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> int:
        indexed_cache_module._sha256_file = _forbid_full_indexed_scan
        indexed_cache_module._audit_indexed_records = (
            _forbid_full_indexed_scan
        )
        progress_index = index % 2
        record = self.store.require(
            self.address,
            progress_index=progress_index,
            source_state=self.states[progress_index],
        )
        return record.address.progress_index


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _state(*, target: bool = False) -> MolecularGraph:
    atom_types = [1, 1 if target else 0, 0, 0]
    hydrogens = [3 if target else 4, 3 if target else 0, 0, 0]
    bonds = np.zeros((4, 4), dtype=np.int32)
    if target:
        bonds[0, 1] = bonds[1, 0] = 1
    return MolecularGraph(
        atom_types=np.asarray(atom_types, dtype=np.int32),
        formal_charges=np.zeros(4, dtype=np.int32),
        implicit_h_counts=np.asarray(hydrogens, dtype=np.int32),
        bonds=bonds,
    )


def _compatibility() -> SuccessorFiberCacheCompatibility:
    return SuccessorFiberCacheCompatibility.from_payload(
        {
            "successor_cache_schema": SUCCESSOR_FIBER_CACHE_SCHEMA,
            "successor_cache_schema_version": (
                SUCCESSOR_FIBER_CACHE_SCHEMA_VERSION
            ),
            "support_signature": {
                "operator_registry_hash": "operator-v1",
                "capability_flags": [["enable_cycle_ops", True]],
                "max_atoms": 4,
                "atom_insert_arity_support": [0, 1],
                "charge_policy": CHARGE_POLICY_VERSION,
            },
            "ordered_family_vocabulary": [
                "atom_insert",
                "atom_delete",
            ],
            "ordered_action_table_vocabulary": [
                "grow_connected",
                "atom_delete",
            ],
            "coordinate_schema_version": 1,
            "operator_registry_hash": "operator-v1",
            "capability_hash": "capability-v1",
            "canonicalization_version": "canonical-state-key-v1",
            "canonicalizer_contract_sha256": "4" * 64,
            "rdkit_version": "test-rdkit",
            "executor_implementation_hash": "executor-v1",
            "action_enumerator_implementation_hash": "enumerator-v1",
            "fiber_compiler_implementation_hash": "e" * 64,
            "charge_policy_version": CHARGE_POLICY_VERSION,
            "persistent_state_digest_schema": (
                "compose.chem.persistent_slot_state"
            ),
            "persistent_state_digest_schema_version": 1,
            "packed_corpus_schema": "compose.data.packed_trace",
            "packed_corpus_schema_version": 1,
            "trace_codec_schema": "compose.rewrite.trace-v1",
            "trace_codec_schema_version": 1,
            "tensorization_implementation_hash": "tensorization-v1",
        }
    )


def test_cache_compatibility_requires_charge_policy_version() -> None:
    payload = dict(_compatibility().payload)
    payload.pop("charge_policy_version")
    with pytest.raises(
        ShardedSuccessorFiberCacheError,
        match="charge_policy_version",
    ):
        SuccessorFiberCacheCompatibility.from_payload(payload)


def _provenance(
    packed_sha256: str,
    compatibility: SuccessorFiberCacheCompatibility,
) -> SuccessorFiberCacheProvenance:
    payload = compatibility.payload
    return SuccessorFiberCacheProvenance(
        operator_registry_hash=payload["operator_registry_hash"],
        capability_hash=payload["capability_hash"],
        support_signature_sha256=_canonical_hash(
            payload["support_signature"]
        ),
        canonicalization_version=payload["canonicalization_version"],
        canonicalizer_contract_sha256=payload[
            "canonicalizer_contract_sha256"
        ],
        packed_corpus_schema=payload["packed_corpus_schema"],
        packed_corpus_schema_version=payload[
            "packed_corpus_schema_version"
        ],
        packed_shard_content_sha256=packed_sha256,
        packed_manifest_sha256=_MANIFEST_SHA256,
        packed_provenance_overlay_sha256=_PROVENANCE_OVERLAY_SHA256,
        unified_packed_manifest_sha256=_UNIFIED_SHA256,
        representability_overlay_sha256=_OVERLAY_SHA256,
        coordinate_schema_version=payload["coordinate_schema_version"],
        tensorization_implementation_hash=payload[
            "tensorization_implementation_hash"
        ],
        fiber_compiler_implementation_hash=payload[
            "fiber_compiler_implementation_hash"
        ],
    )


def _records(
    packed_sha256: str,
    *,
    shard_name: str,
    entry_index: int = 0,
    trace_id: str = "trace-0",
) -> tuple[SuccessorFiberCacheRecord, ...]:
    source = _state()
    target = _state(target=True)
    source_support = StateProductiveSupport(
        source_key="C",
        source_state_sha256=persistent_slot_state_sha256(source),
        virtual_aliases=(),
    )
    teacher = TeacherSuccessorFiber(
        source_key="C",
        target_key="CC",
        target_state_sha256=persistent_slot_state_sha256(target),
        aliases=(
            TeacherSuccessorAlias(
                "atom_insert",
                "grow_connected",
                (0, 1, 0),
            ),
        ),
        state_support=source_support,
    )

    def address(progress_index: int) -> SuccessorFiberCacheAddress:
        return SuccessorFiberCacheAddress(
            packed_shard_content_sha256=packed_sha256,
            packed_shard_name=shard_name,
            entry_index=entry_index,
            layer="mmp_analogue",
            partition="train",
            trace_id=trace_id,
            trace_source_key="C",
            trace_target_key="CC",
            progress_index=progress_index,
            path_length=1,
        )

    return (
        SuccessorFiberCacheRecord(
            address=address(0),
            state_support=source_support,
            teacher_fiber=teacher,
        ),
        SuccessorFiberCacheRecord(
            address=address(1),
            state_support=StateProductiveSupport(
                source_key="CC",
                source_state_sha256=persistent_slot_state_sha256(target),
                virtual_aliases=(),
            ),
            teacher_fiber=None,
        ),
    )


def _write_shard(
    root: Path,
    *,
    packed_sha256: str,
    compatibility: SuccessorFiberCacheCompatibility,
    name: str,
) -> tuple[SuccessorFiberShardSpec, PackedTraceAddress]:
    shard_name = name.replace(".fibers.json", ".packed.jsonl.gz")
    provenance = _provenance(packed_sha256, compatibility)
    path = root / name
    cache = write_successor_fiber_cache(
        path,
        _records(packed_sha256, shard_name=shard_name),
        provenance=provenance,
    )
    summary = summarize_successor_fiber_cache(cache)
    spec = SuccessorFiberShardSpec(
        packed_shard_content_sha256=packed_sha256,
        packed_shard_name=shard_name,
        layer="mmp_analogue",
        partition="train",
        packed_manifest_sha256=_MANIFEST_SHA256,
        packed_provenance_overlay_sha256=_PROVENANCE_OVERLAY_SHA256,
        cache_relative_path=name,
        cache_content_sha256=cache.content_sha256,
        cache_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        cache_file_bytes=path.stat().st_size,
        packed_entry_count=1,
        active_trace_count=summary["active_trace_count"],
        excluded_entry_indices=(),
        progress_record_count=summary["progress_record_count"],
        jump_record_count=summary["jump_record_count"],
        terminal_record_count=summary["terminal_record_count"],
        teacher_alias_count=summary["teacher_alias_count"],
        virtual_alias_count=summary["virtual_alias_count"],
        maximum_alias_count=summary["maximum_alias_count"],
        provenance=provenance,
    )
    address = PackedTraceAddress(
        packed_shard_content_sha256=packed_sha256,
        packed_shard_name=shard_name,
        entry_index=0,
        trace_id="trace-0",
        layer="mmp_analogue",
        partition="train",
        source_key="C",
        target_key="CC",
        path_length=1,
    )
    return spec, address


def _write_indexed_shard(
    root: Path,
    *,
    packed_sha256: str,
    compatibility: SuccessorFiberCacheCompatibility,
    name: str,
) -> tuple[SuccessorFiberShardSpec, PackedTraceAddress]:
    shard_name = name.replace(".fibers.sqlite", ".packed.jsonl.gz")
    provenance = _provenance(packed_sha256, compatibility)
    path = root / name
    metadata = write_indexed_successor_fiber_cache(
        path,
        _records(packed_sha256, shard_name=shard_name),
        provenance=provenance,
    )
    spec = SuccessorFiberShardSpec(
        packed_shard_content_sha256=packed_sha256,
        packed_shard_name=shard_name,
        layer="mmp_analogue",
        partition="train",
        packed_manifest_sha256=_MANIFEST_SHA256,
        packed_provenance_overlay_sha256=_PROVENANCE_OVERLAY_SHA256,
        cache_relative_path=name,
        cache_content_sha256=metadata.content_sha256,
        cache_file_sha256=metadata.file_sha256,
        cache_file_bytes=metadata.file_bytes,
        packed_entry_count=1,
        active_trace_count=metadata.active_trace_count,
        excluded_entry_indices=(),
        progress_record_count=metadata.record_count,
        jump_record_count=metadata.jump_record_count,
        terminal_record_count=metadata.terminal_record_count,
        teacher_alias_count=metadata.teacher_alias_count,
        virtual_alias_count=metadata.virtual_alias_count,
        maximum_alias_count=metadata.maximum_alias_count,
        provenance=provenance,
    )
    address = PackedTraceAddress(
        packed_shard_content_sha256=packed_sha256,
        packed_shard_name=shard_name,
        entry_index=0,
        trace_id="trace-0",
        layer="mmp_analogue",
        partition="train",
        source_key="C",
        target_key="CC",
        path_length=1,
    )
    return spec, address


def _inventory(
    specs: tuple[SuccessorFiberShardSpec, ...],
    compatibility: SuccessorFiberCacheCompatibility,
    *,
    storage_backend: str | None = None,
):
    options = (
        {}
        if storage_backend is None
        else {"storage_backend": storage_backend}
    )
    return seal_successor_fiber_cache_inventory(
        specs,
        source_corpus_inventory_sha256="a" * 64,
        compatibility=compatibility,
        unified_packed_manifest_sha256=_UNIFIED_SHA256,
        representability_overlay_sha256=_OVERLAY_SHA256,
        **options,
    )


def _store(
    root: Path,
    inventory,
    *,
    max_open_shards: int = 2,
) -> ShardedSuccessorFiberCache:
    return ShardedSuccessorFiberCache(
        root,
        inventory,
        expected_inventory_sha256=inventory.inventory_sha256,
        expected_compatibility_sha256=(
            inventory.compatibility.compatibility_sha256
        ),
        max_open_shards=max_open_shards,
    )


def test_inventory_is_deterministic_self_hashed_and_immutable(tmp_path) -> None:
    compatibility = _compatibility()
    first, _ = _write_shard(
        tmp_path,
        packed_sha256="1" * 64,
        compatibility=compatibility,
        name="first.fibers.json",
    )
    second, _ = _write_shard(
        tmp_path,
        packed_sha256="2" * 64,
        compatibility=compatibility,
        name="second.fibers.json",
    )
    encoded_a, inventory_a = _inventory((second, first), compatibility)
    encoded_b, inventory_b = _inventory((first, second), compatibility)
    assert encoded_a == encoded_b
    assert inventory_a == inventory_b

    path = tmp_path / "inventory.json"
    write_successor_fiber_cache_inventory(
        path,
        encoded_a,
        expected_inventory_sha256=inventory_a.inventory_sha256,
        expected_compatibility_sha256=compatibility.compatibility_sha256,
    )
    loaded = read_successor_fiber_cache_inventory(
        path,
        expected_inventory_sha256=inventory_a.inventory_sha256,
        expected_compatibility_sha256=compatibility.compatibility_sha256,
    )
    assert loaded == inventory_a

    modified = encoded_a.replace(b'"entry_count":2', b'"entry_count":3')
    with pytest.raises(ShardedSuccessorFiberCacheError):
        write_successor_fiber_cache_inventory(
            path,
            modified,
            expected_inventory_sha256=inventory_a.inventory_sha256,
            expected_compatibility_sha256=compatibility.compatibility_sha256,
        )
    assert path.read_bytes() == encoded_a


def test_store_requires_exact_address_and_persistent_state(tmp_path) -> None:
    compatibility = _compatibility()
    spec, address = _write_shard(
        tmp_path,
        packed_sha256="1" * 64,
        compatibility=compatibility,
        name="only.fibers.json",
    )
    _, inventory = _inventory((spec,), compatibility)
    store = _store(tmp_path, inventory)

    record = store.require(
        address,
        progress_index=0,
        source_state=_state(),
    )
    assert record.address.entry_index == 0
    with pytest.raises(
        ShardedSuccessorFiberCacheError,
        match="sampled exact state",
    ):
        store.require(
            address,
            progress_index=0,
            source_state=_state(target=True),
        )
    with pytest.raises(
        ShardedSuccessorFiberCacheError,
        match="absent from the frozen",
    ):
        store.require(
            replace(address, packed_shard_content_sha256="f" * 64),
            progress_index=0,
            source_state=_state(),
        )
    with pytest.raises(
        ShardedSuccessorFiberCacheError,
        match="inventory lane",
    ):
        store.require(
            replace(address, layer="cycle_ops"),
            progress_index=0,
            source_state=_state(),
        )


def test_indexed_store_uses_same_exact_join_without_materializing_shard(
    tmp_path,
) -> None:
    compatibility = _compatibility()
    spec, address = _write_indexed_shard(
        tmp_path,
        packed_sha256="1" * 64,
        compatibility=compatibility,
        name="only.fibers.sqlite",
    )
    _, inventory = _inventory(
        (spec,),
        compatibility,
        storage_backend=SUCCESSOR_FIBER_INDEXED_STORAGE_BACKEND,
    )
    store = _store(tmp_path, inventory)
    record = store.require(
        address,
        progress_index=0,
        source_state=_state(),
    )
    assert record.address.entry_index == 0
    assert store.loaded_shard_digests == ("1" * 64,)
    with pytest.raises(
        ShardedSuccessorFiberCacheError,
        match="no exact sampled",
    ):
        store.require(
            replace(address, trace_id="wrong-trace"),
            progress_index=0,
            source_state=_state(),
        )

    restored = pickle.loads(pickle.dumps(store))
    assert restored.loaded_shard_digests == ()
    assert (
        restored.require(
            address,
            progress_index=1,
            source_state=_state(target=True),
        ).teacher_fiber
        is None
    )

    wrong_lane_spec = replace(spec, layer="another-lane")
    _, wrong_lane_inventory = _inventory(
        (wrong_lane_spec,),
        compatibility,
        storage_backend=SUCCESSOR_FIBER_INDEXED_STORAGE_BACKEND,
    )
    with pytest.raises(
        ShardedSuccessorFiberCacheError,
        match="envelope",
    ):
        _store(tmp_path, wrong_lane_inventory).require(
            replace(address, layer="another-lane"),
            progress_index=0,
            source_state=_state(),
        )


@pytest.mark.parametrize(
    "start_method",
    sorted(
        {"fork", "spawn"}.intersection(
            multiprocessing.get_all_start_methods()
        )
    ),
)
def test_prepared_indexed_receipt_survives_worker_process_transfer(
    tmp_path,
    start_method,
) -> None:
    compatibility = _compatibility()
    spec, address = _write_indexed_shard(
        tmp_path,
        packed_sha256="1" * 64,
        compatibility=compatibility,
        name="only.fibers.sqlite",
    )
    _, inventory = _inventory(
        (spec,),
        compatibility,
        storage_backend=SUCCESSOR_FIBER_INDEXED_STORAGE_BACKEND,
    )
    store = _store(tmp_path, inventory)
    assert store.worker_open_receipt_digests == ()
    assert store.prepare_indexed_worker_open_receipts() == ("1" * 64,)
    assert store.worker_open_receipts_prepared

    context = multiprocessing.get_context(start_method)
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker_lookup_with_full_scans_forbidden,
        args=(store, address, _state(), sender),
    )
    process.start()
    sender.close()
    assert receiver.poll(30), f"{start_method} worker did not return"
    result = receiver.recv()
    receiver.close()
    process.join(timeout=30)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        pytest.fail(f"{start_method} worker did not exit")
    assert process.exitcode == 0
    assert result == (
        "ok",
        persistent_slot_state_sha256(_state()),
        ("1" * 64,),
    )


@pytest.mark.skipif(
    "spawn" not in multiprocessing.get_all_start_methods(),
    reason="spawn multiprocessing is unavailable",
)
def test_prepared_indexed_receipt_works_through_torch_dataloader(
    tmp_path,
) -> None:
    torch = pytest.importorskip("torch")
    compatibility = _compatibility()
    spec, address = _write_indexed_shard(
        tmp_path,
        packed_sha256="1" * 64,
        compatibility=compatibility,
        name="only.fibers.sqlite",
    )
    _, inventory = _inventory(
        (spec,),
        compatibility,
        storage_backend=SUCCESSOR_FIBER_INDEXED_STORAGE_BACKEND,
    )
    store = _store(tmp_path, inventory)
    store.prepare_indexed_worker_open_receipts()
    assert_store_is_pickle_safe(store)
    dataset = _PreparedReceiptLookupDataset(store, address, length=512)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=2,
        multiprocessing_context="spawn",
    )

    observed = list(loader)

    assert observed == [index % 2 for index in range(len(dataset))]


def test_prepared_indexed_receipt_never_falls_back_after_file_drift(
    tmp_path,
    monkeypatch,
) -> None:
    compatibility = _compatibility()
    spec, address = _write_indexed_shard(
        tmp_path,
        packed_sha256="1" * 64,
        compatibility=compatibility,
        name="only.fibers.sqlite",
    )
    _, inventory = _inventory(
        (spec,),
        compatibility,
        storage_backend=SUCCESSOR_FIBER_INDEXED_STORAGE_BACKEND,
    )
    store = _store(tmp_path, inventory)
    store.prepare_indexed_worker_open_receipts()
    restored = pickle.loads(pickle.dumps(store))

    def forbid_fallback(*args, **kwargs):
        raise AssertionError("worker fell back to full indexed validation")

    monkeypatch.setattr(
        indexed_cache_module,
        "_sha256_file",
        forbid_fallback,
    )
    (tmp_path / spec.cache_relative_path).touch()
    with pytest.raises(
        ShardedSuccessorFiberCacheError,
        match="failed verification",
    ):
        restored.require(
            address,
            progress_index=0,
            source_state=_state(),
        )

    lost_receipt = pickle.loads(pickle.dumps(store))
    lost_receipt._loaded.clear()
    lost_receipt._worker_open_receipts.clear()
    with pytest.raises(
        ShardedSuccessorFiberCacheError,
        match="lost a required",
    ):
        lost_receipt.require(
            address,
            progress_index=0,
            source_state=_state(),
        )


def test_inventory_rejects_compatibility_and_schema_tampering(tmp_path) -> None:
    compatibility = _compatibility()
    spec, _ = _write_shard(
        tmp_path,
        packed_sha256="1" * 64,
        compatibility=compatibility,
        name="only.fibers.json",
    )
    encoded, inventory = _inventory((spec,), compatibility)
    root = json.loads(encoded)
    root["compatibility_payload"]["rdkit_version"] = "another-version"
    root["compatibility_sha256"] = _canonical_hash(
        root["compatibility_payload"]
    )
    unhashed = dict(root)
    unhashed.pop("inventory_sha256")
    root["inventory_sha256"] = _canonical_hash(unhashed)
    tampered = json.dumps(root).encode()

    with pytest.raises(
        ShardedSuccessorFiberCacheError,
        match="launch compatibility",
    ):
        deserialize_successor_fiber_cache_inventory(
            tampered,
            expected_inventory_sha256=root["inventory_sha256"],
            expected_compatibility_sha256=(
                inventory.compatibility.compatibility_sha256
            ),
        )

    root["unexpected"] = True
    with pytest.raises(ShardedSuccessorFiberCacheError, match="schema mismatch"):
        deserialize_successor_fiber_cache_inventory(
            json.dumps(root).encode(),
            expected_inventory_sha256=root["inventory_sha256"],
            expected_compatibility_sha256=root["compatibility_sha256"],
        )


def test_whole_trace_omission_and_cache_byte_tampering_fail(tmp_path) -> None:
    compatibility = _compatibility()
    spec, address = _write_shard(
        tmp_path,
        packed_sha256="1" * 64,
        compatibility=compatibility,
        name="only.fibers.json",
    )
    incomplete_spec = replace(
        spec,
        packed_entry_count=2,
        active_trace_count=2,
        progress_record_count=4,
        jump_record_count=2,
        terminal_record_count=2,
        teacher_alias_count=2,
    )
    _, incomplete_inventory = _inventory((incomplete_spec,), compatibility)
    with pytest.raises(
        ShardedSuccessorFiberCacheError,
        match="census mismatch",
    ):
        _store(tmp_path, incomplete_inventory).require(
            address,
            progress_index=0,
            source_state=_state(),
        )

    _, valid_inventory = _inventory((spec,), compatibility)
    path = tmp_path / spec.cache_relative_path
    damaged = bytearray(path.read_bytes())
    damaged[-2] ^= 1
    path.write_bytes(damaged)
    with pytest.raises(
        ShardedSuccessorFiberCacheError,
        match="byte SHA-256",
    ):
        _store(tmp_path, valid_inventory).require(
            address,
            progress_index=0,
            source_state=_state(),
        )


def test_lru_eviction_and_pickle_drop_loaded_object_graphs(tmp_path) -> None:
    compatibility = _compatibility()
    first, first_address = _write_shard(
        tmp_path,
        packed_sha256="1" * 64,
        compatibility=compatibility,
        name="first.fibers.json",
    )
    second, second_address = _write_shard(
        tmp_path,
        packed_sha256="2" * 64,
        compatibility=compatibility,
        name="second.fibers.json",
    )
    _, inventory = _inventory((first, second), compatibility)
    store = _store(tmp_path, inventory, max_open_shards=1)

    store.require(first_address, progress_index=0, source_state=_state())
    assert store.loaded_shard_digests == ("1" * 64,)
    store.require(second_address, progress_index=0, source_state=_state())
    assert store.loaded_shard_digests == ("2" * 64,)
    store.require(first_address, progress_index=0, source_state=_state())
    assert store.loaded_shard_digests == ("1" * 64,)

    restored = pickle.loads(pickle.dumps(store))
    assert restored.loaded_shard_digests == ()
    assert_store_is_pickle_safe(store)


def test_inventory_paths_duplicates_and_full_training_are_rejected(tmp_path) -> None:
    compatibility = _compatibility()
    spec, _ = _write_shard(
        tmp_path,
        packed_sha256="1" * 64,
        compatibility=compatibility,
        name="only.fibers.json",
    )
    with pytest.raises(ValueError, match="safe POSIX-relative"):
        replace(spec, cache_relative_path="../escape.json")
    with pytest.raises(ValueError, match="safe POSIX-relative"):
        replace(spec, cache_relative_path="fibers/*.json")
    with pytest.raises(
        ShardedSuccessorFiberCacheError,
        match="repeats a packed shard",
    ):
        _inventory((spec, spec), compatibility)

    _, inventory = _inventory((spec,), compatibility)
    with pytest.raises(
        ShardedSuccessorFiberCacheError,
        match="bounded-development only",
    ):
        _store(tmp_path, inventory).assert_full_corpus_training_authorized()
