"""Fail-closed launcher bridge for bounded canonical-successor training.

The scientific sampler remains :class:`FactorizedMarkDataset`: this module
only joins the already-selected packed trace/progress row to an immutable,
indexed successor-fiber cache.  It deliberately contains no online chemistry
compilation and no cache-miss fallback.

This bridge is restricted to the exactly-50-update P50 sentinel.  P500 and a
future full-corpus launcher need separate frozen authorization contracts; this
module does not create either authority.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compose_v4.data.active8_trace_inventory import (
    Active8TraceAdmission,
    Active8TraceInventoryError,
    load_active8_trace_admission,
)
from compose_v4.data.charge_policy import CHARGE_POLICY_VERSION
from compose_v4.data.packed_trace_store import (
    PACKED_STORE_SCHEMA,
    PACKED_STORE_SCHEMA_VERSION,
)
from compose_v4.data.provenance_overlay import (
    tensorization_implementation_hash,
)
from compose_v4.data.sharded_successor_fiber_cache import (
    SUCCESSOR_FIBER_INDEXED_STORAGE_BACKEND,
    ShardedSuccessorFiberCache,
    ShardedSuccessorFiberCacheError,
    SuccessorFiberCacheInventory,
    read_successor_fiber_cache_inventory,
)
from compose_v4.data.successor_fiber_cache import (
    fiber_compiler_implementation_hash,
)
from compose_v4.experiments import zero_mixture_instrumentation as _zmi
from compose_v4.experiments.cnof_conditional import PathRecord
from compose_v4.experiments.editing_gate_zero_runtime import (
    PRODUCTION_ACTION_TABLE_VOCABULARY,
)
from compose_v4.experiments.factorized_mark_conditional import (
    FactorizedMarkCollator,
    FactorizedMarkDataset,
)
from compose_v4.experiments.factorized_successor_data import (
    FactorizedSuccessorBatch,
    factorized_successor_loader,
)
from compose_v4.experiments.factorized_successor_objective import (
    CanonicalSuccessorTrainingObjective,
    SuccessorObjectiveMode,
)
from compose_v4.experiments.factorized_training_objective import (
    TrainingLoaderFactory,
)
from compose_v4.experiments.ringcore_semantic_sidecar import (
    LoadedSemanticCellSidecar,
    read_semantic_cell_sidecar,
)
from compose_v4.experiments.training_support_cache import (
    ShardedTrainingSupportCache,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    MARK_RULE_NAMES,
    FactorizedTraceletRateModel,
)
from compose_v4.rewrite.ring_system_fiber import (
    warm_ring_system_candidate_indices,
)
from compose_v4.rewrite.trace_shard import (
    TRACE_SCHEMA,
    TRACE_SCHEMA_VERSION,
)
from compose_v4.rewrite.typed_ring_catalog import ring_catalog_fingerprint

CANONICAL_SUCCESSOR_P50_STEPS = 50
SUCCESSOR_COORDINATE_SCHEMA_VERSION = 1
_OPERATOR_REGISTRY_SOURCES = (
    "src/compose_v4/rewrite/operators.py",
    "src/compose_v4/rewrite/kernel.py",
    "src/compose_v4/rewrite/tracelets.py",
    "src/compose_v4/rewrite/factorized_fiber.py",
)
_CANONICALIZER_CONTRACT_SOURCES = (
    "src/compose_v4/chem/molecular_graph.py",
    "src/compose_v4/rewrite/kernel.py",
)


class EditingSuccessorTrainerError(RuntimeError):
    """A bounded successor-training launch is incomplete or off contract."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: str, *, field: str) -> None:
    if not _is_sha256(value):
        raise ValueError(f"{field} must be a lowercase SHA-256")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _source_set_sha256(relative_paths: tuple[str, ...]) -> str:
    repository = Path(__file__).resolve().parents[3]
    digest = hashlib.sha256()
    for relative in relative_paths:
        path = repository / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes() if path.is_file() else b"<MISSING>")
    return digest.hexdigest()


def _require_file_sha256(path: Path, expected: str, *, name: str) -> None:
    if not path.is_file():
        raise EditingSuccessorTrainerError(f"{name} is absent: {path}")
    observed = _file_sha256(path)
    if observed != expected:
        raise EditingSuccessorTrainerError(
            f"{name} SHA-256 mismatch: expected={expected}, observed={observed}"
        )


@dataclass(frozen=True)
class CanonicalSuccessorRuntimeConfig:
    """Externally frozen identities required to open the bounded cache."""

    cache_root: Path
    inventory_path: Path
    inventory_sha256: str
    compatibility_sha256: str
    source_corpus_inventory_sha256: str
    source_corpus_inventory_path: Path
    source_support_contract_sha256: str
    unified_packed_manifest_path: Path
    representability_overlay_path: Path
    semantic_sidecar_path: Path
    semantic_sidecar_manifest_path: Path
    semantic_sidecar_manifest_sha256: str
    semantic_sidecar_config_sha256: str
    semantic_sidecar_provenance_sha256: str
    max_open_shards: int = 2

    def __post_init__(self) -> None:
        for field in (
            "inventory_sha256",
            "compatibility_sha256",
            "source_corpus_inventory_sha256",
            "source_support_contract_sha256",
            "semantic_sidecar_manifest_sha256",
            "semantic_sidecar_config_sha256",
            "semantic_sidecar_provenance_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        if type(self.max_open_shards) is not int or self.max_open_shards <= 0:
            raise ValueError("max_open_shards must be a positive integer")


@dataclass(frozen=True)
class CanonicalSuccessorRuntime:
    """Verified cache and validation labels; never a training authorization."""

    cache: ShardedSuccessorFiberCache
    inventory: SuccessorFiberCacheInventory
    validation_sidecar: LoadedSemanticCellSidecar
    config: CanonicalSuccessorRuntimeConfig
    source_admission: Active8TraceAdmission | None = None

    @property
    def full_training_authorized(self) -> bool:
        return False

    def assert_full_training_authorized(self) -> None:
        raise EditingSuccessorTrainerError(
            "bounded canonical-successor runtime cannot authorize a full run"
        )

    def checkpoint_metadata(self) -> dict[str, object]:
        return {
            "successor_cache_inventory_sha256": (self.inventory.inventory_sha256),
            "successor_cache_compatibility_sha256": (
                self.inventory.compatibility.compatibility_sha256
            ),
            "successor_source_corpus_inventory_sha256": (
                self.inventory.source_corpus_inventory_sha256
            ),
            "successor_source_corpus_inventory_logical_sha256": (
                None
                if self.source_admission is None
                else self.source_admission.inventory_sha256
            ),
            "successor_effective_source_corpus_cache_sha256": (
                None
                if self.source_admission is None
                else self.source_admission.effective_source_corpus_cache_sha256
            ),
            "successor_source_support_contract_sha256": (
                self.config.source_support_contract_sha256
            ),
            "successor_unified_packed_manifest_sha256": (
                self.inventory.unified_packed_manifest_sha256
            ),
            "successor_representability_overlay_sha256": (
                self.inventory.representability_overlay_sha256
            ),
            "successor_semantic_sidecar_manifest_sha256": (
                self.config.semantic_sidecar_manifest_sha256
            ),
            "successor_cache_storage_backend": (self.inventory.storage_backend),
            "successor_training_scope": "bounded_development_p50_only",
            "successor_full_training_authorized": False,
        }


def _as_pair_mapping(value: object, *, field: str) -> dict[str, bool]:
    if not isinstance(value, list):
        raise EditingSuccessorTrainerError(f"{field} must be a JSON list")
    result: dict[str, bool] = {}
    for item in value:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], str)
            or type(item[1]) is not bool
            or item[0] in result
        ):
            raise EditingSuccessorTrainerError(f"{field} must contain unique [name, Boolean] pairs")
        result[item[0]] = item[1]
    return result


def _validate_live_compatibility(
    inventory: SuccessorFiberCacheInventory,
    model: FactorizedTraceletRateModel,
    *,
    max_atoms: int,
) -> None:
    """Bind the cache to the model and current support/tensorization code."""

    from rdkit import rdBase

    payload = inventory.compatibility.payload
    signature = payload["support_signature"]
    if not isinstance(signature, Mapping):
        raise EditingSuccessorTrainerError("successor-cache support signature is not an object")
    expected_operator_registry_source_sha256 = _source_set_sha256(
        _OPERATOR_REGISTRY_SOURCES
    )
    expected_operator_registry_hash = expected_operator_registry_source_sha256[:16]
    expected_canonicalizer = _source_set_sha256(_CANONICALIZER_CONTRACT_SOURCES)
    expected_vocabulary = [
        [str(element), int(valence)] for element, valence in model.atom_vocabulary.classes
    ]
    expected_flags = {
        "enable_ring_restates": bool(model.enable_ring_restates),
        "enable_cyclic_graft": bool(model.enable_cyclic_graft),
        "enable_heteroatom_scan": bool(model.enable_heteroatom_scan),
        "enable_ring_opening": bool(model.enable_ring_opening),
        "enable_cycle_ops": bool(model.enable_cycle_ops),
        "enable_ring_grow_macro": bool(model.enable_ring_grow_macro),
        "enable_ring_system_delete": bool(model.enable_ring_system_delete),
    }
    expected_signature_fields: dict[str, object] = {
        "operator_registry_hash": expected_operator_registry_hash,
        "operator_registry_source_sha256": expected_operator_registry_source_sha256,
        "operator_capability_fingerprint": (model.operator_capabilities.fingerprint()),
        "ordered_family_vocabulary": list(MARK_RULE_NAMES),
        "ordered_action_table_vocabulary": list(PRODUCTION_ACTION_TABLE_VOCABULARY),
        "atom_vocabulary": expected_vocabulary,
        "max_atoms": int(max_atoms),
        "atom_insert_arity_support": [0, 1],
        "charge_policy": CHARGE_POLICY_VERSION,
        "catalog_fingerprint": ring_catalog_fingerprint(model.ring_catalog),
        **expected_flags,
        "embedded_jump_chain_policy": "productive_canonical_successors",
    }
    missing_signature = sorted(set(expected_signature_fields) - set(signature))
    disagreements = {
        field: {
            "cache": signature.get(field),
            "live": expected,
        }
        for field, expected in expected_signature_fields.items()
        if field in signature and signature[field] != expected
    }
    if missing_signature or disagreements:
        raise EditingSuccessorTrainerError(
            "successor-cache support signature does not match the live model: "
            f"missing={missing_signature}, disagreements={disagreements}"
        )

    expected_payload_fields = {
        "ordered_family_vocabulary": list(MARK_RULE_NAMES),
        "ordered_action_table_vocabulary": list(PRODUCTION_ACTION_TABLE_VOCABULARY),
        "coordinate_schema_version": SUCCESSOR_COORDINATE_SCHEMA_VERSION,
        "operator_registry_hash": expected_operator_registry_hash,
        "capability_hash": model.operator_capabilities.fingerprint(),
        "canonicalizer_contract_sha256": expected_canonicalizer,
        "rdkit_version": rdBase.rdkitVersion,
        "packed_corpus_schema": PACKED_STORE_SCHEMA,
        "packed_corpus_schema_version": PACKED_STORE_SCHEMA_VERSION,
        "trace_codec_schema": TRACE_SCHEMA,
        "trace_codec_schema_version": TRACE_SCHEMA_VERSION,
        "tensorization_implementation_hash": (tensorization_implementation_hash()),
        "fiber_compiler_implementation_hash": (fiber_compiler_implementation_hash()),
        "charge_policy_version": CHARGE_POLICY_VERSION,
    }
    payload_disagreements = {
        field: {
            "cache": payload.get(field),
            "live": expected,
        }
        for field, expected in expected_payload_fields.items()
        if payload.get(field) != expected
    }
    if payload_disagreements:
        raise EditingSuccessorTrainerError(
            "successor-cache compatibility differs from live production code: "
            f"{payload_disagreements}"
        )

    raw_flags = signature.get("capability_flags")
    if raw_flags is not None:
        recorded_flags = _as_pair_mapping(
            raw_flags,
            field="support_signature.capability_flags",
        )
        known_recorded = {name: recorded_flags.get(name) for name in expected_flags}
        if known_recorded != expected_flags:
            raise EditingSuccessorTrainerError(
                "successor-cache capability flags differ from the live model"
            )


def open_canonical_successor_runtime(
    config: CanonicalSuccessorRuntimeConfig,
    model: FactorizedTraceletRateModel,
    *,
    max_atoms: int,
    source_admission: Active8TraceAdmission | None = None,
) -> CanonicalSuccessorRuntime:
    """Open and fully parent-validate a bounded indexed cache before training."""

    try:
        inventory = read_successor_fiber_cache_inventory(
            config.inventory_path,
            expected_inventory_sha256=config.inventory_sha256,
            expected_compatibility_sha256=config.compatibility_sha256,
        )
    except ShardedSuccessorFiberCacheError as error:
        raise EditingSuccessorTrainerError(
            "successor-cache inventory failed exact verification"
        ) from error
    if inventory.source_corpus_inventory_sha256 != config.source_corpus_inventory_sha256:
        raise EditingSuccessorTrainerError(
            "successor cache names another frozen source-corpus inventory"
        )
    _require_file_sha256(
        config.source_corpus_inventory_path,
        inventory.source_corpus_inventory_sha256,
        name="frozen source-corpus inventory",
    )
    if source_admission is None:
        try:
            source_admission = load_active8_trace_admission(
                config.source_corpus_inventory_path,
                expected_manifest_file_sha256=(
                    inventory.source_corpus_inventory_sha256
                ),
                expected_support_contract_sha256=(
                    config.source_support_contract_sha256
                ),
            )
        except Active8TraceInventoryError as error:
            raise EditingSuccessorTrainerError(
                "frozen source-corpus inventory is not a valid Active8 "
                "whole-trace admission artifact"
            ) from error
    elif (
        source_admission.manifest_path.resolve()
        != config.source_corpus_inventory_path.resolve()
        or source_admission.manifest_file_sha256
        != inventory.source_corpus_inventory_sha256
    ):
        raise EditingSuccessorTrainerError(
            "preloaded Active8 admission disagrees with the frozen "
            "source-corpus inventory"
        )
    if (
        source_admission.support_contract_sha256
        != config.source_support_contract_sha256
    ):
        raise EditingSuccessorTrainerError(
            "Active8 source inventory names another Gate-0 support contract"
        )
    if inventory.storage_backend != SUCCESSOR_FIBER_INDEXED_STORAGE_BACKEND:
        raise EditingSuccessorTrainerError(
            "bounded training requires the qualified indexed successor cache"
        )
    _require_file_sha256(
        config.unified_packed_manifest_path,
        inventory.unified_packed_manifest_sha256,
        name="unified packed manifest",
    )
    if (
        source_admission.unified_packed_manifest_sha256
        != inventory.unified_packed_manifest_sha256
    ):
        raise EditingSuccessorTrainerError(
            "Active8 source inventory and successor cache name different "
            "unified packed manifests"
        )
    _require_file_sha256(
        config.representability_overlay_path,
        inventory.representability_overlay_sha256,
        name="representability overlay",
    )
    _validate_live_compatibility(inventory, model, max_atoms=max_atoms)

    sidecar = read_semantic_cell_sidecar(
        config.semantic_sidecar_path,
        config.semantic_sidecar_manifest_path,
        expected_manifest_sha256=(config.semantic_sidecar_manifest_sha256),
        expected_config_sha256=config.semantic_sidecar_config_sha256,
        expected_provenance_sha256=(config.semantic_sidecar_provenance_sha256),
    )
    provenance = sidecar.manifest["provenance"]
    if (
        provenance["unified_packed_manifest_sha256"] != inventory.unified_packed_manifest_sha256
        or provenance["representability_overlay_sha256"]
        != inventory.representability_overlay_sha256
    ):
        raise EditingSuccessorTrainerError(
            "validation semantic sidecar and successor cache name different source corpora"
        )

    cache = ShardedSuccessorFiberCache(
        config.cache_root,
        inventory,
        expected_inventory_sha256=config.inventory_sha256,
        expected_compatibility_sha256=config.compatibility_sha256,
        max_open_shards=config.max_open_shards,
    )
    # This scans and validates every indexed shard in the trusted parent and
    # freezes same-launch receipts before any DataLoader worker or optimizer.
    cache.prepare_indexed_worker_open_receipts()
    return CanonicalSuccessorRuntime(
        cache=cache,
        inventory=inventory,
        validation_sidecar=sidecar,
        config=config,
        source_admission=source_admission,
    )


def assert_records_covered_by_successor_cache(
    records: tuple[PathRecord, ...],
    inventory: SuccessorFiberCacheInventory,
    *,
    partition: str,
) -> None:
    """Prove all source records resolve to declared complete cache shards."""

    if not records:
        raise EditingSuccessorTrainerError(f"{partition} successor records are empty")
    specs = inventory.by_packed_digest
    observed: set[tuple[str, int]] = set()
    for record_index, record in enumerate(records):
        address = record.corpus_address
        if address is None:
            raise EditingSuccessorTrainerError(
                f"{partition} record {record_index} has no immutable packed address"
            )
        try:
            spec = specs[address.packed_shard_content_sha256]
        except KeyError:
            raise EditingSuccessorTrainerError(
                f"{partition} record {record_index} uses a shard absent from "
                "the successor-cache inventory"
            ) from None
        if (
            address.packed_shard_name != spec.packed_shard_name
            or address.layer != spec.layer
            or address.partition != spec.partition
            or address.partition != partition
            or address.path_length != record.path.path_length
            or not 0 <= address.entry_index < spec.packed_entry_count
            or address.entry_index in set(spec.excluded_entry_indices)
        ):
            raise EditingSuccessorTrainerError(
                f"{partition} record {record_index} disagrees with its exact "
                "successor-cache shard envelope"
            )
        key = (address.packed_shard_content_sha256, address.entry_index)
        if key in observed:
            raise EditingSuccessorTrainerError(
                f"{partition} corpus repeats exact packed trace {key}"
            )
        observed.add(key)


def _mark_collator(
    model: FactorizedTraceletRateModel,
    *,
    use_aromatic_bond_view: bool,
) -> FactorizedMarkCollator:
    return FactorizedMarkCollator.from_capabilities(
        model.operator_capabilities,
        use_aromatic_bond_view=use_aromatic_bond_view,
        ring_catalog=model.ring_catalog,
    )


def build_canonical_successor_validation_batch(
    runtime: CanonicalSuccessorRuntime,
    model: FactorizedTraceletRateModel,
    records: tuple[PathRecord, ...],
    *,
    batch_size: int,
    seed: int,
    workers: int,
    late_time_fraction: float,
    operational_horizon: float,
    progress_stratification_fraction: float,
    use_aromatic_bond_view: bool,
    ring_electronic_mode: str,
    target_property_conditions: Mapping[str, tuple[float, ...]] | None,
) -> FactorizedSuccessorBatch:
    """Materialize one deterministic validation batch with exact fibers."""

    if batch_size <= 0:
        raise ValueError("validation batch size must be positive")
    warm_ring_system_candidate_indices(model.ring_catalog)
    assert_records_covered_by_successor_cache(
        records,
        runtime.inventory,
        partition="validation",
    )
    dataset = FactorizedMarkDataset(
        records,
        start_index=0,
        length=batch_size,
        seed=seed,
        late_time_fraction=late_time_fraction,
        operational_horizon=operational_horizon,
        progress_stratification_fraction=progress_stratification_fraction,
        ring_catalog=model.ring_catalog,
        ring_electronic_mode=ring_electronic_mode,
        target_property_conditions=target_property_conditions,
        condition_dropout_probability=0.0,
        ring_family_mass_mode=model.ring_family_mass_mode,
    )
    loader = factorized_successor_loader(
        dataset,
        runtime.cache,
        _mark_collator(
            model,
            use_aromatic_bond_view=use_aromatic_bond_view,
        ),
        batch_size=batch_size,
        workers=workers,
        pin_memory=False,
        seed=seed,
        semantic_cell_ids=runtime.validation_sidecar.semantic_cell_ids,
        require_semantic_cell_ids=True,
    )
    try:
        batch = next(iter(loader))
    except StopIteration:
        raise EditingSuccessorTrainerError(
            "successor validation loader emitted no complete batch"
        ) from None
    if batch.batch_size != batch_size:
        raise EditingSuccessorTrainerError("successor validation batch cardinality drifted")
    return batch


def build_canonical_successor_loader_factory(
    runtime: CanonicalSuccessorRuntime,
    model: FactorizedTraceletRateModel,
    records: tuple[PathRecord, ...],
    *,
    steps: int,
    batch_size: int,
    seed: int,
    workers: int,
    data_prefetch_factor: int,
    late_time_fraction: float,
    operational_horizon: float,
    progress_stratification_fraction: float,
    use_aromatic_bond_view: bool,
    ring_electronic_mode: str,
    target_property_conditions: Mapping[str, tuple[float, ...]] | None,
    condition_dropout_probability: float,
    record_index_sampler: object | None,
    training_support_cache: ShardedTrainingSupportCache | None = None,
    require_cached_support: bool = False,
) -> TrainingLoaderFactory:
    """Return the shared trainer's exact post-sampling cache-join factory."""

    if steps != CANONICAL_SUCCESSOR_P50_STEPS:
        raise EditingSuccessorTrainerError(
            "bounded canonical-successor training currently supports exactly "
            f"P50 ({CANONICAL_SUCCESSOR_P50_STEPS} steps); P500 is blocked"
        )
    assert_records_covered_by_successor_cache(
        records,
        runtime.inventory,
        partition="train",
    )
    warm_ring_system_candidate_indices(model.ring_catalog)

    def loader_factory(start_step: int) -> Any:
        if not 0 <= start_step <= steps:
            raise EditingSuccessorTrainerError(
                "successor loader resume step lies outside the bounded horizon"
            )
        _zmi.bump("edit_dataset_constructions")
        _zmi.bump("edit_dataloader_constructions")
        mark_dataset = FactorizedMarkDataset(
            records,
            start_index=start_step * batch_size,
            length=(steps - start_step) * batch_size,
            seed=seed,
            late_time_fraction=late_time_fraction,
            operational_horizon=operational_horizon,
            progress_stratification_fraction=(progress_stratification_fraction),
            ring_catalog=model.ring_catalog,
            ring_electronic_mode=ring_electronic_mode,
            training_support_cache=training_support_cache,
            require_cached_support=require_cached_support,
            target_property_conditions=target_property_conditions,
            condition_dropout_probability=condition_dropout_probability,
            ring_family_mass_mode=model.ring_family_mass_mode,
            record_index_sampler=record_index_sampler,
        )
        return factorized_successor_loader(
            mark_dataset,
            runtime.cache,
            _mark_collator(
                model,
                use_aromatic_bond_view=use_aromatic_bond_view,
            ),
            batch_size=batch_size,
            workers=workers,
            pin_memory=model.device.type == "cuda",
            seed=seed,
            prefetch_factor=data_prefetch_factor,
            semantic_cell_ids=None,
            require_semantic_cell_ids=False,
        )

    return loader_factory


def canonical_successor_objective(
    *,
    mode: SuccessorObjectiveMode,
    hazard_weight: float,
) -> CanonicalSuccessorTrainingObjective:
    """Construct the explicit identity/hazard objective used by P50."""

    return CanonicalSuccessorTrainingObjective(
        mode=mode,
        hazard_weight=hazard_weight,
        require_semantic_cells_for_metrics=True,
    )


__all__ = [
    "CANONICAL_SUCCESSOR_P50_STEPS",
    "PRODUCTION_ACTION_TABLE_VOCABULARY",
    "SUCCESSOR_COORDINATE_SCHEMA_VERSION",
    "CanonicalSuccessorRuntime",
    "CanonicalSuccessorRuntimeConfig",
    "EditingSuccessorTrainerError",
    "assert_records_covered_by_successor_cache",
    "build_canonical_successor_loader_factory",
    "build_canonical_successor_validation_batch",
    "canonical_successor_objective",
    "open_canonical_successor_runtime",
]
