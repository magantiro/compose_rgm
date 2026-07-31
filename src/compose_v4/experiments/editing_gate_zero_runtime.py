"""Fail-closed runtime for the real bounded editing Gate-0 probe.

This module joins four already-defined scientific objects without weakening
any of them:

* the frozen, exact-address validation semantic sidecar;
* the immutable packed validation shards named by that sidecar;
* the production RingCore catalog and scratch broad-organic architecture;
* the bounded successor probe and the step-zero structural support audit.

The emitted evidence is deliberately incapable of making a quality decision.
It contains no empirical threshold, no ``GO`` value, and always records
``training_authorized=false``.  Its only conclusion is that the exact
initialization and support/label checks completed without an exception.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from compose_v4.chem.molecular_graph import (
    ORGANIC_VOCABULARY,
    smiles_to_molecular_graph,
)
from compose_v4.chem.persistent_state_identity import (
    PERSISTENT_STATE_DIGEST_SCHEMA,
    PERSISTENT_STATE_DIGEST_VERSION,
)
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.data.charge_policy import CHARGE_POLICY_VERSION
from compose_v4.data.active8_trace_inventory import (
    Active8TraceAdmission,
    Active8TraceInventoryError,
)
from compose_v4.data.packed_trace_store import (
    PACKED_STORE_SCHEMA,
    PACKED_STORE_SCHEMA_VERSION,
    manifest_path_for,
    read_addressed_packed_shard,
)
from compose_v4.data.provenance_overlay import (
    overlay_path_for,
    tensorization_implementation_hash,
)
from compose_v4.data.successor_fiber_cache import (
    SuccessorFiberCacheAddress,
    SuccessorFiberCacheProvenance,
    fiber_compiler_implementation_hash,
)
from compose_v4.experiments.cnof_conditional import PathRecord
from compose_v4.experiments.editing_p50_gate import (
    state_dict_semantic_sha256,
)
from compose_v4.experiments.editing_step_zero_gate import (
    InitializationParityReport,
    StepZeroSuccessorSupportReport,
    assert_initialization_parity,
    audit_successor_support_before_optimization,
    build_scratch_initialization_plan,
)
from compose_v4.experiments.editing_step_zero_probe import (
    GATE_ZERO_SUCCESSOR_SUPPORT_TIME,
    GateZeroSuccessorProbe,
    GateZeroSuccessorProbeConfig,
    build_gate_zero_successor_probe,
    select_gate_zero_successor_probe_records,
)
from compose_v4.experiments.factorized_mark_conditional import (
    FactorizedMarkCollator,
    FactorizedMarkExample,
)
from compose_v4.experiments.factorized_successor_data import (
    FactorizedSuccessorBatch,
    FactorizedSuccessorCollator,
    FactorizedSuccessorExample,
)
from compose_v4.experiments.ringcore_semantic_sidecar import (
    LoadedSemanticCellSidecar,
    SemanticSidecarError,
    SemanticSidecarSourceShard,
    read_semantic_cell_sidecar,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    MARK_RULE_NAMES,
    FactorizedTraceletRateModel,
)
from compose_v4.rewrite.action_codec import ActionCodecError, canonical_family
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target
from compose_v4.rewrite.typed_ring_catalog import (
    TypedRingCatalog,
    build_typed_ring_catalog,
    ring_catalog_fingerprint,
)

EDITING_GATE_ZERO_RUNTIME_CONTRACT_SCHEMA = "compose.editing.gate_zero_runtime_contract"
EDITING_GATE_ZERO_RUNTIME_CONTRACT_VERSION = 2
EDITING_GATE_ZERO_RUNTIME_EVIDENCE_SCHEMA = "compose.editing.gate_zero_structural_evidence"
EDITING_GATE_ZERO_RUNTIME_EVIDENCE_VERSION = 2
EDITING_GATE_ZERO_RUNTIME_STATUS = "STRUCTURAL_AUDIT_COMPLETE_NO_TRAINING_DECISION"
EDITING_GATE_ZERO_RUNTIME_CONTRACT_STATUS = "FROZEN_STRUCTURAL_PROBE_CONTRACT_NO_TRAINING_AUTHORITY"
PRODUCTION_RINGCORE_CATALOG_FINGERPRINT = "639ff6078c32d43c"
PRODUCTION_RINGCORE_CATALOG_SEEDS = (
    "c1ccccc1",
    "c1ccncc1",
    "C1CCNCC1",
    "C1CCOCC1",
    "c1ccc2ccccc2c1",
)
SUCCESSOR_COORDINATE_SCHEMA_VERSION = 1
PRODUCTION_ACTION_TABLE_VOCABULARY = (
    "grow_root",
    "grow_connected",
    "atom_delete",
    "atom_restate",
    "bond_reorder",
    "bond_reroute",
    "cycle_insert",
    "cycle_attach",
    "ring_system_restate",
)
PRODUCTION_ACTIVE_FAMILIES = tuple(
    family
    for family in MARK_RULE_NAMES
    if family not in {"ring_system_grow", "ring_system_delete"}
)
GATE_ZERO_ACTIVE8_IDENTITY_FIELDS = (
    "active8_inventory_manifest_file_sha256",
    "active8_inventory_sha256",
    "active8_effective_source_corpus_cache_sha256",
    "active8_unified_packed_manifest_sha256",
    "active8_support_contract_sha256",
)
GATE_ZERO_ACTIVE8_CENSUS_FIELDS = (
    "partition",
    "source_shard_count",
    "source_trace_count",
    "source_progress_row_count",
    "admitted_trace_count",
    "excluded_trace_count",
    "admitted_progress_row_count",
    "admitted_nonterminal_row_count",
    "admitted_terminal_row_count",
    "admitted_nonempty_semantic_cell_count",
    "admitted_teacher_examples_by_family",
)

_CONTRACT_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "training_authorized",
    "sidecar",
    "model",
    "required_families",
    "audit_batch_size",
    "contract_sha256",
}
_SIDECAR_CONTRACT_FIELDS = {
    "manifest_sha256",
    "config_sha256",
    "provenance_sha256",
    "semantic_census_sha256",
    "nonempty_semantic_cell_count",
    "unified_packed_manifest_sha256",
    "representability_overlay_sha256",
}
_MODEL_CONTRACT_FIELDS = {
    "seed",
    "max_atoms",
    "hidden_dim",
    "message_passing_steps",
    "mark_dim",
    "atom_vocabulary_class_count",
    "catalog_fingerprint",
    "operator_capability_fingerprint",
    "enable_ring_restates",
    "enable_cyclic_graft",
    "enable_heteroatom_scan",
    "enable_ring_opening",
    "enable_cycle_ops",
    "enable_ring_grow_macro",
    "enable_ring_system_delete",
}
_PRODUCTION_MODEL_STRUCTURE = {
    "max_atoms": 40,
    "hidden_dim": 256,
    "message_passing_steps": 6,
    "mark_dim": 32,
    "atom_vocabulary_class_count": 15,
    "operator_capability_fingerprint": "e787c852410c6b63",
    "enable_ring_restates": True,
    "enable_cyclic_graft": True,
    "enable_heteroatom_scan": True,
    "enable_ring_opening": True,
    "enable_cycle_ops": True,
    "enable_ring_grow_macro": False,
    "enable_ring_system_delete": False,
}
_LAYER_DIRECTORIES = {
    "corruption": ("edit_packed_v1", "corruption", "validation"),
    "cycle_ops": ("edit_packed_v1", "cycle_ops", "validation"),
    "mmp_analogue": ("mmp_packed_v1", "validation"),
}
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


class EditingGateZeroRuntimeError(RuntimeError):
    """The bounded structural probe is absent, off-contract, or tampered."""


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise EditingGateZeroRuntimeError(
            "Gate-0 runtime metadata is not finite canonical JSON"
        ) from error


def _stable_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _source_set_sha256(relative_paths: Iterable[str]) -> str:
    repo = Path(__file__).resolve().parents[3]
    digest = hashlib.sha256()
    for relative in sorted(relative_paths):
        source = repo / relative
        if not source.is_file():
            raise EditingGateZeroRuntimeError(f"Gate-0 provenance source is absent: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(source.read_bytes())
    return digest.hexdigest()


def _require_exact_fields(
    value: object,
    expected: set[str],
    *,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EditingGateZeroRuntimeError(f"{name} must be an object")
    if set(value) != expected:
        raise EditingGateZeroRuntimeError(
            f"{name} fields disagree; "
            f"missing={sorted(expected - set(value))!r}, "
            f"unexpected={sorted(set(value) - expected)!r}"
        )
    return value


@dataclass(frozen=True)
class GateZeroRuntimeContract:
    """Validated, self-hashed structural runtime contract."""

    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        frozen = dict(self.payload)
        _require_exact_fields(frozen, _CONTRACT_FIELDS, name="runtime contract")
        body = {key: value for key, value in frozen.items() if key != "contract_sha256"}
        if (
            frozen["schema"] != EDITING_GATE_ZERO_RUNTIME_CONTRACT_SCHEMA
            or frozen["schema_version"] != EDITING_GATE_ZERO_RUNTIME_CONTRACT_VERSION
            or frozen["status"] != EDITING_GATE_ZERO_RUNTIME_CONTRACT_STATUS
            or frozen["training_authorized"] is not False
            or frozen["contract_sha256"] != _stable_sha256(body)
        ):
            raise EditingGateZeroRuntimeError("runtime contract identity or self-hash is invalid")
        sidecar = _require_exact_fields(
            frozen["sidecar"],
            _SIDECAR_CONTRACT_FIELDS,
            name="sidecar contract",
        )
        for field in _SIDECAR_CONTRACT_FIELDS - {"nonempty_semantic_cell_count"}:
            if not _is_sha256(sidecar[field]):
                raise EditingGateZeroRuntimeError(
                    f"sidecar contract {field} must be a lowercase SHA-256"
                )
        if (
            type(sidecar["nonempty_semantic_cell_count"]) is not int
            or sidecar["nonempty_semantic_cell_count"] <= 0
        ):
            raise EditingGateZeroRuntimeError("nonempty semantic-cell count must be positive")
        model = _require_exact_fields(
            frozen["model"],
            _MODEL_CONTRACT_FIELDS,
            name="model contract",
        )
        for field in (
            "seed",
            "max_atoms",
            "hidden_dim",
            "message_passing_steps",
            "mark_dim",
            "atom_vocabulary_class_count",
        ):
            if type(model[field]) is not int or model[field] <= 0:
                raise EditingGateZeroRuntimeError(
                    f"model contract {field} must be a positive integer"
                )
        for field in (
            "enable_ring_restates",
            "enable_cyclic_graft",
            "enable_heteroatom_scan",
            "enable_ring_opening",
            "enable_cycle_ops",
            "enable_ring_grow_macro",
            "enable_ring_system_delete",
        ):
            if type(model[field]) is not bool:
                raise EditingGateZeroRuntimeError(f"model contract {field} must be Boolean")
        if model["catalog_fingerprint"] != PRODUCTION_RINGCORE_CATALOG_FINGERPRINT:
            raise EditingGateZeroRuntimeError("runtime contract names another RingCore catalog")
        structural_drift = {
            field: {
                "observed": model[field],
                "required": expected,
            }
            for field, expected in _PRODUCTION_MODEL_STRUCTURE.items()
            if model[field] != expected
        }
        if structural_drift:
            raise EditingGateZeroRuntimeError(
                "runtime contract does not name the production 256x6 scratch "
                f"RingCore architecture: {structural_drift}"
            )
        required = frozen["required_families"]
        if not isinstance(required, list) or tuple(required) != PRODUCTION_ACTIVE_FAMILIES:
            raise EditingGateZeroRuntimeError(
                "runtime contract must require the ordered eight active pilot families"
            )
        if type(frozen["audit_batch_size"]) is not int or frozen["audit_batch_size"] <= 0:
            raise EditingGateZeroRuntimeError("audit_batch_size must be a positive resource bound")
        object.__setattr__(self, "payload", MappingProxyType(frozen))

    @property
    def sha256(self) -> str:
        return str(self.payload["contract_sha256"])

    @property
    def sidecar(self) -> Mapping[str, Any]:
        return MappingProxyType(dict(self.payload["sidecar"]))

    @property
    def model(self) -> Mapping[str, Any]:
        return MappingProxyType(dict(self.payload["model"]))

    @property
    def required_families(self) -> tuple[str, ...]:
        return tuple(self.payload["required_families"])

    @property
    def audit_batch_size(self) -> int:
        return int(self.payload["audit_batch_size"])

    @property
    def training_authorized(self) -> bool:
        return False

    def assert_training_authorized(self) -> None:
        raise EditingGateZeroRuntimeError("Gate-0 structural evidence never authorizes training")


def load_gate_zero_runtime_contract(path: Path) -> GateZeroRuntimeContract:
    """Load one exact self-hashed runtime contract."""

    source = Path(path)
    if not source.is_file():
        raise EditingGateZeroRuntimeError(f"Gate-0 runtime contract is absent: {source}")
    try:
        payload = json.loads(source.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EditingGateZeroRuntimeError("Gate-0 runtime contract is invalid JSON") from error
    return GateZeroRuntimeContract(payload)


@dataclass(frozen=True)
class ValidationShardBinding:
    """One sidecar-declared packed shard resolved under the transfer root."""

    spec: SemanticSidecarSourceShard
    packed_path: Path
    manifest_path: Path
    overlay_path: Path


@dataclass(frozen=True)
class FrozenValidationSource:
    """Fully verified validation records and their exact semantic sidecar."""

    records: tuple[PathRecord, ...]
    sidecar: LoadedSemanticCellSidecar
    bindings: tuple[ValidationShardBinding, ...]
    unified_packed_manifest_sha256: str
    representability_overlay_sha256: str

    def __post_init__(self) -> None:
        if not self.records or not self.bindings:
            raise ValueError("frozen validation source cannot be empty")
        if not _is_sha256(self.unified_packed_manifest_sha256) or not _is_sha256(
            self.representability_overlay_sha256
        ):
            raise ValueError("validation source identities must be SHA-256 values")

    @property
    def binding_by_digest(self) -> Mapping[str, ValidationShardBinding]:
        return MappingProxyType(
            {binding.spec.packed_shard_content_sha256: binding for binding in self.bindings}
        )


@dataclass(frozen=True)
class GateZeroActive8ValidationSource:
    """Exact whole-trace Active8 view of one frozen validation source."""

    source: FrozenValidationSource
    admission: Active8TraceAdmission
    records: tuple[PathRecord, ...]
    semantic_cell_ids: Mapping[tuple[str, int, int], str | None]
    admitted_teacher_examples_by_family: Mapping[str, int]

    def __post_init__(self) -> None:
        if not isinstance(self.source, FrozenValidationSource):
            raise TypeError("Active8 validation view requires a FrozenValidationSource")
        if not isinstance(self.admission, Active8TraceAdmission):
            raise TypeError("Active8 validation view requires an Active8TraceAdmission")
        if not self.records:
            raise ValueError("Active8 validation view cannot be empty")
        semantic_cell_ids = dict(self.semantic_cell_ids)
        record_keys = {
            (
                record.corpus_address.packed_shard_content_sha256,
                record.corpus_address.entry_index,
                progress_index,
            )
            for record in self.records
            if record.corpus_address is not None
            for progress_index in range(record.path.path_length + 1)
        }
        if set(semantic_cell_ids) != record_keys:
            raise ValueError(
                "Active8 validation records and semantic rows are not an exact census"
            )
        family_counts = dict(self.admitted_teacher_examples_by_family)
        if (
            tuple(family_counts) != PRODUCTION_ACTIVE_FAMILIES
            or any(type(count) is not int or count < 0 for count in family_counts.values())
        ):
            raise ValueError(
                "Active8 validation family census must cover the ordered pilot families"
            )
        object.__setattr__(
            self,
            "semantic_cell_ids",
            MappingProxyType(semantic_cell_ids),
        )
        object.__setattr__(
            self,
            "admitted_teacher_examples_by_family",
            MappingProxyType(family_counts),
        )

    @property
    def sidecar(self) -> LoadedSemanticCellSidecar:
        return self.source.sidecar

    @property
    def bindings(self) -> tuple[ValidationShardBinding, ...]:
        return self.source.bindings

    @property
    def binding_by_digest(self) -> Mapping[str, ValidationShardBinding]:
        return self.source.binding_by_digest

    @property
    def unified_packed_manifest_sha256(self) -> str:
        return self.source.unified_packed_manifest_sha256

    @property
    def representability_overlay_sha256(self) -> str:
        return self.source.representability_overlay_sha256

    @property
    def admitted_nonterminal_row_count(self) -> int:
        return sum(record.path.path_length for record in self.records)

    @property
    def admitted_terminal_row_count(self) -> int:
        return len(self.records)

    def active8_evidence(self) -> dict[str, Any]:
        """Return the exact identity and admitted validation census."""

        nonempty_cells = {
            cell_id for cell_id in self.semantic_cell_ids.values() if cell_id is not None
        }
        return {
            "active8_inventory_manifest_file_sha256": (
                self.admission.manifest_file_sha256
            ),
            "active8_inventory_sha256": self.admission.inventory_sha256,
            "active8_effective_source_corpus_cache_sha256": (
                self.admission.effective_source_corpus_cache_sha256
            ),
            "active8_unified_packed_manifest_sha256": (
                self.admission.unified_packed_manifest_sha256
            ),
            "active8_support_contract_sha256": (
                self.admission.support_contract_sha256
            ),
            "partition": "validation",
            "source_shard_count": len(self.source.bindings),
            "source_trace_count": len(self.source.records),
            "source_progress_row_count": len(self.source.sidecar.rows),
            "admitted_trace_count": len(self.records),
            "excluded_trace_count": len(self.source.records) - len(self.records),
            "admitted_progress_row_count": len(self.semantic_cell_ids),
            "admitted_nonterminal_row_count": self.admitted_nonterminal_row_count,
            "admitted_terminal_row_count": self.admitted_terminal_row_count,
            "admitted_nonempty_semantic_cell_count": len(nonempty_cells),
            "admitted_teacher_examples_by_family": dict(
                self.admitted_teacher_examples_by_family
            ),
        }


def _semantic_source_shard(value: object) -> SemanticSidecarSourceShard:
    if not isinstance(value, Mapping):
        raise EditingGateZeroRuntimeError("semantic-sidecar source shard is malformed")
    try:
        return SemanticSidecarSourceShard(
            packed_shard_content_sha256=str(value["packed_shard_content_sha256"]),
            packed_shard_name=str(value["packed_shard_name"]),
            layer=str(value["layer"]),
            partition=str(value["partition"]),
            packed_manifest_sha256=str(value["packed_manifest_sha256"]),
            provenance_overlay_sha256=str(value["provenance_overlay_sha256"]),
            packed_entry_count=int(value["packed_entry_count"]),
            effective_trace_count=int(value["effective_trace_count"]),
            progress_row_count=int(value["progress_row_count"]),
            excluded_entry_indices=tuple(value["excluded_entry_indices"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise EditingGateZeroRuntimeError(
            "semantic-sidecar source-shard contract is invalid"
        ) from error


def _require_file_sha256(path: Path, expected: str, *, name: str) -> None:
    if not Path(path).is_file():
        raise EditingGateZeroRuntimeError(f"{name} is absent: {path}")
    observed = _sha256_file(path)
    if observed != expected:
        raise EditingGateZeroRuntimeError(f"{name} hash mismatch: {observed} != {expected}")


def _validated_sidecar_external_identities(
    contract: GateZeroRuntimeContract,
    loaded: LoadedSemanticCellSidecar,
) -> tuple[str, str]:
    """Validate the frozen pre-admission sidecar without conflating its support."""

    sidecar_contract = contract.sidecar
    manifest = loaded.manifest
    if (
        manifest["semantic_census_sha256"]
        != sidecar_contract["semantic_census_sha256"]
        or manifest["counts"]["nonempty_cells"]
        != sidecar_contract["nonempty_semantic_cell_count"]
    ):
        raise EditingGateZeroRuntimeError(
            "semantic-sidecar census disagrees with the runtime contract"
        )
    teacher_families = manifest["counts"]["teacher_family_rows"]
    missing_families = sorted(set(contract.required_families) - set(teacher_families))
    if missing_families:
        raise EditingGateZeroRuntimeError(
            "semantic sidecar is missing required Active8 teacher families: "
            f"{missing_families}"
        )
    provenance = manifest["provenance"]
    unified_sha256 = str(provenance["unified_packed_manifest_sha256"])
    representability_sha256 = str(provenance["representability_overlay_sha256"])
    if (
        unified_sha256 != sidecar_contract["unified_packed_manifest_sha256"]
        or representability_sha256
        != sidecar_contract["representability_overlay_sha256"]
    ):
        raise EditingGateZeroRuntimeError(
            "sidecar external corpus identity disagrees with the runtime contract"
        )
    return unified_sha256, representability_sha256


def _validate_active8_source_identity(
    contract: GateZeroRuntimeContract,
    *,
    unified_packed_manifest_sha256: str,
    admission: Active8TraceAdmission,
) -> None:
    """Bind one verified Active8 inventory to the Gate0 support and corpus."""

    if not isinstance(admission, Active8TraceAdmission):
        raise TypeError("Gate0 requires a verified Active8TraceAdmission")
    if admission.support_contract_sha256 != contract.sha256:
        raise EditingGateZeroRuntimeError(
            "Active8 inventory and Gate0 name different support contracts"
        )
    if admission.unified_packed_manifest_sha256 != unified_packed_manifest_sha256:
        raise EditingGateZeroRuntimeError(
            "Active8 inventory and Gate0 name different unified packed corpora"
        )


def load_frozen_validation_source(
    *,
    contract: GateZeroRuntimeContract,
    transfer_root: Path,
    sidecar_path: Path,
    sidecar_manifest_path: Path,
) -> FrozenValidationSource:
    """Verify the frozen sidecar and load every addressed validation trace."""

    root = Path(transfer_root)
    if not root.is_dir():
        raise EditingGateZeroRuntimeError(f"frozen transfer root is absent: {root}")
    sidecar_contract = contract.sidecar
    try:
        loaded = read_semantic_cell_sidecar(
            sidecar_path,
            sidecar_manifest_path,
            expected_manifest_sha256=str(sidecar_contract["manifest_sha256"]),
            expected_config_sha256=str(sidecar_contract["config_sha256"]),
            expected_provenance_sha256=str(sidecar_contract["provenance_sha256"]),
        )
    except SemanticSidecarError as error:
        raise EditingGateZeroRuntimeError(
            "frozen validation semantic sidecar failed verification"
        ) from error
    manifest = loaded.manifest
    unified_sha256, representability_sha256 = (
        _validated_sidecar_external_identities(contract, loaded)
    )
    provenance = manifest["provenance"]
    _require_file_sha256(
        root / "UNIFIED_PACKED_MANIFEST.json",
        unified_sha256,
        name="unified packed manifest",
    )
    _require_file_sha256(
        root / "REPRESENTABILITY_OVERLAY.json",
        representability_sha256,
        name="representability overlay",
    )

    bindings: list[ValidationShardBinding] = []
    records: list[PathRecord] = []
    for raw_spec in provenance["source_shards"]:
        spec = _semantic_source_shard(raw_spec)
        try:
            directory = root.joinpath(*_LAYER_DIRECTORIES[spec.layer])
        except KeyError:
            raise EditingGateZeroRuntimeError(
                f"unknown validation source layer: {spec.layer!r}"
            ) from None
        packed_path = directory / spec.packed_shard_name
        packed_manifest_path = manifest_path_for(packed_path)
        packed_overlay_path = overlay_path_for(packed_path)
        _require_file_sha256(
            packed_path,
            spec.packed_shard_content_sha256,
            name=f"packed shard {spec.layer}/{spec.packed_shard_name}",
        )
        _require_file_sha256(
            packed_manifest_path,
            spec.packed_manifest_sha256,
            name=f"packed manifest {spec.layer}/{spec.packed_shard_name}",
        )
        _require_file_sha256(
            packed_overlay_path,
            spec.provenance_overlay_sha256,
            name=f"packed overlay {spec.layer}/{spec.packed_shard_name}",
        )
        shard_records: list[PathRecord] = []
        try:
            addressed_rows = read_addressed_packed_shard(
                packed_path,
                verify_fraction=0.0,
            )
            for addressed in addressed_rows:
                shard_records.append(
                    PathRecord(
                        target_key=addressed.address.target_key,
                        path=addressed.path,
                        corpus_address=addressed.address,
                    )
                )
        except Exception as error:
            raise EditingGateZeroRuntimeError(
                f"addressed validation shard failed to decode: {packed_path}"
            ) from error
        if len(shard_records) != spec.effective_trace_count:
            raise EditingGateZeroRuntimeError(f"validation trace census drifted for {packed_path}")
        progress_rows = sum(record.path.path_length + 1 for record in shard_records)
        if progress_rows != spec.progress_row_count:
            raise EditingGateZeroRuntimeError(
                f"validation progress census drifted for {packed_path}"
            )
        observed_indices = tuple(
            record.corpus_address.entry_index
            for record in shard_records
            if record.corpus_address is not None
        )
        expected_indices = tuple(
            index
            for index in range(spec.packed_entry_count)
            if index not in set(spec.excluded_entry_indices)
        )
        if observed_indices != expected_indices:
            raise EditingGateZeroRuntimeError(
                f"validation entry-index census drifted for {packed_path}"
            )
        bindings.append(
            ValidationShardBinding(
                spec=spec,
                packed_path=packed_path,
                manifest_path=packed_manifest_path,
                overlay_path=packed_overlay_path,
            )
        )
        records.extend(shard_records)

    sidecar_keys = set(loaded.semantic_cell_ids)
    record_keys = {
        (
            record.corpus_address.packed_shard_content_sha256,
            record.corpus_address.entry_index,
            progress_index,
        )
        for record in records
        if record.corpus_address is not None
        for progress_index in range(record.path.path_length + 1)
    }
    if sidecar_keys != record_keys:
        raise EditingGateZeroRuntimeError(
            "semantic sidecar and addressed validation rows are not an exact census"
        )
    if len(records) != manifest["counts"]["traces"]:
        raise EditingGateZeroRuntimeError(
            "addressed validation trace count disagrees with the sidecar"
        )
    return FrozenValidationSource(
        records=tuple(records),
        sidecar=loaded,
        bindings=tuple(
            sorted(
                bindings,
                key=lambda item: item.spec.packed_shard_content_sha256,
            )
        ),
        unified_packed_manifest_sha256=unified_sha256,
        representability_overlay_sha256=representability_sha256,
    )


def bind_active8_validation_source(
    source: FrozenValidationSource,
    *,
    contract: GateZeroRuntimeContract,
    admission: Active8TraceAdmission,
) -> GateZeroActive8ValidationSource:
    """Apply exact whole-trace Active8 admission before Gate0 probe selection."""

    if not isinstance(source, FrozenValidationSource):
        raise TypeError("Gate0 Active8 binding requires a FrozenValidationSource")
    _validate_active8_source_identity(
        contract,
        unified_packed_manifest_sha256=source.unified_packed_manifest_sha256,
        admission=admission,
    )
    observed_lanes: set[tuple[str, str, str]] = set()
    try:
        for binding in source.bindings:
            spec = binding.spec
            lane = (spec.layer, spec.partition, spec.packed_shard_name)
            observed_lanes.add(lane)
            admission.assert_complete_source_shard(
                packed_shard_name=spec.packed_shard_name,
                layer=spec.layer,
                partition=spec.partition,
                observed_digest=spec.packed_shard_content_sha256,
                observed_entries=spec.packed_entry_count,
            )
        admission.assert_partition_shards("validation", observed_lanes)
    except Active8TraceInventoryError as error:
        raise EditingGateZeroRuntimeError(
            "Gate0 validation shards disagree with exact Active8 admission"
        ) from error

    admitted_records: list[PathRecord] = []
    admitted_trace_keys: set[tuple[str, int]] = set()
    family_counts: Counter[str] = Counter(
        {family: 0 for family in PRODUCTION_ACTIVE_FAMILIES}
    )
    try:
        for record in source.records:
            address = record.corpus_address
            if address is None:
                raise EditingGateZeroRuntimeError(
                    "Gate0 validation record lacks an immutable packed address"
                )
            if not admission.is_accepted(address):
                continue
            admitted_records.append(record)
            admitted_trace_keys.add(
                (
                    address.packed_shard_content_sha256,
                    address.entry_index,
                )
            )
            for step in record.path.trace.steps:
                family = canonical_family(step.rule_name)
                if family not in family_counts:
                    raise EditingGateZeroRuntimeError(
                        "Active8-admitted trace contains a disabled family"
                    )
                family_counts[family] += 1
    except (Active8TraceInventoryError, ActionCodecError) as error:
        raise EditingGateZeroRuntimeError(
            "Gate0 could not apply exact whole-trace Active8 admission"
        ) from error

    admitted_semantic_cell_ids = {
        row.exact_key: row.semantic_cell_id
        for row in source.sidecar.rows
        if (
            row.packed_shard_content_sha256,
            row.entry_index,
        )
        in admitted_trace_keys
    }
    if not admitted_records:
        raise EditingGateZeroRuntimeError(
            "Active8 admission leaves no Gate0 validation traces"
        )
    admitted_record_keys = {
        (
            record.corpus_address.packed_shard_content_sha256,
            record.corpus_address.entry_index,
            progress_index,
        )
        for record in admitted_records
        if record.corpus_address is not None
        for progress_index in range(record.path.path_length + 1)
    }
    if set(admitted_semantic_cell_ids) != admitted_record_keys:
        raise EditingGateZeroRuntimeError(
            "Active8-admitted validation traces and semantic rows are not an exact census"
        )
    return GateZeroActive8ValidationSource(
        source=source,
        admission=admission,
        records=tuple(admitted_records),
        semantic_cell_ids=admitted_semantic_cell_ids,
        admitted_teacher_examples_by_family={
            family: family_counts[family] for family in PRODUCTION_ACTIVE_FAMILIES
        },
    )


def build_production_ringcore_catalog(*, max_atoms: int) -> TypedRingCatalog:
    """Reconstruct and verify the deterministic five-seed production catalog."""

    if type(max_atoms) is not int or max_atoms <= 0:
        raise ValueError("max_atoms must be a positive integer")

    def seed_trace(smiles: str):
        target = pad_molecular_graph(
            smiles_to_molecular_graph(smiles),
            max_atoms,
        )
        source = DegreeBoundedCarbonTreePrior(sizes=(target.n_real_atoms,)).sample(
            np.random.default_rng(1),
            n_slots=max_atoms,
        )
        return compile_carbon_tree_to_target(
            source,
            target,
            use_bond_reroute=True,
            align_source=True,
        )

    catalog = build_typed_ring_catalog(
        tuple(seed_trace(smiles) for smiles in PRODUCTION_RINGCORE_CATALOG_SEEDS)
    )
    observed = ring_catalog_fingerprint(catalog)
    if observed != PRODUCTION_RINGCORE_CATALOG_FINGERPRINT:
        raise EditingGateZeroRuntimeError(
            "reconstructed production RingCore catalog fingerprint drifted: "
            f"{observed} != {PRODUCTION_RINGCORE_CATALOG_FINGERPRINT}"
        )
    return catalog


def build_scratch_ringcore_model(
    contract: GateZeroRuntimeContract,
) -> tuple[FactorizedTraceletRateModel, InitializationParityReport]:
    """Build the exact scratch model and prove bit-exact scratch parity."""

    config = contract.model
    if int(config["atom_vocabulary_class_count"]) != len(ORGANIC_VOCABULARY.classes):
        raise EditingGateZeroRuntimeError("runtime contract broad-organic vocabulary width drifted")
    catalog = build_production_ringcore_catalog(max_atoms=int(config["max_atoms"]))
    torch.manual_seed(int(config["seed"]))
    model = FactorizedTraceletRateModel(
        catalog,
        hidden_dim=int(config["hidden_dim"]),
        message_passing_steps=int(config["message_passing_steps"]),
        mark_dim=int(config["mark_dim"]),
        enable_ring_restates=bool(config["enable_ring_restates"]),
        enable_cyclic_graft=bool(config["enable_cyclic_graft"]),
        enable_heteroatom_scan=bool(config["enable_heteroatom_scan"]),
        enable_ring_opening=bool(config["enable_ring_opening"]),
        enable_cycle_ops=bool(config["enable_cycle_ops"]),
        enable_ring_grow_macro=bool(config["enable_ring_grow_macro"]),
        enable_ring_system_delete=bool(config["enable_ring_system_delete"]),
        atom_vocabulary=ORGANIC_VOCABULARY,
    ).eval()
    if (
        model.hidden_dim != int(config["hidden_dim"])
        or model.message_passing_steps != int(config["message_passing_steps"])
        or model.mark_dim != int(config["mark_dim"])
        or model.atom_vocabulary is not ORGANIC_VOCABULARY
        or ring_catalog_fingerprint(model.ring_catalog) != config["catalog_fingerprint"]
        or model.operator_capabilities.fingerprint() != config["operator_capability_fingerprint"]
    ):
        raise EditingGateZeroRuntimeError(
            "constructed scratch model disagrees with the frozen architecture"
        )
    fresh_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    plan = build_scratch_initialization_plan(fresh_state)
    parity = assert_initialization_parity(
        plan,
        fresh_target_state=fresh_state,
        initialized_state=model.state_dict(),
    )
    return model, parity


def _support_signature_payload(
    model: FactorizedTraceletRateModel,
    *,
    max_atoms: int,
) -> dict[str, Any]:
    operator_registry_source_sha256 = _source_set_sha256(
        _OPERATOR_REGISTRY_SOURCES
    )
    canonicalizer_contract_sha256 = _source_set_sha256(_CANONICALIZER_CONTRACT_SOURCES)
    return {
        "schema": "compose.editing.gate_zero_support_signature",
        "schema_version": 1,
        # The 16-character value is the frozen historical corpus identity.
        # The full digest separately binds all current source bytes without
        # changing that legacy field's meaning.
        "operator_registry_hash": operator_registry_source_sha256[:16],
        "operator_registry_source_sha256": operator_registry_source_sha256,
        "canonicalizer": "canonical_state_key",
        "canonicalizer_contract_sha256": canonicalizer_contract_sha256,
        "persistent_state_digest_schema": PERSISTENT_STATE_DIGEST_SCHEMA,
        "persistent_state_digest_schema_version": (PERSISTENT_STATE_DIGEST_VERSION),
        "successor_coordinate_schema_version": (SUCCESSOR_COORDINATE_SCHEMA_VERSION),
        "operator_capability_fingerprint": (model.operator_capabilities.fingerprint()),
        "ordered_family_vocabulary": list(MARK_RULE_NAMES),
        "ordered_action_table_vocabulary": list(PRODUCTION_ACTION_TABLE_VOCABULARY),
        "atom_vocabulary": [
            [str(element), int(valence)] for element, valence in model.atom_vocabulary.classes
        ],
        "max_atoms": int(max_atoms),
        "atom_insert_arity_support": [0, 1],
        "bond_vocabulary": [
            "none",
            "single",
            "double",
            "triple",
            "aromatic",
        ],
        "charge_policy": CHARGE_POLICY_VERSION,
        "valence_policy": "declared_element_valence_classes_plus_hydrogen_budget",
        "catalog_fingerprint": ring_catalog_fingerprint(model.ring_catalog),
        "enable_ring_restates": model.enable_ring_restates,
        "enable_cyclic_graft": model.enable_cyclic_graft,
        "enable_heteroatom_scan": model.enable_heteroatom_scan,
        "enable_ring_opening": model.enable_ring_opening,
        "enable_cycle_ops": model.enable_cycle_ops,
        "enable_ring_grow_macro": model.enable_ring_grow_macro,
        "enable_ring_system_delete": model.enable_ring_system_delete,
        "embedded_jump_chain_policy": "productive_canonical_successors",
    }


def build_exact_cache_provenance(
    model: FactorizedTraceletRateModel,
    source: FrozenValidationSource | GateZeroActive8ValidationSource,
    *,
    selected_shard_digests: Iterable[str],
    max_atoms: int,
) -> Mapping[str, SuccessorFiberCacheProvenance]:
    """Build current implementation provenance for exactly selected shards."""

    selected = tuple(sorted(set(selected_shard_digests)))
    if not selected:
        raise EditingGateZeroRuntimeError("successor probe selected no packed shard")
    bindings = source.binding_by_digest
    if set(selected) - set(bindings):
        raise EditingGateZeroRuntimeError("selected probe references an undeclared packed shard")
    support_signature_sha256 = _stable_sha256(
        _support_signature_payload(model, max_atoms=max_atoms)
    )
    operator_registry_hash = _source_set_sha256(_OPERATOR_REGISTRY_SOURCES)[:16]
    canonicalizer_hash = _source_set_sha256(_CANONICALIZER_CONTRACT_SOURCES)
    result: dict[str, SuccessorFiberCacheProvenance] = {}
    for digest in selected:
        binding = bindings[digest]
        result[digest] = SuccessorFiberCacheProvenance(
            operator_registry_hash=operator_registry_hash,
            capability_hash=model.operator_capabilities.fingerprint(),
            support_signature_sha256=support_signature_sha256,
            canonicalization_version=(f"canonical_state_key:{canonicalizer_hash[:16]}"),
            canonicalizer_contract_sha256=canonicalizer_hash,
            packed_corpus_schema=PACKED_STORE_SCHEMA,
            packed_corpus_schema_version=PACKED_STORE_SCHEMA_VERSION,
            packed_shard_content_sha256=digest,
            packed_manifest_sha256=(binding.spec.packed_manifest_sha256),
            packed_provenance_overlay_sha256=(binding.spec.provenance_overlay_sha256),
            unified_packed_manifest_sha256=(source.unified_packed_manifest_sha256),
            representability_overlay_sha256=(source.representability_overlay_sha256),
            coordinate_schema_version=SUCCESSOR_COORDINATE_SCHEMA_VERSION,
            tensorization_implementation_hash=(tensorization_implementation_hash()),
            fiber_compiler_implementation_hash=(fiber_compiler_implementation_hash()),
        )
    return MappingProxyType(result)


def _probe_example(
    record: PathRecord,
    *,
    record_index: int,
    progress_index: int,
    cache_record_by_address: Mapping[
        SuccessorFiberCacheAddress,
        Any,
    ],
    semantic_cell_ids: Mapping[tuple[str, int, int], str | None],
) -> FactorizedSuccessorExample:
    address = record.corpus_address
    if address is None:
        raise EditingGateZeroRuntimeError("selected Gate-0 trace has no immutable address")
    cache_address = SuccessorFiberCacheAddress.from_packed_trace(
        address,
        progress_index=progress_index,
    )
    try:
        cache_record = cache_record_by_address[cache_address]
    except KeyError:
        raise EditingGateZeroRuntimeError(
            "selected progress row is absent from the compiled cache"
        ) from None
    cell_key = (
        address.packed_shard_content_sha256,
        address.entry_index,
        progress_index,
    )
    if cell_key not in semantic_cell_ids:
        raise EditingGateZeroRuntimeError(
            "selected progress row is absent from the semantic sidecar"
        )
    if progress_index == record.path.path_length:
        rule_name = None
        action = None
        teacher_rate = 0.0
    else:
        step = record.path.trace.steps[progress_index]
        rule_name = step.rule_name
        action = step.action
        teacher_rate = float(record.path.operational_jump_rate(progress_index))
    return FactorizedSuccessorExample(
        mark_example=FactorizedMarkExample(
            state=record.path.state_at(progress_index),
            time=GATE_ZERO_SUCCESSOR_SUPPORT_TIME,
            teacher_action=action,
            teacher_rule_name=rule_name,
            teacher_rate=teacher_rate,
            importance_weight=1.0,
            record_index=record_index,
            progress_index=progress_index,
        ),
        cache_record=cache_record,
        semantic_cell_id=semantic_cell_ids[cell_key],
    )


def reconstruct_probe_audit_batches(
    model: FactorizedTraceletRateModel,
    probe: GateZeroSuccessorProbe,
    *,
    batch_size: int,
) -> tuple[FactorizedSuccessorBatch, ...]:
    """Reconstruct every selected trace-progress row exactly once."""

    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    cache_record_by_address = {record.address: record for record in probe.cache_records}
    if len(cache_record_by_address) != len(probe.cache_records):
        raise EditingGateZeroRuntimeError("compiled Gate-0 cache repeats an exact progress address")
    examples = tuple(
        _probe_example(
            record,
            record_index=record_index,
            progress_index=progress_index,
            cache_record_by_address=cache_record_by_address,
            semantic_cell_ids=probe.semantic_cell_ids,
        )
        for record_index, record in enumerate(probe.selected_path_records)
        for progress_index in range(record.path.path_length + 1)
    )
    capabilities = model.operator_capabilities
    collator = FactorizedSuccessorCollator(
        FactorizedMarkCollator(
            use_aromatic_bond_view=True,
            ring_catalog=model.ring_catalog,
            compute_ring_grow_support=(capabilities.compute_ring_grow_support),
            compute_ring_restates=capabilities.compute_ring_restates,
            compute_cyclic_graft=capabilities.compute_cyclic_graft,
            compute_ring_opening=capabilities.compute_ring_opening,
            compute_ring_system_delete=capabilities.compute_ring_system_delete,
        )
    )
    return tuple(
        collator(list(examples[start : start + batch_size]))
        for start in range(0, len(examples), batch_size)
    )


@dataclass(frozen=True)
class GateZeroRuntimeResult:
    """In-memory result ready for an immutable evidence freeze."""

    contract: GateZeroRuntimeContract
    source: GateZeroActive8ValidationSource
    model: FactorizedTraceletRateModel
    initialization_parity: InitializationParityReport
    probe: GateZeroSuccessorProbe
    support_report: StepZeroSuccessorSupportReport

    @property
    def training_authorized(self) -> bool:
        return False

    def assert_training_authorized(self) -> None:
        raise EditingGateZeroRuntimeError("Gate-0 structural evidence never authorizes training")


def run_gate_zero_runtime(
    *,
    contract: GateZeroRuntimeContract,
    transfer_root: Path,
    sidecar_path: Path,
    sidecar_manifest_path: Path,
    active8_admission: Active8TraceAdmission,
) -> GateZeroRuntimeResult:
    """Run the exact bounded structural probe without an optimizer update."""

    frozen_source = load_frozen_validation_source(
        contract=contract,
        transfer_root=transfer_root,
        sidecar_path=sidecar_path,
        sidecar_manifest_path=sidecar_manifest_path,
    )
    source = bind_active8_validation_source(
        frozen_source,
        contract=contract,
        admission=active8_admission,
    )
    required_semantic_cells = tuple(
        sorted({cell for cell in source.semantic_cell_ids.values() if cell is not None})
    )
    if not required_semantic_cells:
        raise EditingGateZeroRuntimeError(
            "Active8-admitted Gate0 validation source has no semantic cells"
        )
    probe_config = GateZeroSuccessorProbeConfig(
        required_families=contract.required_families,
        required_semantic_cells=required_semantic_cells,
    )
    selected_records = select_gate_zero_successor_probe_records(
        source.records,
        semantic_cell_ids=source.semantic_cell_ids,
        config=probe_config,
    )
    selected_keys = {
        (
            record.corpus_address.packed_shard_content_sha256,
            record.corpus_address.entry_index,
            progress_index,
        )
        for record in selected_records
        if record.corpus_address is not None
        for progress_index in range(record.path.path_length + 1)
    }
    selected_cells = {key: source.semantic_cell_ids[key] for key in selected_keys}
    model, parity = build_scratch_ringcore_model(contract)
    selected_shards = {
        record.corpus_address.packed_shard_content_sha256
        for record in selected_records
        if record.corpus_address is not None
    }
    provenance = build_exact_cache_provenance(
        model,
        source,
        selected_shard_digests=selected_shards,
        max_atoms=int(contract.model["max_atoms"]),
    )
    probe = build_gate_zero_successor_probe(
        model,
        selected_records,
        semantic_cell_ids=selected_cells,
        config=probe_config,
        provenance_by_shard=provenance,
    )
    if tuple(probe.selected_path_records) != tuple(selected_records):
        raise EditingGateZeroRuntimeError(
            "preselected complete traces changed under the production probe"
        )
    batches = reconstruct_probe_audit_batches(
        model,
        probe,
        batch_size=contract.audit_batch_size,
    )
    support_report = audit_successor_support_before_optimization(
        model,
        batches,
        required_families=contract.required_families,
        required_semantic_cells=required_semantic_cells,
    )
    if support_report.row_count != len(probe.rows):
        raise EditingGateZeroRuntimeError(
            "step-zero support audit did not consume every selected progress row"
        )
    return GateZeroRuntimeResult(
        contract=contract,
        source=source,
        model=model,
        initialization_parity=parity,
        probe=probe,
        support_report=support_report,
    )


def _atomic_write_if_absent(path: Path, content: bytes) -> bool:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, destination)
            return True
        except FileExistsError:
            if destination.read_bytes() != content:
                raise FileExistsError(
                    f"Gate-0 evidence path already exists with different content: {destination}"
                ) from None
            return False
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _evidence_body(
    result: GateZeroRuntimeResult,
    *,
    cache_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    sidecar_manifest = result.source.sidecar.manifest
    parameter_count = sum(int(parameter.numel()) for parameter in result.model.parameters())
    support_signature = _support_signature_payload(
        result.model,
        max_atoms=int(result.contract.model["max_atoms"]),
    )
    return {
        "schema": EDITING_GATE_ZERO_RUNTIME_EVIDENCE_SCHEMA,
        "schema_version": EDITING_GATE_ZERO_RUNTIME_EVIDENCE_VERSION,
        "status": EDITING_GATE_ZERO_RUNTIME_STATUS,
        "training_authorized": False,
        "quality_thresholds": None,
        "training_decision": None,
        "optimizer_steps": 0,
        "runtime_contract_sha256": result.contract.sha256,
        "active8_admission": result.source.active8_evidence(),
        "source": {
            "semantic_sidecar_manifest_sha256": (sidecar_manifest["manifest_sha256"]),
            "semantic_sidecar_source_sha256": (sidecar_manifest["source_sha256"]),
            "semantic_census_sha256": (sidecar_manifest["semantic_census_sha256"]),
            "unified_packed_manifest_sha256": (result.source.unified_packed_manifest_sha256),
            "representability_overlay_sha256": (result.source.representability_overlay_sha256),
            "validation_trace_count": len(result.source.records),
            "validation_progress_row_count": len(result.source.semantic_cell_ids),
            "validation_shards": [asdict(binding.spec) for binding in result.source.bindings],
        },
        "model": {
            **dict(result.contract.model),
            "state_dict_semantic_sha256": state_dict_semantic_sha256(result.model.state_dict()),
            "parameter_count": parameter_count,
            "initialization_regime": "scratch",
            "initialization_parity": asdict(result.initialization_parity),
        },
        "cache_contract": {
            "support_signature": support_signature,
            "support_signature_sha256": _stable_sha256(support_signature),
            "operator_registry_hash": _source_set_sha256(_OPERATOR_REGISTRY_SOURCES)[:16],
            "operator_registry_source_sha256": _source_set_sha256(
                _OPERATOR_REGISTRY_SOURCES
            ),
            "canonicalizer_contract_sha256": _source_set_sha256(_CANONICALIZER_CONTRACT_SOURCES),
            "tensorization_implementation_hash": (tensorization_implementation_hash()),
            "fiber_compiler_implementation_hash": (fiber_compiler_implementation_hash()),
        },
        "probe": result.probe.manifest(),
        "cache_artifacts": cache_entries,
        "support_audit": asdict(result.support_report),
    }


def freeze_gate_zero_runtime_evidence(
    result: GateZeroRuntimeResult,
    *,
    cache_directory: Path,
    evidence_manifest_path: Path,
) -> dict[str, Any]:
    """Freeze cache bytes and a self-hashed manifest without overwrite."""

    if not isinstance(result, GateZeroRuntimeResult):
        raise TypeError("result must be GateZeroRuntimeResult")
    cache_root = Path(cache_directory)
    manifest_path = Path(evidence_manifest_path)
    cache_entries: list[dict[str, Any]] = []
    cache_payloads: list[tuple[Path, bytes]] = []
    for shard in result.probe.cache_shards:
        filename = (
            f"{shard.packed_shard_content_sha256}.{shard.encoded_sha256}.successor_fiber_cache.json"
        )
        destination = cache_root / filename
        cache_payloads.append((destination, shard.encoded))
        cache_entries.append(
            {
                "packed_shard_content_sha256": (shard.packed_shard_content_sha256),
                "relative_path": filename,
                "cache_content_sha256": shard.content_sha256,
                "cache_file_sha256": shard.encoded_sha256,
                "cache_file_bytes": len(shard.encoded),
                "record_count": len(shard.cache.records),
                "provenance": asdict(shard.cache.provenance),
            }
        )
    body = _evidence_body(result, cache_entries=cache_entries)
    manifest = {**body, "evidence_sha256": _stable_sha256(body)}
    manifest_bytes = _canonical_json_bytes(manifest) + b"\n"

    created_cache_paths: list[Path] = []
    try:
        for destination, encoded in cache_payloads:
            if _atomic_write_if_absent(destination, encoded):
                created_cache_paths.append(destination)
        _atomic_write_if_absent(manifest_path, manifest_bytes)
    except Exception:
        for created in created_cache_paths:
            created.unlink(missing_ok=True)
        raise
    return manifest


__all__ = [
    "EDITING_GATE_ZERO_RUNTIME_CONTRACT_SCHEMA",
    "EDITING_GATE_ZERO_RUNTIME_CONTRACT_STATUS",
    "EDITING_GATE_ZERO_RUNTIME_CONTRACT_VERSION",
    "EDITING_GATE_ZERO_RUNTIME_EVIDENCE_SCHEMA",
    "EDITING_GATE_ZERO_RUNTIME_EVIDENCE_VERSION",
    "EDITING_GATE_ZERO_RUNTIME_STATUS",
    "GATE_ZERO_ACTIVE8_CENSUS_FIELDS",
    "GATE_ZERO_ACTIVE8_IDENTITY_FIELDS",
    "PRODUCTION_ACTIVE_FAMILIES",
    "PRODUCTION_ACTION_TABLE_VOCABULARY",
    "PRODUCTION_RINGCORE_CATALOG_FINGERPRINT",
    "PRODUCTION_RINGCORE_CATALOG_SEEDS",
    "EditingGateZeroRuntimeError",
    "FrozenValidationSource",
    "GateZeroActive8ValidationSource",
    "GateZeroRuntimeContract",
    "GateZeroRuntimeResult",
    "ValidationShardBinding",
    "bind_active8_validation_source",
    "build_exact_cache_provenance",
    "build_production_ringcore_catalog",
    "build_scratch_ringcore_model",
    "freeze_gate_zero_runtime_evidence",
    "load_frozen_validation_source",
    "load_gate_zero_runtime_contract",
    "reconstruct_probe_audit_batches",
    "run_gate_zero_runtime",
]
