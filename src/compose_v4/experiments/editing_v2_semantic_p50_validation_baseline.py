"""Physical prospective validation baseline contract for semantic Editing-V2 P50.

This module does not evaluate a model and grants no training authority.  It
defines two immutable artifacts needed by a future producer:

* a content-addressed, validation-only inventory whose JSONL order is fully
  deterministic; and
* a pre-update scratch-model baseline containing exact family and semantic-cell
  counts with finite canonical-successor negative log likelihoods (NLLs).

Both artifacts bind the current Editing-V2 process, the physical semantic
source inventory, the physical prepared recipe and its frozen policy, and the
scratch initial model-state identity.  Their openers recompute physical and
semantic identities instead of trusting an in-memory receipt.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from compose_v4.chem.persistent_state_identity import persistent_slot_state_sha256
from compose_v4.data.editing_corpus_contract import ACTIVE8_FAMILIES
from compose_v4.data.editing_v2_semantic_capability_cells import (
    SemanticCapabilityCellRegistry,
    classify_verified_structural_transition,
    load_semantic_capability_cell_registry,
)
from compose_v4.data.successor_fiber_cache import (
    SuccessorFiberCacheAddress,
    SuccessorFiberCacheRecord,
    successor_fiber_cache_record_payload,
)
from compose_v4.experiments.editing_p50_gate import state_dict_semantic_sha256
from compose_v4.experiments.editing_v2_semantic_p50_recipe_stream import (
    PREPARED_SCHEMA,
    PREPARED_STATUS,
    load_semantic_p50_recipe_policy,
    semantic_p50_time_hex,
)
from compose_v4.experiments.editing_v2_semantic_p50_source_inventory import (
    VerifiedSemanticP50SourceInventory,
)
from compose_v4.experiments.editing_v2_semantic_p50_successor_cache import (
    SemanticP50SuccessorCache,
)
from compose_v4.experiments.editing_v2_semantic_runtime import SemanticScratchRuntime
from compose_v4.rewrite.editing_v2_process_identity import (
    EditingV2ProcessIdentityError,
    require_editing_v2_process_identity,
)
from compose_v4.rewrite.kernel import canonical_state_key

SCHEMA_VERSION = 1
VALIDATION_PARTITION_ROLE = "validation"

INVENTORY_ROWS_FILENAME = "validation_candidates.jsonl"
INVENTORY_COMPLETION_FILENAME = "SEMANTIC_P50_VALIDATION_INVENTORY_COMPLETE.json"
INVENTORY_SCHEMA = "compose.editing_v2.semantic_p50_validation_inventory"
INVENTORY_STATUS = "FROZEN_VALIDATION_STREAM_NO_P50_AUTHORITY"

BASELINE_RESULT_FILENAME = "validation_baseline_result.json"
BASELINE_COMPLETION_FILENAME = "SEMANTIC_P50_VALIDATION_BASELINE_COMPLETE.json"
BASELINE_RESULT_SCHEMA = "compose.editing_v2.semantic_p50_validation_baseline_result"
BASELINE_COMPLETION_SCHEMA = "compose.editing_v2.semantic_p50_validation_baseline_completion"
BASELINE_STATUS = "COMPLETE_PREUPDATE_SCRATCH_BASELINE_NO_P50_AUTHORITY"

EVALUATION_ENVIRONMENT_RECEIPT_SCHEMA = (
    "compose.editing_v2.semantic_p50_validation_evaluation_environment"
)
EVALUATION_ENVIRONMENT_RECEIPT_STATUS = "COMPLETE_PHYSICAL_EVALUATION_ENVIRONMENT_NO_P50_AUTHORITY"

# These are the tracked production primitives that define canonical-successor
# scoring.  The baseline publisher is deliberately excluded: it validates and
# seals producer output but does not itself evaluate a model.
_EVALUATOR_IMPLEMENTATION_SOURCES = (
    "src/compose_v4/data/successor_fiber_cache.py",
    "src/compose_v4/experiments/editing_successor_trainer.py",
    "src/compose_v4/experiments/factorized_successor_training.py",
    "src/compose_v4/experiments/production_successor_kernel.py",
    "src/compose_v4/experiments/successor_kernel.py",
    "src/compose_v4/model/factorized_tracelet_rate_model.py",
    "src/compose_v4/model/segmented_successor.py",
    "src/compose_v4/rewrite/kernel.py",
)

_NO_AUTHORITY = {
    "training_authorized": False,
    "bounded_p50_authorized": False,
    "p500_authorized": False,
    "long_training_authorized": False,
    "checkpoint_selection_authorized": False,
    "final_test_selection_authorized": False,
}
_SOURCE_NO_AUTHORITY = {
    "training_authorized": False,
    "gate_zero_authorized": False,
    "t1_authorized": False,
    "bounded_p50_authorized": False,
    "long_training_authorized": False,
    "checkpoint_selection_authorized": False,
    "final_test_selection_authorized": False,
}
_HEX = frozenset("0123456789abcdef")
_BASELINE_OBJECTIVE = {
    "unit": "productive_embedded_canonical_successor",
    "metric": "mean_canonical_successor_nll_nats",
    "hazard_included": False,
    "terminal_rows_included": False,
    "evaluation_order": "deterministic_validation_inventory_order",
    "model_state": "exact_pre_update_scratch_initial_model_state",
}
_INVENTORY_COMPLETION_FIELDS = {
    "schema",
    "schema_version",
    "status",
    *_NO_AUTHORITY,
    "run_identity_sha256",
    "binding",
    "partition_role",
    "required_families",
    "required_semantic_cells",
    "candidate_rows_relative_path",
    "candidate_rows_file_sha256",
    "candidate_count",
    "candidate_inventory_sha256",
    "ordered_validation_stream_sha256",
    "ordered_validation_time_stream_sha256",
    "family_counts",
    "semantic_cell_counts",
    "successor_cache_completion_sha256",
    "successor_cache_validation_contract_sha256",
    "successor_cache_validation_candidate_inventory_sha256",
    "completion_sha256",
}
_BASELINE_RESULT_FIELDS = {
    "schema",
    "schema_version",
    "status",
    *_NO_AUTHORITY,
    "binding",
    "validation_inventory_completion_file_sha256",
    "validation_inventory_completion_sha256",
    "ordered_validation_stream_sha256",
    "ordered_validation_time_stream_sha256",
    "objective",
    "evaluated_model_state_sha256",
    "successor_cache_completion_sha256",
    "successor_cache_manifest_sha256",
    "successor_cache_validation_contract_sha256",
    "successor_cache_validation_candidate_inventory_sha256",
    "evaluator_implementation_sha256",
    "evaluation_environment_receipt_file_sha256",
    "evaluation_environment_receipt_sha256",
    "execution_environment",
    "example_count",
    "address_set_sha256",
    "evaluation_records",
    "family_metrics",
    "semantic_cell_metrics",
    "result_sha256",
}
_BASELINE_COMPLETION_FIELDS = {
    "schema",
    "schema_version",
    "status",
    *_NO_AUTHORITY,
    "result_relative_path",
    "result_file_sha256",
    "result_sha256",
    "validation_inventory_completion_file_sha256",
    "validation_inventory_completion_sha256",
    "successor_cache_completion_sha256",
    "successor_cache_validation_contract_sha256",
    "evaluator_implementation_sha256",
    "evaluation_environment_receipt_sha256",
    "binding",
    "completion_sha256",
}

_EVALUATION_ENVIRONMENT_RECEIPT_FIELDS = {
    "schema",
    "schema_version",
    "status",
    *_NO_AUTHORITY,
    "producer_implementation_sha256",
    "execution_environment",
    "execution_environment_sha256",
    "evaluated_model_state_sha256",
    "successor_cache_completion_sha256",
    "successor_cache_manifest_sha256",
    "successor_cache_validation_contract_sha256",
    "successor_cache_validation_candidate_inventory_sha256",
    "objective",
    "evaluation_address_set_sha256",
    "evaluation_records_sha256",
    "receipt_sha256",
}
_EXECUTION_ENVIRONMENT_FIELDS = {
    "hardware_class",
    "device_name",
    "device_capability",
    "accelerator_class",
    "dtype",
    "mixed_precision",
    "batch_size",
    "python_version",
    "torch_version",
    "cuda_version",
    "cudnn_version",
    "rdkit_version",
    "numpy_version",
    "environment_sha256",
}


class SemanticP50ValidationBaselineError(RuntimeError):
    """A validation inventory or baseline is incomplete, leaked, or mutable."""


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    try:
        raw = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SemanticP50ValidationBaselineError(
            "semantic P50 validation artifact is not finite canonical JSON"
        ) from error
    return raw + (b"\n" if newline else b"")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            while block := handle.read(1 << 20):
                digest.update(block)
    except OSError as error:
        raise SemanticP50ValidationBaselineError(
            f"cannot hash semantic P50 validation artifact: {path}"
        ) from error
    return digest.hexdigest()


def semantic_p50_validation_evaluator_implementation_sha256(*, repo_root: Path) -> str:
    """Hash the exact tracked sources that implement successor scoring."""

    root = Path(repo_root).resolve()
    digest = hashlib.sha256()
    for relative in _EVALUATOR_IMPLEMENTATION_SOURCES:
        path = root / relative
        if not path.is_file():
            raise SemanticP50ValidationBaselineError(
                f"semantic P50 evaluator implementation source is absent: {relative}"
            )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _require_sha(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise SemanticP50ValidationBaselineError(f"{field} must be a lowercase SHA-256")
    return value


def _require_text(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any(ord(character) < 32 for character in value)
    ):
        raise SemanticP50ValidationBaselineError(f"{field} must be normalized nonempty text")
    return value


def _require_count(value: object, *, field: str, positive: bool = False) -> int:
    if type(value) is not int or value < int(positive):
        qualifier = "positive" if positive else "nonnegative"
        raise SemanticP50ValidationBaselineError(f"{field} must be a {qualifier} integer")
    return value


def _require_count_map(
    value: object,
    *,
    field: str,
    expected_keys: Sequence[str],
) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(expected_keys):
        raise SemanticP50ValidationBaselineError(
            f"{field} keys differ from the exact expected identities"
        )
    return {
        key: _require_count(value[key], field=f"{field}.{key}", positive=True)
        for key in expected_keys
    }


def _require_nll(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SemanticP50ValidationBaselineError(f"{field} must be a finite NLL")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise SemanticP50ValidationBaselineError(f"{field} must be a finite nonnegative NLL")
    return result


def _load_canonical(path: Path, *, field: str) -> tuple[dict[str, Any], bytes]:
    source = Path(path).resolve()
    if not source.is_file():
        raise SemanticP50ValidationBaselineError(f"{field} is absent: {source}")
    try:
        raw = source.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SemanticP50ValidationBaselineError(f"{field} is unreadable: {source}") from error
    if not isinstance(payload, dict) or raw != _canonical_bytes(payload, newline=True):
        raise SemanticP50ValidationBaselineError(
            f"{field} must be canonical newline-terminated JSON"
        )
    return payload, raw


def _address_payload(address: SuccessorFiberCacheAddress) -> dict[str, object]:
    return asdict(address)


def _validation_time_hex(*, stream_index: int, address: SuccessorFiberCacheAddress) -> str:
    time_hex = semantic_p50_time_hex(
        stream_index=_require_count(stream_index, field="stream_index"),
        address=address,
    )
    value = float.fromhex(time_hex)
    if not 0.0 < value < 1.0 or float.fromhex(time_hex).hex() != time_hex:
        raise SemanticP50ValidationBaselineError(
            "deterministic validation time is outside the open unit interval"
        )
    return time_hex


def _authority_is_false(payload: Mapping[str, Any]) -> bool:
    return all(payload.get(name) is expected for name, expected in _NO_AUTHORITY.items())


@dataclass(frozen=True, slots=True)
class SemanticP50ValidationBinding:
    """Exact upstream identities shared by the inventory and baseline."""

    process_identity_sha256: str
    source_inventory_file_sha256: str
    source_inventory_sha256: str
    model_runtime_identity_sha256: str
    active8_policy_sha256: str
    operator_capability_fingerprint: str
    decision_source_implementation_sha256: str
    prepared_recipe_file_sha256: str
    prepared_recipe_sha256: str
    recipe_policy_file_sha256: str
    recipe_policy_sha256: str
    scratch_initial_model_state_sha256: str
    capability_registry_sha256: str
    classifier_implementation_sha256: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if name == "operator_capability_fingerprint":
                value = getattr(self, name)
                if (
                    not isinstance(value, str)
                    or len(value) != 16
                    or any(character not in _HEX for character in value)
                ):
                    raise SemanticP50ValidationBaselineError(
                        "operator_capability_fingerprint must be 16 lowercase hex characters"
                    )
            else:
                _require_sha(getattr(self, name), field=name)

    def as_payload(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SemanticP50ValidationCandidate:
    """One nonterminal validation transition eligible for baseline scoring."""

    address: SuccessorFiberCacheAddress
    family: str
    semantic_cell_id: str
    data_lane: str
    assignment_sha256: str
    source_key: str
    source_state_sha256: str
    target_key: str
    target_state_sha256: str
    time_hex: str

    def __post_init__(self) -> None:
        if self.address.is_terminal:
            raise SemanticP50ValidationBaselineError(
                "semantic P50 validation candidates must be nonterminal"
            )
        if self.address.partition != VALIDATION_PARTITION_ROLE:
            raise SemanticP50ValidationBaselineError(
                "semantic P50 baseline permits only the validation role; train, "
                "controller_validation, and final_test are sealed against leakage"
            )
        if self.family not in ACTIVE8_FAMILIES:
            raise SemanticP50ValidationBaselineError(
                "semantic P50 validation candidate is outside Active8"
            )
        _require_text(self.semantic_cell_id, field="semantic_cell_id")
        components = self.semantic_cell_id.rsplit(":", 2)
        if len(components) != 3 or components[1] != self.family or not components[2]:
            raise SemanticP50ValidationBaselineError(
                "semantic P50 validation cell identity disagrees with its family"
            )
        _require_text(self.data_lane, field="data_lane")
        if self.address.layer != self.data_lane:
            raise SemanticP50ValidationBaselineError(
                "semantic P50 validation candidate lane differs from its address"
            )
        _require_sha(self.assignment_sha256, field="assignment_sha256")
        _require_text(self.source_key, field="source_key")
        _require_sha(self.source_state_sha256, field="source_state_sha256")
        _require_text(self.target_key, field="target_key")
        _require_sha(self.target_state_sha256, field="target_state_sha256")
        try:
            validation_time = float.fromhex(self.time_hex)
        except (TypeError, ValueError) as error:
            raise SemanticP50ValidationBaselineError(
                "candidate.time_hex must be an exact hexadecimal float"
            ) from error
        if not 0.0 < validation_time < 1.0 or validation_time.hex() != self.time_hex:
            raise SemanticP50ValidationBaselineError(
                "candidate.time_hex must round-trip inside the open unit interval"
            )

    def as_payload(self) -> dict[str, object]:
        return {
            "address": _address_payload(self.address),
            "family": self.family,
            "semantic_cell_id": self.semantic_cell_id,
            "data_lane": self.data_lane,
            "assignment_sha256": self.assignment_sha256,
            "source_key": self.source_key,
            "source_state_sha256": self.source_state_sha256,
            "target_key": self.target_key,
            "target_state_sha256": self.target_state_sha256,
            "time_hex": self.time_hex,
        }


@dataclass(frozen=True, slots=True)
class SemanticP50ValidationEvaluation:
    """One physically cache-bound per-address scratch evaluation receipt."""

    address: SuccessorFiberCacheAddress
    family: str
    semantic_cell_id: str
    cache_record_sha256: str
    time_hex: str
    canonical_successor_nll_nats: float

    def __post_init__(self) -> None:
        if self.address.is_terminal or self.address.partition != VALIDATION_PARTITION_ROLE:
            raise SemanticP50ValidationBaselineError(
                "validation evaluation address must be a nonterminal validation row"
            )
        if self.family not in ACTIVE8_FAMILIES:
            raise SemanticP50ValidationBaselineError(
                "validation evaluation family is outside Active8"
            )
        _require_text(self.semantic_cell_id, field="evaluation.semantic_cell_id")
        _require_sha(self.cache_record_sha256, field="evaluation.cache_record_sha256")
        try:
            validation_time = float.fromhex(self.time_hex)
        except (TypeError, ValueError) as error:
            raise SemanticP50ValidationBaselineError(
                "evaluation.time_hex must be an exact hexadecimal float"
            ) from error
        if not 0.0 < validation_time < 1.0 or validation_time.hex() != self.time_hex:
            raise SemanticP50ValidationBaselineError(
                "evaluation.time_hex must round-trip inside the open unit interval"
            )
        object.__setattr__(
            self,
            "canonical_successor_nll_nats",
            _require_nll(
                self.canonical_successor_nll_nats,
                field="evaluation.canonical_successor_nll_nats",
            ),
        )

    def as_payload(self) -> dict[str, object]:
        body: dict[str, object] = {
            "address": _address_payload(self.address),
            "family": self.family,
            "semantic_cell_id": self.semantic_cell_id,
            "cache_record_sha256": self.cache_record_sha256,
            "time_hex": self.time_hex,
            "canonical_successor_nll_nats": self.canonical_successor_nll_nats,
        }
        return {**body, "evaluation_sha256": _sha(body)}


@dataclass(frozen=True, slots=True)
class VerifiedSemanticP50ValidationEvaluationEnvironment:
    """Physically reopened, non-authorizing score-producer receipt."""

    path: Path
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path).resolve())
        if not isinstance(self.receipt, Mapping):
            raise TypeError("evaluation-environment receipt must be a mapping")


@dataclass(frozen=True, slots=True)
class SemanticP50BaselineMetric:
    """One exact-count, finite mean canonical-successor NLL."""

    example_count: int
    canonical_successor_nll_nats: float

    def __post_init__(self) -> None:
        _require_count(self.example_count, field="example_count", positive=True)
        object.__setattr__(
            self,
            "canonical_successor_nll_nats",
            _require_nll(
                self.canonical_successor_nll_nats,
                field="canonical_successor_nll_nats",
            ),
        )

    def as_payload(self) -> dict[str, object]:
        return {
            "example_count": self.example_count,
            "canonical_successor_nll_nats": self.canonical_successor_nll_nats,
        }


@dataclass(frozen=True, slots=True)
class VerifiedSemanticP50ValidationInventory:
    """A fully reopened physical validation inventory."""

    completion_path: Path
    completion: Mapping[str, Any]
    binding: SemanticP50ValidationBinding
    candidates: tuple[SemanticP50ValidationCandidate, ...]


@dataclass(frozen=True, slots=True)
class VerifiedSemanticP50ValidationBaseline:
    """A fully reopened physical pre-update validation baseline."""

    completion_path: Path
    completion: Mapping[str, Any]
    result: Mapping[str, Any]
    inventory: VerifiedSemanticP50ValidationInventory


def _inside(path: Path, *, root: Path, field: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(root):
        raise SemanticP50ValidationBaselineError(f"{field} is outside artifact_root")
    return resolved


def _candidate_contract_row(
    candidate: SemanticP50ValidationCandidate,
) -> dict[str, object]:
    body: dict[str, object] = {
        "address": _address_payload(candidate.address),
        "family": candidate.family,
        "semantic_cell_id": candidate.semantic_cell_id,
        "data_lane": candidate.data_lane,
        "assignment_sha256": candidate.assignment_sha256,
    }
    return {**body, "candidate_sha256": _sha(body)}


def _validated_successor_cache_projection(
    cache: SemanticP50SuccessorCache,
    *,
    candidates: Sequence[SemanticP50ValidationCandidate],
) -> tuple[
    dict[SuccessorFiberCacheAddress, SuccessorFiberCacheRecord],
    dict[str, str],
]:
    """Project one already strict cache onto the exact validation inventory."""

    if not isinstance(cache, SemanticP50SuccessorCache):
        raise TypeError(
            "validation baseline requires one strictly opened SemanticP50SuccessorCache"
        )
    completion = cache.completion
    manifest = cache.manifest
    contract = manifest.get("validation_contract")
    if not all(isinstance(value, Mapping) for value in (completion, manifest, contract)):
        raise SemanticP50ValidationBaselineError(
            "semantic P50 successor cache lacks completion, manifest, or validation contract"
        )
    assert isinstance(contract, Mapping)
    completion_body = dict(completion)
    completion_sha256 = completion_body.pop("completion_sha256", None)
    manifest_body = dict(manifest)
    manifest_sha256 = manifest_body.pop("manifest_sha256", None)
    contract_body = dict(contract)
    validation_contract_sha256 = contract_body.pop("validation_contract_sha256", None)
    candidate_rows = contract.get("candidate_rows")
    expected_rows = [_candidate_contract_row(item) for item in candidates]
    candidate_inventory_sha256 = _sha(expected_rows)
    if (
        completion_sha256 != _sha(completion_body)
        or manifest_sha256 != _sha(manifest_body)
        or validation_contract_sha256 != _sha(contract_body)
        or completion.get("manifest_sha256") != manifest_sha256
        or completion.get("validation_contract_sha256") != validation_contract_sha256
        or candidate_rows != expected_rows
        or contract.get("candidate_inventory_sha256") != candidate_inventory_sha256
        or contract.get("candidate_count") != len(expected_rows)
        or contract.get("partition_role") != VALIDATION_PARTITION_ROLE
        or tuple(contract.get("required_families", ())) != ACTIVE8_FAMILIES
        or contract.get("complete_required_coverage") is not True
    ):
        raise SemanticP50ValidationBaselineError(
            "semantic P50 cache completion or authenticated validation contract disagrees"
        )
    for value, field in (
        (completion_sha256, "cache.completion_sha256"),
        (manifest_sha256, "cache.manifest_sha256"),
        (validation_contract_sha256, "cache.validation_contract_sha256"),
        (candidate_inventory_sha256, "cache.validation_candidate_inventory_sha256"),
    ):
        _require_sha(value, field=field)

    records = cache.validation_requested_records
    if any(not isinstance(item, SuccessorFiberCacheRecord) for item in records):
        raise TypeError("semantic P50 validation cache records must use SuccessorFiberCacheRecord")
    by_address = {record.address: record for record in records}
    expected_by_address = {candidate.address: candidate for candidate in candidates}
    if (
        len(by_address) != len(records)
        or set(by_address) != set(expected_by_address)
        or _sha(sorted(_sha(_address_payload(item.address)) for item in records))
        != completion.get("validation_requested_address_set_sha256")
    ):
        raise SemanticP50ValidationBaselineError(
            "semantic P50 validation cache address set differs from the authenticated inventory"
        )
    for address, candidate in expected_by_address.items():
        record = by_address[address]
        try:
            indexed_record = cache.record_for_address(address)
        except KeyError as error:
            raise SemanticP50ValidationBaselineError(
                "semantic P50 validation cache address index is incomplete"
            ) from error
        record_sha256 = _sha(successor_fiber_cache_record_payload(record))
        indexed_record_sha256 = _sha(successor_fiber_cache_record_payload(indexed_record))
        if (
            record.teacher_fiber is None
            or record.source_key != candidate.source_key
            or record.source_state_sha256 != candidate.source_state_sha256
            or record.target_key != candidate.target_key
            or record.target_state_sha256 != candidate.target_state_sha256
            or indexed_record_sha256 != record_sha256
        ):
            raise SemanticP50ValidationBaselineError(
                "semantic P50 validation cache record differs from exact address/state evidence"
            )
    return by_address, {
        "completion_sha256": completion_sha256,
        "manifest_sha256": manifest_sha256,
        "validation_contract_sha256": validation_contract_sha256,
        "validation_candidate_inventory_sha256": candidate_inventory_sha256,
    }


def _validate_source_inventory(
    source: VerifiedSemanticP50SourceInventory,
    *,
    root: Path,
    expected: SemanticP50ValidationBinding,
) -> None:
    if not isinstance(source, VerifiedSemanticP50SourceInventory):
        raise TypeError("validation inventory requires a fully verified semantic P50 source")
    physical_path = _inside(source.path, root=root, field="source.path")
    payload, raw = _load_canonical(physical_path, field="semantic P50 source inventory")
    if (
        payload != source.index.identity_payload()
        or source.binding.source_inventory_file_sha256 != hashlib.sha256(raw).hexdigest()
        or source.binding.source_inventory_file_sha256 != expected.source_inventory_file_sha256
        or source.binding.source_inventory_sha256 != expected.source_inventory_sha256
        or source.binding.process_identity_sha256 != expected.process_identity_sha256
        or source.binding.model_runtime_identity_sha256 != expected.model_runtime_identity_sha256
        or source.binding.active8_policy_sha256 != expected.active8_policy_sha256
        or source.binding.operator_capability_fingerprint
        != expected.operator_capability_fingerprint
        or source.binding.decision_source_implementation_sha256
        != expected.decision_source_implementation_sha256
        or any(payload.get(name) is not value for name, value in _SOURCE_NO_AUTHORITY.items())
    ):
        raise SemanticP50ValidationBaselineError(
            "verified semantic source differs from its physical receipt or prepared binding"
        )


def _validate_scratch_runtime(
    scratch: SemanticScratchRuntime,
    *,
    expected: SemanticP50ValidationBinding,
) -> None:
    if not isinstance(scratch, SemanticScratchRuntime):
        raise TypeError("validation inventory requires the exact semantic scratch runtime")
    observed_state = state_dict_semantic_sha256(scratch.model.state_dict())
    if (
        observed_state != scratch.initial_model_state_sha256
        or observed_state != expected.scratch_initial_model_state_sha256
        or scratch.process_identity_sha256 != expected.process_identity_sha256
        or scratch.architecture.operator_capability_fingerprint
        != expected.operator_capability_fingerprint
    ):
        raise SemanticP50ValidationBaselineError(
            "physical scratch runtime differs from the prepared source/process binding"
        )


def _prepared_binding(
    *,
    prepared_recipe_path: Path,
    source: VerifiedSemanticP50SourceInventory,
    scratch_runtime: SemanticScratchRuntime,
    artifact_root: Path,
    recipe_policy_path: Path | None,
) -> tuple[SemanticP50ValidationBinding, dict[str, Any], tuple[str, ...]]:
    root = Path(artifact_root).resolve()
    prepared_recipe_path = _inside(prepared_recipe_path, root=root, field="prepared_recipe_path")
    prepared, prepared_raw = _load_canonical(
        prepared_recipe_path, field="semantic P50 prepared recipe"
    )
    body = dict(prepared)
    supplied = body.pop("prepared_recipe_sha256", None)
    prerequisites = prepared.get("prerequisites")
    validation = prepared.get("validation_contract")
    recipe = prepared.get("recipe")
    if not all(isinstance(item, Mapping) for item in (prerequisites, validation, recipe)):
        raise SemanticP50ValidationBaselineError(
            "semantic P50 prepared recipe lacks prerequisite or validation bindings"
        )
    assert isinstance(prerequisites, Mapping)
    assert isinstance(validation, Mapping)
    assert isinstance(recipe, Mapping)
    policy, policy_file_sha256 = load_semantic_p50_recipe_policy(recipe_policy_path)
    required_cells = tuple(validation.get("required_nonempty_semantic_cells", ()))
    if (
        prepared.get("schema") != PREPARED_SCHEMA
        or prepared.get("schema_version") != SCHEMA_VERSION
        or prepared.get("status") != PREPARED_STATUS
        or supplied != _sha(body)
        or not _authority_is_false(prepared)
        or prepared.get("recipe_policy_file_sha256") != policy_file_sha256
        or prepared.get("recipe_policy_sha256") != policy["policy_sha256"]
        or tuple(prepared.get("required_families", ())) != ACTIVE8_FAMILIES
        or validation.get("partition_role") != VALIDATION_PARTITION_ROLE
        or tuple(validation.get("required_families", ())) != ACTIVE8_FAMILIES
        or not required_cells
        or tuple(sorted(set(required_cells))) != required_cells
        or recipe.get("initialization") != "scratch_from_t1_bound_initial_model_state"
        or recipe.get("t1_selected_checkpoint_used_for_initialization") is not False
    ):
        raise SemanticP50ValidationBaselineError(
            "semantic P50 prepared recipe, policy, or validation contract disagrees"
        )
    for cell in required_cells:
        _require_text(cell, field="validation_contract.required_nonempty_semantic_cells")

    binding = SemanticP50ValidationBinding(
        process_identity_sha256=_require_sha(
            prerequisites.get("process_identity_sha256"),
            field="prerequisites.process_identity_sha256",
        ),
        source_inventory_file_sha256=_require_sha(
            prerequisites.get("source_inventory_file_sha256"),
            field="prerequisites.source_inventory_file_sha256",
        ),
        source_inventory_sha256=_require_sha(
            prerequisites.get("source_inventory_sha256"),
            field="prerequisites.source_inventory_sha256",
        ),
        model_runtime_identity_sha256=_require_sha(
            prerequisites.get("model_runtime_identity_sha256"),
            field="prerequisites.model_runtime_identity_sha256",
        ),
        active8_policy_sha256=_require_sha(
            prerequisites.get("active8_policy_sha256"),
            field="prerequisites.active8_policy_sha256",
        ),
        operator_capability_fingerprint=_require_text(
            prerequisites.get("operator_capability_fingerprint"),
            field="prerequisites.operator_capability_fingerprint",
        ),
        decision_source_implementation_sha256=_require_sha(
            prerequisites.get("decision_source_implementation_sha256"),
            field="prerequisites.decision_source_implementation_sha256",
        ),
        prepared_recipe_file_sha256=hashlib.sha256(prepared_raw).hexdigest(),
        prepared_recipe_sha256=_require_sha(supplied, field="prepared_recipe_sha256"),
        recipe_policy_file_sha256=policy_file_sha256,
        recipe_policy_sha256=_require_sha(
            policy.get("policy_sha256"), field="recipe_policy_sha256"
        ),
        scratch_initial_model_state_sha256=_require_sha(
            prerequisites.get("scratch_initial_model_state_sha256"),
            field="prerequisites.scratch_initial_model_state_sha256",
        ),
        capability_registry_sha256=_require_sha(
            prerequisites.get("capability_registry_sha256"),
            field="prerequisites.capability_registry_sha256",
        ),
        classifier_implementation_sha256=_require_sha(
            prerequisites.get("classifier_implementation_sha256"),
            field="prerequisites.classifier_implementation_sha256",
        ),
    )
    try:
        require_editing_v2_process_identity(binding.process_identity_sha256)
    except EditingV2ProcessIdentityError as error:
        raise SemanticP50ValidationBaselineError(
            "prepared semantic P50 recipe binds a stale Editing-V2 process"
        ) from error
    _validate_source_inventory(source, root=root, expected=binding)
    _validate_scratch_runtime(scratch_runtime, expected=binding)
    return binding, prepared, required_cells


def _candidates_from_source(
    source: VerifiedSemanticP50SourceInventory,
    *,
    registry: SemanticCapabilityCellRegistry,
) -> tuple[SemanticP50ValidationCandidate, ...]:
    """Reclassify every admitted validation transition from the verified source."""

    candidate_inputs: list[dict[str, Any]] = []
    for trace in source.index.iter_accepted_traces_for_partition(VALIDATION_PARTITION_ROLE):
        for transition in source.index.accepted_transitions_for(trace):
            assignment = classify_verified_structural_transition(
                source.index, transition, registry=registry
            )
            progress = transition.step_index
            path = transition.addressed_trace.path
            source_state = path.state_at(progress)
            target_state = path.state_at(progress + 1)
            candidate_inputs.append(
                {
                    "address": SuccessorFiberCacheAddress.from_packed_trace(
                        transition.addressed_trace.address,
                        progress_index=progress,
                    ),
                    "family": assignment.model_family,
                    "semantic_cell_id": assignment.capability_cell_id,
                    "data_lane": assignment.data_lane,
                    "assignment_sha256": assignment.assignment_sha256,
                    "source_key": canonical_state_key(source_state),
                    "source_state_sha256": persistent_slot_state_sha256(source_state),
                    "target_key": canonical_state_key(target_state),
                    "target_state_sha256": persistent_slot_state_sha256(target_state),
                }
            )
    candidate_inputs.sort(key=lambda item: item["address"])
    return tuple(
        SemanticP50ValidationCandidate(
            **item,
            time_hex=_validation_time_hex(stream_index=stream_index, address=item["address"]),
        )
        for stream_index, item in enumerate(candidate_inputs)
    )


def _inventory_rows(
    candidates: Sequence[SemanticP50ValidationCandidate],
    *,
    required_cells: Sequence[str],
) -> tuple[
    tuple[SemanticP50ValidationCandidate, ...],
    list[dict[str, Any]],
    dict[str, int],
    dict[str, int],
]:
    if not candidates:
        raise SemanticP50ValidationBaselineError(
            "semantic P50 validation candidate inventory is empty"
        )
    if any(not isinstance(item, SemanticP50ValidationCandidate) for item in candidates):
        raise TypeError("semantic P50 validation inventory requires typed validation candidates")
    ordered = tuple(sorted(candidates, key=lambda item: item.address))
    addresses = tuple(item.address for item in ordered)
    if len(set(addresses)) != len(addresses):
        raise SemanticP50ValidationBaselineError(
            "semantic P50 validation inventory contains duplicate addresses"
        )
    family_counts = Counter(item.family for item in ordered)
    cell_counts = Counter(item.semantic_cell_id for item in ordered)
    if set(family_counts) != set(ACTIVE8_FAMILIES):
        raise SemanticP50ValidationBaselineError(
            "semantic P50 validation inventory must cover every Active8 family"
        )
    missing_cells = set(required_cells).difference(cell_counts)
    if missing_cells:
        raise SemanticP50ValidationBaselineError(
            "semantic P50 validation inventory omits required semantic cells: "
            f"{sorted(missing_cells)}"
        )
    rows: list[dict[str, Any]] = []
    for stream_index, candidate in enumerate(ordered):
        if candidate.time_hex != _validation_time_hex(
            stream_index=stream_index, address=candidate.address
        ):
            raise SemanticP50ValidationBaselineError(
                "validation candidate time differs from the frozen address stream"
            )
        row_body = {
            "stream_index": stream_index,
            **candidate.as_payload(),
        }
        rows.append({**row_body, "row_sha256": _sha(row_body)})
    return (
        ordered,
        rows,
        {family: family_counts[family] for family in ACTIVE8_FAMILIES},
        {cell: cell_counts[cell] for cell in sorted(cell_counts)},
    )


def _ordered_time_stream_sha256(
    candidates: Sequence[SemanticP50ValidationCandidate],
) -> str:
    return _sha(
        [
            {
                "stream_index": index,
                "address": _address_payload(candidate.address),
                "time_hex": candidate.time_hex,
            }
            for index, candidate in enumerate(candidates)
        ]
    )


def _write_directory_atomically(
    *,
    target: Path,
    files: Mapping[str, bytes],
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".staging", dir=target.parent)
    )
    published = False
    try:
        for name, content in files.items():
            destination = staging / name
            with destination.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        try:
            os.replace(staging, target)
            published = True
        except OSError:
            if not target.is_dir():
                raise
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def materialize_semantic_p50_validation_inventory(
    *,
    prepared_recipe_path: Path,
    source: VerifiedSemanticP50SourceInventory,
    scratch_runtime: SemanticScratchRuntime,
    successor_cache: SemanticP50SuccessorCache,
    artifact_root: Path,
    output_root: Path,
    recipe_policy_path: Path | None = None,
    registry: SemanticCapabilityCellRegistry | None = None,
) -> Path:
    """Publish one deterministic validation-only candidate stream."""

    root = Path(artifact_root).resolve()
    output_root = _inside(output_root, root=root, field="output_root")
    selected_registry = registry or load_semantic_capability_cell_registry()
    if not isinstance(selected_registry, SemanticCapabilityCellRegistry):
        raise TypeError("registry must be a SemanticCapabilityCellRegistry")
    binding, _, required_cells = _prepared_binding(
        prepared_recipe_path=prepared_recipe_path,
        source=source,
        scratch_runtime=scratch_runtime,
        artifact_root=root,
        recipe_policy_path=recipe_policy_path,
    )
    if (
        selected_registry.registry_sha256 != binding.capability_registry_sha256
        or selected_registry.classifier_implementation_sha256
        != binding.classifier_implementation_sha256
        or selected_registry.process_identity_sha256 != binding.process_identity_sha256
    ):
        raise SemanticP50ValidationBaselineError(
            "validation classifier registry differs from the prepared recipe"
        )
    candidates = _candidates_from_source(source, registry=selected_registry)
    ordered, rows, family_counts, cell_counts = _inventory_rows(
        candidates, required_cells=required_cells
    )
    _, cache_binding = _validated_successor_cache_projection(successor_cache, candidates=ordered)
    rows_bytes = b"".join(_canonical_bytes(row, newline=True) for row in rows)
    candidate_inventory_sha256 = _sha([candidate.as_payload() for candidate in ordered])
    ordered_stream_sha256 = _sha(rows)
    ordered_time_stream_sha256 = _ordered_time_stream_sha256(ordered)
    run_identity_body: dict[str, Any] = {
        "schema": INVENTORY_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "binding": binding.as_payload(),
        "partition_role": VALIDATION_PARTITION_ROLE,
        "required_families": list(ACTIVE8_FAMILIES),
        "required_semantic_cells": list(required_cells),
        "candidate_count": len(rows),
        "candidate_inventory_sha256": candidate_inventory_sha256,
        "ordered_validation_stream_sha256": ordered_stream_sha256,
        "ordered_validation_time_stream_sha256": ordered_time_stream_sha256,
        "successor_cache_completion_sha256": cache_binding["completion_sha256"],
        "successor_cache_validation_contract_sha256": cache_binding["validation_contract_sha256"],
        "successor_cache_validation_candidate_inventory_sha256": cache_binding[
            "validation_candidate_inventory_sha256"
        ],
    }
    run_identity_sha256 = _sha(run_identity_body)
    completion_body: dict[str, Any] = {
        "schema": INVENTORY_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": INVENTORY_STATUS,
        **_NO_AUTHORITY,
        "run_identity_sha256": run_identity_sha256,
        "binding": binding.as_payload(),
        "partition_role": VALIDATION_PARTITION_ROLE,
        "required_families": list(ACTIVE8_FAMILIES),
        "required_semantic_cells": list(required_cells),
        "candidate_rows_relative_path": INVENTORY_ROWS_FILENAME,
        "candidate_rows_file_sha256": hashlib.sha256(rows_bytes).hexdigest(),
        "candidate_count": len(rows),
        "candidate_inventory_sha256": candidate_inventory_sha256,
        "ordered_validation_stream_sha256": ordered_stream_sha256,
        "ordered_validation_time_stream_sha256": ordered_time_stream_sha256,
        "family_counts": family_counts,
        "semantic_cell_counts": cell_counts,
        "successor_cache_completion_sha256": cache_binding["completion_sha256"],
        "successor_cache_validation_contract_sha256": cache_binding["validation_contract_sha256"],
        "successor_cache_validation_candidate_inventory_sha256": cache_binding[
            "validation_candidate_inventory_sha256"
        ],
    }
    completion = {
        **completion_body,
        "completion_sha256": _sha(completion_body),
    }
    target = output_root / run_identity_sha256
    _write_directory_atomically(
        target=target,
        files={
            INVENTORY_ROWS_FILENAME: rows_bytes,
            INVENTORY_COMPLETION_FILENAME: _canonical_bytes(completion, newline=True),
        },
    )
    completion_path = target / INVENTORY_COMPLETION_FILENAME
    verified = open_semantic_p50_validation_inventory(
        completion_path,
        prepared_recipe_path=prepared_recipe_path,
        source=source,
        scratch_runtime=scratch_runtime,
        successor_cache=successor_cache,
        artifact_root=root,
        recipe_policy_path=recipe_policy_path,
        registry=selected_registry,
    )
    if tuple(verified.candidates) != ordered:
        raise SemanticP50ValidationBaselineError(
            "published semantic P50 validation inventory differs from input"
        )
    return completion_path


def _load_inventory_rows(
    path: Path,
    *,
    max_row_bytes: int,
) -> tuple[tuple[SemanticP50ValidationCandidate, ...], list[dict[str, Any]], bytes]:
    _require_count(max_row_bytes, field="max_row_bytes", positive=True)
    rows: list[dict[str, Any]] = []
    raw_stream = bytearray()
    try:
        with Path(path).open("rb") as handle:
            while True:
                raw = handle.readline(max_row_bytes + 1)
                if not raw:
                    break
                index = len(rows)
                if len(raw) > max_row_bytes or not raw.endswith(b"\n"):
                    raise SemanticP50ValidationBaselineError(
                        f"validation candidate row {index} exceeds its bound or lacks newline"
                    )
                try:
                    row = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise SemanticP50ValidationBaselineError(
                        f"validation candidate row {index} is invalid JSON"
                    ) from error
                if not isinstance(row, dict) or raw != _canonical_bytes(row, newline=True):
                    raise SemanticP50ValidationBaselineError(
                        f"validation candidate row {index} is not canonical JSONL"
                    )
                rows.append(row)
                raw_stream.extend(raw)
    except OSError as error:
        raise SemanticP50ValidationBaselineError(
            f"cannot read validation candidate rows: {path}"
        ) from error

    candidates: list[SemanticP50ValidationCandidate] = []
    expected_fields = {
        "stream_index",
        "address",
        "family",
        "semantic_cell_id",
        "data_lane",
        "assignment_sha256",
        "source_key",
        "source_state_sha256",
        "target_key",
        "target_state_sha256",
        "time_hex",
        "row_sha256",
    }
    for index, row in enumerate(rows):
        body = dict(row)
        supplied = body.pop("row_sha256", None)
        address = body.get("address")
        try:
            candidate = SemanticP50ValidationCandidate(
                address=SuccessorFiberCacheAddress(**dict(address)),
                family=body.get("family"),
                semantic_cell_id=body.get("semantic_cell_id"),
                data_lane=body.get("data_lane"),
                assignment_sha256=body.get("assignment_sha256"),
                source_key=body.get("source_key"),
                source_state_sha256=body.get("source_state_sha256"),
                target_key=body.get("target_key"),
                target_state_sha256=body.get("target_state_sha256"),
                time_hex=body.get("time_hex"),
            )
        except (TypeError, ValueError) as error:
            raise SemanticP50ValidationBaselineError(
                f"validation candidate row {index} has an invalid address"
            ) from error
        if (
            set(row) != expected_fields
            or _require_count(body.get("stream_index"), field=f"candidate[{index}].stream_index")
            != index
            or supplied != _sha(body)
        ):
            raise SemanticP50ValidationBaselineError(
                f"validation candidate row {index} identity or order disagrees"
            )
        candidates.append(candidate)
    return tuple(candidates), rows, bytes(raw_stream)


def open_semantic_p50_validation_inventory(
    completion_path: Path,
    *,
    prepared_recipe_path: Path,
    source: VerifiedSemanticP50SourceInventory,
    scratch_runtime: SemanticScratchRuntime,
    successor_cache: SemanticP50SuccessorCache,
    artifact_root: Path,
    recipe_policy_path: Path | None = None,
    registry: SemanticCapabilityCellRegistry | None = None,
    max_row_bytes: int = 2 * 1024 * 1024,
) -> VerifiedSemanticP50ValidationInventory:
    """Strictly reopen a physical validation inventory and all bound identities."""

    root = Path(artifact_root).resolve()
    physical_path = _inside(completion_path, root=root, field="completion_path")
    if physical_path.name != INVENTORY_COMPLETION_FILENAME:
        raise SemanticP50ValidationBaselineError(
            f"validation inventory completion must name {INVENTORY_COMPLETION_FILENAME}"
        )
    completion, _ = _load_canonical(
        physical_path, field="semantic P50 validation inventory completion"
    )
    body = dict(completion)
    supplied = body.pop("completion_sha256", None)
    selected_registry = registry or load_semantic_capability_cell_registry()
    if not isinstance(selected_registry, SemanticCapabilityCellRegistry):
        raise TypeError("registry must be a SemanticCapabilityCellRegistry")
    binding, _, required_cells = _prepared_binding(
        prepared_recipe_path=prepared_recipe_path,
        source=source,
        scratch_runtime=scratch_runtime,
        artifact_root=root,
        recipe_policy_path=recipe_policy_path,
    )
    if (
        selected_registry.registry_sha256 != binding.capability_registry_sha256
        or selected_registry.classifier_implementation_sha256
        != binding.classifier_implementation_sha256
        or selected_registry.process_identity_sha256 != binding.process_identity_sha256
    ):
        raise SemanticP50ValidationBaselineError(
            "validation classifier registry differs from the prepared recipe"
        )
    if (
        set(completion) != _INVENTORY_COMPLETION_FIELDS
        or completion.get("schema") != INVENTORY_SCHEMA
        or completion.get("schema_version") != SCHEMA_VERSION
        or completion.get("status") != INVENTORY_STATUS
        or supplied != _sha(body)
        or not _authority_is_false(completion)
        or completion.get("binding") != binding.as_payload()
        or completion.get("partition_role") != VALIDATION_PARTITION_ROLE
        or tuple(completion.get("required_families", ())) != ACTIVE8_FAMILIES
        or tuple(completion.get("required_semantic_cells", ())) != required_cells
        or completion.get("candidate_rows_relative_path") != INVENTORY_ROWS_FILENAME
    ):
        raise SemanticP50ValidationBaselineError(
            "semantic P50 validation inventory completion identity disagrees"
        )
    run_identity = _require_sha(completion.get("run_identity_sha256"), field="run_identity_sha256")
    if physical_path.parent.name != run_identity:
        raise SemanticP50ValidationBaselineError(
            "semantic P50 validation inventory is not at its content-addressed path"
        )
    rows_path = physical_path.parent / INVENTORY_ROWS_FILENAME
    candidates, rows, rows_bytes = _load_inventory_rows(rows_path, max_row_bytes=max_row_bytes)
    exact_source_candidates = tuple(
        sorted(
            _candidates_from_source(source, registry=selected_registry),
            key=lambda item: item.address,
        )
    )
    ordered, expected_rows, family_counts, cell_counts = _inventory_rows(
        candidates, required_cells=required_cells
    )
    _, cache_binding = _validated_successor_cache_projection(successor_cache, candidates=ordered)
    completion_family_counts = _require_count_map(
        completion.get("family_counts"),
        field="completion.family_counts",
        expected_keys=ACTIVE8_FAMILIES,
    )
    completion_cell_counts = _require_count_map(
        completion.get("semantic_cell_counts"),
        field="completion.semantic_cell_counts",
        expected_keys=tuple(sorted(cell_counts)),
    )
    completion_candidate_count = _require_count(
        completion.get("candidate_count"),
        field="completion.candidate_count",
        positive=True,
    )
    candidate_inventory_sha256 = _sha([candidate.as_payload() for candidate in ordered])
    stream_sha256 = _sha(rows)
    time_stream_sha256 = _ordered_time_stream_sha256(ordered)
    run_body = {
        "schema": INVENTORY_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "binding": binding.as_payload(),
        "partition_role": VALIDATION_PARTITION_ROLE,
        "required_families": list(ACTIVE8_FAMILIES),
        "required_semantic_cells": list(required_cells),
        "candidate_count": len(rows),
        "candidate_inventory_sha256": candidate_inventory_sha256,
        "ordered_validation_stream_sha256": stream_sha256,
        "ordered_validation_time_stream_sha256": time_stream_sha256,
        "successor_cache_completion_sha256": cache_binding["completion_sha256"],
        "successor_cache_validation_contract_sha256": cache_binding["validation_contract_sha256"],
        "successor_cache_validation_candidate_inventory_sha256": cache_binding[
            "validation_candidate_inventory_sha256"
        ],
    }
    if (
        rows != expected_rows
        or ordered != exact_source_candidates
        or hashlib.sha256(rows_bytes).hexdigest() != completion.get("candidate_rows_file_sha256")
        or completion_candidate_count != len(rows)
        or completion.get("candidate_inventory_sha256") != candidate_inventory_sha256
        or completion.get("ordered_validation_stream_sha256") != stream_sha256
        or completion.get("ordered_validation_time_stream_sha256") != time_stream_sha256
        or completion_family_counts != family_counts
        or completion_cell_counts != cell_counts
        or completion.get("successor_cache_completion_sha256") != cache_binding["completion_sha256"]
        or completion.get("successor_cache_validation_contract_sha256")
        != cache_binding["validation_contract_sha256"]
        or completion.get("successor_cache_validation_candidate_inventory_sha256")
        != cache_binding["validation_candidate_inventory_sha256"]
        or run_identity != _sha(run_body)
    ):
        raise SemanticP50ValidationBaselineError(
            "semantic P50 validation rows, counts, or content address disagree"
        )
    return VerifiedSemanticP50ValidationInventory(
        completion_path=physical_path,
        completion=completion,
        binding=binding,
        candidates=ordered,
    )


def _metric_rows(
    metrics: Mapping[str, SemanticP50BaselineMetric],
    *,
    expected_counts: Mapping[str, int],
    identity_field: str,
) -> list[dict[str, object]]:
    if set(metrics) != set(expected_counts):
        raise SemanticP50ValidationBaselineError(
            f"baseline {identity_field} metrics differ from the exact inventory"
        )
    identities: Sequence[str]
    if identity_field == "family":
        identities = ACTIVE8_FAMILIES
    else:
        identities = tuple(sorted(expected_counts))
    rows: list[dict[str, object]] = []
    for identity in identities:
        metric = metrics[identity]
        if not isinstance(metric, SemanticP50BaselineMetric):
            raise TypeError("semantic P50 baseline metrics must use the typed schema")
        if metric.example_count != expected_counts[identity]:
            raise SemanticP50ValidationBaselineError(
                f"baseline {identity_field} {identity!r} count differs from inventory"
            )
        rows.append({identity_field: identity, **metric.as_payload()})
    return rows


def _parse_metric_rows(
    value: object,
    *,
    expected_counts: Mapping[str, int],
    identity_field: str,
) -> dict[str, SemanticP50BaselineMetric]:
    if not isinstance(value, list):
        raise SemanticP50ValidationBaselineError(
            f"baseline {identity_field} metrics must be a list"
        )
    metrics: dict[str, SemanticP50BaselineMetric] = {}
    for index, row in enumerate(value):
        if not isinstance(row, Mapping) or set(row) != {
            identity_field,
            "example_count",
            "canonical_successor_nll_nats",
        }:
            raise SemanticP50ValidationBaselineError(
                f"baseline {identity_field} metric {index} fields disagree"
            )
        identity = _require_text(row[identity_field], field=f"baseline metric {identity_field}")
        if identity in metrics:
            raise SemanticP50ValidationBaselineError(
                f"baseline repeats {identity_field} {identity!r}"
            )
        metrics[identity] = SemanticP50BaselineMetric(
            example_count=_require_count(
                row["example_count"],
                field="baseline metric example_count",
                positive=True,
            ),
            canonical_successor_nll_nats=_require_nll(
                row["canonical_successor_nll_nats"],
                field="baseline metric canonical_successor_nll_nats",
            ),
        )
    expected_rows = _metric_rows(
        metrics, expected_counts=expected_counts, identity_field=identity_field
    )
    if [dict(row) for row in value] != expected_rows:
        raise SemanticP50ValidationBaselineError(
            f"baseline {identity_field} metrics are not in deterministic order"
        )
    return metrics


def _evaluation_rows_and_metrics(
    evaluations: Sequence[SemanticP50ValidationEvaluation],
    *,
    inventory: VerifiedSemanticP50ValidationInventory,
    cache_by_address: Mapping[SuccessorFiberCacheAddress, SuccessorFiberCacheRecord],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    str,
]:
    if any(not isinstance(item, SemanticP50ValidationEvaluation) for item in evaluations):
        raise TypeError("baseline requires typed per-address validation evaluations")
    ordered = tuple(sorted(evaluations, key=lambda item: item.address))
    expected_by_address = {candidate.address: candidate for candidate in inventory.candidates}
    if (
        len(ordered) != len(expected_by_address)
        or len({item.address for item in ordered}) != len(ordered)
        or {item.address for item in ordered} != set(expected_by_address)
    ):
        raise SemanticP50ValidationBaselineError(
            "baseline evaluation addresses differ from the complete validation inventory"
        )
    family_values: dict[str, list[float]] = {family: [] for family in ACTIVE8_FAMILIES}
    cell_values: dict[str, list[float]] = {
        cell: [] for cell in inventory.completion["semantic_cell_counts"]
    }
    rows: list[dict[str, object]] = []
    for evaluation in ordered:
        candidate = expected_by_address[evaluation.address]
        cache_record_sha256 = _sha(
            successor_fiber_cache_record_payload(cache_by_address[evaluation.address])
        )
        if (
            evaluation.family != candidate.family
            or evaluation.semantic_cell_id != candidate.semantic_cell_id
            or evaluation.time_hex != candidate.time_hex
            or evaluation.cache_record_sha256 != cache_record_sha256
        ):
            raise SemanticP50ValidationBaselineError(
                "per-address evaluation differs from candidate classification or cache record"
            )
        family_values[candidate.family].append(evaluation.canonical_successor_nll_nats)
        cell_values[candidate.semantic_cell_id].append(evaluation.canonical_successor_nll_nats)
        rows.append(evaluation.as_payload())

    def metrics(values: Mapping[str, Sequence[float]], *, field: str) -> list[dict[str, object]]:
        identities = ACTIVE8_FAMILIES if field == "family" else tuple(sorted(values))
        result: list[dict[str, object]] = []
        for identity in identities:
            observed = values[identity]
            if not observed:
                raise SemanticP50ValidationBaselineError(
                    f"baseline {field} {identity!r} has no per-address evaluations"
                )
            metric = SemanticP50BaselineMetric(
                example_count=len(observed),
                canonical_successor_nll_nats=math.fsum(observed) / len(observed),
            )
            result.append({field: identity, **metric.as_payload()})
        return result

    address_set_sha256 = _sha([_address_payload(item.address) for item in ordered])
    return (
        rows,
        metrics(family_values, field="family"),
        metrics(cell_values, field="semantic_cell_id"),
        address_set_sha256,
    )


def _parse_evaluations(value: object) -> tuple[SemanticP50ValidationEvaluation, ...]:
    if not isinstance(value, list):
        raise SemanticP50ValidationBaselineError("baseline evaluation_records must be a list")
    result: list[SemanticP50ValidationEvaluation] = []
    expected_fields = {
        "address",
        "family",
        "semantic_cell_id",
        "cache_record_sha256",
        "time_hex",
        "canonical_successor_nll_nats",
        "evaluation_sha256",
    }
    for index, row in enumerate(value):
        if not isinstance(row, Mapping) or set(row) != expected_fields:
            raise SemanticP50ValidationBaselineError(
                f"baseline evaluation record {index} fields disagree"
            )
        body = dict(row)
        supplied = body.pop("evaluation_sha256")
        try:
            evaluation = SemanticP50ValidationEvaluation(
                address=SuccessorFiberCacheAddress(**dict(body["address"])),
                family=body["family"],
                semantic_cell_id=body["semantic_cell_id"],
                cache_record_sha256=body["cache_record_sha256"],
                time_hex=body["time_hex"],
                canonical_successor_nll_nats=body["canonical_successor_nll_nats"],
            )
        except (TypeError, ValueError) as error:
            raise SemanticP50ValidationBaselineError(
                f"baseline evaluation record {index} is invalid"
            ) from error
        if supplied != _sha(body) or dict(row) != evaluation.as_payload():
            raise SemanticP50ValidationBaselineError(
                f"baseline evaluation record {index} identity disagrees"
            )
        result.append(evaluation)
    return tuple(result)


def _validated_execution_environment(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _EXECUTION_ENVIRONMENT_FIELDS:
        raise SemanticP50ValidationBaselineError("evaluation execution environment fields disagree")
    environment = dict(value)
    body = dict(environment)
    supplied = body.pop("environment_sha256")
    required_text = (
        "hardware_class",
        "device_name",
        "accelerator_class",
        "dtype",
        "python_version",
        "torch_version",
        "rdkit_version",
        "numpy_version",
    )
    if (
        any(
            not isinstance(environment[name], str) or not environment[name]
            for name in required_text
        )
        or not isinstance(environment["mixed_precision"], bool)
        or type(environment["batch_size"]) is not int
        or environment["batch_size"] <= 0
        or supplied != _sha(body)
    ):
        raise SemanticP50ValidationBaselineError(
            "evaluation execution environment is incomplete or not self-authenticating"
        )
    optional_gpu_text = ("device_capability", "cuda_version", "cudnn_version")
    if environment["accelerator_class"] == "gpu":
        if any(
            not isinstance(environment[name], str) or not environment[name]
            for name in optional_gpu_text
        ):
            raise SemanticP50ValidationBaselineError(
                "GPU evaluation environment lacks capability or CUDA identities"
            )
    elif any(environment[name] is not None for name in optional_gpu_text):
        raise SemanticP50ValidationBaselineError(
            "non-GPU evaluation environment must null GPU-only identities"
        )
    return environment


def build_semantic_p50_validation_evaluation_environment_receipt(
    *,
    repo_root: Path,
    execution_environment: Mapping[str, Any],
    scratch_runtime: SemanticScratchRuntime,
    successor_cache: SemanticP50SuccessorCache,
    evaluations: Sequence[SemanticP50ValidationEvaluation],
) -> dict[str, Any]:
    """Build a non-authorizing receipt for a future physical score producer."""

    environment = _validated_execution_environment(execution_environment)
    cache_contract = successor_cache.manifest.get("validation_contract")
    if not isinstance(cache_contract, Mapping):
        raise SemanticP50ValidationBaselineError(
            "evaluation receipt requires an authenticated cache validation contract"
        )
    evaluator_sha256 = semantic_p50_validation_evaluator_implementation_sha256(repo_root=repo_root)
    ordered = tuple(sorted(evaluations, key=lambda item: item.address))
    evaluation_rows = [item.as_payload() for item in ordered]
    body: dict[str, Any] = {
        "schema": EVALUATION_ENVIRONMENT_RECEIPT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": EVALUATION_ENVIRONMENT_RECEIPT_STATUS,
        **_NO_AUTHORITY,
        "producer_implementation_sha256": evaluator_sha256,
        "execution_environment": environment,
        "execution_environment_sha256": environment["environment_sha256"],
        "evaluated_model_state_sha256": scratch_runtime.initial_model_state_sha256,
        "successor_cache_completion_sha256": successor_cache.completion.get("completion_sha256"),
        "successor_cache_manifest_sha256": successor_cache.manifest.get("manifest_sha256"),
        "successor_cache_validation_contract_sha256": cache_contract.get(
            "validation_contract_sha256"
        ),
        "successor_cache_validation_candidate_inventory_sha256": cache_contract.get(
            "candidate_inventory_sha256"
        ),
        "objective": dict(_BASELINE_OBJECTIVE),
        "evaluation_address_set_sha256": _sha([_address_payload(item.address) for item in ordered]),
        "evaluation_records_sha256": _sha(evaluation_rows),
    }
    for field in (
        "producer_implementation_sha256",
        "execution_environment_sha256",
        "evaluated_model_state_sha256",
        "successor_cache_completion_sha256",
        "successor_cache_manifest_sha256",
        "successor_cache_validation_contract_sha256",
        "successor_cache_validation_candidate_inventory_sha256",
        "evaluation_address_set_sha256",
        "evaluation_records_sha256",
    ):
        _require_sha(body[field], field=f"evaluation_receipt.{field}")
    return {**body, "receipt_sha256": _sha(body)}


def open_semantic_p50_validation_evaluation_environment_receipt(
    path: Path,
    *,
    artifact_root: Path,
    repo_root: Path,
    scratch_runtime: SemanticScratchRuntime,
    successor_cache: SemanticP50SuccessorCache,
    evaluations: Sequence[SemanticP50ValidationEvaluation],
) -> VerifiedSemanticP50ValidationEvaluationEnvironment:
    """Reopen and independently recompute one score-producer receipt."""

    root = Path(artifact_root).resolve()
    physical_path = _inside(path, root=root, field="evaluation_environment_receipt")
    receipt, _ = _load_canonical(physical_path, field="semantic P50 evaluation-environment receipt")
    body = dict(receipt)
    supplied = body.pop("receipt_sha256", None)
    expected = build_semantic_p50_validation_evaluation_environment_receipt(
        repo_root=repo_root,
        execution_environment=receipt.get("execution_environment", {}),
        scratch_runtime=scratch_runtime,
        successor_cache=successor_cache,
        evaluations=evaluations,
    )
    if (
        set(receipt) != _EVALUATION_ENVIRONMENT_RECEIPT_FIELDS
        or supplied != _sha(body)
        or receipt != expected
        or not _authority_is_false(receipt)
    ):
        raise SemanticP50ValidationBaselineError(
            "physical evaluation-environment receipt differs from producer, runtime, "
            "model, cache, or score identities"
        )
    return VerifiedSemanticP50ValidationEvaluationEnvironment(
        path=physical_path,
        receipt=receipt,
    )


def materialize_semantic_p50_validation_baseline(
    *,
    inventory_completion_path: Path,
    prepared_recipe_path: Path,
    source: VerifiedSemanticP50SourceInventory,
    scratch_runtime: SemanticScratchRuntime,
    artifact_root: Path,
    evaluations: Sequence[SemanticP50ValidationEvaluation],
    successor_cache: SemanticP50SuccessorCache,
    evaluation_environment_receipt: VerifiedSemanticP50ValidationEvaluationEnvironment,
    repo_root: Path,
    output_root: Path,
    recipe_policy_path: Path | None = None,
    registry: SemanticCapabilityCellRegistry | None = None,
) -> Path:
    """Seal exact per-address scratch evaluations and recomputed aggregates."""

    root = Path(artifact_root).resolve()
    output_root = _inside(output_root, root=root, field="output_root")
    inventory = open_semantic_p50_validation_inventory(
        inventory_completion_path,
        prepared_recipe_path=prepared_recipe_path,
        source=source,
        scratch_runtime=scratch_runtime,
        successor_cache=successor_cache,
        artifact_root=root,
        recipe_policy_path=recipe_policy_path,
        registry=registry,
    )
    _validate_scratch_runtime(scratch_runtime, expected=inventory.binding)
    model_state_sha256 = scratch_runtime.initial_model_state_sha256
    cache_by_address, cache_binding = _validated_successor_cache_projection(
        successor_cache, candidates=inventory.candidates
    )
    evaluation_rows, family_rows, cell_rows, address_set_sha256 = _evaluation_rows_and_metrics(
        evaluations,
        inventory=inventory,
        cache_by_address=cache_by_address,
    )
    if not isinstance(
        evaluation_environment_receipt,
        VerifiedSemanticP50ValidationEvaluationEnvironment,
    ):
        raise TypeError("baseline requires a physically reopened evaluation-environment receipt")
    verified_environment = open_semantic_p50_validation_evaluation_environment_receipt(
        evaluation_environment_receipt.path,
        artifact_root=root,
        repo_root=repo_root,
        scratch_runtime=scratch_runtime,
        successor_cache=successor_cache,
        evaluations=evaluations,
    )
    if verified_environment.receipt != evaluation_environment_receipt.receipt:
        raise SemanticP50ValidationBaselineError(
            "typed evaluation-environment receipt differs from its physical artifact"
        )
    evaluator_implementation_sha256 = semantic_p50_validation_evaluator_implementation_sha256(
        repo_root=repo_root
    )
    inventory_completion_file_sha256 = _file_sha(inventory.completion_path)
    result_body: dict[str, Any] = {
        "schema": BASELINE_RESULT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": BASELINE_STATUS,
        **_NO_AUTHORITY,
        "binding": inventory.binding.as_payload(),
        "validation_inventory_completion_file_sha256": (inventory_completion_file_sha256),
        "validation_inventory_completion_sha256": inventory.completion["completion_sha256"],
        "ordered_validation_stream_sha256": inventory.completion[
            "ordered_validation_stream_sha256"
        ],
        "ordered_validation_time_stream_sha256": inventory.completion[
            "ordered_validation_time_stream_sha256"
        ],
        "objective": dict(_BASELINE_OBJECTIVE),
        "evaluated_model_state_sha256": model_state_sha256,
        "successor_cache_completion_sha256": cache_binding["completion_sha256"],
        "successor_cache_manifest_sha256": cache_binding["manifest_sha256"],
        "successor_cache_validation_contract_sha256": cache_binding["validation_contract_sha256"],
        "successor_cache_validation_candidate_inventory_sha256": cache_binding[
            "validation_candidate_inventory_sha256"
        ],
        "evaluator_implementation_sha256": evaluator_implementation_sha256,
        "evaluation_environment_receipt_file_sha256": _file_sha(verified_environment.path),
        "evaluation_environment_receipt_sha256": verified_environment.receipt["receipt_sha256"],
        "execution_environment": verified_environment.receipt["execution_environment"],
        "example_count": len(inventory.candidates),
        "address_set_sha256": address_set_sha256,
        "evaluation_records": evaluation_rows,
        "family_metrics": family_rows,
        "semantic_cell_metrics": cell_rows,
    }
    result = {**result_body, "result_sha256": _sha(result_body)}
    result_bytes = _canonical_bytes(result, newline=True)
    completion_body: dict[str, Any] = {
        "schema": BASELINE_COMPLETION_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": BASELINE_STATUS,
        **_NO_AUTHORITY,
        "result_relative_path": BASELINE_RESULT_FILENAME,
        "result_file_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "result_sha256": result["result_sha256"],
        "validation_inventory_completion_file_sha256": (inventory_completion_file_sha256),
        "validation_inventory_completion_sha256": inventory.completion["completion_sha256"],
        "successor_cache_completion_sha256": cache_binding["completion_sha256"],
        "successor_cache_validation_contract_sha256": cache_binding["validation_contract_sha256"],
        "evaluator_implementation_sha256": evaluator_implementation_sha256,
        "evaluation_environment_receipt_sha256": verified_environment.receipt["receipt_sha256"],
        "binding": inventory.binding.as_payload(),
    }
    completion = {
        **completion_body,
        "completion_sha256": _sha(completion_body),
    }
    target = output_root / result["result_sha256"]
    _write_directory_atomically(
        target=target,
        files={
            BASELINE_RESULT_FILENAME: result_bytes,
            BASELINE_COMPLETION_FILENAME: _canonical_bytes(completion, newline=True),
        },
    )
    completion_path = target / BASELINE_COMPLETION_FILENAME
    verified = open_semantic_p50_validation_baseline(
        completion_path,
        inventory_completion_path=inventory_completion_path,
        prepared_recipe_path=prepared_recipe_path,
        source=source,
        scratch_runtime=scratch_runtime,
        successor_cache=successor_cache,
        artifact_root=root,
        evaluation_environment_receipt=evaluation_environment_receipt,
        repo_root=repo_root,
        recipe_policy_path=recipe_policy_path,
        registry=registry,
    )
    if verified.result != result:
        raise SemanticP50ValidationBaselineError(
            "published semantic P50 validation baseline differs from input"
        )
    return completion_path


def open_semantic_p50_validation_baseline(
    completion_path: Path,
    *,
    inventory_completion_path: Path,
    prepared_recipe_path: Path,
    source: VerifiedSemanticP50SourceInventory,
    scratch_runtime: SemanticScratchRuntime,
    artifact_root: Path,
    successor_cache: SemanticP50SuccessorCache,
    evaluation_environment_receipt: VerifiedSemanticP50ValidationEvaluationEnvironment,
    repo_root: Path,
    recipe_policy_path: Path | None = None,
    registry: SemanticCapabilityCellRegistry | None = None,
) -> VerifiedSemanticP50ValidationBaseline:
    """Strictly reopen a baseline result, completion, inventory, and bindings."""

    root = Path(artifact_root).resolve()
    physical_path = _inside(completion_path, root=root, field="completion_path")
    if physical_path.name != BASELINE_COMPLETION_FILENAME:
        raise SemanticP50ValidationBaselineError(
            f"validation baseline completion must name {BASELINE_COMPLETION_FILENAME}"
        )
    completion, _ = _load_canonical(
        physical_path, field="semantic P50 validation baseline completion"
    )
    completion_body = dict(completion)
    completion_sha256 = completion_body.pop("completion_sha256", None)
    inventory = open_semantic_p50_validation_inventory(
        inventory_completion_path,
        prepared_recipe_path=prepared_recipe_path,
        source=source,
        scratch_runtime=scratch_runtime,
        successor_cache=successor_cache,
        artifact_root=root,
        recipe_policy_path=recipe_policy_path,
        registry=registry,
    )
    cache_by_address, cache_binding = _validated_successor_cache_projection(
        successor_cache, candidates=inventory.candidates
    )
    if not isinstance(
        evaluation_environment_receipt,
        VerifiedSemanticP50ValidationEvaluationEnvironment,
    ):
        raise TypeError("baseline requires a physically reopened evaluation-environment receipt")
    evaluator_implementation_sha256 = semantic_p50_validation_evaluator_implementation_sha256(
        repo_root=repo_root
    )
    receipt_path = _inside(
        evaluation_environment_receipt.path,
        root=root,
        field="evaluation_environment_receipt.path",
    )
    inventory_file_sha256 = _file_sha(inventory.completion_path)
    if (
        set(completion) != _BASELINE_COMPLETION_FIELDS
        or completion.get("schema") != BASELINE_COMPLETION_SCHEMA
        or completion.get("schema_version") != SCHEMA_VERSION
        or completion.get("status") != BASELINE_STATUS
        or completion_sha256 != _sha(completion_body)
        or not _authority_is_false(completion)
        or completion.get("result_relative_path") != BASELINE_RESULT_FILENAME
        or completion.get("validation_inventory_completion_file_sha256") != inventory_file_sha256
        or completion.get("validation_inventory_completion_sha256")
        != inventory.completion["completion_sha256"]
        or completion.get("successor_cache_completion_sha256") != cache_binding["completion_sha256"]
        or completion.get("successor_cache_validation_contract_sha256")
        != cache_binding["validation_contract_sha256"]
        or completion.get("evaluator_implementation_sha256") != evaluator_implementation_sha256
        or completion.get("evaluation_environment_receipt_sha256")
        != evaluation_environment_receipt.receipt.get("receipt_sha256")
        or completion.get("binding") != inventory.binding.as_payload()
    ):
        raise SemanticP50ValidationBaselineError(
            "semantic P50 validation baseline completion identity disagrees"
        )
    result_path = physical_path.parent / BASELINE_RESULT_FILENAME
    result, result_raw = _load_canonical(
        result_path, field="semantic P50 validation baseline result"
    )
    result_body = dict(result)
    result_sha256 = result_body.pop("result_sha256", None)
    if (
        set(result) != _BASELINE_RESULT_FIELDS
        or result.get("schema") != BASELINE_RESULT_SCHEMA
        or result.get("schema_version") != SCHEMA_VERSION
        or result.get("status") != BASELINE_STATUS
        or result_sha256 != _sha(result_body)
        or physical_path.parent.name != result_sha256
        or not _authority_is_false(result)
        or hashlib.sha256(result_raw).hexdigest() != completion.get("result_file_sha256")
        or result_sha256 != completion.get("result_sha256")
        or result.get("binding") != inventory.binding.as_payload()
        or result.get("validation_inventory_completion_file_sha256") != inventory_file_sha256
        or result.get("validation_inventory_completion_sha256")
        != inventory.completion["completion_sha256"]
        or result.get("ordered_validation_stream_sha256")
        != inventory.completion["ordered_validation_stream_sha256"]
        or result.get("ordered_validation_time_stream_sha256")
        != inventory.completion["ordered_validation_time_stream_sha256"]
        or result.get("objective") != _BASELINE_OBJECTIVE
        or result.get("evaluated_model_state_sha256")
        != inventory.binding.scratch_initial_model_state_sha256
        or _require_count(result.get("example_count"), field="result.example_count", positive=True)
        != len(inventory.candidates)
        or result.get("successor_cache_completion_sha256") != cache_binding["completion_sha256"]
        or result.get("successor_cache_manifest_sha256") != cache_binding["manifest_sha256"]
        or result.get("successor_cache_validation_contract_sha256")
        != cache_binding["validation_contract_sha256"]
        or result.get("successor_cache_validation_candidate_inventory_sha256")
        != cache_binding["validation_candidate_inventory_sha256"]
        or result.get("evaluator_implementation_sha256") != evaluator_implementation_sha256
        or result.get("evaluation_environment_receipt_file_sha256") != _file_sha(receipt_path)
        or result.get("evaluation_environment_receipt_sha256")
        != evaluation_environment_receipt.receipt.get("receipt_sha256")
        or result.get("execution_environment")
        != evaluation_environment_receipt.receipt.get("execution_environment")
    ):
        raise SemanticP50ValidationBaselineError(
            "semantic P50 validation baseline result identity disagrees"
        )
    parsed_evaluations = _parse_evaluations(result.get("evaluation_records"))
    reopened_environment = open_semantic_p50_validation_evaluation_environment_receipt(
        receipt_path,
        artifact_root=root,
        repo_root=repo_root,
        scratch_runtime=scratch_runtime,
        successor_cache=successor_cache,
        evaluations=parsed_evaluations,
    )
    if reopened_environment.receipt != evaluation_environment_receipt.receipt:
        raise SemanticP50ValidationBaselineError(
            "typed evaluation-environment receipt differs from its physical artifact"
        )
    evaluation_rows, family_rows, cell_rows, address_set_sha256 = _evaluation_rows_and_metrics(
        parsed_evaluations,
        inventory=inventory,
        cache_by_address=cache_by_address,
    )
    if (
        result.get("evaluation_records") != evaluation_rows
        or result.get("family_metrics") != family_rows
        or result.get("semantic_cell_metrics") != cell_rows
        or result.get("address_set_sha256") != address_set_sha256
    ):
        raise SemanticP50ValidationBaselineError(
            "baseline per-address receipts or recomputed aggregates disagree"
        )
    _parse_metric_rows(
        result.get("family_metrics"),
        expected_counts=dict(inventory.completion["family_counts"]),
        identity_field="family",
    )
    _parse_metric_rows(
        result.get("semantic_cell_metrics"),
        expected_counts=dict(inventory.completion["semantic_cell_counts"]),
        identity_field="semantic_cell_id",
    )
    return VerifiedSemanticP50ValidationBaseline(
        completion_path=physical_path,
        completion=completion,
        result=result,
        inventory=inventory,
    )


def semantic_p50_validation_evaluations_from_verified_baseline(
    baseline: VerifiedSemanticP50ValidationBaseline,
) -> tuple[SemanticP50ValidationEvaluation, ...]:
    """Return the exact per-address evaluations from a verified baseline."""

    if not isinstance(baseline, VerifiedSemanticP50ValidationBaseline):
        raise TypeError("baseline must be a verified semantic P50 validation baseline")
    return _parse_evaluations(baseline.result.get("evaluation_records"))


def open_semantic_p50_validation_baseline_from_paths(
    completion_path: Path,
    *,
    inventory_completion_path: Path,
    evaluation_environment_receipt_path: Path,
    prepared_recipe_path: Path,
    source: VerifiedSemanticP50SourceInventory,
    scratch_runtime: SemanticScratchRuntime,
    artifact_root: Path,
    successor_cache: SemanticP50SuccessorCache,
    repo_root: Path,
    recipe_policy_path: Path | None = None,
    registry: SemanticCapabilityCellRegistry | None = None,
) -> VerifiedSemanticP50ValidationBaseline:
    """Strictly reopen a baseline and its typed environment receipt from paths.

    The preliminary result read is used only to recover the evaluation identities
    needed to authenticate the environment receipt.  The ordinary strict opener
    then reopens and cross-validates every baseline, inventory, cache, runtime,
    implementation, and receipt binding before returning a typed artifact.
    """

    root = Path(artifact_root).resolve()
    physical_path = _inside(completion_path, root=root, field="completion_path")
    if physical_path.name != BASELINE_COMPLETION_FILENAME:
        raise SemanticP50ValidationBaselineError(
            f"validation baseline completion must name {BASELINE_COMPLETION_FILENAME}"
        )
    completion, _ = _load_canonical(
        physical_path, field="semantic P50 validation baseline completion"
    )
    if completion.get("result_relative_path") != BASELINE_RESULT_FILENAME:
        raise SemanticP50ValidationBaselineError(
            "semantic P50 validation baseline result path disagrees"
        )
    result, _ = _load_canonical(
        physical_path.parent / BASELINE_RESULT_FILENAME,
        field="semantic P50 validation baseline result",
    )
    evaluations = _parse_evaluations(result.get("evaluation_records"))
    environment = open_semantic_p50_validation_evaluation_environment_receipt(
        evaluation_environment_receipt_path,
        artifact_root=root,
        repo_root=repo_root,
        scratch_runtime=scratch_runtime,
        successor_cache=successor_cache,
        evaluations=evaluations,
    )
    return open_semantic_p50_validation_baseline(
        physical_path,
        inventory_completion_path=inventory_completion_path,
        prepared_recipe_path=prepared_recipe_path,
        source=source,
        scratch_runtime=scratch_runtime,
        artifact_root=root,
        successor_cache=successor_cache,
        evaluation_environment_receipt=environment,
        repo_root=repo_root,
        recipe_policy_path=recipe_policy_path,
        registry=registry,
    )


__all__ = [
    "BASELINE_COMPLETION_FILENAME",
    "BASELINE_RESULT_FILENAME",
    "EVALUATION_ENVIRONMENT_RECEIPT_SCHEMA",
    "INVENTORY_COMPLETION_FILENAME",
    "INVENTORY_ROWS_FILENAME",
    "SemanticP50BaselineMetric",
    "SemanticP50ValidationBaselineError",
    "SemanticP50ValidationBinding",
    "SemanticP50ValidationCandidate",
    "SemanticP50ValidationEvaluation",
    "VerifiedSemanticP50ValidationBaseline",
    "VerifiedSemanticP50ValidationEvaluationEnvironment",
    "VerifiedSemanticP50ValidationInventory",
    "build_semantic_p50_validation_evaluation_environment_receipt",
    "materialize_semantic_p50_validation_baseline",
    "materialize_semantic_p50_validation_inventory",
    "open_semantic_p50_validation_baseline",
    "open_semantic_p50_validation_baseline_from_paths",
    "open_semantic_p50_validation_evaluation_environment_receipt",
    "open_semantic_p50_validation_inventory",
    "semantic_p50_validation_evaluations_from_verified_baseline",
    "semantic_p50_validation_evaluator_implementation_sha256",
]
