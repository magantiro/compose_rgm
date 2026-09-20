"""Pure, fixed-budget semantic Editing-V2 P50 training runtime.

This module performs no artifact publication and exposes no launch or selection
authority.  It consumes an already reopened combined train/validation
successor cache, exact persistent-slot source states, and the T1-bound scratch
runtime.  The only permitted optimization is the frozen 50 by 64 productive
canonical-successor schedule.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import shutil
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from compose_v4.chem.persistent_state_identity import persistent_slot_state_sha256
from compose_v4.data.editing_v2_semantic_capability_cells import ACTIVE8_FAMILIES
from compose_v4.data.successor_fiber_cache import (
    SuccessorFiberCacheAddress,
    SuccessorFiberCacheRecord,
    successor_fiber_cache_record_payload,
)
from compose_v4.experiments.editing_p50_gate import state_dict_semantic_sha256
from compose_v4.experiments.editing_v2_semantic_p50_recipe_stream import (
    BATCH_SIZE,
    MAXIMUM_CELL_P50_NONINCREASE_NATS,
    MAXIMUM_FAMILY_P50_NONINCREASE_NATS,
    OPTIMIZER_STEPS,
    SCHEDULED_EXAMPLES,
    SEED,
    TIME_ALGORITHM,
    semantic_p50_time_hex,
)
from compose_v4.experiments.editing_v2_semantic_p50_successor_cache import (
    SemanticP50SuccessorCache,
    SemanticP50SuccessorCacheError,
    validate_semantic_p50_prepared_recipe,
    validate_semantic_p50_successor_cache_completion,
)
from compose_v4.experiments.editing_v2_semantic_p50_validation_baseline import (
    SemanticP50ValidationEvaluation,
    VerifiedSemanticP50ValidationBaseline,
    semantic_p50_validation_evaluations_from_verified_baseline,
)
from compose_v4.experiments.editing_v2_semantic_runtime import SemanticScratchRuntime
from compose_v4.experiments.editing_v2_semantic_t1_capacity_runner import (
    ACTION_ROUTE_PREFIXES,
    FAMILY_ROUTE_PREFIXES,
    TOTAL_HAZARD_PREFIX,
    SemanticT1CapacityRunnerError,
    _assert_hazard_identity,
    _clone_state_dict,
    _collator,
    _hazard_state,
    _optimizer_configuration,
    _rng_state,
    optimizer_state_semantic_sha256,
)
from compose_v4.experiments.factorized_mark_conditional import FactorizedMarkExample
from compose_v4.experiments.factorized_successor_training import (
    SuccessorTrainingError,
    factorized_successor_identity_loss,
    forward_teacher_successor_batch,
)

SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA = "compose.editing_v2.semantic_p50_checkpoint"
RESULT_SCHEMA = "compose.editing_v2.semantic_p50_training_result"
CHECKPOINT_STATUS = "TERMINAL_P50_STATE_NO_SELECTION_OR_DOWNSTREAM_AUTHORITY"
RESULT_STATUS = "COMPLETED_FIXED_50_STEP_PILOT_NO_DOWNSTREAM_AUTHORITY"
VALIDATION_TIME_STREAM_SCHEMA = "compose.editing_v2.semantic_p50_validation_time_stream"
RUN_COMPLETION_SCHEMA = "compose.editing_v2.semantic_p50_run_completion"
RUN_COMPLETION_STATUS = "COMPLETE_ATOMIC_P50_RUN_NO_SELECTION_OR_DOWNSTREAM_AUTHORITY"
RUN_RESULT_FILENAME = "semantic_p50_result.json"
RUN_CHECKPOINT_FILENAME = "semantic_p50_checkpoint.pt"
RUN_COMPLETION_FILENAME = "SEMANTIC_P50_RUN_COMPLETE.json"

NO_AUTHORITY = {
    "training_authorized": False,
    "bounded_p50_authorized": False,
    "p500_authorized": False,
    "long_training_authorized": False,
    "checkpoint_selection_authorized": False,
    "final_test_selection_authorized": False,
}

_HEX = frozenset("0123456789abcdef")
_EXPECTED_RECIPE = {
    "initialization": "scratch_from_t1_bound_initial_model_state",
    "optimizer_steps": OPTIMIZER_STEPS,
    "batch_size": BATCH_SIZE,
    "scheduled_nonterminal_examples": SCHEDULED_EXAMPLES,
    "optimizer": "adamw",
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "scheduler": "constant",
    "gradient_clip_norm": 10.0,
    "seed": SEED,
    "dtype": "float32",
    "mixed_precision": False,
    "deterministic_algorithms_required": True,
    "resume": False,
    "t1_selected_checkpoint_used_for_initialization": False,
}
_STREAM_ROW_FIELDS = {
    "stream_index",
    "optimizer_step",
    "batch_offset",
    "address",
    "family",
    "semantic_cell_id",
    "data_lane",
    "source_group_id",
    "time_hex",
    "identity_coefficient",
    "importance_correction",
    "path_position_coefficient",
    "row_sha256",
}
_IMPLEMENTATION_SOURCES = (
    "src/compose_v4/experiments/editing_v2_semantic_p50_runner.py",
    "src/compose_v4/experiments/editing_v2_semantic_p50_recipe_stream.py",
    "src/compose_v4/experiments/editing_v2_semantic_p50_successor_cache.py",
    "src/compose_v4/experiments/editing_v2_semantic_p50_validation_baseline.py",
    "src/compose_v4/experiments/editing_v2_semantic_runtime.py",
    "src/compose_v4/experiments/editing_v2_semantic_t1_capacity_runner.py",
    "src/compose_v4/experiments/factorized_mark_conditional.py",
    "src/compose_v4/experiments/factorized_successor_training.py",
    "src/compose_v4/data/editing_v2_semantic_capability_cells.py",
    "src/compose_v4/data/successor_fiber_cache.py",
    "src/compose_v4/model/factorized_tracelet_rate_model.py",
    "src/compose_v4/model/relational_reroute_rate_model.py",
    "src/compose_v4/chem/persistent_state_identity.py",
)


class SemanticP50RunnerError(RuntimeError):
    """The fixed P50 runtime or one of its scientific identities failed closed."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SemanticP50RunnerError(
            "semantic P50 metadata is not finite canonical JSON"
        ) from error


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise SemanticP50RunnerError(f"{field} must be a lowercase SHA-256")
    return value


def _address_payload(address: SuccessorFiberCacheAddress) -> dict[str, object]:
    return asdict(address)


def _address_sha256(address: SuccessorFiberCacheAddress) -> str:
    return _sha(_address_payload(address))


def _parse_address(value: object, *, field: str) -> SuccessorFiberCacheAddress:
    if not isinstance(value, Mapping):
        raise SemanticP50RunnerError(f"{field} is not an address object")
    try:
        address = SuccessorFiberCacheAddress(**dict(value))
    except (TypeError, ValueError) as error:
        raise SemanticP50RunnerError(f"{field} is not a valid exact address") from error
    if _address_payload(address) != dict(value):
        raise SemanticP50RunnerError(f"{field} does not round-trip exactly")
    return address


def semantic_p50_validation_time_hex(
    *, stream_index: int, address: SuccessorFiberCacheAddress
) -> str:
    """Derive the frozen open-unit validation time for one ordered address."""

    if type(stream_index) is not int or stream_index < 0:
        raise SemanticP50RunnerError("validation stream_index must be nonnegative")
    if not isinstance(address, SuccessorFiberCacheAddress):
        raise TypeError("validation time requires SuccessorFiberCacheAddress")
    time_hex = semantic_p50_time_hex(stream_index=stream_index, address=address)
    value = float.fromhex(time_hex)
    if not 0.0 < value < 1.0:
        raise AssertionError("validation time escaped its strict open support")
    if value.hex() != time_hex:
        raise AssertionError("validation time serialization is not exact")
    return time_hex


def semantic_p50_runner_implementation_sha256() -> str:
    """Hash the exact local sources used by the pure P50 score/update path."""

    root = Path(__file__).resolve().parents[3]
    digest = hashlib.sha256()
    for relative in _IMPLEMENTATION_SOURCES:
        path = root / relative
        if not path.is_file():
            raise SemanticP50RunnerError(
                f"semantic P50 runtime implementation source is absent: {relative}"
            )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SemanticP50RuntimeInputs:
    """Already authenticated objects and exact source states needed for P50."""

    prepared_recipe: Mapping[str, Any]
    cache: SemanticP50SuccessorCache
    scratch_runtime: SemanticScratchRuntime
    validation_baseline: VerifiedSemanticP50ValidationBaseline
    states_by_address_sha256: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.prepared_recipe, Mapping):
            raise TypeError("prepared_recipe must be a mapping")
        if not isinstance(self.cache, SemanticP50SuccessorCache):
            raise TypeError("cache must be a SemanticP50SuccessorCache")
        if not isinstance(self.scratch_runtime, SemanticScratchRuntime):
            raise TypeError("scratch_runtime must be a SemanticScratchRuntime")
        if not isinstance(self.validation_baseline, VerifiedSemanticP50ValidationBaseline):
            raise TypeError(
                "validation_baseline must be a verified semantic P50 validation baseline"
            )
        if not isinstance(self.states_by_address_sha256, Mapping):
            raise TypeError("states_by_address_sha256 must be a mapping")


@dataclass(frozen=True, slots=True)
class SemanticP50RunArtifacts:
    """In-memory terminal artifacts; publication remains a separate operation."""

    scratch_validation_evaluations: tuple[SemanticP50ValidationEvaluation, ...]
    final_validation_evaluations: tuple[SemanticP50ValidationEvaluation, ...]
    checkpoint: Mapping[str, Any]
    result: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class VerifiedSemanticP50RunArtifacts:
    """Strictly reopened physical terminal P50 artifacts."""

    completion_path: Path
    completion: Mapping[str, Any]
    result_path: Path
    result: Mapping[str, Any]
    checkpoint_path: Path
    checkpoint: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _ValidatedInputs:
    prepared: Mapping[str, Any]
    train_rows: tuple[Mapping[str, Any], ...]
    validation_rows: tuple[Mapping[str, Any], ...]
    train_records: Mapping[str, SuccessorFiberCacheRecord]
    validation_records: Mapping[str, SuccessorFiberCacheRecord]
    states: Mapping[str, Any]
    frozen_scratch_evaluations: tuple[SemanticP50ValidationEvaluation, ...]


def _validate_stream_rows(prepared: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    rows = prepared.get("ordered_stream_rows")
    required_cells = tuple(prepared.get("declared_nonempty_semantic_cells", ()))
    if (
        not isinstance(rows, list)
        or len(rows) != SCHEDULED_EXAMPLES
        or not required_cells
        or tuple(sorted(set(required_cells))) != required_cells
    ):
        raise SemanticP50RunnerError("P50 stream size or declared cell identity disagrees")
    family_draws: Counter[str] = Counter()
    cell_draws: Counter[str] = Counter()
    family_steps: dict[str, set[int]] = {family: set() for family in ACTIVE8_FAMILIES}
    cell_steps: dict[str, set[int]] = {cell: set() for cell in required_cells}
    parsed: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != _STREAM_ROW_FIELDS:
            raise SemanticP50RunnerError(f"P50 stream row {index} fields disagree")
        body = dict(row)
        supplied_sha = body.pop("row_sha256")
        address = _parse_address(row["address"], field=f"ordered_stream_rows[{index}].address")
        family = row["family"]
        cell = row["semantic_cell_id"]
        cell_components = cell.rsplit(":", 2) if isinstance(cell, str) else []
        try:
            time = float.fromhex(row["time_hex"])
        except (TypeError, ValueError) as error:
            raise SemanticP50RunnerError(f"P50 stream row {index} time is invalid") from error
        if (
            supplied_sha != _sha(body)
            or row["stream_index"] != index
            or row["optimizer_step"] != index // BATCH_SIZE
            or row["batch_offset"] != index % BATCH_SIZE
            or address.partition != "train"
            or address.is_terminal
            or family not in ACTIVE8_FAMILIES
            or cell not in cell_steps
            or len(cell_components) != 3
            or cell_components[1] != family
            or not cell_components[2]
            or row["data_lane"] != address.layer
            or row["source_group_id"] is not None
            or row["time_hex"]
            != semantic_p50_validation_time_hex(stream_index=index, address=address)
            or not 0.0 < time < 1.0
            or time.hex() != row["time_hex"]
            or type(row["identity_coefficient"]) is not float
            or row["identity_coefficient"] != 1.0
            or row["importance_correction"] is not None
            or type(row["path_position_coefficient"]) is not float
            or row["path_position_coefficient"] != 1.0
        ):
            raise SemanticP50RunnerError(f"P50 stream row {index} semantics disagree")
        family_draws[family] += 1
        cell_draws[cell] += 1
        family_steps[family].add(index // BATCH_SIZE)
        cell_steps[cell].add(index // BATCH_SIZE)
        parsed.append(row)

    def expected_exposure(field: str) -> dict[str, tuple[int, int, int]]:
        key = "planned_family_exposure" if field == "family" else "planned_semantic_cell_exposure"
        identity = field if field == "family" else "semantic_cell_id"
        reports = prepared.get(key)
        if not isinstance(reports, list):
            raise SemanticP50RunnerError(f"{key} is absent")
        result: dict[str, tuple[int, int, int]] = {}
        for report in reports:
            if not isinstance(report, Mapping) or identity not in report:
                raise SemanticP50RunnerError(f"{key} row is malformed")
            result[str(report[identity])] = (
                int(report.get("scheduled_draw_count", -1)),
                int(report.get("optimizer_step_opportunity_count", -1)),
                int(report.get("minimum_observed_nonzero_gradient_update_count", -1)),
            )
        return result

    family_expected = expected_exposure("family")
    cell_expected = expected_exposure("semantic_cell_id")
    family_floors = prepared.get("minimum_nonzero_gradient_updates_by_family")
    cell_floors = prepared.get("minimum_nonzero_gradient_updates_by_semantic_cell")
    if (
        set(family_draws) != set(ACTIVE8_FAMILIES)
        or set(cell_draws) != set(required_cells)
        or family_expected
        != {
            family: (
                family_draws[family],
                len(family_steps[family]),
                int(family_floors[family]),
            )
            for family in ACTIVE8_FAMILIES
        }
        or cell_expected
        != {
            cell: (cell_draws[cell], len(cell_steps[cell]), int(cell_floors[cell]))
            for cell in required_cells
        }
    ):
        raise SemanticP50RunnerError("P50 planned family or cell exposure disagrees")
    return tuple(parsed)


def _validate_cache_and_states(
    inputs: SemanticP50RuntimeInputs,
    *,
    prepared: Mapping[str, Any],
) -> tuple[
    tuple[Mapping[str, Any], ...],
    Mapping[str, SuccessorFiberCacheRecord],
    Mapping[str, SuccessorFiberCacheRecord],
]:
    cache = inputs.cache
    try:
        completion = validate_semantic_p50_successor_cache_completion(cache.completion)
    except (TypeError, ValueError, SemanticP50SuccessorCacheError) as error:
        raise SemanticP50RunnerError("combined P50 cache completion is invalid") from error
    manifest = dict(cache.manifest)
    manifest_body = dict(manifest)
    manifest_sha = manifest_body.pop("manifest_sha256", None)
    prerequisites = prepared["prerequisites"]
    validation = manifest.get("validation_contract")
    candidate_rows = (
        None if not isinstance(validation, Mapping) else validation.get("candidate_rows")
    )
    if (
        manifest_sha != _sha(manifest_body)
        or completion["manifest_sha256"] != manifest_sha
        or completion["prepared_recipe_sha256"] != prepared["prepared_recipe_sha256"]
        or manifest.get("prepared_recipe_binding", {}).get("prepared_recipe_sha256")
        != prepared["prepared_recipe_sha256"]
        or completion["validation_contract_sha256"]
        != (validation or {}).get("validation_contract_sha256")
        or completion["scratch_initial_model_state_sha256"]
        != prerequisites["scratch_initial_model_state_sha256"]
        or completion["process_identity_sha256"] != prerequisites["process_identity_sha256"]
        or completion["model_runtime_identity_sha256"]
        != prerequisites["model_runtime_identity_sha256"]
        or not isinstance(candidate_rows, list)
        or not candidate_rows
    ):
        raise SemanticP50RunnerError("combined P50 cache lineage disagrees")

    train_records = {
        _address_sha256(record.address): record for record in cache.train_requested_records
    }
    validation_records = {
        _address_sha256(record.address): record for record in cache.validation_requested_records
    }
    train_addresses = tuple(
        _parse_address(row, field="requested_address_union.address")
        for row in prepared["requested_address_union"]
    )
    train_hashes = {_address_sha256(address) for address in train_addresses}
    parsed_validation: list[Mapping[str, Any]] = []
    validation_hashes: set[str] = set()
    previous: SuccessorFiberCacheAddress | None = None
    for index, candidate in enumerate(candidate_rows):
        if not isinstance(candidate, Mapping):
            raise SemanticP50RunnerError("validation candidate is not an object")
        expected_candidate_fields = {
            "address",
            "family",
            "semantic_cell_id",
            "data_lane",
            "assignment_sha256",
            "candidate_sha256",
        }
        if "time_hex" in candidate:
            expected_candidate_fields.add("time_hex")
        if set(candidate) != expected_candidate_fields:
            raise SemanticP50RunnerError("validation candidate fields disagree")
        body = dict(candidate)
        candidate_sha = body.pop("candidate_sha256", None)
        address = _parse_address(candidate.get("address"), field=f"validation[{index}].address")
        family = candidate.get("family")
        cell = candidate.get("semantic_cell_id")
        cell_components = cell.rsplit(":", 2) if isinstance(cell, str) else []
        if (
            candidate_sha != _sha(body)
            or address.partition != "validation"
            or address.is_terminal
            or (previous is not None and address <= previous)
            or family not in ACTIVE8_FAMILIES
            or not isinstance(cell, str)
            or len(cell_components) != 3
            or cell_components[1] != family
            or not cell_components[2]
            or candidate.get("data_lane") != address.layer
        ):
            raise SemanticP50RunnerError("validation candidate identity or ordering disagrees")
        previous = address
        validation_hashes.add(_address_sha256(address))
        parsed_validation.append(candidate)
    required_cells = set(prepared["validation_contract"]["required_nonempty_semantic_cells"])
    if (
        set(train_records) != train_hashes
        or set(validation_records) != validation_hashes
        or len(train_records) != len(cache.train_requested_records)
        or len(validation_records) != len(cache.validation_requested_records)
        or completion["train_requested_address_count"] != len(train_records)
        or completion["validation_requested_address_count"] != len(validation_records)
        or completion["requested_address_count"] != len(train_records) + len(validation_records)
        or {str(row["family"]) for row in parsed_validation} != set(ACTIVE8_FAMILIES)
        or {_address_sha256(record.address) for record in cache.requested_records}
        != train_hashes | validation_hashes
        or len(cache.requested_records) != len(train_hashes | validation_hashes)
        or train_hashes & validation_hashes
        or not required_cells.issubset({str(row["semantic_cell_id"]) for row in parsed_validation})
    ):
        raise SemanticP50RunnerError("combined cache train/validation requested union disagrees")
    combined = {**train_records, **validation_records}
    if set(inputs.states_by_address_sha256) != set(combined):
        raise SemanticP50RunnerError("exact state inventory differs from cache requested union")
    for address_sha, record in combined.items():
        if (
            not isinstance(record, SuccessorFiberCacheRecord)
            or record.address.is_terminal
            or record.teacher_fiber is None
            or cache.record_for_address(record.address) != record
            or any(
                alias.family_name not in ACTIVE8_FAMILIES for alias in record.teacher_fiber.aliases
            )
        ):
            raise SemanticP50RunnerError("cache contains an unsupported or missing teacher fiber")
        state = inputs.states_by_address_sha256[address_sha]
        if persistent_slot_state_sha256(state) != record.source_state_sha256:
            raise SemanticP50RunnerError("exact source state differs from its cached fiber")
    return tuple(parsed_validation), train_records, validation_records


def _validate_inputs(inputs: SemanticP50RuntimeInputs) -> _ValidatedInputs:
    try:
        prepared = validate_semantic_p50_prepared_recipe(inputs.prepared_recipe)
    except (TypeError, ValueError, SemanticP50SuccessorCacheError) as error:
        raise SemanticP50RunnerError("prepared P50 recipe failed exact validation") from error
    if (
        dict(prepared.get("recipe", {})) != _EXPECTED_RECIPE
        or tuple(prepared.get("required_families", ())) != tuple(ACTIVE8_FAMILIES)
        or prepared.get("unresolved_physical_bindings")
        != list(prepared.get("required_physical_binding_purposes", ()))
        or prepared.get("validation_contract", {}).get(
            "maximum_family_final_minus_baseline_for_p50_nonincrease_nats"
        )
        != MAXIMUM_FAMILY_P50_NONINCREASE_NATS
        or prepared.get("validation_contract", {}).get(
            "maximum_cell_final_minus_baseline_for_p50_nonincrease_nats"
        )
        != MAXIMUM_CELL_P50_NONINCREASE_NATS
        or prepared.get("validation_contract", {}).get(
            "p50_nonincrease_numerical_equivalence_rationale"
        )
        != "one_e_minus_seven_nats_allows_only_float32_reduction_equivalence_not_regression"
        or prepared.get("validation_contract", {}).get("catastrophic_regression_sentinel_rationale")
        != "plus_0_25_nats_is_a_separate_abort_sentinel_and_not_a_learning_criterion"
    ):
        raise SemanticP50RunnerError("prepared P50 optimization contract disagrees")
    train_rows = _validate_stream_rows(prepared)
    validation_rows, train_records, validation_records = _validate_cache_and_states(
        inputs, prepared=prepared
    )
    runtime = inputs.scratch_runtime
    model = runtime.model
    try:
        parameter = next(model.parameters())
    except StopIteration as error:
        raise SemanticP50RunnerError("scratch model has no parameters") from error
    observed_state = state_dict_semantic_sha256(model.state_dict())
    prerequisites = prepared["prerequisites"]
    if (
        observed_state != runtime.initial_model_state_sha256
        or observed_state != prerequisites["scratch_initial_model_state_sha256"]
        or runtime.process_identity_sha256 != prerequisites["process_identity_sha256"]
        or runtime.architecture.operator_capability_fingerprint
        != prerequisites["operator_capability_fingerprint"]
        or parameter.dtype != torch.float32
        or any(
            tensor.dtype != torch.float32
            for tensor in model.state_dict().values()
            if tensor.is_floating_point()
        )
    ):
        raise SemanticP50RunnerError("physical scratch model identity or fp32 contract disagrees")
    baseline = inputs.validation_baseline
    baseline_result = dict(baseline.result)
    baseline_completion = dict(baseline.completion)
    result_body = dict(baseline_result)
    supplied_result_sha256 = result_body.pop("result_sha256", None)
    completion_body = dict(baseline_completion)
    supplied_completion_sha256 = completion_body.pop("completion_sha256", None)
    frozen_scratch_evaluations = semantic_p50_validation_evaluations_from_verified_baseline(
        baseline
    )
    expected_evaluation_metadata: list[tuple[object, ...]] = []
    for index, row in enumerate(validation_rows):
        address = _parse_address(row["address"], field=f"validation[{index}].address")
        record = validation_records[_address_sha256(address)]
        expected_evaluation_metadata.append(
            (
                address,
                str(row["family"]),
                str(row["semantic_cell_id"]),
                _sha(successor_fiber_cache_record_payload(record)),
                semantic_p50_validation_time_hex(stream_index=index, address=address),
            )
        )
    observed_evaluation_metadata = [
        (
            item.address,
            item.family,
            item.semantic_cell_id,
            item.cache_record_sha256,
            item.time_hex,
        )
        for item in frozen_scratch_evaluations
    ]
    binding = baseline.inventory.binding
    if (
        supplied_result_sha256 != _sha(result_body)
        or supplied_completion_sha256 != _sha(completion_body)
        or baseline_completion.get("result_sha256") != supplied_result_sha256
        or baseline_result.get("evaluated_model_state_sha256") != runtime.initial_model_state_sha256
        or baseline_result.get("successor_cache_completion_sha256")
        != inputs.cache.completion["completion_sha256"]
        or baseline_result.get("successor_cache_manifest_sha256")
        != inputs.cache.manifest["manifest_sha256"]
        or baseline_result.get("successor_cache_validation_contract_sha256")
        != inputs.cache.manifest["validation_contract"]["validation_contract_sha256"]
        or baseline_completion.get("successor_cache_completion_sha256")
        != inputs.cache.completion["completion_sha256"]
        or getattr(binding, "prepared_recipe_sha256", None) != prepared["prepared_recipe_sha256"]
        or getattr(binding, "scratch_initial_model_state_sha256", None)
        != runtime.initial_model_state_sha256
        or getattr(binding, "process_identity_sha256", None) != runtime.process_identity_sha256
        or baseline_result.get("evaluation_records")
        != [item.as_payload() for item in frozen_scratch_evaluations]
        or observed_evaluation_metadata != expected_evaluation_metadata
    ):
        raise SemanticP50RunnerError(
            "frozen physical scratch baseline differs from runtime, cache, or validation union"
        )
    return _ValidatedInputs(
        prepared=prepared,
        train_rows=train_rows,
        validation_rows=validation_rows,
        train_records=train_records,
        validation_records=validation_records,
        states=dict(inputs.states_by_address_sha256),
        frozen_scratch_evaluations=frozen_scratch_evaluations,
    )


def _rows_to_batch(
    *,
    rows: Sequence[Mapping[str, Any]],
    records: Mapping[str, SuccessorFiberCacheRecord],
    states: Mapping[str, Any],
    collator: Any,
) -> tuple[Any, tuple[Any, ...]]:
    examples: list[FactorizedMarkExample] = []
    fibers: list[Any] = []
    for row in rows:
        address = _parse_address(row["address"], field="runtime row address")
        address_sha = _address_sha256(address)
        try:
            record = records[address_sha]
            state = states[address_sha]
        except KeyError as error:
            raise SemanticP50RunnerError("runtime row is absent from cache/state union") from error
        fiber = record.teacher_fiber
        if fiber is None:
            raise SemanticP50RunnerError("nonterminal runtime row has no teacher fiber")
        examples.append(
            FactorizedMarkExample(
                state=state,
                time=float.fromhex(str(row["time_hex"])),
                teacher_action=None,
                teacher_rule_name=str(row["family"]),
                teacher_rate=1.0,
                importance_weight=1.0,
            )
        )
        fibers.append(fiber)
    try:
        return collator(examples), tuple(fibers)
    except (TypeError, ValueError, RuntimeError) as error:
        raise SemanticP50RunnerError("P50 batch collation failed closed") from error


def _route_gradient_evidence(
    model: Any,
    prediction: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    group_field: str,
) -> dict[str, dict[str, float | bool]]:
    named = dict(model.named_parameters())
    family_parameters = tuple(
        parameter
        for name, parameter in named.items()
        if name.startswith(FAMILY_ROUTE_PREFIXES) and parameter.requires_grad
    )
    if not family_parameters:
        raise SemanticP50RunnerError("P50 family route parameter surface is absent")
    log_probabilities = prediction.selected_productive_successor_log_probability
    result: dict[str, dict[str, float | bool]] = {}
    for identity in sorted({str(row[group_field]) for row in rows}):
        indices = [index for index, row in enumerate(rows) if row[group_field] == identity]
        families = {str(rows[index]["family"]) for index in indices}
        if len(families) != 1:
            raise SemanticP50RunnerError("one semantic gradient group spans multiple families")
        family = next(iter(families))
        action_parameters = tuple(
            parameter
            for name, parameter in named.items()
            if name.startswith(ACTION_ROUTE_PREFIXES[family]) and parameter.requires_grad
        )
        if not action_parameters:
            raise SemanticP50RunnerError(
                f"P50 action route parameter surface is absent for {family}"
            )
        loss = -log_probabilities[indices].mean()
        gradients = torch.autograd.grad(
            loss,
            (*family_parameters, *action_parameters),
            retain_graph=True,
            allow_unused=True,
        )

        def route(values: Sequence[Tensor | None]) -> tuple[bool, float]:
            finite = all(value is None or bool(torch.isfinite(value).all()) for value in values)
            norm = math.sqrt(
                math.fsum(
                    float(value.detach().norm()) ** 2 for value in values if value is not None
                )
            )
            return finite and math.isfinite(norm) and norm > 0.0, norm

        family_seen, family_norm = route(gradients[: len(family_parameters)])
        action_seen, action_norm = route(gradients[len(family_parameters) :])
        result[identity] = {
            "family_route_finite_nonzero": family_seen,
            "family_route_gradient_norm": family_norm,
            "action_route_finite_nonzero": action_seen,
            "action_route_gradient_norm": action_norm,
        }
    return result


def _new_exposure(
    *, identities: Sequence[str], floors: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    return {
        identity: {
            "scheduled_draw_count": 0,
            "optimizer_step_opportunity_count": 0,
            "finite_nonzero_gradient_update_count": 0,
            "minimum_required_finite_nonzero_gradient_update_count": int(floors[identity]),
            "cumulative_family_route_gradient_norm": 0.0,
            "cumulative_action_route_gradient_norm": 0.0,
        }
        for identity in identities
    }


def _merge_exposure(
    cumulative: dict[str, dict[str, Any]],
    observed: Mapping[str, Mapping[str, float | bool]],
    *,
    rows: Sequence[Mapping[str, Any]],
    group_field: str,
) -> None:
    draws = Counter(str(row[group_field]) for row in rows)
    for identity, count in draws.items():
        evidence = observed[identity]
        nonzero = bool(evidence["family_route_finite_nonzero"]) and bool(
            evidence["action_route_finite_nonzero"]
        )
        if not nonzero:
            raise SemanticP50RunnerError(
                f"zero or nonfinite P50 family/action gradient for {group_field}={identity}"
            )
        target = cumulative[identity]
        target["scheduled_draw_count"] += count
        target["optimizer_step_opportunity_count"] += 1
        target["finite_nonzero_gradient_update_count"] += 1
        target["cumulative_family_route_gradient_norm"] += float(
            evidence["family_route_gradient_norm"]
        )
        target["cumulative_action_route_gradient_norm"] += float(
            evidence["action_route_gradient_norm"]
        )


def _validation_runtime_rows(
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], ...], list[dict[str, Any]]]:
    runtime_rows: list[Mapping[str, Any]] = []
    time_rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        address = _parse_address(candidate["address"], field="validation candidate address")
        time_hex = semantic_p50_validation_time_hex(stream_index=index, address=address)
        supplied_time = candidate.get("time_hex")
        if supplied_time is not None and supplied_time != time_hex:
            raise SemanticP50RunnerError("validation candidate time differs from frozen derivation")
        runtime_rows.append(
            {
                "address": _address_payload(address),
                "family": candidate["family"],
                "semantic_cell_id": candidate["semantic_cell_id"],
                "time_hex": time_hex,
            }
        )
        time_rows.append(
            {
                "stream_index": index,
                "address": _address_payload(address),
                "time_hex": time_hex,
            }
        )
    return tuple(runtime_rows), time_rows


def _evaluate_validation(
    *,
    model: Any,
    candidates: Sequence[Mapping[str, Any]],
    records: Mapping[str, SuccessorFiberCacheRecord],
    states: Mapping[str, Any],
    collator: Any,
) -> tuple[tuple[SemanticP50ValidationEvaluation, ...], list[dict[str, Any]]]:
    runtime_rows, time_rows = _validation_runtime_rows(candidates)
    evaluations: list[SemanticP50ValidationEvaluation] = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, len(runtime_rows), BATCH_SIZE):
            rows = runtime_rows[start : start + BATCH_SIZE]
            batch, fibers = _rows_to_batch(
                rows=rows, records=records, states=states, collator=collator
            )
            device_batch = batch.to(model.device)
            try:
                prediction = forward_teacher_successor_batch(model, device_batch, fibers)
            except (ValueError, RuntimeError, SuccessorTrainingError) as error:
                raise SemanticP50RunnerError(
                    "validation successor scoring failed closed"
                ) from error
            log_probabilities = prediction.selected_productive_successor_log_probability
            if log_probabilities.shape != (len(rows),) or not bool(
                torch.isfinite(log_probabilities).all()
            ):
                raise SemanticP50RunnerError("validation returned missing or nonfinite NLLs")
            for offset, (row, log_probability) in enumerate(
                zip(rows, log_probabilities.detach().cpu(), strict=True)
            ):
                address = _parse_address(row["address"], field="validation evaluation address")
                record = records[_address_sha256(address)]
                nll = -float(log_probability)
                if not math.isfinite(nll) or nll < 0.0:
                    raise SemanticP50RunnerError("validation canonical-successor NLL is invalid")
                try:
                    evaluation = SemanticP50ValidationEvaluation(
                        address=address,
                        family=str(row["family"]),
                        semantic_cell_id=str(row["semantic_cell_id"]),
                        cache_record_sha256=_sha(successor_fiber_cache_record_payload(record)),
                        time_hex=str(row["time_hex"]),
                        canonical_successor_nll_nats=nll,
                    )
                except (TypeError, ValueError, RuntimeError) as error:
                    raise SemanticP50RunnerError(
                        f"validation evaluation {start + offset} failed its typed schema"
                    ) from error
                evaluations.append(evaluation)
    model.train(was_training)
    return tuple(evaluations), time_rows


def _evaluate_and_require_frozen_scratch_baseline(
    *,
    inputs: SemanticP50RuntimeInputs,
    validated: _ValidatedInputs,
    collator: Any,
) -> tuple[tuple[SemanticP50ValidationEvaluation, ...], list[dict[str, Any]]]:
    model = inputs.scratch_runtime.model
    evaluations, time_rows = _evaluate_validation(
        model=model,
        candidates=validated.validation_rows,
        records=validated.validation_records,
        states=validated.states,
        collator=collator,
    )
    if [item.as_payload() for item in evaluations] != [
        item.as_payload() for item in validated.frozen_scratch_evaluations
    ]:
        raise SemanticP50RunnerError(
            "recomputed scratch validation differs exactly from the frozen physical baseline"
        )
    if (
        state_dict_semantic_sha256(model.state_dict())
        != inputs.scratch_runtime.initial_model_state_sha256
    ):
        raise SemanticP50RunnerError("pre-update validation changed the scratch model")
    return evaluations, time_rows


def evaluate_semantic_p50_scratch_validation(
    inputs: SemanticP50RuntimeInputs,
) -> tuple[SemanticP50ValidationEvaluation, ...]:
    """Recompute and exact-compare the frozen scratch baseline without updates."""

    validated = _validate_inputs(inputs)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    evaluations, _ = _evaluate_and_require_frozen_scratch_baseline(
        inputs=inputs,
        validated=validated,
        collator=_collator(inputs.scratch_runtime.model),
    )
    return evaluations


def _normalized_rng_state(value: Mapping[str, Any]) -> dict[str, Any]:
    numpy_state = value["numpy"]
    numpy_values = numpy_state[1]
    if hasattr(numpy_values, "tolist"):
        numpy_values = numpy_values.tolist()
    return {
        "python": value["python"],
        "numpy": [
            numpy_state[0],
            numpy_values,
            int(numpy_state[2]),
            int(numpy_state[3]),
            float(numpy_state[4]),
        ],
        "torch_cpu": value["torch_cpu"],
        "torch_cuda": value["torch_cuda"],
    }


def _rng_state_sha256(value: Mapping[str, Any]) -> str:
    try:
        return optimizer_state_semantic_sha256(_normalized_rng_state(value))
    except (KeyError, TypeError, ValueError, SemanticT1CapacityRunnerError) as error:
        raise SemanticP50RunnerError("P50 RNG state is not semantically hashable") from error


def _execution_environment(model: Any) -> dict[str, Any]:
    parameter = next(model.parameters())
    device = parameter.device
    body = {
        "hardware_class": platform.machine(),
        "device_type": device.type,
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()
        ),
        "dtype": str(parameter.dtype),
        "mixed_precision": False,
        "cuda_matmul_tf32_allowed": (
            bool(torch.backends.cuda.matmul.allow_tf32) if hasattr(torch.backends, "cuda") else None
        ),
        "cudnn_tf32_allowed": (
            bool(torch.backends.cudnn.allow_tf32) if torch.backends.cudnn.is_available() else None
        ),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "numpy_version": str(np.__version__),
    }
    return {**body, "environment_sha256": _sha(body)}


def _validation_nll_checks(
    *,
    scratch: Sequence[SemanticP50ValidationEvaluation],
    final: Sequence[SemanticP50ValidationEvaluation],
    prepared: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Recompute exact-count family/cell means and enforce frozen ceilings."""

    scratch_by_address = {item.address: item for item in scratch}
    final_by_address = {item.address: item for item in final}
    if (
        not scratch_by_address
        or len(scratch_by_address) != len(scratch)
        or len(final_by_address) != len(final)
        or set(scratch_by_address) != set(final_by_address)
    ):
        raise SemanticP50RunnerError(
            "scratch/final validation address inventories differ or repeat"
        )
    for address, before in scratch_by_address.items():
        after = final_by_address[address]
        if (
            before.family != after.family
            or before.semantic_cell_id != after.semantic_cell_id
            or before.cache_record_sha256 != after.cache_record_sha256
            or before.time_hex != after.time_hex
        ):
            raise SemanticP50RunnerError(
                "scratch/final validation classification, cache, or time differs"
            )

    validation_contract = prepared["validation_contract"]
    required_cells = tuple(validation_contract["required_nonempty_semantic_cells"])
    required_cell_set = frozenset(required_cells)
    gated_scratch = tuple(item for item in scratch if item.semantic_cell_id in required_cell_set)
    gated_final = tuple(item for item in final if item.semantic_cell_id in required_cell_set)
    family_ceiling = float(
        validation_contract["maximum_family_final_minus_baseline_successor_nll_nats"]
    )
    cell_ceiling = float(
        validation_contract["maximum_cell_final_minus_baseline_successor_nll_nats"]
    )

    def rows_for(
        *,
        identities: Sequence[str],
        field: str,
        ceiling: float,
        non_increase_ceiling: float,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for identity in identities:
            before_values = [
                item.canonical_successor_nll_nats
                for item in gated_scratch
                if getattr(item, field) == identity
            ]
            after_values = [
                item.canonical_successor_nll_nats
                for item in gated_final
                if getattr(item, field) == identity
            ]
            if not before_values or len(before_values) != len(after_values):
                raise SemanticP50RunnerError(
                    f"validation NLL reducer has missing or mismatched {field}={identity}"
                )
            before_mean = math.fsum(before_values) / len(before_values)
            after_mean = math.fsum(after_values) / len(after_values)
            delta = after_mean - before_mean
            catastrophic_regression_sentinel_passed = math.isfinite(delta) and delta <= ceiling
            non_increase_passed = math.isfinite(delta) and delta <= non_increase_ceiling
            passed = catastrophic_regression_sentinel_passed and non_increase_passed
            rows.append(
                {
                    field: identity,
                    "example_count": len(before_values),
                    "scratch_mean_canonical_successor_nll_nats": before_mean,
                    "final_mean_canonical_successor_nll_nats": after_mean,
                    "final_minus_scratch_nll_nats": delta,
                    "catastrophic_regression_ceiling_nats": ceiling,
                    "catastrophic_regression_sentinel_passed": (
                        catastrophic_regression_sentinel_passed
                    ),
                    "non_increase_numerical_tolerance_nats": (non_increase_ceiling),
                    "non_increase_passed": non_increase_passed,
                    "passed": passed,
                }
            )
        return rows

    family_rows = rows_for(
        identities=ACTIVE8_FAMILIES,
        field="family",
        ceiling=family_ceiling,
        non_increase_ceiling=float(
            validation_contract["maximum_family_final_minus_baseline_for_p50_nonincrease_nats"]
        ),
    )
    cell_rows = rows_for(
        identities=required_cells,
        field="semantic_cell_id",
        ceiling=cell_ceiling,
        non_increase_ceiling=float(
            validation_contract["maximum_cell_final_minus_baseline_for_p50_nonincrease_nats"]
        ),
    )
    if any(not row["passed"] for row in (*family_rows, *cell_rows)):
        failures = [
            str(row.get("family", row.get("semantic_cell_id")))
            for row in (*family_rows, *cell_rows)
            if not row["passed"]
        ]
        raise SemanticP50RunnerError(
            "frozen scratch-to-final validation NLL non-increase or catastrophic "
            f"regression sentinel failed: {failures}"
        )
    return family_rows, cell_rows


def _checkpoint_payload(
    *,
    identity: Mapping[str, Any],
    configuration: Mapping[str, Any],
    model: Any,
    optimizer: torch.optim.Optimizer,
    rng_state: Mapping[str, Any],
    hazard_initial_sha256: str,
) -> dict[str, Any]:
    model_state = _clone_state_dict(model)
    optimizer_state = optimizer.state_dict()
    normalized_rng_state = _normalized_rng_state(rng_state)
    body = {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": CHECKPOINT_STATUS,
        **NO_AUTHORITY,
        "identity": dict(identity),
        "configuration": dict(configuration),
        "completed_optimizer_steps": OPTIMIZER_STEPS,
        "resume_count": 0,
        "model_state_sha256": state_dict_semantic_sha256(model_state),
        "optimizer_state_sha256": optimizer_state_semantic_sha256(optimizer_state),
        "rng_state_sha256": _rng_state_sha256(normalized_rng_state),
        "hazard_initial_state_sha256": hazard_initial_sha256,
        "hazard_final_state_sha256": state_dict_semantic_sha256(_hazard_state(model)),
    }
    return {
        **body,
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "rng_state": normalized_rng_state,
        "checkpoint_sha256": _sha(body),
    }


def validate_semantic_p50_checkpoint_payload(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute all tensor/RNG hashes for one in-memory terminal checkpoint."""

    payload = dict(value)
    expected_fields = {
        "schema",
        "schema_version",
        "status",
        *NO_AUTHORITY,
        "identity",
        "configuration",
        "completed_optimizer_steps",
        "resume_count",
        "model_state_sha256",
        "optimizer_state_sha256",
        "rng_state_sha256",
        "hazard_initial_state_sha256",
        "hazard_final_state_sha256",
        "model_state",
        "optimizer_state",
        "rng_state",
        "checkpoint_sha256",
    }
    body = {
        key: nested
        for key, nested in payload.items()
        if key not in {"model_state", "optimizer_state", "rng_state", "checkpoint_sha256"}
    }
    try:
        identity = payload["identity"]
        configuration = payload["configuration"]
        model_hazard_state = {
            name: tensor
            for name, tensor in payload["model_state"].items()
            if name.startswith(TOTAL_HAZARD_PREFIX)
        }
        valid = (
            set(payload) == expected_fields
            and isinstance(identity, Mapping)
            and set(identity)
            == {
                "prepared_recipe_sha256",
                "recipe_policy_sha256",
                "ordered_address_stream_sha256",
                "ordered_training_stream_sha256",
                "requested_address_union_sha256",
                "cache_completion_sha256",
                "cache_manifest_sha256",
                "cache_validation_contract_sha256",
                "validation_baseline_completion_sha256",
                "validation_baseline_result_sha256",
                "validation_baseline_evaluation_records_sha256",
                "validation_time_stream_sha256",
                "exact_state_inventory_sha256",
                "initial_model_state_sha256",
                "process_identity_sha256",
                "runner_implementation_sha256",
                "execution_environment_sha256",
            }
            and all(
                _require_sha(identity[name], field=f"checkpoint.identity.{name}") == identity[name]
                for name in identity
            )
            and identity["runner_implementation_sha256"]
            == semantic_p50_runner_implementation_sha256()
            and isinstance(configuration, Mapping)
            and set(configuration)
            == {
                "optimization_policy",
                "optimization_policy_sha256",
                "optimizer_configuration",
                "optimizer_configuration_sha256",
                "validation_time_contract",
                "scratch_architecture",
            }
            and configuration["optimization_policy"] == _EXPECTED_RECIPE
            and configuration["optimization_policy_sha256"] == _sha(_EXPECTED_RECIPE)
            and configuration["optimizer_configuration_sha256"]
            == _sha(configuration["optimizer_configuration"])
            and configuration["validation_time_contract"]["ordered_validation_time_stream_sha256"]
            == identity["validation_time_stream_sha256"]
            and payload.get("schema") == CHECKPOINT_SCHEMA
            and payload.get("schema_version") == SCHEMA_VERSION
            and payload.get("status") == CHECKPOINT_STATUS
            and all(payload.get(name) is expected for name, expected in NO_AUTHORITY.items())
            and payload.get("completed_optimizer_steps") == OPTIMIZER_STEPS
            and payload.get("resume_count") == 0
            and payload.get("model_state_sha256")
            == state_dict_semantic_sha256(payload["model_state"])
            and payload.get("optimizer_state_sha256")
            == optimizer_state_semantic_sha256(payload["optimizer_state"])
            and payload.get("rng_state_sha256") == _rng_state_sha256(payload["rng_state"])
            and payload.get("hazard_initial_state_sha256")
            == payload.get("hazard_final_state_sha256")
            and bool(model_hazard_state)
            and payload.get("hazard_final_state_sha256")
            == state_dict_semantic_sha256(model_hazard_state)
            and payload.get("checkpoint_sha256") == _sha(body)
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise SemanticP50RunnerError("P50 checkpoint payload is malformed") from error
    if not valid:
        raise SemanticP50RunnerError("P50 checkpoint semantic identity disagrees")
    return payload


def _parse_result_evaluations(
    value: object, *, field: str
) -> tuple[SemanticP50ValidationEvaluation, ...]:
    if not isinstance(value, list):
        raise SemanticP50RunnerError(f"{field} must be a list")
    result: list[SemanticP50ValidationEvaluation] = []
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise SemanticP50RunnerError(f"{field}[{index}] must be an object")
        try:
            evaluation = SemanticP50ValidationEvaluation(
                address=_parse_address(row.get("address"), field=f"{field}[{index}].address"),
                family=row.get("family"),
                semantic_cell_id=row.get("semantic_cell_id"),
                cache_record_sha256=row.get("cache_record_sha256"),
                time_hex=row.get("time_hex"),
                canonical_successor_nll_nats=row.get("canonical_successor_nll_nats"),
            )
        except (TypeError, ValueError, RuntimeError) as error:
            raise SemanticP50RunnerError(
                f"{field}[{index}] is not a typed validation evaluation"
            ) from error
        if dict(row) != evaluation.as_payload():
            raise SemanticP50RunnerError(f"{field}[{index}] semantic identity disagrees")
        result.append(evaluation)
    return tuple(result)


def _expected_runtime_identity(
    inputs: SemanticP50RuntimeInputs,
    *,
    validated: _ValidatedInputs,
    validation_time_contract: Mapping[str, Any],
    execution_environment: Mapping[str, Any],
) -> dict[str, str]:
    combined_records = {
        **validated.train_records,
        **validated.validation_records,
    }
    return {
        "prepared_recipe_sha256": validated.prepared["prepared_recipe_sha256"],
        "recipe_policy_sha256": validated.prepared["recipe_policy_sha256"],
        "ordered_address_stream_sha256": validated.prepared["ordered_address_stream_sha256"],
        "ordered_training_stream_sha256": validated.prepared["ordered_training_stream_sha256"],
        "requested_address_union_sha256": validated.prepared["requested_address_union_sha256"],
        "cache_completion_sha256": inputs.cache.completion["completion_sha256"],
        "cache_manifest_sha256": inputs.cache.manifest["manifest_sha256"],
        "cache_validation_contract_sha256": inputs.cache.manifest["validation_contract"][
            "validation_contract_sha256"
        ],
        "validation_baseline_completion_sha256": inputs.validation_baseline.completion[
            "completion_sha256"
        ],
        "validation_baseline_result_sha256": inputs.validation_baseline.result["result_sha256"],
        "validation_baseline_evaluation_records_sha256": _sha(
            inputs.validation_baseline.result["evaluation_records"]
        ),
        "validation_time_stream_sha256": validation_time_contract[
            "ordered_validation_time_stream_sha256"
        ],
        "exact_state_inventory_sha256": _sha(
            [
                {
                    "address_sha256": address_sha,
                    "source_state_sha256": combined_records[address_sha].source_state_sha256,
                }
                for address_sha in sorted(validated.states)
            ]
        ),
        "initial_model_state_sha256": inputs.scratch_runtime.initial_model_state_sha256,
        "process_identity_sha256": inputs.scratch_runtime.process_identity_sha256,
        "runner_implementation_sha256": semantic_p50_runner_implementation_sha256(),
        "execution_environment_sha256": execution_environment["environment_sha256"],
    }


def _validate_semantic_p50_result_payload(
    value: Mapping[str, Any],
    *,
    checkpoint: Mapping[str, Any],
    inputs: SemanticP50RuntimeInputs,
) -> dict[str, Any]:
    payload = dict(value)
    expected_fields = {
        "schema",
        "schema_version",
        "status",
        *NO_AUTHORITY,
        "identity",
        "optimization",
        "optimizer_configuration",
        "execution_environment",
        "runtime_observations",
        "validation_cell_coverage",
        "completed_optimizer_steps",
        "scheduled_example_count",
        "resume_count",
        "learning_demonstrated",
        "scientific_interpretation",
        "next_stage_authorized",
        "hazard_parameters_frozen_excluded_and_unchanged",
        "validation_time_contract",
        "scratch_validation_evaluations",
        "final_validation_evaluations",
        "family_validation_nll_checks",
        "semantic_cell_validation_nll_checks",
        "family_exposure_and_gradient_evidence",
        "semantic_cell_exposure_and_gradient_evidence",
        "trajectory",
        "checkpoint_sha256",
        "terminal_model_state_sha256",
        "result_sha256",
    }
    result_body = dict(payload)
    supplied_result_sha256 = result_body.pop("result_sha256", None)
    validated = _validate_inputs(inputs)
    validation_time_contract = payload.get("validation_time_contract")
    execution_environment = payload.get("execution_environment")
    if not isinstance(validation_time_contract, Mapping) or not isinstance(
        execution_environment, Mapping
    ):
        raise SemanticP50RunnerError(
            "P50 result lacks validation-time or execution-environment identity"
        )
    environment_body = dict(execution_environment)
    supplied_environment_sha256 = environment_body.pop("environment_sha256", None)
    time_body = dict(validation_time_contract)
    time_stream_sha256 = time_body.get("ordered_validation_time_stream_sha256")
    expected_time_rows = _validation_runtime_rows(validated.validation_rows)[1]
    expected_identity = _expected_runtime_identity(
        inputs,
        validated=validated,
        validation_time_contract=validation_time_contract,
        execution_environment=execution_environment,
    )
    scratch = _parse_result_evaluations(
        payload.get("scratch_validation_evaluations"),
        field="scratch_validation_evaluations",
    )
    final = _parse_result_evaluations(
        payload.get("final_validation_evaluations"),
        field="final_validation_evaluations",
    )
    expected_family_checks, expected_cell_checks = _validation_nll_checks(
        scratch=scratch,
        final=final,
        prepared=validated.prepared,
    )
    cell_counts = Counter(item.semantic_cell_id for item in scratch)
    required_cells = tuple(
        validated.prepared["validation_contract"]["required_nonempty_semantic_cells"]
    )
    expected_coverage = {
        "required_semantic_cells": list(required_cells),
        "observed_semantic_cell_counts": dict(sorted(cell_counts.items())),
        "extra_observed_semantic_cells_not_gated": sorted(set(cell_counts) - set(required_cells)),
        "all_combined_cache_validation_rows_evaluated": True,
    }
    runtime_observations = payload.get("runtime_observations")
    trajectory = payload.get("trajectory")
    if (
        set(payload) != expected_fields
        or payload.get("schema") != RESULT_SCHEMA
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != RESULT_STATUS
        or supplied_result_sha256 != _sha(result_body)
        or any(payload.get(name) is not expected for name, expected in NO_AUTHORITY.items())
        or payload.get("learning_demonstrated") is not False
        or payload.get("scientific_interpretation")
        != "NON_REGRESSION_ONLY_DOES_NOT_ESTABLISH_LEARNING"
        or payload.get("next_stage_authorized") is not None
        or payload.get("optimization") != _EXPECTED_RECIPE
        or payload.get("completed_optimizer_steps") != OPTIMIZER_STEPS
        or payload.get("scheduled_example_count") != SCHEDULED_EXAMPLES
        or payload.get("resume_count") != 0
        or payload.get("hazard_parameters_frozen_excluded_and_unchanged") is not True
        or validation_time_contract.get("schema") != VALIDATION_TIME_STREAM_SCHEMA
        or validation_time_contract.get("schema_version") != SCHEMA_VERSION
        or validation_time_contract.get("algorithm") != TIME_ALGORITHM
        or validation_time_contract.get("seed") != SEED
        or validation_time_contract.get("rows") != expected_time_rows
        or time_stream_sha256 != _sha(expected_time_rows)
        or supplied_environment_sha256 != _sha(environment_body)
        or payload.get("identity") != expected_identity
        or checkpoint.get("identity") != expected_identity
        or checkpoint.get("configuration", {}).get("validation_time_contract")
        != validation_time_contract
        or payload.get("checkpoint_sha256") != checkpoint.get("checkpoint_sha256")
        or payload.get("terminal_model_state_sha256") != checkpoint.get("model_state_sha256")
        or [item.as_payload() for item in scratch]
        != inputs.validation_baseline.result["evaluation_records"]
        or payload.get("family_validation_nll_checks") != expected_family_checks
        or payload.get("semantic_cell_validation_nll_checks") != expected_cell_checks
        or payload.get("validation_cell_coverage") != expected_coverage
        or not isinstance(runtime_observations, Mapping)
        or not isinstance(runtime_observations.get("wall_time_seconds"), (int, float))
        or not math.isfinite(float(runtime_observations["wall_time_seconds"]))
        or float(runtime_observations["wall_time_seconds"]) <= 0.0
        or not isinstance(trajectory, list)
        or len(trajectory) != OPTIMIZER_STEPS
        or [row.get("optimizer_step") for row in trajectory] != list(range(1, OPTIMIZER_STEPS + 1))
    ):
        raise SemanticP50RunnerError("P50 result semantic identity disagrees")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            while block := handle.read(1 << 20):
                digest.update(block)
    except OSError as error:
        raise SemanticP50RunnerError(f"cannot hash P50 run artifact: {path}") from error
    return digest.hexdigest()


def _load_canonical_json(path: Path, *, field: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = Path(path).read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SemanticP50RunnerError(f"{field} is absent or unreadable: {path}") from error
    if not isinstance(value, dict) or raw != _canonical_bytes(value) + b"\n":
        raise SemanticP50RunnerError(f"{field} must be canonical newline-terminated JSON")
    return value, raw


def publish_semantic_p50_run_artifacts(
    output_root: Path,
    *,
    artifacts: SemanticP50RunArtifacts,
    inputs: SemanticP50RuntimeInputs,
) -> Path:
    """Publish one immutable content-addressed result/checkpoint directory atomically."""

    if not isinstance(artifacts, SemanticP50RunArtifacts):
        raise TypeError("artifacts must be SemanticP50RunArtifacts")
    checkpoint = validate_semantic_p50_checkpoint_payload(artifacts.checkpoint)
    result = _validate_semantic_p50_result_payload(
        artifacts.result, checkpoint=checkpoint, inputs=inputs
    )
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / result["result_sha256"]
    if target.exists():
        raise SemanticP50RunnerError(
            "content-addressed P50 result already exists; publication is write-once"
        )
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".staging", dir=root))
    try:
        result_path = staging / RUN_RESULT_FILENAME
        checkpoint_path = staging / RUN_CHECKPOINT_FILENAME
        result_path.write_bytes(_canonical_bytes(result) + b"\n")
        torch.save(dict(checkpoint), checkpoint_path)
        completion_body = {
            "schema": RUN_COMPLETION_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "status": RUN_COMPLETION_STATUS,
            **NO_AUTHORITY,
            "learning_demonstrated": False,
            "next_stage_authorized": None,
            "result_relative_path": RUN_RESULT_FILENAME,
            "result_file_sha256": _file_sha256(result_path),
            "result_sha256": result["result_sha256"],
            "checkpoint_relative_path": RUN_CHECKPOINT_FILENAME,
            "checkpoint_file_sha256": _file_sha256(checkpoint_path),
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "terminal_model_state_sha256": checkpoint["model_state_sha256"],
            "identity_sha256": _sha(result["identity"]),
        }
        completion = {
            **completion_body,
            "completion_sha256": _sha(completion_body),
        }
        completion_path = staging / RUN_COMPLETION_FILENAME
        completion_path.write_bytes(_canonical_bytes(completion) + b"\n")
        for path in (result_path, checkpoint_path, completion_path):
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(staging, target)
        root_fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(root_fd)
        finally:
            os.close(root_fd)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    completion_path = target / RUN_COMPLETION_FILENAME
    open_semantic_p50_run_artifacts(
        completion_path,
        artifact_root=root,
        inputs=inputs,
    )
    return completion_path


def open_semantic_p50_run_artifacts(
    completion_path: Path,
    *,
    artifact_root: Path,
    inputs: SemanticP50RuntimeInputs,
) -> VerifiedSemanticP50RunArtifacts:
    """Strictly reopen physical P50 output and recompute all bound identities."""

    root = Path(artifact_root).resolve()
    physical_completion = Path(completion_path).resolve()
    if (
        not physical_completion.is_relative_to(root)
        or physical_completion.name != RUN_COMPLETION_FILENAME
    ):
        raise SemanticP50RunnerError(
            "P50 completion is outside artifact_root or has the wrong filename"
        )
    completion, _ = _load_canonical_json(physical_completion, field="P50 run completion")
    completion_body = dict(completion)
    supplied_completion_sha256 = completion_body.pop("completion_sha256", None)
    expected_completion_fields = {
        "schema",
        "schema_version",
        "status",
        *NO_AUTHORITY,
        "learning_demonstrated",
        "next_stage_authorized",
        "result_relative_path",
        "result_file_sha256",
        "result_sha256",
        "checkpoint_relative_path",
        "checkpoint_file_sha256",
        "checkpoint_sha256",
        "terminal_model_state_sha256",
        "identity_sha256",
        "completion_sha256",
    }
    if (
        set(completion) != expected_completion_fields
        or completion.get("schema") != RUN_COMPLETION_SCHEMA
        or completion.get("schema_version") != SCHEMA_VERSION
        or completion.get("status") != RUN_COMPLETION_STATUS
        or supplied_completion_sha256 != _sha(completion_body)
        or any(completion.get(name) is not expected for name, expected in NO_AUTHORITY.items())
        or completion.get("learning_demonstrated") is not False
        or completion.get("next_stage_authorized") is not None
        or completion.get("result_relative_path") != RUN_RESULT_FILENAME
        or completion.get("checkpoint_relative_path") != RUN_CHECKPOINT_FILENAME
        or physical_completion.parent.name != completion.get("result_sha256")
    ):
        raise SemanticP50RunnerError("P50 run completion identity disagrees")
    result_path = physical_completion.parent / RUN_RESULT_FILENAME
    checkpoint_path = physical_completion.parent / RUN_CHECKPOINT_FILENAME
    result, _ = _load_canonical_json(result_path, field="P50 result")
    if (
        _file_sha256(result_path) != completion.get("result_file_sha256")
        or result.get("result_sha256") != completion.get("result_sha256")
        or _file_sha256(checkpoint_path) != completion.get("checkpoint_file_sha256")
    ):
        raise SemanticP50RunnerError("P50 physical result or checkpoint hash disagrees")
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise SemanticP50RunnerError(
            "P50 checkpoint cannot be reopened with weights_only=True"
        ) from error
    if not isinstance(checkpoint, Mapping):
        raise SemanticP50RunnerError("P50 checkpoint payload is not a mapping")
    verified_checkpoint = validate_semantic_p50_checkpoint_payload(checkpoint)
    verified_result = _validate_semantic_p50_result_payload(
        result,
        checkpoint=verified_checkpoint,
        inputs=inputs,
    )
    if (
        completion.get("checkpoint_sha256") != verified_checkpoint["checkpoint_sha256"]
        or completion.get("terminal_model_state_sha256")
        != verified_checkpoint["model_state_sha256"]
        or completion.get("identity_sha256") != _sha(verified_result["identity"])
    ):
        raise SemanticP50RunnerError("P50 completion disagrees with reopened semantic artifacts")
    return VerifiedSemanticP50RunArtifacts(
        completion_path=physical_completion,
        completion=completion,
        result_path=result_path,
        result=verified_result,
        checkpoint_path=checkpoint_path,
        checkpoint=verified_checkpoint,
    )


def run_semantic_p50(inputs: SemanticP50RuntimeInputs) -> SemanticP50RunArtifacts:
    """Execute exactly 50 deterministic, non-resumable P50 optimizer updates."""

    validated = _validate_inputs(inputs)
    runtime = inputs.scratch_runtime
    model = runtime.model
    initial_model_state = _clone_state_dict(model)
    initial_requires_grad = {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    }
    initial_training = model.training
    try:
        torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch.backends, "cuda"):
            torch.backends.cuda.matmul.allow_tf32 = False
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
            torch.cuda.reset_peak_memory_stats(model.device)
        started_at = time.perf_counter()

        hazard_initial = _hazard_state(model)
        hazard_initial_sha = state_dict_semantic_sha256(hazard_initial)
        trainable: list[Tensor] = []
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(not name.startswith(TOTAL_HAZARD_PREFIX))
            if parameter.requires_grad:
                trainable.append(parameter)
        if not trainable or any(
            parameter.requires_grad
            for name, parameter in model.named_parameters()
            if name.startswith(TOTAL_HAZARD_PREFIX)
        ):
            raise SemanticP50RunnerError("hazard was not exactly excluded from optimization")
        optimizer = torch.optim.AdamW(trainable, lr=0.001, weight_decay=0.0)
        optimizer_configuration = _optimizer_configuration(model, optimizer)
        collator = _collator(model)

        scratch_evaluations, validation_time_rows = _evaluate_and_require_frozen_scratch_baseline(
            inputs=inputs,
            validated=validated,
            collator=collator,
        )

        family_exposure = _new_exposure(
            identities=ACTIVE8_FAMILIES,
            floors=validated.prepared["minimum_nonzero_gradient_updates_by_family"],
        )
        cells = tuple(validated.prepared["declared_nonempty_semantic_cells"])
        cell_exposure = _new_exposure(
            identities=cells,
            floors=validated.prepared["minimum_nonzero_gradient_updates_by_semantic_cell"],
        )
        trajectory: list[dict[str, Any]] = []
        model.train()
        for step in range(OPTIMIZER_STEPS):
            rows = validated.train_rows[step * BATCH_SIZE : (step + 1) * BATCH_SIZE]
            if len(rows) != BATCH_SIZE or any(row["optimizer_step"] != step for row in rows):
                raise SemanticP50RunnerError("P50 optimizer batch differs from exact stream")
            batch, fibers = _rows_to_batch(
                rows=rows,
                records=validated.train_records,
                states=validated.states,
                collator=collator,
            )
            device_batch = batch.to(model.device)
            optimizer.zero_grad(set_to_none=True)
            try:
                prediction = forward_teacher_successor_batch(model, device_batch, fibers)
                loss = factorized_successor_identity_loss(prediction, device_batch)
            except (ValueError, RuntimeError, SuccessorTrainingError) as error:
                raise SemanticP50RunnerError(
                    f"P50 successor forward failed before update {step + 1}"
                ) from error
            if not bool(torch.isfinite(loss)):
                raise SemanticP50RunnerError(f"nonfinite P50 loss before update {step + 1}")
            family_routes = _route_gradient_evidence(model, prediction, rows, group_field="family")
            cell_routes = _route_gradient_evidence(
                model, prediction, rows, group_field="semantic_cell_id"
            )
            _merge_exposure(family_exposure, family_routes, rows=rows, group_field="family")
            _merge_exposure(cell_exposure, cell_routes, rows=rows, group_field="semantic_cell_id")
            loss.backward()
            for name, parameter in model.named_parameters():
                if name.startswith(TOTAL_HAZARD_PREFIX):
                    if parameter.grad is not None:
                        raise SemanticP50RunnerError(
                            f"frozen hazard parameter received a gradient: {name}"
                        )
                elif parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                    raise SemanticP50RunnerError(
                        f"nonfinite gradient before P50 update {step + 1}: {name}"
                    )
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 10.0)
            if not bool(torch.isfinite(gradient_norm)):
                raise SemanticP50RunnerError("P50 clipped gradient norm is nonfinite")
            optimizer.step()
            _assert_hazard_identity(model, hazard_initial)
            trajectory.append(
                {
                    "optimizer_step": step + 1,
                    "batch_start_stream_index": step * BATCH_SIZE,
                    "batch_end_stream_index_exclusive": (step + 1) * BATCH_SIZE,
                    "productive_canonical_successor_nll_nats": float(loss.detach().cpu()),
                    "pre_clip_gradient_norm": float(gradient_norm.detach().cpu()),
                    "model_state_sha256": state_dict_semantic_sha256(model.state_dict()),
                }
            )
        if len(trajectory) != OPTIMIZER_STEPS:
            raise SemanticP50RunnerError("P50 did not execute exactly 50 updates")

        def verify_exposure(
            observed: Mapping[str, Mapping[str, Any]],
            *,
            reports_key: str,
            identity_key: str,
        ) -> None:
            expected = {
                str(row[identity_key]): (
                    int(row["scheduled_draw_count"]),
                    int(row["optimizer_step_opportunity_count"]),
                )
                for row in validated.prepared[reports_key]
            }
            for identity, evidence in observed.items():
                if (
                    evidence["scheduled_draw_count"],
                    evidence["optimizer_step_opportunity_count"],
                ) != expected[identity] or evidence[
                    "finite_nonzero_gradient_update_count"
                ] < evidence[
                    "minimum_required_finite_nonzero_gradient_update_count"
                ]:
                    raise SemanticP50RunnerError(
                        f"P50 exposure or gradient floor failed for {identity}"
                    )

        verify_exposure(
            family_exposure,
            reports_key="planned_family_exposure",
            identity_key="family",
        )
        verify_exposure(
            cell_exposure,
            reports_key="planned_semantic_cell_exposure",
            identity_key="semantic_cell_id",
        )
        final_evaluations, final_time_rows = _evaluate_validation(
            model=model,
            candidates=validated.validation_rows,
            records=validated.validation_records,
            states=validated.states,
            collator=collator,
        )
        if final_time_rows != validation_time_rows:
            raise SemanticP50RunnerError("pre-update and final validation time streams differ")
        family_nll_checks, cell_nll_checks = _validation_nll_checks(
            scratch=scratch_evaluations,
            final=final_evaluations,
            prepared=validated.prepared,
        )
        _assert_hazard_identity(model, hazard_initial)
        environment = _execution_environment(model)
        validation_time_contract = {
            "schema": VALIDATION_TIME_STREAM_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "algorithm": TIME_ALGORITHM,
            "seed": SEED,
            "order": "canonical_address_sorted_combined_cache_validation_union",
            "rows": validation_time_rows,
            "ordered_validation_time_stream_sha256": _sha(validation_time_rows),
        }
        combined_records = {
            **validated.train_records,
            **validated.validation_records,
        }
        identity = {
            "prepared_recipe_sha256": validated.prepared["prepared_recipe_sha256"],
            "recipe_policy_sha256": validated.prepared["recipe_policy_sha256"],
            "ordered_address_stream_sha256": validated.prepared["ordered_address_stream_sha256"],
            "ordered_training_stream_sha256": validated.prepared["ordered_training_stream_sha256"],
            "requested_address_union_sha256": validated.prepared["requested_address_union_sha256"],
            "cache_completion_sha256": inputs.cache.completion["completion_sha256"],
            "cache_manifest_sha256": inputs.cache.manifest["manifest_sha256"],
            "cache_validation_contract_sha256": inputs.cache.manifest["validation_contract"][
                "validation_contract_sha256"
            ],
            "validation_baseline_completion_sha256": inputs.validation_baseline.completion[
                "completion_sha256"
            ],
            "validation_baseline_result_sha256": inputs.validation_baseline.result["result_sha256"],
            "validation_baseline_evaluation_records_sha256": _sha(
                inputs.validation_baseline.result["evaluation_records"]
            ),
            "validation_time_stream_sha256": validation_time_contract[
                "ordered_validation_time_stream_sha256"
            ],
            "exact_state_inventory_sha256": _sha(
                [
                    {
                        "address_sha256": address_sha,
                        "source_state_sha256": combined_records[address_sha].source_state_sha256,
                    }
                    for address_sha in sorted(validated.states)
                ]
            ),
            "initial_model_state_sha256": runtime.initial_model_state_sha256,
            "process_identity_sha256": runtime.process_identity_sha256,
            "runner_implementation_sha256": semantic_p50_runner_implementation_sha256(),
            "execution_environment_sha256": environment["environment_sha256"],
        }
        checkpoint_configuration = {
            "optimization_policy": dict(_EXPECTED_RECIPE),
            "optimization_policy_sha256": _sha(_EXPECTED_RECIPE),
            "optimizer_configuration": optimizer_configuration,
            "optimizer_configuration_sha256": _sha(optimizer_configuration),
            "validation_time_contract": validation_time_contract,
            "scratch_architecture": runtime.architecture.as_payload(),
        }
        terminal_rng = _rng_state()
        checkpoint = _checkpoint_payload(
            identity=identity,
            configuration=checkpoint_configuration,
            model=model,
            optimizer=optimizer,
            rng_state=terminal_rng,
            hazard_initial_sha256=hazard_initial_sha,
        )
        validate_semantic_p50_checkpoint_payload(checkpoint)
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        elapsed_seconds = time.perf_counter() - started_at
        if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0.0:
            raise SemanticP50RunnerError("P50 wall-clock observation is invalid")
        runtime_observations = {
            "wall_time_seconds": elapsed_seconds,
            "optimizer_updates_per_second": OPTIMIZER_STEPS / elapsed_seconds,
            "scheduled_examples_per_second": SCHEDULED_EXAMPLES / elapsed_seconds,
            "gpu_peak_memory_bytes": (
                torch.cuda.max_memory_allocated(model.device)
                if model.device.type == "cuda"
                else None
            ),
            "device_type": model.device.type,
            "dtype": "float32",
            "batch_size": BATCH_SIZE,
        }
        validation_cell_counts = Counter(item.semantic_cell_id for item in scratch_evaluations)
        required_validation_cells = tuple(
            validated.prepared["validation_contract"]["required_nonempty_semantic_cells"]
        )
        validation_cell_coverage = {
            "required_semantic_cells": list(required_validation_cells),
            "observed_semantic_cell_counts": dict(sorted(validation_cell_counts.items())),
            "extra_observed_semantic_cells_not_gated": sorted(
                set(validation_cell_counts) - set(required_validation_cells)
            ),
            "all_combined_cache_validation_rows_evaluated": True,
        }
        result_body = {
            "schema": RESULT_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "status": RESULT_STATUS,
            **NO_AUTHORITY,
            "identity": identity,
            "optimization": dict(_EXPECTED_RECIPE),
            "optimizer_configuration": optimizer_configuration,
            "execution_environment": environment,
            "runtime_observations": runtime_observations,
            "validation_cell_coverage": validation_cell_coverage,
            "completed_optimizer_steps": OPTIMIZER_STEPS,
            "scheduled_example_count": SCHEDULED_EXAMPLES,
            "resume_count": 0,
            "learning_demonstrated": False,
            "scientific_interpretation": ("NON_REGRESSION_ONLY_DOES_NOT_ESTABLISH_LEARNING"),
            "next_stage_authorized": None,
            "hazard_parameters_frozen_excluded_and_unchanged": True,
            "validation_time_contract": validation_time_contract,
            "scratch_validation_evaluations": [item.as_payload() for item in scratch_evaluations],
            "final_validation_evaluations": [item.as_payload() for item in final_evaluations],
            "family_validation_nll_checks": family_nll_checks,
            "semantic_cell_validation_nll_checks": cell_nll_checks,
            "family_exposure_and_gradient_evidence": family_exposure,
            "semantic_cell_exposure_and_gradient_evidence": cell_exposure,
            "trajectory": trajectory,
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "terminal_model_state_sha256": checkpoint["model_state_sha256"],
        }
        result = {**result_body, "result_sha256": _sha(result_body)}
        model.load_state_dict(initial_model_state, strict=True)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(initial_requires_grad[name])
        model.train(initial_training)
        return SemanticP50RunArtifacts(
            scratch_validation_evaluations=scratch_evaluations,
            final_validation_evaluations=final_evaluations,
            checkpoint=checkpoint,
            result=result,
        )
    except Exception:
        model.load_state_dict(initial_model_state, strict=True)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(initial_requires_grad[name])
        model.train(initial_training)
        raise


__all__ = [
    "CHECKPOINT_SCHEMA",
    "RESULT_SCHEMA",
    "RUN_CHECKPOINT_FILENAME",
    "RUN_COMPLETION_FILENAME",
    "RUN_RESULT_FILENAME",
    "SemanticP50RunArtifacts",
    "SemanticP50RunnerError",
    "SemanticP50RuntimeInputs",
    "VerifiedSemanticP50RunArtifacts",
    "evaluate_semantic_p50_scratch_validation",
    "open_semantic_p50_run_artifacts",
    "publish_semantic_p50_run_artifacts",
    "run_semantic_p50",
    "semantic_p50_runner_implementation_sha256",
    "semantic_p50_validation_time_hex",
    "validate_semantic_p50_checkpoint_payload",
]
