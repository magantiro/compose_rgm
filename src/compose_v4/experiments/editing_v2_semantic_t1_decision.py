"""Fail-closed semantic Editing-V2 T1 result and decision artifacts.

The prospective capacity policy defines the experiment before optimization.
This module records the result after optimization and derives the only T1
decision that may authorize the bounded P50 pilot.  It deliberately does not
implement optimization or launch infrastructure.

``build_semantic_t1_capacity_result`` is the explicit adapter boundary for a
runner.  The runner may use different internal metric names or tensor layouts;
it must translate them into the small artifact vocabulary documented by that
function.  This keeps the gate schema independent from trainer implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from compose_v4.experiments.editing_v2_semantic_development_cell_roles import (
    load_semantic_development_cell_roles,
)
from compose_v4.experiments.editing_v2_semantic_gate_zero import (
    EVIDENCE_SCHEMA as GATE_ZERO_EVIDENCE_SCHEMA,
)
from compose_v4.experiments.editing_v2_semantic_gate_zero import (
    EVIDENCE_SCHEMA_VERSION as GATE_ZERO_EVIDENCE_SCHEMA_VERSION,
)
from compose_v4.experiments.editing_v2_semantic_gate_zero import (
    EVIDENCE_STATUS as GATE_ZERO_EVIDENCE_STATUS,
)
from compose_v4.experiments.editing_v2_semantic_t1_capacity_policy import (
    SemanticT1CapacityPolicyError,
    load_semantic_t1_capacity_policy,
)
from compose_v4.experiments.editing_v2_semantic_t1_checkpoint import (
    SemanticT1SelectedCheckpointError,
    validate_semantic_t1_selected_checkpoint,
)
from compose_v4.experiments.editing_v2_semantic_t1_prepared_inputs import (
    NO_AUTHORITY as PREPARED_INPUT_NO_AUTHORITY,
)
from compose_v4.experiments.editing_v2_semantic_t1_prepared_inputs import (
    SCHEMA as PREPARED_INPUT_SCHEMA,
)
from compose_v4.experiments.editing_v2_semantic_t1_prepared_inputs import (
    SCHEMA_VERSION as PREPARED_INPUT_SCHEMA_VERSION,
)
from compose_v4.experiments.editing_v2_semantic_t1_prepared_inputs import (
    STATUS as PREPARED_INPUT_STATUS,
)
from compose_v4.experiments.editing_v2_semantic_t1_prepared_inputs import (
    SemanticT1PreparedInputError,
    semantic_t1_prepared_input_implementation_sha256,
    validate_semantic_t1_prepared_inputs,
)
from compose_v4.experiments.editing_v2_semantic_t1_successor_cache import (
    SemanticT1SuccessorCacheError,
    validate_semantic_t1_successor_cache_completion,
)
from compose_v4.experiments.successor_micro_overfit import (
    RINGCORE_EDITING_FAMILIES,
)

CAPACITY_POLICY_RELATIVE_PATH = "configs/editing_v2_semantic_t1_capacity_policy_v1.json"
CAPACITY_POLICY_FILE_SHA256 = "9bb8fce6ee1d3b0110cc19f16affe6b6fc17f873e84a834315ef964838e65862"
CAPACITY_POLICY_SHA256 = "2186dbd9436f39b3caffafd421710f0c82de14b7da9e4c5ab94a252525a46d1e"

RESULT_FILENAME = "SEMANTIC_T1_CAPACITY_RESULT.json"
COMPLETION_FILENAME = "SEMANTIC_T1_CAPACITY_COMPLETE.json"
DECISION_FILENAME = "SEMANTIC_T1_CAPACITY_DECISION.json"

RESULT_SCHEMA = "compose.editing_v2.semantic_t1_capacity_result"
RESULT_SCHEMA_VERSION = 2
RESULT_STATUS = "COMPLETE_UNIQUE_STATE_CAPACITY_RESULT_NO_DOWNSTREAM_AUTHORITY"
COMPLETION_SCHEMA = "compose.editing_v2.semantic_t1_capacity_completion"
COMPLETION_SCHEMA_VERSION = 2
COMPLETION_STATUS = "COMPLETE_SEMANTIC_T1_ARTIFACTS_NO_DOWNSTREAM_AUTHORITY"
DECISION_SCHEMA = "compose.editing_v2.semantic_t1_capacity_decision"
DECISION_SCHEMA_VERSION = 2
DECISION_GO_STATUS = "GO_BOUNDED_P50_SEMANTIC_T1_CAPACITY"
DECISION_NO_GO_STATUS = "NO_GO_BOUNDED_P50_SEMANTIC_T1_CAPACITY"

_NO_AUTHORITY = {
    "training_authorized": False,
    "bounded_p50_authorized": False,
    "long_training_authorized": False,
    "checkpoint_selection_authorized": False,
    "final_test_selection_authorized": False,
}
_GATE_ZERO_NO_AUTHORITY = {
    **_NO_AUTHORITY,
    "gate_zero_authorized": False,
    "t1_authorized": False,
}
_RESULT_PROVENANCE_FIELDS = {
    "capacity_policy_file_sha256",
    "capacity_policy_sha256",
    "cell_role_policy_sha256",
    "cache_completion_file_sha256",
    "cache_completion_sha256",
    "cache_manifest_file_sha256",
    "cache_manifest_sha256",
    "cache_run_identity_sha256",
    "cache_build_identity_sha256",
    "cache_source_revision_sha256",
    "panel_completion_sha256",
    "panel_artifact_sha256",
    "panel_entry_inventory_sha256",
    "panel_entry_metadata_sha256",
    "decision_source_inventory_sha256",
    "gate_zero_evidence_file_sha256",
    "gate_zero_evidence_sha256",
    "initial_model_state_sha256",
    "panel_entry_binding_count",
    "prepared_input_file_sha256",
    "prepared_input_artifact_sha256",
    "prepared_input_implementation_sha256",
    "runner_implementation_sha256",
    "runner_source_revision_sha256",
    "execution_environment",
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
_RUNNER_IMPLEMENTATION_SOURCES = (
    "src/compose_v4/experiments/editing_v2_semantic_t1_capacity_runner.py",
    "src/compose_v4/experiments/editing_v2_semantic_t1_checkpoint.py",
    "src/compose_v4/experiments/editing_v2_semantic_t1_decision.py",
    "src/compose_v4/experiments/editing_v2_semantic_t1_prepared_inputs.py",
    "src/compose_v4/experiments/factorized_successor_training.py",
    "src/compose_v4/experiments/factorized_mark_conditional.py",
    "src/compose_v4/model/factorized_tracelet_rate_model.py",
    "configs/editing_v2_semantic_t1_capacity_policy_v1.json",
    "configs/editing_v2_semantic_development_cell_roles_v1.json",
    "src/compose_v4/experiments/editing_v2_semantic_development_cell_roles.py",
)
_RUN_INTEGRITY_FIELDS = {
    "optimizer_steps_completed",
    "evaluation_steps",
    "selected_step",
    "termination_reason",
    "resume_requested",
    "resume_count",
    "abort_triggered",
    "abort_reasons",
    "nonfinite_event_count",
    "unsupported_teacher_count",
    "missing_candidate_count",
    "provenance_drift_detected",
    "stream_identity_drift_detected",
    "deterministic_algorithms_enabled",
    "dtype",
    "mixed_precision",
    "hazard_included",
    "address_stream_sha256",
    "selected_model_state_sha256",
    "selected_checkpoint_file_sha256",
    "optimizer_state_sha256",
}
_TRAJECTORY_FIELDS = {
    "step",
    "minimum_entry_teacher_successor_probability",
    "mean_entry_canonical_successor_nll",
    "model_state_sha256",
}
_ENTRY_FIELDS = {
    "panel_entry_sha256",
    "family",
    "semantic_cell_id",
    "teacher_successor_probability",
    "canonical_successor_nll",
    "teacher_successor_rank",
    "teacher_successor_top1",
}
_CELL_FIELDS = {
    "family",
    "semantic_cell_id",
    "entry_count",
    "teacher_successor_top1",
    "teacher_successor_probability",
    "canonical_successor_nll",
}
_FAMILY_FIELDS = {
    "family",
    "cell_count",
    "entry_count",
    "teacher_successor_top1",
    "teacher_successor_probability",
    "canonical_successor_nll",
}
_GRADIENT_FIELDS = {
    "family",
    "family_gate_gradient_finite",
    "family_gate_cumulative_l2",
    "family_gate_nonzero_update_steps",
    "action_route_gradient_finite",
    "action_route_cumulative_l2",
    "action_route_nonzero_update_steps",
}


class SemanticT1DecisionError(ValueError):
    """A semantic T1 result chain is malformed, stale, or not P50-ready."""


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SemanticT1DecisionError(
            "semantic T1 artifact is not finite canonical JSON"
        ) from error
    return encoded + (b"\n" if newline else b"")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise SemanticT1DecisionError(f"semantic T1 artifact is absent: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha(value: object, *, field: str) -> str:
    if not _is_sha(value):
        raise SemanticT1DecisionError(f"{field} must be a lowercase SHA-256 digest")
    return str(value)


def semantic_t1_runner_implementation_sha256(*, repo_root: Path) -> str:
    root = Path(repo_root).resolve()
    digest = hashlib.sha256()
    for relative in _RUNNER_IMPLEMENTATION_SOURCES:
        path = root / relative
        if not path.is_file():
            raise SemanticT1DecisionError(
                f"semantic T1 runner implementation source is absent: {relative}"
            )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_execution_environment(
    value: object,
    *,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    environment = dict(
        _exact_mapping(
            value,
            _EXECUTION_ENVIRONMENT_FIELDS,
            field="result.provenance.execution_environment",
        )
    )
    body = dict(environment)
    supplied_sha256 = body.pop("environment_sha256")
    required_strings = (
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
            for name in required_strings
        )
        or type(environment["batch_size"]) is not int
        or environment["batch_size"] <= 0
        or supplied_sha256 != _sha(body)
    ):
        raise SemanticT1DecisionError(
            "semantic T1 execution environment disagrees with the frozen policy"
        )
    if policy is not None and (
        environment["accelerator_class"] != policy["optimization"]["accelerator_class"]
        or environment["dtype"] != policy["optimization"]["dtype"]
        or environment["mixed_precision"] is not policy["optimization"]["mixed_precision"]
        or environment["batch_size"] != policy["optimization"]["batch_size"]
    ):
        raise SemanticT1DecisionError(
            "semantic T1 execution environment disagrees with the frozen policy"
        )
    for name in ("device_capability", "cuda_version", "cudnn_version"):
        item = environment[name]
        if environment["accelerator_class"] == "gpu":
            if not isinstance(item, str) or not item:
                raise SemanticT1DecisionError(f"semantic T1 GPU execution environment lacks {name}")
        elif item is not None:
            raise SemanticT1DecisionError(
                f"semantic T1 non-GPU execution environment must null {name}"
            )
    return environment


def build_semantic_t1_execution_environment(
    *,
    hardware_class: str,
    device_name: str,
    device_capability: str | None,
    accelerator_class: str,
    dtype: str,
    mixed_precision: bool,
    batch_size: int,
    python_version: str,
    torch_version: str,
    cuda_version: str | None,
    cudnn_version: str | None,
    rdkit_version: str,
    numpy_version: str,
) -> dict[str, Any]:
    """Create a self-hashed runtime receipt without inventing version values."""

    body: dict[str, Any] = {
        "hardware_class": hardware_class,
        "device_name": device_name,
        "device_capability": device_capability,
        "accelerator_class": accelerator_class,
        "dtype": dtype,
        "mixed_precision": mixed_precision,
        "batch_size": batch_size,
        "python_version": python_version,
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "cudnn_version": cudnn_version,
        "rdkit_version": rdkit_version,
        "numpy_version": numpy_version,
    }
    return {**body, "environment_sha256": _sha(body)}


def _exact_mapping(
    value: object,
    fields: set[str],
    *,
    field: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        observed = set(value) if isinstance(value, Mapping) else set()
        raise SemanticT1DecisionError(
            f"{field} has missing or unknown fields: "
            f"missing={sorted(fields - observed)}, unknown={sorted(observed - fields)}"
        )
    return value


def _finite_float(value: object, *, field: str, minimum: float | None = None) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise SemanticT1DecisionError(f"{field} must be finite")
    resolved = float(value)
    if minimum is not None and resolved < minimum:
        raise SemanticT1DecisionError(f"{field} must be at least {minimum}")
    return resolved


def _finite_probability(value: object, *, field: str) -> float:
    probability = _finite_float(value, field=field)
    if not 0.0 <= probability <= 1.0:
        raise SemanticT1DecisionError(f"{field} must lie in [0, 1]")
    return probability


def _load_canonical(path: Path, *, field: str) -> tuple[dict[str, Any], bytes]:
    source = Path(path)
    if not source.is_file():
        raise SemanticT1DecisionError(f"{field} is absent: {source}")
    raw = source.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SemanticT1DecisionError(f"{field} is not readable JSON: {source}") from error
    if not isinstance(payload, dict) or raw != _canonical_bytes(payload, newline=True):
        raise SemanticT1DecisionError(f"{field} is not canonical newline-terminated JSON")
    return payload, raw


def _relative_sibling(root: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise SemanticT1DecisionError(f"{field} must be a nonempty relative path")
    candidate = (root / value).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise SemanticT1DecisionError(f"{field} escapes the semantic T1 artifact directory")
    return candidate


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _metric_summary(entries: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return {
        "teacher_successor_top1": _mean(
            [1.0 if entry["teacher_successor_top1"] else 0.0 for entry in entries]
        ),
        "teacher_successor_probability": _mean(
            [float(entry["teacher_successor_probability"]) for entry in entries]
        ),
        "canonical_successor_nll": _mean(
            [float(entry["canonical_successor_nll"]) for entry in entries]
        ),
    }


def _panel_entry_metadata(
    entries: object,
    *,
    family_field: str,
    cell_field: str,
    field: str,
) -> list[dict[str, str]]:
    """Return the canonical entry-to-family/cell projection for lineage checks."""

    if not isinstance(entries, list) or not entries:
        raise SemanticT1DecisionError(f"{field} must contain panel entries")
    projected: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in entries:
        if not isinstance(raw, Mapping):
            raise SemanticT1DecisionError(f"{field} entry must be an object")
        panel_id = _require_sha(
            raw.get("panel_entry_sha256"),
            field=f"{field}.panel_entry_sha256",
        )
        family = raw.get(family_field)
        cell = raw.get(cell_field)
        if (
            panel_id in seen
            or not isinstance(family, str)
            or not family
            or not isinstance(cell, str)
            or not cell
        ):
            raise SemanticT1DecisionError(
                f"{field} entry identity, family, or semantic cell is invalid"
            )
        seen.add(panel_id)
        projected.append(
            {
                "panel_entry_sha256": panel_id,
                "family": family,
                "semantic_cell_id": cell,
            }
        )
    return sorted(projected, key=lambda item: item["panel_entry_sha256"])


def _derived_aggregates(
    entries: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_cell: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    roles = load_semantic_development_cell_roles()
    for entry in entries:
        family = str(entry["family"])
        cell = str(entry["semantic_cell_id"])
        if roles.role_for(cell) != "required_editing":
            raise SemanticT1DecisionError(
                f"semantic T1 entry is outside the required editing role: {cell}"
            )
        by_cell[(family, cell)].append(entry)
        by_family[family].append(entry)
    observed_cells = {cell for _family, cell in by_cell}
    missing_cells = sorted(roles.required_cell_set - observed_cells)
    unexpected_cells = sorted(observed_cells - roles.required_cell_set)
    if missing_cells or unexpected_cells:
        raise SemanticT1DecisionError(
            "semantic T1 entries must exactly cover all frozen required editing cells; "
            f"missing={missing_cells}, unexpected={unexpected_cells}"
        )
    missing_families = [family for family in RINGCORE_EDITING_FAMILIES if not by_family[family]]
    if missing_families:
        raise SemanticT1DecisionError(
            f"semantic T1 entries lack required families: {missing_families}"
        )
    cells = [
        {
            "family": family,
            "semantic_cell_id": cell,
            "entry_count": len(group),
            **_metric_summary(group),
        }
        for (family, cell), group in sorted(by_cell.items())
    ]
    family_cell_counts: dict[str, int] = defaultdict(int)
    for family, _cell in by_cell:
        family_cell_counts[family] += 1
    families = [
        {
            "family": family,
            "cell_count": family_cell_counts[family],
            "entry_count": len(by_family[family]),
            **_metric_summary(by_family[family]),
        }
        for family in RINGCORE_EDITING_FAMILIES
    ]
    return cells, families


def build_semantic_t1_capacity_result(
    *,
    provenance: Mapping[str, Any],
    run_integrity: Mapping[str, Any],
    evaluation_trajectory: Sequence[Mapping[str, Any]],
    entry_metrics: Sequence[Mapping[str, Any]],
    gradient_evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Adapt runner outputs into one self-hashed joint-model result.

    ``entry_metrics`` is the exact selected-state panel projection.  It names
    each panel entry, its frozen family and semantic cell, teacher-successor
    probability, corresponding NLL, and rank.  ``evaluation_trajectory``
    supplies the frozen checkpoint-selection summaries for every state from
    step zero through the terminal state.  ``gradient_evidence`` supplies one
    ordered family-gate and action-route receipt per Active8 family.  These are
    artifact field names, not required internal runner variable names.
    """

    entries = [dict(entry) for entry in entry_metrics]
    cells, families = _derived_aggregates(entries)
    body: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": RESULT_STATUS,
        **_NO_AUTHORITY,
        "objective_unit": "exact_source_frozen_time_canonical_successor",
        "panel_kind": "unique_state_single_target_canonical_successor_capacity",
        "repeated_state_empirical_law_evaluated": False,
        "provenance": dict(provenance),
        "run_integrity": dict(run_integrity),
        "evaluation_trajectory": [dict(item) for item in evaluation_trajectory],
        "entry_metrics": entries,
        "cell_metrics": cells,
        "family_metrics": families,
        "gradient_evidence": [dict(item) for item in gradient_evidence],
    }
    result = {**body, "result_sha256": _sha(body)}
    return validate_semantic_t1_capacity_result(result)


def validate_semantic_t1_capacity_result(value: object) -> dict[str, Any]:
    """Validate raw result identities and internally recompute all aggregates."""

    fields = {
        "schema",
        "schema_version",
        "status",
        *_NO_AUTHORITY,
        "objective_unit",
        "panel_kind",
        "repeated_state_empirical_law_evaluated",
        "provenance",
        "run_integrity",
        "evaluation_trajectory",
        "entry_metrics",
        "cell_metrics",
        "family_metrics",
        "gradient_evidence",
        "result_sha256",
    }
    result = dict(_exact_mapping(value, fields, field="semantic T1 result"))
    body = dict(result)
    supplied_sha = body.pop("result_sha256")
    if (
        result["schema"] != RESULT_SCHEMA
        or result["schema_version"] != RESULT_SCHEMA_VERSION
        or result["status"] != RESULT_STATUS
        or supplied_sha != _sha(body)
        or any(result[name] is not expected for name, expected in _NO_AUTHORITY.items())
        or result["objective_unit"] != "exact_source_frozen_time_canonical_successor"
        or result["panel_kind"] != "unique_state_single_target_canonical_successor_capacity"
        or result["repeated_state_empirical_law_evaluated"] is not False
    ):
        raise SemanticT1DecisionError("semantic T1 result identity or authority disagrees")

    provenance = _exact_mapping(
        result["provenance"], _RESULT_PROVENANCE_FIELDS, field="result.provenance"
    )
    for name, item in provenance.items():
        if name == "panel_entry_binding_count":
            if type(item) is not int or item <= 0:
                raise SemanticT1DecisionError(
                    "result.provenance.panel_entry_binding_count must be positive"
                )
        elif name == "execution_environment":
            _validate_execution_environment(item)
        else:
            _require_sha(item, field=f"result.provenance.{name}")

    integrity = _exact_mapping(
        result["run_integrity"], _RUN_INTEGRITY_FIELDS, field="result.run_integrity"
    )
    steps = integrity["optimizer_steps_completed"]
    selected_step = integrity["selected_step"]
    evaluation_steps = integrity["evaluation_steps"]
    if (
        type(steps) is not int
        or not 1 <= steps <= 500
        or type(selected_step) is not int
        or not isinstance(evaluation_steps, list)
        or evaluation_steps != list(range(steps + 1))
        or selected_step not in evaluation_steps
        or integrity["termination_reason"]
        not in {
            "all_thresholds_passed_early",
            "maximum_optimizer_steps_reached",
            "aborted",
        }
        or type(integrity["resume_count"]) is not int
        or integrity["resume_count"] < 0
        or not isinstance(integrity["abort_reasons"], list)
        or any(not isinstance(reason, str) or not reason for reason in integrity["abort_reasons"])
        or any(
            type(integrity[name]) is not int or integrity[name] < 0
            for name in (
                "nonfinite_event_count",
                "unsupported_teacher_count",
                "missing_candidate_count",
            )
        )
        or any(
            type(integrity[name]) is not bool
            for name in (
                "resume_requested",
                "abort_triggered",
                "provenance_drift_detected",
                "stream_identity_drift_detected",
                "deterministic_algorithms_enabled",
                "mixed_precision",
                "hazard_included",
            )
        )
        or integrity["dtype"] != "float32"
    ):
        raise SemanticT1DecisionError("semantic T1 run-integrity structure disagrees")
    for name in (
        "address_stream_sha256",
        "selected_model_state_sha256",
        "selected_checkpoint_file_sha256",
        "optimizer_state_sha256",
    ):
        _require_sha(integrity[name], field=f"result.run_integrity.{name}")

    trajectory = result["evaluation_trajectory"]
    if not isinstance(trajectory, list) or len(trajectory) != steps + 1:
        raise SemanticT1DecisionError("semantic T1 evaluation trajectory is incomplete")
    normalized_trajectory: list[tuple[int, float, float, str]] = []
    for index, item in enumerate(trajectory):
        row = _exact_mapping(item, _TRAJECTORY_FIELDS, field="evaluation trajectory row")
        if row["step"] != index:
            raise SemanticT1DecisionError(
                "semantic T1 trajectory steps are not complete and ordered"
            )
        minimum = _finite_probability(
            row["minimum_entry_teacher_successor_probability"],
            field="trajectory.minimum_entry_teacher_successor_probability",
        )
        mean_nll = _finite_float(
            row["mean_entry_canonical_successor_nll"],
            field="trajectory.mean_entry_canonical_successor_nll",
            minimum=0.0,
        )
        state_sha = _require_sha(row["model_state_sha256"], field="trajectory.model_state_sha256")
        normalized_trajectory.append((index, minimum, mean_nll, state_sha))
    expected_selected = min(
        normalized_trajectory,
        key=lambda row: (-row[1], row[2], row[0]),
    )
    if (
        selected_step != expected_selected[0]
        or integrity["selected_model_state_sha256"] != expected_selected[3]
    ):
        raise SemanticT1DecisionError(
            "selected semantic T1 state violates the frozen checkpoint rule"
        )

    entries = result["entry_metrics"]
    if (
        not isinstance(entries, list)
        or len(entries) != provenance["panel_entry_binding_count"]
        or not entries
    ):
        raise SemanticT1DecisionError("semantic T1 entry census disagrees with the cache")
    normalized_entries: list[dict[str, Any]] = []
    seen_entries: set[str] = set()
    for item in entries:
        entry = dict(_exact_mapping(item, _ENTRY_FIELDS, field="entry metric"))
        identifier = _require_sha(entry["panel_entry_sha256"], field="entry.panel_entry_sha256")
        if identifier in seen_entries:
            raise SemanticT1DecisionError("semantic T1 entry metrics repeat a panel entry")
        seen_entries.add(identifier)
        if (
            entry["family"] not in RINGCORE_EDITING_FAMILIES
            or not isinstance(entry["semantic_cell_id"], str)
            or not entry["semantic_cell_id"]
            or type(entry["teacher_successor_rank"]) is not int
            or entry["teacher_successor_rank"] <= 0
            or type(entry["teacher_successor_top1"]) is not bool
            or entry["teacher_successor_top1"] is not (entry["teacher_successor_rank"] == 1)
        ):
            raise SemanticT1DecisionError("semantic T1 entry identity or rank is invalid")
        probability = _finite_probability(
            entry["teacher_successor_probability"],
            field="entry.teacher_successor_probability",
        )
        nll = _finite_float(
            entry["canonical_successor_nll"],
            field="entry.canonical_successor_nll",
            minimum=0.0,
        )
        expected_nll = math.inf if probability == 0.0 else -math.log(probability)
        if not math.isclose(nll, expected_nll, rel_tol=1e-10, abs_tol=1e-12):
            raise SemanticT1DecisionError(
                "entry canonical-successor NLL disagrees with teacher probability"
            )
        normalized_entries.append(entry)
    if entries != sorted(
        normalized_entries,
        key=lambda item: (
            RINGCORE_EDITING_FAMILIES.index(item["family"]),
            item["semantic_cell_id"],
            item["panel_entry_sha256"],
        ),
    ):
        raise SemanticT1DecisionError("semantic T1 entry metrics are not canonical")
    entry_metadata = _panel_entry_metadata(
        entries,
        family_field="family",
        cell_field="semantic_cell_id",
        field="result.entry_metrics",
    )
    if provenance["panel_entry_inventory_sha256"] != _sha(
        sorted(item["panel_entry_sha256"] for item in entry_metadata)
    ) or provenance["panel_entry_metadata_sha256"] != _sha(entry_metadata):
        raise SemanticT1DecisionError(
            "semantic T1 result entry inventory or metadata differs from bound provenance"
        )
    selected_trajectory = normalized_trajectory[selected_step]
    selected_minimum = min(float(entry["teacher_successor_probability"]) for entry in entries)
    selected_mean_nll = _mean([float(entry["canonical_successor_nll"]) for entry in entries])
    if not (
        math.isclose(selected_trajectory[1], selected_minimum, rel_tol=1e-12, abs_tol=1e-12)
        and math.isclose(selected_trajectory[2], selected_mean_nll, rel_tol=1e-12, abs_tol=1e-12)
    ):
        raise SemanticT1DecisionError(
            "selected trajectory summary disagrees with exact entry metrics"
        )

    expected_cells, expected_families = _derived_aggregates(entries)
    for rows, expected, fields, name in (
        (result["cell_metrics"], expected_cells, _CELL_FIELDS, "cell metrics"),
        (result["family_metrics"], expected_families, _FAMILY_FIELDS, "family metrics"),
    ):
        if not isinstance(rows, list) or len(rows) != len(expected):
            raise SemanticT1DecisionError(f"semantic T1 {name} census disagrees")
        for row, derived in zip(rows, expected, strict=True):
            _exact_mapping(row, fields, field=name)
            if set(row) != set(derived):
                raise SemanticT1DecisionError(f"semantic T1 {name} fields disagree")
            for key, expected_value in derived.items():
                observed = row[key]
                if isinstance(expected_value, float):
                    if not math.isclose(
                        _finite_float(observed, field=f"{name}.{key}"),
                        expected_value,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    ):
                        raise SemanticT1DecisionError(
                            f"semantic T1 {name} do not recompute from entries"
                        )
                elif observed != expected_value:
                    raise SemanticT1DecisionError(
                        f"semantic T1 {name} do not recompute from entries"
                    )

    gradients = result["gradient_evidence"]
    if not isinstance(gradients, list) or [
        row.get("family") for row in gradients if isinstance(row, Mapping)
    ] != list(RINGCORE_EDITING_FAMILIES):
        raise SemanticT1DecisionError("gradient evidence must exactly cover ordered Active8")
    for item in gradients:
        row = _exact_mapping(item, _GRADIENT_FIELDS, field="gradient evidence")
        for name in ("family_gate_gradient_finite", "action_route_gradient_finite"):
            if type(row[name]) is not bool:
                raise SemanticT1DecisionError(f"gradient evidence {name} must be Boolean")
        for name in ("family_gate_cumulative_l2", "action_route_cumulative_l2"):
            _finite_float(row[name], field=f"gradient evidence.{name}", minimum=0.0)
        for name in (
            "family_gate_nonzero_update_steps",
            "action_route_nonzero_update_steps",
        ):
            if type(row[name]) is not int or not 0 <= row[name] <= steps:
                raise SemanticT1DecisionError(f"gradient evidence {name} is invalid")
    return result


def _validate_policy(repo_root: Path) -> tuple[dict[str, Any], str]:
    path = Path(repo_root).resolve() / CAPACITY_POLICY_RELATIVE_PATH
    if _file_sha(path) != CAPACITY_POLICY_FILE_SHA256:
        raise SemanticT1DecisionError("semantic T1 capacity policy physical hash disagrees")
    try:
        policy = load_semantic_t1_capacity_policy(path)
    except SemanticT1CapacityPolicyError as error:
        raise SemanticT1DecisionError(f"semantic T1 capacity policy is invalid: {error}") from error
    if policy["policy_sha256"] != CAPACITY_POLICY_SHA256:
        raise SemanticT1DecisionError("semantic T1 capacity policy semantic hash disagrees")
    return policy, CAPACITY_POLICY_FILE_SHA256


def _validate_gate_zero_receipt(
    path: Path,
) -> tuple[dict[str, Any], str]:
    evidence, raw = _load_canonical(path, field="semantic Gate0 evidence")
    body = dict(evidence)
    supplied_sha = body.pop("evidence_sha256", None)
    if (
        evidence.get("schema") != GATE_ZERO_EVIDENCE_SCHEMA
        or evidence.get("schema_version") != GATE_ZERO_EVIDENCE_SCHEMA_VERSION
        or evidence.get("status") != GATE_ZERO_EVIDENCE_STATUS
        or evidence.get("structural_result") != "PASS"
        or supplied_sha != _sha(body)
        or any(
            evidence.get(name) is not expected for name, expected in _GATE_ZERO_NO_AUTHORITY.items()
        )
    ):
        raise SemanticT1DecisionError("semantic Gate0 evidence is not an exact structural PASS")
    _require_sha(
        evidence.get("decision_source_inventory_sha256"),
        field="Gate0 decision_source_inventory_sha256",
    )
    runtime = evidence.get("model_runtime_identity")
    if not isinstance(runtime, Mapping):
        raise SemanticT1DecisionError("semantic Gate0 model runtime identity is absent")
    _require_sha(
        runtime.get("initial_model_state_sha256"),
        field="Gate0 initial_model_state_sha256",
    )
    return evidence, hashlib.sha256(raw).hexdigest()


def validate_semantic_gate_zero_evidence_receipt(
    path: Path,
    *,
    expected_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the exact semantic Gate0 receipt consumed by T1 and P50."""

    evidence, observed_file_sha256 = _validate_gate_zero_receipt(path)
    if expected_file_sha256 is not None and observed_file_sha256 != _require_sha(
        expected_file_sha256,
        field="expected semantic Gate0 evidence file SHA-256",
    ):
        raise SemanticT1DecisionError("semantic Gate0 evidence physical hash disagrees")
    return evidence


def _validate_prepared_input_identity(
    path: Path,
    *,
    policy: Mapping[str, Any],
    cache: Mapping[str, Any],
    repo_root: Path,
) -> tuple[dict[str, Any], str]:
    """Bind canonical prepared bytes without duplicating its full decoder."""

    artifact, raw = _load_canonical(path, field="semantic T1 prepared inputs")
    body = dict(artifact)
    supplied_sha256 = body.pop("artifact_sha256", None)
    implementation_sha256 = semantic_t1_prepared_input_implementation_sha256(repo_root=repo_root)
    if (
        artifact.get("schema") != PREPARED_INPUT_SCHEMA
        or artifact.get("schema_version") != PREPARED_INPUT_SCHEMA_VERSION
        or artifact.get("status") != PREPARED_INPUT_STATUS
        or supplied_sha256 != _sha(body)
        or artifact.get("capacity_policy_sha256") != policy["policy_sha256"]
        or artifact.get("cache_completion_sha256") != cache["completion_sha256"]
        or artifact.get("cache_manifest_sha256") != cache["manifest_sha256"]
        or artifact.get("panel_artifact_sha256") != cache["panel_artifact_sha256"]
        or artifact.get("decision_source_inventory_sha256")
        != cache["decision_source_inventory_sha256"]
        or artifact.get("initial_model_state_sha256") != cache["initial_model_state_sha256"]
        or artifact.get("cache_source_revision_sha256") != cache["source_revision_sha256"]
        or artifact.get("implementation_sha256") != implementation_sha256
        or any(
            artifact.get(name) is not expected
            for name, expected in PREPARED_INPUT_NO_AUTHORITY.items()
        )
    ):
        raise SemanticT1DecisionError(
            "semantic T1 prepared-input physical or semantic identity disagrees"
        )
    try:
        validated = validate_semantic_t1_prepared_inputs(
            artifact,
            expected_capacity_policy_sha256=policy["policy_sha256"],
            expected_panel_artifact_sha256=cache["panel_artifact_sha256"],
            expected_cache_completion_sha256=cache["completion_sha256"],
            expected_initial_model_state_sha256=cache["initial_model_state_sha256"],
            repo_root=repo_root,
        )
    except SemanticT1PreparedInputError as error:
        raise SemanticT1DecisionError(
            f"semantic T1 prepared inputs fail full validation: {error}"
        ) from error
    return validated, hashlib.sha256(raw).hexdigest()


def _expected_provenance(
    *,
    policy: Mapping[str, Any],
    cache: Mapping[str, Any],
    cache_file_sha256: str,
    gate_zero: Mapping[str, Any],
    gate_zero_file_sha256: str,
    prepared_input: Mapping[str, Any],
    prepared_input_file_sha256: str,
    runner_implementation_sha256: str,
    runner_source_revision_sha256: str,
    execution_environment: Mapping[str, Any],
) -> dict[str, Any]:
    prepared_entry_metadata = _panel_entry_metadata(
        prepared_input.get("entries"),
        family_field="model_family",
        cell_field="capability_cell_id",
        field="prepared_input.entries",
    )
    return {
        "capacity_policy_file_sha256": CAPACITY_POLICY_FILE_SHA256,
        "capacity_policy_sha256": policy["policy_sha256"],
        "cell_role_policy_sha256": policy["cell_role_policy_sha256"],
        "cache_completion_file_sha256": cache_file_sha256,
        "cache_completion_sha256": cache["completion_sha256"],
        "cache_manifest_file_sha256": cache["manifest_file_sha256"],
        "cache_manifest_sha256": cache["manifest_sha256"],
        "cache_run_identity_sha256": cache["run_identity_sha256"],
        "cache_build_identity_sha256": cache["build_identity_sha256"],
        "cache_source_revision_sha256": cache["source_revision_sha256"],
        "panel_completion_sha256": cache["panel_completion_sha256"],
        "panel_artifact_sha256": cache["panel_artifact_sha256"],
        "panel_entry_inventory_sha256": prepared_input["panel_entry_inventory_sha256"],
        "panel_entry_metadata_sha256": _sha(prepared_entry_metadata),
        "decision_source_inventory_sha256": cache["decision_source_inventory_sha256"],
        "gate_zero_evidence_file_sha256": gate_zero_file_sha256,
        "gate_zero_evidence_sha256": gate_zero["evidence_sha256"],
        "initial_model_state_sha256": cache["initial_model_state_sha256"],
        "panel_entry_binding_count": cache["panel_entry_binding_count"],
        "prepared_input_file_sha256": prepared_input_file_sha256,
        "prepared_input_artifact_sha256": prepared_input["artifact_sha256"],
        "prepared_input_implementation_sha256": prepared_input["implementation_sha256"],
        "runner_implementation_sha256": runner_implementation_sha256,
        "runner_source_revision_sha256": runner_source_revision_sha256,
        "execution_environment": dict(execution_environment),
    }


def build_semantic_t1_result_provenance(
    *,
    cache_completion_path: Path,
    gate_zero_evidence_path: Path,
    prepared_input_path: Path,
    runner_source_revision_sha256: str,
    execution_environment: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Build the exact runner-facing provenance mapping without duplication."""

    policy, _ = _validate_policy(repo_root)
    cache, cache_raw = _load_canonical(cache_completion_path, field="semantic T1 cache completion")
    try:
        cache = validate_semantic_t1_successor_cache_completion(cache)
    except SemanticT1SuccessorCacheError as error:
        raise SemanticT1DecisionError(
            f"semantic T1 cache completion is invalid: {error}"
        ) from error
    gate_zero, gate_zero_file_sha = _validate_gate_zero_receipt(gate_zero_evidence_path)
    if (
        cache["decision_source_inventory_sha256"] != gate_zero["decision_source_inventory_sha256"]
        or cache["initial_model_state_sha256"]
        != gate_zero["model_runtime_identity"]["initial_model_state_sha256"]
    ):
        raise SemanticT1DecisionError("semantic cache and Gate0 provenance identities disagree")
    prepared_input, prepared_input_file_sha256 = _validate_prepared_input_identity(
        prepared_input_path,
        policy=policy,
        cache=cache,
        repo_root=repo_root,
    )
    environment = _validate_execution_environment(
        execution_environment,
        policy=policy,
    )
    return _expected_provenance(
        policy=policy,
        cache=cache,
        cache_file_sha256=hashlib.sha256(cache_raw).hexdigest(),
        gate_zero=gate_zero,
        gate_zero_file_sha256=gate_zero_file_sha,
        prepared_input=prepared_input,
        prepared_input_file_sha256=prepared_input_file_sha256,
        runner_implementation_sha256=semantic_t1_runner_implementation_sha256(repo_root=repo_root),
        runner_source_revision_sha256=_require_sha(
            runner_source_revision_sha256,
            field="runner_source_revision_sha256",
        ),
        execution_environment=environment,
    )


def build_semantic_t1_capacity_completion(
    *,
    artifact_directory: Path,
    result_path: Path,
    cache_completion_path: Path,
    gate_zero_evidence_path: Path,
    prepared_input_path: Path,
    selected_checkpoint_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Cross-link exact result, cache, Gate0, and prospective policy bytes."""

    root = Path(artifact_directory).resolve()
    if Path(result_path).name != RESULT_FILENAME:
        raise SemanticT1DecisionError(f"semantic T1 result must name {RESULT_FILENAME}")
    policy, _ = _validate_policy(repo_root)
    result, result_raw = _load_canonical(result_path, field="semantic T1 result")
    result = validate_semantic_t1_capacity_result(result)
    cache, cache_raw = _load_canonical(cache_completion_path, field="semantic T1 cache completion")
    try:
        cache = validate_semantic_t1_successor_cache_completion(cache)
    except SemanticT1SuccessorCacheError as error:
        raise SemanticT1DecisionError(
            f"semantic T1 cache completion is invalid: {error}"
        ) from error
    gate_zero, gate_zero_file_sha = _validate_gate_zero_receipt(gate_zero_evidence_path)
    if (
        cache["decision_source_inventory_sha256"] != gate_zero["decision_source_inventory_sha256"]
        or cache["initial_model_state_sha256"]
        != gate_zero["model_runtime_identity"]["initial_model_state_sha256"]
    ):
        raise SemanticT1DecisionError("semantic cache and Gate0 provenance identities disagree")
    prepared_input, prepared_input_file_sha256 = _validate_prepared_input_identity(
        prepared_input_path,
        policy=policy,
        cache=cache,
        repo_root=repo_root,
    )
    prepared_entry_metadata = _panel_entry_metadata(
        prepared_input.get("entries"),
        family_field="model_family",
        cell_field="capability_cell_id",
        field="prepared_input.entries",
    )
    result_entry_metadata = _panel_entry_metadata(
        result.get("entry_metrics"),
        family_field="family",
        cell_field="semantic_cell_id",
        field="result.entry_metrics",
    )
    if result_entry_metadata != prepared_entry_metadata:
        raise SemanticT1DecisionError(
            "semantic T1 result entries differ from the exact prepared-panel "
            "entry IDs, families, or semantic cells"
        )
    environment = _validate_execution_environment(
        result["provenance"]["execution_environment"],
        policy=policy,
    )
    runner_implementation_sha256 = semantic_t1_runner_implementation_sha256(repo_root=repo_root)
    runner_source_revision_sha256 = _require_sha(
        result["provenance"]["runner_source_revision_sha256"],
        field="result.provenance.runner_source_revision_sha256",
    )
    expected_provenance = _expected_provenance(
        policy=policy,
        cache=cache,
        cache_file_sha256=hashlib.sha256(cache_raw).hexdigest(),
        gate_zero=gate_zero,
        gate_zero_file_sha256=gate_zero_file_sha,
        prepared_input=prepared_input,
        prepared_input_file_sha256=prepared_input_file_sha256,
        runner_implementation_sha256=runner_implementation_sha256,
        runner_source_revision_sha256=runner_source_revision_sha256,
        execution_environment=environment,
    )
    if result["provenance"] != expected_provenance:
        raise SemanticT1DecisionError("semantic T1 result provenance differs from exact inputs")

    def relative(path: Path, *, field: str) -> str:
        resolved = Path(path).resolve()
        if not resolved.is_relative_to(root):
            raise SemanticT1DecisionError(f"{field} must be inside artifact_directory")
        return resolved.relative_to(root).as_posix()

    selected_checkpoint_file_sha256 = result["run_integrity"]["selected_checkpoint_file_sha256"]
    try:
        selected_checkpoint = validate_semantic_t1_selected_checkpoint(
            selected_checkpoint_path,
            expected_file_sha256=selected_checkpoint_file_sha256,
            expected_selected_step=result["run_integrity"]["selected_step"],
            expected_model_state_sha256=result["run_integrity"]["selected_model_state_sha256"],
            expected_stream_sha256=result["run_integrity"]["address_stream_sha256"],
            expected_identity_fields={
                "capacity_policy_sha256": policy["policy_sha256"],
                "optimization_policy": policy["optimization"],
                "prepared_input_artifact_sha256": prepared_input["artifact_sha256"],
                "cache_completion_sha256": cache["completion_sha256"],
                "cache_manifest_sha256": cache["manifest_sha256"],
                "initial_model_state_sha256": cache["initial_model_state_sha256"],
                "runner_implementation_sha256": runner_implementation_sha256,
                "runner_source_revision_sha256": runner_source_revision_sha256,
                "execution_environment": environment,
                "execution_environment_sha256": environment["environment_sha256"],
            },
        )
    except SemanticT1SelectedCheckpointError as error:
        raise SemanticT1DecisionError(
            f"selected semantic T1 checkpoint is invalid: {error}"
        ) from error
    selected_checkpoint_identity_sha256 = _sha(selected_checkpoint["identity"])

    body: dict[str, Any] = {
        "schema": COMPLETION_SCHEMA,
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "status": COMPLETION_STATUS,
        **_NO_AUTHORITY,
        "capacity_policy_file_sha256": CAPACITY_POLICY_FILE_SHA256,
        "capacity_policy_sha256": policy["policy_sha256"],
        "cell_role_policy_sha256": policy["cell_role_policy_sha256"],
        "result_relative_path": relative(result_path, field="result_path"),
        "result_file_sha256": hashlib.sha256(result_raw).hexdigest(),
        "result_sha256": result["result_sha256"],
        "cache_completion_relative_path": relative(
            cache_completion_path, field="cache_completion_path"
        ),
        "cache_completion_file_sha256": hashlib.sha256(cache_raw).hexdigest(),
        "cache_completion_sha256": cache["completion_sha256"],
        "gate_zero_evidence_relative_path": relative(
            gate_zero_evidence_path, field="gate_zero_evidence_path"
        ),
        "gate_zero_evidence_file_sha256": gate_zero_file_sha,
        "gate_zero_evidence_sha256": gate_zero["evidence_sha256"],
        "prepared_input_relative_path": relative(
            prepared_input_path,
            field="prepared_input_path",
        ),
        "prepared_input_file_sha256": prepared_input_file_sha256,
        "prepared_input_artifact_sha256": prepared_input["artifact_sha256"],
        "prepared_input_implementation_sha256": prepared_input["implementation_sha256"],
        "panel_entry_inventory_sha256": prepared_input["panel_entry_inventory_sha256"],
        "panel_entry_metadata_sha256": _sha(prepared_entry_metadata),
        "cache_source_revision_sha256": cache["source_revision_sha256"],
        "runner_implementation_sha256": runner_implementation_sha256,
        "runner_source_revision_sha256": runner_source_revision_sha256,
        "execution_environment_sha256": environment["environment_sha256"],
        "decision_source_inventory_sha256": cache["decision_source_inventory_sha256"],
        "initial_model_state_sha256": cache["initial_model_state_sha256"],
        "selected_checkpoint_relative_path": relative(
            selected_checkpoint_path,
            field="selected_checkpoint_path",
        ),
        "selected_checkpoint_file_sha256": selected_checkpoint_file_sha256,
        "selected_checkpoint_identity_sha256": selected_checkpoint_identity_sha256,
        "selected_model_state_sha256": result["run_integrity"]["selected_model_state_sha256"],
    }
    return {**body, "completion_sha256": _sha(body)}


def validate_semantic_t1_capacity_completion(
    value: object,
    *,
    completion_path: Path,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reopen and recompute a completion from all exact physical inputs."""

    physical_path = Path(completion_path).resolve()
    if physical_path.name != COMPLETION_FILENAME:
        raise SemanticT1DecisionError(f"semantic T1 completion must name {COMPLETION_FILENAME}")
    physical, _ = _load_canonical(physical_path, field="semantic T1 completion")
    if value != physical:
        raise SemanticT1DecisionError(
            "semantic T1 completion payload differs from its physical artifact"
        )

    fields = {
        "schema",
        "schema_version",
        "status",
        *_NO_AUTHORITY,
        "capacity_policy_file_sha256",
        "capacity_policy_sha256",
        "cell_role_policy_sha256",
        "result_relative_path",
        "result_file_sha256",
        "result_sha256",
        "cache_completion_relative_path",
        "cache_completion_file_sha256",
        "cache_completion_sha256",
        "gate_zero_evidence_relative_path",
        "gate_zero_evidence_file_sha256",
        "gate_zero_evidence_sha256",
        "prepared_input_relative_path",
        "prepared_input_file_sha256",
        "prepared_input_artifact_sha256",
        "prepared_input_implementation_sha256",
        "panel_entry_inventory_sha256",
        "panel_entry_metadata_sha256",
        "cache_source_revision_sha256",
        "runner_implementation_sha256",
        "runner_source_revision_sha256",
        "execution_environment_sha256",
        "decision_source_inventory_sha256",
        "initial_model_state_sha256",
        "selected_checkpoint_relative_path",
        "selected_checkpoint_file_sha256",
        "selected_checkpoint_identity_sha256",
        "selected_model_state_sha256",
        "completion_sha256",
    }
    completion = dict(_exact_mapping(value, fields, field="semantic T1 completion"))
    body = dict(completion)
    supplied_sha = body.pop("completion_sha256")
    if (
        completion["schema"] != COMPLETION_SCHEMA
        or completion["schema_version"] != COMPLETION_SCHEMA_VERSION
        or completion["status"] != COMPLETION_STATUS
        or supplied_sha != _sha(body)
        or any(completion[name] is not expected for name, expected in _NO_AUTHORITY.items())
    ):
        raise SemanticT1DecisionError("semantic T1 completion identity or authority disagrees")
    for name in fields:
        if name.endswith("sha256"):
            _require_sha(completion[name], field=f"completion.{name}")
    root = physical_path.parent
    result_path = _relative_sibling(
        root,
        completion["result_relative_path"],
        field="completion.result_relative_path",
    )
    cache_path = _relative_sibling(
        root,
        completion["cache_completion_relative_path"],
        field="completion.cache_completion_relative_path",
    )
    gate_zero_path = _relative_sibling(
        root,
        completion["gate_zero_evidence_relative_path"],
        field="completion.gate_zero_evidence_relative_path",
    )
    prepared_input_path = _relative_sibling(
        root,
        completion["prepared_input_relative_path"],
        field="completion.prepared_input_relative_path",
    )
    selected_checkpoint_path = _relative_sibling(
        root,
        completion["selected_checkpoint_relative_path"],
        field="completion.selected_checkpoint_relative_path",
    )
    recomputed = build_semantic_t1_capacity_completion(
        artifact_directory=root,
        result_path=result_path,
        cache_completion_path=cache_path,
        gate_zero_evidence_path=gate_zero_path,
        prepared_input_path=prepared_input_path,
        selected_checkpoint_path=selected_checkpoint_path,
        repo_root=repo_root,
    )
    if completion != recomputed:
        raise SemanticT1DecisionError(
            "semantic T1 completion differs from exact physical recomputation"
        )
    result, _ = _load_canonical(result_path, field="semantic T1 result")
    return completion, validate_semantic_t1_capacity_result(result)


def _decision_failures(result: Mapping[str, Any], policy: Mapping[str, Any]) -> list[str]:
    thresholds = policy["thresholds"]
    failures: list[str] = []
    integrity = result["run_integrity"]
    termination_reason = integrity["termination_reason"]
    optimizer_steps = integrity["optimizer_steps_completed"]
    abort_triggered = integrity["abort_triggered"]
    abort_reasons = integrity["abort_reasons"]
    integrity_checks = {
        "resume_requested": integrity["resume_requested"] is False,
        "resume_count_zero": integrity["resume_count"] == 0,
        "abort_not_triggered": integrity["abort_triggered"] is False,
        "abort_reasons_empty": integrity["abort_reasons"] == [],
        "nonfinite_event_count_zero": integrity["nonfinite_event_count"] == 0,
        "unsupported_teacher_count_zero": integrity["unsupported_teacher_count"] == 0,
        "missing_candidate_count_zero": integrity["missing_candidate_count"] == 0,
        "provenance_drift_absent": integrity["provenance_drift_detected"] is False,
        "stream_identity_drift_absent": integrity["stream_identity_drift_detected"] is False,
        "deterministic_algorithms_enabled": integrity["deterministic_algorithms_enabled"] is True,
        "mixed_precision_disabled": integrity["mixed_precision"] is False,
        "hazard_excluded": integrity["hazard_included"] is False,
        "termination_reason_consistent": (
            (
                termination_reason == "all_thresholds_passed_early"
                and optimizer_steps < 500
                and abort_triggered is False
            )
            or (
                termination_reason == "maximum_optimizer_steps_reached"
                and optimizer_steps == 500
                and abort_triggered is False
            )
            or (termination_reason == "aborted" and abort_triggered is True and bool(abort_reasons))
        ),
    }
    failures.extend(
        f"run_integrity:{name}" for name, passed in integrity_checks.items() if not passed
    )

    roles = load_semantic_development_cell_roles()
    observed_cells = {str(row["semantic_cell_id"]) for row in result["cell_metrics"]}
    failures.extend(
        f"required_cell:missing:{cell_id}"
        for cell_id in sorted(roles.required_cell_set - observed_cells)
    )
    failures.extend(
        f"required_cell:unexpected:{cell_id}"
        for cell_id in sorted(observed_cells - roles.required_cell_set)
    )

    for gradient in result["gradient_evidence"]:
        family = gradient["family"]
        checks = {
            "family_gate_gradient_finite": gradient["family_gate_gradient_finite"] is True,
            "family_gate_gradient_nonzero": gradient["family_gate_cumulative_l2"] > 0.0,
            "family_gate_nonzero_updates": gradient["family_gate_nonzero_update_steps"] > 0,
            "action_route_gradient_finite": gradient["action_route_gradient_finite"] is True,
            "action_route_gradient_nonzero": gradient["action_route_cumulative_l2"] > 0.0,
            "action_route_nonzero_updates": gradient["action_route_nonzero_update_steps"] > 0,
        }
        failures.extend(
            f"gradient:{family}:{name}" for name, passed in checks.items() if not passed
        )

    for family in result["family_metrics"]:
        cardinality = policy["panel_cardinality"]
        minimum = cardinality["minimum_entries_by_family"][family["family"]]
        maximum = cardinality["maximum_entries_by_family"][family["family"]]
        checks = {
            "entry_count_minimum": family["entry_count"] >= minimum,
            "entry_count_maximum": family["entry_count"] <= maximum,
            "top1": family["teacher_successor_top1"]
            >= thresholds["minimum_unique_state_teacher_successor_top1"],
            "probability": family["teacher_successor_probability"]
            >= thresholds["minimum_unique_state_teacher_successor_probability"],
            "nll": family["canonical_successor_nll"]
            <= thresholds["maximum_unique_state_teacher_successor_nll"],
        }
        failures.extend(
            f"family:{family['family']}:{name}" for name, passed in checks.items() if not passed
        )
    for cell in result["cell_metrics"]:
        checks = {
            "top1": cell["teacher_successor_top1"]
            >= thresholds["minimum_nonempty_cell_teacher_successor_top1"],
            "probability": cell["teacher_successor_probability"]
            >= thresholds["minimum_nonempty_cell_teacher_successor_probability"],
            "nll": cell["canonical_successor_nll"]
            <= thresholds["maximum_nonempty_cell_teacher_successor_nll"],
        }
        failures.extend(
            f"cell:{cell['family']}:{cell['semantic_cell_id']}:{name}"
            for name, passed in checks.items()
            if not passed
        )
    for entry in result["entry_metrics"]:
        if not entry["teacher_successor_top1"]:
            failures.append(f"entry:{entry['panel_entry_sha256']}:top1")
        if (
            entry["teacher_successor_probability"]
            < thresholds["minimum_every_unique_entry_teacher_successor_probability"]
        ):
            failures.append(f"entry:{entry['panel_entry_sha256']}:probability")
    return sorted(failures)


def build_semantic_t1_capacity_decision(
    *,
    completion_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Derive GO or NO_GO from one fully revalidated physical completion."""

    if Path(completion_path).name != COMPLETION_FILENAME:
        raise SemanticT1DecisionError(f"semantic T1 completion must name {COMPLETION_FILENAME}")
    completion, completion_raw = _load_canonical(completion_path, field="semantic T1 completion")
    completion, result = validate_semantic_t1_capacity_completion(
        completion, completion_path=completion_path, repo_root=repo_root
    )
    policy, _ = _validate_policy(repo_root)
    failures = _decision_failures(result, policy)
    passed = not failures
    body: dict[str, Any] = {
        "schema": DECISION_SCHEMA,
        "schema_version": DECISION_SCHEMA_VERSION,
        "status": DECISION_GO_STATUS if passed else DECISION_NO_GO_STATUS,
        "training_authorized": False,
        "bounded_p50_authorized": passed,
        "long_training_authorized": False,
        "checkpoint_selection_authorized": False,
        "final_test_selection_authorized": False,
        "capacity_policy_file_sha256": CAPACITY_POLICY_FILE_SHA256,
        "capacity_policy_sha256": policy["policy_sha256"],
        "cell_role_policy_sha256": policy["cell_role_policy_sha256"],
        "completion_relative_path": Path(completion_path).name,
        "completion_file_sha256": hashlib.sha256(completion_raw).hexdigest(),
        "completion_sha256": completion["completion_sha256"],
        "result_file_sha256": completion["result_file_sha256"],
        "result_sha256": completion["result_sha256"],
        "cache_completion_file_sha256": completion["cache_completion_file_sha256"],
        "cache_completion_sha256": completion["cache_completion_sha256"],
        "gate_zero_evidence_file_sha256": completion["gate_zero_evidence_file_sha256"],
        "gate_zero_evidence_sha256": completion["gate_zero_evidence_sha256"],
        "prepared_input_file_sha256": completion["prepared_input_file_sha256"],
        "prepared_input_artifact_sha256": completion["prepared_input_artifact_sha256"],
        "prepared_input_implementation_sha256": completion["prepared_input_implementation_sha256"],
        "panel_entry_inventory_sha256": completion["panel_entry_inventory_sha256"],
        "panel_entry_metadata_sha256": completion["panel_entry_metadata_sha256"],
        "cache_source_revision_sha256": completion["cache_source_revision_sha256"],
        "runner_implementation_sha256": completion["runner_implementation_sha256"],
        "runner_source_revision_sha256": completion["runner_source_revision_sha256"],
        "execution_environment_sha256": completion["execution_environment_sha256"],
        "decision_source_inventory_sha256": completion["decision_source_inventory_sha256"],
        "initial_model_state_sha256": completion["initial_model_state_sha256"],
        "selected_checkpoint_file_sha256": completion["selected_checkpoint_file_sha256"],
        "selected_checkpoint_identity_sha256": completion["selected_checkpoint_identity_sha256"],
        "selected_model_state_sha256": completion["selected_model_state_sha256"],
        "required_families": list(RINGCORE_EDITING_FAMILIES),
        "family_count": len(result["family_metrics"]),
        "nonempty_cell_count": len(result["cell_metrics"]),
        "unique_entry_count": len(result["entry_metrics"]),
        "repeated_state_empirical_law_required_for_p50": False,
        "repeated_state_empirical_law_evaluated": False,
        "failed_checks": failures,
    }
    return {**body, "decision_sha256": _sha(body)}


def validate_semantic_t1_capacity_decision(
    value: object,
    *,
    decision_path: Path,
    repo_root: Path,
    expected_gate_zero_evidence_file_sha256: str | None = None,
    require_p50_go: bool = False,
) -> dict[str, Any]:
    """Reopen a decision chain and optionally require exact bounded-P50 GO."""

    physical_path = Path(decision_path).resolve()
    if physical_path.name != DECISION_FILENAME:
        raise SemanticT1DecisionError(f"semantic T1 decision must name {DECISION_FILENAME}")
    physical, _ = _load_canonical(physical_path, field="semantic T1 decision")
    if value != physical:
        raise SemanticT1DecisionError(
            "semantic T1 decision payload differs from its physical artifact"
        )

    fields = {
        "schema",
        "schema_version",
        "status",
        "training_authorized",
        "bounded_p50_authorized",
        "long_training_authorized",
        "checkpoint_selection_authorized",
        "final_test_selection_authorized",
        "capacity_policy_file_sha256",
        "capacity_policy_sha256",
        "cell_role_policy_sha256",
        "completion_relative_path",
        "completion_file_sha256",
        "completion_sha256",
        "result_file_sha256",
        "result_sha256",
        "cache_completion_file_sha256",
        "cache_completion_sha256",
        "gate_zero_evidence_file_sha256",
        "gate_zero_evidence_sha256",
        "prepared_input_file_sha256",
        "prepared_input_artifact_sha256",
        "prepared_input_implementation_sha256",
        "panel_entry_inventory_sha256",
        "panel_entry_metadata_sha256",
        "cache_source_revision_sha256",
        "runner_implementation_sha256",
        "runner_source_revision_sha256",
        "execution_environment_sha256",
        "decision_source_inventory_sha256",
        "initial_model_state_sha256",
        "selected_checkpoint_file_sha256",
        "selected_checkpoint_identity_sha256",
        "selected_model_state_sha256",
        "required_families",
        "family_count",
        "nonempty_cell_count",
        "unique_entry_count",
        "repeated_state_empirical_law_required_for_p50",
        "repeated_state_empirical_law_evaluated",
        "failed_checks",
        "decision_sha256",
    }
    decision = dict(_exact_mapping(value, fields, field="semantic T1 decision"))
    body = dict(decision)
    supplied_sha = body.pop("decision_sha256")
    if (
        decision["schema"] != DECISION_SCHEMA
        or decision["schema_version"] != DECISION_SCHEMA_VERSION
        or supplied_sha != _sha(body)
        or decision["training_authorized"] is not False
        or decision["long_training_authorized"] is not False
        or decision["checkpoint_selection_authorized"] is not False
        or decision["final_test_selection_authorized"] is not False
        or tuple(decision["required_families"]) != RINGCORE_EDITING_FAMILIES
        or decision["nonempty_cell_count"]
        != len(load_semantic_development_cell_roles().required_cell_ids)
        or decision["repeated_state_empirical_law_required_for_p50"] is not False
        or decision["repeated_state_empirical_law_evaluated"] is not False
    ):
        raise SemanticT1DecisionError("semantic T1 decision identity or scope disagrees")
    for name in fields:
        if name.endswith("sha256"):
            _require_sha(decision[name], field=f"decision.{name}")
    root = physical_path.parent
    completion_path = _relative_sibling(
        root,
        decision["completion_relative_path"],
        field="decision.completion_relative_path",
    )
    recomputed = build_semantic_t1_capacity_decision(
        completion_path=completion_path, repo_root=repo_root
    )
    if decision != recomputed:
        raise SemanticT1DecisionError(
            "semantic T1 decision differs from exact physical recomputation"
        )
    if expected_gate_zero_evidence_file_sha256 is not None and decision[
        "gate_zero_evidence_file_sha256"
    ] != _require_sha(
        expected_gate_zero_evidence_file_sha256,
        field="expected Gate0 evidence file SHA-256",
    ):
        raise SemanticT1DecisionError("semantic T1 decision names another Gate0 artifact")
    if require_p50_go and (
        decision["status"] != DECISION_GO_STATUS
        or decision["bounded_p50_authorized"] is not True
        or decision["failed_checks"] != []
    ):
        raise SemanticT1DecisionError("semantic T1 decision is not an exact bounded-P50 GO")
    return decision


__all__ = [
    "CAPACITY_POLICY_FILE_SHA256",
    "CAPACITY_POLICY_RELATIVE_PATH",
    "CAPACITY_POLICY_SHA256",
    "COMPLETION_FILENAME",
    "COMPLETION_SCHEMA",
    "DECISION_FILENAME",
    "DECISION_GO_STATUS",
    "DECISION_NO_GO_STATUS",
    "DECISION_SCHEMA",
    "RESULT_FILENAME",
    "RESULT_SCHEMA",
    "SemanticT1DecisionError",
    "build_semantic_t1_capacity_completion",
    "build_semantic_t1_capacity_decision",
    "build_semantic_t1_capacity_result",
    "build_semantic_t1_execution_environment",
    "build_semantic_t1_result_provenance",
    "semantic_t1_runner_implementation_sha256",
    "validate_semantic_gate_zero_evidence_receipt",
    "validate_semantic_t1_capacity_completion",
    "validate_semantic_t1_capacity_decision",
    "validate_semantic_t1_capacity_result",
]
