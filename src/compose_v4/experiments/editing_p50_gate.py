"""Sealed, fail-closed evidence for the exactly-50-step editing pilot.

This module does not run training and does not choose numeric thresholds.  It
turns already-resolved development thresholds and already-observed P50
diagnostics into one strict, self-hashed qualification artifact.  A PASS is
eligible to unlock P500 only when it is bound to:

* the frozen scientific contract and prerequisite S0/T1 evidence;
* the exact recipe, objective, schedule, corpus, panels, and cache inventory;
* the complete planned and observed ordered training stream, including
  addresses, times, teachers, cells, and objective coefficients;
* separate family-gate and within-family action-route gradient updates;
* baseline and step-50 canonical-successor validation metrics;
* an initialization-appropriate retention statement; and
* a recovery checkpoint whose *current_state_dict* is exactly step 50.

The historical ``best_state_dict`` is never inspected.  P50 is intentionally
non-resumable: fifty steps are cheap enough that a partial run should restart
from its frozen initialization rather than introduce sentinel-resume state.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from math import isclose, isfinite
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor

from compose_v4.experiments.editing_training_gate import (
    P50_INITIALIZATION_REGIMES,
    REQUIRED_P50_FAMILIES,
    ResolvedP50Thresholds,
)

P50_EVIDENCE_SCHEMA = "compose.editing_p50_evidence"
P50_EVIDENCE_SCHEMA_VERSION = 1
P50_EXPECTED_OPTIMIZER_STEPS = 50
P50_SELECTION_METRIC = "production_weighted_canonical_successor_nll"
P50_BALANCED_METRIC = "balanced_semantic_cell_canonical_successor_nll"
P50_CURRENT_STATE_SOURCE = "current_state_dict"

P50Status = Literal["PASS", "FAIL"]
P50InitializationRegime = Literal[
    "scratch",
    "compatible_warm_start",
    "compatible_warm_start_with_retention",
]
P50RetentionApplicability = Literal[
    "NOT_APPLICABLE_SCRATCH",
    "REQUIRED_WARM_START",
]

_INITIALIZATION_REGIMES = set(P50_INITIALIZATION_REGIMES)
_RETENTION_APPLICABILITY = {
    "NOT_APPLICABLE_SCRATCH",
    "REQUIRED_WARM_START",
}
_TOP_LEVEL_KEYS = {
    "schema",
    "schema_version",
    "status",
    "run_identity",
    "resolved_thresholds",
    "completed_optimizer_steps",
    "checkpoint",
    "observed_ordered_training_stream_sha256",
    "gradient_updates",
    "baseline_validation",
    "final_validation",
    "retention",
    "failure_reason",
    "artifact_sha256",
}
_RUN_IDENTITY_KEYS = {
    "training_gate_contract_sha256",
    "resolved_thresholds_sha256",
    "s0_evidence_sha256",
    "t1_evidence_sha256",
    "training_recipe_sha256",
    "initial_state_semantic_sha256",
    "objective_name",
    "selection_metric",
    "optimizer_kind",
    "schedule_sha256",
    "seed",
    "initialization_regime",
    "corpus_manifest_sha256",
    "representability_overlay_sha256",
    "semantic_cell_sidecar_sha256",
    "validation_panel_sha256",
    "successor_cache_inventory_sha256",
    "planned_ordered_training_stream_sha256",
    "expected_optimizer_steps",
    "resume",
}
_THRESHOLD_KEYS = {
    "initialization_regime",
    "required_families",
    "minimum_gradient_updates",
    "maximum_family_nll_regression",
    "inherited_retention_status",
    "maximum_inherited_probe_nll_regression",
}
_CHECKPOINT_KEYS = {
    "state_source",
    "completed_optimizer_steps",
    "current_state_semantic_sha256",
    "checkpoint_file_sha256",
    "checkpoint_file_bytes",
}
_GRADIENT_KEYS = {
    "family_gate_updates",
    "action_route_updates",
}
_RETENTION_KEYS = {
    "applicability",
    "probe_panel_sha256",
    "baseline_checkpoint_sha256",
    "baseline_nll",
    "final_nll",
    "observed_regression",
}


class P50GateError(ValueError):
    """P50 evidence is incomplete, inconsistent, tampered with, or ineligible."""


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
        raise P50GateError("P50 evidence is not finite canonical JSON") from error


def _stable_json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, *, field: str) -> str:
    if not _is_sha256(value):
        raise P50GateError(f"{field} must be a lowercase SHA-256 digest")
    return str(value)


def _require_exact_keys(
    value: object,
    expected: set[str],
    *,
    field: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise P50GateError(f"{field} must be an object")
    observed = set(value)
    if observed != expected:
        raise P50GateError(
            f"{field} has missing or unknown fields: "
            f"missing={sorted(expected - observed)}, "
            f"unknown={sorted(observed - expected)}"
        )
    return value


def _require_nonempty_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise P50GateError(f"{field} must be nonempty text")
    return value


def _require_nonnegative_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise P50GateError(f"{field} must be a nonnegative integer")
    return value


def _require_finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise P50GateError(f"{field} must be a finite JSON number")
    result = float(value)
    if not isfinite(result):
        raise P50GateError(f"{field} must be a finite JSON number")
    return result


@dataclass(frozen=True)
class _FileIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


def _file_identity(metadata: os.stat_result) -> _FileIdentity:
    if not stat.S_ISREG(metadata.st_mode):
        raise P50GateError("P50 checkpoint must be a regular file")
    return _FileIdentity(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        size=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
        ctime_ns=metadata.st_ctime_ns,
    )


def _open_file_sha256(handle: Any) -> str:
    digest = hashlib.sha256()
    while block := handle.read(1 << 20):
        digest.update(block)
    return digest.hexdigest()


def state_dict_semantic_sha256(state_dict: Mapping[str, Tensor]) -> str:
    """Hash tensor names, dtypes, shapes, and bytes in deterministic order."""

    if not isinstance(state_dict, Mapping) or not state_dict:
        raise P50GateError("current_state_dict must be a nonempty tensor mapping")
    names = tuple(state_dict)
    if any(not isinstance(name, str) or not name for name in names):
        raise P50GateError("current_state_dict contains an invalid tensor name")
    if len(names) != len(set(names)):
        raise P50GateError("current_state_dict contains duplicate tensor names")

    digest = hashlib.sha256()
    digest.update(b"compose.current_state_dict.semantic.v1\0")
    for name in sorted(names):
        tensor = state_dict[name]
        if not isinstance(tensor, Tensor):
            raise P50GateError(f"current_state_dict value {name!r} is not a tensor")
        if tensor.device.type == "meta" or tensor.layout != torch.strided:
            raise P50GateError(f"current_state_dict tensor {name!r} is not materialized dense data")
        detached = tensor.detach().cpu().contiguous()
        if (detached.is_floating_point() or detached.is_complex()) and not bool(
            torch.isfinite(detached).all()
        ):
            raise P50GateError(f"current_state_dict tensor {name!r} contains nonfinite values")
        encoded_name = name.encode("utf-8")
        encoded_dtype = str(detached.dtype).encode("ascii")
        encoded_shape = _canonical_json_bytes(list(detached.shape))
        raw = detached.reshape(-1).view(torch.uint8).numpy().tobytes()
        for block in (encoded_name, encoded_dtype, encoded_shape, raw):
            digest.update(len(block).to_bytes(8, "big"))
            digest.update(block)
    return digest.hexdigest()


def _validate_resolved_thresholds(
    thresholds: ResolvedP50Thresholds,
) -> tuple[dict[str, int], dict[str, float]]:
    """Strictly adapt the gate resolver's single authoritative threshold type."""

    if not isinstance(thresholds, ResolvedP50Thresholds):
        raise P50GateError("resolved_thresholds must come from the editing-training gate resolver")
    if thresholds.initialization_regime not in _INITIALIZATION_REGIMES:
        raise P50GateError("resolved P50 thresholds have an unknown initialization regime")
    required = tuple(REQUIRED_P50_FAMILIES)
    if thresholds.required_families != required:
        raise P50GateError(
            "resolved P50 thresholds must cover the ordered eight-family "
            "bounded pilot"
        )

    minimum_pairs = thresholds.minimum_gradient_updates
    regression_pairs = thresholds.maximum_family_nll_regression
    if (
        type(minimum_pairs) is not tuple
        or any(type(pair) is not tuple or len(pair) != 2 for pair in minimum_pairs)
        or tuple(pair[0] for pair in minimum_pairs) != required
    ):
        raise P50GateError("resolved minimum-gradient thresholds are not normalized")
    if (
        type(regression_pairs) is not tuple
        or any(type(pair) is not tuple or len(pair) != 2 for pair in regression_pairs)
        or tuple(pair[0] for pair in regression_pairs) != required
    ):
        raise P50GateError("resolved family-NLL thresholds are not normalized")

    minimum = dict(minimum_pairs)
    regressions = dict(regression_pairs)
    if any(
        type(value) is not int or not 1 <= value <= P50_EXPECTED_OPTIMIZER_STEPS
        for value in minimum.values()
    ):
        raise P50GateError("resolved minimum gradient updates must be integers in [1, 50]")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or float(value) < 0.0
        for value in regressions.values()
    ):
        raise P50GateError(
            "resolved family NLL regression thresholds must be finite and nonnegative"
        )

    inherited = thresholds.maximum_inherited_probe_nll_regression
    if thresholds.initialization_regime == "scratch":
        if thresholds.inherited_retention_status != "NOT_APPLICABLE" or inherited is not None:
            raise P50GateError(
                "scratch resolved thresholds must bind retention status NOT_APPLICABLE"
            )
    else:
        if thresholds.inherited_retention_status != "REQUIRED":
            raise P50GateError("warm-start resolved thresholds must bind retention status REQUIRED")
        if (
            isinstance(inherited, bool)
            or not isinstance(inherited, (int, float))
            or not isfinite(float(inherited))
            or float(inherited) < 0.0
        ):
            raise P50GateError("warm-start resolved thresholds require a finite retention bound")
    return minimum, {family: float(value) for family, value in regressions.items()}


def resolved_p50_thresholds_payload(
    thresholds: ResolvedP50Thresholds,
) -> dict[str, object]:
    """Return the canonical evidence representation of gate-resolved thresholds."""

    minimum, regressions = _validate_resolved_thresholds(thresholds)
    return {
        "initialization_regime": thresholds.initialization_regime,
        "required_families": list(thresholds.required_families),
        "minimum_gradient_updates": minimum,
        "maximum_family_nll_regression": regressions,
        "inherited_retention_status": thresholds.inherited_retention_status,
        "maximum_inherited_probe_nll_regression": (
            thresholds.maximum_inherited_probe_nll_regression
        ),
    }


def resolved_p50_thresholds_sha256(thresholds: ResolvedP50Thresholds) -> str:
    """Hash all resolved values, including initialization and retention status."""

    return _stable_json_sha256(resolved_p50_thresholds_payload(thresholds))


@dataclass(frozen=True)
class P50RunIdentity:
    """Immutable scientific and engineering identity of one P50 attempt."""

    training_gate_contract_sha256: str
    resolved_thresholds_sha256: str
    s0_evidence_sha256: str
    t1_evidence_sha256: str
    training_recipe_sha256: str
    initial_state_semantic_sha256: str
    objective_name: str
    selection_metric: str
    optimizer_kind: str
    schedule_sha256: str
    seed: int
    initialization_regime: P50InitializationRegime
    corpus_manifest_sha256: str
    representability_overlay_sha256: str
    semantic_cell_sidecar_sha256: str
    validation_panel_sha256: str
    successor_cache_inventory_sha256: str
    planned_ordered_training_stream_sha256: str
    expected_optimizer_steps: int = P50_EXPECTED_OPTIMIZER_STEPS
    resume: bool = False

    def __post_init__(self) -> None:
        for field in (
            "training_gate_contract_sha256",
            "resolved_thresholds_sha256",
            "s0_evidence_sha256",
            "t1_evidence_sha256",
            "training_recipe_sha256",
            "initial_state_semantic_sha256",
            "schedule_sha256",
            "corpus_manifest_sha256",
            "representability_overlay_sha256",
            "semantic_cell_sidecar_sha256",
            "validation_panel_sha256",
            "successor_cache_inventory_sha256",
            "planned_ordered_training_stream_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        _require_nonempty_text(self.objective_name, field="objective_name")
        if not self.objective_name.startswith("canonical_successor_"):
            raise P50GateError("P50 must use a canonical-successor objective")
        if self.selection_metric != P50_SELECTION_METRIC:
            raise P50GateError("P50 checkpoint selection must use canonical-successor NLL")
        _require_nonempty_text(self.optimizer_kind, field="optimizer_kind")
        if type(self.seed) is not int or self.seed < 0:
            raise P50GateError("P50 seed must be a nonnegative integer")
        if self.initialization_regime not in _INITIALIZATION_REGIMES:
            raise P50GateError("unknown P50 initialization regime")
        if self.expected_optimizer_steps != P50_EXPECTED_OPTIMIZER_STEPS:
            raise P50GateError("P50 must contain exactly 50 optimizer steps")
        if self.resume is not False:
            raise P50GateError("P50 resume is not authorized")

    def to_payload(self) -> dict[str, object]:
        return {
            "training_gate_contract_sha256": self.training_gate_contract_sha256,
            "resolved_thresholds_sha256": self.resolved_thresholds_sha256,
            "s0_evidence_sha256": self.s0_evidence_sha256,
            "t1_evidence_sha256": self.t1_evidence_sha256,
            "training_recipe_sha256": self.training_recipe_sha256,
            "initial_state_semantic_sha256": self.initial_state_semantic_sha256,
            "objective_name": self.objective_name,
            "selection_metric": self.selection_metric,
            "optimizer_kind": self.optimizer_kind,
            "schedule_sha256": self.schedule_sha256,
            "seed": self.seed,
            "initialization_regime": self.initialization_regime,
            "corpus_manifest_sha256": self.corpus_manifest_sha256,
            "representability_overlay_sha256": (self.representability_overlay_sha256),
            "semantic_cell_sidecar_sha256": self.semantic_cell_sidecar_sha256,
            "validation_panel_sha256": self.validation_panel_sha256,
            "successor_cache_inventory_sha256": (self.successor_cache_inventory_sha256),
            "planned_ordered_training_stream_sha256": (self.planned_ordered_training_stream_sha256),
            "expected_optimizer_steps": self.expected_optimizer_steps,
            "resume": self.resume,
        }


@dataclass(frozen=True)
class P50RetentionEvidence:
    """Initialization-aware inherited-capability evidence."""

    applicability: P50RetentionApplicability
    probe_panel_sha256: str | None = None
    baseline_checkpoint_sha256: str | None = None
    baseline_nll: float | None = None
    final_nll: float | None = None
    observed_regression: float | None = None

    @classmethod
    def scratch(cls) -> P50RetentionEvidence:
        return cls(applicability="NOT_APPLICABLE_SCRATCH")

    def to_payload(self) -> dict[str, object]:
        return {
            "applicability": self.applicability,
            "probe_panel_sha256": self.probe_panel_sha256,
            "baseline_checkpoint_sha256": self.baseline_checkpoint_sha256,
            "baseline_nll": self.baseline_nll,
            "final_nll": self.final_nll,
            "observed_regression": self.observed_regression,
        }


@dataclass(frozen=True)
class P50Evidence:
    """Typed result returned by strict evidence validation."""

    status: P50Status
    run_identity: P50RunIdentity
    resolved_thresholds: ResolvedP50Thresholds
    completed_optimizer_steps: int
    checkpoint: Mapping[str, object]
    observed_ordered_training_stream_sha256: str | None
    family_gate_updates: Mapping[str, int]
    action_route_updates: Mapping[str, int]
    baseline_validation: Mapping[str, float]
    final_validation: Mapping[str, float] | None
    retention: P50RetentionEvidence
    failure_reason: str | None
    artifact_sha256: str


def _required_validation_metric_keys(
    families: tuple[str, ...],
) -> set[str]:
    keys = {P50_SELECTION_METRIC, P50_BALANCED_METRIC}
    for family in families:
        keys.add(f"teacher_examples_{family}")
        keys.add(f"canonical_successor_nll_{family}")
    return keys


def _validation_payload(
    metrics: Mapping[str, float],
    *,
    families: tuple[str, ...],
    field: str,
) -> dict[str, float]:
    required = _required_validation_metric_keys(families)
    _require_exact_keys(metrics, required, field=field)
    result: dict[str, float] = {}
    for key in sorted(required):
        value = _require_finite_number(metrics[key], field=f"{field}.{key}")
        if key.startswith("teacher_examples_"):
            if value <= 0.0 or not value.is_integer():
                raise P50GateError(f"{field}.{key} must be a positive integer-valued count")
        elif value < 0.0:
            raise P50GateError(f"{field}.{key} must be nonnegative")
        result[key] = value
    return result


def _gradient_payload(
    values: Mapping[str, int],
    *,
    families: tuple[str, ...],
    field: str,
    completed_steps: int,
) -> dict[str, int]:
    _require_exact_keys(values, set(families), field=field)
    result: dict[str, int] = {}
    for family in families:
        value = values[family]
        if type(value) is not int or not 0 <= value <= completed_steps:
            raise P50GateError(f"{field}.{family} must be an integer in [0, completed steps]")
        result[family] = value
    return result


def _validate_retention(
    retention: P50RetentionEvidence,
    *,
    run_identity: P50RunIdentity,
    thresholds: ResolvedP50Thresholds,
    status: P50Status,
) -> None:
    if retention.applicability not in _RETENTION_APPLICABILITY:
        raise P50GateError("unknown retention applicability")
    scratch = run_identity.initialization_regime == "scratch"
    fields = (
        retention.probe_panel_sha256,
        retention.baseline_checkpoint_sha256,
        retention.baseline_nll,
        retention.final_nll,
        retention.observed_regression,
    )
    if scratch:
        if retention.applicability != "NOT_APPLICABLE_SCRATCH":
            raise P50GateError("scratch P50 must declare retention NOT_APPLICABLE_SCRATCH")
        if any(value is not None for value in fields):
            raise P50GateError("scratch P50 must not fabricate inherited retention evidence")
        if thresholds.maximum_inherited_probe_nll_regression is not None:
            raise P50GateError("scratch P50 thresholds must mark inherited retention null")
        if thresholds.inherited_retention_status != "NOT_APPLICABLE":
            raise P50GateError("scratch P50 threshold retention status is inconsistent")
        return

    if retention.applicability != "REQUIRED_WARM_START":
        raise P50GateError("warm-start P50 requires inherited retention evidence")
    if thresholds.inherited_retention_status != "REQUIRED":
        raise P50GateError("warm-start P50 threshold retention status is inconsistent")
    _require_sha256(
        retention.probe_panel_sha256,
        field="retention.probe_panel_sha256",
    )
    _require_sha256(
        retention.baseline_checkpoint_sha256,
        field="retention.baseline_checkpoint_sha256",
    )
    baseline = _require_finite_number(
        retention.baseline_nll,
        field="retention.baseline_nll",
    )
    if baseline < 0.0:
        raise P50GateError("retention baseline NLL must be nonnegative")
    threshold = thresholds.maximum_inherited_probe_nll_regression
    if threshold is None:
        raise P50GateError("warm-start P50 lacks an inherited-probe regression threshold")
    if retention.final_nll is None or retention.observed_regression is None:
        if status == "PASS":
            raise P50GateError("warm-start PASS requires final inherited retention metrics")
        return
    final = _require_finite_number(
        retention.final_nll,
        field="retention.final_nll",
    )
    regression = _require_finite_number(
        retention.observed_regression,
        field="retention.observed_regression",
    )
    if final < 0.0:
        raise P50GateError("retention final NLL must be nonnegative")
    if not isclose(regression, final - baseline, rel_tol=1e-9, abs_tol=1e-9):
        raise P50GateError("retention regression disagrees with final - baseline")
    if status == "PASS" and regression > threshold:
        raise P50GateError("warm-start inherited capability exceeds its threshold")


def _inspect_current_checkpoint(path: Path) -> dict[str, object]:
    try:
        before = _file_identity(path.stat())
        handle = path.open("rb")
    except OSError as error:
        raise P50GateError(f"P50 checkpoint does not exist or cannot be opened: {path}") from error
    if before.size <= 0:
        handle.close()
        raise P50GateError("P50 checkpoint file is empty")
    with handle:
        opened = _file_identity(os.fstat(handle.fileno()))
        if opened != before:
            raise P50GateError("P50 checkpoint identity changed before load")
        try:
            payload = torch.load(handle, map_location="cpu", weights_only=True)
        except Exception as error:
            raise P50GateError(
                "P50 checkpoint cannot be safely loaded with weights_only=True"
            ) from error
        after_load = _file_identity(os.fstat(handle.fileno()))
        if after_load != opened:
            raise P50GateError("P50 checkpoint changed while loading")
        handle.seek(0)
        checkpoint_sha256 = _open_file_sha256(handle)
        after_hash = _file_identity(os.fstat(handle.fileno()))
        if after_hash != after_load:
            raise P50GateError("P50 checkpoint changed while hashing")
    try:
        after_path = _file_identity(path.stat())
    except OSError as error:
        raise P50GateError("P50 checkpoint disappeared after hashing") from error
    if after_path != after_hash:
        raise P50GateError("P50 checkpoint path identity changed during inspection")
    if not isinstance(payload, Mapping):
        raise P50GateError("P50 checkpoint payload must be an object")
    completed = payload.get("completed_steps")
    if type(completed) is not int or completed != P50_EXPECTED_OPTIMIZER_STEPS:
        raise P50GateError("P50 checkpoint is not the exact step-50 recovery state")
    current = payload.get(P50_CURRENT_STATE_SOURCE)
    if not isinstance(current, Mapping):
        raise P50GateError("P50 checkpoint lacks current_state_dict; best_state is ineligible")
    return {
        "state_source": P50_CURRENT_STATE_SOURCE,
        "completed_optimizer_steps": completed,
        "current_state_semantic_sha256": state_dict_semantic_sha256(current),
        "checkpoint_file_sha256": checkpoint_sha256,
        "checkpoint_file_bytes": after_hash.size,
    }


def _validate_checkpoint_metadata(
    checkpoint: Mapping[str, Any],
    *,
    completed_steps: int,
    status: P50Status,
) -> None:
    values = tuple(checkpoint[key] for key in sorted(_CHECKPOINT_KEYS))
    if all(value is None for value in values):
        if status == "PASS":
            raise P50GateError("P50 PASS lacks a sealed current checkpoint")
        return
    if any(value is None for value in values):
        raise P50GateError("P50 checkpoint metadata is partially populated")
    if checkpoint["state_source"] != P50_CURRENT_STATE_SOURCE:
        raise P50GateError("P50 checkpoint state source is not current_state_dict")
    checkpoint_steps = checkpoint["completed_optimizer_steps"]
    if (
        type(checkpoint_steps) is not int
        or checkpoint_steps != P50_EXPECTED_OPTIMIZER_STEPS
        or checkpoint_steps != completed_steps
    ):
        raise P50GateError("P50 checkpoint step disagrees with exact evidence")
    _require_sha256(
        checkpoint["current_state_semantic_sha256"],
        field="checkpoint.current_state_semantic_sha256",
    )
    _require_sha256(
        checkpoint["checkpoint_file_sha256"],
        field="checkpoint.checkpoint_file_sha256",
    )
    if (
        type(checkpoint["checkpoint_file_bytes"]) is not int
        or checkpoint["checkpoint_file_bytes"] <= 0
    ):
        raise P50GateError("P50 checkpoint byte count is invalid")


def p50_evidence_self_hash(evidence: Mapping[str, Any]) -> str:
    """Return the canonical hash excluding the artifact's own hash field."""

    payload = dict(evidence)
    payload.pop("artifact_sha256", None)
    return _stable_json_sha256(payload)


def seal_p50_evidence(
    *,
    status: P50Status,
    run_identity: P50RunIdentity,
    resolved_thresholds: ResolvedP50Thresholds,
    completed_optimizer_steps: int,
    checkpoint_path: str | Path | None,
    observed_ordered_training_stream_sha256: str | None,
    family_gate_updates: Mapping[str, int],
    action_route_updates: Mapping[str, int],
    baseline_validation: Mapping[str, float],
    final_validation: Mapping[str, float] | None,
    retention: P50RetentionEvidence,
    failure_reason: str | None = None,
) -> dict[str, object]:
    """Create and validate one strict self-hashed PASS or FAIL artifact."""

    if status not in {"PASS", "FAIL"}:
        raise P50GateError("P50 status must be PASS or FAIL")
    minimum, maximum_regressions = _validate_resolved_thresholds(resolved_thresholds)
    if run_identity.resolved_thresholds_sha256 != resolved_p50_thresholds_sha256(
        resolved_thresholds
    ):
        raise P50GateError("run identity and resolved P50 thresholds have different hashes")
    if run_identity.initialization_regime != resolved_thresholds.initialization_regime:
        raise P50GateError(
            "run identity and resolved thresholds have different initialization regimes"
        )
    completed = _require_nonnegative_int(
        completed_optimizer_steps,
        field="completed_optimizer_steps",
    )
    if completed > P50_EXPECTED_OPTIMIZER_STEPS:
        raise P50GateError("P50 evidence exceeds its 50-step ceiling")
    families = resolved_thresholds.required_families
    gate_updates = _gradient_payload(
        family_gate_updates,
        families=families,
        field="family_gate_updates",
        completed_steps=completed,
    )
    route_updates = _gradient_payload(
        action_route_updates,
        families=families,
        field="action_route_updates",
        completed_steps=completed,
    )
    baseline = _validation_payload(
        baseline_validation,
        families=families,
        field="baseline_validation",
    )
    final = (
        None
        if final_validation is None
        else _validation_payload(
            final_validation,
            families=families,
            field="final_validation",
        )
    )
    _validate_retention(
        retention,
        run_identity=run_identity,
        thresholds=resolved_thresholds,
        status=status,
    )

    checkpoint = (
        {
            "state_source": None,
            "completed_optimizer_steps": None,
            "current_state_semantic_sha256": None,
            "checkpoint_file_sha256": None,
            "checkpoint_file_bytes": None,
        }
        if checkpoint_path is None
        else _inspect_current_checkpoint(Path(checkpoint_path))
    )
    observed = (
        None
        if observed_ordered_training_stream_sha256 is None
        else _require_sha256(
            observed_ordered_training_stream_sha256,
            field="observed_ordered_training_stream_sha256",
        )
    )
    if status == "PASS":
        if completed != P50_EXPECTED_OPTIMIZER_STEPS:
            raise P50GateError("P50 PASS requires exactly 50 optimizer steps")
        if checkpoint_path is None:
            raise P50GateError("P50 PASS requires the exact current checkpoint")
        if final is None:
            raise P50GateError("P50 PASS requires step-50 validation metrics")
        if observed != run_identity.planned_ordered_training_stream_sha256:
            raise P50GateError("P50 observed ordered training stream differs from its frozen plan")
        for family in families:
            if gate_updates[family] < minimum[family]:
                raise P50GateError(f"P50 family-gate gradient threshold failed for {family}")
            if route_updates[family] < minimum[family]:
                raise P50GateError(f"P50 action-route gradient threshold failed for {family}")
            nll_key = f"canonical_successor_nll_{family}"
            regression = final[nll_key] - baseline[nll_key]
            if regression > maximum_regressions[family]:
                raise P50GateError(f"P50 family NLL regression threshold failed for {family}")
        if failure_reason is not None:
            raise P50GateError("P50 PASS cannot carry a failure reason")
    else:
        _require_nonempty_text(failure_reason, field="failure_reason")

    artifact: dict[str, object] = {
        "schema": P50_EVIDENCE_SCHEMA,
        "schema_version": P50_EVIDENCE_SCHEMA_VERSION,
        "status": status,
        "run_identity": run_identity.to_payload(),
        "resolved_thresholds": resolved_p50_thresholds_payload(resolved_thresholds),
        "completed_optimizer_steps": completed,
        "checkpoint": checkpoint,
        "observed_ordered_training_stream_sha256": observed,
        "gradient_updates": {
            "family_gate_updates": gate_updates,
            "action_route_updates": route_updates,
        },
        "baseline_validation": baseline,
        "final_validation": final,
        "retention": retention.to_payload(),
        "failure_reason": failure_reason,
    }
    artifact["artifact_sha256"] = p50_evidence_self_hash(artifact)
    validate_p50_evidence(artifact, expected_run_identity=run_identity)
    return artifact


def _parse_run_identity(payload: object) -> P50RunIdentity:
    value = _require_exact_keys(payload, _RUN_IDENTITY_KEYS, field="run_identity")
    return P50RunIdentity(**dict(value))


def _parse_thresholds(payload: object) -> ResolvedP50Thresholds:
    value = _require_exact_keys(
        payload,
        _THRESHOLD_KEYS,
        field="resolved_thresholds",
    )
    initialization_regime = value["initialization_regime"]
    families = value["required_families"]
    minimum = value["minimum_gradient_updates"]
    regressions = value["maximum_family_nll_regression"]
    retention_status = value["inherited_retention_status"]
    if not isinstance(initialization_regime, str):
        raise P50GateError("resolved_thresholds.initialization_regime is malformed")
    if not isinstance(families, list) or not all(isinstance(family, str) for family in families):
        raise P50GateError("resolved_thresholds.required_families is malformed")
    if not isinstance(minimum, Mapping) or not isinstance(regressions, Mapping):
        raise P50GateError("resolved P50 family thresholds must be objects")
    required = tuple(families)
    _require_exact_keys(
        minimum, set(required), field="resolved_thresholds.minimum_gradient_updates"
    )
    _require_exact_keys(
        regressions,
        set(required),
        field="resolved_thresholds.maximum_family_nll_regression",
    )
    if not isinstance(retention_status, str):
        raise P50GateError("resolved_thresholds.inherited_retention_status is malformed")
    inherited = value["maximum_inherited_probe_nll_regression"]
    thresholds = ResolvedP50Thresholds(
        initialization_regime=initialization_regime,
        required_families=required,
        minimum_gradient_updates=tuple((family, minimum[family]) for family in required),
        maximum_family_nll_regression=tuple((family, regressions[family]) for family in required),
        inherited_retention_status=retention_status,
        maximum_inherited_probe_nll_regression=inherited,
    )
    _validate_resolved_thresholds(thresholds)
    return thresholds


def _parse_retention(payload: object) -> P50RetentionEvidence:
    value = _require_exact_keys(payload, _RETENTION_KEYS, field="retention")
    return P50RetentionEvidence(**dict(value))


def _parse_validation(
    payload: object,
    *,
    families: tuple[str, ...],
    field: str,
    allow_null: bool,
) -> dict[str, float] | None:
    if payload is None and allow_null:
        return None
    if not isinstance(payload, Mapping):
        raise P50GateError(f"{field} must be an object")
    expected = _required_validation_metric_keys(families)
    _require_exact_keys(payload, expected, field=field)
    return _validation_payload(payload, families=families, field=field)


def validate_p50_evidence(
    evidence: Mapping[str, Any],
    *,
    expected_run_identity: P50RunIdentity | None = None,
    checkpoint_path: str | Path | None = None,
) -> P50Evidence:
    """Strictly validate schema, semantics, self-hash, and optional file bytes."""

    root = _require_exact_keys(evidence, _TOP_LEVEL_KEYS, field="P50 evidence")
    if root["schema"] != P50_EVIDENCE_SCHEMA:
        raise P50GateError("unexpected P50 evidence schema")
    if root["schema_version"] != P50_EVIDENCE_SCHEMA_VERSION:
        raise P50GateError("unexpected P50 evidence schema version")
    status = root["status"]
    if status not in {"PASS", "FAIL"}:
        raise P50GateError("P50 evidence status must be PASS or FAIL")
    claimed_hash = _require_sha256(
        root["artifact_sha256"],
        field="artifact_sha256",
    )
    if p50_evidence_self_hash(root) != claimed_hash:
        raise P50GateError("P50 evidence self-hash mismatch")

    run_identity = _parse_run_identity(root["run_identity"])
    if expected_run_identity is not None and run_identity != expected_run_identity:
        raise P50GateError("P50 evidence run identity mismatch")
    thresholds = _parse_thresholds(root["resolved_thresholds"])
    minimum, maximum_regressions = _validate_resolved_thresholds(thresholds)
    if run_identity.resolved_thresholds_sha256 != resolved_p50_thresholds_sha256(thresholds):
        raise P50GateError("P50 resolved-threshold hash mismatch")
    if run_identity.initialization_regime != thresholds.initialization_regime:
        raise P50GateError("P50 threshold initialization regime mismatch")

    completed = _require_nonnegative_int(
        root["completed_optimizer_steps"],
        field="completed_optimizer_steps",
    )
    if completed > P50_EXPECTED_OPTIMIZER_STEPS:
        raise P50GateError("P50 evidence exceeds its step ceiling")
    gradients = _require_exact_keys(
        root["gradient_updates"],
        _GRADIENT_KEYS,
        field="gradient_updates",
    )
    gate_updates = _gradient_payload(
        gradients["family_gate_updates"],
        families=thresholds.required_families,
        field="family_gate_updates",
        completed_steps=completed,
    )
    route_updates = _gradient_payload(
        gradients["action_route_updates"],
        families=thresholds.required_families,
        field="action_route_updates",
        completed_steps=completed,
    )
    baseline = _parse_validation(
        root["baseline_validation"],
        families=thresholds.required_families,
        field="baseline_validation",
        allow_null=False,
    )
    assert baseline is not None
    final = _parse_validation(
        root["final_validation"],
        families=thresholds.required_families,
        field="final_validation",
        allow_null=True,
    )
    retention = _parse_retention(root["retention"])
    _validate_retention(
        retention,
        run_identity=run_identity,
        thresholds=thresholds,
        status=status,
    )

    checkpoint = _require_exact_keys(
        root["checkpoint"],
        _CHECKPOINT_KEYS,
        field="checkpoint",
    )
    _validate_checkpoint_metadata(
        checkpoint,
        completed_steps=completed,
        status=status,
    )
    observed_raw = root["observed_ordered_training_stream_sha256"]
    observed = (
        None
        if observed_raw is None
        else _require_sha256(
            observed_raw,
            field="observed_ordered_training_stream_sha256",
        )
    )
    failure_reason = root["failure_reason"]
    if status == "PASS":
        if completed != P50_EXPECTED_OPTIMIZER_STEPS or final is None:
            raise P50GateError("P50 PASS is incomplete")
        if observed != run_identity.planned_ordered_training_stream_sha256:
            raise P50GateError("P50 PASS has an ordered-training-stream mismatch")
        if failure_reason is not None:
            raise P50GateError("P50 PASS carries a failure reason")
        for family in thresholds.required_families:
            if gate_updates[family] < minimum[family] or route_updates[family] < minimum[family]:
                raise P50GateError(f"P50 gradient thresholds failed for {family}")
            nll_key = f"canonical_successor_nll_{family}"
            if final[nll_key] - baseline[nll_key] > maximum_regressions[family]:
                raise P50GateError(f"P50 validation threshold failed for {family}")
    else:
        _require_nonempty_text(failure_reason, field="failure_reason")

    if checkpoint_path is not None:
        inspected = _inspect_current_checkpoint(Path(checkpoint_path))
        if dict(checkpoint) != inspected:
            raise P50GateError("P50 checkpoint file no longer matches sealed evidence")

    return P50Evidence(
        status=status,
        run_identity=run_identity,
        resolved_thresholds=thresholds,
        completed_optimizer_steps=completed,
        checkpoint=dict(checkpoint),
        observed_ordered_training_stream_sha256=observed,
        family_gate_updates=gate_updates,
        action_route_updates=route_updates,
        baseline_validation=baseline,
        final_validation=final,
        retention=retention,
        failure_reason=failure_reason,
        artifact_sha256=claimed_hash,
    )


def p50_pass_authorizes_p500(
    evidence: Mapping[str, Any],
    *,
    expected_run_identity: P50RunIdentity,
    checkpoint_path: str | Path,
) -> bool:
    """Return true only for a fully verified PASS bound to the current file."""

    validated = validate_p50_evidence(
        evidence,
        expected_run_identity=expected_run_identity,
    )
    if validated.status != "PASS":
        return False
    validate_p50_evidence(
        evidence,
        expected_run_identity=expected_run_identity,
        checkpoint_path=checkpoint_path,
    )
    return True


__all__ = [
    "P50_BALANCED_METRIC",
    "P50_CURRENT_STATE_SOURCE",
    "P50_EVIDENCE_SCHEMA",
    "P50_EVIDENCE_SCHEMA_VERSION",
    "P50_EXPECTED_OPTIMIZER_STEPS",
    "P50_SELECTION_METRIC",
    "P50Evidence",
    "P50GateError",
    "P50RetentionEvidence",
    "P50RunIdentity",
    "ResolvedP50Thresholds",
    "p50_evidence_self_hash",
    "p50_pass_authorizes_p500",
    "resolved_p50_thresholds_payload",
    "resolved_p50_thresholds_sha256",
    "seal_p50_evidence",
    "state_dict_semantic_sha256",
    "validate_p50_evidence",
]
