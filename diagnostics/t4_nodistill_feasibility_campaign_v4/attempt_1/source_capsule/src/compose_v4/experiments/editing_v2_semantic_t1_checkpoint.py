"""Strict semantic validation for the selected Editing-V2 T1 checkpoint."""

from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from compose_v4.experiments.editing_p50_gate import (
    P50GateError,
    state_dict_semantic_sha256,
)

CHECKPOINT_SCHEMA = "compose.editing_v2.semantic_t1_capacity_checkpoint"
CHECKPOINT_SCHEMA_VERSION = 3
CHECKPOINT_STATUS = "RECOVERABLE_T1_CAPACITY_STATE_NO_DOWNSTREAM_AUTHORITY"
NO_AUTHORITY = {
    "training_authorized": False,
    "bounded_p50_authorized": False,
    "long_training_authorized": False,
    "checkpoint_selection_authorized": False,
    "final_test_selection_authorized": False,
}
SELECTED_CHECKPOINT_FIELDS = {
    "schema",
    "schema_version",
    "status",
    *NO_AUTHORITY,
    "identity",
    "selected_step",
    "model_state",
    "model_state_sha256",
    "stream_sha256",
    "torch_version",
}
CHECKPOINT_IDENTITY_FIELDS = {
    "capacity_policy_sha256",
    "optimization_policy",
    "optimization_policy_sha256",
    "prepared_input_artifact_sha256",
    "cache_completion_sha256",
    "cache_manifest_sha256",
    "initial_model_state_sha256",
    "runner_implementation_sha256",
    "runner_source_revision_sha256",
    "execution_environment",
    "execution_environment_sha256",
    "model_device",
    "model_device_type",
    "model_device_index",
    "model_dtype",
    "deterministic_algorithms_enabled",
    "optimizer_configuration",
    "optimizer_configuration_sha256",
}


class SemanticT1SelectedCheckpointError(RuntimeError):
    """The selected T1 checkpoint is malformed or differs from its result chain."""


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
        raise SemanticT1SelectedCheckpointError(
            "selected T1 checkpoint metadata is not canonical finite JSON"
        ) from error


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def validate_semantic_t1_selected_checkpoint(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_selected_step: int,
    expected_model_state_sha256: str,
    expected_stream_sha256: str,
    expected_identity_fields: Mapping[str, object],
) -> dict[str, Any]:
    """Reopen a selected-only checkpoint and bind it to the result provenance."""

    source = Path(path)
    try:
        physical_sha256 = _file_sha(source)
    except OSError as error:
        raise SemanticT1SelectedCheckpointError("selected T1 checkpoint cannot be read") from error
    if not _is_sha(expected_file_sha256) or physical_sha256 != expected_file_sha256:
        raise SemanticT1SelectedCheckpointError(
            "selected T1 checkpoint physical SHA-256 differs from the result"
        )
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except (EOFError, OSError, pickle.PickleError, RuntimeError, TypeError, ValueError) as error:
        raise SemanticT1SelectedCheckpointError(
            "selected T1 checkpoint cannot be decoded"
        ) from error
    if (
        not isinstance(payload, dict)
        or set(payload) != SELECTED_CHECKPOINT_FIELDS
        or payload.get("schema") != CHECKPOINT_SCHEMA
        or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or payload.get("status") != CHECKPOINT_STATUS
        or any(payload.get(name) is not expected for name, expected in NO_AUTHORITY.items())
        or type(payload.get("selected_step")) is not int
        or payload.get("selected_step") != expected_selected_step
        or payload.get("stream_sha256") != expected_stream_sha256
        or payload.get("torch_version") != str(torch.__version__)
        or not _is_sha(payload.get("model_state_sha256"))
    ):
        raise SemanticT1SelectedCheckpointError(
            "selected T1 checkpoint schema, authority, step, stream, or runtime disagrees"
        )
    try:
        observed_model_state_sha256 = state_dict_semantic_sha256(payload.get("model_state"))
    except (P50GateError, TypeError, ValueError) as error:
        raise SemanticT1SelectedCheckpointError(
            "selected T1 checkpoint model state is invalid"
        ) from error
    if (
        observed_model_state_sha256 != payload["model_state_sha256"]
        or observed_model_state_sha256 != expected_model_state_sha256
    ):
        raise SemanticT1SelectedCheckpointError(
            "selected T1 checkpoint embedded model-state identity disagrees"
        )
    identity = payload.get("identity")
    if not isinstance(identity, Mapping) or set(identity) != CHECKPOINT_IDENTITY_FIELDS:
        raise SemanticT1SelectedCheckpointError("selected T1 checkpoint identity fields disagree")
    identity = dict(identity)
    if any(identity.get(name) != expected for name, expected in expected_identity_fields.items()):
        raise SemanticT1SelectedCheckpointError(
            "selected T1 checkpoint identity differs from reopened provenance"
        )
    environment = identity.get("execution_environment")
    optimization = identity.get("optimization_policy")
    optimizer_configuration = identity.get("optimizer_configuration")
    if not isinstance(environment, Mapping):
        raise SemanticT1SelectedCheckpointError(
            "selected T1 checkpoint execution environment is invalid"
        )
    environment_body = dict(environment)
    embedded_environment_sha256 = environment_body.pop("environment_sha256", None)
    expected_device_type = (
        "cuda"
        if environment.get("accelerator_class") == "gpu"
        else environment.get("accelerator_class")
    )
    model_device_index = identity.get("model_device_index")
    expected_model_device = (
        expected_device_type
        if model_device_index is None
        else f"{expected_device_type}:{model_device_index}"
    )
    if (
        embedded_environment_sha256 != _sha(environment_body)
        or identity.get("execution_environment_sha256") != embedded_environment_sha256
        or not isinstance(optimization, Mapping)
        or identity.get("optimization_policy_sha256") != _sha(optimization)
        or not isinstance(optimizer_configuration, Mapping)
        or identity.get("optimizer_configuration_sha256") != _sha(optimizer_configuration)
        or identity.get("deterministic_algorithms_enabled") is not True
        or identity.get("model_device_type") != expected_device_type
        or identity.get("model_device") != expected_model_device
        or identity.get("model_dtype") != "torch.float32"
        or type(model_device_index) not in (int, type(None))
    ):
        raise SemanticT1SelectedCheckpointError(
            "selected T1 checkpoint self-bound runtime identity disagrees"
        )
    return payload


__all__ = [
    "CHECKPOINT_IDENTITY_FIELDS",
    "CHECKPOINT_SCHEMA",
    "CHECKPOINT_SCHEMA_VERSION",
    "CHECKPOINT_STATUS",
    "NO_AUTHORITY",
    "SELECTED_CHECKPOINT_FIELDS",
    "SemanticT1SelectedCheckpointError",
    "validate_semantic_t1_selected_checkpoint",
]
