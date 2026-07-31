"""Prospective schedule-only refinement of the failed atom-insert V2 arm.

V3 changes only the optimizer learning-rate schedule after an exact replay to
the V2-selected update.  It preserves the V2 panel, model, objective, weights,
seed, hazard exclusion, selection rule, and thresholds.  Checkpoints contain
the complete continuation state and are published immutably with a verified
sidecar receipt.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from compose_v4.data.immutable_artifact import write_bytes_if_absent
from compose_v4.experiments.editing_p50_gate import state_dict_semantic_sha256
from compose_v4.experiments.editing_t1_uniform_successor_capacity_v2 import (
    EARLY_STOP_RULE,
    EXPECTED_V1_THRESHOLDS,
    SELECTION_RULE,
    TOTAL_HAZARD_PARAMETER_PREFIX,
    _clone_state_dict,
    _hazard_parameter_state,
    _require_hazard_bit_identity,
    _require_production_point_consistency,
    _required_component_prefixes,
    _selected_metric_checks,
    checkpoint_is_better,
    is_sha256,
    probability_implication_checks,
    stable_sha256,
)
from compose_v4.experiments.factorized_successor_training import (
    factorized_successor_identity_loss,
    forward_teacher_successor_batch,
)
from compose_v4.experiments.successor_micro_overfit import (
    PreparedSuccessorPanel,
    SuccessorMicroOverfitError,
    configure_micro_overfit_parameters,
    successor_panel_metrics,
)
from compose_v4.model.factorized_tracelet_rate_model import FactorizedTraceletRateModel

V3_CONTRACT_SCHEMA = "compose.editing.t1_atom_insert_schedule_contract"
V3_CONTRACT_VERSION = 3
V3_CONTRACT_STATUS = "FROZEN_PROSPECTIVE_OPTIMIZATION_ONLY_DIAGNOSTIC_NO_TRAINING_AUTHORITY"
V3_RESULT_SCHEMA = "compose.editing.t1_atom_insert_schedule_result"
V3_RESULT_VERSION = 3
V3_RESULT_STATUS = "ATOM_INSERT_SCHEDULE_V3_DIAGNOSTIC_COMPLETE_NO_GATE_DECISION"
V3_CHECKPOINT_SCHEMA = "compose.editing.t1_atom_insert_schedule_checkpoint"
V3_CHECKPOINT_VERSION = 1
V3_CHECKPOINT_RECEIPT_SCHEMA = "compose.editing.t1_atom_insert_schedule_checkpoint_receipt"
V3_CHECKPOINT_RECEIPT_VERSION = 1
EXPECTED_V3_CONTRACT_SHA256 = "15933e0859afe938a9a079fa69c2623a381f23d1334797cebb0895ea4ed25fd4"


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def load_atom_insert_schedule_v3_contract(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(Path(path).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SuccessorMicroOverfitError(
            f"atom-insert V3 contract is unreadable: {path}"
        ) from error
    if not isinstance(payload, dict):
        raise SuccessorMicroOverfitError("atom-insert V3 contract must be one object")
    body = {key: value for key, value in payload.items() if key != "contract_sha256"}
    optimization = payload.get("optimization_law")
    checkpoint = payload.get("checkpoint_policy")
    evaluation = payload.get("evaluation")
    if (
        payload.get("schema") != V3_CONTRACT_SCHEMA
        or payload.get("schema_version") != V3_CONTRACT_VERSION
        or payload.get("status") != V3_CONTRACT_STATUS
        or payload.get("training_authorized") is not False
        or payload.get("bounded_p50_authorized") is not False
        or payload.get("gate_decision_emitted") is not False
        or payload.get("family") != "atom_insert"
        or payload.get("contract_sha256") != stable_sha256(body)
        or payload.get("contract_sha256") != EXPECTED_V3_CONTRACT_SHA256
        or not isinstance(optimization, Mapping)
        or not isinstance(checkpoint, Mapping)
        or not isinstance(evaluation, Mapping)
    ):
        raise SuccessorMicroOverfitError("atom-insert V3 contract identity or status is invalid")
    phase1 = optimization.get("phase_1")
    phase2 = optimization.get("phase_2")
    if (
        optimization.get("objective") != "productive_embedded_canonical_successor_nll"
        or optimization.get("scope") != "all"
        or optimization.get("optimizer") != "AdamW"
        or optimization.get("optimizer_betas") != [0.9, 0.999]
        or optimization.get("optimizer_epsilon") != 1e-8
        or optimization.get("retain_optimizer_state_across_phase_boundary") is not True
        or optimization.get("weight_decay") != 0.0
        or optimization.get("gradient_clip_norm") != 10.0
        or optimization.get("seed") != 20260730
        or optimization.get("teacher_rate") != 1.0
        or optimization.get("importance_coefficient") != 1.0
        or optimization.get("hazard_included") is not False
        or optimization.get("excluded_parameter_prefixes") != [TOTAL_HAZARD_PARAMETER_PREFIX]
        or optimization.get("require_excluded_parameter_bit_identity") is not True
        or not isinstance(phase1, Mapping)
        or not isinstance(phase2, Mapping)
        or phase1.get("completed_updates") != 495
        or phase1.get("learning_rate") != 0.001
        or not is_sha256(phase1.get("required_initial_model_state_sha256"))
        or not is_sha256(phase1.get("required_terminal_model_state_sha256"))
        or phase2.get("maximum_additional_updates") != 250
        or phase2.get("learning_rate") != 0.0001
        or optimization.get("maximum_total_updates") != 745
        or optimization.get("checkpoint_selection") != SELECTION_RULE
        or optimization.get("patience_or_plateau_stop") is not False
        or checkpoint.get("required_updates") != [495, 545, 595, 645, 695, 745]
        or checkpoint.get("maximum_rework_updates") != 50
        or evaluation.get("thresholds") != EXPECTED_V1_THRESHOLDS
    ):
        raise SuccessorMicroOverfitError("atom-insert V3 changes its frozen optimization law")
    return payload


def _semantic_digest_update(digest: Any, value: object) -> None:
    if isinstance(value, Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(_canonical_json_bytes(list(tensor.shape)))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _semantic_digest_update(digest, key)
            _semantic_digest_update(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"sequence\0")
        digest.update(str(len(value)).encode("ascii"))
        for item in value:
            _semantic_digest_update(digest, item)
        return
    if value is None:
        digest.update(b"none\0")
        return
    if isinstance(value, bytes):
        digest.update(b"bytes\0")
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
        return
    if isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise SuccessorMicroOverfitError("checkpoint contains a nonfinite scalar")
        digest.update(type(value).__name__.encode("ascii") + b"\0")
        digest.update(_canonical_json_bytes(value))
        return
    raise TypeError(f"unsupported checkpoint semantic value: {type(value).__name__}")


def checkpoint_semantic_sha256(value: object) -> str:
    digest = hashlib.sha256()
    _semantic_digest_update(digest, value)
    return digest.hexdigest()


def _torch_save_bytes(payload: object) -> bytes:
    output = io.BytesIO()
    torch.save(payload, output)
    return output.getvalue()


def publish_v3_checkpoint(
    root: Path,
    body: Mapping[str, object],
) -> dict[str, object]:
    """Publish and immediately round-trip one complete immutable checkpoint."""

    if body.get("schema") != V3_CHECKPOINT_SCHEMA or body.get("schema_version") != 1:
        raise SuccessorMicroOverfitError("atom-insert V3 checkpoint body has invalid identity")
    semantic_sha256 = checkpoint_semantic_sha256(body)
    payload = {**body, "checkpoint_semantic_sha256": semantic_sha256}
    encoded = _torch_save_bytes(payload)
    file_sha256 = hashlib.sha256(encoded).hexdigest()
    update = int(body["completed_update"])
    stem = f"update_{update:04d}.{semantic_sha256[:16]}.{file_sha256[:16]}"
    checkpoint_path = Path(root) / f"{stem}.pt"
    write_bytes_if_absent(checkpoint_path, encoded)
    observed = checkpoint_path.read_bytes()
    if hashlib.sha256(observed).hexdigest() != file_sha256:
        raise SuccessorMicroOverfitError("atom-insert V3 checkpoint file hash changed")
    loaded = torch.load(io.BytesIO(observed), map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict):
        raise SuccessorMicroOverfitError("atom-insert V3 checkpoint round-trip is not an object")
    loaded_hash = loaded.pop("checkpoint_semantic_sha256", None)
    if loaded_hash != semantic_sha256 or checkpoint_semantic_sha256(loaded) != semantic_sha256:
        raise SuccessorMicroOverfitError("atom-insert V3 checkpoint semantic round-trip failed")
    receipt_body = {
        "schema": V3_CHECKPOINT_RECEIPT_SCHEMA,
        "schema_version": V3_CHECKPOINT_RECEIPT_VERSION,
        "status": "COMPLETE_IMMUTABLE_ATOM_INSERT_V3_CHECKPOINT",
        "training_authorized": False,
        "bounded_p50_authorized": False,
        "completed_update": update,
        "reason": body["checkpoint_reason"],
        "checkpoint_path": checkpoint_path.name,
        "checkpoint_file_sha256": file_sha256,
        "checkpoint_file_bytes": len(encoded),
        "checkpoint_semantic_sha256": semantic_sha256,
        "model_state_sha256": body["model_state_sha256"],
        "optimizer_state_sha256": body["optimizer_state_sha256"],
        "rng_state_sha256": body["rng_state_sha256"],
        "identity_sha256": stable_sha256(body["identity"]),
    }
    receipt = {**receipt_body, "receipt_sha256": stable_sha256(receipt_body)}
    receipt_path = Path(root) / f"{stem}.json"
    write_bytes_if_absent(receipt_path, _canonical_json_bytes(receipt) + b"\n")
    return {
        **receipt,
        "checkpoint_local_path": str(checkpoint_path),
        "receipt_local_path": str(receipt_path),
    }


def load_v3_checkpoint(
    path: Path,
    *,
    expected_identity: Mapping[str, object],
) -> dict[str, object]:
    encoded = Path(path).read_bytes()
    receipt_path = Path(path).with_suffix(".json")
    try:
        receipt = json.loads(receipt_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SuccessorMicroOverfitError(
            "atom-insert V3 resume checkpoint receipt is unreadable"
        ) from error
    if not isinstance(receipt, dict):
        raise SuccessorMicroOverfitError("atom-insert V3 resume receipt is not an object")
    receipt_body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if (
        receipt.get("schema") != V3_CHECKPOINT_RECEIPT_SCHEMA
        or receipt.get("schema_version") != V3_CHECKPOINT_RECEIPT_VERSION
        or receipt.get("status") != "COMPLETE_IMMUTABLE_ATOM_INSERT_V3_CHECKPOINT"
        or receipt.get("training_authorized") is not False
        or receipt.get("bounded_p50_authorized") is not False
        or receipt.get("checkpoint_path") != Path(path).name
        or receipt.get("checkpoint_file_sha256") != hashlib.sha256(encoded).hexdigest()
        or receipt.get("checkpoint_file_bytes") != len(encoded)
        or receipt.get("receipt_sha256") != stable_sha256(receipt_body)
    ):
        raise SuccessorMicroOverfitError("atom-insert V3 resume receipt identity is invalid")
    payload = torch.load(io.BytesIO(encoded), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise SuccessorMicroOverfitError("atom-insert V3 resume checkpoint is not an object")
    semantic_sha256 = payload.pop("checkpoint_semantic_sha256", None)
    if (
        payload.get("schema") != V3_CHECKPOINT_SCHEMA
        or payload.get("schema_version") != V3_CHECKPOINT_VERSION
        or payload.get("identity") != dict(expected_identity)
        or not is_sha256(semantic_sha256)
        or receipt.get("checkpoint_semantic_sha256") != semantic_sha256
        or checkpoint_semantic_sha256(payload) != semantic_sha256
    ):
        raise SuccessorMicroOverfitError("atom-insert V3 resume checkpoint identity is invalid")
    if payload.get("model_state_sha256") != state_dict_semantic_sha256(payload["model_state"]):
        raise SuccessorMicroOverfitError("atom-insert V3 resume model hash is invalid")
    if payload.get("optimizer_state_sha256") != checkpoint_semantic_sha256(
        payload["optimizer_state"]
    ):
        raise SuccessorMicroOverfitError("atom-insert V3 resume optimizer hash is invalid")
    rng_body = {
        "cpu": payload["cpu_rng_state"],
        "cuda": payload["cuda_rng_states"],
    }
    if payload.get("rng_state_sha256") != checkpoint_semantic_sha256(rng_body):
        raise SuccessorMicroOverfitError("atom-insert V3 resume RNG hash is invalid")
    payload["_resume_receipt"] = {
        **receipt,
        "checkpoint_local_path": str(Path(path)),
        "receipt_local_path": str(receipt_path),
    }
    return payload


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(value)


def _current_rng_state() -> tuple[Tensor, list[Tensor]]:
    return torch.get_rng_state().clone(), [
        state.clone() for state in torch.cuda.get_rng_state_all()
    ]


def _restore_rng_state(cpu_state: Tensor, cuda_states: Sequence[Tensor]) -> None:
    torch.set_rng_state(cpu_state.cpu())
    if cuda_states:
        if not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count():
            raise SuccessorMicroOverfitError("atom-insert V3 CUDA RNG device identity changed")
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])


def train_atom_insert_schedule_v3(
    model: FactorizedTraceletRateModel,
    panel: PreparedSuccessorPanel,
    *,
    phase1_updates: int,
    phase1_learning_rate: float,
    phase2_updates: int,
    phase2_learning_rate: float,
    expected_initial_model_sha256: str,
    expected_phase1_model_sha256: str | None,
    weight_decay: float,
    scope: str,
    seed: int,
    gradient_clip_norm: float,
    thresholds: Mapping[str, object],
    checkpoint_root: Path,
    checkpoint_identity: Mapping[str, object],
    checkpoint_interval: int = 50,
    resume_checkpoint: Path | None = None,
    checkpoint_publish_callback: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Run or exactly resume the bounded two-phase atom-insert diagnostic."""

    if (
        phase1_updates <= 0
        or phase2_updates <= 0
        or phase1_learning_rate <= 0
        or phase2_learning_rate <= 0
        or checkpoint_interval <= 0
        or dict(thresholds) != EXPECTED_V1_THRESHOLDS
    ):
        raise ValueError("invalid atom-insert V3 schedule")
    families = tuple(sorted({row.family_name for row in panel.examples}))
    if families != ("atom_insert",):
        raise SuccessorMicroOverfitError("atom-insert V3 requires exactly atom_insert")
    maximum_updates = phase1_updates + phase2_updates
    torch.manual_seed(int(seed))
    configured_names = configure_micro_overfit_parameters(model, families, scope=scope)
    initial_hazard_state = _hazard_parameter_state(model)
    for name, parameter in model.named_parameters():
        if name.startswith(TOTAL_HAZARD_PARAMETER_PREFIX):
            parameter.requires_grad_(False)
            parameter.grad = None
    trainable_names = tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    excluded_hazard_names = tuple(sorted(set(configured_names) - set(trainable_names)))
    if excluded_hazard_names != tuple(sorted(initial_hazard_state)):
        raise SuccessorMicroOverfitError("atom-insert V3 failed exact hazard exclusion")
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(phase1_learning_rate),
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=float(weight_decay),
    )
    device = next(model.parameters()).device
    device_batch = panel.batch.to(device)
    observed_initial_sha256 = state_dict_semantic_sha256(model.state_dict())
    initial_hazard_state_sha256 = state_dict_semantic_sha256(initial_hazard_state)
    if observed_initial_sha256 != expected_initial_model_sha256:
        raise SuccessorMicroOverfitError("atom-insert V3 initial model hash differs from V2")

    required_component_prefixes = _required_component_prefixes(families)
    initial_metrics = successor_panel_metrics(model, panel, include_per_example=False)
    finite_nonzero_gradient_seen = {name: False for name in trainable_names}
    gradient_update_counts = {name: 0 for name in trainable_names}
    component_gradient_update_counts = {component: 0 for component in required_component_prefixes}
    optimizer_steps_with_nonzero_gradient = 0
    trajectory: list[dict[str, object]] = []
    selected_criterion_point: dict[str, object] | None = None
    selected_update: int | None = None
    selected_state: dict[str, Tensor] | None = None
    completed_updates = 0
    observed_phase1_model_sha256: str | None = None
    checkpoint_receipts: list[dict[str, object]] = []

    if resume_checkpoint is not None:
        resumed = load_v3_checkpoint(resume_checkpoint, expected_identity=checkpoint_identity)
        resume_receipt = resumed.pop("_resume_receipt")
        model.load_state_dict(resumed["model_state"], strict=True)
        optimizer.load_state_dict(resumed["optimizer_state"])
        _restore_rng_state(resumed["cpu_rng_state"], resumed["cuda_rng_states"])
        completed_updates = int(resumed["completed_update"])
        trajectory = copy.deepcopy(resumed["trajectory"])
        selected_criterion_point = copy.deepcopy(resumed["selected_criterion_point"])
        selected_update = int(resumed["selected_update"])
        selected_state = copy.deepcopy(resumed["selected_state"])
        finite_nonzero_gradient_seen = dict(resumed["finite_nonzero_gradient_seen"])
        gradient_update_counts = dict(resumed["gradient_update_counts"])
        component_gradient_update_counts = dict(resumed["component_gradient_update_counts"])
        optimizer_steps_with_nonzero_gradient = int(
            resumed["optimizer_steps_with_nonzero_gradient"]
        )
        initial_metrics = copy.deepcopy(resumed["initial_metrics"])
        observed_phase1_model_sha256 = resumed["phase1_model_state_sha256"]
        checkpoint_receipts = copy.deepcopy(resumed["checkpoint_receipts"])
        if not any(
            receipt.get("checkpoint_semantic_sha256")
            == resume_receipt["checkpoint_semantic_sha256"]
            for receipt in checkpoint_receipts
        ):
            checkpoint_receipts.append(copy.deepcopy(resume_receipt))
        if len(trajectory) != completed_updates:
            raise SuccessorMicroOverfitError(
                "atom-insert V3 resumes only pre-evaluation continuation checkpoints"
            )
        if completed_updates < phase1_updates:
            expected_lr = phase1_learning_rate
        else:
            expected_lr = phase2_learning_rate
        if any(float(group["lr"]) != float(expected_lr) for group in optimizer.param_groups):
            raise SuccessorMicroOverfitError("atom-insert V3 resume learning rate is invalid")
        if completed_updates >= phase1_updates and (
            expected_phase1_model_sha256 is None
            or observed_phase1_model_sha256 != expected_phase1_model_sha256
        ):
            raise SuccessorMicroOverfitError("atom-insert V3 resume lacks the exact V2 boundary")
        _require_hazard_bit_identity(model, initial_hazard_state, state_label="resume")

    def publish(reason: str) -> None:
        nonlocal checkpoint_receipts
        cpu_rng, cuda_rng = _current_rng_state()
        model_state = _clone_state_dict(model)
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        rng_body = {"cpu": cpu_rng, "cuda": cuda_rng}
        body: dict[str, object] = {
            "schema": V3_CHECKPOINT_SCHEMA,
            "schema_version": V3_CHECKPOINT_VERSION,
            "status": "COMPLETE_ATOM_INSERT_V3_CONTINUATION_STATE",
            "training_authorized": False,
            "bounded_p50_authorized": False,
            "identity": dict(checkpoint_identity),
            "checkpoint_reason": reason,
            "completed_update": completed_updates,
            "phase": "phase_1" if completed_updates < phase1_updates else "phase_2",
            "model_state": model_state,
            "model_state_sha256": state_dict_semantic_sha256(model_state),
            "optimizer_state": optimizer_state,
            "optimizer_state_sha256": checkpoint_semantic_sha256(optimizer_state),
            "cpu_rng_state": cpu_rng,
            "cuda_rng_states": cuda_rng,
            "rng_state_sha256": checkpoint_semantic_sha256(rng_body),
            "trajectory": copy.deepcopy(trajectory),
            "selected_criterion_point": copy.deepcopy(selected_criterion_point),
            "selected_update": selected_update,
            "selected_state": copy.deepcopy(selected_state),
            "finite_nonzero_gradient_seen": dict(finite_nonzero_gradient_seen),
            "gradient_update_counts": dict(gradient_update_counts),
            "component_gradient_update_counts": dict(component_gradient_update_counts),
            "optimizer_steps_with_nonzero_gradient": optimizer_steps_with_nonzero_gradient,
            "initial_metrics": copy.deepcopy(initial_metrics),
            "initial_model_state_sha256": observed_initial_sha256,
            "phase1_model_state_sha256": observed_phase1_model_sha256,
            "initial_hazard_state": copy.deepcopy(initial_hazard_state),
            "checkpoint_receipts": copy.deepcopy(checkpoint_receipts),
        }
        checkpoint_receipts.append(publish_v3_checkpoint(checkpoint_root, body))
        if checkpoint_publish_callback is not None:
            checkpoint_publish_callback()

    model.train()
    stop_reason = "MAXIMUM_TOTAL_UPDATES_REACHED"
    while True:
        if completed_updates == phase1_updates:
            observed_boundary = state_dict_semantic_sha256(model.state_dict())
            if (
                expected_phase1_model_sha256 is not None
                and observed_boundary != expected_phase1_model_sha256
            ):
                raise SuccessorMicroOverfitError(
                    "atom-insert V3 update-495 model hash differs from V2"
                )
            observed_phase1_model_sha256 = observed_boundary
            if any(
                float(group["lr"]) != float(phase2_learning_rate)
                for group in optimizer.param_groups
            ):
                _set_optimizer_lr(optimizer, phase2_learning_rate)
            if (
                not checkpoint_receipts
                or int(checkpoint_receipts[-1]["completed_update"]) != completed_updates
            ):
                publish("PHASE_BOUNDARY")

        optimizer.zero_grad(set_to_none=True)
        prediction = forward_teacher_successor_batch(model, device_batch, panel.fibers)
        log_probabilities = prediction.selected_productive_successor_log_probability
        probabilities = log_probabilities.exp()
        implication_checks, point_metrics = probability_implication_checks(
            probabilities,
            thresholds=thresholds,
            log_probabilities=log_probabilities,
        )
        missing_components = tuple(
            sorted(
                component
                for component, count in component_gradient_update_counts.items()
                if count == 0
            )
        )
        gradient_checks = {
            "minimum_nonzero_gradient_optimizer_steps": optimizer_steps_with_nonzero_gradient
            >= int(thresholds["minimum_nonzero_gradient_optimizer_steps"]),
            "require_every_declared_component_nonzero_gradient": (
                not bool(thresholds["require_every_declared_component_nonzero_gradient"])
                or not missing_components
            ),
        }
        point: dict[str, object] = {
            "update": completed_updates,
            **point_metrics,
            "minimum_teacher_successor_log_probability": float(log_probabilities.detach().min()),
            "probability_nll_top1_implication_checks": implication_checks,
            "optimizer_steps_with_nonzero_gradient_before_next_update": optimizer_steps_with_nonzero_gradient,
            "component_gradient_update_counts_before_next_update": dict(
                component_gradient_update_counts
            ),
            "gradient_gate_checks_before_next_update": gradient_checks,
            "required_components_without_gradient_before_next_update": list(missing_components),
            "all_early_stop_checks_pass": all(implication_checks.values())
            and all(gradient_checks.values()),
            "optimizer_update_executed_after_evaluation": False,
            "loss_for_executed_update": None,
            "gradient_norm_for_executed_update": None,
            "learning_rate_for_executed_update": None,
        }
        if checkpoint_is_better(point, selected_criterion_point):
            selected_criterion_point = copy.deepcopy(point)
            selected_update = completed_updates
            selected_state = _clone_state_dict(model)
        should_stop = completed_updates >= phase1_updates and bool(
            point["all_early_stop_checks_pass"]
        )
        at_limit = completed_updates == maximum_updates
        if should_stop or at_limit:
            trajectory.append(point)
            stop_reason = "ALL_FROZEN_EARLY_STOP_CHECKS_PROVEN" if should_stop else stop_reason
            publish("SUCCESS" if should_stop else "TERMINAL")
            break

        loss = factorized_successor_identity_loss(prediction, device_batch)
        if not bool(torch.isfinite(loss)):
            raise SuccessorMicroOverfitError("nonfinite atom-insert V3 successor loss")
        loss.backward()
        squared_gradient_norm = 0.0
        nonzero_gradient_names: set[str] = set()
        for name, parameter in model.named_parameters():
            if name.startswith(TOTAL_HAZARD_PARAMETER_PREFIX):
                if parameter.grad is not None:
                    raise SuccessorMicroOverfitError(
                        "frozen atom-insert V3 hazard received a gradient"
                    )
                continue
            if not parameter.requires_grad or parameter.grad is None:
                continue
            if not bool(torch.isfinite(parameter.grad).all()):
                raise SuccessorMicroOverfitError(f"nonfinite atom-insert V3 gradient: {name}")
            norm = float(parameter.grad.norm())
            squared_gradient_norm += norm * norm
            if norm > 0:
                finite_nonzero_gradient_seen[name] = True
                gradient_update_counts[name] += 1
                nonzero_gradient_names.add(name)
        if nonzero_gradient_names:
            optimizer_steps_with_nonzero_gradient += 1
        for component, prefixes in required_component_prefixes.items():
            if any(name.startswith(prefixes) for name in nonzero_gradient_names):
                component_gradient_update_counts[component] += 1
        current_lr = float(optimizer.param_groups[0]["lr"])
        if any(float(group["lr"]) != current_lr for group in optimizer.param_groups):
            raise SuccessorMicroOverfitError(
                "atom-insert V3 optimizer groups disagree on learning rate"
            )
        torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=float(gradient_clip_norm))
        optimizer.step()
        completed_updates += 1
        _require_hazard_bit_identity(
            model, initial_hazard_state, state_label=f"post_update_{completed_updates}"
        )
        point["optimizer_update_executed_after_evaluation"] = True
        point["loss_for_executed_update"] = float(loss.detach())
        point["gradient_norm_for_executed_update"] = squared_gradient_norm**0.5
        point["learning_rate_for_executed_update"] = current_lr
        trajectory.append(point)
        if (
            completed_updates > phase1_updates
            and (completed_updates - phase1_updates) % checkpoint_interval == 0
        ):
            publish("PERIODIC")

    if selected_state is None or selected_update is None or selected_criterion_point is None:
        raise AssertionError("atom-insert V3 selected no checkpoint")
    terminal_state = _clone_state_dict(model)
    terminal_model_state_sha256 = state_dict_semantic_sha256(terminal_state)
    terminal_hazard_state_sha256 = state_dict_semantic_sha256(_hazard_parameter_state(model))
    terminal_metrics = successor_panel_metrics(model, panel, include_per_example=True)
    _require_production_point_consistency(terminal_metrics, trajectory[-1], state_label="terminal")
    model.load_state_dict(selected_state, strict=True)
    selected_model_state_sha256 = state_dict_semantic_sha256(model.state_dict())
    selected_hazard_state_sha256 = state_dict_semantic_sha256(_hazard_parameter_state(model))
    hazard_bit_identity = (
        initial_hazard_state_sha256 == terminal_hazard_state_sha256 == selected_hazard_state_sha256
    )
    if not hazard_bit_identity:
        raise SuccessorMicroOverfitError("atom-insert V3 total-hazard parameters changed")
    selected_metrics = successor_panel_metrics(model, panel, include_per_example=True)
    selected_point = trajectory[selected_update]
    _require_production_point_consistency(selected_metrics, selected_point, state_label="selected")
    missing = tuple(
        sorted(
            component for component, count in component_gradient_update_counts.items() if count == 0
        )
    )
    checks, floors = _selected_metric_checks(
        selected_metrics,
        optimizer_steps_with_nonzero_gradient=optimizer_steps_with_nonzero_gradient,
        component_gradient_update_counts=component_gradient_update_counts,
        required_components_without_gradient=missing,
        thresholds=thresholds,
    )
    return {
        "schema": "compose.editing.t1_atom_insert_schedule_training_report",
        "schema_version": 1,
        "training_authorized": False,
        "bounded_p50_authorized": False,
        "family": "atom_insert",
        "scope": scope,
        "seed": seed,
        "phase1_updates": phase1_updates,
        "phase1_learning_rate": phase1_learning_rate,
        "phase2_maximum_updates": phase2_updates,
        "phase2_learning_rate": phase2_learning_rate,
        "completed_full_panel_updates": completed_updates,
        "early_stopped": completed_updates < maximum_updates,
        "stop_reason": stop_reason,
        "optimizer": "AdamW",
        "optimizer_state_retained_at_phase_boundary": True,
        "weight_decay": weight_decay,
        "gradient_clip_norm": gradient_clip_norm,
        "selection_rule": SELECTION_RULE,
        "early_stop_rule": EARLY_STOP_RULE,
        "initial": initial_metrics,
        "terminal": terminal_metrics,
        "selected": selected_metrics,
        "selected_update": selected_update,
        "terminal_update": completed_updates,
        "trajectory": trajectory,
        "trajectory_sha256": stable_sha256(trajectory),
        "trajectory_state_count": len(trajectory),
        "initial_model_state_sha256": observed_initial_sha256,
        "phase1_model_state_sha256": observed_phase1_model_sha256,
        "terminal_model_state_sha256": terminal_model_state_sha256,
        "selected_model_state_sha256": selected_model_state_sha256,
        "returned_model_state_sha256": selected_model_state_sha256,
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in trainable_parameters)
        ),
        "trainable_tensor_count": len(trainable_names),
        "trainable_parameter_names_sha256": stable_sha256(list(trainable_names)),
        "optimizer_steps_with_nonzero_gradient": optimizer_steps_with_nonzero_gradient,
        "gradient_update_counts": gradient_update_counts,
        "component_gradient_update_counts": component_gradient_update_counts,
        "required_components_without_gradient": list(missing),
        "never_received_nonzero_gradient": sorted(
            name for name, seen in finite_nonzero_gradient_seen.items() if not seen
        ),
        "thresholds": dict(thresholds),
        "threshold_checks": checks,
        "per_example_floor_metrics": floors,
        "all_threshold_checks_pass": all(checks.values()),
        "hazard_freeze": {
            "parameter_prefix": TOTAL_HAZARD_PARAMETER_PREFIX,
            "excluded_parameter_names": list(excluded_hazard_names),
            "excluded_parameter_count": len(excluded_hazard_names),
            "requires_grad": False,
            "optimizer_included": False,
            "gradient_observed": False,
            "bit_identity_required": True,
            "bit_identity_pass": hazard_bit_identity,
            "initial_parameter_state_sha256": initial_hazard_state_sha256,
            "terminal_parameter_state_sha256": terminal_hazard_state_sha256,
            "selected_parameter_state_sha256": selected_hazard_state_sha256,
        },
        "execution_environment": {
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda) if torch.version.cuda else None,
            "device_type": device.type,
            "device_index": device.index,
            "observed_gpu_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "model_parameter_dtype": str(next(model.parameters()).dtype),
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        },
        "checkpoint_receipts": checkpoint_receipts,
        "checkpoint_identity_sha256": stable_sha256(checkpoint_identity),
        "resumed_from_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
    }


def validate_atom_insert_v3_training_report(
    report: object,
    *,
    contract: Mapping[str, object],
) -> Mapping[str, object]:
    """Recompute the schedule, selection, threshold, and checkpoint invariants."""

    if not isinstance(report, Mapping):
        raise SuccessorMicroOverfitError("atom-insert V3 report must be one object")
    optimization = contract.get("optimization_law")
    if not isinstance(optimization, Mapping):
        raise SuccessorMicroOverfitError("atom-insert V3 contract optimization law is invalid")
    phase1 = optimization["phase_1"]
    phase2 = optimization["phase_2"]
    assert isinstance(phase1, Mapping)
    assert isinstance(phase2, Mapping)
    boundary = int(phase1["completed_updates"])
    maximum = int(optimization["maximum_total_updates"])
    completed = report.get("completed_full_panel_updates")
    trajectory = report.get("trajectory")
    if (
        report.get("schema") != "compose.editing.t1_atom_insert_schedule_training_report"
        or report.get("schema_version") != 1
        or report.get("training_authorized") is not False
        or report.get("bounded_p50_authorized") is not False
        or report.get("family") != "atom_insert"
        or report.get("scope") != optimization["scope"]
        or report.get("seed") != optimization["seed"]
        or report.get("phase1_updates") != boundary
        or report.get("phase1_learning_rate") != phase1["learning_rate"]
        or report.get("phase2_maximum_updates") != phase2["maximum_additional_updates"]
        or report.get("phase2_learning_rate") != phase2["learning_rate"]
        or report.get("optimizer") != "AdamW"
        or report.get("optimizer_state_retained_at_phase_boundary") is not True
        or report.get("selection_rule") != SELECTION_RULE
        or report.get("early_stop_rule") != EARLY_STOP_RULE
        or report.get("thresholds") != EXPECTED_V1_THRESHOLDS
        or not isinstance(completed, int)
        or not boundary <= completed <= maximum
        or not isinstance(trajectory, list)
        or len(trajectory) != completed + 1
        or report.get("trajectory_state_count") != len(trajectory)
        or report.get("trajectory_sha256") != stable_sha256(trajectory)
        or [point.get("update") for point in trajectory] != list(range(completed + 1))
        or report.get("phase1_model_state_sha256") != phase1["required_terminal_model_state_sha256"]
    ):
        raise SuccessorMicroOverfitError("atom-insert V3 report identity or schedule is invalid")
    for index, point in enumerate(trajectory):
        if not isinstance(point, Mapping):
            raise SuccessorMicroOverfitError("atom-insert V3 trajectory point is invalid")
        executed = index < completed
        expected_lr = (
            float(phase1["learning_rate"]) if index < boundary else float(phase2["learning_rate"])
        )
        if (
            point.get("optimizer_update_executed_after_evaluation") is not executed
            or (
                executed
                and (
                    point.get("learning_rate_for_executed_update") != expected_lr
                    or not isinstance(point.get("loss_for_executed_update"), float)
                    or not math.isfinite(float(point["loss_for_executed_update"]))
                    or not isinstance(point.get("gradient_norm_for_executed_update"), float)
                    or not math.isfinite(float(point["gradient_norm_for_executed_update"]))
                )
            )
            or (
                not executed
                and (
                    point.get("learning_rate_for_executed_update") is not None
                    or point.get("loss_for_executed_update") is not None
                    or point.get("gradient_norm_for_executed_update") is not None
                )
            )
            or (index < boundary and point.get("all_early_stop_checks_pass") is True)
        ):
            raise SuccessorMicroOverfitError("atom-insert V3 trajectory execution is invalid")
    recomputed_selected: Mapping[str, object] | None = None
    for point in trajectory:
        if checkpoint_is_better(point, recomputed_selected):
            recomputed_selected = point
    assert recomputed_selected is not None
    if report.get("selected_update") != recomputed_selected["update"]:
        raise SuccessorMicroOverfitError("atom-insert V3 selected checkpoint is invalid")
    selected = report.get("selected")
    terminal = report.get("terminal")
    if not isinstance(selected, Mapping) or not isinstance(terminal, Mapping):
        raise SuccessorMicroOverfitError("atom-insert V3 production metrics are invalid")
    _require_production_point_consistency(selected, recomputed_selected, state_label="selected")
    _require_production_point_consistency(terminal, trajectory[-1], state_label="terminal")
    component_counts = report.get("component_gradient_update_counts")
    missing = report.get("required_components_without_gradient")
    nonzero_steps = report.get("optimizer_steps_with_nonzero_gradient")
    if (
        not isinstance(component_counts, Mapping)
        or not isinstance(missing, list)
        or not isinstance(nonzero_steps, int)
    ):
        raise SuccessorMicroOverfitError("atom-insert V3 gradient evidence is invalid")
    checks, floors = _selected_metric_checks(
        selected,
        optimizer_steps_with_nonzero_gradient=nonzero_steps,
        component_gradient_update_counts={
            str(key): int(value) for key, value in component_counts.items()
        },
        required_components_without_gradient=tuple(str(value) for value in missing),
        thresholds=EXPECTED_V1_THRESHOLDS,
    )
    if (
        report.get("threshold_checks") != checks
        or report.get("per_example_floor_metrics") != floors
        or report.get("all_threshold_checks_pass") is not all(checks.values())
    ):
        raise SuccessorMicroOverfitError("atom-insert V3 threshold evidence is invalid")
    hazard = report.get("hazard_freeze")
    if (
        not isinstance(hazard, Mapping)
        or hazard.get("parameter_prefix") != TOTAL_HAZARD_PARAMETER_PREFIX
        or hazard.get("requires_grad") is not False
        or hazard.get("optimizer_included") is not False
        or hazard.get("gradient_observed") is not False
        or hazard.get("bit_identity_required") is not True
        or hazard.get("bit_identity_pass") is not True
        or not is_sha256(hazard.get("initial_parameter_state_sha256"))
        or len(
            {
                hazard.get("initial_parameter_state_sha256"),
                hazard.get("terminal_parameter_state_sha256"),
                hazard.get("selected_parameter_state_sha256"),
            }
        )
        != 1
    ):
        raise SuccessorMicroOverfitError("atom-insert V3 hazard-freeze evidence is invalid")
    passed = bool(report["all_threshold_checks_pass"])
    if (
        report.get("early_stopped") is not (completed < maximum)
        or (passed and report.get("stop_reason") != "ALL_FROZEN_EARLY_STOP_CHECKS_PROVEN")
        or (not passed and report.get("stop_reason") != "MAXIMUM_TOTAL_UPDATES_REACHED")
        or (passed and trajectory[-1].get("all_early_stop_checks_pass") is not True)
        or (not passed and completed != maximum)
    ):
        raise SuccessorMicroOverfitError("atom-insert V3 stop decision is invalid")
    receipts = report.get("checkpoint_receipts")
    if not isinstance(receipts, list) or not receipts:
        raise SuccessorMicroOverfitError("atom-insert V3 checkpoint evidence is absent")
    required_updates = {boundary}
    required_updates.update(range(boundary + 50, completed + 1, 50))
    observed_updates = {
        int(receipt["completed_update"]) for receipt in receipts if isinstance(receipt, Mapping)
    }
    for receipt in receipts:
        if not isinstance(receipt, Mapping):
            raise SuccessorMicroOverfitError("atom-insert V3 checkpoint receipt is malformed")
        receipt_body = {
            key: value
            for key, value in receipt.items()
            if key
            not in {
                "receipt_sha256",
                "checkpoint_local_path",
                "receipt_local_path",
            }
        }
        if receipt.get("receipt_sha256") != stable_sha256(receipt_body):
            raise SuccessorMicroOverfitError("atom-insert V3 checkpoint receipt hash is invalid")
    if not required_updates.issubset(observed_updates) or completed not in observed_updates:
        raise SuccessorMicroOverfitError("atom-insert V3 checkpoint schedule is incomplete")
    return report


__all__ = [
    "EXPECTED_V3_CONTRACT_SHA256",
    "V3_CONTRACT_SCHEMA",
    "V3_RESULT_SCHEMA",
    "V3_RESULT_STATUS",
    "V3_RESULT_VERSION",
    "checkpoint_semantic_sha256",
    "load_atom_insert_schedule_v3_contract",
    "load_v3_checkpoint",
    "publish_v3_checkpoint",
    "train_atom_insert_schedule_v3",
    "validate_atom_insert_v3_training_report",
]
