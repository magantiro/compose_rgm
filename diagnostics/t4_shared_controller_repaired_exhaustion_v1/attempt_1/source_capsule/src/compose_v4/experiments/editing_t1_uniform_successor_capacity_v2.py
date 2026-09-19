"""Prospective worst-example checkpointing for the uniform T1 diagnostic.

The V1 family-local diagnostic reported only its terminal checkpoint.  This
module leaves the frozen panel, objective, coefficients, and thresholds alone,
but records every full-panel pre-update state and selects the state with the
best bottleneck canonical-successor probability.  It is a bounded capacity
diagnostic and cannot authorize P50 or any longer training run.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from compose_v4.experiments.editing_p50_gate import state_dict_semantic_sha256
from compose_v4.experiments.factorized_successor_training import (
    factorized_successor_identity_loss,
    forward_teacher_successor_batch,
)
from compose_v4.experiments.successor_micro_overfit import (
    _HEAD_PREFIXES_BY_FAMILY,
    PreparedSuccessorPanel,
    RINGCORE_EDITING_FAMILIES,
    SuccessorMicroOverfitError,
    configure_micro_overfit_parameters,
    successor_panel_metrics,
)
from compose_v4.model.factorized_tracelet_rate_model import FactorizedTraceletRateModel

UNIFORM_V2_CONTRACT_SCHEMA = "compose.editing.t1_uniform_successor_capacity_contract"
UNIFORM_V2_CONTRACT_VERSION = 2
UNIFORM_V2_RESULT_SCHEMA = "compose.editing.t1_uniform_successor_capacity_result"
UNIFORM_V2_RESULT_VERSION = 2
UNIFORM_V2_RESULT_STATUS = (
    "UNIFORM_CANONICAL_SUCCESSOR_CAPACITY_V2_DIAGNOSTIC_COMPLETE_NO_GATE_DECISION"
)
UNIFORM_V2_CONTRACT_STATUS = "FROZEN_BOUNDED_DEVELOPMENT_DIAGNOSTIC_NO_TRAINING_AUTHORITY"
TOTAL_HAZARD_PARAMETER_PREFIX = "total_hazard_head."
SELECTION_RULE = (
    "maximize_minimum_per_example_productive_canonical_successor_probability"
    "__tie_lower_mean_canonical_successor_nll__tie_earlier_update"
)
EARLY_STOP_RULE = (
    "stop_before_next_update_only_if_all_frozen_probability_nll_top1_implications_"
    "and_gradient_component_gates_pass"
)
EXPECTED_UNIFORM_V2_CONTRACT_SHA256 = (
    "bd96dc3ac19e37daf0ec249e9470a8e12d6d3f699f9ec392dc9b57313d0ff60a"
)

EXPECTED_V1_THRESHOLDS: dict[str, float | int | bool] = {
    "minimum_unique_state_teacher_successor_top1": 0.95,
    "minimum_unique_state_teacher_successor_probability": 0.8,
    "maximum_unique_state_teacher_successor_nll": 0.22314355131420976,
    "require_every_example_teacher_successor_top1": True,
    "minimum_every_example_teacher_successor_probability": 0.8,
    "minimum_nonzero_gradient_optimizer_steps": 1,
    "require_every_declared_component_nonzero_gradient": True,
}
SERIALIZED_SOURCE_SUBTREES = (
    "src",
    "scripts",
    "modal_apps",
    "configs",
    "diagnostics/coherence",
)


def canonical_json_bytes(value: object) -> bytes:
    """Encode one deterministic, finite JSON value."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def stable_sha256(value: object) -> str:
    """Hash one deterministic JSON value."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def serialized_source_tree_sha256(root: Path) -> str:
    """Hash the exact serialized source tree while ignoring Python bytecode."""

    base = Path(root).resolve()
    digest = hashlib.sha256()
    seen: set[str] = set()
    for subtree in SERIALIZED_SOURCE_SUBTREES:
        source = base / subtree
        if not source.is_dir():
            raise SuccessorMicroOverfitError(
                f"uniform T1 V2 serialized source subtree is absent: {source}"
            )
        for path in sorted(candidate for candidate in source.rglob("*") if candidate.is_file()):
            relative = path.relative_to(base).as_posix()
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            if relative in seen:
                raise SuccessorMicroOverfitError(
                    f"uniform T1 V2 serialized source path is duplicated: {relative}"
                )
            seen.add(relative)
            encoded_path = relative.encode("utf-8")
            content = path.read_bytes()
            digest.update(len(encoded_path).to_bytes(8, "big"))
            digest.update(encoded_path)
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
    if not seen:
        raise SuccessorMicroOverfitError("uniform T1 V2 serialized source tree is empty")
    return digest.hexdigest()


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_uniform_capacity_v2_contract(path: Path) -> Mapping[str, object]:
    """Load and fully validate the frozen prospective V2 contract."""

    source = Path(path)
    try:
        payload = json.loads(source.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SuccessorMicroOverfitError(
            f"uniform T1 V2 contract is unreadable: {source}"
        ) from error
    if not isinstance(payload, dict):
        raise SuccessorMicroOverfitError("uniform T1 V2 contract must be one object")
    body = {key: value for key, value in payload.items() if key != "contract_sha256"}
    if (
        payload.get("schema") != UNIFORM_V2_CONTRACT_SCHEMA
        or payload.get("schema_version") != UNIFORM_V2_CONTRACT_VERSION
        or payload.get("status") != UNIFORM_V2_CONTRACT_STATUS
        or payload.get("training_authorized") is not False
        or payload.get("bounded_p50_authorized") is not False
        or payload.get("contract_sha256") != stable_sha256(body)
        or payload.get("contract_sha256") != EXPECTED_UNIFORM_V2_CONTRACT_SHA256
    ):
        raise SuccessorMicroOverfitError(
            "uniform T1 V2 contract identity, status, or self-hash is invalid"
        )
    families = payload.get("families")
    panel = payload.get("panel_policy")
    optimization = payload.get("optimization_law")
    evaluation = payload.get("evaluation")
    decision = payload.get("decision_policy")
    if (
        not isinstance(families, list)
        or tuple(families) != RINGCORE_EDITING_FAMILIES
        or not all(
            isinstance(value, Mapping) for value in (panel, optimization, evaluation, decision)
        )
    ):
        raise SuccessorMicroOverfitError("uniform T1 V2 contract sections are malformed")
    assert isinstance(panel, Mapping)
    assert isinstance(optimization, Mapping)
    assert isinstance(evaluation, Mapping)
    assert isinstance(decision, Mapping)
    if (
        panel.get("panel_kind") != "unique_state"
        or panel.get("examples_per_family") != 64
        or panel.get("duplicate_exact_source_time_policy") != "reject"
        or panel.get("panel_identity_change_from_v1") != "none"
        or optimization.get("objective") != "productive_embedded_canonical_successor_nll"
        or optimization.get("scope") != "all"
        or optimization.get("maximum_full_panel_updates") != 500
        or optimization.get("optimizer") != "AdamW"
        or optimization.get("schedule") != "constant"
        or optimization.get("seed") != 20260730
        or optimization.get("learning_rate") != 0.001
        or optimization.get("weight_decay") != 0.0
        or optimization.get("gradient_clip_norm") != 10.0
        or optimization.get("teacher_rate") != 1.0
        or optimization.get("importance_coefficient") != 1.0
        or optimization.get("hazard_included") is not False
        or optimization.get("excluded_parameter_prefixes") != [TOTAL_HAZARD_PARAMETER_PREFIX]
        or optimization.get("require_excluded_parameter_bit_identity") is not True
        or optimization.get("checkpoint_selection") != SELECTION_RULE
        or optimization.get("checkpoint_selection_numerics")
        != "compare_minimum_log_probability_as_the_monotone_nonunderflowing_"
        "representation_of_minimum_probability"
        or optimization.get("early_stop") != EARLY_STOP_RULE
        or evaluation.get("thresholds") != EXPECTED_V1_THRESHOLDS
        or decision.get("formal_gate_decision_emitted") is not False
        or decision.get("family_local_arms_alone_are_insufficient_for_p50") is not True
        or decision.get("v2_result_may_not_retroactively_change_v1") is not True
    ):
        raise SuccessorMicroOverfitError(
            "uniform T1 V2 contract changes its frozen capacity law or boundary"
        )
    return payload


def _clone_state_dict(model: torch.nn.Module) -> dict[str, Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def _hazard_parameter_state(model: torch.nn.Module) -> dict[str, Tensor]:
    state = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.startswith(TOTAL_HAZARD_PARAMETER_PREFIX)
    }
    if not state:
        raise SuccessorMicroOverfitError("uniform T1 V2 found no total-hazard parameters to freeze")
    return state


def _require_hazard_bit_identity(
    model: torch.nn.Module,
    reference: Mapping[str, Tensor],
    *,
    state_label: str,
) -> None:
    current = {
        name: parameter.detach()
        for name, parameter in model.named_parameters()
        if name.startswith(TOTAL_HAZARD_PARAMETER_PREFIX)
    }
    if tuple(sorted(current)) != tuple(sorted(reference)):
        raise SuccessorMicroOverfitError(
            f"total-hazard parameter identity changed at {state_label}"
        )
    changed = [
        name for name in sorted(reference) if not torch.equal(current[name], reference[name])
    ]
    if changed:
        raise SuccessorMicroOverfitError(
            f"total-hazard parameters changed at {state_label}: {changed}"
        )


def _required_component_prefixes(families: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    return {
        "family_head": ("family_head.",),
        **{family: _HEAD_PREFIXES_BY_FAMILY[family] for family in families},
    }


def probability_implication_checks(
    probabilities: Tensor,
    *,
    thresholds: Mapping[str, object],
    log_probabilities: Tensor | None = None,
) -> tuple[dict[str, bool], dict[str, float]]:
    """Prove every V1 probability, NLL, and top-1 implication from one vector.

    The productive successor law is normalized.  A target with probability
    greater than one half is therefore the unique top-1 successor.  The frozen
    per-example probability floor is 0.8, so satisfying it proves both top-1
    requirements without invoking the slower production ranking evaluator at
    every update.
    """

    if probabilities.ndim != 1 or probabilities.numel() == 0:
        raise SuccessorMicroOverfitError("uniform T1 V2 requires a nonempty probability vector")
    detached = probabilities.detach()
    if not bool(torch.isfinite(detached).all()) or not bool(
        ((detached >= 0.0) & (detached <= 1.0)).all()
    ):
        raise SuccessorMicroOverfitError("uniform T1 V2 observed invalid successor probabilities")
    minimum_probability = float(detached.min())
    mean_probability = float(detached.mean())
    if log_probabilities is None:
        nlls = -torch.log(detached.clamp_min(torch.finfo(detached.dtype).tiny))
    else:
        detached_logs = log_probabilities.detach()
        if (
            detached_logs.shape != detached.shape
            or not bool(torch.isfinite(detached_logs).all())
            or not bool((detached_logs <= 0.0).all())
        ):
            raise SuccessorMicroOverfitError(
                "uniform T1 V2 observed invalid successor log probabilities"
            )
        nlls = -detached_logs
    mean_nll = float(nlls.mean())
    maximum_nll = float(nlls.max())
    every_top1_proven = minimum_probability > 0.5
    checks = {
        "minimum_unique_state_teacher_successor_top1": (
            every_top1_proven
            and 1.0 >= float(thresholds["minimum_unique_state_teacher_successor_top1"])
        ),
        "minimum_unique_state_teacher_successor_probability": (
            mean_probability
            >= float(thresholds["minimum_unique_state_teacher_successor_probability"])
        ),
        "maximum_unique_state_teacher_successor_nll": (
            mean_nll <= float(thresholds["maximum_unique_state_teacher_successor_nll"])
        ),
        "require_every_example_teacher_successor_top1": (
            not bool(thresholds["require_every_example_teacher_successor_top1"])
            or every_top1_proven
        ),
        "minimum_every_example_teacher_successor_probability": (
            minimum_probability
            >= float(thresholds["minimum_every_example_teacher_successor_probability"])
        ),
    }
    metrics = {
        "minimum_teacher_successor_probability": minimum_probability,
        "mean_teacher_successor_probability": mean_probability,
        "mean_canonical_successor_nll": mean_nll,
        "maximum_teacher_successor_nll": maximum_nll,
    }
    return checks, metrics


def checkpoint_is_better(
    candidate: Mapping[str, object],
    incumbent: Mapping[str, object] | None,
) -> bool:
    """Apply the frozen lexicographic V2 selection rule without tolerance."""

    if incumbent is None:
        return True

    def bottleneck_log_probability(point: Mapping[str, object]) -> float:
        if "minimum_teacher_successor_log_probability" in point:
            return float(point["minimum_teacher_successor_log_probability"])
        probability = float(point["minimum_teacher_successor_probability"])
        return math.log(probability) if probability > 0.0 else -math.inf

    candidate_minimum = bottleneck_log_probability(candidate)
    incumbent_minimum = bottleneck_log_probability(incumbent)
    if candidate_minimum != incumbent_minimum:
        return candidate_minimum > incumbent_minimum
    candidate_nll = float(candidate["mean_canonical_successor_nll"])
    incumbent_nll = float(incumbent["mean_canonical_successor_nll"])
    if candidate_nll != incumbent_nll:
        return candidate_nll < incumbent_nll
    return int(candidate["update"]) < int(incumbent["update"])


def _selected_metric_checks(
    metrics: Mapping[str, object],
    *,
    optimizer_steps_with_nonzero_gradient: int,
    component_gradient_update_counts: Mapping[str, int],
    required_components_without_gradient: tuple[str, ...],
    thresholds: Mapping[str, object],
) -> tuple[dict[str, bool], dict[str, float | int]]:
    per_example = metrics.get("per_example")
    if not isinstance(per_example, list) or not per_example:
        raise SuccessorMicroOverfitError("selected checkpoint lacks production per-example metrics")
    probabilities = [float(row["teacher_successor_probability"]) for row in per_example]
    nlls = [float(row["teacher_successor_nll"]) for row in per_example]
    ranks = [int(row["teacher_successor_rank"]) for row in per_example]
    if (
        any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities)
        or any(not math.isfinite(value) or value < 0.0 for value in nlls)
        or any(rank <= 0 for rank in ranks)
        or not math.isfinite(float(metrics["teacher_successor_top1_recall"]))
        or not math.isfinite(float(metrics["teacher_successor_probability"]))
        or not math.isfinite(float(metrics["canonical_successor_nll"]))
    ):
        raise SuccessorMicroOverfitError("selected checkpoint contains invalid production metrics")
    checks = {
        "minimum_unique_state_teacher_successor_top1": (
            float(metrics["teacher_successor_top1_recall"])
            >= float(thresholds["minimum_unique_state_teacher_successor_top1"])
        ),
        "minimum_unique_state_teacher_successor_probability": (
            float(metrics["teacher_successor_probability"])
            >= float(thresholds["minimum_unique_state_teacher_successor_probability"])
        ),
        "maximum_unique_state_teacher_successor_nll": (
            float(metrics["canonical_successor_nll"])
            <= float(thresholds["maximum_unique_state_teacher_successor_nll"])
        ),
        "require_every_example_teacher_successor_top1": (
            not bool(thresholds["require_every_example_teacher_successor_top1"])
            or all(rank == 1 for rank in ranks)
        ),
        "minimum_every_example_teacher_successor_probability": (
            min(probabilities)
            >= float(thresholds["minimum_every_example_teacher_successor_probability"])
        ),
        "minimum_nonzero_gradient_optimizer_steps": (
            optimizer_steps_with_nonzero_gradient
            >= int(thresholds["minimum_nonzero_gradient_optimizer_steps"])
        ),
        "require_every_declared_component_nonzero_gradient": (
            not bool(thresholds["require_every_declared_component_nonzero_gradient"])
            or (
                not required_components_without_gradient
                and all(value > 0 for value in component_gradient_update_counts.values())
            )
        ),
    }
    floors: dict[str, float | int] = {
        "minimum_teacher_successor_probability": min(probabilities),
        "maximum_teacher_successor_nll": max(nlls),
        "maximum_teacher_successor_rank": max(ranks),
        "top1_example_count": sum(rank == 1 for rank in ranks),
        "example_count": len(ranks),
    }
    return checks, floors


def _require_production_point_consistency(
    metrics: Mapping[str, object],
    point: Mapping[str, object],
    *,
    state_label: str,
) -> None:
    """Bind the independently enumerated production law to one trajectory state."""

    per_example = metrics.get("per_example")
    if not isinstance(per_example, list) or not per_example:
        raise SuccessorMicroOverfitError(
            f"{state_label} production metrics lack per-example probabilities"
        )
    try:
        production_probabilities = [
            float(row["teacher_successor_probability"]) for row in per_example
        ]
        comparisons = (
            (
                "minimum_teacher_successor_probability",
                min(production_probabilities),
                float(point["minimum_teacher_successor_probability"]),
            ),
            (
                "mean_teacher_successor_probability",
                float(metrics["teacher_successor_probability"]),
                float(point["mean_teacher_successor_probability"]),
            ),
            (
                "mean_canonical_successor_nll",
                float(metrics["canonical_successor_nll"]),
                float(point["mean_canonical_successor_nll"]),
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SuccessorMicroOverfitError(
            f"{state_label} production/trajectory metrics are malformed"
        ) from error
    for name, production_value, trajectory_value in comparisons:
        if not (
            math.isfinite(production_value)
            and math.isfinite(trajectory_value)
            and math.isclose(
                production_value,
                trajectory_value,
                rel_tol=1e-4,
                abs_tol=1e-10 if "probability" in name else 1e-6,
            )
        ):
            raise SuccessorMicroOverfitError(
                f"{state_label} production {name} disagrees with its trajectory point: "
                f"{production_value} != {trajectory_value}"
            )


def train_uniform_successor_capacity_v2(
    model: FactorizedTraceletRateModel,
    panel: PreparedSuccessorPanel,
    *,
    maximum_updates: int,
    learning_rate: float,
    weight_decay: float,
    scope: str,
    seed: int,
    gradient_clip_norm: float,
    thresholds: Mapping[str, object],
) -> dict[str, Any]:
    """Fit one family-local panel with prospective bottleneck selection.

    The returned model is restored to the selected state.  The report retains
    hashes and metrics for both the optimization terminal state and the
    selected state, so a nonterminal selection cannot be mistaken for the last
    optimizer update.
    """

    if (
        maximum_updates <= 0
        or learning_rate <= 0.0
        or weight_decay < 0.0
        or gradient_clip_norm <= 0.0
    ):
        raise ValueError("invalid uniform T1 V2 optimization settings")
    if dict(thresholds) != EXPECTED_V1_THRESHOLDS:
        raise ValueError("uniform T1 V2 thresholds differ from frozen V1 thresholds")

    torch.manual_seed(int(seed))
    families = tuple(sorted({row.family_name for row in panel.examples}))
    if len(families) != 1:
        raise SuccessorMicroOverfitError(
            "uniform T1 V2 family-local diagnostic requires exactly one family"
        )
    configured_names = configure_micro_overfit_parameters(model, families, scope=scope)
    initial_hazard_state = _hazard_parameter_state(model)
    for name, parameter in model.named_parameters():
        if name.startswith(TOTAL_HAZARD_PARAMETER_PREFIX):
            parameter.requires_grad_(False)
            parameter.grad = None
    trainable_names = tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    if not trainable_names or any(
        name.startswith(TOTAL_HAZARD_PARAMETER_PREFIX) for name in trainable_names
    ):
        raise SuccessorMicroOverfitError("uniform T1 V2 failed to exclude total-hazard parameters")
    excluded_hazard_names = tuple(sorted(set(configured_names) - set(trainable_names)))
    if excluded_hazard_names != tuple(sorted(initial_hazard_state)):
        raise SuccessorMicroOverfitError(
            "uniform T1 V2 excluded parameter set is not exactly total_hazard_head"
        )

    required_component_prefixes = _required_component_prefixes(families)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    device = next(model.parameters()).device
    device_batch = panel.batch.to(device)
    initial_model_state_sha256 = state_dict_semantic_sha256(model.state_dict())
    initial_hazard_state_sha256 = state_dict_semantic_sha256(initial_hazard_state)
    initial_metrics = successor_panel_metrics(model, panel, include_per_example=False)

    finite_nonzero_gradient_seen = {name: False for name in trainable_names}
    gradient_update_counts = {name: 0 for name in trainable_names}
    component_gradient_update_counts = {component: 0 for component in required_component_prefixes}
    optimizer_steps_with_nonzero_gradient = 0
    trajectory: list[dict[str, object]] = []
    selected_criterion_point: dict[str, object] | None = None
    selected_update: int | None = None
    selected_state: dict[str, Tensor] | None = None
    stop_reason = "MAXIMUM_FULL_PANEL_UPDATES_REACHED"
    completed_updates = 0

    model.train()
    while True:
        optimizer.zero_grad(set_to_none=True)
        prediction = forward_teacher_successor_batch(
            model,
            device_batch,
            panel.fibers,
        )
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
            "minimum_nonzero_gradient_optimizer_steps": (
                optimizer_steps_with_nonzero_gradient
                >= int(thresholds["minimum_nonzero_gradient_optimizer_steps"])
            ),
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
            "optimizer_steps_with_nonzero_gradient_before_next_update": (
                optimizer_steps_with_nonzero_gradient
            ),
            "component_gradient_update_counts_before_next_update": dict(
                component_gradient_update_counts
            ),
            "gradient_gate_checks_before_next_update": gradient_checks,
            "required_components_without_gradient_before_next_update": list(missing_components),
            "all_early_stop_checks_pass": (
                all(implication_checks.values()) and all(gradient_checks.values())
            ),
            "optimizer_update_executed_after_evaluation": False,
            "loss_for_executed_update": None,
            "gradient_norm_for_executed_update": None,
        }
        if checkpoint_is_better(point, selected_criterion_point):
            selected_criterion_point = copy.deepcopy(point)
            selected_update = completed_updates
            selected_state = _clone_state_dict(model)

        should_stop_early = bool(point["all_early_stop_checks_pass"])
        at_update_limit = completed_updates == maximum_updates
        if should_stop_early or at_update_limit:
            if should_stop_early and not at_update_limit:
                stop_reason = "ALL_FROZEN_EARLY_STOP_CHECKS_PROVEN"
            trajectory.append(point)
            break

        loss = factorized_successor_identity_loss(prediction, device_batch)
        if not bool(torch.isfinite(loss)):
            raise SuccessorMicroOverfitError(
                f"non-finite successor identity loss before update {completed_updates + 1}"
            )
        loss.backward()
        squared_gradient_norm = 0.0
        nonzero_gradient_names: set[str] = set()
        for name, parameter in model.named_parameters():
            if name.startswith(TOTAL_HAZARD_PARAMETER_PREFIX):
                if parameter.grad is not None:
                    raise SuccessorMicroOverfitError(
                        f"frozen total-hazard parameter received a gradient: {name}"
                    )
                continue
            if not parameter.requires_grad or parameter.grad is None:
                continue
            if not bool(torch.isfinite(parameter.grad).all()):
                raise SuccessorMicroOverfitError(
                    f"non-finite gradient for {name!r} before update {completed_updates + 1}"
                )
            gradient_norm = float(parameter.grad.norm())
            squared_gradient_norm += gradient_norm * gradient_norm
            if gradient_norm > 0.0:
                finite_nonzero_gradient_seen[name] = True
                gradient_update_counts[name] += 1
                nonzero_gradient_names.add(name)
        if nonzero_gradient_names:
            optimizer_steps_with_nonzero_gradient += 1
        for component, prefixes in required_component_prefixes.items():
            if any(name.startswith(prefixes) for name in nonzero_gradient_names):
                component_gradient_update_counts[component] += 1
        torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            max_norm=float(gradient_clip_norm),
        )
        optimizer.step()
        completed_updates += 1
        _require_hazard_bit_identity(
            model,
            initial_hazard_state,
            state_label=f"post_update_{completed_updates}",
        )
        point["optimizer_update_executed_after_evaluation"] = True
        point["loss_for_executed_update"] = float(loss.detach())
        point["gradient_norm_for_executed_update"] = squared_gradient_norm**0.5
        trajectory.append(point)

    if selected_criterion_point is None or selected_update is None or selected_state is None:
        raise AssertionError("uniform T1 V2 selected no checkpoint")
    selected_point = copy.deepcopy(trajectory[selected_update])
    terminal_state = _clone_state_dict(model)
    terminal_model_state_sha256 = state_dict_semantic_sha256(terminal_state)
    _require_hazard_bit_identity(model, initial_hazard_state, state_label="terminal")
    terminal_hazard_state_sha256 = state_dict_semantic_sha256(_hazard_parameter_state(model))
    terminal_metrics = successor_panel_metrics(model, panel, include_per_example=True)

    _require_production_point_consistency(
        terminal_metrics,
        trajectory[-1],
        state_label="terminal",
    )

    model.load_state_dict(selected_state, strict=True)
    _require_hazard_bit_identity(model, initial_hazard_state, state_label="selected_restore")
    selected_model_state_sha256 = state_dict_semantic_sha256(model.state_dict())
    selected_hazard_state_sha256 = state_dict_semantic_sha256(_hazard_parameter_state(model))
    selected_metrics = successor_panel_metrics(model, panel, include_per_example=True)
    _require_production_point_consistency(
        selected_metrics,
        selected_point,
        state_label="selected",
    )
    required_components_without_gradient = tuple(
        sorted(
            component for component, count in component_gradient_update_counts.items() if count == 0
        )
    )
    threshold_checks, floor_metrics = _selected_metric_checks(
        selected_metrics,
        optimizer_steps_with_nonzero_gradient=optimizer_steps_with_nonzero_gradient,
        component_gradient_update_counts=component_gradient_update_counts,
        required_components_without_gradient=required_components_without_gradient,
        thresholds=thresholds,
    )
    trajectory_sha256 = stable_sha256(trajectory)
    terminal_point = trajectory[-1]
    hazard_bit_identity = (
        initial_hazard_state_sha256 == terminal_hazard_state_sha256 == selected_hazard_state_sha256
    )
    if not hazard_bit_identity:
        raise SuccessorMicroOverfitError(
            "total-hazard parameter hash changed despite explicit exclusion"
        )
    return {
        "scope": scope,
        "seed": int(seed),
        "maximum_full_panel_updates": maximum_updates,
        "completed_full_panel_updates": completed_updates,
        "early_stopped": completed_updates < maximum_updates,
        "stop_reason": stop_reason,
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "gradient_clip_norm": float(gradient_clip_norm),
        "optimizer": "AdamW",
        "schedule": "constant",
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
        "selection_rule": SELECTION_RULE,
        "early_stop_rule": EARLY_STOP_RULE,
        "families": list(families),
        "example_count": len(panel.examples),
        "selected_update": selected_update,
        "terminal_update": completed_updates,
        "trajectory_state_count": len(trajectory),
        "trajectory_sha256": trajectory_sha256,
        "trajectory": trajectory,
        "initial": initial_metrics,
        "terminal": terminal_metrics,
        "selected": selected_metrics,
        "terminal_vs_selected": {
            "same_update": selected_update == completed_updates,
            "selected_minus_terminal_minimum_teacher_successor_probability": (
                float(selected_point["minimum_teacher_successor_probability"])
                - float(terminal_point["minimum_teacher_successor_probability"])
            ),
            "selected_minus_terminal_mean_canonical_successor_nll": (
                float(selected_point["mean_canonical_successor_nll"])
                - float(terminal_point["mean_canonical_successor_nll"])
            ),
            "selected_model_state_sha256": selected_model_state_sha256,
            "terminal_model_state_sha256": terminal_model_state_sha256,
        },
        "selected_differentiable_point": selected_point,
        "terminal_differentiable_point": terminal_point,
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in trainable_parameters)
        ),
        "trainable_tensor_count": len(trainable_names),
        "trainable_parameter_names_sha256": stable_sha256(list(trainable_names)),
        "never_received_nonzero_gradient": sorted(
            name for name, seen in finite_nonzero_gradient_seen.items() if not seen
        ),
        "gradient_update_counts": gradient_update_counts,
        "component_gradient_update_counts": component_gradient_update_counts,
        "optimizer_steps_with_nonzero_gradient": optimizer_steps_with_nonzero_gradient,
        "required_components_without_gradient": list(required_components_without_gradient),
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
        "initial_model_state_sha256": initial_model_state_sha256,
        "terminal_model_state_sha256": terminal_model_state_sha256,
        "selected_model_state_sha256": selected_model_state_sha256,
        "returned_model_state_sha256": selected_model_state_sha256,
        "thresholds": dict(thresholds),
        "threshold_checks": threshold_checks,
        "per_example_floor_metrics": floor_metrics,
        "all_threshold_checks_pass": all(threshold_checks.values()),
    }


def validate_v2_training_report(
    report: object,
    *,
    family: str,
    maximum_updates: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    seed: int,
) -> Mapping[str, object]:
    """Recompute trajectory, selection, hazard, and threshold invariants."""

    if not isinstance(report, dict):
        raise SuccessorMicroOverfitError("uniform T1 V2 report must be one object")
    if (
        report.get("scope") != "all"
        or report.get("maximum_full_panel_updates") != maximum_updates
        or not isinstance(report.get("completed_full_panel_updates"), int)
        or not 1 <= int(report["completed_full_panel_updates"]) <= maximum_updates
        or report.get("families") != [family]
        or report.get("example_count") != 64
        or report.get("selection_rule") != SELECTION_RULE
        or report.get("early_stop_rule") != EARLY_STOP_RULE
        or report.get("optimizer") != "AdamW"
        or report.get("schedule") != "constant"
        or report.get("learning_rate") != learning_rate
        or report.get("weight_decay") != weight_decay
        or report.get("gradient_clip_norm") != gradient_clip_norm
        or report.get("seed") != seed
        or not isinstance(report.get("execution_environment"), Mapping)
    ):
        raise SuccessorMicroOverfitError("uniform T1 V2 report law is invalid")
    trajectory = report.get("trajectory")
    environment = report["execution_environment"]
    assert isinstance(environment, Mapping)
    if (
        not isinstance(environment.get("torch_version"), str)
        or environment.get("device_type") not in {"cpu", "cuda", "mps"}
        or not isinstance(environment.get("model_parameter_dtype"), str)
        or not isinstance(environment.get("deterministic_algorithms_enabled"), bool)
    ):
        raise SuccessorMicroOverfitError("uniform T1 V2 execution environment is malformed")
    completed = int(report["completed_full_panel_updates"])
    if (
        not isinstance(trajectory, list)
        or len(trajectory) != completed + 1
        or report.get("trajectory_state_count") != len(trajectory)
        or report.get("trajectory_sha256") != stable_sha256(trajectory)
        or [point.get("update") for point in trajectory] != list(range(completed + 1))
    ):
        raise SuccessorMicroOverfitError("uniform T1 V2 trajectory is invalid")
    for index, point in enumerate(trajectory):
        if not isinstance(point, Mapping):
            raise SuccessorMicroOverfitError("uniform T1 V2 trajectory point is invalid")
        try:
            minimum_probability = float(point["minimum_teacher_successor_probability"])
            minimum_log_probability = float(point["minimum_teacher_successor_log_probability"])
            mean_probability = float(point["mean_teacher_successor_probability"])
            mean_nll = float(point["mean_canonical_successor_nll"])
            maximum_nll = float(point["maximum_teacher_successor_nll"])
        except (KeyError, TypeError, ValueError) as error:
            raise SuccessorMicroOverfitError(
                "uniform T1 V2 trajectory metric is malformed"
            ) from error
        if (
            not all(
                math.isfinite(value)
                for value in (
                    minimum_probability,
                    minimum_log_probability,
                    mean_probability,
                    mean_nll,
                    maximum_nll,
                )
            )
            or not 0.0 <= minimum_probability <= mean_probability <= 1.0
            or minimum_log_probability > 0.0
            or mean_nll < 0.0
            or maximum_nll < mean_nll
            or not math.isclose(
                minimum_probability,
                math.exp(minimum_log_probability),
                rel_tol=1e-6,
                abs_tol=1e-12,
            )
        ):
            raise SuccessorMicroOverfitError("uniform T1 V2 trajectory metric is invalid")
        every_top1_proven = minimum_probability > 0.5
        expected_implications = {
            "minimum_unique_state_teacher_successor_top1": (
                every_top1_proven
                and 1.0
                >= float(EXPECTED_V1_THRESHOLDS["minimum_unique_state_teacher_successor_top1"])
            ),
            "minimum_unique_state_teacher_successor_probability": (
                mean_probability
                >= float(
                    EXPECTED_V1_THRESHOLDS["minimum_unique_state_teacher_successor_probability"]
                )
            ),
            "maximum_unique_state_teacher_successor_nll": (
                mean_nll
                <= float(EXPECTED_V1_THRESHOLDS["maximum_unique_state_teacher_successor_nll"])
            ),
            "require_every_example_teacher_successor_top1": (
                not bool(EXPECTED_V1_THRESHOLDS["require_every_example_teacher_successor_top1"])
                or every_top1_proven
            ),
            "minimum_every_example_teacher_successor_probability": (
                minimum_probability
                >= float(
                    EXPECTED_V1_THRESHOLDS["minimum_every_example_teacher_successor_probability"]
                )
            ),
        }
        component_counts = point.get("component_gradient_update_counts_before_next_update")
        if (
            not isinstance(component_counts, Mapping)
            or set(component_counts) != {"family_head", family}
            or any(
                not isinstance(value, int) or not 0 <= value <= index
                for value in component_counts.values()
            )
        ):
            raise SuccessorMicroOverfitError(
                "uniform T1 V2 trajectory component-gradient evidence is invalid"
            )
        nonzero_steps = point.get("optimizer_steps_with_nonzero_gradient_before_next_update")
        missing_components = sorted(
            component for component, count in component_counts.items() if count == 0
        )
        expected_gradient_checks = {
            "minimum_nonzero_gradient_optimizer_steps": (
                isinstance(nonzero_steps, int)
                and nonzero_steps
                >= int(EXPECTED_V1_THRESHOLDS["minimum_nonzero_gradient_optimizer_steps"])
            ),
            "require_every_declared_component_nonzero_gradient": (
                not bool(
                    EXPECTED_V1_THRESHOLDS["require_every_declared_component_nonzero_gradient"]
                )
                or not missing_components
            ),
        }
        if (
            not isinstance(nonzero_steps, int)
            or not 0 <= nonzero_steps <= index
            or point.get("probability_nll_top1_implication_checks") != expected_implications
            or point.get("required_components_without_gradient_before_next_update")
            != missing_components
            or point.get("gradient_gate_checks_before_next_update") != expected_gradient_checks
            or point.get("all_early_stop_checks_pass")
            is not (all(expected_implications.values()) and all(expected_gradient_checks.values()))
        ):
            raise SuccessorMicroOverfitError("uniform T1 V2 trajectory proof evidence is invalid")
        executed = point.get("optimizer_update_executed_after_evaluation")
        if executed is not (index < completed):
            raise SuccessorMicroOverfitError(
                "uniform T1 V2 trajectory update execution is inconsistent"
            )
        if index < completed and point.get("all_early_stop_checks_pass") is True:
            raise SuccessorMicroOverfitError(
                "uniform T1 V2 continued after a proven early-stop state"
            )
    recomputed_selected: Mapping[str, object] | None = None
    for point in trajectory:
        if checkpoint_is_better(point, recomputed_selected):
            recomputed_selected = point
    if (
        recomputed_selected is None
        or report.get("selected_update") != recomputed_selected["update"]
        or report.get("selected_differentiable_point") != recomputed_selected
        or report.get("terminal_update") != completed
        or report.get("terminal_differentiable_point") != trajectory[-1]
    ):
        raise SuccessorMicroOverfitError(
            "uniform T1 V2 selected checkpoint violates the frozen rule"
        )
    terminal_metrics = report.get("terminal")
    selected_metrics = report.get("selected")
    if not isinstance(terminal_metrics, Mapping) or not isinstance(selected_metrics, Mapping):
        raise SuccessorMicroOverfitError("uniform T1 V2 production metrics are absent")
    _require_production_point_consistency(
        terminal_metrics,
        trajectory[-1],
        state_label="terminal",
    )
    _require_production_point_consistency(
        selected_metrics,
        recomputed_selected,
        state_label="selected",
    )
    expected_terminal_vs_selected = {
        "same_update": int(recomputed_selected["update"]) == completed,
        "selected_minus_terminal_minimum_teacher_successor_probability": (
            float(recomputed_selected["minimum_teacher_successor_probability"])
            - float(trajectory[-1]["minimum_teacher_successor_probability"])
        ),
        "selected_minus_terminal_mean_canonical_successor_nll": (
            float(recomputed_selected["mean_canonical_successor_nll"])
            - float(trajectory[-1]["mean_canonical_successor_nll"])
        ),
        "selected_model_state_sha256": report.get("selected_model_state_sha256"),
        "terminal_model_state_sha256": report.get("terminal_model_state_sha256"),
    }
    if report.get("terminal_vs_selected") != expected_terminal_vs_selected:
        raise SuccessorMicroOverfitError(
            "uniform T1 V2 terminal-versus-selected evidence is invalid"
        )
    stopped_early = completed < maximum_updates
    if (
        report.get("early_stopped") is not stopped_early
        or (
            stopped_early
            and (
                report.get("stop_reason") != "ALL_FROZEN_EARLY_STOP_CHECKS_PROVEN"
                or trajectory[-1].get("all_early_stop_checks_pass") is not True
            )
        )
        or (not stopped_early and report.get("stop_reason") != "MAXIMUM_FULL_PANEL_UPDATES_REACHED")
    ):
        raise SuccessorMicroOverfitError("uniform T1 V2 early-stop evidence is invalid")
    hazard = report.get("hazard_freeze")
    if (
        not isinstance(hazard, Mapping)
        or hazard.get("parameter_prefix") != TOTAL_HAZARD_PARAMETER_PREFIX
        or hazard.get("requires_grad") is not False
        or hazard.get("optimizer_included") is not False
        or hazard.get("gradient_observed") is not False
        or hazard.get("bit_identity_required") is not True
        or hazard.get("bit_identity_pass") is not True
        or not isinstance(hazard.get("excluded_parameter_names"), list)
        or not hazard["excluded_parameter_names"]
        or any(
            not str(name).startswith(TOTAL_HAZARD_PARAMETER_PREFIX)
            for name in hazard["excluded_parameter_names"]
        )
        or hazard.get("excluded_parameter_count") != len(hazard["excluded_parameter_names"])
        or len(
            {
                hazard.get("initial_parameter_state_sha256"),
                hazard.get("terminal_parameter_state_sha256"),
                hazard.get("selected_parameter_state_sha256"),
            }
        )
        != 1
    ):
        raise SuccessorMicroOverfitError("uniform T1 V2 hazard freeze evidence is invalid")
    for field in (
        "trajectory_sha256",
        "trainable_parameter_names_sha256",
        "initial_model_state_sha256",
        "terminal_model_state_sha256",
        "selected_model_state_sha256",
        "returned_model_state_sha256",
    ):
        if not is_sha256(report.get(field)):
            raise SuccessorMicroOverfitError(f"uniform T1 V2 report {field} is invalid")
    if (
        report["returned_model_state_sha256"] != report["selected_model_state_sha256"]
        or report.get("thresholds") != EXPECTED_V1_THRESHOLDS
    ):
        raise SuccessorMicroOverfitError(
            "uniform T1 V2 selected-state or threshold identity is invalid"
        )
    component_updates = report.get("component_gradient_update_counts")
    missing = report.get("required_components_without_gradient")
    if (
        not isinstance(component_updates, Mapping)
        or not isinstance(missing, list)
        or not isinstance(selected_metrics, Mapping)
        or dict(component_updates)
        != trajectory[-1]["component_gradient_update_counts_before_next_update"]
        or report.get("optimizer_steps_with_nonzero_gradient")
        != trajectory[-1]["optimizer_steps_with_nonzero_gradient_before_next_update"]
        or missing != trajectory[-1]["required_components_without_gradient_before_next_update"]
    ):
        raise SuccessorMicroOverfitError("uniform T1 V2 gradient or selected metrics are malformed")
    checks, floors = _selected_metric_checks(
        selected_metrics,
        optimizer_steps_with_nonzero_gradient=int(report["optimizer_steps_with_nonzero_gradient"]),
        component_gradient_update_counts={
            str(name): int(value) for name, value in component_updates.items()
        },
        required_components_without_gradient=tuple(str(name) for name in missing),
        thresholds=EXPECTED_V1_THRESHOLDS,
    )
    if (
        report.get("threshold_checks") != checks
        or report.get("per_example_floor_metrics") != floors
        or report.get("all_threshold_checks_pass") is not all(checks.values())
    ):
        raise SuccessorMicroOverfitError("uniform T1 V2 selected threshold checks are invalid")
    return report


__all__ = [
    "EARLY_STOP_RULE",
    "EXPECTED_V1_THRESHOLDS",
    "EXPECTED_UNIFORM_V2_CONTRACT_SHA256",
    "SELECTION_RULE",
    "TOTAL_HAZARD_PARAMETER_PREFIX",
    "UNIFORM_V2_CONTRACT_SCHEMA",
    "UNIFORM_V2_CONTRACT_STATUS",
    "UNIFORM_V2_CONTRACT_VERSION",
    "UNIFORM_V2_RESULT_SCHEMA",
    "UNIFORM_V2_RESULT_STATUS",
    "UNIFORM_V2_RESULT_VERSION",
    "canonical_json_bytes",
    "checkpoint_is_better",
    "is_sha256",
    "load_uniform_capacity_v2_contract",
    "probability_implication_checks",
    "serialized_source_tree_sha256",
    "stable_sha256",
    "train_uniform_successor_capacity_v2",
    "validate_v2_training_report",
]
