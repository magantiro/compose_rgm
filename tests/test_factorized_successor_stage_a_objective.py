"""Focused contract tests for the Stage-A semantic-cell objective."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from compose_v4.experiments import factorized_successor_objective as objective_module
from compose_v4.experiments.factorized_successor_data import (
    FactorizedSuccessorBatch,
)
from compose_v4.experiments.factorized_successor_objective import (
    CanonicalSuccessorTrainingObjective,
)
from compose_v4.experiments.factorized_successor_training import (
    SuccessorTrainingError,
)


def _batch(
    *,
    teacher_rates: tuple[float, ...],
    importance_weights: tuple[float, ...],
    semantic_cell_ids: tuple[str | None, ...],
) -> FactorizedSuccessorBatch:
    row_count = len(teacher_rates)
    assert len(importance_weights) == row_count
    assert len(semantic_cell_ids) == row_count
    mark_batch = SimpleNamespace(
        batch_size=row_count,
        teacher_rates=torch.tensor(teacher_rates, dtype=torch.float32),
        importance_weights=torch.tensor(importance_weights, dtype=torch.float32),
    )
    supports = tuple(object() for _ in range(row_count))
    fibers = tuple(
        None if rate == 0.0 else SimpleNamespace(state_support=support)
        for rate, support in zip(teacher_rates, supports, strict=True)
    )
    addresses = tuple(SimpleNamespace(is_terminal=rate == 0.0) for rate in teacher_rates)
    return FactorizedSuccessorBatch(
        mark_batch=mark_batch,
        fibers=fibers,
        state_supports=supports,
        cache_addresses=addresses,
        semantic_cell_ids=semantic_cell_ids,
    )


def _objective_loss(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: str,
    log_probabilities: torch.Tensor,
    teacher_rates: tuple[float, ...],
    importance_weights: tuple[float, ...],
    semantic_cell_ids: tuple[str | None, ...],
    total_hazard: torch.Tensor | None = None,
) -> torch.Tensor:
    batch = _batch(
        teacher_rates=teacher_rates,
        importance_weights=importance_weights,
        semantic_cell_ids=semantic_cell_ids,
    )
    prediction = SimpleNamespace(
        selected_productive_successor_log_probability=log_probabilities,
        total_hazard=(torch.ones_like(log_probabilities) if total_hazard is None else total_hazard),
    )
    monkeypatch.setattr(
        objective_module,
        "assert_teachers_in_exact_candidates",
        lambda _mark_batch: None,
    )
    monkeypatch.setattr(
        objective_module,
        "forward_teacher_successor_batch",
        lambda *_args, **_kwargs: prediction,
    )
    objective = CanonicalSuccessorTrainingObjective(
        mode=mode,
        hazard_weight=0.0,
    )
    return objective.loss(SimpleNamespace(), batch)


def test_stage_a_objective_is_invariant_to_between_cell_imbalance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    balanced = _objective_loss(
        monkeypatch,
        mode="balanced_semantic_cell_productive_identity",
        log_probabilities=torch.tensor((-1.0, -3.0)),
        teacher_rates=(1.0, 1.0),
        importance_weights=(1.0, 1.0),
        semantic_cell_ids=("cell-a", "cell-b"),
    )
    imbalanced = _objective_loss(
        monkeypatch,
        mode="balanced_semantic_cell_productive_identity",
        log_probabilities=torch.tensor((-1.0, -1.0, -1.0, -1.0, -3.0)),
        teacher_rates=(1.0, 1.0, 1.0, 1.0, 1.0),
        importance_weights=(1.0, 1.0, 1.0, 1.0, 1.0),
        semantic_cell_ids=("cell-a", "cell-a", "cell-a", "cell-a", "cell-b"),
    )

    assert float(balanced) == pytest.approx(2.0)
    assert float(imbalanced) == pytest.approx(float(balanced))


def test_stage_a_objective_averages_within_cell_and_ignores_legacy_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loss = _objective_loss(
        monkeypatch,
        mode="balanced_semantic_cell_productive_identity",
        log_probabilities=torch.tensor((-1.0, -3.0, -10.0)),
        teacher_rates=(1.0, 1.0, 1.0),
        importance_weights=(1000.0, 0.001, 77.0),
        semantic_cell_ids=("cell-a", "cell-a", "cell-b"),
    )

    assert float(loss) == pytest.approx(6.0)


def test_stage_a_objective_rejects_missing_nonterminal_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SuccessorTrainingError, match="frozen semantic cell"):
        _objective_loss(
            monkeypatch,
            mode="balanced_semantic_cell_productive_identity",
            log_probabilities=torch.tensor((-1.0, -9.0)),
            teacher_rates=(1.0, 0.0),
            importance_weights=(1.0, 1.0),
            semantic_cell_ids=(None, None),
        )


def test_stage_a_objective_has_no_hazard_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="requires hazard_weight=0"):
        CanonicalSuccessorTrainingObjective(
            mode="balanced_semantic_cell_productive_identity",
            hazard_weight=0.1,
        )

    identity_log_probabilities = torch.tensor((-1.0, -3.0), requires_grad=True)
    hazard_logit = torch.tensor(0.5, requires_grad=True)
    total_hazard = torch.nn.functional.softplus(hazard_logit).expand(2)
    loss = _objective_loss(
        monkeypatch,
        mode="balanced_semantic_cell_productive_identity",
        log_probabilities=identity_log_probabilities,
        teacher_rates=(1.0, 1.0),
        importance_weights=(100.0, 0.01),
        semantic_cell_ids=("cell-a", "cell-b"),
        total_hazard=total_hazard,
    )

    loss.backward()

    assert identity_log_probabilities.grad is not None
    assert hazard_logit.grad is None


def test_stage_a_objective_has_precise_name_and_legacy_mode_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage_a = CanonicalSuccessorTrainingObjective(
        mode="balanced_semantic_cell_productive_identity",
        hazard_weight=0.0,
    )
    assert stage_a.name == ("canonical_successor_balanced_semantic_cell_productive_identity_v1")

    legacy = _objective_loss(
        monkeypatch,
        mode="productive_identity",
        log_probabilities=torch.tensor((-1.0, -3.0)),
        teacher_rates=(1.0, 1.0),
        importance_weights=(100.0, 1.0),
        semantic_cell_ids=(None, None),
    )
    assert float(legacy) == pytest.approx(103.0 / 101.0)
