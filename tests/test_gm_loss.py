from __future__ import annotations

import pytest
import torch

from compose_v4.gm.loss import multi_successor_rate_bregman_loss, rate_bregman_loss


def test_single_successor_optimum_matches_teacher_rate() -> None:
    teacher = torch.tensor(3.25)
    optimum = teacher.detach().clone().requires_grad_(True)
    (zero_gradient,) = torch.autograd.grad(
        rate_bregman_loss(optimum, optimum, teacher),
        optimum,
    )
    below = torch.tensor(1.0, requires_grad=True)
    above = torch.tensor(5.0, requires_grad=True)
    (below_gradient,) = torch.autograd.grad(
        rate_bregman_loss(below, below, teacher),
        below,
    )
    (above_gradient,) = torch.autograd.grad(
        rate_bregman_loss(above, above, teacher),
        above,
    )
    assert zero_gradient.item() == pytest.approx(0.0, abs=1e-7)
    assert below_gradient.item() < 0.0
    assert above_gradient.item() > 0.0


def test_non_teacher_hazard_is_penalized() -> None:
    teacher_rate = torch.tensor([2.0])
    teacher_prediction = torch.tensor([2.0])
    clean = rate_bregman_loss(teacher_prediction, teacher_prediction, teacher_rate)
    extra = rate_bregman_loss(
        teacher_prediction + 1.5,
        teacher_prediction,
        teacher_rate,
    )
    assert (extra - clean).item() == pytest.approx(1.5)


def test_successor_rate_must_be_part_of_total_hazard() -> None:
    with pytest.raises(ValueError):
        rate_bregman_loss(torch.tensor(1.0), torch.tensor(2.0), torch.tensor(1.0))


def test_multi_successor_optimum_matches_teacher_vector() -> None:
    teacher = torch.tensor([1.5, 2.5])
    predicted = teacher.detach().clone().requires_grad_(True)
    loss = multi_successor_rate_bregman_loss(
        predicted.sum(),
        predicted,
        teacher,
    )
    (gradient,) = torch.autograd.grad(loss, predicted)
    assert torch.allclose(gradient, torch.zeros_like(gradient), atol=1e-7)
