"""The segmented reduction replaces the objective's aggregation step.

A segmented logsumexp reassociates the sum, so it is not automatically bitwise
equal to stacking and reducing. These tests pin that it agrees with
``torch.logsumexp`` to a stated tolerance AND that the gradients agree, because
a reduction that produces the right value with a wrong gradient trains
silently wrong.
"""

from __future__ import annotations

import pytest
import torch

from compose_v4.experiments.editing_v2_vectorized_successor_scoring import (
    segment_logsumexp,
)


def _reference(values: torch.Tensor, segments_ids: torch.Tensor, segments: int):
    out = []
    for segment in range(segments):
        selected = values[segments_ids == segment]
        out.append(
            torch.logsumexp(selected, dim=0)
            if selected.numel()
            else torch.tensor(float("-inf"), dtype=values.dtype)
        )
    return torch.stack(out)


def test_matches_torch_logsumexp_per_segment() -> None:
    torch.manual_seed(0)
    values = torch.randn(37, dtype=torch.float64)
    ids = torch.tensor([i % 5 for i in range(37)])
    assert torch.allclose(
        segment_logsumexp(values, ids, segments=5), _reference(values, ids, 5), atol=1e-12
    )


def test_gradients_match_the_stacked_reduction() -> None:
    """Right value, wrong gradient would train silently wrong."""

    torch.manual_seed(1)
    base = torch.randn(24, dtype=torch.float64)
    ids = torch.tensor([i % 4 for i in range(24)])

    a = base.clone().requires_grad_(True)
    segment_logsumexp(a, ids, segments=4).sum().backward()
    b = base.clone().requires_grad_(True)
    _reference(b, ids, 4).sum().backward()
    assert torch.allclose(a.grad, b.grad, atol=1e-12)


def test_a_segment_with_one_member_is_that_member() -> None:
    values = torch.tensor([2.5, -1.0], dtype=torch.float64)
    ids = torch.tensor([0, 1])
    assert torch.allclose(
        segment_logsumexp(values, ids, segments=2), values, atol=1e-12
    )


def test_empty_segments_are_negative_infinity_not_nan() -> None:
    """log(0) plus -inf is nan; an empty row must stay -inf.

    Terminal rows contribute no alias, so empty segments are a normal state,
    not an error -- and a nan here would poison the whole batch's gradient.
    """

    values = torch.tensor([1.0, 2.0], dtype=torch.float64)
    ids = torch.tensor([0, 0])
    out = segment_logsumexp(values, ids, segments=3)
    assert torch.isfinite(out[0])
    assert out[1] == float("-inf") and out[2] == float("-inf")
    assert not bool(torch.isnan(out).any())


def test_no_values_at_all_returns_all_negative_infinity() -> None:
    out = segment_logsumexp(
        torch.zeros(0), torch.zeros(0, dtype=torch.long), segments=4
    )
    assert out.shape == (4,) and bool((out == float("-inf")).all())


def test_large_magnitudes_do_not_overflow() -> None:
    """The shift-by-max is what makes this usable on real logits."""

    values = torch.tensor([800.0, 801.0, -800.0], dtype=torch.float64)
    ids = torch.tensor([0, 0, 1])
    out = segment_logsumexp(values, ids, segments=2)
    assert torch.isfinite(out).all()
    assert out[0] == pytest.approx(float(torch.logsumexp(values[:2], dim=0)), abs=1e-9)


def test_unsorted_segment_ids_are_handled() -> None:
    """Aliases arrive grouped by table, so ids are NOT sorted by row."""

    torch.manual_seed(2)
    values = torch.randn(16, dtype=torch.float64)
    ids = torch.tensor([3, 0, 2, 1, 3, 1, 0, 2, 2, 3, 0, 1, 1, 2, 3, 0])
    assert torch.allclose(
        segment_logsumexp(values, ids, segments=4), _reference(values, ids, 4), atol=1e-12
    )
