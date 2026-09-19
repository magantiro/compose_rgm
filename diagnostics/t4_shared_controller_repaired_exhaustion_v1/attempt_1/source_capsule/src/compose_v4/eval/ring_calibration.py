"""Diagnostics for state-conditional ring-template calibration.

These helpers deliberately operate on template logits and exact executor support
without changing the production sampler.  They make it possible to distinguish
three otherwise conflated causes of undesirable ring generation:

* the legal action set exposes too much undesirable topology;
* the empirical catalog prior overweights it after conditioning on legality; or
* the learned residual logits amplify it beyond both baselines.
"""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import torch
from torch import Tensor

from compose_v4.rewrite.typed_ring_catalog import RingSystemTemplate


def ring_template_cycle_sizes(template: RingSystemTemplate) -> tuple[int, ...]:
    """Return minimum-cycle-basis sizes for a ring-system target pattern."""

    graph = nx.Graph()
    graph.add_nodes_from(range(int(template.span)))
    graph.add_edges_from(
        (int(left), int(right)) for left, right, _ in template.target_bonds
    )
    return tuple(sorted(len(cycle) for cycle in nx.minimum_cycle_basis(graph)))


def undesirable_small_ring_mask(
    templates: Sequence[RingSystemTemplate],
    *,
    maximum_size: int = 4,
) -> Tensor:
    """Identify templates whose target pattern contains a 3/4-membered ring."""

    if maximum_size < 3:
        raise ValueError("maximum small-ring size must be at least three")
    return torch.tensor(
        tuple(
            bool(cycle_sizes) and min(cycle_sizes) <= maximum_size
            for cycle_sizes in map(ring_template_cycle_sizes, templates)
        ),
        dtype=torch.bool,
    )


def masked_category_mass(logits: Tensor, support: Tensor, category: Tensor) -> float:
    """Return conditional softmax mass assigned to one template category."""

    if logits.ndim != 1:
        raise ValueError("template logits must be one-dimensional")
    if support.shape != logits.shape or category.shape != logits.shape:
        raise ValueError("support/category masks must align with template logits")
    support = support.to(device=logits.device, dtype=torch.bool)
    category = category.to(device=logits.device, dtype=torch.bool)
    if not bool(support.any()):
        raise ValueError("conditional category mass requires non-empty support")
    probabilities = torch.softmax(
        logits.masked_fill(~support, float("-inf")),
        dim=0,
    )
    return float(probabilities[category & support].sum().detach().cpu())


def uniform_category_mass(support: Tensor, category: Tensor) -> float:
    """Return category prevalence among legal templates, without any logits."""

    if support.shape != category.shape:
        raise ValueError("support/category masks must align")
    support = support.to(dtype=torch.bool)
    category = category.to(dtype=torch.bool)
    legal = int(support.sum())
    if legal == 0:
        raise ValueError("uniform category mass requires non-empty support")
    return float((support & category).sum()) / float(legal)


__all__ = [
    "masked_category_mass",
    "ring_template_cycle_sizes",
    "undesirable_small_ring_mask",
    "uniform_category_mass",
]
