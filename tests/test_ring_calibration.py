from __future__ import annotations

import pytest
import torch

from compose_v4.eval.ring_calibration import (
    masked_category_mass,
    ring_template_cycle_sizes,
    undesirable_small_ring_mask,
    uniform_category_mass,
)
from compose_v4.rewrite.typed_ring_catalog import RingSystemTemplate


def _cycle_template(size: int) -> RingSystemTemplate:
    atoms = tuple((6, 0, 2) for _ in range(size))
    path = tuple((index, index + 1, 1) for index in range(size - 1))
    cycle = (*path, (0, size - 1, 1))
    return RingSystemTemplate(
        source_atoms=atoms,
        target_atoms=atoms,
        source_bonds=path,
        target_bonds=cycle,
        grow_bond_reorders=(),
        grow_atom_payloads=(),
        inserted_bonds=((0, size - 1, 1),),
        deleted_bonds=(),
        delete_atom_payloads=(),
        delete_bond_reorders=(),
        source_external_bonds=tuple(() for _ in range(size)),
        target_aromatic_edges=(),
        topology_class="single_ring",
    )


def test_ring_template_cycle_sizes_and_small_mask() -> None:
    templates = (_cycle_template(3), _cycle_template(5), _cycle_template(6))

    assert ring_template_cycle_sizes(templates[0]) == (3,)
    assert undesirable_small_ring_mask(templates).tolist() == [True, False, False]


def test_masked_category_mass_distinguishes_support_and_logits() -> None:
    logits = torch.tensor((0.0, 1.0, 9.0))
    support = torch.tensor((True, True, False))
    category = torch.tensor((True, False, True))

    assert uniform_category_mass(support, category) == pytest.approx(0.5)
    assert masked_category_mass(logits, support, category) == pytest.approx(
        1.0 / (1.0 + torch.exp(torch.tensor(1.0)).item())
    )


def test_category_mass_rejects_empty_support() -> None:
    support = torch.zeros(2, dtype=torch.bool)
    category = torch.tensor((True, False))

    with pytest.raises(ValueError, match="non-empty support"):
        uniform_category_mass(support, category)
    with pytest.raises(ValueError, match="non-empty support"):
        masked_category_mass(torch.zeros(2), support, category)
