"""The H40 fixed-size intervention masks families before normalizing the draw."""

import numpy as np
import pytest
import torch

from compose_v4.experiments import hphi_lazy_family_scores
from compose_v4.experiments.hphi_lazy_sampler import (
    fixed_size_allowed_families,
    sample_one_transition,
)


class _Model:
    def _encode_batch(self, batch):
        return None, None, None

    def _family_base_logits(self, batch, glob):
        return torch.tensor([[100.0, 0.0]])


def _helpers(*, restate_legal=True):
    def family_mask(model, table, state, batch, pair, glob, node):
        return torch.tensor([table != "atom_restate" or restate_legal])

    return {
        "build_batch": lambda state, time: object(),
        "family_names": ("atom_insert", "atom_restate"),
        "family_mask": family_mask,
        "legality": {},
    }


def test_fixed_size_mask_precedes_family_draw(monkeypatch):
    monkeypatch.setattr(
        hphi_lazy_family_scores,
        "LAZY_SCORERS",
        {
            "grow_connected": lambda *args: torch.tensor([[0.0]]),
            "atom_restate": lambda *args: torch.tensor([[0.0]]),
        },
    )
    model = _Model()
    unrestricted = sample_one_transition(
        model, object(), 0.5, np.random.default_rng(3), helpers=_helpers()
    )
    restricted = sample_one_transition(
        model,
        object(),
        0.5,
        np.random.default_rng(3),
        helpers=_helpers(),
        allowed_families=frozenset({"atom_restate"}),
    )
    assert unrestricted.table == "grow_connected"
    assert restricted.table == "atom_restate"
    assert restricted.family_redraws == 0


def test_empty_restricted_fiber_returns_no_mark(monkeypatch):
    monkeypatch.setattr(
        hphi_lazy_family_scores,
        "LAZY_SCORERS",
        {"atom_restate": lambda *args: torch.tensor([[0.0]])},
    )
    draw = sample_one_transition(
        _Model(),
        object(),
        0.5,
        np.random.default_rng(4),
        helpers=_helpers(restate_legal=False),
        allowed_families=frozenset({"atom_restate"}),
    )
    assert draw.table is None
    assert draw.coordinate is None
    assert draw.empty_families == ["atom_restate"]


def test_unknown_allowed_family_fails_loudly():
    with pytest.raises(ValueError, match="unknown model families"):
        sample_one_transition(
            _Model(),
            object(),
            0.5,
            np.random.default_rng(5),
            helpers=_helpers(),
            allowed_families=frozenset({"not_a_family"}),
        )


def test_deployed_active8_fixed_size_family_set():
    deployed = (
        "atom_insert",
        "atom_delete",
        "atom_restate",
        "bond_reorder",
        "bond_reroute",
        "cycle_insert",
        "cycle_attach",
        "ring_system_restate",
    )
    assert fixed_size_allowed_families(deployed) == frozenset(
        {
            "atom_restate",
            "bond_reorder",
            "bond_reroute",
            "cycle_insert",
            "cycle_attach",
            "ring_system_restate",
        }
    )
    with pytest.raises(ValueError, match="unique"):
        fixed_size_allowed_families(("atom_restate", "atom_restate"))
