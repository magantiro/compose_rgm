from __future__ import annotations

import pytest
import torch

from scripts.train_tracelet_cnof_gate import _compatible_checkpoint_initialization


def test_compatible_initialization_transfers_only_matching_named_shapes() -> None:
    target = {
        "shared.weight": torch.zeros((2, 3)),
        "resized.weight": torch.full((3, 2), 7.0),
        "new_head.bias": torch.full((2,), 9.0),
    }
    payload = {
        "best_state_dict": {
            "shared.weight": torch.ones((2, 3)),
            "resized.weight": torch.ones((2, 2)),
            "old_only.bias": torch.ones((2,)),
        }
    }

    initialized, transferred, retained = _compatible_checkpoint_initialization(
        target,
        payload,
    )

    assert transferred == ("shared.weight",)
    assert retained == ("resized.weight", "new_head.bias")
    assert torch.equal(initialized["shared.weight"], torch.ones((2, 3)))
    assert torch.equal(initialized["resized.weight"], target["resized.weight"])
    assert torch.equal(initialized["new_head.bias"], target["new_head.bias"])


def test_compatible_initialization_requires_a_transferable_tensor() -> None:
    with pytest.raises(ValueError, match="no shape-compatible tensors"):
        _compatible_checkpoint_initialization(
            {"new.weight": torch.zeros((2, 2))},
            {"state_dict": {"old.weight": torch.ones((2, 2))}},
        )
