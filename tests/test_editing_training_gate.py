from __future__ import annotations

import copy
from pathlib import Path

import pytest

from compose_v4.experiments.editing_training_gate import (
    EditingTrainingGateError,
    assert_full_training_launch_authorized,
    full_training_launch_blockers,
    load_editing_training_gate,
    validate_editing_training_gate,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs" / "editing_training_v2_gate.json"


def test_gate_contract_is_valid_but_blocks_full_training() -> None:
    contract = load_editing_training_gate(CONTRACT_PATH)
    blockers = full_training_launch_blockers(contract)
    assert "full_training_authorized is not true" in blockers
    assert any("unfrozen threshold" in blocker for blocker in blockers)
    with pytest.raises(EditingTrainingGateError, match="blocks a full run"):
        assert_full_training_launch_authorized(contract)


def test_family_choice_accuracy_cannot_be_primary() -> None:
    contract = load_editing_training_gate(CONTRACT_PATH)
    broken = copy.deepcopy(contract)
    broken["metric_contract"]["primary"][0] = "family_accuracy"
    with pytest.raises(EditingTrainingGateError, match="primary metrics"):
        validate_editing_training_gate(broken)


def test_gate_sequence_cannot_skip_short_pilots() -> None:
    contract = load_editing_training_gate(CONTRACT_PATH)
    broken = copy.deepcopy(contract)
    del broken["gates"][3]
    with pytest.raises(EditingTrainingGateError, match="gate sequence"):
        validate_editing_training_gate(broken)


def test_test_partition_cannot_select_checkpoint() -> None:
    contract = load_editing_training_gate(CONTRACT_PATH)
    broken = copy.deepcopy(contract)
    broken["checkpoint_selection"]["partition"] = "test"
    with pytest.raises(EditingTrainingGateError, match="validation only"):
        validate_editing_training_gate(broken)
