from __future__ import annotations

import copy
from pathlib import Path

import pytest

from compose_v4.experiments.editing_training_gate import (
    REQUIRED_P50_FAMILIES,
    EditingTrainingGateError,
    assert_full_training_launch_authorized,
    assert_p50_launch_authorized,
    full_training_launch_blockers,
    load_editing_training_gate,
    p50_launch_blockers,
    resolve_p50_thresholds,
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


def test_p50_is_exactly_fifty_optimizer_steps() -> None:
    contract = load_editing_training_gate(CONTRACT_PATH)
    p50 = next(
        gate
        for gate in contract["gates"]
        if gate["id"] == "P50_gradient_and_collapse_sentinel"
    )
    assert p50["maximum_optimizer_steps"] == 50

    broken = copy.deepcopy(contract)
    broken["gates"][3]["maximum_optimizer_steps"] = 51
    with pytest.raises(EditingTrainingGateError, match="exactly 50"):
        validate_editing_training_gate(broken)


def test_test_partition_cannot_select_checkpoint() -> None:
    contract = load_editing_training_gate(CONTRACT_PATH)
    broken = copy.deepcopy(contract)
    broken["checkpoint_selection"]["partition"] = "test"
    with pytest.raises(EditingTrainingGateError, match="validation only"):
        validate_editing_training_gate(broken)


def _freeze_p50_thresholds(contract):
    frozen = copy.deepcopy(contract)
    thresholds = frozen["gates"][3]["numeric_thresholds"]
    thresholds["minimum_gradient_updates_per_required_slice"] = {
        family: 1 for family in REQUIRED_P50_FAMILIES
    }
    thresholds["maximum_required_slice_successor_nll_regression"] = {
        family: 0.25 for family in REQUIRED_P50_FAMILIES
    }
    thresholds["maximum_inherited_probe_nll_regression"] = {
        "scratch": "NOT_APPLICABLE",
        "compatible_warm_start": 0.2,
        "compatible_warm_start_with_retention": 0.1,
    }
    return frozen


def test_unfrozen_p50_thresholds_refuse_before_loader_construction() -> None:
    contract = load_editing_training_gate(CONTRACT_PATH)
    blockers = p50_launch_blockers(
        contract,
        initialization_regime="scratch",
    )
    assert blockers
    assert "per-family object" in blockers[0]
    with pytest.raises(
        EditingTrainingGateError,
        match="before loader construction",
    ):
        assert_p50_launch_authorized(
            contract,
            initialization_regime="scratch",
        )


def test_p50_thresholds_exactly_cover_families_and_retention_regime() -> None:
    contract = _freeze_p50_thresholds(
        load_editing_training_gate(CONTRACT_PATH)
    )
    scratch = resolve_p50_thresholds(
        contract,
        initialization_regime="scratch",
    )
    assert scratch.inherited_retention_status == "NOT_APPLICABLE"
    assert scratch.maximum_inherited_probe_nll_regression is None
    assert set(scratch.minimum_gradient_update_map) == set(
        REQUIRED_P50_FAMILIES
    )

    warm = resolve_p50_thresholds(
        contract,
        initialization_regime="compatible_warm_start",
    )
    assert warm.inherited_retention_status == "REQUIRED"
    assert warm.maximum_inherited_probe_nll_regression == 0.2

    missing = copy.deepcopy(contract)
    del missing["gates"][3]["numeric_thresholds"][
        "minimum_gradient_updates_per_required_slice"
    ]["cycle_attach"]
    with pytest.raises(
        EditingTrainingGateError,
        match="exactly cover required families",
    ):
        resolve_p50_thresholds(
            missing,
            initialization_regime="scratch",
        )


def test_scratch_and_warm_start_retention_cannot_masquerade() -> None:
    contract = _freeze_p50_thresholds(
        load_editing_training_gate(CONTRACT_PATH)
    )
    contract["gates"][3]["numeric_thresholds"][
        "maximum_inherited_probe_nll_regression"
    ]["scratch"] = 0.0
    with pytest.raises(
        EditingTrainingGateError,
        match="NOT_APPLICABLE",
    ):
        resolve_p50_thresholds(
            contract,
            initialization_regime="scratch",
        )

    contract = _freeze_p50_thresholds(
        load_editing_training_gate(CONTRACT_PATH)
    )
    contract["gates"][3]["numeric_thresholds"][
        "maximum_inherited_probe_nll_regression"
    ]["compatible_warm_start"] = "NOT_APPLICABLE"
    with pytest.raises(
        EditingTrainingGateError,
        match="warm-start",
    ):
        resolve_p50_thresholds(
            contract,
            initialization_regime="compatible_warm_start",
        )


def test_full_launch_finds_nested_unfrozen_thresholds() -> None:
    contract = _freeze_p50_thresholds(
        load_editing_training_gate(CONTRACT_PATH)
    )
    contract["gates"][3]["numeric_thresholds"][
        "minimum_gradient_updates_per_required_slice"
    ]["bond_reroute"] = None
    blockers = full_training_launch_blockers(contract)
    assert any(
        blocker.endswith(
            "minimum_gradient_updates_per_required_slice.bond_reroute"
        )
        for blocker in blockers
    )
