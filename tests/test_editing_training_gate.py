from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from compose_v4.experiments.editing_training_gate import (
    BOUNDED_PILOT_DISABLED_FAMILIES,
    P50_PREREQUISITE_EVIDENCE_FIELDS,
    REQUIRED_P50_FAMILIES,
    EditingTrainingGateError,
    assert_full_training_launch_authorized,
    assert_p50_launch_authorized,
    full_training_launch_blockers,
    load_editing_training_gate,
    p50_launch_blockers,
    resolve_p50_prerequisite_evidence,
    resolve_p50_thresholds,
    validate_editing_training_gate,
    verify_p50_prerequisite_artifacts,
)
from compose_v4.experiments.successor_micro_overfit import (
    RINGCORE_EDITING_FAMILIES,
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


def test_bounded_pilot_requires_exact_active_eight_without_production_promotion() -> None:
    contract = load_editing_training_gate(CONTRACT_PATH)
    freeze = contract["bounded_pilot_operator_freeze"]
    assert REQUIRED_P50_FAMILIES == RINGCORE_EDITING_FAMILIES
    assert tuple(freeze["required_families"]) == RINGCORE_EDITING_FAMILIES
    assert tuple(freeze["disabled_families"]) == BOUNDED_PILOT_DISABLED_FAMILIES
    assert freeze["final_production_support_authorized"] is False
    assert "ring_system_restate" in contract["development_panels"][
        "required_semantic_slices"
    ]
    assert contract["development_panels"]["conditional_slices"] == []
    assert contract["development_panels"][
        "unique_state_deterministic_panels_required_before_p50"
    ] is True
    empirical = contract["development_panels"][
        "repeated_state_empirical_law_panel"
    ]
    assert empirical["bounded_p50_capacity_prerequisite"] is False
    assert empirical["raw_record_multiplicity_is_observation_count"] is False
    assert empirical["mark_alias_multiplicity_is_observation_count"] is False
    assert empirical["required_for_empirical_law_claims"] is True

    for mutation in ("missing_restate", "enable_delete", "promote_production"):
        broken = copy.deepcopy(contract)
        if mutation == "missing_restate":
            broken["bounded_pilot_operator_freeze"]["required_families"].pop()
        elif mutation == "enable_delete":
            broken["bounded_pilot_operator_freeze"]["disabled_families"].remove(
                "ring_system_delete"
            )
        else:
            broken["bounded_pilot_operator_freeze"][
                "final_production_support_authorized"
            ] = True
        with pytest.raises(EditingTrainingGateError, match="operator freeze"):
            validate_editing_training_gate(broken)

    conditional = copy.deepcopy(contract)
    conditional["development_panels"]["required_semantic_slices"].remove(
        "ring_system_restate"
    )
    conditional["development_panels"]["conditional_slices"] = [
        "ring_system_restate"
    ]
    with pytest.raises(EditingTrainingGateError, match="required bounded-pilot"):
        validate_editing_training_gate(conditional)


def test_test_partition_cannot_select_checkpoint() -> None:
    contract = load_editing_training_gate(CONTRACT_PATH)
    broken = copy.deepcopy(contract)
    broken["checkpoint_selection"]["partition"] = "test"
    with pytest.raises(EditingTrainingGateError, match="validation only"):
        validate_editing_training_gate(broken)


def test_repeated_state_counts_cannot_be_fabricated_for_t1() -> None:
    contract = load_editing_training_gate(CONTRACT_PATH)
    for field in (
        "bounded_p50_capacity_prerequisite",
        "raw_record_multiplicity_is_observation_count",
        "mark_alias_multiplicity_is_observation_count",
    ):
        broken = copy.deepcopy(contract)
        broken["development_panels"]["repeated_state_empirical_law_panel"][
            field
        ] = True
        with pytest.raises(EditingTrainingGateError, match="observation receipts"):
            validate_editing_training_gate(broken)

    broken = copy.deepcopy(contract)
    t1 = next(
        gate
        for gate in broken["gates"]
        if gate["id"] == "T1_true_successor_micro_overfit"
    )
    t1["requirements"][1] = "repeated_state_panels_fit_raw_duplicate_frequency"
    with pytest.raises(EditingTrainingGateError, match="separate unique-state"):
        validate_editing_training_gate(broken)


def test_t1_unique_state_thresholds_are_frozen_before_results() -> None:
    contract = load_editing_training_gate(CONTRACT_PATH)
    t1 = next(
        gate
        for gate in contract["gates"]
        if gate["id"] == "T1_true_successor_micro_overfit"
    )
    assert t1["numeric_thresholds"] == {
        "minimum_unique_state_teacher_successor_top1": 0.95,
        "minimum_unique_state_teacher_successor_probability": 0.8,
        "maximum_unique_state_teacher_successor_nll": 0.22314355131420976,
    }

    broken = copy.deepcopy(contract)
    broken["gates"][2]["numeric_thresholds"][
        "minimum_unique_state_teacher_successor_probability"
    ] = None
    with pytest.raises(EditingTrainingGateError, match="thresholds must be frozen"):
        validate_editing_training_gate(broken)

    broken = copy.deepcopy(contract)
    broken["dependencies"]["semantic_t1_capacity_policy_sha256"] = "0" * 64
    with pytest.raises(EditingTrainingGateError, match="exact prospective"):
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
    assert any("per-family object" in blocker for blocker in blockers)
    assert any("prerequisite evidence is not frozen" in blocker for blocker in blockers)
    with pytest.raises(
        EditingTrainingGateError,
        match="before loader construction",
    ):
        assert_p50_launch_authorized(
            contract,
            initialization_regime="scratch",
        )


def test_frozen_numeric_thresholds_cannot_bypass_missing_p50_prerequisites() -> None:
    contract = _freeze_p50_thresholds(
        load_editing_training_gate(CONTRACT_PATH)
    )
    blockers = p50_launch_blockers(
        contract,
        initialization_regime="scratch",
    )
    assert len(blockers) == len(P50_PREREQUISITE_EVIDENCE_FIELDS) + 2
    assert any("bounded_p50_authorized is not true" in item for item in blockers)
    assert any("status is not FROZEN_BOUNDED_P50_AUTHORIZED" in item for item in blockers)
    assert sum(
        "prerequisite evidence is not frozen" in item for item in blockers
    ) == len(P50_PREREQUISITE_EVIDENCE_FIELDS)

    for index, field in enumerate(P50_PREREQUISITE_EVIDENCE_FIELDS, start=1):
        contract["gates"][3]["prerequisite_evidence"][field] = str(index) * 64
    contract["status"] = "FROZEN_BOUNDED_P50_AUTHORIZED"
    contract["bounded_p50_authorized"] = True
    resolved = assert_p50_launch_authorized(
        contract,
        initialization_regime="scratch",
    )
    assert resolved.required_families == REQUIRED_P50_FAMILIES
    assert resolve_p50_prerequisite_evidence(contract) == {
        field: str(index) * 64
        for index, field in enumerate(
            P50_PREREQUISITE_EVIDENCE_FIELDS,
            start=1,
        )
    }


def test_hash_matching_arbitrary_json_cannot_authorize_p50(
    tmp_path,
) -> None:
    contract = _freeze_p50_thresholds(
        load_editing_training_gate(CONTRACT_PATH)
    )
    paths = {}
    for index, field in enumerate(P50_PREREQUISITE_EVIDENCE_FIELDS):
        path = tmp_path / f"prerequisite-{index}.json"
        path.write_text(f'{{"field": "{field}"}}\n')
        paths[field] = path
        contract["gates"][3]["prerequisite_evidence"][field] = (
            hashlib.sha256(path.read_bytes()).hexdigest()
        )

    contract["status"] = "FROZEN_BOUNDED_P50_AUTHORIZED"
    contract["bounded_p50_authorized"] = True
    with pytest.raises(
        EditingTrainingGateError,
        match="prerequisite semantic validation failed",
    ):
        verify_p50_prerequisite_artifacts(
            contract,
            paths,
            expected_launch={},
        )

    paths[P50_PREREQUISITE_EVIDENCE_FIELDS[-1]].write_text("mutated recipe")
    with pytest.raises(
        EditingTrainingGateError,
        match="prerequisite artifact SHA-256 mismatch",
    ):
        verify_p50_prerequisite_artifacts(
            contract,
            paths,
            expected_launch={},
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
