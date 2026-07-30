from __future__ import annotations

import copy
from pathlib import Path

import pytest

from compose_v4.data.editing_corpus_contract import (
    EditingCorpusContractError,
    assert_training_launch_authorized,
    load_editing_corpus_contract,
    training_launch_blockers,
    validate_editing_corpus_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs" / "editing_corpus_v2_contract.json"


def test_v2_contract_is_well_formed_but_intentionally_blocks_training() -> None:
    contract = load_editing_corpus_contract(CONTRACT_PATH)
    blockers = training_launch_blockers(contract)
    assert "training_authorized is not true" in blockers
    assert any(item.startswith("unfrozen threshold:") for item in blockers)
    with pytest.raises(EditingCorpusContractError, match="blocks training"):
        assert_training_launch_authorized(contract)


def test_contract_rejects_unknown_capability_lane() -> None:
    contract = load_editing_corpus_contract(CONTRACT_PATH)
    broken = copy.deepcopy(contract)
    broken["capabilities"][0]["evidence_lanes"].append("invented_lane")
    with pytest.raises(EditingCorpusContractError, match="unknown lanes"):
        validate_editing_corpus_contract(broken)


@pytest.mark.parametrize("mutation", ["reorder", "evidence_class"])
def test_contract_rejects_lane_identity_or_evidence_drift(
    mutation: str,
) -> None:
    contract = load_editing_corpus_contract(CONTRACT_PATH)
    broken = copy.deepcopy(contract)
    if mutation == "reorder":
        broken["data_lanes"][0], broken["data_lanes"][1] = (
            broken["data_lanes"][1],
            broken["data_lanes"][0],
        )
    else:
        broken["data_lanes"][0]["evidence_class"] = "synthetic"
    with pytest.raises(EditingCorpusContractError, match="data-lane"):
        validate_editing_corpus_contract(broken)


def test_contract_rejects_required_disabled_operator_overlap() -> None:
    contract = load_editing_corpus_contract(CONTRACT_PATH)
    broken = copy.deepcopy(contract)
    broken["operator_basis"]["disabled_by_default"].append("atom_insert")
    with pytest.raises(EditingCorpusContractError, match="both required and disabled"):
        validate_editing_corpus_contract(broken)


@pytest.mark.parametrize(
    "roles",
    [
        ["train", "validation", "final_test"],
        ["train", "controller_validation", "validation", "final_test"],
    ],
)
def test_contract_rejects_missing_or_reordered_partition_roles(
    roles: list[str],
) -> None:
    contract = load_editing_corpus_contract(CONTRACT_PATH)
    broken = copy.deepcopy(contract)
    broken["split_contract"]["partition_roles"] = roles
    with pytest.raises(EditingCorpusContractError, match="partition"):
        validate_editing_corpus_contract(broken)
