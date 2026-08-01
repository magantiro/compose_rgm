from __future__ import annotations

import json
from pathlib import Path

from compose_v4.data.editing_corpus_contract import load_editing_corpus_contract
from compose_v4.data.editing_v2_split_census import default_split_census_policy

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs" / "editing_corpus_v2_contract.json"
POLICY_PATH = ROOT / "configs" / "editing_v2_split_census_policy_v4.json"


def test_frozen_split_census_policy_is_the_exact_structural_default() -> None:
    policy = json.loads(POLICY_PATH.read_bytes())
    contract = load_editing_corpus_contract(CONTRACT_PATH)

    assert policy == default_split_census_policy(
        contract,
        identity_definitions=policy["identity_definitions"],
    )
    assert policy["edge_modes"]["exact_molecule"] == "hard"
    assert policy["edge_modes"]["partition_scaffold"] == "hard"
    assert policy["edge_modes"]["source_group"] == "hard"
    assert policy["edge_modes"]["transformation_signature"] == "diagnostic"
