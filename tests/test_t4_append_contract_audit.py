"""Offline audit boundaries on a synthetic closure, not a model sample."""

from copy import deepcopy

import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.rewrite.action_codec_v4 import encode_action
from compose_v4.rewrite.kernel import editing_v2_rewrite_system
from compose_v4.rewrite.operators import CycleCloseEdge
from compose_v4.rewrite.trace_shard import encode_state
from tools.t4_append_contract_audit import SCHEMA, audit_fixture


def fixture():
    source = pad_molecular_graph(smiles_to_molecular_graph("c1ccccc1CCCCCC"), 48)
    mark = CycleCloseEdge(6, 11, 1)
    product = editing_v2_rewrite_system().apply(source, "cycle_close", mark)
    row = {
        "bundle_id": "model-free-fixture",
        "source": encode_state(source),
        "mark": encode_action("cycle_close", mark),
        "product": encode_state(product),
        "expected_accept": True,
        "historical_accept": False,
    }
    return {
        "schema_version": SCHEMA,
        "minimum_new_system_atoms": 6,
        "rows": [{**row, "arm": arm} for arm in ("reference", "committor")],
    }


def test_audit_preserves_paired_denominators_but_shares_exact_replays():
    result = audit_fixture(fixture())
    assert result["all_expected_decisions_match"]
    assert result["unique_source_mark_replays"] == 1
    assert result["new_model_calls"] == result["new_oracle_calls"] == 0
    for values in result["metrics"].values():
        assert values["closure_products"] == values["true_positive"] == 1
        assert values["coverage"] == values["precision"] == 1


@pytest.mark.parametrize("field", ["schema_version", "minimum_new_system_atoms"])
def test_incompatible_fixture_fails(field):
    data = fixture()
    data[field] = "wrong"
    with pytest.raises(ValueError):
        audit_fixture(data)


def test_corrupt_saved_product_fails_exact_replay():
    data = deepcopy(fixture())
    data["rows"][0]["product"] = data["rows"][0]["source"]
    with pytest.raises(ValueError, match="replay differs"):
        audit_fixture(data)


def test_historical_decision_drift_fails():
    data = fixture()
    data["rows"][0]["historical_accept"] = True
    with pytest.raises(ValueError, match="historical decision"):
        audit_fixture(data)


def test_mismatched_expected_label_reports_false_positive_not_success():
    data = fixture()
    data["rows"][0]["expected_accept"] = False
    result = audit_fixture(data)
    assert not result["all_expected_decisions_match"]
    assert result["metrics"]["reference"]["false_positive"] == 1
    assert result["metrics"]["reference"]["coverage"] is None
    assert result["metrics"]["reference"]["precision"] == 0
