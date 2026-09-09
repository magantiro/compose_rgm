import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.rewrite.trace_shard import encode_state
from tools.t4_target_path_values import curve_summary, score_saved_path


def test_curve_keeps_the_enabling_declines_not_just_endpoint_gain():
    result = curve_summary([0.4, 0.5, 0.3, 0.2, 0.6, 1.0])
    assert result["decreasing_steps"] == [2, 3]
    assert result["max_drawdown"] == pytest.approx(0.3)
    assert result["max_drawdown_step"] == 3
    assert result["terminal_similarity"] == 1


def test_constant_curve_is_not_improvement():
    result = curve_summary([0.5, 0.5, 0.5])
    assert result["decreasing_edits"] == 0
    assert result["max_drawdown"] == 0


@pytest.mark.parametrize("values", [[], [1], [0, float("nan")], [0, 1.1]])
def test_bad_curve_fails(values):
    with pytest.raises(ValueError):
        curve_summary(values)


def test_saved_state_scoring_checks_exact_endpoint_and_endpoint_feasibility():
    row = {"source": "C", "target": "CC", "pair_id": "fixture"}
    payload = {
        "path": {
            "status": "witness_found",
            "actions": [{"fixture_only": True}],
            "states": [encode_state(smiles_to_molecular_graph(s)) for s in ("C", "CC")],
        }
    }
    result = score_saved_path(row, payload)
    assert result["states"][-1]["exact_target"]
    assert result["terminal_similarity"] == 1
    assert result["t4_infeasible_steps"] == [0, 1]
    with pytest.raises(ValueError, match="does not equal"):
        score_saved_path({**row, "target": "CCC"}, payload)
