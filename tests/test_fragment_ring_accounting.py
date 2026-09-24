import pytest

from tools.repair_fragment_ring_accounting import repair_attempt
from tools.run_fragment_joint_prior_gate import prompt_ring_count


@pytest.mark.parametrize("smiles, expected", [("CCO", 0), ("c1ccccc1", 1), ("c1ccc2ccccc2c1", 2)])
def test_ring_info_parent_remains_alive(smiles, expected):
    assert prompt_ring_count(smiles) == expected


def test_accounting_repair_changes_only_capability_labels():
    attempt = {
        "committed_smiles": "Cc1ccccc1",
        "selected_capabilities": {"new_ring": True},
        "offered": [
            {
                "endpoint": "Cc1ccccc1",
                "capabilities": {"new_ring": True},
                "actions": [1, 2],
                "mean_log_mark": -2.0,
            }
        ],
        "rng_state_after": {"exact": [12, 99]},
        "selection": {"candidate_index": 0},
    }
    corrected, changes = repair_attempt(attempt, 1)
    assert changes == 2
    assert not corrected["selected_capabilities"]["new_ring"]
    assert not corrected["offered"][0]["capabilities"]["new_ring"]
    corrected["selected_capabilities"]["new_ring"] = True
    corrected["offered"][0]["capabilities"]["new_ring"] = True
    assert corrected == attempt
