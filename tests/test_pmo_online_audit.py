"""Tiny accounting fixtures, no chemistry generation or oracle evaluations."""

import copy

import pytest

from tools.pmo_online_audit import summarize


def test_warm_window_accounting_and_lock_failures():
    result = {
        "status": "complete_bounded_development",
        "evaluation_policy_frozen": False,
        "new_oracle_calls": 0,
        "historical_calls": 1,
        "seconds": 1,
        "online_updates": [{"model_sha256": str(i), "seconds": 0} for i in range(1, 8)],
        "arms": {},
    }
    phases = {
        n: {
            "choices": {
                "choices": [],
                "policy_by_arm": {"balanced": "fixed", "frozen": "fixed", "learned": str(n - 1)},
            },
            "proposals": {"worker": {"attempts": [], "seconds": 0, "initialization_seconds": 0}},
        }
        for n in range(1, 9)
    }
    for arm in ("balanced", "frozen", "learned"):
        result["arms"][arm] = {
            "initial": {"best": 0.5},
            "best": 0.5,
            "top10_sum": 0.5,
            "mean_selected_score": None,
            "mean_parent_delta": None,
            "selections": [],
            "curve": [{"best": 0.5, "top10_sum": 0.5}] * 32,
        }
        for n in range(1, 9):
            for slot in range(4):
                result["arms"][arm]["selections"].append(
                    {"round": n, "slot": slot, "status": "no_unqueried_candidate"}
                )
                phases[n]["choices"]["choices"].append(
                    {
                        "arm": arm,
                        "slot": slot,
                        "selected": None,
                        "pool": [],
                        "excluded": [],
                        "parent": {"smiles": "C"},
                    }
                )
    prepared = {"observed": {"C": 0.5}, "historical_calls": 1}
    report = summarize(result, prepared, phases)
    assert report["arms"]["balanced"]["abstentions"] == 32
    assert report["arms"]["learned"]["distinct_policy_hashes"] == 8
    assert report["arms"]["learned"]["warm_window_top10_trapezoid_auc"] == pytest.approx(0.05)
    bad = copy.deepcopy(result)
    bad["arms"]["balanced"]["curve"][0]["best"] = 0.6
    with pytest.raises(ValueError, match="curve does not replay"):
        summarize(bad, prepared, phases)
    bad_phases = copy.deepcopy(phases)
    bad_phases[1]["choices"]["choices"][0]["selected"] = 0
    with pytest.raises(ValueError, match="abstention disagrees"):
        summarize(result, prepared, bad_phases)
