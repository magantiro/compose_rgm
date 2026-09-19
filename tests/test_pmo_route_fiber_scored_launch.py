from pathlib import Path

from compose_v4.experiments.pmo_route_fiber_scored_pilot import (
    aggregate_scored_units,
    build_scored_launch_receipt,
)
from compose_v4.experiments.pmo_route_fiber_scored_pilot_contract import (
    ARMS,
    TASKS,
)

ROOT = Path(__file__).resolve().parents[1]


def test_scored_launch_receipt_binds_frozen_inputs():
    launch = build_scored_launch_receipt(ROOT, code_revision="a" * 40)
    assert launch["scored_launch_authorized"] is True
    assert launch["charged_call_ceiling"] == 384
    assert launch["automatic_retries"] == 0
    assert set(launch["tasks"]) == set(TASKS)
    assert set(launch["arms"]) == set(ARMS)


def test_scored_aggregate_requires_complete_eight_unit_census():
    launch = build_scored_launch_receipt(ROOT, code_revision="b" * 40)
    results = []
    for task in TASKS:
        for arm in ARMS:
            route = arm == "additive_route_fiber_control"
            results.append(
                {
                    "schema_version": "pmo_route_fiber_scored_unit_v1",
                    "status": "complete",
                    "task": task,
                    "arm": arm,
                    "charged_calls": 48,
                    "new_oracle_calls": 48,
                    "auc_top10_48": 0.8 if route else 0.7,
                    "best_reward": 0.9 if route else 0.8,
                    "observations": [
                        {
                            "proposal_lane": "route" if route else "v0",
                            "improvement": 0.1 if route else 0.0,
                        }
                    ],
                }
            )
    result = aggregate_scored_units(launch, results)
    assert result["decision"] == "PROMOTE"
    assert result["charged_calls"] == 384
