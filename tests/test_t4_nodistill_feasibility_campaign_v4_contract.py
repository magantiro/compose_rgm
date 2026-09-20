import json
import subprocess
import sys
from pathlib import Path

import pytest

from compose_v4.experiments.t4_nodistill_feasibility_campaign_v4_contract import (
    AUTHORIZATION_RELATIVE_PATH,
    CAPSULE_INCLUDE,
    PREPARATION_RELATIVE_PATH,
    SUPPORT_GATE_RELATIVE_PATH,
    _validate_preparation,
    _validate_support_gate,
    required_authorization_statement,
)

ROOT = Path(__file__).resolve().parents[1]


def test_v4_preparation_binds_three_fresh_49_call_cells_and_four_experts():
    preparation = _validate_preparation(ROOT / PREPARATION_RELATIVE_PATH)
    assert [row["cell_key"] for row in preparation["cells"]] == [
        "jak2_0_d06",
        "parp1_0_d04",
        "5ht1b_1_d04",
    ]
    assert preparation["charged_calls_per_cell"] == 49
    assert preparation["total_charged_call_ceiling"] == 147
    assert preparation["controller"]["batch"] == 8
    assert preparation["controller"]["proposal_experts"] == [
        "shallow",
        "anchored_replacement",
        "route_complete_region",
        "generic_feasibility_headroom_v4",
    ]
    assert preparation["trajectory_distillation"]["enabled"] is False
    assert all("checkpoint" not in path for path in CAPSULE_INCLUDE)


def test_v4_preparation_rejects_support_or_expert_order_drift(tmp_path):
    preparation = json.loads((ROOT / PREPARATION_RELATIVE_PATH).read_text())
    proposal = preparation["controller"]["proposal"]
    proposal["generic_feasibility_headroom_v4"]["maximum_blocks"] = 9
    preparation["controller"]["proposal_experts"].reverse()
    path = tmp_path / "preparation.json"
    path.write_text(json.dumps(preparation))
    with pytest.raises(ValueError, match="append-only|support"):
        _validate_preparation(path)


def test_v4_support_gate_is_frozen_zero_oracle_and_covers_every_scored_cell():
    identity, support = _validate_support_gate(ROOT / SUPPORT_GATE_RELATIVE_PATH)
    assert len(identity) == 64
    assert set(support) == {"jak2_0_d06", "parp1_0_d04", "5ht1b_1_d04"}
    assert all(row["archive_ready_unique"] >= 8 for row in support.values())
    assert all(row["macro_plan_identities"] >= 3 for row in support.values())
    assert all(row["macro_modes"] >= 3 for row in support.values())


def test_v4_authorization_is_payload_specific_and_not_precreated():
    first = required_authorization_statement("a" * 64)
    second = required_authorization_statement("b" * 64)
    assert first != second
    assert "147 scored calls" in first
    assert not (ROOT / AUTHORIZATION_RELATIVE_PATH).exists()


def test_v4_wrapper_selects_isolated_three_cell_modal_variant():
    script = """
import json
import modal_apps.t4_nodistill_feasibility_v4_app
import modal_apps.t4_shared_controller_completion_v1_app as shared
print(json.dumps({
    "app": shared.APP_NAME,
    "cells": list(shared.CELL_KEYS),
    "variant": shared._CAMPAIGN_VARIANT,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload == {
        "app": "compose-t4-nodistill-feasibility-v4",
        "cells": ["jak2_0_d06", "parp1_0_d04", "5ht1b_1_d04"],
        "variant": "nodistill_feasibility_v4",
    }


def test_v4_shared_app_dispatches_generation_before_runtime_binding():
    source = (ROOT / "modal_apps/t4_shared_controller_completion_v1_app.py").read_text()
    start = source.index('elif expert_name == "generic_feasibility_headroom_v4":')
    end = source.index("    else:", start)
    branch = source[start:end]
    assert "feasibility_headroom_v4_records(" in branch
    assert 'row["parent_score"] = request["parent_score"]' in branch
    assert 'row["delta"] = cell["delta"]' in branch
    assert branch.index("feasibility_headroom_v4_records(") < branch.index(
        'row["parent_score"]'
    )
