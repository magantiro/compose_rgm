import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from compose_v4.experiments.t4_repaired_exhaustion_contract import (
    CONTROLLER_IMPLEMENTATION_REVISION,
    PREPARATION_RELATIVE_PATH,
    SCHEMA_VERSION,
    _validate_controller_implementation,
    _validate_support_gate,
    publish_authorization_receipt,
    required_authorization_sentence,
    validate_preparation,
)
from compose_v4.experiments.t4_shared_controller_completion_contract import (
    payload_identity,
)
from compose_v4.experiments.t4_shared_controller_scored_runtime import (
    make_launch_task,
)

ROOT = Path(__file__).resolve().parents[1]


def _preparation():
    return validate_preparation(ROOT / PREPARATION_RELATIVE_PATH)


def test_preparation_binds_one_identical_four_expert_controller_to_two_cells():
    preparation = _preparation()

    assert preparation["campaign_cell_count"] == 2
    assert preparation["charged_calls_per_cell"] == 49
    assert preparation["total_charged_call_ceiling"] == 98
    assert [row["cell_key"] for row in preparation["cells"]] == [
        "braf_0_d06",
        "5ht1b_2_d06",
    ]
    assert len({row["volume"] for row in preparation["cells"]}) == 2
    assert all("controller" not in row for row in preparation["cells"])
    assert all("proposal" not in row for row in preparation["cells"])
    assert preparation["controller"]["proposal_experts"] == [
        "shallow",
        "anchored_replacement",
        "route_complete_region",
        "protonation_aware_retained_subgraph",
    ]
    protonation = preparation["controller"]["proposal"][
        "protonation_aware_retained_subgraph"
    ]
    assert protonation["task_or_target_input_used"] is False
    assert protonation["teacher_or_endpoint_input_used"] is False


def test_preparation_rejects_any_cell_specific_controller_override(tmp_path):
    preparation = copy.deepcopy(_preparation())
    preparation["cells"][0]["controller"] = {"proposal": "override"}
    path = tmp_path / "preparation.json"
    path.write_text(json.dumps(preparation))

    with pytest.raises(ValueError, match="cell-specific controller settings"):
        validate_preparation(path)


def test_controller_and_zero_oracle_gate_are_exactly_bound():
    implementation = _validate_controller_implementation(ROOT)
    _, support_identity = _validate_support_gate(
        ROOT
        / "diagnostics/t4_shared_controller_production_support_gate_v1/attempt_1/result.json"
    )

    assert implementation["source_snapshot_revision"] == (
        CONTROLLER_IMPLEMENTATION_REVISION
    )
    assert implementation["support_repair_code_revision"].startswith("7c3d18f7")
    assert implementation["support_gate_evidence_revision"].startswith("4bc29828")
    assert support_identity == (
        "a587a846b031493077c31de97f63fdc18d2a33b2a7d2bf4b359da2d7a05b020d"
    )


def test_two_cell_launch_task_is_exactly_98_calls_and_payload_authorized():
    preparation = _preparation()
    contract = {
        **preparation,
        "scored_calls_requested": 98,
        "total_charged_call_ceiling": 98,
    }
    contract_identity = "a" * 64
    authorization = {
        "schema_version": "t4_shared_controller_completion_scored_authorization_v4",
        "contract_payload_sha256": contract_identity,
        "authorized_scored_calls": 98,
        "user_statement": required_authorization_sentence(contract_identity),
    }

    task = make_launch_task(
        contract=contract,
        contract_payload_sha256=contract_identity,
        contract_file_sha256="b" * 64,
        authorization_receipt=authorization,
        authorization_receipt_sha256="c" * 64,
        code_revision="d" * 40,
        source_capsule_payload_sha256="e" * 64,
    )

    assert task["charged_call_ceiling_per_cell"] == 49
    assert task["total_charged_call_ceiling"] == 98
    assert task["cell_keys"] == ["braf_0_d06", "5ht1b_2_d06"]
    assert task["automatic_retries"] == 0
    assert task["replacement"] is False
    assert task["backfill"] is False


def test_authorization_is_exactly_payload_bound_and_publish_once(tmp_path):
    first = required_authorization_sentence("a" * 64)
    second = required_authorization_sentence("b" * 64)
    assert first != second
    assert "49 charged docking calls per cell and 98 total" in first
    assert "zero retry, replacement, backfill" in first

    contract_payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "SEALED_PENDING_EXACT_USER_AUTHORIZATION",
    }
    contract_identity = payload_identity(contract_payload)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "payload": contract_payload,
                "payload_sha256": contract_identity,
            }
        )
    )
    authorization_path = tmp_path / "authorization.json"
    with pytest.raises(ValueError, match="must exactly equal"):
        publish_authorization_receipt(
            contract_path=contract_path,
            authorization_path=authorization_path,
            user_statement="broad authorization",
        )
    exact = required_authorization_sentence(contract_identity)
    publish_authorization_receipt(
        contract_path=contract_path,
        authorization_path=authorization_path,
        user_statement=exact,
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        publish_authorization_receipt(
            contract_path=contract_path,
            authorization_path=authorization_path,
            user_statement=exact,
        )


def test_wrapper_selects_isolated_two_cell_app_and_preflight_is_inert():
    script = """
import json
import modal_apps.t4_shared_controller_repaired_exhaustion_v1_app as wrapper
import modal_apps.t4_shared_controller_completion_v1_app as shared
print(json.dumps({
    "app": shared.APP_NAME,
    "cells": list(shared.CELL_KEYS),
    "variant": shared._CAMPAIGN_VARIANT,
    "volumes": list(shared.CELL_VOLUMES.values()),
    "preflight": wrapper.scored_preflight_report(),
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

    assert payload["app"] == "compose-t4-repaired-exhaustion-v1"
    assert payload["variant"] == "repaired_exhaustion_v1"
    assert payload["cells"] == ["braf_0_d06", "5ht1b_2_d06"]
    assert len(set(payload["volumes"])) == 2
    assert payload["preflight"]["ready"] is False
    assert payload["preflight"]["modal_calls_created"] == 0
    assert "authorization" in payload["preflight"]["missing"]


def test_launcher_reserves_before_spawn_and_never_retries():
    source = (
        ROOT / "tools/launch_t4_shared_controller_repaired_exhaustion.py"
    ).read_text()
    reserve = source.index("launcher._replace_launch_receipt(receipt_path, updated)")
    spawn = source.index("call = _driver(cell_key).spawn")

    assert reserve < spawn
    assert '"automatic_retries": 0' in source
    assert '"replacement": False' in source
    assert '"backfill": False' in source
    assert "_plan_driver_resume(" in source
