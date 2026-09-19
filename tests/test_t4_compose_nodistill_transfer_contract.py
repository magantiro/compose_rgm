import json
import subprocess
import sys
from pathlib import Path

import pytest

from compose_v4.experiments.t4_compose_nodistill_transfer_contract import (
    BRAF_SUPPORT_RELATIVE_PATH,
    CAPSULE_INCLUDE,
    JAK2_SUPPORT_RELATIVE_PATH,
    PREPARATION_RELATIVE_PATH,
    _validate_braf_support,
    _validate_jak2_support,
    _validate_preparation,
    required_authorization_statement,
)

ROOT = Path(__file__).resolve().parents[1]


def test_transfer_preparation_is_matched_modern_nodistill():
    preparation = _validate_preparation(ROOT / PREPARATION_RELATIVE_PATH)
    assert [row["cell_key"] for row in preparation["cells"]] == [
        "jak2_1_d06",
        "braf_1_d04",
        "braf_0_d06",
    ]
    assert preparation["trajectory_distillation"]["enabled"] is False
    assert preparation["trajectory_distillation"]["target_independent_scale_floor"]
    assert preparation["controller"]["scale_floor_scope"] == "all_generic"
    assert preparation["controller"]["route_scale_floor_rounds"] == 2
    assert "one_to_32_primitive_protected_program_support" in preparation[
        "generic_capabilities_retained"
    ]
    assert all("checkpoint" not in path for path in CAPSULE_INCLUDE)


def test_transfer_preparation_rejects_route_checkpoint(tmp_path):
    preparation = json.loads((ROOT / PREPARATION_RELATIVE_PATH).read_text())
    preparation["shared_route_checkpoint"] = {"path": "forbidden"}
    path = tmp_path / "preparation.json"
    path.write_text(json.dumps(preparation))
    with pytest.raises(ValueError, match="may not bind"):
        _validate_preparation(path)


def test_transfer_support_gates_are_zero_oracle_and_cover_all_cells():
    _, braf = _validate_braf_support(ROOT / BRAF_SUPPORT_RELATIVE_PATH)
    _, jak2 = _validate_jak2_support(ROOT / JAK2_SUPPORT_RELATIVE_PATH)
    assert set(braf) == {"braf_1_d04", "braf_0_d06"}
    assert set(jak2) == {"jak2_1_d06"}
    assert all(value > 0 for value in {**braf, **jak2}.values())


def test_transfer_authorization_is_payload_specific():
    first = required_authorization_statement("a" * 64)
    second = required_authorization_statement("b" * 64)
    assert first != second
    assert "a" * 64 in first
    assert "147 scored calls" in first


def test_transfer_wrapper_selects_isolated_modal_variant():
    script = """
import json
import modal_apps.t4_compose_nodistill_transfer_v1_app
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
        "app": "compose-t4-compose-nodistill-transfer-v1",
        "cells": ["jak2_1_d06", "braf_1_d04", "braf_0_d06"],
        "variant": "nodistill_transfer_v1",
    }
