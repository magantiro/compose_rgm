import json
from pathlib import Path

import pytest

from compose_v4.experiments.t4_compose_nodistill_contract import (
    CAPSULE_INCLUDE,
    PREPARATION_RELATIVE_PATH,
    USER_AUTHORIZATION_STATEMENT,
    _validate_preparation,
)

ROOT = Path(__file__).resolve().parents[1]


def test_nodistill_preparation_removes_only_explicit_route_distillation():
    preparation = _validate_preparation(ROOT / PREPARATION_RELATIVE_PATH)
    assert preparation["trajectory_distillation"] == {
        "enabled": False,
        "stored_structural_template_library": False,
        "route_marginal_or_scale_weights": False,
        "direct_template_rebinding": False,
        "route_template_particles": False,
        "route_template_or_scale_quota": False,
        "route_checkpoint_allowed": False,
    }
    assert preparation["controller"]["route_scale_floor_rounds"] == 0
    assert preparation["controller"]["expert_floor_rounds"] == 2
    assert (
        "recursive_archive_descendants" in preparation["generic_capabilities_retained"]
    )
    assert "diagnostics/t4_shared_retained_rewrite_v1/checkpoint.json" not in (
        CAPSULE_INCLUDE
    )
    assert USER_AUTHORIZATION_STATEMENT


def test_nodistill_preparation_fails_if_route_checkpoint_is_added(tmp_path):
    preparation = json.loads((ROOT / PREPARATION_RELATIVE_PATH).read_text())
    preparation["shared_route_checkpoint"] = {"path": "forbidden"}
    path = tmp_path / "preparation.json"
    path.write_text(json.dumps(preparation))
    with pytest.raises(ValueError, match="may not bind"):
        _validate_preparation(path)
