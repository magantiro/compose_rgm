from __future__ import annotations

import json
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import sha256_file
from compose_v4.experiments.t4_matched_pilot import unseal

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/t4_integrated_route_fiber_parp1_v1.json"
REFERENCE = ROOT / "configs/t4_integrated_route_fiber_v1_1.json"


def test_parp1_transfer_preserves_the_frozen_controller() -> None:
    candidate = unseal(CONTRACT)
    reference = unseal(REFERENCE)

    for field in (
        "delta",
        "support",
        "charged_calls_per_cell",
        "batch",
        "parents",
        "parent_explore",
        "exploration",
        "expert_floor_rounds",
        "value_penalty",
        "docking_seed",
        "round_lock_policy",
    ):
        assert candidate[field] == reference[field]
    assert candidate["proposal"] == {
        **reference["proposal"],
        "route_complete_region": {
            **reference["proposal"]["route_complete_region"],
            "training_split": "leave-PARP1-out",
        },
    }
    assert [row["cell"] for row in candidate["cells"]] == [
        "parp1_0",
        "parp1_1",
        "parp1_2",
    ]


def test_parp1_route_checkpoint_and_runtime_inputs_are_bound() -> None:
    candidate = unseal(CONTRACT)
    checkpoint_path = (
        ROOT / "diagnostics/t4_integrated_route_fiber_parp1_v1/route_expert_checkpoint.json"
    )
    checkpoint = json.loads(checkpoint_path.read_text())

    assert identity(checkpoint["payload"]) == checkpoint["payload_sha256"]
    assert checkpoint["payload"]["split_audit"]["held_target_absent_from_training"] is True
    assert checkpoint["payload"]["split_audit"]["split"] == "leave_one_target_out"
    serialized = json.dumps(checkpoint["payload"]["expert"], sort_keys=True).lower()
    assert "parp1" not in serialized

    for relative, expected in candidate["runtime_inputs_sha256"].items():
        assert sha256_file(ROOT / relative) == expected
