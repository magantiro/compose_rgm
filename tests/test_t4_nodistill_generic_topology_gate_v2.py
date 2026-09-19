from __future__ import annotations

import inspect
import json
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.control.generic_complete_program_composer_v2 import (
    TOPOLOGY_MACRO_PLANS,
    propose_generic_topology_macro_programs,
)
from compose_v4.experiments import t4_nodistill_generic_topology_gate_v2 as gate

ROOT = Path(__file__).resolve().parents[1]


def test_v2_contract_is_self_hashed_and_zero_oracle() -> None:
    envelope = json.loads((ROOT / gate.CONTRACT_RELATIVE_PATH).read_text())
    assert envelope["payload_sha256"] == identity(envelope["payload"])
    contract, contract_identity = gate.load_contract(ROOT)
    assert contract_identity == envelope["payload_sha256"]
    assert contract["costs"] == {
        "oracle_calls": 0,
        "docking_calls": 0,
        "modal_launches": 0,
        "gpu_seconds": 0,
    }


def test_v2_macro_plans_predeclare_repeated_topology_support() -> None:
    cycle_support = {
        cycle
        for plan in TOPOLOGY_MACRO_PLANS
        for cycle in range(plan.target_delta_cycle_rank[0], plan.target_delta_cycle_rank[1] + 1)
    }
    assert {2, 3} <= cycle_support
    assert all(len(plan.families) >= 2 for plan in TOPOLOGY_MACRO_PLANS)
    assert any("segment_replace" in plan.families for plan in TOPOLOGY_MACRO_PLANS)
    assert any("substituent_delete" in plan.families for plan in TOPOLOGY_MACRO_PLANS)
    assert sum(plan.candidate_quota for plan in TOPOLOGY_MACRO_PLANS) == 24


def test_v2_runtime_api_has_no_task_route_or_teacher_input() -> None:
    signature = inspect.signature(propose_generic_topology_macro_programs)
    assert tuple(signature.parameters) == (
        "source",
        "seed",
        "attempts_per_plan",
        "maximum_primitives",
        "maximum_blocks",
    )
    for forbidden in (
        "target_name",
        "cell_key",
        "route",
        "teacher",
        "endpoint",
        "score",
    ):
        assert forbidden not in signature.parameters


def test_v2_teacher_loading_is_post_lock_only() -> None:
    assert "_teacher_descriptors" not in inspect.getsource(gate.lock_cell)
    assert "_teacher_descriptors" in inspect.getsource(gate.evaluate_locks)


def test_v2_sealed_result_and_unscored_panel_are_self_hashed() -> None:
    root = ROOT / gate.ARTIFACT_ROOT_RELATIVE_PATH
    result = json.loads((root / "result.json").read_text())
    assert result["payload_sha256"] == identity(result["payload"])
    assert result["payload"]["decision"] == "PASS_TOPOLOGY_SUPPORT_GATE"
    assert result["payload"]["gates"]["passed"] is True
    panel = json.loads((root / "scored_candidate_proposal.json").read_text())
    assert panel["payload_sha256"] == identity(panel["payload"])
    assert panel["payload"]["candidate_count"] == 4
    assert panel["payload"]["status"] == (
        "PROPOSED_REQUIRES_SEPARATE_SCORED_CONTRACT_AND_AUTHORIZATION"
    )
    assert panel["payload"]["costs_spent"]["docking_calls"] == 0
