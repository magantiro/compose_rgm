from pathlib import Path

from compose_v4.experiments.pmo_route_fiber_support_audit import (
    build_support_audit,
    contract_envelope,
)

ROOT = Path(__file__).resolve().parents[1]


def test_support_audit_fails_closed_before_asymmetric_sampling():
    result = build_support_audit(ROOT)["payload"]
    assert result["decision"] == "BLOCKED_EXPORT_NOT_PRODUCTION_BINDABLE"
    assert result["new_oracle_calls"] == 0
    assert result["matched_sampler_comparison"] == {
        "executed": False,
        "old_sampler_attempts": 0,
        "route_sampler_attempts": 0,
        "reason": (
            "running only the old sampler would not be an equal-attempt comparison; "
            "the route arm has no executable production policy"
        ),
    }
    assert result["initialization"]["sources"] == 16
    assert result["initialization"]["task_or_score_fields"] == 0


def test_export_has_parameters_but_no_runtime_binding_bridge():
    result = build_support_audit(ROOT)["payload"]
    rows = result["row_support"]
    assert rows["decisions"] == 6143
    assert rows["structural_operand_rows"] == 6143
    assert rows["typed_parameter_rows"] == 4557
    assert rows["parameterless_rows_expected_by_rule"] == 1586
    assert rows["parameter_schema_mismatches"] == {}
    assert rows["runtime_binding_key_hits"] == []
    assert rows["runtime_binding_rows"] == 0
    assert rows["exact_source_graph_rows"] == 0
    assert rows["executable_action_rows"] == 0
    assert rows["complete_region_target_rows"] == 0
    assert result["route_export"]["binding_prototype_targets_emitted"] is False


def test_contract_authorizes_no_scoring_or_modal_launch():
    contract = contract_envelope()["payload"]
    assert contract["oracle_calls_authorized"] == 0
    assert contract["scored_launch_authorized"] is False
    assert contract["attempt_matching"] == (
        "required, but prohibited until both actual samplers exist"
    )
