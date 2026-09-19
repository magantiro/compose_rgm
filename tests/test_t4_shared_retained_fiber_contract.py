import json
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import sha256_file
from compose_v4.experiments.t4_matched_pilot import unseal

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = (
    ROOT / "configs/t4_shared_retained_fiber_parp1_v1.json",
    ROOT / "configs/t4_shared_retained_fiber_jak2_v1.json",
)


def _keys(value) -> set[str]:
    if isinstance(value, dict):
        result = set(map(str, value))
        for row in value.values():
            result.update(_keys(row))
        return result
    if isinstance(value, list):
        result = set()
        for row in value:
            result.update(_keys(row))
        return result
    return set()


def test_shared_scored_contracts_are_sealed_and_runtime_inputs_match():
    contracts = [unseal(path) for path in CONFIGS]
    for contract in contracts:
        assert contract["status"] == "AUTHORIZED_FROZEN_PRIMARY_CONTROLLER"
        assert len(contract["cells"]) == 3
        assert contract["charged_calls_per_cell"] == 49
        assert contract["total_charged_call_ceiling"] == 147
        assert contract["route_scale_floor_rounds"] == 2
        assert contract["proposal"]["route_complete_region"] == {
            "pool_size": 192,
            "realization_limit": 96,
            "beam_width": 48,
            "expansion_width": 48,
            "max_bindings_per_template": 4,
            "maximum_expansions": 4000,
            "scale_balanced": True,
            "training_scope": "one shared task-independent expert fit on all 77 locked routes",
        }
        for relative, expected in contract["runtime_inputs_sha256"].items():
            assert sha256_file(ROOT / relative) == expected

    invariant = (
        "support",
        "charged_calls_per_cell",
        "batch",
        "parents",
        "parent_explore",
        "exploration",
        "expert_floor_rounds",
        "route_scale_floor_rounds",
        "value_penalty",
        "docking_seed",
        "proposal",
    )
    assert all(contracts[0][key] == contracts[1][key] for key in invariant)
    assert {contract["delta"] for contract in contracts} == {0.4, 0.6}


def test_shared_checkpoint_contains_no_runtime_answer_map():
    checkpoint_path = ROOT / "diagnostics/t4_shared_retained_rewrite_v1/checkpoint.json"
    envelope = json.loads(checkpoint_path.read_text())
    assert identity(envelope["payload"]) == envelope["payload_sha256"]
    forbidden = {
        "cell",
        "endpoint",
        "route_id",
        "smiles",
        "source_graph",
        "source_group",
        "target",
        "target_name",
    }
    assert forbidden.isdisjoint(_keys(envelope["payload"]))


def test_shared_apps_enable_scale_balancing_and_route_scale_floor():
    for name in ("parp1", "jak2"):
        source = (ROOT / f"modal_apps/t4_shared_retained_fiber_{name}_app.py").read_text()
        assert 'scale_balanced=route["scale_balanced"]' in source
        assert 'route_scale_floor_rounds=contract["route_scale_floor_rounds"]' in source
        assert 'automatic_retries": 0' in source
