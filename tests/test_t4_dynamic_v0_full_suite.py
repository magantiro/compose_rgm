import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from compose_v4.control.dynamic_program_synthesis import ONLINE_COMPOSITION_PROBABILITY
from compose_v4.experiments import t4_frozen_program_benchmark as benchmark
from compose_v4.experiments.t4_dynamic_v0_full_suite import (
    configured,
    dynamic_v0_namespace,
)
from compose_v4.experiments.t4_matched_pilot import unseal

ROOT = Path(__file__).resolve().parents[1]


def _tool_module():
    spec = importlib.util.spec_from_file_location(
        "t4_dynamic_v0_full_suite_tool", ROOT / "tools/t4_dynamic_v0_full_suite.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dynamic_v0_configuration_is_the_historical_route_free_recipe():
    contract = unseal(ROOT / "configs/t4_no_complete_routes_diagnostic_v1.json")
    unit = next(row for row in contract["units"] if row["unit_id"] == "braf_1_r0")

    result = configured(contract, unit)

    assert result.current_state_edit_probability == 0.0
    assert result.composition_probability == ONLINE_COMPOSITION_PROBABILITY
    assert result.max_composed_programs == 3
    source = dict(contract["controller"])
    source["channel_probabilities"] = tuple(source["channel_probabilities"])
    for key, value in source.items():
        if key not in {
            "current_state_edit_probability",
            "composition_probability",
            "max_composed_programs",
            "seed",
        }:
            assert asdict(result)[key] == value


def test_dynamic_v0_namespace_is_additive_and_restores_benchmark_globals():
    original = (
        benchmark.KIND,
        benchmark.load_library,
        benchmark.competitive_plateau,
    )

    with dynamic_v0_namespace():
        assert benchmark.load_library(None, None) == ()
        assert benchmark.competitive_plateau([], -10.0, {}) == {
            "stop": False,
            "reason": "disabled_by_frozen_contract",
            "calls": 0,
        }

    assert (
        benchmark.KIND,
        benchmark.load_library,
        benchmark.competitive_plateau,
    ) == original


def test_empty_library_literal_has_no_program_rows(tmp_path):
    path = tmp_path / "empty_library.json"
    path.write_text("[]\n")
    assert json.loads(path.read_text()) == []


def test_preflight_executes_each_wall_clock_bounded_batch_once(monkeypatch, tmp_path):
    tool = _tool_module()
    cells = {f"cell_{index}": {"source_state": {}} for index in range(15)}
    units = [
        {
            "unit_id": f"cell_{index}_r0",
            "cell": f"cell_{index}",
            "target": "target",
            "source_idx": index,
            "original_seed": "C",
            "oracle_protocol": f"protocol-{index}",
        }
        for index in range(15)
    ]
    contract = {"cells": cells, "units": units}
    calls = []

    def fake_batch(source, entries, config, **kwargs):
        del source, entries, config, kwargs
        calls.append(len(calls))
        return {
            "attempts": [{"attempt": len(calls)}],
            "candidates": [],
            "batch_id": f"batch-{len(calls)}",
        }

    import compose_v4.control.dynamic_program_synthesis as synthesis

    monkeypatch.setattr(tool, "ROOT", tmp_path)
    monkeypatch.setattr(tool, "load_contract", lambda root: contract)
    monkeypatch.setattr(tool, "decode_state", lambda payload: object())
    monkeypatch.setattr(
        tool,
        "configured",
        lambda contract, unit: SimpleNamespace(max_primitives=32, max_blocks=8),
    )
    monkeypatch.setattr(tool, "sha256_file", lambda path: "contract-sha")
    monkeypatch.setattr(synthesis, "initial_dynamic_program_batch", fake_batch)

    result = tool.build_preflight(code_revision="revision")

    assert result["passed"] is True
    assert result["schema_version"] == "t4_dynamic_v0_full_suite_preflight_v2"
    assert result["new_oracle_calls"] == 0
    assert len(calls) == 15
