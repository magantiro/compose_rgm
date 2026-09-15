import json
from dataclasses import asdict
from pathlib import Path

from compose_v4.control.dynamic_program_synthesis import ONLINE_COMPOSITION_PROBABILITY
from compose_v4.experiments import t4_frozen_program_benchmark as benchmark
from compose_v4.experiments.t4_dynamic_v0_full_suite import (
    configured,
    dynamic_v0_namespace,
)
from compose_v4.experiments.t4_matched_pilot import unseal

ROOT = Path(__file__).resolve().parents[1]


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
