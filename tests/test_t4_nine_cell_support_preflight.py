from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from compose_v4.experiments.t4_nine_cell_support_preflight import (
    EXPERTS,
    aggregate_cells,
    scientific_projection,
    validate_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "t4_nine_cell_support_preflight_v1.json"


def test_sealed_contract_has_exact_production_census_and_checkpoint() -> None:
    contract, checkpoint = validate_contract(ROOT, CONTRACT)

    assert len(contract["cells"]) == 9
    assert contract["experts"] == list(EXPERTS)
    assert contract["deferred_joint_planning"]["enabled"] is False
    assert checkpoint["training_routes"] == 77
    assert checkpoint["training_regions"] == 147
    assert len(checkpoint["expert"]["templates"]) == 137


def test_contract_has_no_teacher_or_scored_runtime_input() -> None:
    payload = json.loads(CONTRACT.read_text())["payload"]
    serialized = json.dumps(payload, sort_keys=True).lower()

    assert "teacher endpoint" in serialized  # It occurs only in the forbidden list.
    assert "teacher_endpoint" not in serialized
    assert payload["costs"] == {
        "oracle_calls": 0,
        "docking_calls": 0,
        "modal_launches": 0,
    }


def test_scientific_projection_removes_timing_recursively() -> None:
    left = {"wall_seconds": 2.0, "cells": [{"elapsed_seconds": 1.0, "count": 4}]}
    right = {"wall_seconds": 7.0, "cells": [{"elapsed_seconds": 8.0, "count": 4}]}

    assert scientific_projection(left) == scientific_projection(right)


def test_aggregate_fails_closed_on_one_zero_support_cell() -> None:
    contract, _ = validate_contract(ROOT, CONTRACT)
    prototype = {
        "cell_id": "placeholder",
        "gate": {"nonzero_eligible_support": True},
        "experts": [
            {
                "expert": expert,
                "attempted": 1,
                "generated": 1,
                "exact_unique": 1,
                "valid_unique": 1,
                "eligible_unique": 1,
            }
            for expert in EXPERTS
        ],
    }
    cells = []
    for index, spec in enumerate(contract["cells"]):
        cell = deepcopy(prototype)
        cell["cell_id"] = spec["cell_id"]
        if index == 0:
            cell["gate"]["nonzero_eligible_support"] = False
        cells.append(cell)

    result = aggregate_cells(
        root=ROOT, contract=contract, cells=cells, wall_seconds=1.0
    )

    assert result["gate"]["passed"] is False
    assert result["gate"]["zero_eligible_cells"] == [contract["cells"][0]["cell_id"]]
