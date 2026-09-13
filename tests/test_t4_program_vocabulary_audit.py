import copy
import json
from pathlib import Path

import pytest

from compose_v4.control.edit_program import EditProgram
from compose_v4.experiments.t4_matched_pilot import unseal
from tools.t4_program_vocabulary_audit import (
    inspect_program_payload,
    origin_relation,
    source_group_map,
)

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_source_groups_map_exactly_to_all_fifteen_cells():
    contract = unseal(ROOT / "configs/t4_frozen_program_benchmark_v2.json")
    registry = json.loads((ROOT / "docs/GENMOL_T4_SEEDS.json").read_text())
    library = json.loads(
        (
            ROOT
            / "diagnostics/t4_shared_program_controller/attempt_2/shared_library.json"
        ).read_text()
    )

    mapped = source_group_map(contract, registry)
    observed = {group for row in library for group in row["source_groups"]}

    assert len(mapped) == 15
    assert observed == set(mapped)
    assert {row["cell"] for row in mapped.values()} == set(contract["cells"])


def test_program_inspection_reports_typed_handles_and_literal_labels():
    library = json.loads(
        (
            ROOT
            / "diagnostics/t4_shared_program_controller/attempt_2/shared_library.json"
        ).read_text()
    )
    payload = copy.deepcopy(library[0]["program"])
    payload["blocks"][0]["label"] = "jak2"

    result = inspect_program_payload(
        payload,
        targets={"jak2"},
        seeds={"seed-smiles"},
        winner_endpoints={"winner-smiles"},
    )

    assert result["literal_target_values"] == ["jak2"]
    assert result["literal_seed_smiles"] == []
    assert result["literal_winner_endpoint_smiles"] == []
    assert result["typed_atom_operands"] > 0
    assert result["raw_integer_atom_operands"] == 0
    assert result["malformed_atom_operands"] == 0


def test_edit_program_rejects_a_raw_source_slot_operand():
    library = json.loads(
        (
            ROOT
            / "diagnostics/t4_shared_program_controller/attempt_2/shared_library.json"
        ).read_text()
    )
    payload = copy.deepcopy(library[0]["program"])
    record = json.loads(payload["marks"][0])
    if record["executor_rule"] == "atom_insert":
        record["payload"]["slot"] = 17
    elif record["executor_rule"] in ("atom_delete", "atom_restate_semantic"):
        record["payload"]["v"] = 17
    else:
        pytest.skip("fixture's first rule has no single slot operand")
    payload["marks"][0] = json.dumps(record, sort_keys=True, separators=(",", ":"))

    with pytest.raises(
        ValueError, match="typed atom reference|birth handles must be sequential"
    ):
        EditProgram.from_payload(payload)


@pytest.mark.parametrize(
    ("recipient", "origins", "expected"),
    [
        (
            "braf_1",
            [{"cell": "braf_1", "target": "braf"}],
            "includes_same_cell",
        ),
        (
            "braf_1",
            [{"cell": "braf_0", "target": "braf"}],
            "includes_same_target_other_seed",
        ),
        (
            "braf_1",
            [{"cell": "jak2_0", "target": "jak2"}],
            "cross_target_only",
        ),
    ],
)
def test_origin_relation(recipient, origins, expected):
    assert origin_relation(recipient, origins) == expected
