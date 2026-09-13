import json
import sys
from pathlib import Path

import pytest

from compose_v4.control.edit_program import EditProgram
from compose_v4.experiments.parent_edit_cycles import pmo_oracle
from compose_v4.experiments.pmo_winner_program_curriculum import (
    build_curriculum,
    load_contract,
    locked_queries,
    score_queries,
)

ROOT = Path(__file__).resolve().parents[1]


def test_winner_curriculum_replays_all_locked_programs():
    contract = load_contract(ROOT)
    curriculum = build_curriculum(ROOT, contract)

    assert curriculum["structural_gate"] == {
        "expected_programs": 11,
        "replayed_programs": 11,
        "unique_endpoints": 11,
        "unique_programs": 11,
        "all_within_support": True,
        "passed": True,
    }
    assert len(curriculum["route_audit"]) == 5
    assert sum(row["primitive_steps"] for row in curriculum["route_audit"]) == 215
    assert all(EditProgram.from_payload(row["program"]) for row in curriculum["programs"])
    queries = locked_queries(curriculum, contract)
    assert len(queries) == 15
    assert len({row["smiles"] for row in queries}) == 15
    assert sum(row["role"] == "charged_root" for row in queries) == 4
    assert sum(row["role"] == "complete_program_output" for row in queries) == 11


def test_official_auc_uses_flat_10000_query_tail():
    queries = [{"smiles": f"C{'C' * i}"} for i in range(15)]
    scores = iter([0.1] * 4 + [0.8] * 11)
    result = score_queries(queries, lambda _smiles: next(scores), budget=10000, frequency=100)

    assert result["oracle_calls"] == 15
    assert result["best_score"] == 0.8
    assert result["final_top10"] == pytest.approx(0.8)
    assert result["auc_top10_official_10k"] == pytest.approx(0.7994)


def test_contract_records_winner_informed_role():
    contract = load_contract(ROOT)
    assert contract["information_regime"] == {
        "role": "answer-known winner-informed Perindopril development",
        "candidate_injection": False,
        "program_supervision": "public endpoint plus exact target-known COMPOSE recovery witnesses",
        "retrospective_probe_disclosed": True,
        "held_out_claim": False,
        "general_pmo_claim": False,
        "t4_affected": False,
    }
    raw = json.loads((ROOT / "docs/invirtuogen_pmo_targets.json").read_text())
    assert raw["no_prescreen_targets_partial"]["perindopril_mpo"] == 0.645
    assert raw["targets"]["perindopril_mpo"] == 0.753


def test_pinned_oracle_adapter_installs_legacy_rdkit_compatibility_before_import():
    sys.modules.pop("rdkit.six", None)

    evaluate = pmo_oracle("perindopril_mpo", {"perindopril_mpo": "pinned"})

    assert callable(evaluate)
    assert "rdkit.six" in sys.modules
    import tdc.chem_utils.oracle.oracle as oracle_module

    assert Path(oracle_module.__file__).is_file()
