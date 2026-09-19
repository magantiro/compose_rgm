from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools/audit_t4_5ht1b2_exhaustion.py"
SPEC = importlib.util.spec_from_file_location("audit_t4_5ht1b2_exhaustion", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def _row(
    smiles: str,
    *,
    structural_valid: bool,
    similarity: float,
    qed: float,
    sa: float,
) -> dict:
    return {
        "smiles": smiles,
        "structural_valid": structural_valid,
        "similarity": similarity,
        "qed": qed,
        "sa": sa,
        "heavy": 20,
    }


def test_classify_candidates_reports_every_gate_intersection() -> None:
    rows = [
        _row("C", structural_valid=True, similarity=0.7, qed=0.7, sa=3.0),
        _row("CC", structural_valid=True, similarity=0.7, qed=0.5, sa=3.0),
        _row("CCC", structural_valid=False, similarity=0.5, qed=0.7, sa=5.0),
    ]

    result = AUDIT.classify_candidates(rows, delta=0.6)

    assert result["unique_endpoints"] == 3
    assert len(result["gate_intersections"]) == 15
    assert result["gate_intersections"]["structural_valid"] == 2
    assert result["gate_intersections"]["structural_valid&similarity"] == 2
    assert result["gate_intersections"]["structural_valid&similarity&qed&sa"] == 1
    assert result["all_eligible"] == 1


def test_closest_candidates_are_deterministic_and_include_raw_margins() -> None:
    rows = [
        _row("CC", structural_valid=True, similarity=0.59, qed=0.7, sa=3.0),
        _row("C", structural_valid=True, similarity=0.6, qed=0.7, sa=3.0),
    ]

    closest = AUDIT.classify_candidates(rows, delta=0.6)["closest_to_feasible"]

    assert [row["smiles"] for row in closest] == ["C", "CC"]
    assert closest[0]["similarity_margin"] == 0.0
    assert closest[1]["similarity_margin"] == pytest.approx(-0.01)
    assert closest[1]["feasibility_distance"] == pytest.approx(0.01)
