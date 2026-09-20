from __future__ import annotations

import json
from pathlib import Path

from compose_v4.control.docking_value import identity
from tools.t4_nodistill_retained_interface_failure_audit import run_audit

ROOT = Path(__file__).resolve().parents[1]
LOCK = (
    ROOT / "diagnostics/t4_nodistill_retained_interface_gate_v3/attempt_1/locks/"
    "jak2_0_d06.json.gz"
)
SEALED = (
    ROOT / "diagnostics/t4_nodistill_retained_interface_gate_v3/attempt_1/"
    "jak2_failure_margin_audit.json"
)


def test_sealed_jak2_margin_audit_is_self_hashed_and_localizes_failure() -> None:
    envelope = json.loads(SEALED.read_text())
    assert envelope["payload_sha256"] == identity(envelope["payload"])
    counts = envelope["payload"]["counts"]
    assert counts["candidate_count"] == 34
    assert counts["all_free_constraints_pass"] == 0
    assert counts["pass_by_constraint"] == {
        "connected": 34,
        "heavy_atom_capacity": 34,
        "qed": 2,
        "sa": 8,
        "similarity": 0,
        "structural_validity": 34,
    }
    nearest = envelope["payload"]["nearest_candidates"][0]
    assert nearest["failed_free_constraints"] == ["similarity"]
    assert nearest["margins"]["similarity"] < 0


def test_margin_audit_is_deterministic(tmp_path: Path) -> None:
    output = tmp_path / "audit.json"
    result = run_audit(repository_root=ROOT, lock_path=LOCK, output_path=output)
    observed = json.loads(output.read_text())
    sealed = json.loads(SEALED.read_text())
    assert result["counts"] == sealed["payload"]["counts"]
    assert (
        observed["payload"]["nearest_candidates"]
        == sealed["payload"]["nearest_candidates"]
    )
    assert (
        observed["payload"]["all_candidate_margins"]
        == sealed["payload"]["all_candidate_margins"]
    )
