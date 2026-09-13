"""Read-only checks on the optional, documented local throughput evidence."""

import json
from pathlib import Path

import pytest

from compose_v4.control.docking_value import identity
from tools.fast_program_search import compare_profiles


def test_retained_speed_assay_preserves_every_exact_candidate_and_attempt(tmp_path):
    root = Path("diagnostics/fast_program_search")
    if not (root / "fixed_1/result.json").exists():
        pytest.skip("documented local fast-program profile absent")
    output = tmp_path / "comparison.json"
    compare_profiles(root / "fixed_1", root / "bindings_1", output)
    result = json.loads(output.read_text())
    assert (
        identity({k: v for k, v in result.items() if k != "result_sha256"})
        == result["result_sha256"]
    )
    assert len(result["units"]) == 4
    assert all(r["exact_candidate_and_attempt_equivalence"] for r in result["units"])
    assert sum(r["after_new_candidates"] for r in result["units"]) == 48
    assert result["new_oracle_calls"] == 0
