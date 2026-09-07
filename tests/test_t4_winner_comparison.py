"""Endpoint diagnostics must not conflate ring count, systems, and reachability."""

import runpy
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


@pytest.fixture(scope="module")
def describe():
    path = Path(__file__).resolve().parents[1] / "tools/t4_compare_winners.py"
    return runpy.run_path(str(path))["describe"]


@pytest.mark.parametrize(
    "smiles,cycles,systems",
    [("CCCC", 0, 0), ("c1ccc2ccccc2c1", 2, 1), ("c1ccccc1-c1ccccc1", 2, 2)],
)
def test_endpoint_ring_axes_remain_distinct(describe, smiles, cycles, systems):
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    seed_fp = fp.GetFingerprint(Chem.MolFromSmiles(smiles))
    result = describe(smiles, seed_fp, 0.4)
    assert result["cycle_rank"] == cycles
    assert result["ring_system_count"] == systems
    assert result["similarity_to_seed"] == 1.0
    assert "reachable" not in result
    assert "executor_path" not in result


def test_endpoint_analysis_rejects_invalid_smiles(describe):
    with pytest.raises(ValueError, match="invalid diagnostic molecule"):
        describe("not-a-molecule", None, 0.4)
