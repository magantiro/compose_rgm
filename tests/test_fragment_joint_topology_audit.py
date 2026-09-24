"""Denominators for the read-only task-topology/quality decomposition."""

import pytest
from audit_fragment_joint_topology_pilot import _summary, audit


def test_summary_keeps_no_output_attempts_in_denominator():
    result = _summary(
        [
            {"smiles": None},
            {
                "smiles": "CC",
                "qed_pass": True,
                "sa_pass": False,
                "joint_pass": False,
                "heavy_atoms": 2,
                "planned_heavy_atoms": 3,
                "qed": 0.7,
                "sa": 4.5,
            },
        ]
    )
    assert result["attempts"] == 2
    assert result["outputs"] == 1
    assert result["qed_pass_outputs"] == 1
    assert result["sa_pass_outputs"] == result["joint_pass_outputs"] == 0
    assert result["mean_heavy_atoms"] == 2
    assert result["mean_planned_heavy_atoms"] == 3


def test_audit_requires_frozen_pilot_manifest(tmp_path):
    with pytest.raises(FileNotFoundError):
        audit(tmp_path)
