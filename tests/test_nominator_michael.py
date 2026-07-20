"""Operational nominator on real Michael-addition lipids: the AD gate admits them.

BEAE's chemistry class (Michael-addition lipids) reuses known ionizable amine
heads, so the deployed head-aware AD gate should ADMIT the overwhelming majority
rather than abstain-on-novel-head. This is the operational bridge: the "novel
head -> abstain" path does not bite BEAE-class lipids; the open question is
ranking quality (leave-library + positive-control), not admission.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DIAG = REPO_ROOT / "diagnostics/nominator_michael_operational.json"

pytestmark = pytest.mark.skipif(not DIAG.exists(), reason="nominator-Michael diagnostic not generated")


def _load() -> dict:
    return json.loads(DIAG.read_text())


def test_michael_lipids_are_admitted_not_abstained() -> None:
    d = _load()
    m = d["per_library"]["michael_addition"]
    # known amine heads -> admitted; the novel-head abstention path does not fire.
    assert m["admission_rate"] >= 0.9
    assert m["abstain_novel_head_rate"] <= 0.05


def test_reference_library_also_admitted() -> None:
    d = _load()
    r = d["per_library"]["reductive_amination"]
    assert r["admission_rate"] >= 0.9
