"""Lock the component-transfer qualification finding (head vs tail asymmetry)."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DIAG = REPO_ROOT / "diagnostics/pan_lung_component_transfer.json"


def _spearman(block: dict, split: str) -> float:
    return next(s for s in block["splits"] if s["split"] == split)["spearman"]


def test_a549_head_transfers_worse_than_tail() -> None:
    diag = json.loads(DIAG.read_text())
    a = diag["datasets"]["a549_in_vitro"]
    held_head = _spearman(a, "held_head")
    held_tail = _spearman(a, "held_tail1")
    held_lipid = _spearman(a, "held_lipid_baseline")
    # tail transfer is close to the held-lipid baseline; head transfer collapses.
    assert held_tail > 0.4
    assert held_head < 0.3
    assert held_tail - held_head > 0.25
    assert held_lipid > held_head


def test_lumi_high_cardinality_head_axis_transfers_worst() -> None:
    diag = json.loads(DIAG.read_text())
    lumi = diag["datasets"]["lumi_hbe_in_vitro"]
    r1 = _spearman(lumi, "held_R1")  # amine/head-like, cardinality 20
    others = [_spearman(lumi, f"held_{r}") for r in ("R2", "R3", "R4")]
    # R1 (head-like) is the worst-transferring component axis.
    assert r1 == min([r1] + others)
    assert min(others) > r1


def test_oracle_design_consequence_is_recorded() -> None:
    diag = json.loads(DIAG.read_text())
    text = diag["oracle_design_consequence"].lower()
    # the decision-relevant guidance must be present for cross-stream coordination.
    assert "head-aware" in text
    assert "active-learning" in text or "active learning" in text
    assert "abstain" in text
