"""Nominator qualification: admits real lipids, abstains on novel heads."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DIAG = REPO_ROOT / "diagnostics/pan_lung_nominator_qualification.json"


def test_most_real_lipids_are_admitted() -> None:
    d = json.loads(DIAG.read_text())
    assert d["admission_rate"] >= 0.9  # real in-domain lipids rank, not abstain


def test_scores_track_potency_in_sample() -> None:
    d = json.loads(DIAG.read_text())
    # directional wiring check: score correlates with measured potency (in-sample).
    assert d["in_sample_ranking_spearman_score_vs_potency"] > 0.5


def test_all_novel_and_non_ionizable_heads_abstain() -> None:
    d = json.loads(DIAG.read_text())
    assert d["all_novel_heads_abstain"] is True
    for name, v in d["novel_head_candidates"].items():
        assert v["abstained"] is True, name
