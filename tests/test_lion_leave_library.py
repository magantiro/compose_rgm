"""LiON leave-library-out: cross-chemistry rank transfer is weak (justifies AD gate).

The LiON A549 lung screen carries per-library annotations, including a
Michael-addition-branched library -- the novel-linker chemistry class. Holding
out an entire library and predicting it with an oracle trained on the OTHER
chemistries tests whether the oracle can rank a chemistry it never saw. Weak
transfer (especially to the Michael-addition library) is the real-data evidence
for the head-aware applicability-domain gate + active-learning calibration.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DIAG = REPO_ROOT / "diagnostics/lion_leave_library_transfer.json"

pytestmark = pytest.mark.skipif(not DIAG.exists(), reason="LiON diagnostic not generated")


def _load() -> dict:
    return json.loads(DIAG.read_text())


def test_genuine_leave_library_split() -> None:
    d = _load()
    # at least two distinct chemistry libraries held out => a real leave-library-out
    assert d["n_libraries"] >= 2
    for lib, r in d["per_library"].items():
        assert r["train_lipids"] >= 30 and r["held_out_lipids"] >= 10, lib


def test_michael_library_present_and_weakly_transferred() -> None:
    d = _load()
    michael = d["michael_addition_library"]
    assert michael is not None, "Michael-addition library not found in LiON A549 slice"
    # near-zero global rank transfer to the novel-linker chemistry class: an
    # oracle trained on other chemistries cannot rank Michael-addition lipids.
    assert d["michael_transfer_spearman"] < 0.25


def test_cross_chemistry_transfer_is_limited() -> None:
    d = _load()
    # no held-out chemistry library is well-ranked from the others (all Spearman < 0.4):
    # supports domain-bounded nomination over blind extrapolation across chemistries.
    assert all(r["spearman"] < 0.4 for r in d["per_library"].values())


def test_positive_control_michael_is_rankable_when_seen() -> None:
    d = _load()
    # deploy config: with Michael-addition data represented in training, the oracle
    # ranks Michael-addition lipids well -- and far better than the blind leave-out.
    pc = d["positive_control_michael_included"]
    assert pc["michael_subset_spearman"] > 0.4
    assert pc["michael_subset_spearman"] > d["michael_transfer_spearman"] + 0.3
