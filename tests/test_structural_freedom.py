"""Structural-freedom guarantee: every DOF matches R0 ratios AND every family is present.

The corpus must span each ionizable-lipid degree of freedom (number of tails,
tail length, branchedness, unsaturation, linker type, head size, ...) at
realistic ratios (low Jensen-Shannon to the R0 real anchor) -- covering an axis
is not enough, its RATIOS must match reality. Simultaneously, the reaction-family
coverage floor must keep every qualified family visible so the generator can
learn every linker (the diversity mandate), even families whose linker is rare
in R0. These two pull against each other; the two-tier realism-match satisfies both.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT = REPO_ROOT / "artifacts/datasets/compose_lipid_pretraining_v1/structural_freedom_audit.json"
METRICS = REPO_ROOT / "artifacts/datasets/compose_lipid_pretraining_v1/corpus_pilot_v1/pilot_metrics.json"

pytestmark = pytest.mark.skipif(not AUDIT.exists(), reason="structural-freedom audit not generated")


def test_every_degree_of_freedom_matches_r0_ratios() -> None:
    d = json.loads(AUDIT.read_text())
    # no axis is off-ratio: the corpus spans every DOF at realistic ratios.
    assert d["off_ratio_axes"] == []
    for axis, v in d["degrees_of_freedom"].items():
        assert v["jensen_shannon_to_r0"] < 0.12, (axis, v["jensen_shannon_to_r0"])


def test_all_reaction_families_visible_above_floor() -> None:
    m = json.loads(METRICS.read_text())
    balance = m["selection"]["family_balance_selected"]
    # every qualified family survives the realism resample (diversity mandate):
    # >= 6 complementary families, and no family collapsed to a token count.
    assert len(balance) >= 6
    assert all(n >= 30 for n in balance.values()), balance


def test_no_single_family_dominates_to_collapse() -> None:
    m = json.loads(METRICS.read_text())
    balance = m["selection"]["family_balance_selected"]
    total = sum(balance.values())
    # realistic ratios let aza-Michael lead (it dominates real lipidoids), but the
    # corpus is not a single-family corpus: the tail beyond the top family is broad.
    assert max(balance.values()) / total < 0.6
    assert len([n for n in balance.values() if n >= total * 0.02]) >= 6
