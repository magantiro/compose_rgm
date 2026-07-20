"""Lock the head-pKa transfer finding: intrinsic head pKa lifts held-head transfer."""

from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DIAG = REPO_ROOT / "diagnostics/pan_lung_head_pka_transfer.json"
CACHE = REPO_ROOT / "artifacts/oracles/head_pka_v1/a549_head_pka.csv"


def test_head_pka_improves_held_head_transfer() -> None:
    diag = json.loads(DIAG.read_text())
    t = diag["held_head_transfer"]
    # adding intrinsic head pKa must improve held-head transfer over R1 alone.
    assert t["R1_plus_head_pka"]["spearman"] > t["R1_only"]["spearman"]
    assert t["improvement_spearman"] >= 0.05
    # the pKa features alone carry most of the head-transfer signal.
    assert t["head_pka_ionization_only"]["spearman"] > 0.10


def test_intrinsic_head_pka_matches_amine_range() -> None:
    diag = json.loads(DIAG.read_text())
    pk = diag["intrinsic_head_pka"]
    # intrinsic amine pKa ~9 (Whitehead & Arral), NOT apparent LNP ~6.4.
    assert 8.0 <= pk["median"] <= 11.0


def test_pka_cache_has_head_values() -> None:
    with CACHE.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) >= 150
    with_pka = [r for r in rows if r["head_pka"] not in ("", "None")]
    assert len(with_pka) >= 150  # nearly all lipids get a head pKa
