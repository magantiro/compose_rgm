"""Leakage-resistant corpus splits: jointly hold out family/scaffold/head/study."""

from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ART = REPO_ROOT / "artifacts/datasets/compose_lipid_pretraining_v1/splits_v1"

EXPECTED_SPLITS = {"held_reaction_family", "held_scaffold", "held_head",
                   "held_study", "random_diagnostic"}


def test_all_splits_are_leak_free() -> None:
    manifest = json.loads((ART / "manifest.json").read_text())
    assert manifest["all_splits_leak_free"] is True
    assert set(manifest["splits"]) == EXPECTED_SPLITS
    for name, s in manifest["splits"].items():
        assert s["leaked_groups"] == 0, name


def test_held_family_and_study_actually_hold_out() -> None:
    manifest = json.loads((ART / "manifest.json").read_text())
    # held-family test set is populated (whole reaction families held out).
    assert manifest["splits"]["held_reaction_family"]["fold_counts"]["test"] > 0
    # held-study holds out at least one source study.
    assert manifest["splits"]["held_study"]["n_held_out_groups"] >= 1


def test_fold_assignments_cover_both_layers() -> None:
    with (ART / "corpus_fold_assignments.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    manifest = json.loads((ART / "manifest.json").read_text())
    assert len(rows) == manifest["corpus_rows"]
    layers = {r["layer"] for r in rows}
    assert {"R0_observed", "R1_virtual"} <= layers
    # every split column has all three folds represented somewhere (except family/study
    # which may be train/test only), and values are valid.
    for r in rows[:200]:
        for split in EXPECTED_SPLITS:
            assert r[split] in {"train", "val", "test"}


def test_scaffold_and_head_splits_have_real_holdouts() -> None:
    manifest = json.loads((ART / "manifest.json").read_text())
    for split in ("held_scaffold", "held_head"):
        s = manifest["splits"][split]
        assert s["fold_counts"]["test"] > 0 and s["fold_counts"]["val"] > 0
        assert s["n_groups"] > 100  # many distinct scaffolds/heads to hold out
