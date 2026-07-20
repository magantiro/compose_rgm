"""Invariants for the canonical row-level pan-lung manifest.

Reads the committed artifacts (no external raw files) and enforces typed-head
separation, expected counts, the full-structure/component boundary, and absence
of cross-source structure leakage.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ART = REPO_ROOT / "artifacts/oracles/pan_lung_canonical_v1"
CSV = ART / "canonical_rows.csv"
MANIFEST = ART / "manifest.json"

EXPECTED_HEAD_COUNTS = {
    "airway_a549_in_vitro_expression": 1801,
    "airway_hbe_in_vitro_expression": 1920,
    "airway_hbec_ali_in_vitro_expression": 29,
    "local_intratracheal_functional_expression": 49,
    "systemic_iv_barcoded_lung_uptake": 96,
    "systemic_iv_lung_expression": 444,
    "systemic_iv_lung_selectivity": 444,
}


def _rows() -> list[dict]:
    with CSV.open() as handle:
        return list(csv.DictReader(handle))


def test_manifest_hash_matches_committed_csv() -> None:
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["canonical_rows_csv"]["sha256"] == hashlib.sha256(CSV.read_bytes()).hexdigest()


def test_typed_head_counts_are_exact() -> None:
    rows = _rows()
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["typed_head"]] = counts.get(row["typed_head"], 0) + 1
    assert counts == EXPECTED_HEAD_COUNTS
    assert len(rows) == sum(EXPECTED_HEAD_COUNTS.values()) == 4783


def test_readout_types_stay_semantically_distinct() -> None:
    head_to_readout = {r["typed_head"]: r["readout_type"] for r in _rows()}
    # in-vitro transfection, functional in-vivo expression, biodistribution, and
    # selectivity must never collapse to a single readout.
    assert head_to_readout["airway_a549_in_vitro_expression"] == "in_vitro_transfection"
    assert head_to_readout["local_intratracheal_functional_expression"] == "functional_in_vivo_expression"
    assert head_to_readout["systemic_iv_barcoded_lung_uptake"] == "biodistribution_barcode_uptake"
    assert head_to_readout["systemic_iv_lung_selectivity"] == "selectivity"
    assert len(set(head_to_readout.values())) >= 4


def test_lut_is_component_only_and_others_have_structure() -> None:
    for row in _rows():
        if row["source_id"] == "lut_444_2026":
            assert row["has_full_structure"] == "False"
            assert row["raw_smiles"] == ""
        else:
            assert row["has_full_structure"] == "True"
            assert row["raw_smiles"] != ""


def test_no_cross_source_structure_leakage_lnpdb_lumi() -> None:
    manifest = json.loads(MANIFEST.read_text())
    overlaps = manifest["overlap"]["cross_source_exact_structure_overlap"]
    assert overlaps["lnpdb_2026__lumi_lab_4cr1920"] == 0


def test_every_row_has_study_and_chemical_group() -> None:
    for row in _rows():
        assert row["study_group"], row
        assert row["chemical_group"], row
