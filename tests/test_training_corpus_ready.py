"""Training-readiness contract for the generator handoff.

The COMPOSE-Lipid generator trains on R0 (real anchor) + R1 (reaction-grounded
reachable support). This locks the materialized training corpus + manifest: the
full R1 set is written with provenance and realism sampling weights, every
qualified family is present, the kernel profile the model must support is
declared, and the realism weighting matches R0 on the fit axes.
"""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPO_ROOT / "artifacts/datasets/compose_lipid_pretraining_v1"
MANIFEST = DATASET / "training_corpus_manifest_v1.json"
R1_CSV = DATASET / "r1_reaction_grounded_corpus_v1.csv"

pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="training corpus not materialized")

EXPECTED_COLUMNS = ["canonical_smiles", "reaction_family", "reactant_ids", "reactant_roles",
                    "size_bin", "charge_bin", "ring_bin", "n_tails_bin", "tail_length_bin",
                    "linker_type", "realism_weight"]


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def test_manifest_declares_both_layers() -> None:
    m = _manifest()
    assert m["format"] == "compose_lipid_training_corpus_v1"
    r0 = m["layers"]["r0_observed_real"]
    r1 = m["layers"]["r1_reaction_grounded"]
    assert r0["row_count"] > 15000 and r0["never_inherits_labels"] is True
    assert r1["row_count"] > 200000  # the full reachable support, not the 3k pilot
    for layer in (r0, r1):
        assert len(layer["sha256"]) == 64 and layer["path"]


def test_all_qualified_families_present_in_r1() -> None:
    m = _manifest()
    fams = m["layers"]["r1_reaction_grounded"]["family_counts"]
    assert len(fams) >= 6
    assert all(n > 0 for n in fams.values()), fams


def test_kernel_profile_declares_lipid_elements() -> None:
    m = _manifest()
    els = set(m["kernel_readiness_profile"]["elements_present"])
    # lipid kernel must span beyond C/N/O/F: S (disulfide/thioether) + P (iPhos)
    assert {"C", "N", "O", "S", "P"}.issubset(els)
    assert m["kernel_readiness_profile"]["heavy_atoms"]["max"] > 64


def test_realism_weighting_matches_r0_on_fit_axes() -> None:
    m = _manifest()
    rw = m["realism_weighting"]
    assert rw["axes_fit_at_scale"] == ["linker_type", "n_tails", "tail_length", "size"]
    assert all(js < 0.1 for js in rw["weighted_js_to_r0"].values()), rw["weighted_js_to_r0"]


def test_r1_csv_header_and_weights() -> None:
    with R1_CSV.open() as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == EXPECTED_COLUMNS
        rows = list(itertools.islice(reader, 200))
    assert rows and all(float(r["realism_weight"]) >= 0 for r in rows)
    assert all(r["reaction_family"] and r["canonical_smiles"] for r in rows)
