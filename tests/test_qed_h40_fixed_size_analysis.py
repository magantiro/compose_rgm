"""Paired H40 analysis must fail closed on missing or inconsistent source records."""

import json

import pytest
from rdkit import Chem
from rdkit.Chem import QED, rdFingerprintGenerator

from scripts.qed_h40_fixed_size_analysis import (
    _validate_record,
    analyze,
    paired_counts,
)


def test_paired_categories_keep_source_as_unit():
    assert paired_counts(
        [True, True, False, False], [True, False, True, False]
    ) == {"both": 1, "full_only": 1, "fixed_only": 1, "neither": 1}
    with pytest.raises(ValueError, match="different lengths"):
        paired_counts([True], [])


def test_missing_sources_produce_no_paired_estimate(tmp_path):
    sources = tmp_path / "sources.txt"
    sources.write_text("\n".join(f"source-{index}" for index in range(800)))
    result = analyze(
        source_path=sources, full_dir=tmp_path / "full", fixed_dir=tmp_path / "fixed"
    )
    assert result["status"] == "INCOMPLETE"
    assert result["missing"] == {"full": list(range(800)), "fixed": list(range(800))}
    assert "summary" not in result


def test_fixed_size_record_rejects_changed_endpoint_atom_count(tmp_path):
    source = "CC"
    returned = "CCC"
    record = {
        "index": 0,
        "source": source,
        "horizon": 40,
        "budget_max": 24,
        "head_dir": "hphi_v2",
        "size_fixed": True,
        "allowed_families": [
            "atom_restate", "bond_reorder", "bond_reroute", "cycle_insert",
            "cycle_attach", "ring_system_restate",
        ],
        "empty_restricted_fiber_rule": "kill_particle_no_retry",
        "arms": {"restart": {
            "success": False,
            "candidates": [{
                "k": index,
                "returned": returned,
                "terminal_qed": QED.qed(Chem.MolFromSmiles(returned)),
                "terminal_sim": 0.0,
                "success": False,
                "extinct": False,
            } for index in range(8)],
        }},
    }
    path = tmp_path / "000_H40_hphi_v2_fixed-size_k0-8.json"
    path.write_text(json.dumps(record))
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    with pytest.raises(ValueError, match="changed atom count"):
        _validate_record(
            path, index=0, source=source, fixed=True,
            fingerprint_generator=generator,
        )
