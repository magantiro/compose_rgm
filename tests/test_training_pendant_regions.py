"""Split-first smaller-side extraction and source balancing."""

import pytest
from rdkit import Chem

from compose_v4.benchmark.training_attachment_fragments import physical_sha256
from compose_v4.benchmark.training_pendant_regions import (
    build_pendant_catalog,
    observed_pendant_sides,
)


def test_smaller_side_uses_all_acyclic_single_bonds_and_retains_one_atom_ends():
    pieces = observed_pendant_sides(Chem.MolFromSmiles("CCO"))
    assert sorted(row["rooted_smiles"] for row in pieces) == ["[1*]C", "[1*]O"]
    assert all(row["heavy_atoms"] == 1 and row["ring_count"] == 0 for row in pieces)
    ring_pieces = observed_pendant_sides(Chem.MolFromSmiles("Cc1ccccc1-c1ccccc1"))
    assert any(row["ring_count"] == 1 for row in ring_pieces)
    assert all(row["heavy_atoms"] <= 6 for row in ring_pieces)


def test_catalog_weights_each_source_molecule_once_and_checks_source_hash(tmp_path):
    source = tmp_path / "train.smiles"
    source.write_text("CCO\nCc1ccccc1-c1ccccc1\n")
    base = {
        "schema": "split_first_training_region_catalog_v1",
        "source_sha256": physical_sha256(source),
        "training_molecules": 2,
        "accepted_source_rows": [1, 2],
        "split": {"partition": "train"},
    }
    result = build_pendant_catalog(base, source)
    assert result["schema"] == "split_first_training_pendant_catalog_v1"
    assert result["census"]["molecules_with_pendant"] == 2
    assert result["census"]["one_atom_observations"] >= 2
    assert result["census"]["ring_containing_observations"] >= 1
    assert sum(entry["source_balanced_weight"] for entry in result["entries"]) == pytest.approx(2)
    assert all(set(entry["source_rows"]) <= {1, 2} for entry in result["entries"])
    source.write_text("CCN\nCc1ccccc1-c1ccccc1\n")
    with pytest.raises(ValueError, match="hash changed"):
        build_pendant_catalog(base, source)
