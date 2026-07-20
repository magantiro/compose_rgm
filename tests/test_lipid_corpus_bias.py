from __future__ import annotations

import csv
import hashlib
import json

import numpy as np
import pytest
from rdkit import Chem

from compose_v4.lipids.corpus_bias import (
    choose_equal_priority_minimax,
    expected_weighted_unique_coverage,
    heteroatom_connector_core,
    hill_effective_numbers,
    jensen_shannon,
    lipid_topology_features,
)
from compose_v4.lipids.layered_sampler import (
    LayerAwareStructureSampler,
    LayeredSamplerError,
)


def molecule(smiles: str) -> Chem.Mol:
    parsed = Chem.MolFromSmiles(smiles)
    assert parsed is not None
    return parsed


def test_lipid_topology_proxies_are_explicit_and_chemically_interpretable() -> None:
    saturated_ester = lipid_topology_features(molecule("CCCCCCCCOC(=O)CCCCCCCCN"))
    branched_unsaturated = lipid_topology_features(
        molecule("CC(C)CCCC/C=C/CCCCNCC(=O)OCC")
    )
    assert saturated_ester["motif_ester_count"] == 1
    assert "ester" in str(saturated_ester["cleavable_motif_proxy"])
    assert saturated_ester["aromatic_bin"] == "nonaromatic"
    assert branched_unsaturated["branching_bin"] != "0"
    assert branched_unsaturated["unsaturation_bin"] == "present"


def test_heteroatom_connector_removes_terminal_carbon_tails() -> None:
    first = molecule("CCCCCCCCOC(=O)CNCC")
    second = molecule("CCCCOC(=O)CNCCCCCC")
    first_core = heteroatom_connector_core(first)
    second_core = heteroatom_connector_core(second)
    assert "O" in first_core and "N" in first_core
    assert len(first_core) < len(Chem.MolToSmiles(first))
    assert len(second_core) < len(Chem.MolToSmiles(second))


def test_diversity_and_tradeoff_metrics_have_expected_boundaries() -> None:
    assert jensen_shannon(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == 0.0
    assert np.isclose(
        jensen_shannon(np.array([1.0, 0.0]), np.array([0.0, 1.0])), 1.0
    )
    effective = hill_effective_numbers(["a", "a", "b", "b"])
    assert np.isclose(effective["shannon_q1"], 2.0)
    assert np.isclose(effective["simpson_q2"], 2.0)
    probabilities = np.array([0.5, 0.5])
    novelty = np.array([1.0, 1.0])
    assert expected_weighted_unique_coverage(probabilities, novelty, 0.0, 10) == 0.0
    assert expected_weighted_unique_coverage(probabilities, novelty, 0.5, 10) > 0.0
    selected = choose_equal_priority_minimax(
        [0.0, 0.1, 0.2],
        [0.0, 0.4, 1.0],
        [0.0, 0.7, 1.0],
    )
    assert selected["layer_probability"] == 0.1


def test_layer_sampler_verifies_hash_preserves_layers_and_applies_split_exclusions(
    tmp_path,
) -> None:
    rows_path = tmp_path / "rows.csv"
    fields = [
        "layer_id",
        "structure_sha256",
        "canonical_isomeric_smiles",
        "split_group_tokens_json",
        "within_layer_probability",
    ]
    rows = [
        {
            "layer_id": "observed",
            "structure_sha256": "a",
            "canonical_isomeric_smiles": "CCN",
            "split_group_tokens_json": json.dumps(["study=A"]),
            "within_layer_probability": "0.75",
        },
        {
            "layer_id": "observed",
            "structure_sha256": "b",
            "canonical_isomeric_smiles": "CCCN",
            "split_group_tokens_json": json.dumps(["study=B"]),
            "within_layer_probability": "0.25",
        },
        {
            "layer_id": "auxiliary",
            "structure_sha256": "c",
            "canonical_isomeric_smiles": "CCCCN",
            "split_group_tokens_json": json.dumps(["reaction=Ugi"]),
            "within_layer_probability": "1.0",
        },
    ]
    with rows_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    artifact_hash = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "layers": {
                    "observed": {"layer_probability": 0.8},
                    "auxiliary": {"layer_probability": 0.2},
                },
                "sampler_rows": {
                    "path": str(rows_path),
                    "sha256": artifact_hash,
                    "row_count": 3,
                },
            }
        )
    )
    sampler = LayerAwareStructureSampler.load(manifest_path)
    selected = sampler.sample(20_000, seed=7)
    auxiliary_fraction = sum(row.layer_id == "auxiliary" for row in selected) / len(
        selected
    )
    assert abs(auxiliary_fraction - 0.2) < 0.02
    held_out = sampler.sample(2_000, seed=8, excluded_split_group_tokens={"study=A"})
    assert all(row.structure_sha256 != "a" for row in held_out)
    with pytest.raises(LayeredSamplerError, match="removed every row"):
        sampler.sample(10, seed=9, excluded_split_group_tokens={"reaction=Ugi"})

    rows_path.write_text(rows_path.read_text() + "\n")
    with pytest.raises(LayeredSamplerError, match="hash mismatch"):
        LayerAwareStructureSampler.load(manifest_path)
