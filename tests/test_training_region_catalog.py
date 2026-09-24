from __future__ import annotations

import hashlib

from rdkit import Chem

from compose_v4.benchmark.training_region_catalog import build_region_catalog, observed_regions
from compose_v4.data.scaffold_partition import murcko_scaffold, partition_for_scaffold


def test_ring_heterocycle_and_two_boundary_content_are_not_removed():
    molecule = Chem.MolFromSmiles("CCOC(=O)c1ccc(NC(=O)C2CCNCC2)cc1")
    regions = list(observed_regions(molecule))
    # RingInfo is owned by its Mol; keep the owners alive across the C++ call.
    parsed = [Chem.MolFromSmiles(text) for _, text in regions]
    assert any(mol.GetRingInfo().NumRings() > 0 for mol in parsed)
    assert any(len(contexts) == 2 for contexts, _ in regions)
    assert any("N" in text and "1" in text for _, text in regions)


def test_scaffold_split_precedes_every_extraction(tmp_path):
    molecules = [
        "CCOC(=O)c1ccccc1",
        "CCOc1ccncc1",
        "CCOC(=O)C1CCCCC1",
        "CCNC(=O)c1ccc2ccccc2c1",
        "CCOC(=O)C1CC1",
        "CCOc1nccnc1",
        "CCOC(=O)C1CCCCCCCCCCCCCC1",
    ]
    train = [s for s in molecules if partition_for_scaffold(murcko_scaffold(s)) == "train"]
    held = [s for s in molecules if partition_for_scaffold(murcko_scaffold(s)) != "train"]
    assert train and held, "fixture must exercise both source partitions"
    source = tmp_path / "source.smi"
    ordered = [held[0], train[0], train[-1]]
    source.write_text("\n".join(ordered) + "\n")
    catalog = build_region_catalog(
        source,
        expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        excluded_canonical=frozenset({Chem.MolToSmiles(Chem.MolFromSmiles(train[0]))}),
        training_molecules=1,
    )
    assert catalog["accepted_source_rows"] == [3]
    assert catalog["entries"]
    assert all(set(e["source_rows"]) == {3} for e in catalog["entries"])
    assert catalog["exclusions"]["benchmark_reference_excluded"] == 1
