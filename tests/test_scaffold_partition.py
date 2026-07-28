"""Scaffold-keyed partitioning: the scaffold is the partition unit, assigned BEFORE trajectory generation.

`load_organic_corpus_split` partitions by INDEX, so molecules sharing a Bemis-Murcko scaffold can land in
different partitions. For an edit model that leaks: corrupting a source yields many trajectories around one
scaffold, so a scaffold present in both train and validation lets the model memorize an edit neighbourhood
and score well on "held-out" data it has effectively seen.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from compose_v4.data.scaffold_partition import (  # noqa: E402
    PARTITIONS,
    assign_partitions,
    murcko_scaffold,
    partition_for_scaffold,
    verify_partition_disjointness,
)

_MOLS = [
    "CC(=O)Nc1ccccc1", "CCC(=O)Nc1ccccc1", "COc1ccc(cc1)CCN", "O=C1NC(=O)c2ccccc21",
    "C1CCNCC1C(=O)O", "c1ccc2c(c1)ccc1ccccc12", "CCO", "CCCCO",
    "O=C(O)c1ccc(cc1)S(=O)(=O)N", "Clc1ccccc1C(=O)Nc1ccncc1", "C[N+](C)(C)CCO", "CC(=O)[O-]",
]


def test_partition_is_a_pure_function_of_scaffold():
    for _ in range(3):
        assert partition_for_scaffold("c1ccccc1") == partition_for_scaffold("c1ccccc1")
    assert partition_for_scaffold("c1ccccc1") in PARTITIONS


def test_same_scaffold_always_same_partition():
    """Two DIFFERENT molecules sharing a scaffold must never straddle partitions."""
    a, b = "CC(=O)Nc1ccccc1", "CCC(=O)Nc1ccccc1"
    assert murcko_scaffold(a) == murcko_scaffold(b), "fixture must share a scaffold"
    part, _scaf, _stats = assign_partitions([a, b])
    assert part[a] == part[b]


def test_assignment_is_order_independent():
    forward, _, _ = assign_partitions(_MOLS)
    reverse, _, _ = assign_partitions(list(reversed(_MOLS)))
    assert forward == reverse


def test_zero_source_and_scaffold_overlap():
    part, scaf, _stats = assign_partitions(_MOLS)
    report = verify_partition_disjointness(part, scaf)
    assert report["source_overlap"] == 0
    assert report["scaffold_overlap"] == 0
    assert report["verified"] is True


def test_leak_is_detected_when_injected():
    """The verifier must actually catch a straddling scaffold, not just report zeros."""
    part, scaf, _stats = assign_partitions(_MOLS)
    a = next(iter(part))
    # force a second molecule with the SAME scaffold into a different partition
    other = "___leaked___"
    scaf[other] = scaf[a]
    part[other] = "validation" if part[a] != "validation" else "test"
    with pytest.raises(ValueError, match="scaffolds"):
        verify_partition_disjointness(part, scaf)


def test_acyclic_molecules_do_not_collapse_into_one_bucket():
    """Acyclic molecules have an empty Murcko scaffold; they must not all share one partition key."""
    assert murcko_scaffold("CCO") != murcko_scaffold("CCCCCCCCO")
    assert murcko_scaffold("CCO").startswith("<acyclic:")


def test_acyclic_key_separates_distinct_skeletons_of_equal_size():
    """Heavy-atom count alone was too coarse: measured on held-out data it put 60 acyclic molecules into
    25 keys and 59/1/0 across partitions, leaving held-out sets with no acyclic coverage. The key is now a
    label-reduced (carbonized) Weisfeiler-Lehman skeleton hash, so equal-size but differently-shaped
    molecules separate."""
    linear = murcko_scaffold("CCCCCC")        # straight chain, 6 heavy
    branched = murcko_scaffold("CCC(C)(C)C")  # branched, also 6 heavy
    assert linear is not None and branched is not None
    assert linear != branched, "distinct acyclic skeletons of equal size must get distinct keys"


def test_acyclic_key_is_composition_blind_so_a_skeleton_stays_together():
    """The key is carbonized: same shape, different heteroatoms -> SAME partition. That preserves the leak
    guarantee (a skeleton's edit neighbourhood cannot straddle partitions)."""
    a = murcko_scaffold("CCCCO")
    b = murcko_scaffold("CCCCN")
    assert a == b, "same skeleton must share a key regardless of composition"


def test_unparseable_molecules_are_dropped_not_defaulted_to_train():
    part, _scaf, stats = assign_partitions(["CC(=O)Nc1ccccc1", "not_a_molecule("])
    assert stats["unassignable_sources"] == 1
    assert "not_a_molecule(" not in part


def test_ratios_are_approximately_respected_at_scale():
    import hashlib

    # synthetic distinct scaffolds; only the ratio behaviour is under test
    mols = [f"C1CC{'C' * (i % 7)}1c1ccccc1{'C' * (i // 7)}" for i in range(1)]
    scaffolds = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(4000)]
    counts = {p: 0 for p in PARTITIONS}
    for s in scaffolds:
        counts[partition_for_scaffold(s)] += 1
    assert 0.85 < counts["train"] / len(scaffolds) < 0.95
    assert counts["validation"] > 0 and counts["test"] > 0
    assert mols  # keep the fixture referenced
