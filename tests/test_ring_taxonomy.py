from __future__ import annotations

import pytest

from compose_v4.eval.ring_taxonomy import ring_taxonomy_report


def test_ring_taxonomy_keeps_overlapping_topology_and_chemistry_axes() -> None:
    report = ring_taxonomy_report(
        (
            "CCCC",
            "c1ccccc1",
            "O1CCNCC1",
            "c1ccc2ccccc2c1",
            "C1CCC2(CC1)CCCC2",
            "C1CC2CCC1C2",
            "C1CCCCCCCCCCC1",
        )
    )
    prevalence = report["molecule_prevalence"]

    assert prevalence["acyclic"]["count"] == 1
    assert prevalence["cyclic"]["count"] == 6
    assert prevalence["aromatic"]["count"] == 2
    assert prevalence["heterocycle"]["count"] == 1
    assert prevalence["fused"]["count"] >= 1
    assert prevalence["spiro"]["count"] == 1
    assert prevalence["bridged"]["count"] >= 1
    assert prevalence["macrocycle_ge_12"]["count"] == 1
    assert report["means"]["cycle_rank"] == pytest.approx(9 / 7)


def test_ring_taxonomy_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        ring_taxonomy_report(())
