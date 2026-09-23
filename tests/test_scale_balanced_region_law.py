"""The scale-balanced region law: equal mass per occupied class, and no filtering."""
from __future__ import annotations

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.scale_balanced_region_law import (
    DEFAULT_EDGES,
    ScaleBalancedRegionLaw,
)

SOURCE = "CCC(NC(=O)C1CCOCC1)c1cn(C)nc1C"


def _graph(smiles=SOURCE):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), 48)


def test_class_boundaries_are_half_open():
    law = ScaleBalancedRegionLaw()
    assert [law.scale_class(s) for s in (1, 4, 5, 11, 12, 25)] == [0, 0, 1, 1, 2, 2]


def test_edges_must_be_increasing_and_positive():
    with pytest.raises(ValueError):
        ScaleBalancedRegionLaw(edges=(12, 5))
    with pytest.raises(ValueError):
        ScaleBalancedRegionLaw(edges=(0,))
    with pytest.raises(ValueError):
        ScaleBalancedRegionLaw(edges=())


def test_every_weight_is_strictly_positive_so_the_law_never_filters():
    law, g = ScaleBalancedRegionLaw(), _graph()
    regions = law.regions(g)
    assert regions, "fixture must offer regions or the test cannot bind"
    assert all(w > 0.0 for w in law.weights(g, regions))


def test_support_equals_the_unconditioned_support():
    """Re-ranking, not filtering: the ordered set is the enumerated set."""
    law, g = ScaleBalancedRegionLaw(), _graph()
    ordered = law.order(g, np.random.default_rng(0))
    assert len(ordered) == len(law.regions(g))


def test_a_lone_large_region_outweighs_many_small_ones():
    """The property a uniform law cannot have at any cap."""
    law, g = ScaleBalancedRegionLaw(), _graph()
    regions = law.regions(g)
    sizes = [r.size for r in regions]
    classes = {law.scale_class(s) for s in sizes}
    if len(classes) < 2:
        pytest.skip("fixture occupies one class; the property cannot bind here")
    weights = law.weights(g, regions)
    by_class: dict[int, float] = {}
    for region, weight in zip(regions, weights):
        by_class.setdefault(law.scale_class(region.size), 0.0)
        by_class[law.scale_class(region.size)] += weight
    # every occupied class carries the same total mass
    totals = sorted(by_class.values())
    assert totals[-1] - totals[0] < 1e-9


def test_empty_classes_are_skipped_not_padded():
    law = ScaleBalancedRegionLaw()
    g = _graph("CCCC")
    regions = law.regions(g)
    if not regions:
        pytest.skip("no bridge regions on this fixture")
    occupied = {law.scale_class(r.size) for r in regions}
    weights = law.weights(g, regions)
    assert abs(sum(weights) - 1.0) < 1e-9 or len(occupied) < len(DEFAULT_EDGES) + 1


def test_order_is_deterministic_given_a_seed():
    law, g = ScaleBalancedRegionLaw(), _graph()
    a = [r.fragment for r in law.order(g, np.random.default_rng(7))]
    b = [r.fragment for r in law.order(g, np.random.default_rng(7))]
    assert a == b


def test_the_law_reads_no_witness_constant():
    """REGRESSION: bin edges are an engineering parameter, never a fitted witness value.

    19 (the required region) and 0.30 (the required retention) must not appear.
    """
    import pathlib

    src = pathlib.Path("src/compose_v4/control/scale_balanced_region_law.py").read_text()
    body = "\n".join(
        line for line in src.splitlines() if not line.strip().startswith("#")
    )
    code = body.split('"""')[-1]  # everything after the module docstring
    assert "19" not in code
    assert "0.30" not in code and "0.3 " not in code
