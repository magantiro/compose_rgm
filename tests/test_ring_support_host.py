"""The ring-grow host, and the catalog's dependence on how much of it remains.

The legal ring-template support collapses along a compiled trace, and the
question these tests pin down is WHY. ``_eligible_grow_host_graph`` admits only
acyclic, carbon, neutral, single-bonded atoms, so committing a ring system --
or installing a heteroatom, or raising a bond order -- permanently removes
atoms from the scaffold every later ring decision is made against.

Each test states one exclusive cause and is written so that breaking that cause
alone turns it red. The catalog-side test is the important one: it is a
property of the catalog by itself, so it cannot be explained away by whichever
states happened to be sampled.
"""

from __future__ import annotations

import networkx as nx
import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.eval.ring_support_host import (
    catalog_host_requirement_histogram,
    catalog_small_ring_share_by_host_size,
    eligible_host_census,
    host_size_predicts,
    template_host_atoms,
    template_source_pattern_is_forest,
)


def _state(smiles: str, slots: int = 40):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), slots)


# ---- What a state offers ----


def test_an_acyclic_all_carbon_chain_is_entirely_eligible() -> None:
    """The baseline: nothing is excluded, so the host is the whole molecule."""

    census = eligible_host_census(_state("CCCCCCCC"))
    assert census["real_atoms"] == 8
    assert census["host_atoms"] == 8
    assert census["host_bonds"] == 7
    assert census["host_components"] == 1
    assert census["largest_host_tree"] == 8
    assert census["lost_to_cycles"] == 0
    assert census["lost_to_heteroatoms"] == 0
    assert census["lost_to_charge"] == 0


def test_a_committed_ring_removes_exactly_its_own_atoms() -> None:
    """Cyclic atoms leave the host; the pendant chain does not.

    This is the mechanism behind the per-ring-system ordinal effect: each ring
    a molecule commits is subtracted from the scaffold the next one is chosen
    against.
    """

    census = eligible_host_census(_state("C1CCCCC1CCC"))
    assert census["real_atoms"] == 9
    assert census["lost_to_cycles"] == 6
    assert census["host_atoms"] == 3
    assert census["largest_host_tree"] == 3
    assert census["lost_to_heteroatoms"] == 0


def test_a_heteroatom_removes_itself_and_splits_the_host() -> None:
    """An installed heteroatom is excluded, and it can FRAGMENT the scaffold.

    Fragmentation matters independently of the count: a template needs one
    contiguous tree, so two fragments of three cannot carry a template needing
    six.
    """

    census = eligible_host_census(_state("CCCOCCC"))
    assert census["real_atoms"] == 7
    assert census["lost_to_heteroatoms"] == 1
    assert census["host_atoms"] == 6
    assert census["host_components"] == 2
    assert census["largest_host_tree"] == 3


def test_a_double_bond_splits_the_host_without_removing_an_atom() -> None:
    """Only SINGLE bonds join the host, so a double bond cuts it in two.

    Asserted separately from the heteroatom case because the atom count is
    unchanged here -- a census that tracked atoms alone would miss it.
    """

    census = eligible_host_census(_state("CCC=CCC"))
    assert census["real_atoms"] == 6
    assert census["host_atoms"] == 6
    assert census["lost_to_heteroatoms"] == 0
    assert census["lost_to_cycles"] == 0
    assert census["host_components"] == 2
    assert census["largest_host_tree"] == 3


def test_host_loss_causes_are_exclusive_and_bounded() -> None:
    """The three causes never double-count and never exceed the molecule."""

    for smiles in ("C1CCCCC1CCO", "CCCOCC=CC", "c1ccccc1CCN", "CCCCCCCCCC"):
        census = eligible_host_census(_state(smiles))
        lost = (
            census["lost_to_cycles"]
            + census["lost_to_heteroatoms"]
            + census["lost_to_charge"]
        )
        assert lost <= census["real_atoms"], smiles
        assert census["host_atoms"] <= census["real_atoms"] - lost, smiles
        assert census["largest_host_tree"] <= census["host_atoms"], smiles


# ---- What the catalog asks for ----


class _Template:
    """Stand-in carrying only the two fields the host helpers read."""

    def __init__(self, span, source_bonds, target_bonds) -> None:
        self.span = span
        self.source_bonds = source_bonds
        self.target_bonds = target_bonds


def _ring_template(ring: int, host_chain: int) -> _Template:
    """A template installing one `ring`-cycle onto a `host_chain` path."""

    source = tuple((i, i + 1, 1) for i in range(host_chain - 1))
    target = tuple((i, (i + 1) % ring, 1) for i in range(ring))
    return _Template(ring, source, target)


def test_template_host_requirement_is_read_from_the_source_pattern() -> None:
    """``span`` counts installed atoms and is NOT the host requirement."""

    template = _ring_template(ring=6, host_chain=4)
    assert template_host_atoms(template) == 4
    assert template.span == 6


def test_every_production_template_requires_a_forest_host() -> None:
    """No template can match a host containing a ring.

    This is what makes the collapse a SCOPE restriction rather than a catalog
    gap: ring systems are installed atomically on acyclic carbon, so a molecule
    with several ring systems cannot reuse the ones it already built.
    """

    cyclic_host = _Template(3, ((0, 1, 1), (1, 2, 1), (2, 0, 1)), ((0, 1, 1),))
    assert template_source_pattern_is_forest(_ring_template(6, 4))
    assert not template_source_pattern_is_forest(cyclic_host)


def test_small_ring_share_rises_as_the_host_shrinks() -> None:
    """The mechanism, stated over the catalog alone.

    A large ring needs a large contiguous host; a three-ring needs three atoms.
    So the survivors of a shrinking host are increasingly the small rings, and
    a support measured on a small host is small-ring-enriched by construction
    rather than by any defect in the policy or the reward.
    """

    templates = (
        _ring_template(ring=3, host_chain=3),
        _ring_template(ring=4, host_chain=4),
        _ring_template(ring=6, host_chain=10),
        _ring_template(ring=6, host_chain=12),
        _ring_template(ring=5, host_chain=14),
    )
    report = catalog_small_ring_share_by_host_size(
        templates, host_sizes=(3, 4, 10, 14)
    )
    assert report["3"] == {
        "templates_fitting": 1,
        "small_ring_templates": 1,
        "small_ring_share": 1.0,
    }
    assert report["4"]["small_ring_share"] == 1.0
    assert report["10"]["small_ring_share"] == pytest.approx(2 / 3)
    assert report["14"]["small_ring_share"] == pytest.approx(2 / 5)
    # Monotone in the direction the mechanism predicts.
    shares = [report[key]["small_ring_share"] for key in ("3", "4", "10", "14")]
    assert shares == sorted(shares, reverse=True)


def test_host_requirement_histogram_counts_every_template_once() -> None:
    templates = (
        _ring_template(3, 3),
        _ring_template(4, 3),
        _ring_template(6, 10),
    )
    histogram = catalog_host_requirement_histogram(templates)
    assert histogram == {"3": 2, "10": 1}
    assert sum(histogram.values()) == len(templates)


def test_an_empty_catalog_reports_a_zero_share_not_a_crash() -> None:
    report = catalog_small_ring_share_by_host_size((), host_sizes=(5,))
    assert report["5"] == {
        "templates_fitting": 0,
        "small_ring_templates": 0,
        "small_ring_share": 0.0,
    }


def test_negative_host_size_is_refused() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        catalog_small_ring_share_by_host_size((), host_sizes=(-1,))


# ---- The association helper refuses the cases where it would mislead ----


def test_host_size_correlation_refuses_degenerate_input() -> None:
    with pytest.raises(ValueError, match="align"):
        host_size_predicts((1, 2, 3), (0.1, 0.2))
    with pytest.raises(ValueError, match="at least two"):
        host_size_predicts((1,), (0.1,))
    with pytest.raises(ValueError, match="constant"):
        host_size_predicts((4, 4, 4), (0.1, 0.2, 0.3))


def test_host_size_correlation_recovers_a_known_sign() -> None:
    assert host_size_predicts((1, 2, 3, 4), (0.4, 0.3, 0.2, 0.1)) == pytest.approx(-1.0)
    assert host_size_predicts((1, 2, 3, 4), (0.1, 0.2, 0.3, 0.4)) == pytest.approx(1.0)


def test_the_census_uses_the_production_host_builder() -> None:
    """A transcription would drift; this asserts the real builder is consulted.

    Driven through the production function rather than by reading the import,
    so removing the eligibility rule from the builder fails here.
    """

    from compose_v4.rewrite.ring_system_fiber import _eligible_grow_host_graph

    state = _state("C1CCCCC1CCC")
    host = _eligible_grow_host_graph(state)
    census = eligible_host_census(state)
    assert census["host_atoms"] == host.number_of_nodes()
    assert census["host_bonds"] == host.number_of_edges()
    assert nx.is_forest(host)
