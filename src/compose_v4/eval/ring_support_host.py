"""Why the legal ring-template support collapses: it is the HOST, not the catalog.

``_eligible_grow_host_graph`` restricts v1 ring installation to a scaffold of
**acyclic, carbon, neutral, single-bonded** atoms, and asserts that scaffold is
a forest.  Every one of the catalog's templates therefore describes a FOREST
source pattern; none can require an existing ring.  Two consequences follow,
and both are measurable rather than arguable:

* Committing a ring system removes its atoms from the host permanently, and so
  does restating an atom to a heteroatom or raising a bond order.  The ring
  systems of one molecule are competing for a single shrinking carbon forest.
* The catalog's small-ring share is a steep function of how much host remains.
  A template installing a large ring needs a large contiguous carbon tree to
  sit on; a three-membered ring needs three atoms.  So as the host shrinks the
  survivors are increasingly the small rings -- from the catalog's own ~9%
  small-ring share up to 100% on a three-atom host.

This module measures both, through the production host builder rather than a
transcription of it, so a change to the eligibility rule moves these numbers
instead of silently disagreeing with them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import networkx as nx
import numpy as np

from compose_v4.chem.molecular_graph import ELEMENT_TO_IDX, MolecularGraph
from compose_v4.eval.ring_calibration import ring_template_cycle_sizes
from compose_v4.rewrite.ring_system_fiber import (
    _cyclic_atoms,
    _eligible_grow_host_graph,
    _state_graph,
)

CARBON = int(ELEMENT_TO_IDX["C"])


# ---- The host a state offers ----


def eligible_host_census(state: MolecularGraph) -> dict[str, int]:
    """Decompose what a state offers the ring-grow support, and what it lost.

    The host itself comes from the production builder.  The loss decomposition
    beside it is computed independently and the three exclusive causes are
    reported separately, because "the support collapsed" is not actionable
    while "the support collapsed because 14 atoms are now cyclic" is.

    ``largest_host_tree`` is the operative number, not ``host_atoms``: a
    template needs one CONTIGUOUS tree to sit on, so a host split into three
    fragments of four atoms cannot carry a template needing seven.
    """

    full = _state_graph(state, include_atom_labels=False)
    cyclic = _cyclic_atoms(full)
    host = _eligible_grow_host_graph(state)
    real = [int(node) for node in full.nodes()]

    lost_cyclic = sum(1 for v in real if v in cyclic)
    lost_heteroatom = sum(
        1 for v in real if v not in cyclic and int(state.atom_types[v]) != CARBON
    )
    lost_charged = sum(
        1
        for v in real
        if v not in cyclic
        and int(state.atom_types[v]) == CARBON
        and int(state.formal_charges[v]) != 0
    )
    components = (
        [len(part) for part in nx.connected_components(host)]
        if host.number_of_nodes()
        else []
    )
    return {
        "real_atoms": len(real),
        "host_atoms": host.number_of_nodes(),
        "host_bonds": host.number_of_edges(),
        "host_components": len(components),
        "largest_host_tree": max(components, default=0),
        "lost_to_cycles": lost_cyclic,
        "lost_to_heteroatoms": lost_heteroatom,
        "lost_to_charge": lost_charged,
    }


# ---- What the catalog asks of a host ----


def template_host_atoms(template: Any) -> int:
    """Atoms of the host one template consumes.

    Read from ``source_bonds`` -- the pattern the template matches against --
    not from ``span``, which counts the atoms of the INSTALLED system and
    coincides with the host requirement only when the template inserts none.
    """

    used = {int(a) for a, _b, _order in template.source_bonds}
    used |= {int(b) for _a, b, _order in template.source_bonds}
    return len(used)


def template_source_pattern_is_forest(template: Any) -> bool:
    """Whether a template could ever match a host containing a ring."""

    graph = nx.Graph()
    graph.add_edges_from(
        (int(a), int(b)) for a, b, _order in template.source_bonds
    )
    return graph.number_of_nodes() == 0 or nx.is_forest(graph)


def catalog_small_ring_share_by_host_size(
    templates: Sequence[Any],
    *,
    host_sizes: Sequence[int],
    maximum_small_ring: int = 4,
) -> dict[str, dict[str, float | int]]:
    """Small-ring share of the templates that fit on a host of each size.

    This is the mechanism behind the ordinal effect, expressed without
    reference to any particular molecule: it is a property of the catalog
    alone, so it cannot be explained away by the sample of states measured.
    """

    if any(size < 0 for size in host_sizes):
        raise ValueError("host sizes must be non-negative")
    requirement = [template_host_atoms(template) for template in templates]
    is_small = [
        bool(sizes) and min(sizes) <= maximum_small_ring
        for sizes in map(ring_template_cycle_sizes, templates)
    ]
    report: dict[str, dict[str, float | int]] = {}
    for size in host_sizes:
        fitting = [index for index, need in enumerate(requirement) if need <= size]
        small = sum(1 for index in fitting if is_small[index])
        report[str(size)] = {
            "templates_fitting": len(fitting),
            "small_ring_templates": small,
            "small_ring_share": (small / len(fitting)) if fitting else 0.0,
        }
    return report


def catalog_host_requirement_histogram(templates: Sequence[Any]) -> dict[str, int]:
    """How many host atoms each template needs, as a distribution."""

    histogram: dict[str, int] = {}
    for template in templates:
        key = str(template_host_atoms(template))
        histogram[key] = histogram.get(key, 0) + 1
    return dict(sorted(histogram.items(), key=lambda item: int(item[0])))


def host_size_predicts(
    host_sizes: Sequence[int],
    observed: Sequence[float],
) -> float:
    """Pearson correlation between host size and an observed support statistic.

    Returned as a bare float so a caller cannot mistake it for a fitted model;
    it is a descriptive association, not an attribution.
    """

    if len(host_sizes) != len(observed):
        raise ValueError("host sizes and observations must align")
    if len(host_sizes) < 2:
        raise ValueError("correlation needs at least two observations")
    left = np.asarray(host_sizes, dtype=float)
    right = np.asarray(observed, dtype=float)
    if left.std() == 0.0 or right.std() == 0.0:
        raise ValueError("correlation is undefined for a constant sequence")
    return float(np.corrcoef(left, right)[0, 1])
