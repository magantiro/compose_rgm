"""An explicit region-replacement option: choose WHERE, then choose WHAT to rebuild.

WHY THIS EXISTS, measured rather than assumed.  The PMO shallow lane factors a proposal
as a draw over thirteen generic module families, of which exactly two consult a region
law.  A law over regions therefore governs a small minority of module draws and cannot
move a program-level statistic however well it is designed -- measured, forcing the
region option multiplies its effect on MCS retention about fivefold (-0.039 -> -0.19).
That is the case for making the region replacement an OPTION, selected before the family
lottery, rather than a law inside it.

The second half of the defect is what gets put BACK.  ``segment_replace`` excises a
bridge-separated region and then always rebuilds with ``_grow_actions``: a LINEAR
SINGLE-BONDED CHAIN over C/N/O off one anchor, which can never install a ring, a branch
or any other element.  Measured against the global-delta census, 80.6% of productive PMO
transitions install RING content and only 22.2% install something a linear C/N/O chain
could build.  So the replacement stage is structurally incapable of matching, and no
amount of region conditioning reaches it.

THE FACTORIZATION, which is the point:

    q(R | G)          choose a coherent region on the parent (the region law)
    q(o | G, R)       choose a REBUILD OPTION from COMPOSE's existing macro vocabulary
    q(omega | G,R,o)  realize that option exactly, through the production executor

COMPOSE already knows how to build rings.  This module does NOT introduce ring
chemistry; every option below is dispatched to a builder that already ships and is
already exercised by the generic module table -- ``_grow_actions`` for chain regrowth
and ``_ring_module`` for pendant and fused ring construction, the latter reached through
its ``locus`` argument so the ring lands AT the anchor the excision retained.  Reusing
those builders rather than reimplementing them beside their originals is the correctness
guarantee: a region replacement and an ordinary module build the same chemistry by
construction, because they call the same code.

INFORMATION BOUNDARY.  Nothing here reads a task, an oracle, a reference molecule, a
similarity threshold or a target.  The option weights are ENGINEERING PARAMETERS and are
not fitted to any answer-known witness; no witness constant (a required region size, a
required retention) appears in this module or may be introduced into it.  The draw reads
only the parent's own bridge structure and the contracted graph's own free valence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from compose_v4.chem.molecular_graph import is_element
from compose_v4.control.dynamic_program_synthesis import (
    GENERIC_MODULES,
    MAX_SEGMENT_LENGTH,
    _delete_pendant_fragment,
    _grow_actions,
    _ring_module,
    _stage,
    compile_generic_module,
)
from compose_v4.experiments.whole_ring_plan import execute_program

#: The rebuild vocabulary is the FULL generic module table plus two anchored composites.
#:
#: The region selects WHERE COMPOSE acts; it must not also decide WHAT COMPOSE may do there.
#: An earlier version of this module offered four hand-picked rebuild choices, and a parity
#: report measured the cost: all thirteen generic families are applicable on every Celecoxib
#: macro-boundary state, and TEN of them were applicable on a contracted state while being
#: unreachable from the rebuild stage (`diagnostics/pmo_macro_parity_v1/`).  A macro test run
#: against that reduced vocabulary could not have distinguished a bad search from a capability
#: that was never exposed.
#:
#: Every entry dispatches to a builder that already ships.  ``regrow`` and the two ring
#: entries are ANCHORED at the region's retained anchor; ``ring_then_grow`` composes two of
#: them; every other entry is the production ``compile_generic_module`` family applied to the
#: contracted state.  Families that cannot apply refuse through their own normal validity
#: path and are recorded as refusals -- no capability class is silently omitted.
ANCHORED_OPTIONS: tuple[str, ...] = ("regrow", "append_ring", "fuse_ring", "ring_then_grow")
#: `append_ring` and `fuse_ring` also name generic families.  The anchored branch claims those
#: names, so listing them twice would silently double their draw mass rather than add anything;
#: dedupe so the table is exactly one entry per distinct rebuild behaviour.
REPLACEMENT_OPTIONS: tuple[str, ...] = (
    *ANCHORED_OPTIONS,
    *(family for family in GENERIC_MODULES if family not in ANCHORED_OPTIONS),
)

#: Equal mass over every declared option.  Deliberately flat: the census says the shipped lane
#: is effectively all plain growth, so any weighting that is not flat would be choosing a
#: rebuild distribution before measuring one.  An engineering choice, not a fitted one.
DEFAULT_OPTION_WEIGHTS: tuple[float, ...] = tuple(1.0 for _ in REPLACEMENT_OPTIONS)


def _new_slots(before, after) -> list[int]:
    """Real slots present in ``after`` and absent from ``before``."""
    was = {int(i) for i in np.flatnonzero(is_element(before.atom_types))}
    now = {int(i) for i in np.flatnonzero(is_element(after.atom_types))}
    return sorted(now - was)


def _fusion_locus(graph, anchor: int) -> frozenset[int]:
    """The anchor plus its neighbours, so a fused ring can reach a ring it sits in."""
    neighbours = {
        int(j)
        for j in np.flatnonzero(is_element(graph.atom_types))
        if graph.bonds[anchor, int(j)]
    }
    return frozenset({int(anchor), *neighbours})


def _build(option: str, contracted, rng, *, anchor: int, budget: int):
    """Realize one rebuild option on the contracted graph.  Returns (actions, product)."""
    capacity = min(budget, 40 - contracted.n_real_atoms)
    if capacity < 1:
        raise ValueError("region replacement has no rebuild capacity")
    if option == "regrow":
        length = int(rng.integers(1, capacity + 1))
        actions, _tip, elements = _grow_actions(
            contracted, rng, length=length, elements=("C", "N", "O"), anchor=anchor
        )
        product, _receipt = execute_program(contracted, list(actions))
        return actions, product, {"inserted_atoms": length, "elements": list(elements)}
    if option in {"append_ring", "ring_then_grow"}:
        product, stage = _ring_module(
            contracted, rng, topology="pendant", label=option, locus={anchor}
        )
        actions = list(stage["actions"])
        detail = {"ring": stage["parameters"]}
        if option == "append_ring":
            return actions, product, detail
        fresh = [
            slot
            for slot in _new_slots(contracted, product)
            if int(product.implicit_h_counts[slot]) >= 1
        ]
        if not fresh:
            raise ValueError("ring offers no hydrogen-bearing atom to grow from")
        remaining = min(budget, 40 - product.n_real_atoms)
        if remaining < 1:
            raise ValueError("ring consumed the rebuild capacity")
        tail = int(rng.integers(1, remaining + 1))
        grown, _tip, elements = _grow_actions(
            product,
            rng,
            length=tail,
            elements=("C", "N", "O"),
            anchor=fresh[int(rng.integers(len(fresh)))],
        )
        final, _receipt = execute_program(product, list(grown))
        detail |= {"inserted_atoms": tail, "elements": list(elements)}
        return [*actions, *grown], final, detail
    if option == "fuse_ring":
        product, stage = _ring_module(
            contracted,
            rng,
            topology="fused",
            label=option,
            locus=_fusion_locus(contracted, anchor),
        )
        return list(stage["actions"]), product, {"ring": stage["parameters"]}
    if option in GENERIC_MODULES:
        # The full production vocabulary, reached through the production compiler so the
        # chemistry is byte-identical to an ordinary module.  These families carry no locus
        # argument, so the region constrains them by having already excised R rather than by
        # pinning the edit to the anchor; that is recorded rather than hidden.
        product, stage = compile_generic_module(contracted, rng, option)
        return (
            list(stage["actions"]),
            product,
            {"generic_module": stage["parameters"], "anchored_at_region": False},
        )
    raise ValueError(f"unknown region replacement option: {option}")


@dataclass(frozen=True)
class RegionReplacementOption:
    """``q(R|G)`` then ``q(o|G,R)`` then exact realization, over existing builders."""

    options: tuple[str, ...] = REPLACEMENT_OPTIONS
    weights: tuple[float, ...] = DEFAULT_OPTION_WEIGHTS
    region_law: object = None
    maximum_rebuild: int | None = None
    _validated: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self):
        if not self.options:
            raise ValueError("a region replacement needs at least one rebuild option")
        if len(self.options) != len(self.weights):
            raise ValueError("option and weight counts differ")
        if any(option not in REPLACEMENT_OPTIONS for option in self.options):
            raise ValueError(f"unknown rebuild option in {self.options}")
        if any(weight <= 0.0 for weight in self.weights):
            raise ValueError("every declared rebuild option needs positive mass")

    def order(self, rng) -> list[str]:
        """A weighted draw ORDER over the declared options, sampled without replacement."""
        remaining = list(self.options)
        weights = list(self.weights)
        drawn = []
        while remaining:
            mass = np.asarray(weights, dtype=float)
            at = int(rng.choice(len(remaining), p=mass / mass.sum()))
            drawn.append(remaining.pop(at))
            weights.pop(at)
        return drawn

    def compile(self, source, rng):
        """Excise a region, then rebuild it with the first option that executes."""
        delete_actions, contracted, anchor, path = _delete_pendant_fragment(
            source, rng, law=self.region_law
        )
        removed = source.n_real_atoms - contracted.n_real_atoms
        ceiling = self.maximum_rebuild
        if ceiling is None:
            # A replacement may be as large as what it replaced.  Without a region law
            # the excision is already bounded by MAX_SEGMENT_LENGTH, so this reduces to
            # v1's ceiling and cannot widen an unlawed draw.
            ceiling = max(removed, MAX_SEGMENT_LENGTH)
        failures = {}
        for option in self.order(rng):
            try:
                build_actions, product, detail = _build(
                    option, contracted, rng, anchor=anchor, budget=ceiling
                )
            except ValueError as error:
                failures[option] = str(error)
                continue
            actions = [*delete_actions, *build_actions]
            product, receipt = execute_program(source, list(actions))
            return product, _stage(
                "region_replace",
                receipt,
                {
                    "rebuild_option": option,
                    "deleted_path": path,
                    "deleted_atoms": len(path),
                    "excised_atoms": removed,
                    "cycle_openings": len(delete_actions) - len(path),
                    "retained_anchor": anchor,
                    "rebuild_capacity": ceiling,
                    "rebuild_failures": failures,
                    **detail,
                },
            )
        raise ValueError(f"no rebuild option executed: {failures}")
