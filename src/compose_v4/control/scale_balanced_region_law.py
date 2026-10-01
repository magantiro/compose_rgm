"""A region law that balances draw mass across SIZE CLASSES, not across regions.

WHY THIS EXISTS, measured rather than assumed.  Removing the size cap on the region
draw was shown not to move PMO's realized changed-region size: end to end the median
went 4.000 to 4.167 against a paired standard error of 0.216, and conditional on
``segment_replace`` firing only 0.2% of draws reached fifteen changed atoms
(``diagnostics/pmo_region_law_v1/``).  The reason is that an unconditioned law draws
UNIFORMLY over bridge-separated regions and most of them are small, so uncapping only
adds large regions to the menu without giving any reason to prefer one.  COMPOSE can
already execute the large change -- an isolated uncapped delete reaches twenty-five
atoms at full validity -- so the missing thing is structural INTENT, not capability.

THE LAW.  Factor the draw as

    q(R | G) = q(s | G) * q(R | s, G)

choosing a size CLASS first and a region within it second.  Every class the state
actually offers receives equal mass, so a state carrying one large region and thirty
small ones draws the large one as often as the small ones collectively -- which a
uniform law over regions cannot do at any cap.

INFORMATION BOUNDARY.  ``edges`` is an ENGINEERING PARAMETER.  It is NOT fitted to the
answer-known witness routes, and no witness constant (the required region of nineteen
atoms, the required retention of 0.30) appears anywhere in this module or may be
introduced into it.  The law reads only the state's own bridge structure.

The support floor keeps every drawable region strictly positive, so this re-ranks and
never filters: the reachable set is exactly the reachable set of the uniform law.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.control.bridge_region_law import BridgeRegion, bridge_separated_regions

#: Class boundaries in heavy atoms: small ``< 5``, medium ``5..11``, large ``>= 12``.
#: An engineering choice, deliberately coarse, and not derived from any witness.
DEFAULT_EDGES: tuple[int, ...] = (5, 12)

#: Minimum relative weight, so the law re-ranks and never filters.
#:
#: Deliberately far below :data:`BridgeRegionLaw`'s 0.05.  There the floor guards an
#: EXPONENTIAL margin tilt that can drive a weight to zero; here weights are
#: ``share / members`` and are bounded below by construction, so a large floor would
#: only re-inflate a crowded class -- at 0.05 a state with thirty small regions gives
#: the small class 1.5 against the large class's 0.33, undoing the balance this law
#: exists to impose.  A test pins the class totals to equality.
SUPPORT_FLOOR = 1e-3


@dataclass(frozen=True)
class ScaleBalancedRegionLaw:
    """Equal draw mass per occupied size class, uniform within a class."""

    edges: tuple[int, ...] = DEFAULT_EDGES
    floor: float = SUPPORT_FLOOR
    maximum: int | None = None

    def __post_init__(self) -> None:
        if not self.edges or any(e < 1 for e in self.edges):
            raise ValueError("class edges must be positive heavy-atom counts")
        if list(self.edges) != sorted(set(self.edges)):
            raise ValueError("class edges must be strictly increasing and unique")
        if not 0.0 < self.floor <= 1.0:
            raise ValueError("support floor must lie in (0, 1]")
        if self.maximum is not None and self.maximum < 1:
            raise ValueError("region size cap must be at least one atom")

    @property
    def conditioned(self) -> bool:
        return True

    def scale_class(self, size: int) -> int:
        """Index of the size class a region of ``size`` atoms falls in."""
        for index, edge in enumerate(self.edges):
            if size < edge:
                return index
        return len(self.edges)

    def regions(self, graph: MolecularGraph) -> tuple[BridgeRegion, ...]:
        return bridge_separated_regions(graph, maximum=self.maximum)

    def weights(
        self, graph: MolecularGraph, regions: tuple[BridgeRegion, ...]
    ) -> list[float]:
        """Relative draw weight of each region; strictly positive everywhere.

        Each OCCUPIED class receives equal total mass, split evenly inside it, so a
        lone large region carries the whole large-class share rather than one vote
        among many.  Empty classes are skipped, never padded.

        The floor is a support guard and is set low enough not to bind on realistic
        region counts; where it does bind, class balance is traded for support and the
        support wins.
        """
        if not regions:
            return []
        members: dict[int, int] = {}
        for region in regions:
            key = self.scale_class(region.size)
            members[key] = members.get(key, 0) + 1
        share = 1.0 / len(members)
        return [
            max(self.floor, share / members[self.scale_class(region.size)])
            for region in regions
        ]

    def order(self, graph: MolecularGraph, rng) -> list[BridgeRegion]:
        """Regions in weighted draw order, sampled without replacement.

        Efraimidis-Spirakis, matching :class:`BridgeRegionLaw`: sorting by
        ``-log(u_i) / w_i`` ascending is successive weighted selection without
        replacement, so the head is one draw from the law and the tail is the law
        conditioned on that draw having been refused by the executor.
        """
        regions = self.regions(graph)
        if not regions:
            return []
        weights = self.weights(graph, regions)
        keys = [
            -math.log(max(float(rng.random()), 1e-300)) / weight
            for weight in weights
        ]
        return [region for _, region in sorted(zip(keys, regions), key=lambda p: p[0])]
