"""Task-independent cold-start coverage for shared T4 FiberControl.

The first value-free batch cannot use learned utility to distinguish proposal
families.  This policy therefore spends its fixed batch on deterministic coverage
of transformation scale and non-route proposal experts.  It contains no target,
cell, molecule, route, or score information.

Version 1 is deliberately narrow: it applies to round one of an eight-query
batch.  Route-complete-region proposals receive one small, one medium, and three
large slots.  Nonempty shallow and anchored-replacement pools receive one slot
each.  The final slot remains ordinary exploration unless the optional
protonation-aware pool is nonempty, in which case that expert receives the slot.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

SCHEMA_VERSION = "t4_cold_start_allocation_v1"
ROUTE_EXPERT = "route_complete_region"
SHALLOW_EXPERT = "shallow"
ANCHORED_EXPERT = "anchored_replacement"
PROTONATION_EXPERT = "protonation_aware_retained_subgraph"

KNOWN_EXPERTS = (
    SHALLOW_EXPERT,
    ANCHORED_EXPERT,
    ROUTE_EXPERT,
    PROTONATION_EXPERT,
)
ROUTE_SCALE_FLOOR_ITEMS = (("small", 1), ("medium", 1), ("large", 3))


@dataclass(frozen=True)
class ColdStartAllocationPlan:
    """Immutable description of the selector floors for one round.

    Tuple storage keeps the plan hashable.  The mapping properties return fresh
    dictionaries suitable for the existing selector APIs.
    """

    schema_version: str
    active: bool
    round_index: int
    batch_size: int
    route_scale_floor_items: tuple[tuple[str, int], ...]
    expert_floor_items: tuple[tuple[str, int], ...]
    exploration_slots: int

    @property
    def route_scale_floor_counts(self) -> dict[str, int]:
        return dict(self.route_scale_floor_items)

    @property
    def expert_floor_counts(self) -> dict[str, int]:
        return dict(self.expert_floor_items)


def cold_start_allocation_v1(
    *,
    round_index: int,
    batch_size: int,
    available_experts: Iterable[str],
) -> ColdStartAllocationPlan:
    """Return the frozen round-one allocation for nonempty expert pools.

    ``available_experts`` means experts with at least one candidate in the
    current union, not merely experts registered in the runtime vocabulary.
    Missing pools therefore release their slots back to ordinary acquisition.
    Unknown expert names fail loudly rather than silently changing coverage.
    """

    if isinstance(round_index, bool) or not isinstance(round_index, int) or round_index < 1:
        raise ValueError(f"round_index must be a positive integer, got {round_index!r}")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size != 8:
        raise ValueError(f"{SCHEMA_VERSION} requires batch_size=8, got {batch_size!r}")

    available = frozenset(str(expert) for expert in available_experts)
    unknown = available.difference(KNOWN_EXPERTS)
    if unknown:
        raise ValueError(f"unknown proposal experts: {sorted(unknown)}")

    if round_index != 1:
        return ColdStartAllocationPlan(
            schema_version=SCHEMA_VERSION,
            active=False,
            round_index=round_index,
            batch_size=batch_size,
            route_scale_floor_items=(),
            expert_floor_items=(),
            exploration_slots=batch_size,
        )

    route_items = ROUTE_SCALE_FLOOR_ITEMS if ROUTE_EXPERT in available else ()
    expert_items = tuple(
        (expert, 1)
        for expert in (SHALLOW_EXPERT, ANCHORED_EXPERT, PROTONATION_EXPERT)
        if expert in available
    )
    guaranteed_slots = sum(count for _, count in route_items) + sum(
        count for _, count in expert_items
    )
    if guaranteed_slots > batch_size:
        raise ValueError(
            f"cold-start floors require {guaranteed_slots} slots, batch has {batch_size}"
        )
    return ColdStartAllocationPlan(
        schema_version=SCHEMA_VERSION,
        active=True,
        round_index=round_index,
        batch_size=batch_size,
        route_scale_floor_items=route_items,
        expert_floor_items=expert_items,
        exploration_slots=batch_size - guaranteed_slots,
    )


__all__ = [
    "ANCHORED_EXPERT",
    "KNOWN_EXPERTS",
    "PROTONATION_EXPERT",
    "ROUTE_EXPERT",
    "ROUTE_SCALE_FLOOR_ITEMS",
    "SCHEMA_VERSION",
    "SHALLOW_EXPERT",
    "ColdStartAllocationPlan",
    "cold_start_allocation_v1",
]
