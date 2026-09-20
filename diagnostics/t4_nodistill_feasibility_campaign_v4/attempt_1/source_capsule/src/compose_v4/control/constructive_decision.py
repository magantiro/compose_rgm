"""Generic descriptions of realized constructive program decisions.

The route corpus and the runtime proposal law must describe a construction with the
same object.  This module owns that shared description so an offline teacher decision
and a newly compiled proposal cannot silently use different WHERE/HOW semantics.
"""

from __future__ import annotations


def attachment_sites(actions, indices, source_atom_count: int) -> list[int]:
    """Return pre-existing atoms a bounded construction attaches to."""
    created, sites = set(), set()
    for index in indices:
        payload = actions[index].get("payload") or {}
        for key in ("v", "slot", "fresh", "target"):
            value = payload.get(key)
            if isinstance(value, int):
                created.add(value)
        for neighbour in payload.get("neighbors") or payload.get("neighbours") or []:
            slot = neighbour[0] if isinstance(neighbour, (list, tuple)) else neighbour
            if isinstance(slot, int) and slot < source_atom_count and slot not in created:
                sites.add(slot)
    return sorted(sites)


def stage_decision(source_state, actions) -> dict | None:
    """Describe one realized constructive stage as its generic site and mode."""
    if not actions:
        return None
    created = sum(1 for action in actions if action.get("executor_rule") == "atom_insert")
    closes = any(action.get("executor_rule") == "cycle_close" for action in actions)
    if not created and not closes:
        return None
    source_atoms = len(source_state.get("atom_types", []))
    sites = attachment_sites(actions, range(len(actions)), source_atoms)
    if not sites:
        return None
    return {
        "site": min(sites),
        "mode": {
            "attachment_count": len(sites),
            "created_atoms": created,
            "closes_ring": closes,
            "opens_ring": any(action.get("executor_rule") == "cycle_open" for action in actions),
            "primitive_count": len(actions),
        },
    }
