"""Constructive decisions at the level a proposal prior can actually learn.

Teacher-forcing the production sampler at 308 decision points across the 77 exact
routes reproduced a teacher decision 406 times in 31,235 proposals -- but every one of
those reproductions came from a destructive family. Constructive families produced
zero in 11,762 proposals, while compiling at 285 to 308 of the 308 states. The
vocabulary is present; the probability over WHERE, attachments, elements, topology and
bond orders is diffuse enough that it never lands on the teacher's choice.

The 213-option decision menu cannot express the fix, because it records those
constructions as primitive fallback: 64 of 77 routes close a ring, 130 closures in
total, and none of it surfaces as a ring *option*. So the unit here is the dependency
region -- a connected component of the exact trace, carrying its own created handles
and their dependency edges -- which is where a coherent construction is visible.

This module extracts those decisions and describes what a prior would have to place
mass on. It fits nothing; teacher routes are diagnostic and no runtime component reads
them.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

from compose_v4.control.dependency_region_program import dependency_region_program

SCHEMA_VERSION = "t4_constructive_decisions_v1"


# A component counts as constructive when it adds atoms or closes a ring. Deletions
# and restatements are the half the sampler already reproduces at 4.5 to 10.8%.
def is_constructive(component: dict) -> bool:
    return bool(component["created_outputs"]) or component["contains_cycle_close"]


def attachment_sites(actions, indices, source_atom_count: int) -> list[int]:
    """Pre-existing atoms a component builds onto.

    An inserted atom names its neighbours in the payload; a neighbour whose slot
    predates the component is an attachment point, which is the WHERE a prior has to
    choose. Slots created inside the component are excluded.
    """
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


def describe_component(component: dict, actions, source_atom_count: int) -> dict:
    indices = component["primitive_indices"]
    sites = attachment_sites(actions, indices, source_atom_count)
    return {
        "primitive_count": component["primitive_count"],
        "created_atoms": component["created_outputs"],
        "changed_atom_lifetimes": component["changed_atom_lifetime_count"],
        "rule_counts": dict(sorted(component["rule_counts"].items())),
        "closes_ring": component["contains_cycle_close"],
        "opens_ring": component["contains_cycle_open"],
        "restates_ring": component["contains_ring_restate"],
        "attachment_sites": sites,
        "attachment_count": len(sites),
        "constructive": is_constructive(component),
    }


def route_decisions(receipt: Path) -> dict:
    """Decompose one teacher route into its dependency-region decisions."""
    payload = json.loads(gzip.decompress(Path(receipt).read_bytes()))["payload"]["path"]
    states, actions = tuple(payload["states"]), tuple(payload["actions"])
    program = dependency_region_program(states, actions)
    source_atoms = len(states[0].get("atom_types", []))
    described = [describe_component(c, actions, source_atoms) for c in program["components"]]
    return {
        "receipt": str(receipt),
        "primitives": len(actions),
        "components": described,
        "component_count": program["component_count"],
        "constructive_components": sum(d["constructive"] for d in described),
        "created_handles": len(program["created_handles"]),
        "created_dependency_edges": len(program["created_dependency_edges"]),
        "exact_replay": program["exact_replay"],
        "complete_representation_supported": program["complete_representation_supported"],
        "abstention_reason": program["abstention_reason"],
    }


def census(routes) -> dict:
    """What a constructive proposal prior would have to place mass on."""
    constructive = [c for r in routes for c in r["components"] if c["constructive"]]
    destructive = [c for r in routes for c in r["components"] if not c["constructive"]]
    if not constructive:
        return {"constructive_components": 0}

    def spread(values):
        ordered = sorted(values)
        return {
            "n": len(ordered),
            "min": ordered[0],
            "p50": ordered[len(ordered) // 2],
            "p90": ordered[int(0.9 * (len(ordered) - 1))],
            "max": ordered[-1],
        }

    return {
        "routes": len(routes),
        "routes_with_a_constructive_decision": sum(
            1 for r in routes if r["constructive_components"]
        ),
        "constructive_components": len(constructive),
        "destructive_components": len(destructive),
        "primitives_per_constructive_decision": spread(
            [c["primitive_count"] for c in constructive]
        ),
        "atoms_created_per_decision": spread([c["created_atoms"] for c in constructive]),
        "attachment_sites_per_decision": spread([c["attachment_count"] for c in constructive]),
        "decisions_closing_a_ring": sum(c["closes_ring"] for c in constructive),
        "rule_mix": dict(
            Counter(
                rule for c in constructive for rule, n in c["rule_counts"].items() for _ in range(n)
            ).most_common()
        ),
        "attachment_count_histogram": dict(
            sorted(Counter(c["attachment_count"] for c in constructive).items())
        ),
        "interpretation": (
            "each of these is one coherent construction the sampler must place mass on: "
            "a choice of attachment sites, how many atoms to add, which rules, and "
            "whether to close a ring. The 213-option menu records none of it."
        ),
        "new_oracle_calls": 0,
    }
