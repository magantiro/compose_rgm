"""Population-derived pendant exchange compiled to existing primitive edits.

Donors are complete molecules, not a fixed fragment vocabulary. This optional
controller proposal is executor-supported; it is NOT claimed to be a Doob
transform of, or absolutely continuous with, the frozen learned marked law.
Hypothetical endpoints are plans, never committed molecular transitions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from compose_v4.chem.molecular_graph import MolecularGraph, is_element
from compose_v4.chem.state import empty_molecular_graph
from compose_v4.experiments.winner_paths import PathConfig, replay, search_mapping
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import encode_state


@dataclass(frozen=True)
class PendantCut:
    anchor: int
    root: int
    component: tuple[int, ...]


def pendant_cuts(graph: MolecularGraph) -> tuple[PendantCut, ...]:
    """Both orientations of single-bond bridges, in persistent coordinates."""
    real = tuple(int(i) for i in np.flatnonzero(is_element(graph.atom_types)))
    neighbors = {i: tuple(int(j) for j in np.flatnonzero(graph.bonds[i])) for i in real}
    result = []
    for anchor in real:
        for root in neighbors[anchor]:
            if graph.bonds[anchor, root] != 1:
                continue
            seen, frontier = {root}, [root]
            while frontier:
                at = frontier.pop()
                for other in neighbors[at]:
                    if {at, other} == {anchor, root} or other in seen:
                        continue
                    seen.add(other)
                    frontier.append(other)
            if anchor not in seen:
                result.append(PendantCut(anchor, root, tuple(sorted(seen))))
    return tuple(result)


def transplant_plan(source, donor, source_cut: PendantCut, donor_cut: PendantCut):
    """Build a desired graph and known correspondence, without MCS or replay parsing."""
    if len(source.atom_types) != 48 or len(donor.atom_types) != 48:
        raise ValueError("donor programs require exact 48-slot states")
    if source_cut not in pendant_cuts(source) or donor_cut not in pendant_cuts(donor):
        raise ValueError("cut is not an oriented single-bond bridge of its exact graph")
    retained = sorted(
        set(np.flatnonzero(is_element(source.atom_types))) - set(source_cut.component)
    )
    added = donor_cut.component
    if not 1 <= len(retained) + len(added) <= 40:
        return {"status": "unsupported_size"}
    # Packed target coordinates are a proposed endpoint only. search_mapping
    # transports them to source slots and executor replay preserves those slots.
    target = empty_molecular_graph(48)
    left = {int(old): new for new, old in enumerate(retained)}
    right = {old: len(retained) + new for new, old in enumerate(added)}
    for graph, mapping in ((source, left), (donor, right)):
        for old, new in mapping.items():
            for field in ("atom_types", "formal_charges", "implicit_h_counts"):
                getattr(target, field)[new] = getattr(graph, field)[old]
            for old_other, new_other in mapping.items():
                target.bonds[new, new_other] = graph.bonds[old, old_other]
    a, b = left[source_cut.anchor], right[donor_cut.root]
    target.bonds[a, b] = target.bonds[b, a] = 1
    return {
        "status": "planned",
        "target": target,
        "mapping": sorted(left.items()),
        "retained_slots": sorted(left),
        "released_fraction": len(source_cut.component) / source.n_real_atoms,
        "removed_atoms": len(source_cut.component),
        "added_atoms": len(added),
    }


def compile_transplant(source, donor, source_cut, donor_cut, *, config=None):
    config = PathConfig() if config is None else config
    plan = transplant_plan(source, donor, source_cut, donor_cut)
    if plan["status"] != "planned":
        return plan
    desired = canonical_state_key(plan["target"])
    metadata = {k: v for k, v in plan.items() if k not in ("status", "target", "mapping")}
    if desired == canonical_state_key(source):
        return {**metadata, "status": "self_proposal"}
    result = search_mapping(source, plan["target"], plan["mapping"], config)
    if "best_residual" in result:
        result["best_residual"] = int(result["best_residual"])
    if result["status"] != "witness_found":
        return {**metadata, **result}
    states = replay(encode_state(source), result["actions"], desired)
    # The frozen context is stronger than merely matching an endpoint SMILES.
    for state in states:
        from compose_v4.rewrite.trace_shard import decode_state

        graph = decode_state(state)
        for field in ("atom_types", "formal_charges"):
            if not np.array_equal(
                getattr(graph, field)[plan["retained_slots"]],
                getattr(source, field)[plan["retained_slots"]],
            ):
                raise ValueError("compiled donor program changed retained atom identity/charge")
        slots = np.ix_(plan["retained_slots"], plan["retained_slots"])
        if not np.array_equal(graph.bonds[slots], source.bonds[slots]):
            raise ValueError("compiled donor program changed retained connectivity")
    return {
        **metadata,
        **result,
        "status": "compiled",
        "smiles": desired,
        "states": states,
        "primitive_steps": len(result["actions"]),
        "support": "frozen_executor_only; frozen_reference_probability_not_certified",
    }
