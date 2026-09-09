"""Target-informed diagnostic witnesses, never generator proposals or rewards."""

from __future__ import annotations

import heapq
from collections import Counter
from dataclasses import dataclass
from itertools import count

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFMCS

from compose_v4.chem.molecular_graph import (
    ORGANIC_VOCABULARY,
    MolecularGraph,
    MolecularGraphError,
    is_element,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import empty_molecular_graph, pad_molecular_graph
from compose_v4.control.option_continuation import exact_graph_key
from compose_v4.data.charge_policy import charge_policy_preserved
from compose_v4.rewrite.action_codec_v4 import decode_action, encode_action
from compose_v4.rewrite.kernel import (
    InvalidRewrite,
    canonical_state_key,
    editing_v2_semantic_rewrite_system,
)
from compose_v4.rewrite.operators import (
    AtomDelete,
    AtomInsert,
    BondReorder,
    BondReroute,
    CycleCloseEdge,
    CycleOpenEdge,
    SemanticAtomRestate,
)
from compose_v4.rewrite.trace_shard import decode_state, encode_state
from compose_v4.rewrite.tracelets import BondOrderChange, RingSystemRestate


@dataclass(frozen=True)
class PathConfig:
    slots: int = 48
    max_active: int = 40
    mapping_timeout_seconds: int = 1
    matches_per_molecule: int = 4
    max_expansions: int = 128
    children_per_expansion: int = 4
    max_steps: int = 64

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in vars(self).values()):
            raise ValueError("path-search allocations must be positive integers")
        if self.max_active != 40 or self.slots != 48:
            raise ValueError("diagnostic preserves the existing 40-active/48-slot support")


def graph_stats(graph: MolecularGraph) -> dict:
    n = graph.n_real_atoms
    edges = int(np.count_nonzero(np.triu(graph.bonds, 1)))
    return {"heavy_atoms": n, "cycle_rank": edges - n + 1}


def _class(graph, slot):
    return ORGANIC_VOCABULARY.class_index(
        int(graph.atom_types[slot]),
        int(graph.bonds[slot].sum()),
        int(graph.implicit_h_counts[slot]),
        int(graph.formal_charges[slot]),
    )


def alignments(source_smiles: str, target_smiles: str, config: PathConfig) -> list[dict]:
    """At most one best sampled correspondence per declared MCS comparison mode."""
    source, target = map(Chem.MolFromSmiles, (source_smiles, target_smiles))
    if source is None or target is None:
        raise ValueError("invalid source or target SMILES")
    results = []
    for mode, comparator in (
        ("elements", rdFMCS.AtomCompare.CompareElements),
        ("topology", rdFMCS.AtomCompare.CompareAny),
    ):
        p = rdFMCS.MCSParameters()
        p.Timeout = config.mapping_timeout_seconds
        p.MaximizeBonds = False
        p.AtomTyper = comparator
        p.BondTyper = rdFMCS.BondCompare.CompareAny
        p.AtomCompareParameters.MatchFormalCharge = True
        mcs = rdFMCS.FindMCS([source, target], p)
        query = Chem.MolFromSmarts(mcs.smartsString) if mcs.smartsString else None
        record = {
            "mode": mode,
            "timed_out": mcs.canceled,
            "matched_atoms": mcs.numAtoms,
            "smarts": mcs.smartsString,
        }
        if query is None:
            results.append({**record, "mapping": [], "status": "no_mapping"})
            continue
        limit = config.matches_per_molecule
        left = source.GetSubstructMatches(query, uniquify=False, maxMatches=limit + 1)
        right = target.GetSubstructMatches(query, uniquify=False, maxMatches=limit + 1)

        def rank(mapping):
            atom_changes = sum(
                source.GetAtomWithIdx(a).GetAtomicNum() != target.GetAtomWithIdx(b).GetAtomicNum()
                for a, b in mapping
            )
            target_of = dict(mapping)
            edge_changes = 0
            for a, b in mapping:
                for c, d in mapping:
                    if a >= c:
                        continue
                    x, y = source.GetBondBetweenAtoms(a, c), target.GetBondBetweenAtoms(b, d)
                    edge_changes += (x is None) != (y is None)
                    if x is not None and y is not None:
                        edge_changes += x.GetBondType() != y.GetBondType()
            return atom_changes + edge_changes, tuple(sorted(target_of.items()))

        candidates = [tuple(zip(a, b, strict=True)) for a in left[:limit] for b in right[:limit]]
        mapping = min(candidates, key=rank) if candidates else ()
        results.append(
            {
                **record,
                "mapping": [list(pair) for pair in sorted(mapping)],
                "match_enumeration_truncated": len(left) > limit or len(right) > limit,
                "status": "mapped" if mapping else "no_mapping",
            }
        )
    return results


def _target_layout(source, target, mapping, config):
    # New atoms prefer originally empty slots; retired source slots may be reused.
    target_to_slot = {b: a for a, b in mapping}
    retired = frozenset(range(source.n_real_atoms)) - set(target_to_slot.values())
    free = [i for i in range(source.n_real_atoms, config.slots)] + sorted(retired)
    for b in range(target.n_real_atoms):
        if b not in target_to_slot:
            target_to_slot[b] = free.pop(0)
    aligned = empty_molecular_graph(config.slots)
    for b, slot in target_to_slot.items():
        for field in ("atom_types", "formal_charges", "implicit_h_counts"):
            getattr(aligned, field)[slot] = getattr(target, field)[b]
        for d, other in target_to_slot.items():
            aligned.bonds[slot, other] = target.bonds[b, d]
    return aligned, retired, target_to_slot


def _distance(graph, target, retire):
    active = set(np.flatnonzero(is_element(graph.atom_types))) - retire
    desired = set(np.flatnonzero(is_element(target.atom_types)))
    common = active & desired
    atom = len(retire) + len(desired - active) + len(active - desired)
    atom += sum(_class(graph, i) != _class(target, i) for i in common)
    edges = sum(
        bool(graph.bonds[a, b]) != bool(target.bonds[a, b])
        or (graph.bonds[a, b] and target.bonds[a, b] and graph.bonds[a, b] != target.bonds[a, b])
        for a in common
        for b in common
        if a < b
    )
    # Missing atoms each cost a birth; edges to them are handled after their birth.
    return atom + edges


def _actions(graph, target, retire):
    """Difference-directed proposals only; the production executor decides legality."""
    real = {int(i) for i in np.flatnonzero(is_element(graph.atom_types))}
    wanted = {int(i) for i in np.flatnonzero(is_element(target.atom_types))}
    live = sorted(real & wanted - retire)
    for a in sorted(retire & real):
        yield "atom_delete", AtomDelete(a)
    for a in live:
        cls = _class(target, a)
        if cls is not None and _class(graph, a) != cls:
            yield "atom_restate_semantic", SemanticAtomRestate(a, cls)
    for b in sorted(wanted - real):
        cls = _class(target, b)
        if cls is None:
            continue
        for a in live:
            desired = int(target.bonds[a, b])
            if desired:
                for order in dict.fromkeys((desired, 1)):
                    h = ORGANIC_VOCABULARY.h_count(cls, order, int(target.formal_charges[b]))
                    if h is not None:
                        yield (
                            "atom_insert",
                            AtomInsert(
                                b,
                                int(target.atom_types[b]),
                                int(target.formal_charges[b]),
                                h,
                                ((a, order),),
                            ),
                        )
    remove, add, reorders = [], [], []
    for a in live:
        for b in live:
            if a >= b:
                continue
            old, new = int(graph.bonds[a, b]), int(target.bonds[a, b])
            if old and not new:
                remove.append((a, b))
                yield "cycle_open", CycleOpenEdge(a, b)
            elif new and not old:
                add.append((a, b, new))
                yield "cycle_close", CycleCloseEdge(a, b, new)
            elif old and new and old != new:
                reorders.append(BondOrderChange(a, b, new))
                yield "bond_reorder", BondReorder(a, b, new)
    for a, b in remove:
        for u, v, order in add:
            yield "bond_reroute", BondReroute(a, b, u, v, order)
    if len(reorders) > 1:
        yield "ring_system_restate", RingSystemRestate(tuple(reorders))


def search_mapping(source, target, mapping, config: PathConfig) -> dict:
    goal, retire, target_slots = _target_layout(source, target, mapping, config)
    if not charge_policy_preserved(source, goal):
        return {"status": "mapping_violates_charged_center_policy", "attempts": 0}
    system = editing_v2_semantic_rewrite_system()
    desired = canonical_state_key(target)
    serial = count()
    queue = [(float(_distance(source, goal, retire)), next(serial), source, retire, ())]
    visited, failures = set(), Counter()
    expanded = attempted = 0
    best = None
    while queue and expanded < config.max_expansions:
        _, _, graph, pending, path = heapq.heappop(queue)
        key = (exact_graph_key(graph), pending)
        if key in visited:
            continue
        visited.add(key)
        expanded += 1
        error = _distance(graph, goal, pending)
        if best is None or error < best[0]:
            best = error, graph, path
        current_smiles = canonical_state_key(graph)
        if current_smiles == desired:
            return {
                "status": "witness_found",
                "actions": list(path),
                "expanded": expanded,
                "attempts": attempted,
                "rejections": dict(sorted(failures.items())),
                "target_slots": {str(k): v for k, v in sorted(target_slots.items())},
            }
        if len(path) >= config.max_steps:
            continue
        children = []
        for family, action in _actions(graph, goal, pending):
            attempted += 1
            try:
                product = system.apply(graph, family, action)
            except InvalidRewrite as exc:
                failures[f"{family}:{str(exc).split(':')[0]}"] += 1
                continue
            if not 1 <= product.n_real_atoms <= config.max_active:
                failures["active_atom_limit"] += 1
                continue
            if canonical_state_key(product) == current_smiles:
                failures["nonproductive_alias"] += 1
                continue
            remaining = pending - {action.v} if family == "atom_delete" else pending
            if (exact_graph_key(product), remaining) in visited:
                continue
            actions = path + (encode_action(family, action),)
            score = _distance(product, goal, remaining) + 0.05 * len(actions)
            children.append((score, next(serial), product, remaining, actions))
        for child in sorted(children, key=lambda c: c[:2])[: config.children_per_expansion]:
            heapq.heappush(queue, child)
    return {
        "status": "search_unresolved",
        "expanded": expanded,
        "attempts": attempted,
        "rejections": dict(sorted(failures.items())),
        "best_residual": best[0],
        "best_partial_actions": list(best[2]),
        "best_partial_smiles": canonical_state_key(best[1]),
    }


def replay(source_payload: dict, actions: list[dict], expected_smiles: str) -> list[dict]:
    """Replay saved actions, never rebuild a trajectory state from its SMILES."""
    system = editing_v2_semantic_rewrite_system()
    graph = decode_state(source_payload)
    states = [encode_state(graph)]
    for mark in actions:
        family, action = decode_action(mark)
        graph = system.apply(graph, family, action)
        if not 1 <= graph.n_real_atoms <= 40:
            raise ValueError("witness leaves the non-null 1..40-atom editing support")
        states.append(encode_state(graph))
    if canonical_state_key(graph) != expected_smiles:
        raise ValueError("witness replay endpoint is not the exact supported 2D target")
    return states


def find_path(source_smiles: str, target_smiles: str, config: PathConfig) -> dict:
    try:
        source = smiles_to_molecular_graph(source_smiles)
        target = smiles_to_molecular_graph(target_smiles)
    except MolecularGraphError as exc:
        return {"status": "unsupported_representation", "reason": str(exc)}
    if max(source.n_real_atoms, target.n_real_atoms) > config.max_active:
        return {"status": "unsupported_size"}
    source = pad_molecular_graph(source, config.slots)
    target = pad_molecular_graph(target, config.slots)
    initial, final = graph_stats(source), graph_stats(target)
    dn, dc = (final[k] - initial[k] for k in ("heavy_atoms", "cycle_rank"))
    result = {
        "source_state": encode_state(source),
        "target_2d": canonical_state_key(target),
        "source_stats": initial,
        "target_stats": final,
        "primitive_lower_bound": abs(dn) + max(0, dc),
        "lower_bound_reason": "one-neighbor birth changes atom count but not cycle rank; only closure increases cycle rank",
    }
    if sum(source.formal_charges) != sum(target.formal_charges):
        return {**result, "status": "unreachable_charge_change"}
    mappings = alignments(source_smiles, target_smiles, config)
    attempts = []
    for mapping in mappings:
        if not mapping["mapping"]:
            attempts.append({"mapping": mapping, "status": "no_mapping"})
            continue
        candidate = search_mapping(source, target, mapping["mapping"], config)
        attempts.append({"mapping": mapping, **candidate})
        if candidate["status"] == "witness_found":
            states = replay(result["source_state"], candidate["actions"], result["target_2d"])
            return {
                **result,
                "status": "witness_found",
                "attempts": attempts,
                "actions": candidate["actions"],
                "states": states,
                "witness_steps": len(candidate["actions"]),
                "within_16_edit_witness": len(candidate["actions"]) <= 16,
            }
    return {**result, "status": "search_unresolved", "attempts": attempts}
