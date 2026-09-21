#!/usr/bin/env python3
"""Zero-oracle PMO capability matrix: where autonomous search loses the teacher move.

Phase A of ``pmo_teacher_route_gap_v1`` decomposed the sealed 184-route teacher
corpus and Phase B drove the live proposer for THREE tasks.  This driver extends the
comparison to every PMO task that has teacher routes, and attributes each task's loss
to exactly one stage of the structural pipeline.

The structural action is ``Z = (R, H, alpha, D)``:

    R      retained / released region -- which source atoms survive the route
    H      replacement topology       -- ring and heavy-atom delta, move classes used
    alpha  attachment                 -- final bonds joining created topology to the
                                        retained core, and the retained atoms they land on
    D      dependencies               -- created-handle reuse, dependency edges, the
                                        longest producer->consumer chain

Teachers are used as COUNTERFACTUALS ONLY.  Nothing here injects a teacher endpoint,
copies a route into a policy, or builds a task-specific template; every proposer
number comes from the unmodified production proposal law.

No oracle is constructed and no call is charged.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pmo_teacher_route_gap_v1 import (
    MAX_PRIMITIVES,
    characterize,
    load_routes,
    regions,
    route_objective,
)

from compose_v4.control.dependency_region_program import trace_structure

# ---- Z completion: the attachment factor the Phase-A characterization omits ----


def attachment(route: dict) -> dict:
    """alpha: where created topology joins the retained core, in the FINAL state.

    An attachment bond is a bond of the terminal graph whose two endpoints carry
    tokens of different provenance -- one surviving source atom, one route-created
    atom.  ``anchor_atoms`` counts the distinct retained atoms that carry at least
    one such bond; that is the number of attachment SITES the route commits to.
    """
    states = tuple(route["states"])
    actions = tuple(route["actions"])
    structure = trace_structure(states, actions)
    final_graph = structure["graphs"][-1]
    token_of = structure["final_active_slots"]
    bonds = np.asarray(final_graph.bonds)
    anchors: set[int] = set()
    created_ends: set[int] = set()
    attachment_bonds = 0
    for left, right in zip(*np.nonzero(np.triu(bonds != 0, 1)), strict=True):
        left, right = int(left), int(right)
        left_token, right_token = token_of.get(left), token_of.get(right)
        if left_token is None or right_token is None:
            continue
        kinds = {left_token[0], right_token[0]}
        if kinds != {"source", "created"}:
            continue
        attachment_bonds += 1
        for slot, token in ((left, left_token), (right, right_token)):
            (anchors if token[0] == "source" else created_ends).add(slot)

    # D: longest producer -> consumer chain over created handles.
    incoming: dict[int, list[int]] = defaultdict(list)
    for edge in structure["dependency_edges"]:
        incoming[edge["consumer"]].append(edge["producer"])
    depth: dict[int, int] = {}
    for step in range(len(actions)):
        best = 0
        for producer in incoming.get(step, ()):
            best = max(best, depth.get(producer, 0) + 1)
        depth[step] = best
    return {
        "attachment_bonds": attachment_bonds,
        "anchor_atoms": len(anchors),
        "created_endpoints_attached": len(created_ends),
        "dependency_chain_depth": max(depth.values()) if depth else 0,
        "max_primitive_lag": max(
            (edge["primitive_lag"] for edge in structure["dependency_edges"]), default=0
        ),
    }


def released_region(route: dict) -> dict:
    """R: the released half of the retained/released split, in heavy atoms."""
    features = characterize(route)
    released = features["source_atom_count"] - features["retained_source_atoms"]
    return {
        "source_atoms": features["source_atom_count"],
        "retained_atoms": features["retained_source_atoms"],
        "released_atoms": released,
        "released_fraction": released / features["source_atom_count"]
        if features["source_atom_count"]
        else 0.0,
    }


def describe_route(route: dict) -> dict:
    features = characterize(route)
    alpha = attachment(route)
    region = released_region(route)
    return {
        **features,
        **alpha,
        **region,
        "distinct_move_classes": len(features["rule_counts"]),
        "primary_regions": regions(route, join=True)["component_count"],
        "strict_regions": regions(route, join=False)["component_count"],
    }


def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "median": ordered[len(ordered) // 2],
        "mean": sum(ordered) / len(ordered),
        "max": ordered[-1],
    }


def aggregate(rows: list[dict]) -> dict:
    return {
        "routes": len(rows),
        "primitive_count": _stats([row["primitive_count"] for row in rows]),
        "distinct_move_classes": _stats([row["distinct_move_classes"] for row in rows]),
        "released_atoms": _stats([row["released_atoms"] for row in rows]),
        "released_fraction": _stats([row["released_fraction"] for row in rows]),
        "retained_fraction": _stats([row["retained_fraction"] for row in rows]),
        "delta_heavy_atoms": _stats([row["delta_heavy_atoms"] for row in rows]),
        "delta_rings": _stats(
            [row["delta_rings"] for row in rows if row["delta_rings"] is not None]
        ),
        "attachment_bonds": _stats([row["attachment_bonds"] for row in rows]),
        "anchor_atoms": _stats([row["anchor_atoms"] for row in rows]),
        "dependency_chain_depth": _stats([row["dependency_chain_depth"] for row in rows]),
        "max_primitive_lag": _stats([row["max_primitive_lag"] for row in rows]),
        "created_handles": _stats([row["created_handles"] for row in rows]),
        "strict_regions": _stats([row["strict_regions"] for row in rows]),
        "direction": dict(Counter(row["direction"] for row in rows)),
        "move_class_support": dict(
            sorted(Counter(rule for row in rows for rule in row["rule_counts"]).items())
        ),
        "ring_topology_rate": sum(row["touches_ring_topology"] for row in rows) / len(rows),
        "reused_handle_rate": sum(row["created_handles_reused"] > 0 for row in rows) / len(rows),
        "within_runtime_horizon_rate": sum(
            row["primitive_count"] <= MAX_PRIMITIVES for row in rows
        )
        / len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    options = parser.parse_args()

    corpus = load_routes()
    by_objective: dict[str, list[dict]] = defaultdict(list)
    root_of: dict[str, set[str]] = defaultdict(set)
    # Root ids follow CORPUS order so that every artifact in this family numbers the
    # roots identically; numbering them in task order would silently renumber them.
    root_ids: dict[str, int] = {}
    for route in corpus:
        objective = route_objective(route)
        by_objective[objective].append(route)
        key = json.dumps(route["source_state"], sort_keys=True)
        root_of[objective].add(key)
        root_ids.setdefault(key, len(root_ids))

    tasks = {}
    per_route = {}
    for objective, routes in sorted(by_objective.items()):
        rows = [describe_route(route) for route in routes]
        per_route[objective] = rows
        tasks[objective] = {
            **aggregate(rows),
            "root_ids": sorted(root_ids[key] for key in root_of[objective]),
            "source_smiles": sorted({row["source_smiles"] for row in rows}),
        }

    payload = {
        "schema_version": "pmo_teacher_capability_matrix_teacher_side_v1",
        "charged_oracle_calls": 0,
        "corpus": {
            "path": "diagnostics/pmo_dependency_region_program_v2/attempt_1/"
            "training_dependency_region_corpus.json.gz",
            "routes": len(corpus),
            "distinct_roots": len(root_ids),
        },
        "runtime_maximum_primitives": MAX_PRIMITIVES,
        "tasks": tasks,
    }
    Path(options.out).parent.mkdir(parents=True, exist_ok=True)
    Path(options.out).write_text(json.dumps(payload, sort_keys=True, indent=1) + "\n")
    print(f"wrote {options.out}")


if __name__ == "__main__":
    main()
