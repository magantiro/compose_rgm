#!/usr/bin/env python3
"""PMO route archaeology: teacher dependency-region programs vs the live proposer.

The T4 analogue (``tools/t4_dependency_region_audit.py`` plus
``scripts/t4_teacher_program_factors.py``) established a template: decompose exact
winning traces into dependency regions, characterize them, then ask which factor of
the proposal law destroys the probability of reproducing them.  This script runs the
same two passes for PMO against the sealed 184-route exact corpus.

Nothing here charges an oracle call.  Phase B drives the real
``PmoPopulationController.propose_batch`` with an injected free scorer, exactly as
``scripts/pmo_free_oracle_smoke.py`` does, so the proposal law is the production one
while the objective is a deterministic hash.

Region counts are reported under BOTH union rules, because they answer different
questions:
  primary (join_lifetime_neighbors=True)  -- edits on bonded atoms merge; this is the
      runtime representation's own rule and is structurally coarse on a connected
      drug-like molecule.
  strict  (join_lifetime_neighbors=False) -- only operand overlap and created-handle
      dataflow merge; this is the honest count of independently schedulable regions.
Quoting only the primary count would overstate the collapse.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from compose_v4.chem.molecular_graph import is_element, molecular_graph_to_smiles
from compose_v4.control.dependency_region_program import (
    DependencyRegionConfig,
    dependency_region_program,
    dependency_region_summary,
    trace_structure,
)
from compose_v4.control.edit_program import extract_program
from compose_v4.control.edit_program_graph import compile_program_graph, program_size_profile
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]
CORPUS = (
    ROOT
    / "diagnostics/pmo_dependency_region_program_v2/attempt_1"
    / "training_dependency_region_corpus.json.gz"
)
MAX_REGIONS = 8
MAX_PRIMITIVES = 32


# ---- corpus ----------------------------------------------------------------


def load_routes() -> list[dict]:
    with gzip.open(CORPUS) as handle:
        payload = json.load(handle)["payload"]
    if payload["schema_version"] != "pmo_dependency_region_training_corpus_v2":
        raise ValueError("unexpected PMO dependency-region corpus schema")
    return payload["routes"]


def route_objective(route: dict) -> str:
    """The concrete PMO objective, from member provenance, not the coarse family."""
    tasks = {member["task"] for member in route["members"]}
    return min(tasks) if len(tasks) == 1 else "|".join(sorted(tasks))


# ---- per-route structural characterization ---------------------------------


def _ring_count(smiles: str) -> int | None:
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else mol.GetRingInfo().NumRings()


def characterize(route: dict) -> dict:
    """Retention, size/topology direction and handle reuse for one exact route."""
    states = tuple(route["states"])
    actions = tuple(route["actions"])
    source = decode_state(route["source_state"])
    final = decode_state(states[-1])

    structure = trace_structure(states, actions)
    initial_tokens = set(structure["initial_token_slots"])
    surviving = {
        token for token in structure["final_active_slots"].values() if token[0] == "source"
    }
    retained_fraction = len(surviving) / len(initial_tokens) if initial_tokens else 0.0

    source_atoms = int(is_element(source.atom_types).sum())
    final_atoms = int(is_element(final.atom_types).sum())
    rules = Counter(str(action["executor_rule"]) for action in actions)

    source_smiles = molecular_graph_to_smiles(source)
    final_smiles = route["terminal_endpoint"]
    source_rings, final_rings = _ring_count(source_smiles), _ring_count(final_smiles)

    inserted, deleted = rules.get("atom_insert", 0), rules.get("atom_delete", 0)
    # Direction is a description of the realized edit, not a policy label.
    if retained_fraction < 0.5:
        direction = "remodel"
    elif inserted and deleted:
        direction = "replace"
    elif inserted:
        direction = "grow"
    elif deleted:
        direction = "prune"
    else:
        direction = "restate_only"

    return {
        "source_smiles": source_smiles,
        "terminal_endpoint": final_smiles,
        "primitive_count": len(actions),
        "source_heavy_atoms": source_atoms,
        "final_heavy_atoms": final_atoms,
        "delta_heavy_atoms": final_atoms - source_atoms,
        "retained_source_atoms": len(surviving),
        "source_atom_count": len(initial_tokens),
        "retained_fraction": retained_fraction,
        "source_rings": source_rings,
        "final_rings": final_rings,
        "delta_rings": (
            None if source_rings is None or final_rings is None else final_rings - source_rings
        ),
        "direction": direction,
        "rule_counts": dict(sorted(rules.items())),
        "atoms_inserted": inserted,
        "atoms_deleted": deleted,
        "touches_ring_topology": bool(
            rules.get("cycle_open", 0)
            or rules.get("cycle_close", 0)
            or rules.get("ring_system_restate", 0)
        ),
        "created_handles": len(structure["created_handles"]),
        "created_handles_reused": sum(
            bool(handle["consumers"]) for handle in structure["created_handles"]
        ),
        "created_dependency_edges": len(structure["dependency_edges"]),
    }


def program_geometry(route: dict) -> dict:
    """Independent re-derivation through the required EditProgram machinery.

    ``extract_program`` is given one stage PER PRIMITIVE so that
    ``compile_program_graph`` can expose the dataflow between them; a single stage
    would yield one block and no edges, which measures nothing.
    """
    source = decode_state(route["source_state"])
    states, actions = route["states"], route["actions"]
    stages = [
        {
            "name": f"step_{index}",
            "actions": [action],
            "states": [states[index], states[index + 1]],
            "endpoint": canonical_state_key(decode_state(states[index + 1])),
        }
        for index, action in enumerate(actions)
    ]
    program, roots = extract_program(source, stages)
    graph = compile_program_graph(program)
    profile = program_size_profile(graph, int(is_element(source.atom_types).sum()))

    adjacency: dict[int, set[int]] = defaultdict(set)
    for left, right in graph.dependencies:
        adjacency[left].add(right)
        adjacency[right].add(left)
    seen, blocks = set(), []
    for node in range(len(program.blocks)):
        if node in seen:
            continue
        stack, component = [node], []
        seen.add(node)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current] - seen:
                seen.add(neighbor)
                stack.append(neighbor)
        blocks.append(sorted(component))
    return {
        "program_id": program.program_id,
        "input_roots": len(roots),
        "marks": len(program.marks),
        "dataflow_dependencies": len(graph.dependencies),
        "operand_conflicts": len(graph.conflicts),
        "dataflow_components": len(blocks),
        "peak_heavy_atoms": profile["peak_heavy_atoms"],
        "delta_heavy_atoms": profile["delta_heavy_atoms"],
    }


def regions(route: dict, *, join: bool) -> dict:
    config = DependencyRegionConfig(
        runtime_maximum_primitives=MAX_PRIMITIVES,
        runtime_maximum_components=MAX_REGIONS,
        join_lifetime_neighbors=join,
    )
    return dependency_region_program(tuple(route["states"]), tuple(route["actions"]), config)


def _compression(summary: dict) -> dict:
    routes, distribution = summary["routes"], summary["component_count_distribution"]

    def at_most(maximum: int) -> float:
        return (
            sum(count for value, count in distribution.items() if int(value) <= maximum) / routes
        )

    return {
        "primitives_per_region": summary["primitive_transitions"] / summary["components"],
        "routes_at_most_1_region": at_most(1),
        "routes_at_most_2_regions": at_most(2),
        "routes_at_most_4_regions": at_most(4),
        "routes_at_most_8_regions": at_most(8),
    }


def _five(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "n": int(array.size),
        "minimum": float(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "maximum": float(array.max()),
    }


def phase_a() -> dict:
    routes = load_routes()
    rows = []
    for route in routes:
        primary = regions(route, join=True)
        strict = regions(route, join=False)
        rows.append(
            {
                "trace_identity": route["trace_identity"],
                "objective": route_objective(route),
                "task_family": route["task_family"],
                "test_fold": route["test_fold"],
                "characterization": characterize(route),
                "geometry": program_geometry(route),
                "primary_regions": primary["component_count"],
                "strict_regions": strict["component_count"],
                "dependency_region_program": primary,
                "strict_region_program": strict,
            }
        )

    primary_summary = dependency_region_summary(
        [{"dependency_region_program": row["dependency_region_program"]} for row in rows]
    )
    strict_summary = dependency_region_summary(
        [{"dependency_region_program": row["strict_region_program"]} for row in rows]
    )

    by_objective: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_objective[row["objective"]].append(row)

    objectives = {}
    for objective, group in sorted(by_objective.items()):
        chars = [row["characterization"] for row in group]
        objectives[objective] = {
            "routes": len(group),
            "task_families": dict(Counter(row["task_family"] for row in group)),
            "primitive_count": _five([c["primitive_count"] for c in chars]),
            "primary_regions": _five([row["primary_regions"] for row in group]),
            "strict_regions": _five([row["strict_regions"] for row in group]),
            "dataflow_components": _five([row["geometry"]["dataflow_components"] for row in group]),
            "retained_fraction": _five([c["retained_fraction"] for c in chars]),
            "delta_heavy_atoms": _five([c["delta_heavy_atoms"] for c in chars]),
            "delta_rings": _five(
                [c["delta_rings"] for c in chars if c["delta_rings"] is not None]
            ),
            "direction": dict(Counter(c["direction"] for c in chars)),
            "touches_ring_topology": sum(c["touches_ring_topology"] for c in chars),
            "routes_with_reused_created_handles": sum(
                c["created_handles_reused"] > 0 for c in chars
            ),
            "created_dependency_edges_total": sum(c["created_dependency_edges"] for c in chars),
            "rule_totals": dict(
                sorted(
                    Counter(
                        rule
                        for c in chars
                        for rule, count in c["rule_counts"].items()
                        for _ in range(count)
                    ).items()
                )
            ),
        }

    return {
        "corpus": {
            "path": str(CORPUS.relative_to(ROOT)),
            "routes": len(rows),
            "objectives": dict(Counter(row["objective"] for row in rows)),
            "task_families": dict(Counter(row["task_family"] for row in rows)),
        },
        "region_summary_primary": primary_summary,
        "region_summary_strict": strict_summary,
        "compression_primary": _compression(primary_summary),
        "compression_strict": _compression(strict_summary),
        "by_objective": objectives,
        "routes": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--phase", default="a", choices=["a"])
    options = parser.parse_args()
    payload = phase_a()
    Path(options.out).parent.mkdir(parents=True, exist_ok=True)
    Path(options.out).write_text(json.dumps(payload, sort_keys=True, indent=1) + "\n")
    print(json.dumps(payload["compression_primary"], indent=1))
    print(json.dumps(payload["compression_strict"], indent=1))


if __name__ == "__main__":
    main()
