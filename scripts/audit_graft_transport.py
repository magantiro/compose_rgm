"""Audit atom-retaining Graft transport against primitive delete/regrow paths."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from compose_v4.chem.molecular_graph import (
    is_element,
    smiles_to_molecular_graph,
)
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.chem.state import is_connected_or_null, is_valid_state, pad_molecular_graph
from compose_v4.data.cnof import load_cnof_corpus_split
from compose_v4.rewrite.trace import execute_trace
from compose_v4.rewrite.tracelet_compiler import compile_null_to_target_tracelets
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _edge_set(state) -> set[tuple[int, int]]:
    real = tuple(int(v) for v in np.flatnonzero(is_element(state.atom_types)))
    return {
        (int(a), int(b))
        for offset, a in enumerate(real)
        for b in real[offset + 1 :]
        if int(state.bonds[a, b]) != 0
    }


def _same_state(left, right) -> bool:
    return all(
        np.array_equal(getattr(left, name), getattr(right, name))
        for name in ("atom_types", "formal_charges", "implicit_h_counts", "bonds")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("smiles_file", type=Path)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--max-atoms", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/graft_transport_audit.json"),
    )
    args = parser.parse_args()
    if args.limit <= 0 or args.max_atoms <= 0:
        raise ValueError("limit and max-atoms must be positive")

    split = load_cnof_corpus_split(
        args.smiles_file,
        train_size=args.limit,
        validation_size=0,
        test_size=0,
        max_atoms=args.max_atoms,
        seed=args.seed,
        scan_all=True,
    )
    targets = tuple(
        pad_molecular_graph(smiles_to_molecular_graph(text), args.max_atoms)
        for text in split.train
    )
    proposals = tuple(
        compile_null_to_target_tracelets(target, typed_ring_payloads=True)
        for target in targets
    )
    catalog = build_typed_ring_catalog(
        proposals,
        max_cycle_templates=512,
        max_attach_templates=512,
        max_ear_templates=512,
    )
    prior = DegreeBoundedCarbonTreePrior.from_size_counts(
        dict(Counter(target.n_real_atoms for target in targets))
    )
    rng = np.random.default_rng(args.seed + 1)

    primitive_lengths: list[float] = []
    graft_lengths: list[float] = []
    retained_edges: list[float] = []
    retained_fractions: list[float] = []
    compression: list[float] = []
    compile_seconds = defaultdict(float)
    action_counts = {"primitive": Counter(), "graft": Counter()}
    planner_counts = Counter()
    planner_lengths = defaultdict(list)
    failures = []
    cohort = defaultdict(lambda: {"count": 0, "graft_steps": []})

    for index, target in enumerate(targets):
        source = prior.sample_size(
            rng,
            n_slots=args.max_atoms,
            size=target.n_real_atoms,
        )
        try:
            started = perf_counter()
            primitive = compile_carbon_tree_to_target(
                source,
                target,
                ring_catalog=catalog,
            )
            compile_seconds["primitive"] += perf_counter() - started
            started = perf_counter()
            graft = compile_carbon_tree_to_target(
                source,
                target,
                use_bond_reroute=True,
                align_source=True,
                ring_catalog=catalog,
            )
            compile_seconds["graft"] += perf_counter() - started
            endpoint, states = execute_trace(
                graft.source,
                graft.steps,
                return_states=True,
            )
            if not _same_state(endpoint, target):
                raise RuntimeError("Graft endpoint does not equal target")
            if not all(
                is_valid_state(state) and is_connected_or_null(state)
                for state in states
            ):
                raise RuntimeError("Graft trace left the valid connected state space")
            if any(
                step.rule_name in {"atom_insert", "atom_delete"}
                for step in graft.steps
            ):
                raise RuntimeError("size-matched Graft trace changed atom count")
        except Exception as error:  # audit records the exact compiler surface
            failures.append(
                {
                    "index": index,
                    "target_atoms": target.n_real_atoms,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue

        primitive_lengths.append(float(len(primitive.steps)))
        graft_lengths.append(float(len(graft.steps)))
        compression.append(float(len(primitive.steps) - len(graft.steps)))
        action_counts["primitive"].update(step.rule_name for step in primitive.steps)
        action_counts["graft"].update(step.rule_name for step in graft.steps)
        planner = str(graft.metadata["graft_planner"])
        planner_counts[planner] += 1
        planner_lengths[planner].append(float(len(graft.steps)))
        source_edges = _edge_set(graft.source)
        target_edges = _edge_set(target)
        retained = len(source_edges & target_edges)
        retained_edges.append(float(retained))
        retained_fractions.append(float(retained / max(len(source_edges), 1)))
        cycle_rank = max(len(target_edges) - target.n_real_atoms + 1, 0)
        label = "acyclic" if cycle_rank == 0 else ("monocyclic" if cycle_rank == 1 else "polycyclic")
        cohort[label]["count"] += 1
        cohort[label]["graft_steps"].append(float(len(graft.steps)))

    succeeded = len(graft_lengths)
    report = {
        "input": {
            "smiles_file": str(args.smiles_file),
            "requested": args.limit,
            "max_atoms": args.max_atoms,
            "seed": args.seed,
            "scanned_lines": split.scanned_lines,
            "eligible_molecules": split.eligible_molecules,
        },
        "catalog": catalog.statistics(),
        "attempted": len(targets),
        "succeeded": succeeded,
        "failed": len(failures),
        "success_fraction": float(succeeded / max(len(targets), 1)),
        "primitive_path_steps": _quantiles(primitive_lengths) if primitive_lengths else {},
        "graft_path_steps": _quantiles(graft_lengths) if graft_lengths else {},
        "steps_saved": _quantiles(compression) if compression else {},
        "aligned_source_edges_retained": _quantiles(retained_edges) if retained_edges else {},
        "aligned_source_edge_fraction_retained": (
            _quantiles(retained_fractions) if retained_fractions else {}
        ),
        "compile_seconds": dict(compile_seconds),
        "compile_molecules_per_second": {
            name: float(succeeded / max(seconds, 1e-12))
            for name, seconds in compile_seconds.items()
        },
        "action_counts": {
            name: dict(sorted(counts.items()))
            for name, counts in action_counts.items()
        },
        "graft_planners": {
            name: {
                "count": int(planner_counts[name]),
                "fraction": float(planner_counts[name] / max(succeeded, 1)),
                "graft_path_steps": _quantiles(planner_lengths[name]),
            }
            for name in sorted(planner_counts)
        },
        "cohorts": {
            name: {
                "count": values["count"],
                "graft_path_steps": _quantiles(values["graft_steps"]),
            }
            for name, values in sorted(cohort.items())
        },
        "failures": failures[:50],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
