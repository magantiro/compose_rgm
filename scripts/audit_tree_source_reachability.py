#!/usr/bin/env python
"""Measure exact valid-state reachability from independent carbon-tree sources."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median

import numpy as np

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.rewrite.trace import execute_trace
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("smiles", type=Path)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--trees-per-target", type=int, default=4)
    parser.add_argument("--max-atoms", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument(
        "--transport-strategy",
        choices=("primitive", "reroute"),
        default="primitive",
    )
    parser.add_argument(
        "--source-size-mode",
        choices=("matched", "uniform"),
        default="matched",
        help="uniform samples source N independently from 1..max-atoms",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.limit <= 0 or args.trees_per_target <= 0 or args.max_atoms <= 0:
        raise ValueError("limit, trees-per-target, and max-atoms must be positive")
    if args.transport_strategy == "reroute" and args.source_size_mode != "matched":
        raise ValueError("bond-reroute ablation currently requires matched source size")

    rng = np.random.default_rng(args.seed)
    targets = []
    parse_failures = 0
    with args.smiles.open("r", encoding="utf-8") as handle:
        for line in handle:
            token = line.strip().split()[0] if line.strip() else ""
            if not token:
                continue
            try:
                molecule = smiles_to_molecular_graph(token)
            except Exception:
                parse_failures += 1
                continue
            if molecule.n_real_atoms > args.max_atoms:
                continue
            targets.append((token, pad_molecular_graph(molecule, args.max_atoms)))
            if len(targets) >= args.limit:
                break

    attempts = []
    target_successes = Counter()
    failure_types = Counter()
    for target_index, (smiles, target) in enumerate(targets):
        if args.source_size_mode == "matched":
            prior = DegreeBoundedCarbonTreePrior(sizes=(target.n_real_atoms,))
        else:
            prior = DegreeBoundedCarbonTreePrior(
                sizes=tuple(range(1, args.max_atoms + 1))
            )
        for tree_index in range(args.trees_per_target):
            source = prior.sample(rng, n_slots=args.max_atoms)
            try:
                trace = compile_carbon_tree_to_target(
                    source,
                    target,
                    use_bond_reroute=args.transport_strategy == "reroute",
                )
                endpoint = execute_trace(trace.source, trace.steps)
                exact = bool(
                    np.array_equal(endpoint.atom_types, target.atom_types)
                    and np.array_equal(endpoint.formal_charges, target.formal_charges)
                    and np.array_equal(
                        endpoint.implicit_h_counts,
                        target.implicit_h_counts,
                    )
                    and np.array_equal(endpoint.bonds, target.bonds)
                )
                if not exact:
                    raise RuntimeError("compiler endpoint mismatch")
                target_successes[target_index] += 1
                attempts.append(
                    {
                        "target_index": target_index,
                        "tree_index": tree_index,
                        "smiles": smiles,
                        "success": True,
                        **trace.metadata,
                    }
                )
            except Exception as exc:
                failure = type(exc).__name__
                failure_types[failure] += 1
                attempts.append(
                    {
                        "target_index": target_index,
                        "tree_index": tree_index,
                        "smiles": smiles,
                        "success": False,
                        "failure_type": failure,
                        "failure": str(exc),
                    }
                )

    successes = [item for item in attempts if item["success"]]
    path_lengths = [int(item["visible_steps"]) for item in successes]
    reroutes = [int(item["bond_reroute_steps"]) for item in successes]
    closures = [int(item["scalar_closure_steps"]) for item in successes]
    atom_deletes = [int(item["atom_delete_steps"]) for item in successes]
    atom_inserts = [int(item["atom_insert_steps"]) for item in successes]
    size_changes = [
        int(item["target_heavy_atoms"]) - int(item["source_heavy_atoms"])
        for item in successes
    ]
    summary = {
        "seed": args.seed,
        "transport_strategy": args.transport_strategy,
        "source_size_mode": args.source_size_mode,
        "targets": len(targets),
        "trees_per_target": args.trees_per_target,
        "attempts": len(attempts),
        "successful_attempts": len(successes),
        "attempt_reachability": len(successes) / max(len(attempts), 1),
        "targets_reachable_from_every_tree": sum(
            target_successes[index] == args.trees_per_target
            for index in range(len(targets))
        ),
        "targets_reachable_from_at_least_one_tree": sum(
            target_successes[index] > 0 for index in range(len(targets))
        ),
        "mean_visible_steps": float(np.mean(path_lengths)) if path_lengths else None,
        "median_visible_steps": float(median(path_lengths)) if path_lengths else None,
        "mean_bond_reroutes": float(np.mean(reroutes)) if reroutes else None,
        "mean_scalar_closures": float(np.mean(closures)) if closures else None,
        "mean_atom_deletes": float(np.mean(atom_deletes)) if atom_deletes else None,
        "mean_atom_inserts": float(np.mean(atom_inserts)) if atom_inserts else None,
        "mean_absolute_size_change": (
            float(np.mean(np.abs(size_changes))) if size_changes else None
        ),
        "parse_failures_before_limit": parse_failures,
        "failure_types": dict(failure_types),
        "ring_commitment_statuses": sorted(
            {str(item["ring_commitment_status"]) for item in successes}
        ),
    }
    payload = {"summary": summary, "attempts": attempts}
    rendered = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
