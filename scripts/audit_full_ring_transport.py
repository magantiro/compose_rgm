"""Audit closure-free full-ring transport from an independent carbon-tree prior."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
from multiprocessing import get_context
from pathlib import Path
from time import monotonic

import numpy as np

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.chem.state import is_connected_or_null, is_valid_state, pad_molecular_graph
from compose_v4.data.cnof import load_cnof_corpus_split
from compose_v4.rewrite.trace import execute_trace
from compose_v4.rewrite.tracelets import RingSystemGrow
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target


@dataclass(frozen=True)
class _Task:
    index: int
    smiles: str
    max_atoms: int
    prior_sizes: tuple[int, ...]
    prior_probabilities: tuple[float, ...]
    seed: int


def _same_state(left, right) -> bool:
    return all(
        np.array_equal(getattr(left, field), getattr(right, field))
        for field in ("atom_types", "formal_charges", "implicit_h_counts", "bonds")
    )


def _audit_one(task: _Task) -> dict[str, object]:
    try:
        target = pad_molecular_graph(
            smiles_to_molecular_graph(task.smiles),
            task.max_atoms,
        )
        prior = DegreeBoundedCarbonTreePrior(
            sizes=task.prior_sizes,
            probabilities=task.prior_probabilities,
        )
        source = prior.sample(
            np.random.default_rng(task.seed + task.index),
            n_slots=task.max_atoms,
        )
        trace = compile_carbon_tree_to_target(
            source,
            target,
            use_bond_reroute=True,
            flexible_size=True,
        )
        endpoint, states = execute_trace(
            trace.source,
            trace.steps,
            return_states=True,
        )
        if not _same_state(endpoint, target):
            raise RuntimeError("compiled trace missed its target")
        if not all(is_valid_state(state) for state in states):
            raise RuntimeError("compiled trace exposed an invalid state")
        if not all(is_connected_or_null(state) for state in states):
            raise RuntimeError("compiled trace exposed a disconnected state")
        rule_counts = Counter(step.rule_name for step in trace.steps)
        if rule_counts["ring_ear_insert"] or rule_counts["bond_insert"]:
            raise RuntimeError("transport exposed a model-level closure action")
        macros = tuple(
            step.action
            for step in trace.steps
            if isinstance(step.action, RingSystemGrow)
        )
        return {
            "ok": True,
            "source_atoms": source.n_real_atoms,
            "target_atoms": target.n_real_atoms,
            "steps": len(trace.steps),
            "rules": dict(rule_counts),
            "ring_systems": tuple(
                {
                    "topology_class": action.topology_class,
                    "atoms": len(action.system_atoms),
                    "cycle_rank_delta": len(action.bond_insertions),
                    "aromatic_bonds": len(action.aromatic_edges),
                    "new_atoms": len(action.atom_insertions),
                }
                for action in macros
            ),
        }
    except Exception as error:  # pragma: no cover - summarized by the audit report
        return {
            "ok": False,
            "index": task.index,
            "smiles": task.smiles,
            "error_type": type(error).__name__,
            "error": str(error),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("smiles_file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--max-atoms", type=int, default=40)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()
    if min(args.samples, args.max_atoms, args.workers) <= 0:
        raise ValueError("samples, max atoms, and workers must be positive")

    split = load_cnof_corpus_split(
        args.smiles_file,
        train_size=args.samples,
        validation_size=0,
        test_size=0,
        max_atoms=args.max_atoms,
        seed=args.seed,
        scan_all=False,
        workers=0,
    )
    sizes = Counter(
        pad_molecular_graph(
            smiles_to_molecular_graph(smiles),
            args.max_atoms,
        ).n_real_atoms
        for smiles in split.train
    )
    prior_sizes = tuple(sorted(sizes))
    prior_probabilities = tuple(sizes[size] / len(split.train) for size in prior_sizes)
    tasks = tuple(
        _Task(
            index=index,
            smiles=smiles,
            max_atoms=args.max_atoms,
            prior_sizes=prior_sizes,
            prior_probabilities=prior_probabilities,
            seed=args.seed,
        )
        for index, smiles in enumerate(split.train)
    )
    started = monotonic()
    if args.workers == 1:
        results = tuple(_audit_one(task) for task in tasks)
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=get_context("spawn"),
        ) as executor:
            results = tuple(executor.map(_audit_one, tasks, chunksize=8))

    successes = tuple(result for result in results if result["ok"])
    failures = tuple(result for result in results if not result["ok"])
    rule_counts: Counter[str] = Counter()
    topology_counts: Counter[str] = Counter()
    system_size_counts: Counter[int] = Counter()
    cycle_rank_counts: Counter[int] = Counter()
    aromatic_systems = 0
    new_atom_systems = 0
    for result in successes:
        rule_counts.update(result["rules"])
        for system in result["ring_systems"]:
            topology_counts[str(system["topology_class"])] += 1
            system_size_counts[int(system["atoms"])] += 1
            cycle_rank_counts[int(system["cycle_rank_delta"])] += 1
            aromatic_systems += int(system["aromatic_bonds"] > 0)
            new_atom_systems += int(system["new_atoms"] > 0)
    step_counts = np.asarray(
        [int(result["steps"]) for result in successes],
        dtype=np.float64,
    )
    report = {
        "audit": "independent_tree_to_atomic_ring_system_v1",
        "samples": len(tasks),
        "successes": len(successes),
        "failures": len(failures),
        "success_fraction": len(successes) / len(tasks),
        "seconds": monotonic() - started,
        "workers": args.workers,
        "source_size_prior": {
            "sizes": prior_sizes,
            "probabilities": prior_probabilities,
            "independent_of_target": True,
        },
        "visible_rule_counts": dict(sorted(rule_counts.items())),
        "ring_topology_counts": dict(sorted(topology_counts.items())),
        "ring_system_size_counts": dict(sorted(system_size_counts.items())),
        "ring_cycle_rank_counts": dict(sorted(cycle_rank_counts.items())),
        "aromatic_ring_systems": aromatic_systems,
        "size_growing_ring_systems": new_atom_systems,
        "visible_path_steps": {
            "mean": float(step_counts.mean()) if len(step_counts) else None,
            "p50": float(np.quantile(step_counts, 0.5)) if len(step_counts) else None,
            "p95": float(np.quantile(step_counts, 0.95)) if len(step_counts) else None,
            "max": int(step_counts.max()) if len(step_counts) else None,
        },
        "first_failures": failures[:20],
        "invariants": {
            "model_level_ring_ear_count": rule_counts["ring_ear_insert"],
            "model_level_scalar_bond_insert_count": rule_counts["bond_insert"],
            "all_visible_states_valid": not failures,
            "all_visible_states_connected_or_null": not failures,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
