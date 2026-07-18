"""Measure useful partial-order concurrency in typed construction teachers."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from compose_v4.data.cnof import load_cnof_corpus_split
from compose_v4.experiments.tracelet_conditional import build_tracelet_path_records
from compose_v4.rewrite.causal_trace import CausalTraceCTMC
from compose_v4.rewrite.kernel import canonical_state_key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("smiles_file", type=Path)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--max-atoms", type=int, default=16)
    parser.add_argument("--train-size", type=int, default=1067)
    parser.add_argument("--validation-size", type=int, default=133)
    parser.add_argument("--test-size", type=int, default=133)
    parser.add_argument("--orders-per-molecule", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.orders_per_molecule <= 0:
        raise ValueError("orders-per-molecule must be positive")

    split = load_cnof_corpus_split(
        args.smiles_file,
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        max_atoms=args.max_atoms,
        seed=args.seed,
        scan_all=True,
    )
    report = {}
    for offset, (name, smiles) in enumerate(
        (
            ("train", split.train),
            ("validation", split.validation),
            ("test", split.test),
        )
    ):
        records = build_tracelet_path_records(
            smiles,
            n_slots=args.max_atoms,
            typed_ring_payloads=True,
        )
        report[name] = _audit(
            records,
            orders_per_molecule=args.orders_per_molecule,
            rng=np.random.default_rng(args.seed + offset),
        )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


def _audit(records, *, orders_per_molecule: int, rng) -> dict[str, object]:
    frontier_histogram = Counter()
    molecules_with_branching = 0
    endpoint_failures = 0
    sampled_nonterminal_ideals = 0
    endpoint_checks = min(64, len(records))
    for record_index, record in enumerate(records):
        causal = CausalTraceCTMC(record.path.trace)
        molecule_branched = False
        for draw in range(orders_per_molecule):
            for progress in range(causal.path_length):
                completed, _ = causal.sample_ideal_indices(progress, rng)
                size = len(causal.frontier(completed))
                frontier_histogram[size] += 1
                sampled_nonterminal_ideals += 1
                molecule_branched |= size > 1
            if draw == 0 and record_index < endpoint_checks:
                _, order, state = causal.sample_ideal(causal.path_length, rng)
                endpoint_failures += int(
                    canonical_state_key(state) != record.target_key
                )
        molecules_with_branching += int(molecule_branched)
    weighted_frontier = sum(
        size * count for size, count in frontier_histogram.items()
    )
    branching_ideals = sum(
        count for size, count in frontier_histogram.items() if size > 1
    )
    return {
        "molecules": len(records),
        "orders_per_molecule": orders_per_molecule,
        "sampled_nonterminal_ideals": sampled_nonterminal_ideals,
        "frontier_size_counts": {
            str(size): count for size, count in sorted(frontier_histogram.items())
        },
        "mean_frontier_size": weighted_frontier / max(sampled_nonterminal_ideals, 1),
        "branching_ideal_fraction": branching_ideals
        / max(sampled_nonterminal_ideals, 1),
        "molecules_with_branching_fraction": molecules_with_branching
        / max(len(records), 1),
        "maximum_sampled_frontier": max(frontier_histogram, default=0),
        "endpoint_failures": endpoint_failures,
        "alternate_order_endpoint_checks": endpoint_checks,
    }


if __name__ == "__main__":
    main()
