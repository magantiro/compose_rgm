"""Audit the legal trace compiler on a newline-delimited SMILES corpus."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from compose_v4.chem.molecular_graph import MolecularGraphError, smiles_to_molecular_graph
from compose_v4.chem.state import is_valid_state
from compose_v4.rewrite.compiler import TraceCompilationError, compile_null_to_target
from compose_v4.rewrite.trace import execute_trace, inverse_step


def _same_arrays(left, right) -> bool:
    return bool(
        np.array_equal(left.atom_types, right.atom_types)
        and np.array_equal(left.formal_charges, right.formal_charges)
        and np.array_equal(left.implicit_h_counts, right.implicit_h_counts)
        and np.array_equal(left.bonds, right.bonds)
    )


def audit(path: Path, *, limit: int, max_atoms: int) -> dict[str, object]:
    counts: Counter[str] = Counter()
    failure_examples: dict[str, list[dict[str, str]]] = {}
    committed_rewrites = 0

    with path.open() as handle:
        for line in handle:
            if counts["read"] >= limit:
                break
            smiles = line.strip().split()[0] if line.strip() else ""
            if not smiles:
                continue
            counts["read"] += 1
            try:
                target = smiles_to_molecular_graph(smiles)
            except (MolecularGraphError, ValueError) as exc:
                counts["parse_or_vocabulary_failure"] += 1
                _remember(failure_examples, "parse_or_vocabulary_failure", smiles, exc)
                continue
            if target.n_real_atoms > max_atoms:
                counts["out_of_scope_size"] += 1
                continue
            counts["eligible"] += 1
            try:
                trace = compile_null_to_target(target)
                endpoint, states = execute_trace(
                    trace.source,
                    trace.steps,
                    return_states=True,
                )
                inverse = tuple(
                    inverse_step(states[i], trace.steps[i])
                    for i in range(len(trace.steps))
                )[::-1]
                restored = execute_trace(
                    endpoint,
                    inverse,
                )
            except (TraceCompilationError, ValueError) as exc:
                counts["compile_or_execute_failure"] += 1
                _remember(failure_examples, "compile_or_execute_failure", smiles, exc)
                continue
            if not _same_arrays(endpoint, target):
                counts["inexact_forward"] += 1
                continue
            if not _same_arrays(restored, trace.source):
                counts["inexact_reverse"] += 1
                continue
            # Every transition has already crossed RewriteSystem.apply, which
            # checks both source and successor with this same predicate. Keep
            # endpoint checks here without redundantly sanitizing all states.
            if not is_valid_state(endpoint) or not is_valid_state(restored):
                counts["invalid_visible_state"] += 1
                continue
            counts["exact_round_trip"] += 1
            committed_rewrites += len(trace.steps) + len(inverse)

    eligible_failures = (
        counts["compile_or_execute_failure"]
        + counts["inexact_forward"]
        + counts["inexact_reverse"]
        + counts["invalid_visible_state"]
    )
    return {
        "corpus": str(path),
        "limit": limit,
        "max_atoms": max_atoms,
        "counts": dict(sorted(counts.items())),
        "eligible_success_rate": (
            counts["exact_round_trip"] / counts["eligible"]
            if counts["eligible"]
            else 0.0
        ),
        "eligible_failures": eligible_failures,
        "committed_forward_and_reverse_rewrites": committed_rewrites,
        "failure_examples": failure_examples,
    }


def _remember(
    examples: dict[str, list[dict[str, str]]],
    category: str,
    smiles: str,
    error: Exception,
) -> None:
    bucket = examples.setdefault(category, [])
    if len(bucket) < 5:
        bucket.append({"smiles": smiles, "error": str(error)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("smiles_file", type=Path)
    parser.add_argument("--limit", type=int, default=10_000)
    parser.add_argument("--max-atoms", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = audit(args.smiles_file, limit=args.limit, max_atoms=args.max_atoms)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
