"""Audit the structural ring transactions induced by the typed compiler."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from compose_v4.data.cnof import load_cnof_corpus_split
from compose_v4.experiments.tracelet_conditional import build_tracelet_path_records
from compose_v4.rewrite.ring_junctions import describe_ring_transaction
from compose_v4.rewrite.tracelets import CycleAttach, CycleInsert, RingEarInsert


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("smiles_file", type=Path)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--max-atoms", type=int, default=16)
    parser.add_argument("--train-size", type=int, default=1067)
    parser.add_argument("--validation-size", type=int, default=133)
    parser.add_argument("--test-size", type=int, default=133)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

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
    for name, smiles in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        records = build_tracelet_path_records(
            smiles,
            n_slots=args.max_atoms,
            typed_ring_payloads=True,
        )
        report[name] = _summarize(records)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


def _summarize(records) -> dict[str, object]:
    kinds = Counter()
    sizes = Counter()
    joint = Counter()
    distances = Counter()
    for record in records:
        for index, step in enumerate(record.path.trace.steps):
            if not isinstance(step.action, (CycleInsert, CycleAttach, RingEarInsert)):
                continue
            descriptor = describe_ring_transaction(
                record.path.states[index],
                step.action,
            )
            kinds[descriptor.kind] += 1
            sizes[descriptor.nominal_ring_size] += 1
            joint[(descriptor.kind, descriptor.nominal_ring_size)] += 1
            distances[descriptor.anchor_distance] += 1
    total = sum(kinds.values())
    return {
        "molecules": len(records),
        "ring_events": total,
        "junction_counts": dict(sorted(kinds.items())),
        "junction_fractions": {
            key: value / max(total, 1) for key, value in sorted(kinds.items())
        },
        "nominal_ring_size_counts": {
            str(key): value for key, value in sorted(sizes.items())
        },
        "anchor_distance_counts": {
            str(key): value for key, value in sorted(distances.items())
        },
        "junction_size_counts": {
            f"{kind}:{size}": value
            for (kind, size), value in sorted(joint.items())
        },
    }


if __name__ == "__main__":
    main()
