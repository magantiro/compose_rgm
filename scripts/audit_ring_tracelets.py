"""Measure ring-family and derived-rewrite frequencies on a C/N/O/F split."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.data.cnof import load_cnof_corpus_split
from compose_v4.eval.ring_taxonomy import ring_taxonomy_report
from compose_v4.rewrite.tracelet_compiler import compile_null_to_target_tracelets
from compose_v4.rewrite.tracelets import CycleInsert, RingEarInsert, RingSystemRestate


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
    partitions = {
        "train": split.train,
        "validation": split.validation,
        "test": split.test,
        "all": (*split.train, *split.validation, *split.test),
    }
    report = {
        "split": {
            "seed": args.seed,
            "max_atoms": args.max_atoms,
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
            "eligible_molecules": split.eligible_molecules,
            "scanned_lines": split.scanned_lines,
        },
        "ring_taxonomy": {
            name: ring_taxonomy_report(tuple(smiles))
            for name, smiles in partitions.items()
        },
        "train_tracelet_events": _tracelet_event_report(
            split.train,
            n_slots=args.max_atoms,
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


def _tracelet_event_report(
    smiles: tuple[str, ...],
    *,
    n_slots: int,
) -> dict[str, object]:
    families: Counter[str] = Counter()
    marks: Counter[str] = Counter()
    visible_steps = 0
    lowered_steps = 0
    for text in smiles:
        target = pad_molecular_graph(smiles_to_molecular_graph(text), n_slots)
        trace = compile_null_to_target_tracelets(target)
        visible_steps += len(trace.steps)
        lowered_steps += int(trace.metadata["micro_lowered_steps"])
        for step in trace.steps:
            families[step.rule_name] += 1
            action = step.action
            if isinstance(action, CycleInsert):
                marks[f"cycle_span_{len(action.atoms)}"] += 1
            elif isinstance(action, RingEarInsert):
                ear_kind = "petal" if action.a == action.b else "open"
                marks[f"{ear_kind}_ear_span_{len(action.atoms)}"] += 1
            elif isinstance(action, RingSystemRestate):
                marks[f"ring_restate_bonds_{len(action.changes)}"] += 1

    total_events = sum(families.values())
    return {
        "molecule_count": len(smiles),
        "visible_steps": visible_steps,
        "micro_lowered_steps": lowered_steps,
        "path_compression": float(lowered_steps / max(visible_steps, 1)),
        "family_counts": dict(sorted(families.items())),
        "family_fractions": {
            family: float(count / total_events)
            for family, count in sorted(families.items())
        },
        "macro_mark_counts": dict(sorted(marks.items())),
    }


if __name__ == "__main__":
    main()
