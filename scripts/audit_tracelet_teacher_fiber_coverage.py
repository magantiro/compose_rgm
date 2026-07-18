"""Verify that every compiled teacher successor is in the production fiber."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.data.cnof import load_cnof_corpus_split
from compose_v4.experiments.tracelet_conditional import (
    canonical_tracelet_representative,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.progress import TraceProgressCTMC
from compose_v4.rewrite.tracelet_compiler import compile_null_to_target_tracelets
from compose_v4.rewrite.tracelet_fiber import (
    enumerate_aromatic_closure_transitions,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("smiles_file", type=Path)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--max-atoms", type=int, default=16)
    parser.add_argument("--train-size", type=int, default=1067)
    parser.add_argument("--validation-size", type=int, default=133)
    parser.add_argument("--test-size", type=int, default=133)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tracelet_teacher_fiber_coverage.json"),
    )
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
    corpus = (*split.train, *split.validation, *split.test)
    family_counts: Counter[str] = Counter()
    missing = []
    compiled_steps_seen = 0
    checked_steps = 0
    for molecule_index, text in enumerate(corpus):
        target = pad_molecular_graph(
            smiles_to_molecular_graph(text),
            args.max_atoms,
        )
        path = TraceProgressCTMC(compile_null_to_target_tracelets(target))
        for progress, step in enumerate(path.trace.steps):
            compiled_steps_seen += 1
            family_counts[step.rule_name] += 1
            if step.rule_name != "ring_system_restate":
                continue
            raw_state = path.states[progress]
            state_key = canonical_state_key(raw_state)
            state = canonical_tracelet_representative(raw_state, state_key)
            transitions = enumerate_aromatic_closure_transitions(state)
            teacher_key = canonical_state_key(path.states[progress + 1])
            successor_keys = {
                transition.successor_key for transition in transitions
            }
            checked_steps += 1
            if teacher_key not in successor_keys:
                missing.append(
                    {
                        "molecule_index": molecule_index,
                        "smiles": text,
                        "progress": progress,
                        "teacher_rule": step.rule_name,
                        "state": state_key,
                        "teacher_successor": teacher_key,
                        "candidate_successors": tuple(sorted(successor_keys)),
                    }
                )
        if (molecule_index + 1) % 100 == 0:
            print(
                json.dumps(
                    {
                        "phase": "audit",
                        "molecules": molecule_index + 1,
                        "compiled_steps": compiled_steps_seen,
                        "closure_steps": checked_steps,
                        "missing": len(missing),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    report = {
        "molecules": len(corpus),
        "compiled_steps_seen": compiled_steps_seen,
        "checked_teacher_steps": checked_steps,
        "covered_teacher_steps": checked_steps - len(missing),
        "coverage_fraction": (
            (checked_steps - len(missing)) / checked_steps if checked_steps else 1.0
        ),
        "teacher_family_counts": dict(sorted(family_counts.items())),
        "missing": missing,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
