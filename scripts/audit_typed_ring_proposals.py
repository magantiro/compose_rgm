"""Audit whether bounded typed ring proposals generalize beyond the train split."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from compose_v4.data.cnof import load_cnof_corpus_split
from compose_v4.experiments.tracelet_conditional import build_tracelet_path_records
from compose_v4.rewrite.tracelets import CycleAttach, CycleInsert, RingEarInsert
from compose_v4.rewrite.typed_ring_catalog import (
    attach_template,
    cycle_template,
    ear_template,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("smiles_file", type=Path)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--max-atoms", type=int, default=16)
    parser.add_argument("--train-size", type=int, default=1067)
    parser.add_argument("--validation-size", type=int, default=133)
    parser.add_argument("--test-size", type=int, default=133)
    parser.add_argument("--limits", type=int, nargs="+", default=(8, 16, 32, 64, 128))
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
    records = {
        name: build_tracelet_path_records(
            smiles,
            n_slots=args.max_atoms,
            typed_ring_payloads=True,
        )
        for name, smiles in (
            ("train", split.train),
            ("validation", split.validation),
            ("test", split.test),
        )
    }
    train_cycles, train_attachments, train_ears = _template_counts(records["train"])
    ranked_cycles = tuple(
        template
        for template, _ in sorted(
            train_cycles.items(), key=lambda item: (-item[1], item[0])
        )
    )
    ranked_ears = tuple(
        template
        for template, _ in sorted(
            train_ears.items(), key=lambda item: (-item[1], item[0])
        )
    )
    ranked_attachments = tuple(
        template
        for template, _ in sorted(
            train_attachments.items(), key=lambda item: (-item[1], item[0])
        )
    )
    report = {
        "unique_train_templates": {
            "cycle": len(ranked_cycles),
            "attachment": len(ranked_attachments),
            "ear": len(ranked_ears),
        },
        "partition_events": {
            name: _event_summary(partition) for name, partition in records.items()
        },
        "coverage_by_limit": {},
    }
    for limit in sorted(set(args.limits)):
        cycle_support = frozenset(ranked_cycles[:limit])
        attachment_support = frozenset(ranked_attachments[:limit])
        ear_support = frozenset(ranked_ears[:limit])
        report["coverage_by_limit"][str(limit)] = {
            name: _coverage(
                partition,
                cycle_support,
                attachment_support,
                ear_support,
            )
            for name, partition in records.items()
        }

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


def _template_counts(records) -> tuple[Counter, Counter, Counter]:
    cycles = Counter()
    attachments = Counter()
    ears = Counter()
    for record in records:
        for step in record.path.trace.steps:
            if isinstance(step.action, CycleInsert):
                cycles[cycle_template(step.action)] += 1
            elif isinstance(step.action, CycleAttach):
                attachments[attach_template(step.action)] += 1
            elif isinstance(step.action, RingEarInsert):
                ears[ear_template(step.action)] += 1
    return cycles, attachments, ears


def _event_summary(records) -> dict[str, object]:
    families = Counter(
        step.rule_name for record in records for step in record.path.trace.steps
    )
    return {
        "molecules": len(records),
        "mean_steps": sum(record.path.path_length for record in records) / len(records),
        "family_counts": dict(sorted(families.items())),
    }


def _coverage(
    records,
    cycle_support,
    attachment_support,
    ear_support,
) -> dict[str, float | int]:
    cycle_total = cycle_hit = 0
    attachment_total = attachment_hit = 0
    ear_total = ear_hit = supported_molecules = 0
    for record in records:
        supported = True
        for step in record.path.trace.steps:
            if isinstance(step.action, CycleInsert):
                cycle_total += 1
                hit = cycle_template(step.action) in cycle_support
                cycle_hit += int(hit)
                supported &= hit
            elif isinstance(step.action, CycleAttach):
                attachment_total += 1
                hit = attach_template(step.action) in attachment_support
                attachment_hit += int(hit)
                supported &= hit
            elif isinstance(step.action, RingEarInsert):
                ear_total += 1
                hit = ear_template(step.action) in ear_support
                ear_hit += int(hit)
                supported &= hit
        supported_molecules += int(supported)
    count = len(records)
    return {
        "cycle_event_fraction": cycle_hit / max(cycle_total, 1),
        "attachment_event_fraction": attachment_hit / max(attachment_total, 1),
        "ear_event_fraction": ear_hit / max(ear_total, 1),
        "fully_supported_molecules": supported_molecules,
        "fully_supported_fraction": supported_molecules / count,
    }


if __name__ == "__main__":
    main()
