#!/usr/bin/env python3
"""Operator-SUBTYPE / ring-TEMPLATE supervision gate.

The earlier supervision audits (cold_vocab_audit.py, Stage-6 coverage) counted positive targets only per
(element,valence) ATOM class, from atom_insert/atom_restate marks. They had no notion of ring-operator
subtype or ring-template supervision -- so a `production_enabled` operator family with ZERO positive selected
targets (ring_system_grow) was invisible and the gate falsely passed.

This gate closes that gap: it counts positive SELECTED-target teacher marks per operator family (subtype) and
per ring template, and FAILS when any production-enabled family has zero positive targets. Use the operator
registry's `production_enabled` set as the required families.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable


def subtype_target_counts(
    family_sequences: Iterable[Iterable[str]],
    family_aliases: dict[str, str] | None = None,
) -> Counter:
    """Positive-target count per operator family, over the teacher-mark (rule_name) sequences of every
    training record (both directions, after all pipeline transformations).

    ``family_aliases`` maps a recorded EXECUTOR rule_name to the MODEL family that scores it, so a teacher
    mark recorded under one name counts toward the family the dense head actually supervises. The
    compositional ring ops need it: ``build_cycle_op_records`` records ``bond_insert``/``bond_delete`` steps
    that the model scores under families ``cycle_insert``/``cycle_attach`` (via ``_CYCLE_OP_EXECUTOR_TO_FAMILY``
    in the rate model). Default identity → callers with no aliasing are byte-identical."""
    aliases = family_aliases or {}
    counts: Counter = Counter()
    for sequence in family_sequences:
        for family in sequence:
            counts[aliases.get(str(family), str(family))] += 1
    return counts


def check_operator_subtype_supervision(
    family_sequences: Iterable[Iterable[str]],
    enabled_families: Iterable[str],
    family_aliases: dict[str, str] | None = None,
) -> tuple[bool, dict]:
    """(ok, report). ok is False iff any production-enabled family has ZERO positive selected targets.
    This is intentionally at the operator-SUBTYPE level, not the atom-class level. ``family_aliases`` is
    applied to the recorded rule_names before counting (see ``subtype_target_counts``)."""
    counts = subtype_target_counts(family_sequences, family_aliases)
    required = list(dict.fromkeys(str(f) for f in enabled_families))
    unsupervised = [f for f in required if counts.get(f, 0) == 0]
    return (not unsupervised, {
        "positive_targets_by_family": dict(counts),
        "required_families": required,
        "unsupervised_enabled_families": unsupervised,
    })


def family_sequences_from_records(records) -> list[list[str]]:
    """Extract the teacher-mark family (rule_name) sequence from each PathRecord's trace steps."""
    sequences: list[list[str]] = []
    for record in records:
        steps = getattr(getattr(getattr(record, "path", None), "trace", None), "steps", ())
        sequences.append([str(step.rule_name) for step in steps])
    return sequences


def main() -> int:
    import argparse
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, help="edit pool jsonl (records with a 'steps' family list)")
    parser.add_argument("--enabled", nargs="+", required=True, help="production-enabled operator families")
    parser.add_argument("--limit", type=int, default=100000)
    parser.add_argument(
        "--family-alias", nargs="*", default=(),
        help="executor=family pairs mapping recorded rule_names to their scoring family "
        "(e.g. bond_insert=cycle_insert bond_delete=cycle_attach)",
    )
    args = parser.parse_args()

    aliases = dict(pair.split("=", 1) for pair in args.family_alias) or None
    sequences: list[list[str]] = []
    if args.pool is not None:
        with args.pool.open() as handle:
            for i, line in enumerate(handle):
                if i >= args.limit or not line.strip():
                    break
                record = json.loads(line)
                steps = record.get("steps") or []
                sequences.append([str(s.get("rule_name", s) if isinstance(s, dict) else s) for s in steps])
    ok, report = check_operator_subtype_supervision(sequences, args.enabled, aliases)
    print(json.dumps({"verdict": "OK" if ok else "NO_GO_OPERATOR_SUBTYPE_SUPERVISION", **report}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
