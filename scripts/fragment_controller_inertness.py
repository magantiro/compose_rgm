#!/usr/bin/env python3
"""Is the attachment controller INERT on a prompt that declares no interface?

The controller is meant to activate from the constraint SPECIFICATION, never
from a drug or a task label.  ``superstructure_generation`` declares no
attachment interface, so switching the controller on must change nothing there
-- and "nothing" has to mean the molecules, not just an aggregate that rounds to
the same number.  An aggregate delta of zero is consistent with two arms that
emitted different molecules and happened to score alike.

This compares the two arms row by row on a task with no declared interface and
reports how many rows emitted a DIFFERENT set of molecules.  The answer should
be zero.  It also reports the controller's own counters, which should be zero
for the same reason and are independent evidence: a controller that redirected
or rejected anything was not inert, whatever the molecules say.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MOLECULE_FIELDS = ("emitted_samples", "committed_endpoint_smiles")
CONTROLLER_COUNTERS = (
    "redirections",
    "interface_rejections",
    "staging_rejections",
    "separation_failures",
)


def _rows(shards: Path, task: str) -> dict[tuple[str, int], dict]:
    out: dict[tuple[str, int], dict] = {}
    for shard in sorted(shards.glob(f"{task}*.json")):
        payload = json.loads(shard.read_text())
        for result in payload.get("results", {}).values():
            for drug, entries in result.get("per_drug", {}).items():
                for entry in entries:
                    out[(drug, int(entry["seed"]))] = entry
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-shards", required=True, type=Path)
    parser.add_argument("--attachment-shards", required=True, type=Path)
    parser.add_argument("--task", default="superstructure_generation")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    before = _rows(args.baseline_shards, args.task)
    after = _rows(args.attachment_shards, args.task)
    keys = sorted(set(before) & set(after))
    if not keys:
        raise SystemExit(f"no paired rows for {args.task}")

    molecules_differ = []
    declared = set()
    counters = dict.fromkeys(CONTROLLER_COUNTERS, 0)
    for key in keys:
        b, a = before[key], after[key]
        declared.add(len(a.get("declared_interfaces") or ()))
        for field in CONTROLLER_COUNTERS:
            counters[field] += int(a.get(field, 0))
        if any(b.get(f) != a.get(f) for f in MOLECULE_FIELDS):
            molecules_differ.append([key[0], key[1]])

    inert = not molecules_differ and not any(counters.values())
    out = {
        "schema": "compose_fragment_controller_inertness_v1",
        "question": (
            "does switching the attachment controller on change anything on a "
            "task that declares no attachment interface?"
        ),
        "task": args.task,
        "paired_rows": len(keys),
        "declared_interface_counts_seen": sorted(declared),
        "rows_whose_molecules_differ": molecules_differ,
        "controller_counters_in_attachment_arm": counters,
        "verdict": "INERT" if inert else "NOT INERT",
        "note": (
            "Molecule-level equality is the claim; the controller counters are "
            "independent evidence, since a controller that redirected or "
            "rejected anything was not inert whatever the molecules say. "
            "Metrics are deliberately NOT compared: official diversity is "
            "hash-order dependent at ~1 ULP across processes."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
