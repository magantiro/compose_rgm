#!/usr/bin/env python3
"""Does the production chemistry kernel reproduce the laptop kernel's fragment rows?

The attachment-control sweep was first run on the laptop stack (python 3.12 /
rdkit 2026.03.6) and then re-run, shard for shard, on the pinned production
stack (python 3.11 / rdkit 2024.3.5 / numpy 1.26.4).  The two are known to
disagree on states a search INVENTS -- a T4 feasibility grid diverged on 105 of
600 evaluations over a Kekule-degenerate hypervalent-sulfur ring -- so parity on
one shard is not parity on a panel, and a fragment row cannot be published off
the unpinned kernel merely because a drug-like rescoring agreed.

This compares the two runs shard by shard over EVERY field the row is built
from, including the emitted SMILES and the committed endpoints themselves, and
reports disagreements per field.  Wall-clock is excluded: it is not a result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Wall-clock differs by construction (different interpreter, different machine
# load) and carries no scientific content.
IGNORED_FIELDS = frozenset({"seconds"})

# A float that differs only in its last bits differs because the two builds
# summed in a different order, not because the chemistry moved.  Anything an
# order-of-summation difference cannot explain has to be reported as a real
# disagreement, so the threshold is deliberately far below any quantity the
# benchmark reports (validity is in percent, diversity in [0, 1]).
FLOAT_NOISE_TOLERANCE = 1e-12


def _numeric_deltas(left, right):
    """Yield (name, |delta|) for a changed field, descending into a metric dict.

    A non-numeric change -- a SMILES string, a count, a family census -- yields
    infinity, so it can never be written off as float noise.
    """
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            for name, delta in _numeric_deltas(left.get(key), right.get(key)):
                yield (f"{key}.{name}" if name else key), delta
        return
    if isinstance(left, bool) or isinstance(right, bool):
        yield "", 0.0 if left == right else float("inf")
        return
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        yield "", abs(float(left) - float(right))
        return
    yield "", 0.0 if left == right else float("inf")


def _verdict(per_field, worst):
    if per_field:
        return "DIVERGENT"
    if any(value > 0.0 for value in worst.values()):
        return "PARITY_TO_FLOAT_NOISE"
    return "EXACT_PARITY"


def _rows(shard: Path) -> dict[tuple[str, str, int], dict]:
    payload = json.loads(shard.read_text())
    out: dict[tuple[str, str, int], dict] = {}
    for task, result in payload.get("results", {}).items():
        for drug, entries in result.get("per_drug", {}).items():
            for entry in entries:
                out[(task, drug, int(entry["seed"]))] = entry
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unpinned-root", required=True, type=Path)
    parser.add_argument("--pinned-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    per_field: dict[str, int] = {}
    worst: dict[str, float] = {}
    compared = 0
    missing: list[str] = []
    kernels: dict[str, set[str]] = {"unpinned": set(), "pinned": set()}
    examples: list[dict] = []

    for arm in ("baseline", "attachment"):
        unpinned_dir = args.unpinned_root / arm
        pinned_dir = args.pinned_root / arm
        for shard in sorted(unpinned_dir.glob("*.json")):
            twin = pinned_dir / shard.name
            if not twin.exists():
                missing.append(f"{arm}/{shard.name}")
                continue
            kernels["unpinned"].add(json.loads(shard.read_text())["kernel"]["rdkit"])
            kernels["pinned"].add(json.loads(twin.read_text())["kernel"]["rdkit"])
            left, right = _rows(shard), _rows(twin)
            for key in sorted(set(left) & set(right)):
                compared += 1
                a, b = left[key], right[key]
                for field in sorted(set(a) | set(b)):
                    if field in IGNORED_FIELDS:
                        continue
                    left_value, right_value = a.get(field), b.get(field)
                    if left_value == right_value:
                        continue
                    for metric, delta in _numeric_deltas(left_value, right_value):
                        name = f"{field}.{metric}" if metric else field
                        worst[name] = max(worst.get(name, 0.0), delta)
                        if delta > FLOAT_NOISE_TOLERANCE:
                            per_field[name] = per_field.get(name, 0) + 1
                            if len(examples) < 10:
                                examples.append({
                                    "arm": arm, "key": list(key), "field": name,
                                    "unpinned": left_value, "pinned": right_value,
                                })

    payload = {
        "schema": "compose_fragment_kernel_parity_panel_v1",
        "question": (
            "does the pinned production kernel reproduce the laptop kernel's "
            "fragment rows across the whole panel, not just one shard?"
        ),
        "unpinned_rdkit": sorted(kernels["unpinned"]),
        "pinned_rdkit": sorted(kernels["pinned"]),
        "rows_compared": compared,
        "shards_missing_a_pinned_twin": missing,
        "ignored_fields": sorted(IGNORED_FIELDS),
        "float_noise_tolerance": FLOAT_NOISE_TOLERANCE,
        "max_abs_delta_by_field": {k: v for k, v in sorted(worst.items())},
        "disagreements_beyond_noise_by_field": per_field,
        "disagreement_examples": examples,
        "verdict": _verdict(per_field, worst),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: v for k, v in payload.items() if k != "disagreement_examples"}, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
