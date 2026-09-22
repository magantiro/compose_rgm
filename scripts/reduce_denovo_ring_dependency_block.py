"""Reduce the ring dependency-block acceptance artifact to its decision.

The gate is stated over ONE number -- uniform mass on 3/4-ring templates over
the exact executable support, at the states where ring events fire -- but that
number alone cannot be acted on, so four things are reported beside it and the
verdict names which of them it rests on:

* the PAIRED delta, because both arms share every source draw and a matched
  difference is the stronger statement than two marginal means;
* the per-ring-system ORDINAL breakdown, because committing one ring system
  constricts the support the next is decided against, and an aggregate that
  mixes them hides whether the repair works or merely works once;
* endpoint exactness as N of N, because a schedule that moves an endpoint is a
  different corpus rather than a repair;
* the full per-minimum-ring-size histogram, because two supports with the same
  small-ring fraction can differ completely in what else they offer.

Molecules carrying no ring system are counted and excluded: the repair has no
work to do on them and including them would dilute the very statistic the gate
is stated over.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

ARMS = ("sequential", "exact_early_ring", "ring_dependency_block")
BASELINE_ARM = "sequential"
SHIPPED_ARM = "exact_early_ring"
REPAIR_ARM = "ring_dependency_block"
GATE = 0.15


def _events(row: dict, arm: str) -> list[dict]:
    entry = row.get("arms", {}).get(arm, {})
    if "compile_error" in entry:
        return []
    return entry.get("ring_events", [])


def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None, "stderr": None}
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "median": median(values),
        "stderr": (variance / n) ** 0.5,
        "min": min(values),
        "max": max(values),
    }


def reduce_artifact(rows: list[dict]) -> dict:
    parsed = [row for row in rows if "target_parse_error" not in row]
    both = [
        row
        for row in parsed
        if all(
            "compile_error" not in row.get("arms", {}).get(arm, {"compile_error": 1})
            for arm in ARMS
        )
    ]
    ring_free = [row for row in both if not _events(row, BASELINE_ARM)]
    in_scope = [row for row in both if _events(row, BASELINE_ARM)]

    report: dict = {
        "gate": {
            "statistic": "P_support(3- or 4-ring | at a ring decision point)",
            "definition": "uniform mass on 3/4-ring templates over the exact "
                          "executable ring-template support at each ring_system_grow state",
            "threshold": GATE,
            "direction": "lower is better",
        },
        "molecules": {
            "rows": len(rows),
            "target_parse_failed": len(rows) - len(parsed),
            "compiled_under_every_arm": len(both),
            "without_a_ring_system_out_of_scope": len(ring_free),
            "in_repair_scope": len(in_scope),
        },
        "arms": {},
    }

    for arm in ARMS:
        events = [event for row in in_scope for event in _events(row, arm)]
        masses = [e["small_mass_uniform_support"] for e in events]
        legal = [float(e["legal_template_count"]) for e in events]
        fractions = [e["fraction_of_trace"] for e in events]
        histogram: Counter = Counter()
        for event in events:
            histogram.update(event["size_histogram"])
        total = sum(histogram.values())
        entries = [row["arms"][arm] for row in both]
        report["arms"][arm] = {
            "small_ring_support_mass": _stats(masses),
            "legal_template_count": _stats(legal),
            "ring_event_fraction_of_trace": _stats(fractions),
            "endpoint_canonical_key_matches": sum(
                1 for e in entries if e.get("endpoint_canonical_key_matches")
            ),
            "endpoint_array_exact": sum(
                1 for e in entries if e.get("endpoint_array_exact")
            ),
            "endpoint_checked": len(entries),
            "support_size_histogram": dict(sorted(histogram.items(), key=_size_key)),
            "support_size_histogram_share": (
                {k: v / total for k, v in sorted(histogram.items(), key=_size_key)}
                if total
                else {}
            ),
        }

    # Paired: same molecule, same source draw, same ring-system ordinal.  The
    # repair is compared against BOTH the legacy schedule and the shipped
    # scheduler, because "better than the corpus we trained on" and "better
    # than what already ships" are different claims.
    for label, left_arm, right_arm in (
        ("repair_minus_sequential", BASELINE_ARM, REPAIR_ARM),
        ("repair_minus_shipped_scheduler", SHIPPED_ARM, REPAIR_ARM),
        ("shipped_scheduler_minus_sequential", BASELINE_ARM, SHIPPED_ARM),
    ):
        deltas = []
        for row in in_scope:
            for left, right in zip(_events(row, left_arm), _events(row, right_arm)):
                deltas.append(
                    right["small_mass_uniform_support"]
                    - left["small_mass_uniform_support"]
                )
        entry = _stats(deltas)
        entry["improved"] = sum(1 for d in deltas if d < 0)
        entry["worsened"] = sum(1 for d in deltas if d > 0)
        entry["unchanged"] = sum(1 for d in deltas if d == 0)
        report.setdefault("paired_deltas", {})[label] = entry

    # Per ring-system ordinal within a molecule.
    ordinals: dict[str, dict[int, list[float]]] = {
        arm: defaultdict(list) for arm in ARMS
    }
    for row in in_scope:
        for arm in ARMS:
            for position, event in enumerate(_events(row, arm)):
                ordinals[arm][position].append(event["small_mass_uniform_support"])
    report["by_ring_system_ordinal"] = {
        arm: {
            str(position): _stats(values)
            for position, values in sorted(ordinals[arm].items())
        }
        for arm in ARMS
    }

    # How often the repair could not act, which bounds what a retrain would fix.
    report["repair_reach"] = {
        "molecules_with_a_deferred_system": sum(
            1
            for row in both
            if (row["arms"][REPAIR_ARM].get("ring_dependency_block_deferred") or 0) > 0
        ),
        "molecules_fell_back_to_sequential": sum(
            1
            for row in both
            if row["arms"][REPAIR_ARM].get(
                "ring_dependency_block_fell_back_to_sequential"
            )
        ),
    }

    repair = report["arms"][REPAIR_ARM]["small_ring_support_mass"]
    baseline = report["arms"][BASELINE_ARM]["small_ring_support_mass"]
    exact_ok = all(
        report["arms"][arm]["endpoint_array_exact"]
        == report["arms"][arm]["endpoint_checked"]
        for arm in ARMS
    )
    report["verdict"] = {
        "endpoint_exactness_holds": exact_ok,
        "baseline_mean": baseline["mean"],
        "shipped_scheduler_mean": report["arms"][SHIPPED_ARM][
            "small_ring_support_mass"
        ]["mean"],
        "repair_mean": repair["mean"],
        "gate_passed_on_mean": bool(
            exact_ok and repair["mean"] is not None and repair["mean"] < GATE
        ),
        "gate_passed_on_median": bool(
            exact_ok and repair["median"] is not None and repair["median"] < GATE
        ),
    }
    return report


def _size_key(item: tuple[str, int]) -> tuple[int, str]:
    key = item[0]
    return (int(key), key) if key.isdigit() else (10_000, key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, help="full acceptance JSON")
    parser.add_argument("--rows", type=Path, help="JSONL sidecar, for a partial run")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.rows and args.rows.exists() and args.rows.stat().st_size:
        rows = [json.loads(line) for line in args.rows.read_text().splitlines() if line]
        source = str(args.rows)
    elif args.artifact:
        payload = json.loads(args.artifact.read_text())
        rows = payload["rows"]
        source = str(args.artifact)
    else:
        raise SystemExit("no readable input")

    report = reduce_artifact(rows)
    report["source"] = source
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps(report, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
