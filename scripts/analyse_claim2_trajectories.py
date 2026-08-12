"""Turn Claim-2 trajectory shards into the mobility--fidelity frontier.

Runs entirely locally on committed shards.  Nothing here needs the model, the
checkpoint or Modal, so a metric bug costs a rerun of this script rather than a
container-hour.

Order of operations, and why it is this order
---------------------------------------------
1.  **Instrument checks first.**  If the three arms did not actually diverge,
    or this kernel disagreed with ``canonical_successor_result``, then every
    downstream number describes something other than what it claims to and the
    run is reported ``INVALID_INSTRUMENT``.  Checking after computing the
    headline is how a broken instrument gets published.
2.  **Average within source, then across sources.**  The source molecule is the
    independent statistical unit; several rollout seeds from one source are
    repeated measures.  Pooling trajectories directly would compute intervals
    against a denominator the experiment does not have.
3.  **Verdict last, and only if powered.**  Below ``MIN_SOURCES_FOR_VERDICT``
    sources no Pareto verdict is emitted at all, whatever the point estimates
    look like.  A smoke measures cost and sanity; it does not decide a claim.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path
from typing import Any

from compose_v4.experiments.claim2_trajectory_metrics import (
    BANNED_STATISTICS,
    DescriptorEnvelope,
    FrontierPoint,
    descriptor_vector,
    family_coverage,
    family_entropy,
    frontier_verdict,
    multi_family_trajectory,
    paired_bootstrap_interval,
    resolved,
    ring_system_count,
    tanimoto_distance,
    trajectory_health,
    validate_metric_registry,
)
from compose_v4.experiments.claim2_transport_laws import ARM_REFERENCE, ARMS

#: Below this many sources no Pareto verdict is emitted. An 8-source smoke
#: cannot decide a claim, and saying so in code stops it being decided anyway.
MIN_SOURCES_FOR_VERDICT = 20

SUPPORT_BAND_ORDER = ("0", "1-4", "5-24", "25+")


def load_shards(directory: Path) -> list[dict[str, Any]]:
    shards = [json.loads(path.read_text()) for path in sorted(directory.glob("source-*.json"))]
    if not shards:
        raise SystemExit(f"no shards under {directory}")
    return shards


def instrument_checks(shards: list[dict[str, Any]]) -> dict[str, Any]:
    """Everything that could make the headline meaningless, measured first."""
    disagreements, unchecked = [], 0
    for shard in shards:
        agreement = shard.get("kernel_agreement")
        if agreement is None:
            unchecked += 1
        elif not agreement.get("agrees"):
            disagreements.append(
                {
                    "source": shard["source"],
                    "keys_match": agreement.get("keys_match"),
                    "alias_counts_match": agreement.get("alias_counts_match"),
                    "max_absolute_probability_difference": agreement.get(
                        "max_absolute_probability_difference"
                    ),
                }
            )

    minima, degenerate, states = [], 0, 0
    for shard in shards:
        for row in shard.get("state_divergence", {}).values():
            states += 1
            minima.append(float(row["minimum"]))
            degenerate += bool(row["degenerate"])

    singleton_support = sum(
        1
        for shard in shards
        for row in shard.get("state_divergence", {}).values()
        if int(row["support_size"]) <= 1
    )
    # An enumeration failure truncates a trajectory but is NOT a chemical dead
    # end, so it must be visible rather than absorbed into the dead-end rate.
    failures = {
        state: reason
        for shard in shards
        for state, reason in (shard.get("enumeration_failures") or {}).items()
    }
    rates = [
        shard["seconds_per_kernel_call"]
        for shard in shards
        if shard.get("seconds_per_kernel_call")
    ]
    truncated = sum(
        1
        for shard in shards
        for trajectory in shard["trajectories"]
        if trajectory["stop_reason"] in {"enumeration_failed", "budget_exhausted"}
    )
    return {
        "kernel_cross_check_disagreements": disagreements,
        "kernel_cross_check_unchecked_sources": unchecked,
        "states_measured": states,
        "arm_divergence_min": min(minima) if minima else None,
        "arm_divergence_median": statistics.median(minima) if minima else None,
        "arm_divergence_max": max(minima) if minima else None,
        "degenerate_states": degenerate,
        "degenerate_state_fraction": degenerate / states if states else None,
        "singleton_support_states": singleton_support,
        "budget_exhausted_sources": sum(1 for s in shards if s.get("budget_exhausted")),
        "enumeration_failures": failures,
        "trajectories_truncated_by_failure_or_budget": truncated,
        "measured_seconds_per_kernel_call": {
            "median": statistics.median(rates) if rates else None,
            "min": min(rates) if rates else None,
            "max": max(rates) if rates else None,
        },
        "banned_statistics": sorted(BANNED_STATISTICS),
    }


def trajectory_metrics(
    trajectory: dict[str, Any],
    envelope: DescriptorEnvelope,
    horizon: int,
) -> dict[str, Any]:
    """Every per-trajectory quantity in the frozen suite."""
    states = trajectory["states"]
    source, endpoint = states[0], states[-1]
    health = trajectory_health(states, horizon)

    source_vector = descriptor_vector(source)
    inside, measured, drifts = 0, 0, []
    for state in states[1:]:
        vector = descriptor_vector(state)
        if vector is None:
            continue
        measured += 1
        inside += int(envelope.contains(vector))
        if source_vector is not None:
            drifts.append(envelope.drift(source_vector, vector))

    source_rings = ring_system_count(source)
    endpoint_rings = ring_system_count(endpoint)
    source_heavy = None if source_vector is None else int(source_vector[8])
    endpoint_vector = descriptor_vector(endpoint)
    endpoint_heavy = None if endpoint_vector is None else int(endpoint_vector[8])

    return {
        "arm": trajectory["arm"],
        "seed": trajectory["seed"],
        "endpoint": endpoint,
        "committed_edits": health.committed_edits,
        "endpoint_tanimoto_distance": tanimoto_distance(source, endpoint),
        "heavy_atom_change": (
            None if source_heavy is None or endpoint_heavy is None
            else endpoint_heavy - source_heavy
        ),
        "ring_system_change": (
            None if source_rings is None or endpoint_rings is None
            else endpoint_rings - source_rings
        ),
        "unique_state_fraction": health.unique_state_fraction,
        "immediate_reversal_rate": health.immediate_reversal_rate,
        "two_cycle_rate": health.two_cycle_rate,
        "any_state_revisit_rate": health.revisit_rate,
        "early_dead_end": bool(health.dead_end),
        "stop_reason": trajectory["stop_reason"],
        "operator_family_entropy": family_entropy(trajectory["families"]),
        "families_used": sorted(family_coverage(trajectory["families"])),
        "cells_used": sorted({cell for step in trajectory["cells"] for cell in step}),
        "multi_family": multi_family_trajectory(trajectory["families"]),
        "envelope_states_measured": measured,
        "envelope_states_inside": inside,
        "envelope_retention": inside / measured if measured else None,
        "mean_standardized_drift": statistics.mean(drifts) if drifts else None,
        "final_standardized_drift": drifts[-1] if drifts else None,
        "mean_support_size": (
            statistics.mean(trajectory["support_sizes"])
            if trajectory["support_sizes"] else None
        ),
    }


def _mean(values: list[float | None]) -> float | None:
    present = [float(v) for v in values if v is not None]
    return statistics.mean(present) if present else None


def per_source_by_arm(
    shards: list[dict[str, Any]], envelope: DescriptorEnvelope
) -> dict[str, dict[str, dict[str, Any]]]:
    """arm -> source -> metrics averaged over that source's seeds."""
    out: dict[str, dict[str, dict[str, Any]]] = {arm: {} for arm in ARMS}
    for shard in shards:
        horizon = int(shard["horizon"])
        rows = [trajectory_metrics(t, envelope, horizon) for t in shard["trajectories"]]
        by_arm: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for row in rows:
            by_arm[row["arm"]].append(row)
        for arm, arm_rows in by_arm.items():
            endpoints = [row["endpoint"] for row in arm_rows]
            out.setdefault(arm, {})[shard["source"]] = {
                "support_band": shard.get("support_band"),
                "size_band": shard.get("size_band"),
                "seeds": len(arm_rows),
                "endpoint_tanimoto_distance": _mean(
                    [row["endpoint_tanimoto_distance"] for row in arm_rows]
                ),
                "heavy_atom_change": _mean([row["heavy_atom_change"] for row in arm_rows]),
                "ring_system_change": _mean([row["ring_system_change"] for row in arm_rows]),
                "unique_state_fraction": _mean(
                    [row["unique_state_fraction"] for row in arm_rows]
                ),
                "immediate_reversal_rate": _mean(
                    [row["immediate_reversal_rate"] for row in arm_rows]
                ),
                "any_state_revisit_rate": _mean(
                    [row["any_state_revisit_rate"] for row in arm_rows]
                ),
                "operator_family_entropy": _mean(
                    [row["operator_family_entropy"] for row in arm_rows]
                ),
                "multi_family_fraction": _mean(
                    [float(row["multi_family"]) for row in arm_rows]
                ),
                "envelope_retention": _mean([row["envelope_retention"] for row in arm_rows]),
                "mean_standardized_drift": _mean(
                    [row["mean_standardized_drift"] for row in arm_rows]
                ),
                "early_dead_end_rate": _mean([float(row["early_dead_end"]) for row in arm_rows]),
                "endpoint_uniqueness": len(set(endpoints)) / len(endpoints),
                "families_used": sorted({f for row in arm_rows for f in row["families_used"]}),
                "cells_used": sorted({c for row in arm_rows for c in row["cells_used"]}),
                "committed_edits": _mean([float(row["committed_edits"]) for row in arm_rows]),
            }
    return out


def frontier_point(arm: str, sources: dict[str, dict[str, Any]]) -> FrontierPoint:
    mobility = [
        row["endpoint_tanimoto_distance"]
        for row in sources.values()
        if row["endpoint_tanimoto_distance"] is not None
    ]
    fidelity = [
        row["envelope_retention"]
        for row in sources.values()
        if row["envelope_retention"] is not None
    ]
    return FrontierPoint(
        arm=arm,
        mobility=statistics.median(mobility) if mobility else float("nan"),
        fidelity=statistics.mean(fidelity) if fidelity else float("nan"),
        source_count=len(sources),
        trajectory_count=sum(row["seeds"] for row in sources.values()),
        state_count=sum(int(row["seeds"] * (row["committed_edits"] or 0)) for row in sources.values()),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", required=True, type=Path)
    parser.add_argument(
        "--envelope", type=Path, default=Path("diagnostics/claim2_descriptor_envelope.json")
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--status", default="SMOKE_HELD_IN")
    args = parser.parse_args()

    validate_metric_registry()
    envelope = DescriptorEnvelope.from_json(json.loads(args.envelope.read_text()))
    shards = load_shards(args.shards)
    checks = instrument_checks(shards)

    print(f"shards: {len(shards)} sources")
    print(f"instrument checks:")
    print(f"  kernel cross-check disagreements: {len(checks['kernel_cross_check_disagreements'])}")
    print(
        f"  arm divergence (min pairwise TV): min {checks['arm_divergence_min']} "
        f"median {checks['arm_divergence_median']} max {checks['arm_divergence_max']}"
    )
    print(
        f"  degenerate states: {checks['degenerate_states']}/{checks['states_measured']} "
        f"({checks['singleton_support_states']} with |N+(x)| <= 1)"
    )
    print(
        f"  enumeration failures: {len(checks['enumeration_failures'])}; "
        f"trajectories truncated by failure or budget: "
        f"{checks['trajectories_truncated_by_failure_or_budget']}"
    )
    print(
        f"  MEASURED seconds per enumeration: "
        f"{checks['measured_seconds_per_kernel_call']['median']} "
        f"(min {checks['measured_seconds_per_kernel_call']['min']}, "
        f"max {checks['measured_seconds_per_kernel_call']['max']})"
    )

    invalid = bool(checks["kernel_cross_check_disagreements"])
    if invalid:
        print(
            "\nINVALID_INSTRUMENT: this kernel disagrees with canonical_successor_result. "
            "Every arm comparison below would describe a private kernel. Not reporting a frontier."
        )

    by_arm = per_source_by_arm(shards, envelope)
    points = {arm: frontier_point(arm, by_arm[arm]) for arm in ARMS if by_arm.get(arm)}
    source_count = min((point.source_count for point in points.values()), default=0)

    print(f"\n{'arm':>20} {'mobility':>10} {'fidelity':>10} {'sources':>8} {'trajs':>7}")
    for arm in ARMS:
        point = points.get(arm)
        if point is None:
            continue
        print(
            f"{arm:>20} {point.mobility:>10.4f} {point.fidelity:>10.4f} "
            f"{point.source_count:>8} {point.trajectory_count:>7}"
        )

    comparisons: dict[str, Any] = {}
    powered = source_count >= MIN_SOURCES_FOR_VERDICT and not invalid
    for arm in ARMS:
        if arm == ARM_REFERENCE or arm not in points:
            continue
        record: dict[str, Any] = {}
        for axis, key, better in (
            ("mobility", "endpoint_tanimoto_distance", "higher"),
            ("fidelity", "envelope_retention", "higher"),
        ):
            left = {
                s: r[key] for s, r in by_arm[ARM_REFERENCE].items() if r[key] is not None
            }
            right = {s: r[key] for s, r in by_arm[arm].items() if r[key] is not None}
            interval = paired_bootstrap_interval(left, right)
            record[axis] = {
                "difference": interval[0],
                "ci_low": interval[1],
                "ci_high": interval[2],
                "resolved": resolved(interval),
                "direction_meaning": f"{better} is more {axis}",
            }
        record["verdict"] = (
            frontier_verdict(
                points[ARM_REFERENCE],
                points[arm],
                mobility_resolved=record["mobility"]["resolved"],
                fidelity_resolved=record["fidelity"]["resolved"],
            )
            if powered
            else "UNDERPOWERED_NO_VERDICT"
        )
        comparisons[f"{ARM_REFERENCE}_vs_{arm}"] = record

    print("\nr_theta versus each unlearned arm (paired over sources):")
    for name, record in comparisons.items():
        print(f"  {name}: {record['verdict']}")
        for axis in ("mobility", "fidelity"):
            entry = record[axis]
            print(
                f"    {axis:>8} {entry['difference']:+.4f} "
                f"[{entry['ci_low']:+.4f}, {entry['ci_high']:+.4f}] "
                f"{'resolved' if entry['resolved'] else 'unresolved'}"
            )
    if not powered and not invalid:
        print(
            f"\nNO VERDICT: {source_count} sources is below the pre-declared "
            f"{MIN_SOURCES_FOR_VERDICT}-source floor. A smoke measures cost and "
            "instrument sanity; it does not decide the claim."
        )

    by_band: dict[str, Any] = {}
    for band in SUPPORT_BAND_ORDER:
        entry: dict[str, Any] = {}
        for arm in ARMS:
            rows = [r for r in by_arm.get(arm, {}).values() if r["support_band"] == band]
            if not rows:
                continue
            entry[arm] = {
                "sources": len(rows),
                "mobility": _mean([r["endpoint_tanimoto_distance"] for r in rows]),
                "fidelity": _mean([r["envelope_retention"] for r in rows]),
                "revisit_rate": _mean([r["any_state_revisit_rate"] for r in rows]),
                "family_entropy": _mean([r["operator_family_entropy"] for r in rows]),
            }
        if entry:
            by_band[band] = entry

    families = {
        arm: sorted({f for row in by_arm.get(arm, {}).values() for f in row["families_used"]})
        for arm in ARMS
    }
    cells = {
        arm: sorted({c for row in by_arm.get(arm, {}).values() for c in row["cells_used"]})
        for arm in ARMS
    }
    print("\noperator families committed:")
    for arm in ARMS:
        print(f"  {arm:>20}: {len(families[arm])} families, {len(cells[arm])} family:table cells")

    payload = {
        "schema": "compose.claim2.trajectory_analysis",
        "status": "INVALID_INSTRUMENT" if invalid else args.status,
        "shard_directory": str(args.shards),
        "envelope_sha256": json.loads(args.envelope.read_text()).get("envelope_sha256"),
        "source_count": source_count,
        "minimum_sources_for_verdict": MIN_SOURCES_FOR_VERDICT,
        "verdict_emitted": powered,
        "instrument_checks": checks,
        "frontier": {arm: point.to_json() for arm, point in points.items()},
        "comparisons": comparisons,
        "by_support_band": by_band,
        "families_by_arm": families,
        "cells_by_arm": cells,
        "per_source": by_arm,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {args.out}")
    return 1 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
