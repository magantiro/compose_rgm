"""Apply the frozen cycle-attribution definitions to saved trajectories.

Definitions live in ``compose_v4.experiments.claim2_cycle_attribution`` and were
committed before this script was first run. Nothing here defines a metric; it
only reads traces and reports.

Two trajectory sources, because the comparison between them is the point:

* **uncontrolled** -- the Claim-2 arm shards. Does the reference law oscillate?
* **controlled** -- committed goal-directed trajectories (retarget prefixes and
  intervention arms). Does goal control suppress that oscillation?

If the uncontrolled reference is locally reversible while control produces
directed progress, that is a coherent and publishable story about what the
reference law is, rather than a defect in it.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from compose_v4.experiments.claim2_cycle_attribution import (
    attribute_cycles,
    attribution_verdict,
    edges_from_trajectory,
    net_displacement_efficiency,
    reversibility_census,
    summarize,
)
from compose_v4.experiments.claim2_trajectory_metrics import tanimoto_distance
from compose_v4.experiments.claim2_transport_laws import ARMS


def load_arm_trajectories(directory: Path) -> dict[str, list[dict[str, Any]]]:
    """arm -> list of {source, states, families} from the Claim-2 shards."""
    out: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    for path in sorted(directory.glob("source-*.json")):
        shard = json.loads(path.read_text())
        for trajectory in shard["trajectories"]:
            out.setdefault(trajectory["arm"], []).append(
                {
                    "source": shard["source"],
                    "states": trajectory["states"],
                    "families": trajectory["families"],
                }
            )
    return out


def load_controlled_trajectories(repo: Path, shards: Path | None) -> dict[str, list[dict]]:
    """Committed goal-directed trajectories, keyed by controller arm.

    Family labels are not recorded by the controller runs, so these support the
    state-level diagnostics (two-cycles, revisits, cancellation, net edits) but
    not the family-level ones. Reported as such rather than imputed.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    prefixes = repo / "diagnostics/retarget_prefixes_committed.json"
    if prefixes.exists():
        payload = json.loads(prefixes.read_text())
        for row in payload.get("prefixes", []):
            states = row.get("trajectory")
            if states and len(states) > 1:
                out.setdefault("controlled_greedy_prefix", []).append(
                    {"source": row.get("source"), "states": states, "families": None}
                )
    if shards and shards.exists():
        for path in sorted(shards.glob("*.json")):
            payload = json.loads(path.read_text())
            for name, arm in (payload.get("arms") or {}).items():
                states = arm.get("trajectory")
                if states and len(states) > 1:
                    out.setdefault(f"controlled_{name}", []).append(
                        {"source": payload.get("source"), "states": states, "families": None}
                    )
    return out


def describe(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    """Cycle attribution plus net-displacement efficiency for one arm."""
    attributions, efficiencies, per_source_cancel = [], [], {}
    family_undone: Counter[str] = Counter()
    family_total: Counter[str] = Counter()
    edges: list[tuple[str, str]] = []
    for row in rows:
        states = row["states"]
        families = row["families"] if row["families"] is not None else [[]] * (len(states) - 1)
        attribution = attribute_cycles(states, families)
        attributions.append(attribution)
        edges.extend(edges_from_trajectory(states))
        distance = tanimoto_distance(states[0], states[-1])
        efficiencies.append(net_displacement_efficiency(distance, attribution))
        per_source_cancel.setdefault(row["source"], []).append(attribution.cancelled_fraction)
        for name, stats in attribution.reversal_by_family.items():
            family_total[name] += int(stats["committed"])
            family_undone[name] += int(stats["undone_next_step"])

    def mean(values):
        present = [v for v in values if v is not None]
        return statistics.mean(present) if present else None

    census = reversibility_census(edges, label=label)
    return {
        "trajectories": len(rows),
        "sources": len(per_source_cancel),
        "mean": summarize(attributions),
        "net_displacement": {
            "per_committed_edit": mean([e["per_committed_edit"] for e in efficiencies]),
            "per_net_edit": mean([e["per_net_edit"] for e in efficiencies]),
        },
        "reversal_by_family": {
            name: {
                "committed": family_total[name],
                "undone_next_step": family_undone[name],
                "rate": family_undone[name] / family_total[name] if family_total[name] else 0.0,
            }
            for name in sorted(family_total)
        },
        "own_trajectory_reversibility": census.to_json(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", required=True, type=Path)
    parser.add_argument("--controlled-shards", type=Path, default=None)
    parser.add_argument("--corpus-census", type=Path, default=None)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--status", default="SMOKE_HELD_IN")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    arms = load_arm_trajectories(args.shards)
    controlled = load_controlled_trajectories(args.repo_root, args.controlled_shards)

    report: dict[str, Any] = {}
    for arm, rows in arms.items():
        if rows:
            report[arm] = describe(rows, arm)
    for name, rows in controlled.items():
        if rows:
            report[name] = describe(rows, name)

    corpus_fraction = None
    corpus_payload = None
    if args.corpus_census and args.corpus_census.exists():
        corpus_payload = json.loads(args.corpus_census.read_text())
        corpus_fraction = corpus_payload.get("mutual_edge_fraction")

    reference_cancelled = report.get("r_theta", {}).get("mean", {}).get(
        "cancelled_fraction", 0.0
    )
    verdict = attribution_verdict(corpus_fraction, reference_cancelled)

    width = max(len(name) for name in report)
    print(f"{'arm':>{width}} {'trajs':>6} {'commit':>7} {'net':>6} {'cancel%':>8} "
          f"{'2cyc':>6} {'longer':>7} {'invpair':>8} {'disp/net':>9}")
    for name, entry in report.items():
        m, d = entry["mean"], entry["net_displacement"]
        per_net = d["per_net_edit"]
        print(
            f"{name:>{width}} {entry['trajectories']:>6} {m['committed_edits']:>7.2f} "
            f"{m['net_edits']:>6.2f} {m['cancelled_fraction']*100:>7.1f}% "
            f"{m['immediate_two_cycles']:>6.2f} {m['longer_revisits']:>7.2f} "
            f"{m['inverse_family_pairs']:>8.2f} "
            + (f"{per_net:>9.4f}" if per_net is not None else f"{'n/a':>9}")
        )

    print("\nreversal rate by operator family (uncontrolled arms):")
    for arm in ARMS:
        entry = report.get(arm)
        if not entry or not entry["reversal_by_family"]:
            continue
        ranked = sorted(
            entry["reversal_by_family"].items(), key=lambda kv: -kv[1]["rate"]
        )
        summary = "  ".join(
            f"{name}={stats['rate']:.2f}({stats['committed']})" for name, stats in ranked
        )
        print(f"  {arm}: {summary}")

    print(f"\ncorpus mutual-edge fraction: {corpus_fraction}")
    print(f"PREREGISTERED VERDICT: {verdict}")
    if corpus_fraction is None:
        print(
            "  The corpus census is unavailable, so the central causal question is "
            "UNANSWERED. This is inconclusive, not favourable."
        )

    payload = {
        "schema": "compose.claim2.cycle_attribution",
        "status": args.status,
        "definitions_frozen_at_commit": "5c81818",
        "shard_directory": str(args.shards),
        "by_arm": report,
        "corpus_census": corpus_payload,
        "corpus_mutual_edge_fraction": corpus_fraction,
        "reference_cancelled_fraction": reference_cancelled,
        "preregistered_verdict": verdict,
        "not_computable_from_these_traces": [
            "reverse_edge_probability R_theta(x|y): needs the successor row at y, "
            "which the smoke shards do not carry; recorded by the rollout app from "
            "the 36-source run onward"
        ],
        "controlled_family_labels_unavailable": (
            "controller runs do not record operator families, so controlled arms "
            "support state-level diagnostics only"
        ),
        "controlled_comparison_has_a_near_fixed_sign": (
            "The controlled arms are DETERMINISTIC argmax-utility policies, which are "
            "monotone non-decreasing in utility, so an immediate two-cycle requires an "
            "exact utility tie and is close to structurally excluded. 'Control has less "
            "cycling than a stochastic sampler' is therefore largely definitional and "
            "is NOT clean evidence that control suppresses cycling. It is reported as "
            "context. The falsifiable comparison is r_theta versus empirical_family: "
            "both are stochastic samplers over the identical support, so a difference "
            "between them could have come out either way."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
