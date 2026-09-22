#!/usr/bin/env python
"""Apply the PREDECLARED scored-A/B rule verbatim to the two arms' artifacts."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from compose_v4.control.program_task import pmo_top_ten_auc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "diagnostics/pmo_completion_repair_v1/scored_ab"
CHECKPOINTS = (100, 150, 200, 250)


def charged_sequence(folder: Path) -> list[dict]:
    """The charged calls in ledger order, from the durable per-query receipts."""

    rows = []
    for path in sorted((folder / "oracle").rglob("result.json")):
        payload = json.loads(path.read_text())
        payload["index"] = int(payload["index"])
        payload["score"] = float(payload["score"])
        rows.append(payload)
    rows.sort(key=lambda row: row["index"])
    return rows


def endpoint_completions(folder: Path) -> dict[str, dict]:
    """Map each proposed endpoint to the completion provenance of its program."""

    out: dict[str, dict] = {}
    for path in (folder / "campaign").rglob("*.json"):
        try:
            payload = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            continue
        stack = [payload]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                endpoint = node.get("endpoint")
                if isinstance(endpoint, str) and "trace" not in node:
                    found, sizes = [], []
                    inner = [node]
                    while inner:
                        item = inner.pop()
                        if isinstance(item, dict):
                            if "completion" in item and "requested_size" in item:
                                found.append(str(item["completion"]))
                                sizes.append(int(item.get("requested_size") or 0))
                            inner.extend(item.values())
                        elif isinstance(item, list):
                            inner.extend(item)
                    if found:
                        prior = out.setdefault(
                            endpoint, {"completions": [], "max_requested_size": 0}
                        )
                        prior["completions"].extend(found)
                        prior["max_requested_size"] = max(
                            prior["max_requested_size"], max(sizes)
                        )
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
    return out


def throughput(result: dict) -> dict:
    rounds = result.get("round_progress") or []
    if not rounds:
        return {}
    attempts = [r["proposal_attempts"] for r in rounds if r.get("proposal_attempts")]
    pools = [r["pool_size"] for r in rounds if r.get("pool_size")]
    return {
        "rounds": len(rounds),
        "mean_proposal_attempts": round(sum(attempts) / max(1, len(attempts)), 1),
        "mean_pool_size": round(sum(pools) / max(1, len(pools)), 1),
    }


def top_ten_mean(values: list[float]) -> float:
    return sum(sorted(values, reverse=True)[:10]) / min(10, max(1, len(values)))


def completion_provenance(folder: Path) -> dict:
    counts, sizes = {}, []
    for path in folder.rglob("*.json"):
        try:
            payload = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            continue
        stack = [payload]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                if "completion" in node and "requested_size" in node:
                    counts[str(node["completion"])] = counts.get(
                        str(node["completion"]), 0
                    ) + 1
                    sizes.append(int(node.get("requested_size") or 0))
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
    return {
        "counts": counts,
        "requested_size_gt_8": sum(1 for s in sizes if s > 8),
        "max_requested_size": max(sizes) if sizes else 0,
    }


def main() -> None:
    task = sys.argv[1] if len(sys.argv) > 1 else "celecoxib_rediscovery"
    arms = {}
    for arm in ("A_deployed_b", "B_completion"):
        folder = OUT / task / arm
        result = json.loads((folder / "result.json").read_text())
        values = [row["score"] for row in charged_sequence(folder)]
        if not values:
            values = [point["score"] for point in result.get("score_curve", [])]
        arms[arm] = {
            "charged": result["charged_oracle_calls"],
            "best": result["best_score"],
            "auc": result["auc_top10_at_budget"],
            "auc_budget": result["auc_budget"],
            "wall_seconds": result.get("wall_seconds"),
            "arm_spec": result.get("arm"),
            "top10_at": {
                str(n): round(top_ten_mean(values[:n]), 6)
                for n in CHECKPOINTS
                if len(values) >= n
            },
            "auc_at": {
                str(n): round(pmo_top_ten_auc(values[:n], budget=n, finish=True), 6)
                for n in CHECKPOINTS
                if len(values) >= n
            },
            "completion_provenance": completion_provenance(folder),
            "throughput": throughput(result),
            "n_scored_values": len(values),
        }
        joined = endpoint_completions(folder)
        with_law, without_law = [], []
        for row in charged_sequence(folder):
            if row.get("role") == "initialization":
                continue
            target = with_law if row["endpoint"] in joined else without_law
            target.append(row["score"])
        buckets: dict[str, list[float]] = {}
        for row in charged_sequence(folder):
            if row.get("role") == "initialization":
                continue
            info = joined.get(row["endpoint"])
            size = info["max_requested_size"] if info else 0
            key = (
                "none" if not info else
                "1-4" if size <= 4 else
                "5-8" if size <= 8 else
                "9-16" if size <= 16 else "17+"
            )
            buckets.setdefault(key, []).append(row["score"])
        arms[arm]["score_by_requested_size"] = {
            key: {
                "n": len(values),
                "mean": round(sum(values) / len(values), 6),
                "max": round(max(values), 6),
            }
            for key, values in sorted(buckets.items())
        }
        arms[arm]["score_by_completion_provenance"] = {
            "with_a_repaired_completion": {
                "n": len(with_law),
                "mean": round(sum(with_law) / len(with_law), 6) if with_law else None,
                "max": round(max(with_law), 6) if with_law else None,
            },
            "without": {
                "n": len(without_law),
                "mean": (
                    round(sum(without_law) / len(without_law), 6)
                    if without_law
                    else None
                ),
                "max": round(max(without_law), 6) if without_law else None,
            },
        }

    a, b = arms["A_deployed_b"], arms["B_completion"]
    leads = [n for n in CHECKPOINTS
             if str(n) in a["top10_at"] and b["top10_at"][str(n)] > a["top10_at"][str(n)]]
    treatment = b["completion_provenance"]
    valid = bool(treatment["counts"].get("component") or treatment["requested_size_gt_8"])
    control_clean = not a["completion_provenance"]["counts"]
    if not valid or not control_clean:
        verdict = "VOID"
    elif b["auc"] >= 1.10 * a["auc"] and len(leads) >= 3:
        verdict = "QUALIFY"
    elif b["auc"] <= 0.90 * a["auc"]:
        verdict = "REJECT"
    else:
        verdict = "INCONCLUSIVE"

    report = {
        "schema_version": "pmo_completion_scored_ab_report_v1",
        "task": task,
        "decision_rule": "diagnostics/pmo_completion_repair_v1/decision_rule_v1.json",
        "arms": arms,
        "relative_auc_b_over_a": round(b["auc"] / a["auc"], 4) if a["auc"] else None,
        "checkpoints_where_b_leads": leads,
        "validity_check": {
            "treatment_carries_the_law": valid,
            "control_is_clean": control_clean,
        },
        "verdict": verdict,
        "caveats": [
            "one task, one seed, 250 charged calls per arm",
            "top_auc trapezoids from (0,0), so a 250-call AUC is structurally "
            "depressed ~20% at a constant level; never compare it to a 10k figure",
            "run locally on the pinned PMO kernel, not in the deployed Modal image",
        ],
    }
    path = ROOT / f"diagnostics/pmo_completion_repair_v1/scored_ab_report_{task}.json"
    path.write_text(json.dumps(report, sort_keys=True, indent=1))
    print(f"{'arm':14s}{'charged':>9}{'best':>10}{'auc':>10}"
          + "".join(f"{'t10@'+str(n):>11}" for n in CHECKPOINTS))
    for name in ("A_deployed_b", "B_completion"):
        row = arms[name]
        print(f"{name:14s}{row['charged']:>9}{row['best']:>10.4f}{row['auc']:>10.4f}"
              + "".join(f"{row['top10_at'].get(str(n), float('nan')):>11.4f}"
                        for n in CHECKPOINTS))
    print(f"\nB/A AUC ratio {report['relative_auc_b_over_a']}  "
          f"B leads at {leads}")
    for name in ("A_deployed_b", "B_completion"):
        print(f"  {name:14s} throughput={arms[name]['throughput']}  "
              f"score_split={arms[name]['score_by_completion_provenance']}")
        print(f"  {name:14s} by_size={arms[name]['score_by_requested_size']}")
    print("treatment provenance:", treatment)
    print("control provenance  :", a["completion_provenance"])
    print("\nVERDICT:", verdict)
    print("wrote", path)


main()
