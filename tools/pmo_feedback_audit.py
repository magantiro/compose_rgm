"""Offline accounting of score information in the completed development pair."""

import argparse
import json
import platform
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from compose_v4.control.archive_allocation import ArchiveCredit
from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from compose_v4.experiments.t4_matched_pilot import unseal

ROOT = Path(__file__).resolve().parents[1]
TOL = 1e-12


def audit(result, snapshot_records=None):
    if result["status"] != "complete" or result["oracle_calls"] != 100:
        raise ValueError("feedback audit requires the completed 100-query pilot")
    nodes = {r["id"]: r for r in result["archive"]}
    scores, rows, groups = {}, [], defaultdict(list)
    credit = ArchiveCredit(discount=0.9, pseudocount=2.0)
    for attempt in result["attempts"]:
        edge = f"attempts/{attempt['attempt']:04}"
        parent = attempt["parent"]
        if parent not in nodes:
            raise ValueError(f"missing exact archived parent {parent}")
        parent_score = scores.get(parent)
        value, trials = credit.values("what", [attempt["option"]])
        row = {
            **attempt,
            "id": edge,
            "parent_score": parent_score,
            "parent_delta": None,
            "prior_option_credit": float(value[0]),
            "prior_option_trials": trials[0],
        }
        if snapshot_records and edge in snapshot_records:
            record = snapshot_records[edge]
            if record["parent_id"] != parent or record["status"] != attempt["status"]:
                raise ValueError("snapshot attempt disagrees with final result")
            for event in record["events"]:
                if event.get("stage") not in ("where", "what"):
                    continue
                saved = event["allocation"]
                q, _ = credit.distribution(
                    saved["reference"],
                    event["stage"],
                    saved["cells"],
                    adaptive=result["case"]["arm"] == "adaptive",
                )
                if not np.allclose(q, saved["probabilities"], rtol=0, atol=TOL):
                    raise ValueError("chronological allocation replay differs from receipt")
                if event["stage"] == "what":
                    j = event["selected"]
                    row["selected_option_allocation"] = {
                        "reference": saved["reference"][j],
                        "controlled": saved["probabilities"][j],
                        "replayed": True,
                    }
        if attempt["status"] == "complete":
            scores[edge] = attempt["score"]
            if parent_score is not None:
                row["parent_delta"] = attempt["score"] - parent_score
        credit.observe(
            edge,
            parent_chain=nodes[parent]["chain"],
            scale=attempt["scale"],
            option=attempt["option"],
            reward=attempt["reward"],
        )
        rows.append(row)
        if attempt["option"] is not None:
            groups[attempt["option"]].append(row)
    if credit.edges != result["credit"]:
        raise ValueError("production credit replay differs from final recorded credit")
    known = [r for r in rows if r["parent_delta"] is not None]
    improved = [r for r in known if r["parent_delta"] > TOL]
    worsened = [r for r in known if r["parent_delta"] < -TOL]
    zero_credit_improvements = [r for r in improved if r["reward"] <= TOL]
    credited_losses = [r for r in worsened if r["reward"] > TOL]
    option_rows = []
    for option, records in sorted(groups.items()):
        completed = [r for r in records if r["score"] is not None]
        estimate, _ = credit.values("what", [option])
        option_rows.append(
            {
                "option": option,
                "attempts": len(records),
                "completed": len(completed),
                "scores": [r["score"] for r in completed],
                "parent_deltas": [r["parent_delta"] for r in records],
                "direct_credit": sum(credit.edges[r["id"]]["direct"] for r in records),
                "total_credit": sum(credit.edges[r["id"]]["credit"] for r in records),
                "final_estimate": float(estimate[0]),
            }
        )
    return {
        "case": result["case"],
        "attempts": len(rows),
        "completed": sum(r["status"] == "complete" for r in rows),
        "production_credit_replay_equal": True,
        "zero_archive_reward": sum(r["reward"] <= TOL for r in rows),
        "known_parent_comparisons": len(known),
        "excluded_parent_comparisons": [
            {
                "id": r["id"],
                "reason": "root_parent_label_not_embedded"
                if r["parent"].startswith("root_")
                else r["status"],
            }
            for r in rows
            if r["parent_delta"] is None
        ],
        "parent_improvements": len(improved),
        "parent_losses": len(worsened),
        "parent_improvements_zero_archive_reward": zero_credit_improvements,
        "parent_losses_positive_archive_reward": credited_losses,
        "final_option_values": sorted(
            option_rows, key=lambda r: (-r["final_estimate"], r["option"])
        ),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adaptive-snapshot", type=Path)
    args = parser.parse_args()
    paths = [
        ROOT / f"diagnostics/pmo_archive_pilot_100/{arm}.json" for arm in ("balanced", "adaptive")
    ]
    inputs = [json.loads(p.read_text()) for p in paths]
    if len({r["run_id"] for r in inputs}) != 1 or len({r["contract_sha256"] for r in inputs}) != 1:
        raise ValueError("inputs are not the same locked development pair")
    snapshots, snapshot_hashes = {}, {}
    if args.adaptive_snapshot:
        source = args.adaptive_snapshot / "source_identity.json"
        identity = json.loads(source.read_text())
        expected = f"pmo_archive_pilot/{inputs[1]['run_id']}/perindopril_mpo__adaptive__0"
        if identity != {"prefix": expected, "volume": "compose-v4-artifacts"}:
            raise ValueError("snapshot source does not match adaptive result")
        snapshot_hashes[str(source)] = sha256_file(source)
        for path in sorted((args.adaptive_snapshot / "attempts").glob("*.json")):
            snapshots[f"attempts/{path.stem}"] = unseal(path)
            snapshot_hashes[str(path)] = sha256_file(path)
    dependencies = [
        Path(__file__),
        ROOT / "src/compose_v4/control/archive_allocation.py",
        ROOT / "src/compose_v4/control/continuation.py",
        ROOT / "src/compose_v4/control/region_rewrite.py",
        ROOT / "src/compose_v4/experiments/pmo_archive_pilot.py",
    ]
    report = {
        "schema_version": "pmo_feedback_audit_v1",
        "at": datetime.now(timezone.utc).isoformat(),
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "input_hashes": {str(p.relative_to(ROOT)): sha256_file(p) for p in paths},
        "snapshot_hashes": snapshot_hashes,
        "implementation_hashes": {str(p.relative_to(ROOT)): sha256_file(p) for p in dependencies},
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "configuration": {
            "tolerance": TOL,
            "discount": 0.9,
            "pseudocount": 2.0,
            "oracle_calls": 0,
            "random_sampling": False,
            "split": "development",
            "device": "cpu",
            "precision": "float64",
        },
        "arms": [audit(inputs[0]), audit(inputs[1], snapshots)],
        "interpretation": "Retrospective score-information accounting. No counterfactual outcomes, training, or policy change. Root-parent label exclusions are explicit.",
    }
    publish_json(args.output, report)
    for arm in report["arms"]:
        print(
            json.dumps(
                {
                    k: v
                    for k, v in arm.items()
                    if k
                    not in (
                        "rows",
                        "final_option_values",
                        "excluded_parent_comparisons",
                        "parent_improvements_zero_archive_reward",
                        "parent_losses_positive_archive_reward",
                    )
                }
            )
        )
        print("zero-credit improvements", len(arm["parent_improvements_zero_archive_reward"]))
        print("positive-credit losses", len(arm["parent_losses_positive_archive_reward"]))
        print("top option estimates", json.dumps(arm["final_option_values"][:3]))


if __name__ == "__main__":
    main()
