"""Reconcile the workshop T4 table with saved controller rows, without docking."""

import argparse
import json
import platform
import statistics
import subprocess
from collections import Counter
from pathlib import Path

from compose_v4.experiments.continuation_profile import publish_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]


def analyze(root):
    table_path = root / "diagnostics/t4_combined_table.json"
    table = json.loads(table_path.read_text())
    new_path = root / table["sources"]["new_panel"]
    old_path = root / table["sources"]["old_panel"]
    new = json.loads(new_path.read_text())["runs"]
    old = {
        f"{r['target']}_s{r['idx']}_d{r['delta']}": r
        for r in json.loads(old_path.read_text())["runs"]
    }
    rows = []
    for row in table["rows"]:
        selected = [
            r
            for r in new
            if r["cell"] == row["cell"]
            and r["arm"] == "pooled"
            and r["acts"] == 22
            and int(r["refine"] or 0) == 0
            and int(r["rounds"]) * int(r["dpr"]) <= 100
        ]
        scores = [r["best"] for r in selected if r["best"] is not None]
        mean = round(statistics.mean(scores), 3) if scores else None
        previous = old.get(row["cell"])
        old_score = previous["best_ds"] if previous and previous.get("reached_feasible") else None
        choices = [
            (v, name) for v, name in ((mean, "new@100"), (old_score, "old@200")) if v is not None
        ]
        combined, source = min(choices) if choices else (None, None)
        expected = (
            row["new100"],
            row["old200"],
            row["combined"],
            row["source"],
            row["new_total_runs"],
            row["new_feasible_runs"],
        )
        if (mean, old_score, combined, source, len(selected), len(scores)) != expected:
            raise ValueError(f"saved table/controller lineage disagreement: {row['cell']}")
        rows.append(
            {
                "cell": row["cell"],
                "source": source,
                "combined_reported_score": combined,
                "compact_controller_feasible_mean": mean,
                "old_controller_score": old_score,
                "compact_runs": selected,
                "compact_feasible_runs": len(scores),
                "compact_reported_docking_calls": sum(r["docked"] for r in selected),
                "old_declared_budget": None if previous is None else previous["budget"],
            }
        )
    inputs = [
        table_path,
        new_path,
        old_path,
        root / "scripts/t4_combined_table.py",
        root / "docs/T4_CONTROLLER_CONCLUSION_2026-08-27.md",
    ]
    return {
        "schema_version": "t4_controller_lineage_v1",
        "evidence_class": "computed reconciliation of historical reported run summaries, not fresh oracle validation",
        "input_paths_sha256": {str(p.relative_to(root)): sha256_file(p) for p in inputs},
        "producer_sha256": sha256_file(Path(__file__)),
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "configuration": {
            "arm": "pooled",
            "actions": 22,
            "refine": 0,
            "maximum_planned_calls": 100,
        },
        "python": platform.python_version(),
        "new_oracle_calls": 0,
        "rows": rows,
        "selected_source_counts": dict(Counter(r["source"] or "no_feasible_result" for r in rows)),
        "compact_run_count_distribution": dict(Counter(len(r["compact_runs"]) for r in rows)),
        "interpretation": "The workshop table selects between controller generations and averages only feasible compact runs; neither a single-controller 500-call run nor a uniformly replicated matched comparison. Preserve component baselines and verify their oracle configuration before new comparative claims.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(ROOT)
    publish_json(args.output, result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "selected_source_counts",
                    "compact_run_count_distribution",
                    "new_oracle_calls",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
