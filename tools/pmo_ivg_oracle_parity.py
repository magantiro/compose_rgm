"""Prepare or run the no-reselection PMO IVG-oracle parity audit."""

import argparse
import json
from pathlib import Path

from compose_v4.experiments.pmo_ivg_oracle_parity import OUTPUT, prepare, run

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, default=ROOT / OUTPUT)
    args = parser.parse_args()
    result = (
        prepare(ROOT, args.output)
        if args.action == "prepare"
        else run(ROOT, args.output)
    )
    if "results" in result:
        compact = {
            "authoritative_oracle_calls": result["authoritative_oracle_calls"],
            "summary": result["summary"],
            "tasks": {
                task: {
                    key: row[key]
                    for key in (
                        "auc_top10_official_10k",
                        "best_score",
                        "margin_no_prescreen",
                        "margin_prescreen",
                    )
                }
                for task, row in result["results"].items()
            },
        }
    else:
        compact = {
            "task_count": result["task_count"],
            "authoritative_query_count": result["authoritative_query_count"],
            "new_oracle_calls": result["new_oracle_calls"],
        }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
