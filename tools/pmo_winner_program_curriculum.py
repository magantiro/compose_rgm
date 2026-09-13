"""Prepare or score the bounded winner-informed Perindopril curriculum."""

import argparse
import json
from pathlib import Path

from compose_v4.experiments.pmo_winner_program_curriculum import OUTPUT, prepare, run

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, default=ROOT / OUTPUT)
    args = parser.parse_args()
    result = prepare(ROOT, args.output) if args.action == "prepare" else run(ROOT, args.output)
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "structural_gate",
                    "oracle_calls",
                    "best_score",
                    "final_top10",
                    "auc_top10_official_10k",
                    "comparisons",
                )
                if key in result
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
