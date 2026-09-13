"""Prepare or run the bounded PMO formula/median panel wave."""

import argparse
import json
from pathlib import Path

from compose_v4.experiments.pmo_formula_median_panel_wave import OUTPUT, prepare, run

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
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("structural_gate", "oracle_calls", "summary")
                if key in result
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
