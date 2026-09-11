"""Offline complete-edit policy development, no oracle calls or remote jobs."""

import argparse
import json
from pathlib import Path

from compose_v4.experiments.pmo_edit_chooser import evaluate, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(
        json.dumps((prepare if args.mode == "prepare" else evaluate)(root, args.output), indent=2)
    )


if __name__ == "__main__":
    main()
