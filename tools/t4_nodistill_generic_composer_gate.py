"""Generate or evaluate the zero-oracle NoDistill generic-composer gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compose_v4.experiments.t4_nodistill_generic_composer_gate import (
    ARTIFACT_ROOT_RELATIVE_PATH,
    evaluate_locks,
    lock_cell,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    lock = subparsers.add_parser("lock-cell")
    lock.add_argument("--cell-key", required=True)
    lock.add_argument("--output", type=Path)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--artifact-root", type=Path)
    evaluate.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    artifact_root = (
        root / ARTIFACT_ROOT_RELATIVE_PATH
        if getattr(arguments, "artifact_root", None) is None
        else arguments.artifact_root.resolve()
    )
    if arguments.command == "lock-cell":
        output = (
            artifact_root / "locks" / f"{arguments.cell_key}.json.gz"
            if arguments.output is None
            else arguments.output.resolve()
        )
        result = lock_cell(
            repository_root=root,
            cell_key=arguments.cell_key,
            output_path=output,
        )
    else:
        output = (
            artifact_root / "result.json"
            if arguments.output is None
            else arguments.output.resolve()
        )
        result = evaluate_locks(
            repository_root=root,
            artifact_root=artifact_root,
            output_path=output,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
