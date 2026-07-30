"""Run the bounded RingCore marked-support invariance development gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from compose_v4.experiments.support_invariance_qualification import (
    build_default_ringcore_support_fixture,
    qualify_stratified_support_invariance,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Machine-readable passing diagnostic path.",
    )
    arguments = parser.parse_args()

    models, panel = build_default_ringcore_support_fixture()
    report = qualify_stratified_support_invariance(models, panel)
    repo = Path(__file__).resolve().parents[1]
    implementation_files = (
        "src/compose_v4/experiments/support_invariance_qualification.py",
        "src/compose_v4/experiments/factorized_successor_training.py",
        "src/compose_v4/experiments/production_successor_kernel.py",
        "src/compose_v4/model/factorized_tracelet_rate_model.py",
        "src/compose_v4/rewrite/action_codec.py",
        "src/compose_v4/rewrite/kernel.py",
    )
    payload = {
        **report.to_payload(),
        "implementation_sha256s": {
            relative: hashlib.sha256((repo / relative).read_bytes()).hexdigest()
            for relative in implementation_files
        },
        "evidence_command": (
            "PYTHONPATH=src .venv/bin/python "
            "scripts/run_ringcore_support_invariance_qualification.py "
            f"--output {arguments.output.as_posix()}"
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "output": arguments.output.as_posix(),
                "covered_families": report.covered_families,
                "maximum_probability_l1": report.maximum_probability_l1,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
