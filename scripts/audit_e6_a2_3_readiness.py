#!/usr/bin/env python3
"""Freeze the fail-closed E6 A2.3 readiness audit; do not run a solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compose_v4.experiments.e6_a2_3_readiness import (
    freeze_a2_3_readiness_artifact,
    load_a2_3_readiness_contract,
    run_a2_3_readiness_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/exact_control_a2_3_readiness_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "diagnostics/exactness/"
            "e6_a2_3_readiness_blocked_v1_2026-07-30.json"
        ),
    )
    arguments = parser.parse_args()

    contract = load_a2_3_readiness_contract(arguments.contract)
    artifact = run_a2_3_readiness_audit(contract)
    freeze_a2_3_readiness_artifact(
        artifact,
        arguments.output,
        contract=contract,
    )
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "control_solver_run": artifact["control_solver_run"],
                "paper_claim_authorized": artifact["paper_claim_authorized"],
                "missing_decisions": [
                    item["decision_id"] for item in artifact["missing_decisions"]
                ],
                "missing_task_artifacts": artifact["missing_task_artifacts"],
                "artifact_sha256": artifact["artifact_sha256"],
                "output": str(arguments.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
