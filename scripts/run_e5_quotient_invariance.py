#!/usr/bin/env python3
"""Run the bounded E5 quotient-invariance foundation without making a paper claim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compose_v4.experiments.quotient_invariance import (
    freeze_e5_artifact,
    load_e5_contract,
    run_e5_foundation,
    validate_e5_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/tasks/quotient.development.json"),
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("configs/experiment_registry.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "diagnostics/coherence/"
            "e5_quotient_invariance_foundation_v1_2026-07-30.json"
        ),
    )
    args = parser.parse_args()

    contract = load_e5_contract(args.contract, registry_path=args.registry)
    artifact = run_e5_foundation(contract, registry_path=args.registry)
    validate_e5_artifact(
        artifact,
        contract=contract,
        registry_path=args.registry,
    )
    freeze_e5_artifact(
        artifact,
        args.output,
        contract=contract,
        registry_path=args.registry,
    )
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "paper_claim_authorized": False,
                "checkpoint_used": None,
                "output": str(args.output),
                "artifact_sha256": artifact["artifact_sha256"],
                "state_count": artifact["summary"]["state_count"],
                "all_checks_pass": all(artifact["checks"].values()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
