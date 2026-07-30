#!/usr/bin/env python3
"""Build and freeze the checkpoint-independent A2.2b exact-graph audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compose_v4.experiments.e6_graph_audit import (
    freeze_a2_2b_artifact,
    load_a2_2b_contract,
    run_a2_2b_graph_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/exact_control_a2_2b_graph_audit_v1.json"),
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
            "diagnostics/exactness/"
            "e6_carbon_6_slots_a2_2b_graph_audit_v1_2026-07-30.json"
        ),
    )
    arguments = parser.parse_args()

    contract = load_a2_2b_contract(
        arguments.contract,
        registry_path=arguments.registry,
    )
    artifact = run_a2_2b_graph_audit(
        contract,
        registry_path=arguments.registry,
    )
    freeze_a2_2b_artifact(
        artifact,
        arguments.output,
        contract=contract,
        registry_path=arguments.registry,
    )
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "paper_claim_authorized": False,
                "checkpoint_used": None,
                "control_solver_run": False,
                "structural_graph_fingerprint": artifact["benchmark"][
                    "structural_graph_fingerprint"
                ],
                "graph_audit_hash": artifact["graph_audit_hash"],
                "artifact_sha256": artifact["artifact_sha256"],
                "n_states": artifact["benchmark"]["n_states"],
                "n_edges": artifact["benchmark"]["n_edges"],
                "output": str(arguments.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
