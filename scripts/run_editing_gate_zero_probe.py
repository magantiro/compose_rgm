#!/usr/bin/env python3
"""Run and freeze the real bounded editing Gate-0 structural probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compose_v4.data.active8_trace_inventory import (
    load_active8_trace_admission,
)
from compose_v4.experiments.editing_gate_zero_runtime import (
    freeze_gate_zero_runtime_evidence,
    load_gate_zero_runtime_contract,
    run_gate_zero_runtime,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--transfer-root", type=Path, required=True)
    parser.add_argument("--semantic-sidecar", type=Path, required=True)
    parser.add_argument(
        "--semantic-sidecar-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--active8-inventory", type=Path, required=True)
    parser.add_argument(
        "--active8-inventory-file-sha256",
        required=True,
        help="Expected SHA-256 of the physical Active8 inventory manifest.",
    )
    parser.add_argument("--output-cache-directory", type=Path, required=True)
    parser.add_argument("--output-evidence-manifest", type=Path, required=True)
    args = parser.parse_args()

    contract = load_gate_zero_runtime_contract(args.contract)
    active8_admission = load_active8_trace_admission(
        args.active8_inventory,
        expected_manifest_file_sha256=args.active8_inventory_file_sha256,
        expected_support_contract_sha256=contract.sha256,
    )
    result = run_gate_zero_runtime(
        contract=contract,
        transfer_root=args.transfer_root,
        sidecar_path=args.semantic_sidecar,
        sidecar_manifest_path=args.semantic_sidecar_manifest,
        active8_admission=active8_admission,
    )
    manifest = freeze_gate_zero_runtime_evidence(
        result,
        cache_directory=args.output_cache_directory,
        evidence_manifest_path=args.output_evidence_manifest,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "training_authorized": False,
                "training_decision": None,
                "optimizer_steps": 0,
                "evidence_manifest": str(args.output_evidence_manifest),
                "evidence_sha256": manifest["evidence_sha256"],
                "probe_sha256": manifest["probe"]["probe_sha256"],
                "selected_trace_count": len(manifest["probe"]["selected_path_addresses"]),
                "selected_progress_row_count": len(manifest["probe"]["rows"]),
                "cache_artifact_count": len(manifest["cache_artifacts"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
