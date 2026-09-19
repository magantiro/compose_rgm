#!/usr/bin/env python3
"""Build and validate the exact-source four-call execution package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compose_v4.experiments.t4_nodistill_topology_four_call_contract import (
    CAPSULE_INCLUDE,
    CAPSULE_MANIFEST_RELATIVE_PATH,
    CAPSULE_ROOT_RELATIVE_PATH,
    EXECUTION_CONTRACT_RELATIVE_PATH,
    build_exact_commit_capsule,
    seal_execution_contract,
    validate_execution_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("capsule", "seal", "verify"))
    parser.add_argument("--revision", default="HEAD")
    arguments = parser.parse_args()
    capsule_root = ROOT / CAPSULE_ROOT_RELATIVE_PATH
    capsule_manifest = ROOT / CAPSULE_MANIFEST_RELATIVE_PATH
    execution_contract = ROOT / EXECUTION_CONTRACT_RELATIVE_PATH
    if arguments.mode == "capsule":
        result = build_exact_commit_capsule(
            repository_root=ROOT,
            revision=arguments.revision,
            include=CAPSULE_INCLUDE,
            capsule_root=capsule_root,
            manifest_path=capsule_manifest,
        )
    elif arguments.mode == "seal":
        result = seal_execution_contract(
            repository_root=ROOT,
            capsule_root=capsule_root,
            capsule_manifest_path=capsule_manifest,
            output_path=execution_contract,
        )
    else:
        execution, identity, scientific = validate_execution_contract(ROOT)
        result = {
            "status": execution["status"],
            "execution_contract_payload_sha256": identity,
            "scientific_contract_payload_sha256": execution[
                "scientific_scored_contract"
            ]["payload_sha256"],
            "authorized_scored_calls": scientific["budget"][
                "total_charged_call_ceiling"
            ],
            "modal_calls_created": 0,
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
