#!/usr/bin/env python3
"""Seal or authorize the exact nine-cell T4 scored contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compose_v4.experiments.t4_shared_controller_completion_contract import (
    build_exact_commit_capsule,
)
from compose_v4.experiments.t4_shared_controller_scored_contract import (
    AUTHORIZATION_RELATIVE_PATH,
    CAPSULE_INCLUDE,
    FINAL_CONTRACT_RELATIVE_PATH,
    publish_authorization_receipt,
    seal_scored_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capsule-root", type=Path)
    parser.add_argument("--capsule-manifest", type=Path)
    parser.add_argument("--build-capsule", action="store_true")
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument(
        "--reconciliation",
        type=Path,
        default=ROOT / "diagnostics/t4_shared_controller_completion_v1/"
        "stale_braf_v4_reconciliation.json",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / FINAL_CONTRACT_RELATIVE_PATH
    )
    parser.add_argument("--authorize", action="store_true")
    parser.add_argument("--user-statement")
    arguments = parser.parse_args()
    if arguments.authorize and arguments.build_capsule:
        raise ValueError("--authorize and --build-capsule are mutually exclusive")
    if arguments.build_capsule:
        if arguments.capsule_root is None or arguments.capsule_manifest is None:
            raise ValueError(
                "--build-capsule requires --capsule-root and --capsule-manifest"
            )
        result = build_exact_commit_capsule(
            repository_root=ROOT,
            revision=arguments.revision,
            include=CAPSULE_INCLUDE,
            capsule_root=arguments.capsule_root,
            manifest_path=arguments.capsule_manifest,
        )
    elif arguments.authorize:
        if not arguments.user_statement:
            raise ValueError("--authorize requires --user-statement")
        result = publish_authorization_receipt(
            contract_path=arguments.output,
            authorization_path=ROOT / AUTHORIZATION_RELATIVE_PATH,
            user_statement=arguments.user_statement,
        )
    else:
        if arguments.capsule_root is None or arguments.capsule_manifest is None:
            raise ValueError("sealing requires --capsule-root and --capsule-manifest")
        result = seal_scored_contract(
            repository_root=ROOT,
            capsule_root=arguments.capsule_root,
            capsule_manifest_path=arguments.capsule_manifest,
            reconciliation_path=arguments.reconciliation,
            output_path=arguments.output,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
