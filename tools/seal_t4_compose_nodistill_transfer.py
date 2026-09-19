"""Build and seal the exact COMPOSE-NoDistill transfer campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compose_v4.experiments.t4_compose_nodistill_transfer_contract import (
    AUTHORIZATION_RELATIVE_PATH,
    CAPSULE_INCLUDE,
    CAPSULE_MANIFEST_RELATIVE_PATH,
    CAPSULE_ROOT_RELATIVE_PATH,
    FINAL_CONTRACT_RELATIVE_PATH,
    build_exact_commit_capsule,
    publish_authorization_receipt,
    seal_scored_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("capsule", "seal", "authorize"))
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--user-statement", default="")
    arguments = parser.parse_args()
    capsule_root = ROOT / CAPSULE_ROOT_RELATIVE_PATH
    capsule_manifest = ROOT / CAPSULE_MANIFEST_RELATIVE_PATH
    contract_path = ROOT / FINAL_CONTRACT_RELATIVE_PATH
    if arguments.mode == "capsule":
        result = build_exact_commit_capsule(
            repository_root=ROOT,
            revision=arguments.revision,
            include=CAPSULE_INCLUDE,
            capsule_root=capsule_root,
            manifest_path=capsule_manifest,
        )
    elif arguments.mode == "seal":
        result = seal_scored_contract(
            repository_root=ROOT,
            capsule_root=capsule_root,
            capsule_manifest_path=capsule_manifest,
            output_path=contract_path,
        )
    else:
        if not arguments.user_statement:
            raise ValueError("--user-statement is required for authorization")
        result = publish_authorization_receipt(
            contract_path=contract_path,
            authorization_path=ROOT / AUTHORIZATION_RELATIVE_PATH,
            user_statement=arguments.user_statement,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
