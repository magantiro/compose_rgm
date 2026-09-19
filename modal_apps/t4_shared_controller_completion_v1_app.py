"""Preparation-only entrypoint for the nine-cell shared T4 completion campaign.

There are intentionally no remote functions, image, volume mounts, or scored
launch path in this revision.  A later revision may register the shared cell
runtime only after the nine-cell support artifact and exact authorization exist.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compose_v4.experiments.t4_shared_controller_completion_contract import (
    CONTRACT_RELATIVE_PATH,
    assert_scored_launch_blocked,
    cell_runtime_payload,
    validate_preparation_contract,
    verify_exact_commit_capsule,
)

app = modal.App("compose-t4-shared-controller-completion-v1-preparation")


def preparation_report() -> dict:
    return validate_preparation_contract(ROOT, ROOT / CONTRACT_RELATIVE_PATH)


def runtime_payload(cell_key: str) -> dict:
    contract = json.loads((ROOT / CONTRACT_RELATIVE_PATH).read_text())
    return cell_runtime_payload(contract, cell_key)


@app.local_entrypoint()
def main(
    mode: str = "preflight",
    cell_key: str = "",
    capsule_root: str = "",
    capsule_manifest: str = "",
) -> None:
    if mode == "preflight":
        print(json.dumps(preparation_report(), indent=2, sort_keys=True))
        return
    if mode == "runtime-payload":
        if not cell_key:
            raise ValueError(
                "runtime-payload mode requires one delta-qualified cell key"
            )
        print(json.dumps(runtime_payload(cell_key), indent=2, sort_keys=True))
        return
    if mode == "capsule-preflight":
        if not capsule_root or not capsule_manifest:
            raise ValueError("capsule-preflight requires capsule root and manifest")
        print(
            json.dumps(
                verify_exact_commit_capsule(Path(capsule_root), Path(capsule_manifest)),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if mode == "launch":
        assert_scored_launch_blocked(ROOT, ROOT / CONTRACT_RELATIVE_PATH)
    raise ValueError(
        "use mode=preflight, runtime-payload, or capsule-preflight; scored launch "
        "is not implemented in the preparation revision"
    )


__all__ = ["app", "main", "preparation_report", "runtime_payload"]
