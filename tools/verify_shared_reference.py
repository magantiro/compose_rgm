"""Verify the one frozen molecular reference named by all task examples."""

from __future__ import annotations

import json
from pathlib import Path

from compose_v4.model.reference_checkpoint import load_frozen_reference

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments/reference/model.json"


def verify() -> dict:
    spec = json.loads(MANIFEST.read_text())
    checkpoint = spec["checkpoint"]
    catalog = spec["catalog"]
    reference = load_frozen_reference(
        MANIFEST.parent / checkpoint["path"],
        expected_sha256=checkpoint["sha256"],
        expected_catalog_fingerprint=catalog["fingerprint"],
        catalog_path=MANIFEST.parent / catalog["path"],
        expected_catalog_sha256=catalog["sha256"],
    )
    fragment = json.loads((ROOT / "experiments/fragments/assets.json").read_text())
    if (
        fragment["assets"]["checkpoint"]["sha256"] != reference.checkpoint_sha256
        or fragment["assets"]["catalog"]["sha256"] != reference.catalog_sha256
        or fragment["catalog_fingerprint"] != reference.catalog_fingerprint
    ):
        raise ValueError("fragment assets do not name the shared reference")
    configs = (
        "experiments/pmo/example.json",
        "experiments/pmo/uniform_chain.json",
        "experiments/pmo/created_atom_rebinding.json",
        "experiments/t4/example.json",
    )
    for filename in configs:
        task = json.loads((ROOT / filename).read_text())
        asset = task["reference"]
        if (
            asset["sha256"] != reference.checkpoint_sha256
            or asset["catalog_sha256"] != reference.catalog_sha256
            or asset["catalog_fingerprint"] != reference.catalog_fingerprint
        ):
            raise ValueError(f"{filename} does not name the shared reference")
    return {
        "schema": "compose.reference_verification.v1",
        "status": "verified",
        "checkpoint_sha256": reference.checkpoint_sha256,
        "catalog_sha256": reference.catalog_sha256,
        "catalog_fingerprint": reference.catalog_fingerprint,
        "process_semantics": reference.model.editing_process_semantics,
        "max_active_atoms": reference.max_active_atoms,
        "task_configs": ["experiments/fragments/assets.json", *configs],
        "oracle_calls": 0,
        "docking_calls": 0,
    }


def main() -> None:
    print(json.dumps(verify(), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
