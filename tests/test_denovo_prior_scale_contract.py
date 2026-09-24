"""Fail-closed identity gates for the frozen de novo training comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from compose_v4.data.frozen_cnof_prior_split import frozen_cnof_arm_identity

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/denovo_native_prior_scale_training_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def test_training_contract_is_self_hashed_and_all_local_inputs_match():
    envelope = json.loads(CONTRACT.read_text())
    payload = envelope["payload"]
    actual_payload_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert envelope["payload_sha256"] == actual_payload_hash

    for arm in payload["arms"].values():
        assert _sha256(ROOT / arm["recipe"]) == arm["recipe_sha256"]
    for relative_path, expected in payload["implementation_sha256"].items():
        assert _sha256(ROOT / relative_path) == expected

    data = payload["frozen_data"]
    manifest = ROOT / data["manifest_path_in_repo"]
    assert _sha256(manifest) == data["manifest_sha256"]
    for arm in payload["arms"].values():
        identity = frozen_cnof_arm_identity(manifest, train_size=arm["train_size"])
        for key in (
            "validation_sha256",
            "iid_test_sha256",
            "scaffold_test_sha256",
            "shared_source_prior_prefix_size",
            "shared_source_prior_sha256",
        ):
            assert identity[key] == data[key]
