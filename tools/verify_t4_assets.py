"""Verify the five T4 program template priors without running docking."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "experiments" / "t4" / "assets.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def verify_t4_assets(manifest_path: Path, *, asset_root: Path | None = None) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "compose_t4_route_assets_v1":
        raise ValueError(f"{manifest_path}: unsupported T4 asset schema")
    experts = manifest.get("route_experts")
    if not isinstance(experts, dict) or set(experts) != {"5ht1b", "braf", "fa7", "jak2", "parp1"}:
        raise ValueError(f"{manifest_path}: expected exactly five T4 program template priors")
    checked = {}
    base = manifest_path.parent if asset_root is None else asset_root
    for target, specification in sorted(experts.items()):
        path = base / specification["file"]
        if not path.is_file():
            raise FileNotFoundError(f"missing {target} program template prior: {path}")
        digest = _sha256(path)
        if digest != specification["sha256"]:
            raise ValueError(
                f"{target} program template prior SHA-256 mismatch: {path}; got {digest}"
            )
        envelope = json.loads(path.read_text())
        if not isinstance(envelope, dict) or set(envelope) != {"payload", "payload_sha256"}:
            raise ValueError(f"{path}: invalid program template prior envelope")
        payload = envelope["payload"]
        if not isinstance(payload, dict) or _payload_hash(payload) != envelope["payload_sha256"]:
            raise ValueError(f"{path}: program template prior payload identity mismatch")
        split = payload.get("split_audit")
        if not isinstance(split, dict) or split.get("split") != "leave_one_target_out":
            raise ValueError(f"{path}: missing leave-one-target-out split audit")
        if split.get("held_target_absent_from_training") is not True:
            raise ValueError(f"{path}: held target was not excluded from training")
        if payload.get("new_oracle_calls") != 0:
            raise ValueError(f"{path}: program template prior unexpectedly records oracle calls")
        checked[target] = {"path": str(path), "sha256": digest}
    return {"schema": "compose_t4_asset_check_v1", "status": "verified", "experts": checked}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--asset-root", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            verify_t4_assets(args.manifest, asset_root=args.asset_root),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
