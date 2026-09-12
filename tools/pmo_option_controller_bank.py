"""Prepare, durably spawn, and inspect the zero-oracle option bank."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "launch", "status"))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from compose_v4.control.docking_value import identity
    from compose_v4.experiments.continuation_profile import publish_json, sha256_file
    from compose_v4.experiments.pmo_option_controller_bank import (
        APP,
        APP_NAME,
        CONTRACT,
        KIND,
        PREPARED,
        PROTOCOL,
        load_contract,
        prepare,
    )

    if args.mode == "prepare":
        print(json.dumps(prepare(ROOT, args.report), sort_keys=True))
        return
    if args.output is None or args.output.resolve().is_relative_to(ROOT):
        parser.error("launch/status output must be outside the serialized tree")

    import modal

    if args.mode == "launch":
        from modal_apps.run_process_v2_p50_app import local_image_revision
        from tools.preflight import assert_synced

        preflight = assert_synced(strict=True)
        contract = load_contract(ROOT, require_proposal_authority=True)
        revision = local_image_revision(
            expected_commit=preflight["commit"], repo_root=ROOT
        )
        body = {
            "contract_sha256": contract["contract_sha256"],
            "image_revision": revision,
            "app_sha256": sha256_file(ROOT / APP),
        }
        task = {**body, "run_id": identity(body)}
        folder = args.output / task["run_id"]
        receipt_path = folder / "spawn.json"
        if receipt_path.exists():
            raise SystemExit("existing spawn receipt: inspect it, do not duplicate")
        folder.mkdir(parents=True, exist_ok=True)
        source_paths = sorted(
            set(revision["serialized_sources"])
            | {
                APP,
                PROTOCOL,
                PREPARED,
                CONTRACT,
                "modal_apps/pmo_online_policy_app.py",
                "docs/PMO_ONLINE_POLICY.md",
                "diagnostics/pmo_online_policy/prepared.json",
                "diagnostics/pmo_branch_policy/prepared.json",
                "diagnostics/pmo_branch_policy/result_sealed.json",
                "diagnostics/pmo_inference_speed/result_sealed.json",
                "diagnostics/pmo_inference_speed/package_manifest.json",
                "diagnostics/pmo_inference_speed/reference_law_sealed.json",
            }
        )
        archive = folder / "source_snapshot.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for path in source_paths:
                tar.add(ROOT / path, arcname=path, recursive=False)
        receipt = {
            "schema_version": "pmo_option_controller_bank_spawn_v1",
            "task": task,
            "status": "spawn_started",
            "run_id": task["run_id"],
            "prefix": f"{KIND}/{task['run_id']}",
            "volume": "compose-v4-artifacts",
            "source_snapshot_sha256": sha256_file(archive),
        }
        publish_json(receipt_path, receipt)
        call = modal.Function.from_name(APP_NAME, "run").spawn(task)
        receipt.update(status="spawned", call_id=call.object_id)
        publish_json(receipt_path, receipt)
        print(receipt_path)
        print(call.object_id)
        return

    receipt = json.loads(args.output.read_text())
    try:
        result = modal.FunctionCall.from_id(receipt["call_id"]).get(timeout=0)
    except TimeoutError:
        volume = modal.Volume.from_name(receipt["volume"])
        prefix = receipt["prefix"]
        try:
            entries = {
                entry.path.rsplit("/", 1)[-1] for entry in volume.listdir(prefix)
            }
        except modal.exception.NotFoundError:
            print("Parent call live; waiting for its first durable receipt")
            return
        if "heartbeat.json" in entries:
            print(json.loads(b"".join(volume.read_file(f"{prefix}/heartbeat.json"))))
        else:
            workers = sum(name == "complete.json" for name in entries)
            print(
                f"Parent call live; initializing (top-level complete receipts: {workers})"
            )
    else:
        result_path = args.output.parent / "result.json"
        publish_json(result_path, result)
        print(result_path)
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
