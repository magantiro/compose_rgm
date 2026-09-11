"""Prepare, durably launch, and inspect the target-free value probe."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "launch", "status"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from compose_v4.control.docking_value import identity
    from compose_v4.experiments.continuation_profile import publish_json, sha256_file
    from compose_v4.experiments.pmo_trajectory_value import (
        APP,
        APP_NAME,
        KIND,
        load_contract,
        prepare,
    )

    if args.mode == "prepare":
        c = prepare(ROOT)
        print(json.dumps({k: c[k] for k in ("contract_sha256", "new_oracle_limit", "compute")}))
        return
    if args.output is None or args.output.resolve().is_relative_to(ROOT):
        parser.error("launch/status requires --output outside serialized source tree")
    import modal

    if args.mode == "launch":
        from modal_apps.run_process_v2_p50_app import local_image_revision
        from tools.preflight import assert_synced

        state = assert_synced(strict=True)
        c = load_contract(ROOT)
        body = {
            "contract_sha256": c["contract_sha256"],
            "image_revision": local_image_revision(expected_commit=state["commit"]),
            "app_sha256": sha256_file(ROOT / APP),
        }
        task = {**body, "run_id": identity(body)}
        path = args.output / task["run_id"] / "spawn.json"
        if path.exists():
            raise SystemExit("spawn exists; inspect instead of duplicating")
        receipt = {
            "schema_version": "trajectory_spawn_v1",
            "task": task,
            "run_id": task["run_id"],
            "volume": "compose-v4-artifacts",
            "prefix": f"{KIND}/{task['run_id']}",
            "status": "spawn_started",
        }
        publish_json(path, receipt)
        call = modal.Function.from_name(APP_NAME, "run").spawn(task)
        receipt.update(call_id=call.object_id, status="spawned")
        publish_json(path, receipt)
        print(path)
        print(call.object_id)
        return
    receipt = json.loads(args.output.read_text())
    try:
        result = modal.FunctionCall.from_id(receipt["call_id"]).get(timeout=0)
    except TimeoutError:
        content = b"".join(
            modal.Volume.from_name(receipt["volume"]).read_file(
                f"{receipt['prefix']}/heartbeat.json"
            )
        )
        print(content.decode())
    else:
        path = args.output.parent / "result.json"
        publish_json(path, result)
        print(path)
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "status",
                        "new_oracle_calls",
                        "training_calls",
                        "evaluation_calls",
                        "seconds",
                        "arms",
                    )
                }
            )
        )


if __name__ == "__main__":
    main()
