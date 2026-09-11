"""Prepare, durably launch and monitor the bounded continuation comparison."""

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "launch", "status"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from compose_v4.control.docking_value import identity
    from compose_v4.experiments.continuation_profile import publish_json, sha256_file
    from compose_v4.experiments.pmo_continuation_choice import (
        APP,
        APP_NAME,
        KIND,
        load_contract,
        prepare,
    )

    if args.mode == "prepare":
        contract = prepare(ROOT)
        print(
            json.dumps(
                {
                    k: contract[k]
                    for k in ("contract_sha256", "new_oracle_limit", "prepared", "compute")
                }
            )
        )
        return
    if args.output is None or args.output.resolve().is_relative_to(ROOT):
        parser.error("launch/status requires --output outside the serialized source tree")
    import modal

    if args.mode == "launch":
        from modal_apps.run_process_v2_p50_app import local_image_revision
        from tools.preflight import assert_synced

        state = assert_synced(strict=True)
        contract = load_contract(ROOT)
        body = {
            "contract_sha256": contract["contract_sha256"],
            "image_revision": local_image_revision(expected_commit=state["commit"]),
            "app_sha256": sha256_file(ROOT / APP),
        }
        task = {**body, "run_id": identity(body)}
        receipt = args.output / task["run_id"] / "launch_spawn.json"
        if receipt.exists():
            raise SystemExit("spawn receipt exists; inspect it rather than launching a duplicate")
        payload = {
            "schema_version": "pmo_continuation_spawn_v1",
            "run_id": task["run_id"],
            "volume": "compose-v4-artifacts",
            "prefix": f"{KIND}/{task['run_id']}",
            "task": task,
            "status": "spawn_started",
        }
        publish_json(receipt, payload)
        call = modal.Function.from_name(APP_NAME, "run").spawn(task)
        payload.update(call_id=call.object_id, status="spawned")
        publish_json(receipt, payload)
        print(receipt)
        print(call.object_id)
        return
    receipt = json.loads(args.output.read_text())
    try:
        result = modal.FunctionCall.from_id(receipt["call_id"]).get(timeout=0)
    except TimeoutError:
        target = args.output.parent / "heartbeat.json"
        fetched = subprocess.run(
            [
                "modal",
                "volume",
                "get",
                receipt["volume"],
                f"{receipt['prefix']}/heartbeat.json",
                str(target),
                "--force",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        print(
            target.read_text()
            if fetched.returncode == 0
            else f"pending heartbeat: {fetched.stderr.strip()}"
        )
    else:
        target = args.output.parent / "result.json"
        publish_json(target, result)
        print(target)
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "status",
                        "new_oracle_calls",
                        "seconds",
                        "changed_choices",
                        "arms",
                        "best_seen",
                    )
                }
            )
        )


if __name__ == "__main__":
    main()
