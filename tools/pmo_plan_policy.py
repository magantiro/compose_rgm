"""Prepare, admit, launch, and inspect the learned-plan comparison."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "check", "launch", "status"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from compose_v4.control.docking_value import identity
    from compose_v4.experiments.continuation_profile import publish_json, sha256_file
    from compose_v4.experiments.pmo_plan_policy import APP_NAME, KIND, load_contract, prepare

    if args.mode == "prepare":
        print(json.dumps(prepare(ROOT), indent=2))
        return
    if args.output is None:
        parser.error("check, launch and status require --output outside source")
    import modal

    if args.mode in ("check", "launch"):
        from modal_apps.run_process_v2_p50_app import local_image_revision
        from tools.preflight import assert_synced

        state = assert_synced(strict=True)
        contract = load_contract(ROOT)
        body = {
            "contract_sha256": contract["contract_sha256"],
            "image_revision": local_image_revision(expected_commit=state["commit"]),
            "app_sha256": sha256_file(ROOT / "modal_apps/pmo_plan_policy_app.py"),
        }
        task = {**body, "run_id": identity(body)}
        receipt = args.output / task["run_id"] / f"{args.mode}_spawn.json"
        if receipt.exists():
            raise SystemExit("spawn receipt exists; inspect it instead of duplicating the run")
        payload = {
            "schema_version": "pmo_plan_policy_spawn_v1",
            "run_id": task["run_id"],
            "volume": "compose-v4-artifacts",
            "prefix": f"{KIND}/{task['run_id']}",
            "task": task,
            "status": "spawn_started",
        }
        publish_json(receipt, payload)
        function = "check" if args.mode == "check" else "run"
        call = modal.Function.from_name(APP_NAME, function).spawn(task)
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
        print(target.read_text() if fetched.returncode == 0 else fetched.stderr.strip())
    else:
        target = args.output.parent / args.output.name.replace("_spawn.json", "_result.json")
        publish_json(target, result)
        print(target)
        if "arms" in result:
            print(
                json.dumps(
                    {
                        "status": result["status"],
                        "new_oracle_calls": result["new_oracle_calls"],
                        "seconds": result["seconds"],
                        "arms": result["arms"],
                        "policy_updates": [
                            {arm: row["policy_update"] for arm, row in round_["arms"].items()}
                            for round_ in result["rounds"]
                        ],
                    },
                    indent=2,
                )
            )
        else:
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
