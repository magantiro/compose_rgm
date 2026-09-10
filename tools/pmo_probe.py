"""Prepare, launch or inspect only the frozen small PMO transfer probe."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "launch", "status"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from compose_v4.control.docking_value import identity
    from compose_v4.experiments.continuation_profile import publish_json, sha256_file
    from compose_v4.experiments.pmo_macro_probe import (
        CONTRACT,
        KIND,
        case_name,
        cases,
        load_contract,
        prepare_contract,
    )

    if args.mode == "prepare":
        contract = prepare_contract(ROOT)
        print(
            json.dumps(
                {
                    "contract": CONTRACT,
                    "tasks": contract["tasks"],
                    "roots": [r["smiles"] for r in contract["roots"]],
                    "case_count": 12,
                    "max_calls_per_case": 44,
                },
                indent=2,
            )
        )
        return
    import modal

    if args.mode == "launch":
        from modal_apps.run_process_v2_p50_app import local_image_revision
        from tools.preflight import assert_synced

        state = assert_synced(strict=True)
        contract = load_contract(ROOT)
        revision = local_image_revision(expected_commit=state["commit"])
        body = {
            "contract_sha256": contract["contract_sha256"],
            "image_revision": revision,
            "app_sha256": sha256_file(ROOT / "modal_apps/pmo_macro_probe_app.py"),
        }
        run_id = identity(body)
        folder = args.output / run_id
        function = modal.Function.from_name("compose-pmo-macro-probe", "probe_case")
        receipt = folder / "spawn.json"
        if receipt.exists():
            raise SystemExit("spawn receipt exists; use status, not duplicate launch")
        payload = {
            "schema_version": "pmo_macro_probe_spawn_v1",
            "run_id": run_id,
            "volume": "compose-v4-artifacts",
            "prefix": f"{KIND}/{run_id}",
            "cases": [],
        }
        publish_json(receipt, payload)
        for case in cases(contract):
            task = {**body, "run_id": run_id, "case": case}
            call = function.spawn(task)
            payload["cases"].append({"case": case, "call_id": call.object_id})
            publish_json(receipt, payload)
            print(case_name(case), call.object_id, flush=True)
        print(receipt)
        return
    receipt = json.loads(args.output.read_text())
    for row in receipt["cases"]:
        name = case_name(row["case"])
        folder = args.output.parent / name
        folder.mkdir(parents=True, exist_ok=True)
        call = modal.FunctionCall.from_id(row["call_id"])
        try:
            result = call.get(timeout=0)
        except TimeoutError:
            process = subprocess.run(
                [
                    "modal",
                    "volume",
                    "get",
                    receipt["volume"],
                    f"{receipt['prefix']}/{name}/heartbeat.json",
                    str(folder / "heartbeat.json"),
                    "--force",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            print(
                name,
                (folder / "heartbeat.json").read_text()
                if process.returncode == 0
                else "pending heartbeat",
            )
        else:
            publish_json(folder / "result.json", result)
            print(
                name,
                {
                    k: result[k]
                    for k in (
                        "status",
                        "oracle_calls",
                        "best",
                        "initial_best",
                        "best_improvement",
                        "seconds",
                    )
                },
            )


if __name__ == "__main__":
    main()
