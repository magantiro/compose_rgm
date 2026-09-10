"""Prepare, durably launch, and inspect the bounded archive-controller pilot."""

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
    parser.add_argument("--prior-launch", type=Path)
    args = parser.parse_args()
    from compose_v4.control.docking_value import identity
    from compose_v4.experiments.continuation_profile import publish_json, sha256_file
    from compose_v4.experiments.pmo_archive_pilot import (
        KIND,
        case_name,
        cases,
        load_contract,
        prepare_contract,
    )

    if args.mode == "prepare":
        if args.prior_launch is None:
            parser.error("prepare requires --prior-launch")
        contract = prepare_contract(ROOT, args.prior_launch)
        print(contract["contract_sha256"])
        return
    import modal

    if args.mode == "launch":
        from modal_apps.run_process_v2_p50_app import local_image_revision
        from tools.preflight import assert_synced

        state = assert_synced(strict=True)
        contract = load_contract(ROOT)
        body = {
            "contract_sha256": contract["contract_sha256"],
            "image_revision": local_image_revision(expected_commit=state["commit"]),
            "app_sha256": sha256_file(ROOT / "modal_apps/pmo_archive_pilot_app.py"),
        }
        run_id = identity(body)
        receipt = args.output / run_id / "spawn.json"
        if receipt.exists():
            raise SystemExit(
                "spawn receipt exists; inspect or explicitly resume, never duplicate launch"
            )
        payload = {
            "schema_version": "pmo_archive_spawn_v1",
            "run_id": run_id,
            "volume": "compose-v4-artifacts",
            "prefix": f"{KIND}/{run_id}",
            "cases": [],
        }
        publish_json(receipt, payload)
        function = modal.Function.from_name("compose-pmo-archive-pilot", "archive_case")
        for case in cases(contract):
            call = function.spawn({**body, "run_id": run_id, "case": case})
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
        try:
            result = modal.FunctionCall.from_id(row["call_id"]).get(timeout=0)
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
        except (ImportError, RuntimeError, ValueError) as error:
            print(name, "FAILED", type(error).__name__, str(error), flush=True)
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
                        "top10_mean",
                        "auc_top10",
                        "seconds",
                    )
                },
            )


if __name__ == "__main__":
    main()
