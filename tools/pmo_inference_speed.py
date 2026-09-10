"""Prepare, export, probe and inspect frozen-model inference acceleration."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def prepare():
    from compose_v4.control.docking_value import identity
    from compose_v4.experiments.continuation_profile import publish_json, sha256_file
    from compose_v4.experiments.inference_package import CACHE_SOURCES
    from compose_v4.experiments.pmo_inference_speed import CONTRACT, PREPARED
    from compose_v4.experiments.t4_matched_pilot import unseal
    from compose_v4.rewrite.trace_shard import decode_state, encode_state

    worker_path = ROOT / "diagnostics/pmo_branch_policy/slow_worker_sealed.json"
    archive_path = ROOT / "diagnostics/pmo_branch_policy/prepared.json"
    worker = unseal(worker_path)
    archive = json.loads(archive_path.read_text())["archive"]
    parent_id = worker["candidates"][0]["chain"][-2]
    parent = next(a for a in archive if a["id"] == parent_id)
    selected = [parent, worker["candidates"][0], worker["candidates"][3]]
    states = [
        {"id": c["id"], "graph": encode_state(decode_state(c["node"]["graph"]))} for c in selected
    ]

    def committed(revision, path):
        return subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT)

    baseline = committed("8ef5eb220435aa76f373b0099fbde858391e7292", CACHE_SOURCES[0])
    data = {
        "schema_version": "inference_speed_states_v1",
        "states": states,
        "baseline_source": baseline.decode(),
        "inputs": {str(p.relative_to(ROOT)): sha256_file(p) for p in (worker_path, archive_path)},
        "selection": "exact slow-worker parent and completed pendant six-ring draws 0 and 3; no score selection",
    }
    publish_json(ROOT / PREPARED, data)
    hashes = json.loads((ROOT / "configs/pmo_archive_pilot.json").read_text())[
        "expected_input_sha256"
    ]
    contract = {
        "schema_version": "inference_speed_contract_v1",
        "oracle_calls": 0,
        "repeats": 3,
        "prepared_sha256": sha256_file(ROOT / PREPARED),
        "protocol_sha256": sha256_file(ROOT / "docs/PMO_INFERENCE_SPEED.md"),
        "baseline_serializer_sha256": hashlib.sha256(baseline).hexdigest(),
        "frozen_process_sha256": "0c938177a34819e6e828920c1f66e240c6eb251fe7c9ea6cfe6757829dceb2dd",
        "cache_sources": {
            p: hashlib.sha256(committed("ff31dbd6e0138243743dd9eadc2e02e27fb511de", p)).hexdigest()
            for p in CACHE_SOURCES
        },
        "reference_inputs": {
            "checkpoint": {
                "path": "/artifacts/editing_v2/r_theta_run/runs/run_v2_01/R_THETA_CHECKPOINT.pt",
                "sha256": hashes["r_theta_checkpoint"],
            },
            "run_paths": {
                "path": "/artifacts/editing_v2/r_theta_run/run_inputs/RUN_PATHS.json",
                "sha256": hashes["r_theta_run_paths"],
            },
        },
        "compute": {
            "workers": 1,
            "cpu": 1,
            "memory_mib": 8192,
            "timeout_seconds": 1200,
            "retries": 0,
        },
    }
    contract["contract_sha256"] = identity(contract)
    publish_json(ROOT / CONTRACT, contract)
    print(
        json.dumps(
            {
                "contract_sha256": contract["contract_sha256"],
                "states": len(states),
                "oracle_calls": 0,
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "export", "probe", "status"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--export-result", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
        return
    if args.output is None:
        parser.error("--output required (directory for launch, spawn receipt for status)")
    import modal

    from compose_v4.control.docking_value import identity
    from compose_v4.experiments.continuation_profile import publish_json, sha256_file
    from compose_v4.experiments.pmo_inference_speed import APP, read_contract

    if args.mode != "status":
        from modal_apps.run_process_v2_p50_app import local_image_revision
        from tools.preflight import assert_synced

        state = assert_synced(strict=True)
        contract = read_contract(ROOT)
        body = {
            "mode": args.mode,
            "contract_sha256": contract["contract_sha256"],
            "image_revision": local_image_revision(expected_commit=state["commit"]),
            "app_sha256": sha256_file(ROOT / APP),
        }
        if args.mode == "probe":
            if args.export_result is None:
                parser.error("probe requires --export-result from verified export")
            body["export"] = json.loads(args.export_result.read_text())
        task = {**body, "run_id": identity(body)}
        receipt = args.output / task["run_id"] / "spawn.json"
        if receipt.exists():
            raise SystemExit("spawn receipt exists; inspect rather than duplicate")
        payload = {
            "schema_version": "inference_speed_spawn_v1",
            "task": task,
            "run_id": task["run_id"],
            "volume": "compose-v4-artifacts",
            "prefix": f"pmo_inference_speed/{task['run_id']}",
            "status": "spawn_started",
        }
        publish_json(receipt, payload)
        call = modal.Function.from_name(
            "compose-pmo-inference-speed", "export_model" if args.mode == "export" else "probe"
        ).spawn(task)
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
            target.read_text() if fetched.returncode == 0 else f"pending: {fetched.stderr.strip()}"
        )
    else:
        target = args.output.parent / "result.json"
        publish_json(target, result)
        print(target)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
