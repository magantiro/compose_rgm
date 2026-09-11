"""Prepare, durably spawn and retrieve the zero-oracle support repair."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "launch", "status"))
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from compose_v4.control.docking_value import identity
    from compose_v4.experiments.continuation_profile import publish_json, sha256_file
    from compose_v4.experiments.pmo_route_support import APP, APP_NAME, KIND, load_contract, prepare
    from compose_v4.experiments.t4_matched_pilot import _stamp

    if args.mode == "prepare":
        if args.prior is None:
            parser.error("prepare requires the prior spawned coverage receipt")
        print(json.dumps(prepare(ROOT, args.prior)))
        return
    if args.output is None or args.output.resolve().is_relative_to(ROOT):
        parser.error("launch/status output must be outside the serialized tree")
    import modal

    if args.mode == "launch":
        from modal_apps.run_process_v2_p50_app import local_image_revision
        from tools.preflight import assert_synced

        commit = assert_synced(strict=True)["commit"]
        c, _ = load_contract(ROOT)
        body = {
            "contract_sha256": c["contract_sha256"],
            "image_revision": local_image_revision(expected_commit=commit),
            "app_sha256": sha256_file(ROOT / APP),
        }
        task = {**body, "run_id": identity(body), "started_at": _stamp()}
        path = args.output / task["run_id"] / "spawn.json"
        if path.exists():
            raise SystemExit("existing spawn receipt: inspect, do not duplicate")
        receipt = {
            "schema_version": "route_support_spawn_v1",
            "task": task,
            "run_id": task["run_id"],
            "status": "spawn_started",
            "prefix": f"{KIND}/{task['run_id']}",
            "volume": "compose-v4-artifacts",
        }
        publish_json(path, receipt)
        call = modal.Function.from_name(APP_NAME, "run").spawn(task)
        receipt.update(status="spawned", call_id=call.object_id)
        publish_json(path, receipt)
        print(path)
        print(call.object_id)
        return
    receipt = json.loads(args.output.read_text())
    try:
        result = modal.FunctionCall.from_id(receipt["call_id"]).get(timeout=0)
    except TimeoutError:
        volume = modal.Volume.from_name(receipt["volume"])
        for name in ("current_best", *(f"original_root_{i}" for i in range(4))):
            prefix = f"{receipt['prefix']}/{name}"
            entries = {e.path.rsplit("/", 1)[-1] for e in volume.listdir(prefix)}
            file = "result.json" if "result.json" in entries else "heartbeat.json"
            if file not in entries:
                print(name, "initializing")
                continue
            row = json.loads(b"".join(volume.read_file(f"{prefix}/{file}")))
            row = row.get("payload", row)
            print(
                json.dumps(
                    {
                        k: row[k]
                        for k in (
                            "source",
                            "status",
                            "phase",
                            "completed_transitions",
                            "completed_original_steps",
                            "primitive_steps",
                            "fresh_laws",
                            "seconds",
                        )
                        if k in row
                    }
                )
            )
    else:
        path = args.output.parent / "result.json"
        publish_json(path, result)
        print(path)
        print(
            json.dumps(
                [
                    {
                        k: r[k]
                        for k in (
                            "source",
                            "status",
                            "original_steps",
                            "completed_original_steps",
                            "primitive_steps",
                            "seconds",
                            "law_work",
                        )
                    }
                    for r in result["results"]
                ]
            )
        )


if __name__ == "__main__":
    main()
