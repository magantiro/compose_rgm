"""Read the spawned diagnostic's durable receipts; never launch or retry work."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

from compose_v4.experiments.continuation_profile import publish_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    destination = Path(__file__).resolve().parent / "attempt_1"
    receipt_path = destination / "launch.json"
    if args.receipt:
        incoming = json.loads(args.receipt.read_text())
        if receipt_path.exists() and incoming != json.loads(receipt_path.read_text()):
            raise ValueError("refusing to mix distinct diagnostic launches")
        publish_json(receipt_path, incoming)
    receipt = json.loads(receipt_path.read_text())
    if receipt["oracle_calls"] != 0 or len(receipt["cases"]) != 3:
        raise ValueError("expected exactly three zero-oracle workers")
    volume = modal.Volume.from_name(receipt["volume"])

    def get(case):
        directory = destination / f"case_{case['case_index']}"
        statuses = {}
        for name in (
            "result",
            "failure",
            "heartbeat",
            "runtime_gate",
            "law_cache_inventory",
            "generation_lock",
            "scored_lock",
        ):
            remote = case["volume_path"].lstrip("/") + f"/{name}.json"
            try:
                payload = json.loads(b"".join(volume.read_file(remote)))
            except FileNotFoundError:
                continue
            # Canonical bytes match the producer's publication convention.
            publish_json(directory / f"{name}.json", payload)
            if name in ("result", "failure", "heartbeat"):
                statuses[name] = payload
        if "failure" in statuses:
            return {"case": case["case_index"], "status": "failed", **statuses["failure"]}
        if "result" in statuses:
            result = statuses["result"]
            return {
                "case": case["case_index"],
                "status": "complete",
                **{
                    k: result.get(k)
                    for k in ("completed", "unique_new_eligible", "elapsed_seconds", "oracle_calls")
                },
            }
        return {"case": case["case_index"], "status": "pending", **statuses.get("heartbeat", {})}

    with ThreadPoolExecutor(max_workers=3) as pool:
        for row in pool.map(get, receipt["cases"]):
            print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
