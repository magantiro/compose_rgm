"""Read durable reset progress/results without launching, retrying or scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.experiments.t4_matched_pilot import _stamp, seal, unseal


def main():
    import modal

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("launch_file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    launch = unseal(args.launch_file)
    task = launch["task"]
    volume = modal.Volume.from_name(launch["volume"], create_if_missing=False)
    inputs = {}

    def fetch(suffix):
        path = f"/{launch['output_prefix']}/{suffix}"
        try:
            data = b"".join(volume.read_file(path))
        except FileNotFoundError:
            return None
        inputs[path] = hashlib.sha256(data).hexdigest()
        wrapped = json.loads(data)
        if identity(wrapped["payload"]) != wrapped["payload_sha256"]:
            raise ValueError(f"corrupt remote payload: {path}")
        return wrapped["payload"]

    cells = ("5ht1b_0", "braf_1", "jak2_1", "parp1_0", "fa7_0")
    arms = ("zero_oracle",) if task["mode"] == "probe" else ("support_control", "objective_search")
    if task["mode"] == "score":
        cells = ("braf_1", "jak2_1", "fa7_0")
    rows = []
    for cell in cells:
        for arm in arms:
            folder = f"{cell}_{arm}"
            result = fetch(f"{folder}/result.json")
            progress = fetch(f"{folder}/progress.json")
            row = {"cell": cell, "arm": arm, "finished": result is not None, "progress": progress}
            if result:
                row["result"] = result
                seal(args.output.parent / "unit_results" / f"{folder}.json", result)
            rows.append(row)
            compact = {**(progress or {}), "cell": cell, "arm": arm, "finished": result is not None}
            if result:
                compact.update(
                    {
                        k: result[k]
                        for k in ("eligible", "attempts", "query_count", "termination")
                        if k in result
                    }
                )
            print(json.dumps(compact, sort_keys=True), flush=True)
    result = {
        "schema_version": "t4_reset_status_v1",
        "observed_at_utc": _stamp(),
        "launch": launch,
        "rows": rows,
        "coordinator": fetch("summary.json"),
        "inputs_sha256": inputs,
        "status_query_new_oracle_calls": 0,
    }
    seal(args.output, result)


if __name__ == "__main__":
    main()
