"""Read durable reset progress/results without launching, retrying or scoring."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.experiments.t4_matched_pilot import _stamp, seal, unseal


def historical_at(cell, calls, units, observations):
    """Exact matched-call comparisons only, never carry final scores backward."""
    result = {}
    for arm in ("v0", "v1", "v21", "full146"):
        matches = [
            u for u in units if u["cell"] == cell and u["arm"] == arm and u["replicate"] == 0
        ]
        if not matches:
            result[arm] = None
            continue
        unit = matches[0]
        if (
            not calls
            or unit["query_count"] < calls
            or not unit["query_mapping_verified"]
            or unit["rounds"] != unit["rounds_recovered"]
        ):
            result[arm] = None
            continue
        values = [
            r["score"]
            for r in observations
            if r["cell"] == cell
            and r["arm"] == arm
            and r["replicate"] == 0
            and r["query_verified"]
            and r["query"] is not None
            and r["query"] <= calls
        ]
        result[arm] = min(values) if values else None
    return result


def main():
    import modal

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("launch_file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-probe-rounds", action="store_true")
    args = parser.parse_args()
    launch = unseal(args.launch_file)
    task = launch["task"]
    volume = modal.Volume.from_name(launch["volume"], create_if_missing=False)
    inputs = {}
    historical, observations = None, []
    if task["mode"] == "score":
        evidence = Path(__file__).resolve().parents[1] / "diagnostics/t4_strategy_reset/20260916"
        audit_path, rows_path = evidence / "audit.json", evidence / "scored_rows.jsonl.gz"
        historical = json.loads(audit_path.read_text())
        with gzip.open(rows_path, "rt") as handle:
            observations = [json.loads(line) for line in handle]
        for path in (audit_path, rows_path):
            inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

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
                if args.include_probe_rounds and task["mode"] == "probe":
                    for stage in result["rounds"]:
                        name = f"round_{stage['round']:04d}.json"
                        payload = fetch(f"{folder}/rounds/{name}")
                        if payload is None:
                            raise FileNotFoundError(f"completed probe missing {folder}/{name}")
                        seal(args.output.parent / "unit_rounds" / folder / name, payload)
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
            if historical:
                calls = result["query_count"] if result else (progress or {}).get("calls", 0)
                comparison = historical_at(cell, calls, historical["units"], observations)
                row["historical_at_same_charged_calls"] = comparison
                compact["historical_at_same_charged_calls"] = comparison
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
