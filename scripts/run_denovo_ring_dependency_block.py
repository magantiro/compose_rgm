"""Launch the de-novo ring dependency-block acceptance measurement on Modal.

Shards are small and idempotent on purpose.  A preempted container restarts
with the same input, and the lever that makes a fan-out survive preemption is
WORK-UNIT SIZE, not retry count, so each shard is a short stride rather than a
long slice.  Shard results are written as they return, so a fan-out that dies
part way is still scorable instead of discarding every healthy sibling.

Trains nothing, writes no checkpoint, calls no oracle.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from denovo_ring_schedule_acceptance import summarize

    from modal_apps.denovo_ring_dependency_block import (
        ARMS,
        acceptance_shard,
        app,
        select_targets,
    )

    started = time.time()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sidecar = args.output.with_suffix(".rows.jsonl")

    artifact: dict = {
        "measurement": "denovo_ring_dependency_block_acceptance_v1",
        "trains_nothing": True,
        "oracle_calls": 0,
        "arms": list(ARMS),
        "sampling": {
            "rule": "uniform at random without replacement from the train partition",
            "seed": args.seed,
            "sample_size_requested": args.sample_size,
        },
    }

    rows: list[dict] = []
    failures: list[str] = []
    with modal.enable_output(), app.run():
        print(f"[targets] selecting {args.sample_size}", flush=True)
        selection = select_targets.remote(args.sample_size, args.seed)
        targets = selection["targets"]
        indices = selection["indices"]
        artifact["target_selection"] = {
            key: value
            for key, value in selection.items()
            if key not in {"targets", "indices"}
        }

        payloads = []
        for shard in range(args.shards):
            stride = list(range(shard, len(targets), args.shards))
            if not stride:
                continue
            payloads.append(
                {
                    "shard": shard,
                    "indices": [indices[i] for i in stride],
                    "targets": [targets[i] for i in stride],
                }
            )
        print(f"[fanout] {len(payloads)} shards over {len(targets)} molecules",
              flush=True)

        support_evaluations = 0
        container_seconds = 0.0
        versions = None
        catalog = {}
        with sidecar.open("w") as handle:
            # Unordered: results are reduced as a set, never by position, and
            # ordered output blocks every finished shard behind the slowest
            # one -- which on a preemptible fan-out can mean reading nothing
            # at all from a run that did most of its work.
            for result in acceptance_shard.map(
                payloads,
                return_exceptions=True,
                wrap_returned_exceptions=False,
                order_outputs=False,
            ):
                if not isinstance(result, dict):
                    failures.append(repr(result)[:300])
                    print(f"[shard-failed] {repr(result)[:160]}", flush=True)
                    continue
                for row in result["rows"]:
                    rows.append(row)
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                support_evaluations += result["support_evaluations"]
                container_seconds += result["elapsed_seconds"]
                versions = versions or result.get("versions")
                catalog = {
                    "template_count": result["catalog_template_count"],
                    "small_template_count": result["catalog_small_template_count"],
                }
                print(
                    f"[shard {result['shard']}] molecules={len(result['rows'])} "
                    f"total={len(rows)}",
                    flush=True,
                )

    artifact["status"] = "COMPLETE" if not failures else "PARTIAL"
    artifact["shard_failures"] = failures[:5]
    artifact["shards_failed"] = len(failures)
    artifact["molecules_completed"] = len(rows)
    artifact["rows_sidecar"] = sidecar.name
    artifact["versions"] = versions
    artifact["catalog"] = catalog
    artifact["cost_counters"] = {
        "support_evaluations": support_evaluations,
        "container_seconds": container_seconds,
        "molecules": len(rows),
    }
    artifact["wall_seconds"] = time.time() - started
    artifact["rows"] = rows
    artifact["summary"] = summarize(rows)
    args.output.write_text(json.dumps(artifact, indent=1, sort_keys=True) + "\n")
    print(json.dumps(artifact["summary"], indent=1, sort_keys=True), flush=True)
    print(f"[written] {args.output}", flush=True)


if __name__ == "__main__":
    main()
