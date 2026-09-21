"""Drive the zero-training de-novo ring-schedule probe and reduce its artifact.

Order is deliberate and is not a convenience: the instrument check runs FIRST and
its verdict is recorded before any arm number is computed, so an enumeration that
disagrees with the committed reference audit cannot be discovered after the fact
and rationalised against the arm results it was supposed to validate.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_AUDIT = ROOT / "diagnostics" / "ring_calibration" / "step2500_exact_support_audit_12.json"
OUTPUT_DIR = ROOT / "diagnostics" / "denovo_schedule_probe_v1"


def _reference_rows() -> list[dict]:
    payload = json.loads(REFERENCE_AUDIT.read_text())
    return [
        {
            "row": row["row"],
            "before": row["before"],
            "legal_template_count": row["legal_template_count"],
            "legal_small_template_count": row["legal_small_template_count"],
            "small_mass_uniform_support": row["small_mass_uniform_support"],
        }
        for row in payload["rows"]
    ]


def _reference_summary() -> dict:
    payload = json.loads(REFERENCE_AUDIT.read_text())
    return payload["summary"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--model-label", type=str, default="step1000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-instrument", action="store_true",
                        help="only for a cost smoke; never for a reported run")
    args = parser.parse_args()

    from modal_apps.denovo_schedule_probe import (
        app,
        catalog_identity,
        instrument_check,
        probe_shard,
        select_targets,
    )

    started = time.time()
    artifact: dict = {
        "probe": "denovo_ring_schedule_probe_v1",
        "predeclaration": "diagnostics/denovo_schedule_probe_v1/PREDECLARATION.json",
        "trains_nothing": True,
        "oracle_calls": 0,
        "model_label": args.model_label,
        "sample_size_requested": args.sample_size,
        "shards": args.shards,
        "seed": args.seed,
        "reference_summary": _reference_summary(),
    }

    with modal.enable_output(), app.run():
        if not args.skip_instrument:
            rows = _reference_rows()
            print(f"[instrument] replicating {len(rows)} reference rows", flush=True)
            artifact["instrument_check"] = instrument_check.remote(rows, "step2500")
            print(
                "[instrument] agreeing "
                f"{artifact['instrument_check']['rows_agreeing']}"
                f"/{artifact['instrument_check']['rows_total']}",
                flush=True,
            )
            artifact["catalog_identity"] = list(
                catalog_identity.map(["step1000", "step2500"])
            )

        print(f"[targets] selecting {args.sample_size}", flush=True)
        selection = select_targets.remote(args.sample_size, args.seed)
        targets = selection["targets"]
        artifact["target_selection"] = {
            key: value for key, value in selection.items() if key != "targets"
        }

        payloads = []
        for shard in range(args.shards):
            indices = list(range(shard, len(targets), args.shards))
            payloads.append(
                {
                    "shard": shard,
                    "offsets": indices,
                    "targets": [targets[index] for index in indices],
                    "seed": args.seed,
                    "model_label": args.model_label,
                }
            )
        print(f"[arms] {args.shards} shards, {len(targets)} targets", flush=True)
        results = list(probe_shard.map(payloads, return_exceptions=True))

    ok = [r for r in results if isinstance(r, dict)]
    failed = [repr(r)[:300] for r in results if not isinstance(r, dict)]
    artifact["shards_ok"] = len(ok)
    artifact["shards_failed"] = len(failed)
    artifact["shard_failures"] = failed[:5]
    artifact["shard_results"] = [
        {key: value for key, value in shard.items() if key != "events"} for shard in ok
    ]
    artifact["events"] = [event for shard in ok for event in shard["events"]]
    artifact["wall_seconds"] = time.time() - started
    artifact["cost_counters"] = {
        "traces_compiled": sum(shard["compiled"] for shard in ok),
        "compile_failures": sum(shard["compile_failures"] for shard in ok),
        "support_evaluations": sum(shard["support_evaluations"] for shard in ok),
        "container_seconds": sum(shard["elapsed_seconds"] for shard in ok),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=1, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "phase": "written",
                "output": str(args.output),
                "events": len(artifact["events"]),
                **artifact["cost_counters"],
                "wall_seconds": round(artifact["wall_seconds"], 1),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
