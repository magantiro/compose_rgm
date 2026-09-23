"""Submit one Modal container per PMO target, all at once.

Parallelism is ACROSS targets. Each campaign is stateful and runs its rounds serially inside
its own container, so the launcher's only job is to submit every target without waiting and
to hand back a call id per target so each can be watched independently.

It refuses two things by construction rather than by care: two targets sharing a volume
namespace (concurrent writers on one mutable checkpoint) and a seed derived from anything
process-dependent.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from datetime import UTC, datetime

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

SIMILARITY_PANEL = (
    "celecoxib_rediscovery",
    "albuterol_similarity",
    "mestranol_similarity",
    "thiothixene_rediscovery",
    "troglitazone_rediscovery",
)


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", default=list(SIMILARITY_PANEL))
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=700)
    parser.add_argument("--queries", type=int, default=16)
    parser.add_argument("--base-seed", type=int, default=20260923)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--stage", choices=("smoke", "scored"), required=True)
    parser.add_argument("--receipt", default="diagnostics/pmo_fibercontrol_targets_v1")
    arguments = parser.parse_args()

    import modal

    from modal_apps.pmo_fibercontrol_targets_app import derived_seed

    function = modal.Function.from_name(
        "compose-pmo-fibercontrol-targets", "run_target"
    )
    commit = _commit()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    submitted, labels = [], set()
    for task in arguments.targets:
        seed = derived_seed(task, arguments.base_seed, arguments.replicate)
        label = f"{arguments.stage}_{task}_seed{seed}_{stamp}"
        volume = f"pmo-fiber-{task.replace('_', '-')}-seed{seed}"
        if label in labels or volume in {s["volume"] for s in submitted}:
            raise SystemExit(f"two targets would share a namespace: {label} / {volume}")
        labels.add(label)
        spec = {
            "task": task,
            "budget": arguments.budget,
            "rounds": arguments.rounds,
            "queries": arguments.queries,
            "seed": seed,
            "base_seed": arguments.base_seed,
            "replicate": arguments.replicate,
            "label": label,
            "volume": volume,
            "git_commit": commit,
        }
        # SPAWN, never call: target B must be submitted without waiting for target A.
        handle = function.spawn(spec)
        spec["modal_call_id"] = handle.object_id
        submitted.append(spec)
        print(f"  {task:28s} seed={seed}  call={handle.object_id}", flush=True)

    receipt = pathlib.Path(arguments.receipt) / f"launch_{arguments.stage}_{stamp}.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with open(receipt, "w") as handle:
        json.dump(
            {
                "schema_version": "pmo_fibercontrol_launch_v1",
                "stage": arguments.stage,
                "git_commit": commit,
                "base_seed": arguments.base_seed,
                "budget": arguments.budget,
                "submitted": submitted,
            },
            handle,
            indent=1,
        )
    print(f"\n{len(submitted)} targets submitted; receipt {receipt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
