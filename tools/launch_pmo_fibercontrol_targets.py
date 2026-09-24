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
    parser.add_argument("--transplant-share", type=float, default=0.0,
                        help="fraction of recombination draws using region transplant")
    parser.add_argument("--parent-weighting", choices=("uniform", "measured_score"),
                        default="uniform",
                        help="bootstrap draw over charged initialization molecules; "
                             "'uniform' is byte-identical to every run so far")
    # Two arms of one matched A/B share task, seed and stamp, so without this they would
    # collide on the same volume namespace and the launcher would refuse the second.
    parser.add_argument("--label-suffix", default="",
                        help="arm tag appended to the label and namespace")
    parser.add_argument("--transplant-lane", action="store_true",
                        help="dedicated deduplicated transplant proposal lane")
    parser.add_argument("--donor-reservoir", action="store_true",
                        help="draw transplant donors from the scaffold-deduplicated bank")
    parser.add_argument("--graft-init", action="store_true",
                        help="initialize from the target-blind ring-graft entry bank")
    parser.add_argument("--prescreen", action="store_true",
                        help="initialize from the per-task ZINC250k prescreen bank")
    parser.add_argument("--receipt", default="diagnostics/pmo_fibercontrol_targets_v1")
    arguments = parser.parse_args()

    import modal

    from modal_apps.pmo_fibercontrol_targets_app import RUN_APP, derived_seed

    # Same override the app honours, so launcher and deployment cannot address
    # different apps.
    function = modal.Function.from_name(RUN_APP, "run_target")
    commit = _commit()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    submitted, labels = [], set()
    for task in arguments.targets:
        seed = derived_seed(task, arguments.base_seed, arguments.replicate)
        suffix = f"_{arguments.label_suffix}" if arguments.label_suffix else ""
        label = f"{arguments.stage}_{task}_seed{seed}{suffix}_{stamp}"
        volume = f"pmo-fiber-{task.replace('_', '-')}-seed{seed}{suffix.replace('_', '-')}"
        if label in labels or volume in {s["volume"] for s in submitted}:
            raise SystemExit(f"two targets would share a namespace: {label} / {volume}")
        labels.add(label)
        spec = {
            "task": task,
            "prescreen": bool(arguments.prescreen),
            "transplant_share": float(arguments.transplant_share),
            "parent_weighting": arguments.parent_weighting,
            "graft_init": bool(arguments.graft_init),
            "transplant_lane": bool(arguments.transplant_lane),
            "donor_reservoir": bool(arguments.donor_reservoir),
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

    # The stamp is second-resolution, so two arms of one A/B launched back to back land
    # on the SAME path and the second silently overwrites the first's call ids. The arm
    # tag is part of the receipt identity for exactly that reason.
    tag = f"_{arguments.label_suffix}" if arguments.label_suffix else ""
    receipt = pathlib.Path(arguments.receipt) / f"launch_{arguments.stage}{tag}_{stamp}.json"
    if receipt.exists():
        raise SystemExit(f"refusing to overwrite an existing launch receipt: {receipt}")
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
