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
    # ARM SELECTOR for the proposal ablation.  Absent is arm A and leaves the worker
    # environment untouched, which is byte-identical to the unpatched tree (verified by
    # execution).  The label carries the arm so two arms can never share a namespace.
    parser.add_argument("--uniform-chain-arm", action="store_true")
    # ARM C: the structured recipe is kept and only its created-atom operands are
    # resampled.  Mutually exclusive with arm B, which discards the recipe entirely;
    # rebinding a uniform chain would measure neither arm.
    parser.add_argument("--binding-rebind-arm", action="store_true")
    parser.add_argument("--receipt", default="diagnostics/pmo_fibercontrol_targets_v1")
    arguments = parser.parse_args()

    import modal

    from modal_apps.pmo_fibercontrol_targets_app import RUN_APP, derived_seed

    # Same override the app honours, so launcher and deployment cannot address
    # different apps.
    function = modal.Function.from_name(RUN_APP, "run_target")
    commit = _commit()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    if arguments.uniform_chain_arm and arguments.binding_rebind_arm:
        raise SystemExit("--uniform-chain-arm and --binding-rebind-arm are exclusive arms")

    submitted, labels = [], set()
    for task in arguments.targets:
        seed = derived_seed(task, arguments.base_seed, arguments.replicate)
        arm = (
            "chain"
            if arguments.uniform_chain_arm
            else "rebind" if arguments.binding_rebind_arm else "struct"
        )
        label = f"{arguments.stage}_{arm}_{task}_seed{seed}_{stamp}"
        volume = f"pmo-fiber-{arm}-{task.replace('_', '-')}-seed{seed}"
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
            "uniform_chain_arm": bool(arguments.uniform_chain_arm),
            "binding_rebind_arm": bool(arguments.binding_rebind_arm),
            "arm": arm,
        }
        # SPAWN, never call: target B must be submitted without waiting for target A.
        handle = function.spawn(spec)
        spec["modal_call_id"] = handle.object_id
        submitted.append(spec)
        print(f"  {task:28s} seed={seed}  call={handle.object_id}", flush=True)

    # The stamp is SECOND-resolution, so two launches of the same stage within one second
    # wrote the same path and the second silently destroyed the first's call ids -- which
    # happened on the arm-C launch and cost a receipt that had to be reconstructed from
    # stdout. The name now carries the ARM and the REPLICATE, and an existing path is a
    # hard error rather than an overwrite: a launch receipt is the only durable record of
    # which call ids belong to which campaigns.
    receipt = pathlib.Path(arguments.receipt) / (
        f"launch_{arguments.stage}_{arm}_r{arguments.replicate}_{stamp}.json"
    )
    receipt.parent.mkdir(parents=True, exist_ok=True)
    if receipt.exists():
        raise SystemExit(
            f"refusing to overwrite an existing launch receipt: {receipt}\n"
            f"the campaigns WERE submitted; their call ids are above. Record them by hand."
        )
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
