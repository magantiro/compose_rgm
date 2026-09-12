"""Prepare or durably launch the small winner-informed multi-site docking pool."""

import argparse
import subprocess
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from compose_v4.experiments.t4_matched_pilot import seal, unseal
from compose_v4.experiments.t4_program_pool import APP, LOCK, prepare

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "launch"))
    args = parser.parse_args()
    if args.mode == "prepare":
        if (ROOT / LOCK).exists():
            raise ValueError("candidate lock already exists; inspect it instead of regenerating")
        lock = prepare(ROOT)
        seal(ROOT / LOCK, lock)
        print(
            f"Locked {len(lock['take'])} candidates; at most {lock['compute']['new_oracle_call_limit']} new calls"
        )
        return
    import modal

    from modal_apps.run_process_v2_p50_app import local_image_revision
    from tools.preflight import assert_synced

    preflight = assert_synced(strict=True)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("scientific launch requires a fully clean committed source tree")
    lock = unseal(ROOT / LOCK)
    task = {
        "lock_sha256": sha256_file(ROOT / LOCK),
        "program_app_sha256": sha256_file(ROOT / APP),
        "app_sha256": sha256_file(ROOT / "modal_apps/genmol_t4_opt_app.py"),
        "image_revision": local_image_revision(expected_commit=preflight["commit"]),
    }
    task["run_id"] = identity(task)
    call = modal.Function.from_name("compose-t4-program-pool", "run_pool").spawn(task)
    receipt = {
        "task": task,
        "call_id": call.object_id,
        "new_oracle_call_limit": lock["compute"]["new_oracle_call_limit"],
        "volume": "compose-v4-artifacts",
        "volume_path": f"t4_program_pool/{task['run_id']}",
    }
    publish_json(ROOT / "diagnostics/t4_program_pool_spawn.json", receipt)
    print(f"Spawned {call.object_id}; at most {receipt['new_oracle_call_limit']} new calls")
    print(receipt["volume_path"])


if __name__ == "__main__":
    main()
