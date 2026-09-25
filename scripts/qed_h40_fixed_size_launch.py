"""Build exact H40 QED tasks; launch only with explicit --execute on Nitya.

The app must first be deployed from the same clean committed worktree. The
default invocation only validates inputs and prints the immutable task census.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modal_apps.hphi_h40head_ab_app import official_tasks

CONTRACT = ROOT / "configs/qed_h40_fixed_size_ablation_v1.json"
APP = "hphi-h40head-ab"
OUTPUTS = {
    "parity": ("hphi_official800_k8_nitya_parity_v1", (10,)),
    "recover": ("hphi_official800_k8_full_recovery_v1", (135, 408)),
    "fixed": ("hphi_official800_k8_fixed_size_v1", tuple(range(800))),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()


def load_contract() -> dict[str, Any]:
    contract = json.loads(CONTRACT.read_text())
    expected = contract["contract_sha256"]
    actual = _canonical_sha256({
        key: value for key, value in contract.items() if key != "contract_sha256"
    })
    if expected != actual:
        raise ValueError(f"QED contract self-hash mismatch: {actual} != {expected}")
    files = {
        contract["source_panel"]["path"]: contract["source_panel"]["sha256"],
        contract["sampler"]["runner"].split("::")[0]: contract["sampler"]["runner_sha256"],
        "src/compose_v4/experiments/hphi_lazy_sampler.py": contract["sampler"]["lazy_sampler_sha256"],
        "src/compose_v4/experiments/hphi_smc.py": contract["sampler"]["smc_sha256"],
        "src/compose_v4/experiments/hphi_lazy_helpers.py": contract["sampler"]["helpers_sha256"],
    }
    for relative, digest in files.items():
        if _sha256(ROOT / relative) != digest:
            raise ValueError(f"QED contract input hash mismatch: {relative}")
    return contract


def build_tasks(mode: str, contract: dict[str, Any]) -> list[dict[str, Any]]:
    if mode not in OUTPUTS:
        raise ValueError(f"unknown QED launch mode {mode}")
    out_dir, indices = OUTPUTS[mode]
    tasks: list[dict[str, Any]] = []
    for index in indices:
        task = official_tasks(ROOT, out_dir, 0, 8, index, index + 1)[0]
        if mode == "fixed":
            task["size_fixed"] = True
        tasks.append(task)
    if mode == "fixed" and len(tasks) != contract["source_panel"]["distinct_sources"]:
        raise ValueError("fixed-size task count differs from declared source panel")
    if mode == "recover" and [t["index"] for t in tasks] != contract["baseline"]["missing_source_indices"]:
        raise ValueError("baseline recovery indices differ from contract")
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=OUTPUTS)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    contract = load_contract()
    tasks = build_tasks(args.mode, contract)
    print(json.dumps({
        "mode": args.mode,
        "tasks": len(tasks),
        "indices": [task["index"] for task in tasks] if len(tasks) < 10 else [0, 799],
        "out_dir": tasks[0]["out_dir"],
        "contract_sha256": contract["contract_sha256"],
        "tasks_sha256": _canonical_sha256(tasks),
        "execute": args.execute,
    }, sort_keys=True), flush=True)
    if not args.execute:
        return
    if os.environ.get("MODAL_PROFILE") != contract["inputs"]["modal_profile"]:
        raise ValueError("MODAL_PROFILE must be nitya for this QED contract")
    import modal

    call = modal.Function.from_name(APP, "drive").spawn(tasks)
    print(json.dumps({"spawned_call_id": call.object_id, "mode": args.mode}), flush=True)


if __name__ == "__main__":
    main()
