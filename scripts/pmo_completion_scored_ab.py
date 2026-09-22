#!/usr/bin/env python
"""Matched scored A/B for the PMO completion repair, run locally on the pinned kernel.

    arm A   deployed B                       (online memory ON, completion law ABSENT)
    arm B   deployed B + the frozen law      (online memory ON, completion law ON)

Everything else is identical by construction: both arms take the same contract,
the same initialization lock, the same joint checkpoint, the same controller
seed, the same budget, the same oracle object and the same code.  The ONLY
difference is one entry in ``optimizer_kwargs``, which ``run_program_campaign``
folds into the run identity so the arms cannot share one.

Before the first charged call this runs a POSITIVE CONTROL that CALLS the
oracle against pinned reference values including a nontrivial intermediate.
Construction success is not evidence that an oracle scores.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "configs/pmo_completion_ab_local_contract_v1.json"
OUT = ROOT / "diagnostics/pmo_completion_repair_v1/scored_ab"

#: Pinned reference values, measured in this environment before any arm ran.
#: A constant oracle passes any nonzero check and fails these.
REFERENCE_PANEL = {
    "celecoxib_rediscovery": (
        ("CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F", 1.0),
        ("c1ccccc1", 0.07228915662650602),
        ("CC(=O)Oc1ccccc1C(=O)O", 0.11578947368421053),
    ),
}


def _rdkit_six_shim() -> None:
    import types

    import rdkit

    shim = types.ModuleType("rdkit.six")
    shim.iteritems = lambda value, **kwargs: iter(value.items())
    shim.itervalues = lambda value, **kwargs: iter(value.values())
    shim.iterkeys = lambda value, **kwargs: iter(value.keys())
    shim.string_types = (str,)
    sys.modules["rdkit.six"] = shim
    rdkit.six = shim


def load_local_contract() -> dict:
    """Same verification as the production loader, with the authorization inverted.

    ``pmo_population_v1.load_contract`` REFUSES a contract whose
    ``scored_launch_authorized`` is anything but ``False`` -- it is the preflight
    loader.  This is the scored local loader, so the flag must be True, and every
    other check is kept: envelope hash, initialization count and lock, joint
    checkpoint payload identity, and every pinned implementation file.
    """

    from compose_v4.control.docking_value import identity
    from compose_v4.experiments.continuation_profile import verify_file
    from compose_v4.experiments.pmo_population_v1 import (
        INIT_COUNT,
        _load_checkpoint,
        _load_initialization,
    )

    envelope = json.loads((ROOT / CONTRACT).read_text())
    contract = envelope["payload"]
    if envelope.get("payload_sha256") != identity(contract):
        raise ValueError("local A/B contract envelope hash changed")
    if contract.get("scored_launch_authorized") is not True:
        raise ValueError("local A/B contract does not authorize scoring")
    initialization = _load_initialization(ROOT, {"initialization": contract["initialization"]})
    if initialization.get("count") != INIT_COUNT:
        raise ValueError("initialization count changed")
    _load_checkpoint(ROOT, contract)
    for path, digest in contract["implementation_sha256"].items():
        verify_file(ROOT / path, digest)
    return contract


def build_oracle(task_name: str):
    _rdkit_six_shim()
    from tdc import Oracle

    oracle = Oracle(name=task_name)
    panel = REFERENCE_PANEL.get(task_name)
    control = []
    if panel is None:
        raise ValueError(
            f"no pinned reference panel for {task_name}; a scored run may not start "
            "without a positive control that CALLS the oracle"
        )
    for smiles, expected in panel:
        observed = float(oracle(smiles))
        control.append(
            {"smiles": smiles, "expected": expected, "observed": observed,
             "abs_delta": abs(observed - expected)}
        )
        if abs(observed - expected) > 1e-9:
            raise ValueError(
                f"oracle positive control FAILED for {task_name}: {smiles} "
                f"expected {expected}, observed {observed}"
            )
    if len({row["observed"] for row in control}) < len(control):
        raise ValueError("reference panel does not discriminate a constant oracle")
    return oracle, control


def run_arm(contract: dict, arm: str, task_name: str, *, resume: bool) -> dict:
    from compose_v4.experiments.pmo_population_v1 import execute_task

    spec = contract["arms"][arm]
    folder = OUT / task_name / arm
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / "result.json"
    if result_path.exists():
        if not resume:
            raise RuntimeError(
                f"{result_path} exists; refusing to re-run a scored arm and "
                "re-charge its budget"
            )
        return json.loads(result_path.read_text())

    oracle, control = build_oracle(task_name)
    (folder / "oracle_positive_control.json").write_text(
        json.dumps({"task": task_name, "panel": control, "oracle_called": True},
                   sort_keys=True, indent=1)
    )

    calls = {"n": 0}

    def evaluate(smiles: str) -> float:
        calls["n"] += 1
        return float(oracle(smiles))

    began = time.time()
    rounds = []

    def progress(row):
        if row.get("phase") != "scored":
            return
        keep = {
            k: row.get(k)
            for k in (
                "round",
                "oracle_calls_including_initialization",
                "top_k_utility",
                "best_new_utility",
                "proposal_attempts",
                "pool_size",
            )
        }
        rounds.append(keep)
        print(
            f"    round {keep['round']:>3} "
            f"charged={keep['oracle_calls_including_initialization']:>4} "
            f"top10={keep['top_k_utility']} best={keep['best_new_utility']}",
            flush=True,
        )

    result = execute_task(
        contract,
        ROOT,
        folder,
        task_name,
        evaluate=evaluate,
        charged_calls_per_task=int(contract["charged_calls_per_task"]),
        progress=progress,
        enable_online_memory=bool(spec["enable_online_memory"]),
        completion_law=spec["completion_law"],
    )
    result["arm_name"] = arm
    result["arm_spec"] = spec
    result["wall_seconds"] = round(time.time() - began, 1)
    result["evaluate_invocations"] = calls["n"]
    result["round_progress"] = rounds
    result["oracle_positive_control"] = control
    result["contract_payload_sha256"] = json.loads(
        (ROOT / CONTRACT).read_text()
    )["payload_sha256"]
    result_path.write_text(json.dumps(result, sort_keys=True, indent=1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="celecoxib_rediscovery")
    parser.add_argument("--arm", required=True, choices=["A_deployed_b", "B_completion"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    contract = load_local_contract()
    if args.task not in contract["tasks"]:
        raise SystemExit(f"{args.task} is outside this contract's task lock")
    result = run_arm(contract, args.arm, args.task, resume=args.resume)
    print(
        f"{args.task}/{args.arm}: charged={result['charged_oracle_calls']} "
        f"best={result['best_score']} auc={result['auc_top10_at_budget']:.4f} "
        f"({result['wall_seconds']}s)"
    )


if __name__ == "__main__":
    main()
