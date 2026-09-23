"""Build the payload-bound held-target JAK2 scored contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.experiments.t4_matched_pilot import unseal

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/t4_integrated_route_fiber_parp1_v1.json"
JAK2_BASE = ROOT / "configs/t4_shared_retained_fiber_jak2_v2.json"
CHECKPOINT = "diagnostics/t4_held_target_distillation_quality_v1/jak2_checkpoint.json"
SUPPORT = "diagnostics/t4_held_target_distillation_quality_v1/jak2_smoke.json"
APP = "modal_apps/t4_integrated_route_fiber_parp1_app.py"
DEFAULT_OUTPUT = ROOT / "configs/t4_held_target_distilled_jak2_d06_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = dict(unseal(BASE))
    jak2 = unseal(JAK2_BASE)
    payload.update(
        {
            "schema_version": "t4_held_target_distilled_jak2_d06_contract_v1",
            "status": "SEALED_PENDING_EXACT_USER_AUTHORIZATION",
            "scientific_problem": "test transfer of a route-trained structural-action prior to a target with every JAK2 route excluded from fitting",
            "primary_model_output": "FiberControl selection over complete shallow, anchored, and leave-JAK2-out route programs",
            "claim_boundary": "held-target JAK2 development panel under one docking seed; no JAK2 route enters the prior, and this is not benchmark-wide held-target evidence",
            "cells": jak2["cells"],
            "delta": 0.6,
            "docking_box": jak2["docking_box"],
            "evaluator_sha256": jak2["evaluator_sha256"],
            "frozen_from": {
                **payload["frozen_from"],
                "controller_contract": "configs/t4_integrated_route_fiber_parp1_v1.json",
                "allowed_adapter_changes": [
                    "leave-JAK2-out route checkpoint",
                    "JAK2 cells and docking evaluator",
                    "held-target output namespace",
                    "payload-bound contract identity",
                ],
            },
            "zero_oracle_route_support": {
                "artifact": SUPPORT,
                "artifact_sha256": sha256(ROOT / SUPPORT),
                "result": "leave-JAK2-out route expert produced exact programs and eligible endpoints on all three JAK2 seeds",
                "interpretation": "support gate is sufficient for a scored held-target panel",
            },
            "runtime_inputs_sha256": dict(payload["runtime_inputs_sha256"]),
        }
    )
    payload["runtime_inputs_sha256"].pop(
        "diagnostics/t4_integrated_route_fiber_parp1_v1/route_expert_checkpoint.json", None
    )
    payload["runtime_inputs_sha256"].pop("modal_apps/t4_integrated_route_fiber_parp1_app.py", None)
    payload["runtime_inputs_sha256"][CHECKPOINT] = sha256(ROOT / CHECKPOINT)
    payload["runtime_inputs_sha256"][APP] = sha256(ROOT / APP)
    for relative in tuple(payload["runtime_inputs_sha256"]):
        path = ROOT / relative
        if not path.exists():
            raise FileNotFoundError(path)
        payload["runtime_inputs_sha256"][relative] = sha256(path)
    payload["proposal"]["route_complete_region"]["training_split"] = "leave-JAK2-out"
    payload["proposal"]["route_complete_region"]["pool_size"] = 64
    payload["total_charged_call_ceiling"] = 147
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(envelope, sort_keys=True, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps({"path": str(output), "payload_sha256": envelope["payload_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
