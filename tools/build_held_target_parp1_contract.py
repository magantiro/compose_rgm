"""Build the payload-bound held-target PARP1 scored contract from the sealed prior."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.experiments.t4_matched_pilot import unseal

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/t4_integrated_route_fiber_parp1_v1.json"
OUTPUT = ROOT / "configs/t4_held_target_distilled_parp1_d06_v1.json"
CHECKPOINT = "diagnostics/t4_held_target_distillation_quality_v1/parp1_checkpoint.json"
SUPPORT = "diagnostics/t4_held_target_distillation_quality_v1/parp1_smoke.json"
APP = "modal_apps/t4_integrated_route_fiber_parp1_app.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    payload = dict(unseal(BASE))
    payload.update(
        {
            "schema_version": "t4_held_target_distilled_parp1_d06_contract_v1",
            "status": "SEALED_PENDING_EXACT_USER_AUTHORIZATION",
            "scientific_problem": "test transfer of a route-trained structural-action prior to a target with every PARP1 route excluded from fitting",
            "primary_model_output": "FiberControl selection over complete shallow, anchored, and leave-PARP1-out route programs",
            "claim_boundary": "held-target PARP1 development panel under one docking seed; no PARP1 route enters the prior, and this is not benchmark-wide held-target evidence",
            "frozen_from": {
                **payload["frozen_from"],
                "controller_contract": "configs/t4_integrated_route_fiber_parp1_v1.json",
                "allowed_adapter_changes": [
                    "leave-PARP1-out route checkpoint",
                    "held-target output namespace",
                    "payload-bound contract identity",
                ],
            },
            "zero_oracle_route_support": {
                "artifact": SUPPORT,
                "artifact_sha256": sha256(ROOT / SUPPORT),
                "result": "leave-PARP1-out route expert produced exact programs and eligible endpoints on PARP1-0 and PARP1-1; PARP1-2 abstention is preserved",
                "interpretation": "support gate is partial and the scored run must preserve exhaustion as a valid outcome",
            },
            "runtime_inputs_sha256": {
                **payload["runtime_inputs_sha256"],
                CHECKPOINT: sha256(ROOT / CHECKPOINT),
                APP: sha256(ROOT / APP),
            },
        }
    )
    payload["runtime_inputs_sha256"].pop(
        "diagnostics/t4_integrated_route_fiber_parp1_v1/route_expert_checkpoint.json",
        None,
    )
    payload["runtime_inputs_sha256"].pop(
        "modal_apps/t4_integrated_route_fiber_parp1_app.py", None
    )
    payload["runtime_inputs_sha256"][CHECKPOINT] = sha256(ROOT / CHECKPOINT)
    payload["runtime_inputs_sha256"][APP] = sha256(ROOT / APP)
    payload["proposal"]["route_complete_region"]["training_split"] = "leave-PARP1-out"
    payload["proposal"]["route_complete_region"]["pool_size"] = 64
    payload["total_charged_call_ceiling"] = 147
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    temporary.write_text(json.dumps(envelope, sort_keys=True, indent=2) + "\n")
    temporary.replace(OUTPUT)
    print(json.dumps({"path": str(OUTPUT), "payload_sha256": envelope["payload_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
