"""Build a payload-bound 250-call held-target contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.experiments.t4_matched_pilot import unseal

ROOT = Path(__file__).resolve().parents[1]
APP = "modal_apps/t4_integrated_route_fiber_parp1_app.py"
CHECKPOINT_ROOT = "diagnostics/t4_held_target_distillation_quality_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--support", required=True)
    parser.add_argument("--app-wrapper", required=True)
    parser.add_argument("--volume", required=True)
    parser.add_argument("--receptor", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    base_path = ROOT / args.base
    payload = dict(unseal(base_path))
    target = args.target
    payload.update(
        {
            "schema_version": f"t4_held_target_distilled_{target}_d06_250_contract_v1",
            "status": "SEALED_PENDING_EXACT_USER_AUTHORIZATION",
            "scientific_problem": f"test transfer of a route-trained structural-action prior to {target} with every {target} route excluded from fitting",
            "primary_model_output": "FiberControl selection over complete shallow, anchored, and leave-target-out route programs",
            "claim_boundary": f"held-target {target} panel at delta 0.6 with 250 charged calls per cell; no {target} route enters the prior",
            "frozen_from": {
                **payload.get("frozen_from", {}),
                "controller_contract": args.base,
                "allowed_adapter_changes": [
                    f"leave-{target}-out route checkpoint",
                    f"{target} cells and docking evaluator",
                    "held-target output namespace",
                    "250-call payload-bound contract identity",
                ],
            },
            "zero_oracle_route_support": {
                "artifact": args.support,
                "artifact_sha256": sha256(ROOT / args.support),
                "result": f"leave-{target}-out route support artifact is frozen for the scored panel",
                "interpretation": "support outcome is preserved and is not treated as oracle evidence",
            },
            "charged_calls_per_cell": 250,
            "total_charged_call_ceiling": 250 * len(payload["cells"]),
            "runtime_inputs_sha256": dict(payload.get("runtime_inputs_sha256", {})),
        }
    )

    runtime = payload["runtime_inputs_sha256"]
    for key in list(runtime):
        if key.endswith("route_expert_checkpoint.json") or key.endswith("checkpoint.json"):
            runtime.pop(key, None)
        if key.startswith("modal_apps/"):
            runtime.pop(key, None)
    runtime[args.checkpoint] = sha256(ROOT / args.checkpoint)
    runtime[APP] = sha256(ROOT / APP)
    for relative in tuple(runtime):
        path = ROOT / relative
        if not path.exists():
            raise FileNotFoundError(path)
        runtime[relative] = sha256(path)
    payload["proposal"]["route_complete_region"]["training_split"] = f"leave-{target}-out"
    payload["proposal"]["route_complete_region"]["pool_size"] = 64
    payload["proposal"]["route_complete_region"]["realization_limit"] = min(
        64, int(payload["proposal"]["route_complete_region"].get("realization_limit", 64))
    )
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    output = (ROOT / args.output).resolve()
    if output.exists() and not args.force:
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(envelope, sort_keys=True, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps({"path": str(output), "payload_sha256": envelope["payload_sha256"], "volume": args.volume}, indent=2))


if __name__ == "__main__":
    main()
