"""Re-pin the held-target 250-call contracts to the resume-capable app.

The contracts pin the controller app inside `runtime_inputs_sha256`, so repairing
`run_cell` to resume a preempted campaign necessarily invalidates them. Re-sealing
would normally orphan the existing checkpoints, which record the predecessor payload
hash. This tool re-seals only after proving that the app pin is the single field that
moved, and records the predecessor hash so a resumed cell can accept exactly the one
declared ancestor and nothing else.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.experiments.t4_matched_pilot import seal, unseal

ROOT = Path(__file__).resolve().parents[1]
APP = "modal_apps/t4_integrated_route_fiber_parp1_app.py"
TARGETS = ("parp1", "jak2", "braf", "5ht1b", "fa7")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reseal(target: str, *, apply: bool) -> dict:
    path = ROOT / f"configs/t4_held_target_distilled_{target}_d06_250.json"
    payload = dict(unseal(path))
    previous = identity(payload)
    actual = sha256(ROOT / APP)

    updated = dict(payload)
    inputs = dict(updated["runtime_inputs_sha256"])
    pinned = inputs.get(APP)
    if pinned is None:
        raise ValueError(f"{target}: contract does not pin {APP}")

    # The ancestor must stay the contract the existing checkpoints were written under,
    # so re-sealing twice never chains predecessors and orphans them.
    existing = payload.get("resume_predecessor") or {}
    ancestor = existing.get("contract_payload_sha256", previous)
    origin_app = existing.get("previous_app_sha256", pinned)

    inputs[APP] = actual
    updated["runtime_inputs_sha256"] = inputs
    updated["resume_predecessor"] = {
        "contract_payload_sha256": ancestor,
        "reason": "run_cell gained the resume path; no scientific field changed",
        "changed_runtime_input": APP,
        "previous_app_sha256": origin_app,
        "current_app_sha256": actual,
    }

    # Prove the app pin is the ONLY difference from the ancestor: strip the provenance
    # block, restore the ancestor's app hash, and require the payload to hash to it.
    probe = dict(updated)
    probe.pop("resume_predecessor")
    probe["runtime_inputs_sha256"] = {**inputs, APP: origin_app}
    if identity(probe) != ancestor:
        raise ValueError(f"{target}: re-seal would change a field other than the app pin")

    for key in ("delta", "charged_calls_per_cell", "cells", "proposal", "docking_seed",
                "evaluator_sha256", "support", "value_penalty", "exploration"):
        if key in payload and payload[key] != updated.get(key):
            raise ValueError(f"{target}: {key} moved during re-seal")

    record = {
        "target": target,
        "previous_contract_payload_sha256": previous,
        "new_contract_payload_sha256": identity(updated),
        "ancestor_contract_payload_sha256": ancestor,
        "previous_app_sha256": origin_app,
        "current_app_sha256": actual,
        "already_current": pinned == actual,
    }
    if apply and pinned != actual:
        seal(path, updated)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    records = [reseal(target, apply=args.apply) for target in TARGETS]
    print(json.dumps({"applied": args.apply, "contracts": records}, indent=2))


if __name__ == "__main__":
    main()
