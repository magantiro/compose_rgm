"""Seal the JAK2 delta=0.6 arm that the 250-call panel is missing.

`configs/t4_held_target_distilled_jak2_d06_250.json` carries `delta = 0.4`, inherited
from its `frozen_from.controller_contract` (a delta=0.4 lineage) while
`charged_calls_per_cell` was correctly updated to 250.  `delta` is executable -- it
constructs the `Fiber` -- while the contract's `claim_boundary` prose says 0.6, so that
run is a valid looser-constraint experiment mislabelled as the headline.

This builds the companion arm with the SAME frozen controller, the SAME held-target JAK2
prior and the SAME 250-call budget, changing only the constraint to a correctly sealed
0.6.  It is a fresh arm with no predecessor, so `resume_predecessor` is dropped rather
than chained: there is no prior progress at this identity to resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from compose_v4.control.docking_value import identity

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "configs/t4_held_target_distilled_jak2_d06_250.json"
OUTPUT = "configs/t4_held_target_distilled_jak2_true_d06_250.json"
BASELINE = "configs/t4_published_invirtuogen_baseline.json"


def sha256(relative: str) -> str:
    digest = hashlib.sha256()
    with (ROOT / relative).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    envelope = json.loads((ROOT / SOURCE).read_text())
    payload = dict(envelope["payload"])
    if identity(payload) != envelope["payload_sha256"]:
        raise ValueError("source jak2 contract payload hash does not match its payload")
    if payload["delta"] != 0.4:
        raise ValueError(f"expected the mis-sealed delta 0.4 source, found {payload['delta']}")

    published = json.loads((ROOT / BASELINE).read_text())["delta_0_6"]["jak2"]
    if len(published) != len(payload["cells"]):
        raise ValueError("published jak2 baseline does not cover the sealed cells")

    payload["delta"] = 0.6
    payload["schema_version"] = "t4_held_target_distilled_jak2_true_d06_250_contract_v1"
    payload["status"] = "SEALED_PENDING_EXACT_USER_AUTHORIZATION"
    payload.pop("resume_predecessor", None)
    payload["reported_ivg_delta_0_6"] = published
    payload["reported_ivg_source"] = BASELINE
    payload["experimental_setting"] = (
        "prospective three-cell JAK2 delta=0.6 held-target panel; 250 charged calls per cell; "
        "companion arm to the delta=0.4 run sealed as "
        "t4_held_target_distilled_jak2_d06_250.json, which remains valid at its own constraint"
    )
    payload["supersedes_delta"] = {
        "contract": SOURCE,
        "contract_payload_sha256": envelope["payload_sha256"],
        "delta": 0.4,
        "reason": (
            "the 0.4 value was inherited from the frozen_from controller contract during "
            "sealing while the call budget was updated; both arms are kept"
        ),
    }
    for relative in sorted(payload["runtime_inputs_sha256"]):
        payload["runtime_inputs_sha256"][relative] = sha256(relative)

    sealed = {"payload": payload, "payload_sha256": identity(payload)}
    if not args.apply:
        print(json.dumps({
            "output": OUTPUT,
            "payload_sha256": sealed["payload_sha256"],
            "delta": payload["delta"],
            "cells": [c["cell"] for c in payload["cells"]],
            "charged_calls_per_cell": payload["charged_calls_per_cell"],
        }, indent=2))
        print("dry run: pass --apply to write")
        return
    target = ROOT / OUTPUT
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(sealed, sort_keys=True, indent=2) + "\n")
    temporary.replace(target)
    print(json.dumps({"output": OUTPUT, "payload_sha256": sealed["payload_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
