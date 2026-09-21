"""Re-pin the two protonation-rescue contracts onto the repaired launch path.

Why this exists
---------------
Each contract pins its own Modal wrapper in ``runtime_inputs_sha256``, because the
remote worker must prove it is running the protonation-wired code.  But that same file
also carries the LOCAL launch plumbing, and that plumbing was broken: it validated
against the superseded v1 contract module and demanded an authorized status inside the
scientific payload.  Repairing it necessarily moves the wrapper's hash, which moves the
payload hash that both the zero-oracle feasibility gate and the owner's authorization
receipt address.

So the re-pin records exactly what moved, in a ``launch_path_revision`` block:

    supersedes_payload_sha256   the identity the gate and the receipt name
    changed_runtime_inputs      {path: {from, to}}   -- the wrapper
    added_runtime_inputs        {path: sha256}       -- the v2 validator module

``t4_protonation_rescue_contract_v2.reconstruct_superseded_payload`` undoes exactly
those edits and re-hashes; the preflight requires the result to equal the superseded
identity.  Continuity is therefore PROVED, not asserted: any other mutation -- a delta,
a ceiling, a cell, a proposal setting, an added field -- survives the reconstruction and
breaks the hash.

This script refuses to run twice and refuses to touch a contract whose current identity
is not the one the authorization receipt names.  It charges no oracle calls and launches
nothing.

    python3 tools/t4_5ht1b2_protonation_rescue_repin_launch_path.py [--apply]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compose_v4.control.docking_value import identity
from compose_v4.experiments.t4_protonation_rescue_contract_v2 import (
    ARMS,
    AUTHORIZATION_RECEIPT,
    reconstruct_superseded_payload,
)

ADDED_MODULE = "src/compose_v4/experiments/t4_protonation_rescue_contract_v2.py"
REASON = (
    "the wrapper pinned here validated against the superseded v1 contract module and "
    "required an authorized status inside the scientific payload; both are repaired. "
    "Only the wrapper pin moved and only the v2 validator pin was added -- no delta, "
    "ceiling, cell, proposal, allocation or endpoint-gate field changed, which "
    "reconstruct_superseded_payload proves by rebuilding the superseded payload."
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dump(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write the contracts")
    args = parser.parse_args()

    receipt = json.loads((ROOT / AUTHORIZATION_RECEIPT).read_text())
    named = receipt["authorized_payload_sha256_at_authorization"]

    for arm, spec in ARMS.items():
        contract_path = ROOT / spec["contract"]
        document = json.loads(contract_path.read_text())
        payload = document["payload"]
        current = identity(payload)
        if identity(payload) != document["payload_sha256"]:
            raise SystemExit(f"{arm}: contract envelope is corrupt")
        if "launch_path_revision" in payload:
            raise SystemExit(f"{arm}: already carries a launch_path_revision; refusing")
        if current != named.get(arm):
            raise SystemExit(
                f"{arm}: contract is at {current}, but the authorization receipt names "
                f"{named.get(arm)}; refusing to re-pin an unauthorized payload"
            )

        app_relative = spec["app"]
        old_app = payload["runtime_inputs_sha256"][app_relative]
        new_app = _sha256_file(ROOT / app_relative)
        new_module = _sha256_file(ROOT / ADDED_MODULE)
        if old_app == new_app and payload["runtime_inputs_sha256"].get(
            ADDED_MODULE
        ) == new_module:
            raise SystemExit(f"{arm}: nothing to re-pin")

        revised = json.loads(json.dumps(payload))
        revised["runtime_inputs_sha256"][app_relative] = new_app
        revised["runtime_inputs_sha256"][ADDED_MODULE] = new_module
        revised["launch_path_revision"] = {
            "added_runtime_inputs": {ADDED_MODULE: new_module},
            "changed_runtime_inputs": {app_relative: {"from": old_app, "to": new_app}},
            "reason": REASON,
            "supersedes_payload_sha256": current,
        }

        rebuilt = identity(reconstruct_superseded_payload(revised))
        if rebuilt != current:
            raise SystemExit(
                f"{arm}: reconstruction gives {rebuilt}, expected {current}"
            )
        revised_identity = identity(revised)
        print(
            f"{arm}: {current} -> {revised_identity}\n"
            f"     wrapper {old_app[:16]} -> {new_app[:16]}\n"
            f"     added   {ADDED_MODULE} {new_module[:16]}\n"
            f"     reconstruction back to the authorized payload: OK"
        )
        if args.apply:
            _dump(
                contract_path,
                {"payload": revised, "payload_sha256": revised_identity},
            )
            print(f"     wrote {spec['contract']}")

    if not args.apply:
        print("\ndry run; pass --apply to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
