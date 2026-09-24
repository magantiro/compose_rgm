"""Seal the v3 base contract: v2 plus the measured-score bootstrap draw.

v2 is left BYTE-UNCHANGED. The running 10k campaigns are bound to it through their baked
image, and re-pinning a contract a live run validates against is the failure this repo has
already paid for twice. v3 is a separate artifact carrying `supersedes_payload_sha256`, the
same shape v2 used to supersede v1.

Only PINNED files whose bytes actually moved are re-hashed, and the script REFUSES to write
if any other field of the payload differs -- so "only plumbing moved" is checked, not asserted.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from compose_v4.control.edit_program import identity  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/pmo_population_controller_v2_proposal_repair.json"
OUT = ROOT / "configs/pmo_population_controller_v3_seed_weighting.json"

REVISION = (
    "seed_weighting_v3: the bootstrap parent draw over charged initialization molecules "
    "moves from uniform to the archive's own 1/rank rule "
    "(initial_parent_weighting='measured_score'). An archive entry carries an EditProgram, so "
    "a molecule handed to the campaign can never be one -- measured 0 of 16 initialization "
    "molecules among 112 entries -- and the bootstrap draw is its only access path. Uses ONLY "
    "counted feedback: the rows are charged through ledger.query(role='initialization'). "
    "Absent/'uniform' reproduces v2's draw byte-identically."
)


def main() -> int:
    envelope = json.loads(BASE.read_text())
    payload = envelope["payload"]
    if envelope["payload_sha256"] != identity(payload):
        raise SystemExit("v2 envelope hash does not match its own payload; refusing to seal")

    new = json.loads(json.dumps(payload))
    moved = {}
    for rel, pinned in new["implementation_sha256"].items():
        current = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        if current != pinned:
            moved[rel] = (pinned, current)
            new["implementation_sha256"][rel] = current
    if not moved:
        raise SystemExit("no pinned file moved; v3 would be a duplicate of v2")

    new["runtime_revision"] = REVISION
    new["supersedes_payload_sha256"] = envelope["payload_sha256"]

    # Everything OTHER than the three fields above must be byte-identical to v2, so the
    # claim that only plumbing moved is checkable rather than asserted.
    allowed = {"implementation_sha256", "runtime_revision", "supersedes_payload_sha256"}
    unexplained = [
        k for k in set(new) | set(payload)
        if k not in allowed and json.dumps(new.get(k), sort_keys=True)
        != json.dumps(payload.get(k), sort_keys=True)
    ]
    if unexplained:
        raise SystemExit(f"unexplained payload differences: {sorted(unexplained)}")

    OUT.write_text(json.dumps({"payload": new, "payload_sha256": identity(new)}, indent=1) + "\n")
    print(f"sealed {OUT.name}")
    print(f"  supersedes {envelope['payload_sha256'][:16]}")
    print(f"  payload    {identity(new)[:16]}")
    for rel, (old, cur) in sorted(moved.items()):
        print(f"  re-pinned  {rel}  {old[:8]} -> {cur[:8]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
