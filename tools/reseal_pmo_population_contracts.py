"""Re-pin the PMO population contracts after a deliberate controller revision.

The joint credit object is now the allocation authority inside
`PmoPopulationController`, and `pmo_credit.py` is therefore load-bearing for every
number the controller produces.  A contract that pins the controller but not its credit
module would authorize a runtime whose behaviour can change without the identity moving,
which is the defect this repository has already paid for once in a cache file list.

The re-pin is mechanical and guarded: only `*_sha256` pointer values may move.  Any
change to a policy, budget, task list or authorization flag aborts the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from compose_v4.control.docking_value import identity

ROOT = Path(__file__).resolve().parents[1]
CREDIT = "src/compose_v4/control/pmo_credit.py"
# The Modal app defines the remote image (including the torch install whose absence
# killed the first scored attempt), so it is part of the experimental identity: a
# smoke that passes on one image and a launch that runs on another must not be able
# to share a contract hash.
REMOTE_APP = "modal_apps/pmo_population_v1_app.py"
# Archive-continuity bookkeeping decides which measured programs the archive admits,
# so it is load-bearing for every number the controller produces.
CONTINUITY = "src/compose_v4/control/bootstrap_pool_continuity.py"
# Asset resolution and the pre-launch positive control sit on the scoring path: they
# decide whether a score is the oracle's output or PyTDC's swallowed default, so a
# contract that did not pin them could authorize a runtime whose numbers can change
# without the identity moving.
ORACLE_ASSETS = "src/compose_v4/experiments/pmo_oracle_assets.py"
BASE = "configs/pmo_population_controller_v1.json"
DEPENDENTS = (
    "configs/pmo_population_controller_v1_scored_contract.json",
    "configs/pmo_population_controller_v1_scored_contract_corrected.json",
    "configs/pmo_population_live_parent_gate_v2.json",
)
CONTROLLER = "src/compose_v4/control/pmo_population_controller.py"


def sha256(relative: str) -> str:
    digest = hashlib.sha256()
    with (ROOT / relative).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _strip_hashes(node):
    """The payload with every 64-hex pointer blanked, so only semantics remain."""
    if isinstance(node, dict):
        return {key: _strip_hashes(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_strip_hashes(value) for value in node]
    if isinstance(node, str) and len(node) == 64 and all(c in "0123456789abcdef" for c in node):
        return "<hash>"
    return node


def reseal(path: Path, *, base_payload_sha256: str | None) -> tuple[str, bool]:
    envelope = json.loads(path.read_text())
    payload = envelope["payload"]
    if identity(payload) != envelope["payload_sha256"]:
        raise ValueError(f"{path.name}: stored payload_sha256 does not match its payload")
    before = _strip_hashes(payload)
    declared_new_dependency = False

    implementation = payload.get("implementation_sha256")
    if implementation and CONTROLLER in implementation:
        # Declaring the new dependency is the ONE structural change this tool may make.
        # It is recorded so the guard below can allow exactly it and nothing else.
        declared = (CREDIT, REMOTE_APP, CONTINUITY, ORACLE_ASSETS)
        additions = [name for name in declared if name not in implementation]
        declared_new_dependency = bool(additions)
        for name in declared:
            implementation.setdefault(name, "")
        for relative in sorted(implementation):
            implementation[relative] = sha256(relative)
    if base_payload_sha256 and "controller_contract_sha256" in payload:
        payload["controller_contract_sha256"] = base_payload_sha256

    after = _strip_hashes(payload)
    if declared_new_dependency:
        expected = json.loads(json.dumps(before))
        for name in additions:
            expected["implementation_sha256"][name] = "<hash>"
        before = expected
    if before != after:
        raise ValueError(f"{path.name}: re-seal would change something other than a hash pointer")

    envelope["payload_sha256"] = identity(payload)
    moved = envelope["payload_sha256"] != json.loads(path.read_text())["payload_sha256"]
    path.write_text(json.dumps(envelope, sort_keys=True, indent=2) + "\n")
    return envelope["payload_sha256"], moved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.apply:
        print("dry run: pass --apply to write")
        return
    base_hash, moved = reseal(ROOT / BASE, base_payload_sha256=None)
    print(f"{BASE:62s} -> {base_hash[:16]} {'(moved)' if moved else '(unchanged)'}")
    for relative in DEPENDENTS:
        new_hash, moved = reseal(ROOT / relative, base_payload_sha256=base_hash)
        print(f"{relative:62s} -> {new_hash[:16]} {'(moved)' if moved else '(unchanged)'}")


if __name__ == "__main__":
    main()
