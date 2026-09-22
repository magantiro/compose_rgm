#!/usr/bin/env python
"""Seal the LOCAL scored contract for the completion-repair A/B.

WHY A SEPARATE CONTRACT
-----------------------
``configs/pmo_population_controller_v1.json`` pins ``implementation_sha256`` over
the controller sources, and the completion repair moves three of them.  That
contract also carries ``scored_launch_authorized: false`` and is the artifact the
DEPLOYED Modal app validates against.  Re-pointing its pins or flipping its
authorization flag would forge an authorization for a runtime nobody approved,
so this writes a NEW, separate, local-only contract instead and leaves the
deployed one byte-identical.

This contract authorizes exactly what the session instruction authorizes: a
matched 250-call celecoxib A/B, and 250 calls each on perindopril_mpo and jnk3
only if the celecoxib arm qualifies.  It carries no Modal authorization.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from compose_v4.control.docking_value import identity  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE = "configs/pmo_population_controller_v1.json"
OUT = "configs/pmo_completion_ab_local_contract_v1.json"
EXTRA_IMPLEMENTATION = (
    "src/compose_v4/control/completion_component_law.py",
    "src/compose_v4/control/completion_law_contract.py",
    "src/compose_v4/control/dynamic_program_synthesis.py",
    "src/compose_v4/control/dynamic_program_synthesis_v1.py",
    "src/compose_v4/control/dynamic_program_synthesis_v2.py",
    "src/compose_v4/control/dynamic_program_synthesis_v21.py",
    "src/compose_v4/experiments/pmo_population_v1.py",
    "configs/pmo_completion_component_bank_v1.json",
)


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    base = json.loads((ROOT / BASE).read_text())["payload"]
    files = dict.fromkeys([*base["implementation_sha256"], *EXTRA_IMPLEMENTATION])
    files.pop("modal_apps/pmo_population_v1_app.py", None)
    payload = {
        "schema_version": "pmo_completion_ab_local_contract_v1",
        "derived_from": {"path": BASE, "payload_sha256": identity(base)},
        "scientific_question": (
            "Does repairing the completion half of a generic edit module -- its SCALE "
            "and its CONTENT -- improve scored PMO search, holding initialization, "
            "memory, allocator, region selector, proposal allocation, oracle, budget "
            "and seeds identical?"
        ),
        "controller": base["controller"],
        "initialization": base["initialization"],
        "joint_checkpoint": base["joint_checkpoint"],
        "oracle": {**base["oracle"], "asset_backed": False},
        "tasks": ["celecoxib_rediscovery", "perindopril_mpo", "jnk3"],
        "arms": {
            "A_deployed_b": {
                "enable_online_memory": True,
                "completion_law": None,
                "role": "control -- the deployed measured winner, v1 completion verbatim",
            },
            "B_completion": {
                "enable_online_memory": True,
                "completion_law": None,  # filled by the freeze step
                "role": "treatment -- the same controller plus the frozen completion law",
            },
        },
        "charged_calls_per_task": 250,
        "development_budget": {
            "celecoxib_rediscovery": {"arms": 2, "calls_per_arm": 250},
            "qualification_panel": {
                "tasks": ["perindopril_mpo", "jnk3"],
                "calls_per_task": 250,
                "conditional_on": "the celecoxib A/B qualifying",
            },
        },
        "runtime": "local, pinned PMO kernel; NOT the deployed Modal app",
        "scored_launch_authorized": True,
        "modal_launch_authorized": False,
        "authorization": {
            "granted_by": "session instruction, 2026-09-22",
            "text": (
                "AUTHORIZED: a zero-oracle 2x2 FIRST, then a matched 250-call "
                "celecoxib A/B (500 charged calls), and only if that qualifies, 250 "
                "calls each on perindopril_mpo and jnk3 (500 more). Nothing else."
            ),
            "total_charged_calls_authorized": 1000,
        },
        "prohibitions": base.get("prohibitions"),
        "implementation_sha256": {
            path: _sha256(ROOT / path) for path in sorted(files)
        },
    }
    if len(sys.argv) > 1:
        payload["arms"]["B_completion"]["completion_law"] = sys.argv[1]
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    (ROOT / OUT).write_text(json.dumps(envelope, sort_keys=True, indent=1))
    print("wrote", OUT)
    print("payload_sha256", envelope["payload_sha256"])
    print("B arm completion_law:", payload["arms"]["B_completion"]["completion_law"])


main()
