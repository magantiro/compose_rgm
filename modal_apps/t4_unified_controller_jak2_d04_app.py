"""Unified T4 controller launcher for jak2 at delta=0.4.

The delta=0.4 sibling of `t4_unified_controller_jak2_app.py`.  It sets only the
arm's identity -- contract, route-expert checkpoint, volume, output namespace, app
name and receptor -- so the CONTROLLER is byte-identical to every other arm and both
the protein and the similarity threshold are benchmark INPUTS rather than
configuration.

This wrapper is PINNED in its contract's `runtime_inputs_sha256` and baked into the
image, because `_validate_task` re-hashes every pinned entry inside the container and
the wrapper is what selects the arm.
"""

from __future__ import annotations

import os

os.environ.setdefault(
    "COMPOSE_HELD_CONTRACT", "configs/t4_unified_controller_jak2_d04_v1.json"
)
os.environ.setdefault(
    "COMPOSE_HELD_CHECKPOINT",
    "diagnostics/t4_held_target_distillation_quality_v1/jak2_checkpoint.json",
)
os.environ.setdefault("COMPOSE_HELD_VOLUME", "compose-t4-unified-controller-jak2-v1")
os.environ.setdefault("COMPOSE_HELD_OUTPUT", "/unified_jak2")
os.environ.setdefault("COMPOSE_HELD_APP", "compose-t4-unified-controller-jak2-d04-v1")
os.environ.setdefault("COMPOSE_HELD_RECEPTOR_NAME", "jak2")
os.environ.setdefault(
    "COMPOSE_HELD_WRAPPER", "modal_apps/t4_unified_controller_jak2_d04_app.py"
)

from modal_apps.t4_unified_controller_app import app, main  # noqa: F401
