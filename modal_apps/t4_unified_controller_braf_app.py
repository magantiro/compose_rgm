"""Unified T4 controller launcher for braf at delta=0.6.

One of five thin wrappers over `modal_apps/t4_unified_controller_app.py`.  Every
wrapper sets only the arm's identity -- contract, route-expert checkpoint, volume,
output namespace, app name and receptor -- so the CONTROLLER is byte-identical
across the panel and the protein is an input rather than a configuration.

This wrapper is PINNED in its contract's `runtime_inputs_sha256` and baked into
the image, because `_validate_task` re-hashes every pinned entry inside the
container and the wrapper is what selects the arm.
"""

from __future__ import annotations

import os

os.environ.setdefault(
    "COMPOSE_HELD_CONTRACT", "configs/t4_unified_controller_braf_d06_v1.json"
)
os.environ.setdefault(
    "COMPOSE_HELD_CHECKPOINT",
    "diagnostics/t4_held_target_distillation_quality_v1/braf_checkpoint.json",
)
os.environ.setdefault("COMPOSE_HELD_VOLUME", "compose-t4-unified-controller-braf-v1")
os.environ.setdefault("COMPOSE_HELD_OUTPUT", "/unified_braf")
os.environ.setdefault("COMPOSE_HELD_APP", "compose-t4-unified-controller-braf-v1")
os.environ.setdefault("COMPOSE_HELD_RECEPTOR_NAME", "braf")
os.environ.setdefault(
    "COMPOSE_HELD_WRAPPER", "modal_apps/t4_unified_controller_braf_app.py"
)

from modal_apps.t4_unified_controller_app import app, main  # noqa: F401
