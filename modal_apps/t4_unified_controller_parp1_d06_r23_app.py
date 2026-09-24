"""Unified T4 controller launcher for parp1 at delta=0.6, replicates 2 and 3.

Sets only the arm's identity. The CONTROLLER is byte-identical to every other arm;
the protein, the similarity threshold and the stochastic replicate seed are all
INPUTS rather than configuration.

Pinned in its contract's `runtime_inputs_sha256` and baked into the image, because
`_validate_task` re-hashes every pinned entry inside the container and the wrapper
is what selects the arm.
"""

from __future__ import annotations

import os

os.environ.setdefault(
    "COMPOSE_HELD_CONTRACT",
    "configs/t4_unified_controller_parp1_d06_r23_v1.json",
)
os.environ.setdefault(
    "COMPOSE_HELD_CHECKPOINT",
    "diagnostics/t4_held_target_distillation_quality_v1/parp1_checkpoint.json",
)
os.environ.setdefault("COMPOSE_HELD_VOLUME", "compose-t4-unified-controller-parp1-v1")
os.environ.setdefault("COMPOSE_HELD_OUTPUT", "/unified_parp1")
os.environ.setdefault("COMPOSE_HELD_APP", "compose-t4-unified-controller-parp1-r23-v1")
os.environ.setdefault("COMPOSE_HELD_RECEPTOR_NAME", "parp1")
os.environ.setdefault(
    "COMPOSE_HELD_WRAPPER",
    "modal_apps/t4_unified_controller_parp1_d06_r23_app.py",
)

from modal_apps.t4_unified_controller_app import app, main  # noqa: F401
