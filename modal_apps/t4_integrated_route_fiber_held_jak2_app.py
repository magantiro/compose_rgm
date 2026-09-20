"""Held-target JAK2 launcher using the shared PARP1 controller implementation.

The imported implementation is configured before import, so the deployed image,
contract, checkpoint and output namespace are all JAK2-specific while the
FiberControl/archive/executor code remains identical.
"""

from __future__ import annotations

import os

os.environ.setdefault("COMPOSE_HELD_CONTRACT", "configs/t4_held_target_distilled_jak2_d06_v1.json")
os.environ.setdefault(
    "COMPOSE_HELD_CHECKPOINT",
    "diagnostics/t4_held_target_distillation_quality_v1/jak2_checkpoint.json",
)
os.environ.setdefault("COMPOSE_HELD_VOLUME", "compose-t4-held-target-distilled-jak2-d06-v1")
os.environ.setdefault("COMPOSE_HELD_OUTPUT", "/held_target_jak2_v1")
os.environ.setdefault("COMPOSE_HELD_APP", "compose-t4-held-target-distilled-jak2-d06-v1")

from modal_apps.t4_integrated_route_fiber_parp1_app import app, main  # noqa: E402,F401

