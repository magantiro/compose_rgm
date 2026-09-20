"""Held-target JAK2 launcher at the correctly sealed delta=0.6.

Companion to `t4_integrated_route_fiber_held_jak2_app.py`, which runs the same frozen
controller and the same held-target JAK2 prior at delta=0.4 (a value inherited during
sealing).  Both arms are valid at their own constraint; this one supplies the delta=0.6
comparison the T4 panel is missing.  Output namespace, volume and app name are distinct
so the two arms cannot collide.
"""

from __future__ import annotations

import os

os.environ.setdefault(
    "COMPOSE_HELD_CONTRACT", "configs/t4_held_target_distilled_jak2_true_d06_250.json"
)
os.environ.setdefault(
    "COMPOSE_HELD_CHECKPOINT",
    "diagnostics/t4_held_target_distillation_quality_v1/jak2_checkpoint.json",
)
os.environ.setdefault("COMPOSE_HELD_VOLUME", "compose-t4-held-target-jak2-true-d06-250")
os.environ.setdefault("COMPOSE_HELD_OUTPUT", "/held_target_jak2_true_d06_250")
os.environ.setdefault("COMPOSE_HELD_APP", "compose-t4-held-target-jak2-true-d06-250")
os.environ.setdefault("COMPOSE_HELD_RECEPTOR_NAME", "jak2")

from modal_apps.t4_integrated_route_fiber_parp1_app import app, main  # noqa: F401
