"""Held-target BRAF launcher at delta=0.4, the paired arm of the frozen delta=0.6 panel.

Companion to `t4_integrated_route_fiber_held_braf_app.py`, which runs the same frozen
controller and the same held-target BRAF prior at delta=0.6.  The two arms differ in
exactly one executable field, the similarity corridor, so the per-cell comparison between
them is paired.  App name, volume and output namespace are distinct so the arms cannot
collide, and neither can touch the live delta=0.6 run.
"""

from __future__ import annotations

import os

os.environ.setdefault(
    "COMPOSE_HELD_CONTRACT", "configs/t4_held_target_distilled_braf_d04_250.json"
)
os.environ.setdefault(
    "COMPOSE_HELD_CHECKPOINT",
    "diagnostics/t4_held_target_distillation_quality_v1/braf_checkpoint.json",
)
os.environ.setdefault("COMPOSE_HELD_VOLUME", "compose-t4-held-target-distilled-braf-d04-250")
os.environ.setdefault("COMPOSE_HELD_OUTPUT", "/held_target_braf_d04_250")
os.environ.setdefault("COMPOSE_HELD_APP", "compose-t4-held-target-distilled-braf-d04-250")
os.environ.setdefault("COMPOSE_HELD_RECEPTOR_NAME", "braf")

from modal_apps.t4_integrated_route_fiber_parp1_app import app, main  # noqa: F401
