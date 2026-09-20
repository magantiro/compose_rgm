"""Held-target 5HT1B launcher for the shared 250-call controller."""
import os
os.environ.setdefault("COMPOSE_HELD_CONTRACT", "configs/t4_held_target_distilled_5ht1b_d06_250.json")
os.environ.setdefault("COMPOSE_HELD_CHECKPOINT", "diagnostics/t4_held_target_distillation_quality_v1/5ht1b_checkpoint.json")
os.environ.setdefault("COMPOSE_HELD_VOLUME", "compose-t4-held-target-distilled-5ht1b-d06-250")
os.environ.setdefault("COMPOSE_HELD_OUTPUT", "/held_target_5ht1b_d06_250")
os.environ.setdefault("COMPOSE_HELD_APP", "compose-t4-held-target-distilled-5ht1b-d06-250")
os.environ.setdefault("COMPOSE_HELD_RECEPTOR_NAME", "5ht1b")
from modal_apps.t4_integrated_route_fiber_parp1_app import app, main  # noqa: E402,F401
