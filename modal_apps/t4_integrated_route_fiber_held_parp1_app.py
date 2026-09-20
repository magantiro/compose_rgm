"""Held-target PARP1 launcher for the shared 250-call controller."""
import os
os.environ.setdefault("COMPOSE_HELD_CONTRACT", "configs/t4_held_target_distilled_parp1_d06_250.json")
os.environ.setdefault("COMPOSE_HELD_CHECKPOINT", "diagnostics/t4_held_target_distillation_quality_v1/parp1_checkpoint.json")
os.environ.setdefault("COMPOSE_HELD_VOLUME", "compose-t4-held-target-distilled-parp1-d06-250")
os.environ.setdefault("COMPOSE_HELD_OUTPUT", "/held_target_parp1_d06_250")
os.environ.setdefault("COMPOSE_HELD_APP", "compose-t4-held-target-distilled-parp1-d06-250")
os.environ.setdefault("COMPOSE_HELD_RECEPTOR_NAME", "parp1")
from modal_apps.t4_integrated_route_fiber_parp1_app import app, main  # noqa: E402,F401
