"""Revival launcher for jak2 at delta=0.6: the round-0-preempted cells.

Sets only the arm's identity. The CONTROLLER is byte-identical to every other arm
and to the live rahul-94866 panel -- `modal_apps/t4_unified_controller_app.py` is
unchanged from 4a600557 -- so the only thing that differs is WHICH cells run and
WHERE the artifacts land.

Pinned in its contract's `runtime_inputs_sha256` and baked into the image, because
`_validate_task` re-hashes every pinned entry inside the container and the wrapper
is what selects the arm.
"""

from __future__ import annotations

import os

os.environ.setdefault(
    "COMPOSE_HELD_CONTRACT",
    "configs/t4_unified_controller_jak2_d06_rev_v1.json",
)
os.environ.setdefault(
    "COMPOSE_HELD_CHECKPOINT",
    "diagnostics/t4_held_target_distillation_quality_v1/jak2_checkpoint.json",
)
os.environ.setdefault("COMPOSE_HELD_VOLUME", "compose-t4-unified-controller-jak2-rev-v1")
os.environ.setdefault("COMPOSE_HELD_OUTPUT", "/unified_jak2_rev")
os.environ.setdefault("COMPOSE_HELD_APP", "compose-t4-unified-controller-jak2-d06-rev-v1")
os.environ.setdefault("COMPOSE_HELD_RECEPTOR_NAME", "jak2")
os.environ.setdefault(
    "COMPOSE_HELD_WRAPPER",
    "modal_apps/t4_unified_controller_jak2_d06_rev_app.py",
)

from modal_apps.t4_unified_controller_app import app, main  # noqa: F401
