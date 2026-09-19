"""Three-cell COMPOSE-NoDistill transfer app using the shared production runtime."""

from __future__ import annotations

import os

os.environ["COMPOSE_T4_CAMPAIGN_VARIANT"] = "nodistill_transfer_v1"

from modal_apps.t4_shared_controller_completion_v1_app import (
    CELL_FUNCTIONS,
    app,
    local_scored_context,
    main,
    scored_preflight_report,
)

__all__ = [
    "CELL_FUNCTIONS",
    "app",
    "local_scored_context",
    "main",
    "scored_preflight_report",
]
