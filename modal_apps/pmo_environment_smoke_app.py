"""Zero-oracle-call environment smoke for the PMO population image.

The first scored PMO attempt died on every task with `ModuleNotFoundError: No module
named 'torch'`, raised transitively by `from tdc import Oracle`, before any oracle call
was charged.  This app proves the production image can walk that exact import and
instantiation path.

It deliberately imports `image` from the scored app rather than redefining one, because a
smoke that passes on a different image proves nothing about the image that will run.  It
scores NO molecule: the oracle is constructed and then discarded, and the controller is
built but never stepped.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

from modal_apps.pmo_population_v1_app import (
    ORACLE_CONTRACT,
    REMOTE_ROOT,
    _rdkit_six_shim,
    image,
)

app = modal.App("compose-pmo-environment-smoke")


@app.function(image=image, cpu=(1.0, 1.0), memory=8192, timeout=30 * 60, retries=0)
def smoke(task_name: str = "gsk3b") -> dict:
    """Import torch, instantiate the TDC oracle, build the controller, charge nothing."""

    import os
    import sys

    report: dict = {"schema_version": "pmo_environment_smoke_v1", "task": task_name}

    import torch

    report["torch_version"] = torch.__version__
    report["torch_file"] = torch.__file__

    from compose_v4.experiments.pmo_dynamic_v21 import verify_runtime_environment

    root = REMOTE_ROOT
    old_oracle = json.loads((root / ORACLE_CONTRACT).read_text())["payload"]
    report["runtime_environment"] = verify_runtime_environment(root, old_oracle)

    _rdkit_six_shim()
    from tdc import Oracle

    report["tdc_module"] = sys.modules["tdc"].__file__

    previous = Path.cwd()
    os.chdir(root / "diagnostics/pmo_ivg_oracle_parity/ivg_oracle_assets")
    try:
        oracle = Oracle(name=task_name)
    finally:
        os.chdir(previous)
    report["oracle_constructed"] = type(oracle).__name__
    report["oracle_called"] = False

    # Build the frozen controller exactly as the scored task would, and stop there.
    from compose_v4.control.pmo_population_controller import PmoPopulationController
    from compose_v4.experiments.pmo_population_v1 import (
        _load_checkpoint,
        configuration,
        load_contract,
    )

    contract = load_contract(root)
    checkpoint = _load_checkpoint(root, contract)
    controller = PmoPopulationController(
        configuration(),
        source_group="pmo-smoke",
        oracle_protocol="pmo-smoke",
        hierarchy=None,
        jump_checkpoint=checkpoint,
    )
    report["controller"] = type(controller).__name__
    report["credit_cells"] = len(controller.credit.cells)
    report["exploration_floor"] = controller.credit.exploration_floor
    report["scored_launch_authorized_on_disk"] = contract.get("scored_launch_authorized")
    report["charged_oracle_calls"] = 0
    return report


@app.local_entrypoint()
def main(task: str = "gsk3b") -> None:
    print(json.dumps(smoke.remote(task), indent=2, sort_keys=True))
