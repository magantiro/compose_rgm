"""Regression: a preemption retry must not carry BOTH an initialization flag and --resume-checkpoint.

Modal restarts a preempted container with the SAME input, so a warm-started run still carries
``--initialize-compatible-checkpoint`` on the retry. The gate rejects initialization together with
resume as mutually exclusive, so the retry died in argument parsing -- which is exactly how the first
RingCore-V1 scientific run ended: preempted at step 1500 with a valid recovery checkpoint on the
volume, then two failed restarts and a stopped app.

The launcher now drops the initialization arguments when it points the recipe at the recovery state.
These tests assert the resolution against the GATE's own exclusivity rule rather than a duplicated
list of flag names, so a new checkpoint-loading flag added to the gate cannot silently reintroduce the
collision.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_APP = _ROOT / "modal_apps" / "train_tracelet_gm.py"
_GATE = _ROOT / "scripts" / "train_tracelet_cnof_gate.py"


def _load_app_module():
    """Import the Modal app for its pure helpers (module import must not need a Modal connection)."""
    spec = importlib.util.spec_from_file_location("_train_tracelet_gm_under_test", _APP)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:  # pragma: no cover - surfaces an import-time regression clearly
        pytest.skip(f"modal app not importable in this environment: {error}")
    return module


def _gate_exclusive_arguments() -> set[str]:
    """The argument names the gate's mutual-exclusivity check counts, read from the gate itself."""
    text = _GATE.read_text()
    match = re.search(
        r"checkpoint_modes\s*=\s*sum\(\s*\(?\s*bool\(\s*\(?(.*?)\)?\s*\)?\s*\)\s*\)",
        text,
        re.DOTALL,
    )
    if match is None:
        # Fall back to the argument list that appears in the error message's vicinity.
        match = re.search(r"checkpoint_modes = sum\((.*?)\n    \)", text, re.DOTALL)
    assert match is not None, "could not locate the gate's checkpoint mutual-exclusivity check"
    return set(re.findall(r"args\.([a-z_]+)", match.group(1)))


def test_resume_drops_every_initialization_argument_the_gate_excludes():
    module = _load_app_module()
    excluded = _gate_exclusive_arguments()
    assert "resume_checkpoint" in excluded, "gate no longer excludes resume_checkpoint"

    # Every excluded argument other than resume itself must be handled by the launcher.
    initialization = excluded - {"resume_checkpoint"}
    handled = set(module._INITIALIZATION_ARGUMENTS)
    missing = initialization - handled
    assert not missing, (
        f"the gate excludes {sorted(missing)} against --resume-checkpoint but the launcher does not "
        "drop them on resume; a preemption retry would die in argument parsing"
    )


def test_warm_started_recipe_becomes_a_single_checkpoint_mode_on_resume():
    module = _load_app_module()
    # The exact argument set the killed run carried into its retry.
    arguments = {
        "initialize_compatible_checkpoint": (
            "/artifacts/compose-v4-stage3-flexible-graft-3k-1ac6f19-v1/checkpoint.best_so_far.pt"
        ),
        "model": "from_scratch",
        "training_steps": 16000,
    }
    dropped = module._supersede_initialization_for_resume(
        arguments, recovery_path=Path("/artifacts/run/checkpoint.recovery.pt")
    )

    assert dropped == ["initialize_compatible_checkpoint"]
    assert arguments["resume_checkpoint"] == "/artifacts/run/checkpoint.recovery.pt"
    # exactly one checkpoint mode remains -> the gate's guard passes
    modes = sum(bool(arguments.get(name)) for name in _gate_exclusive_arguments())
    assert modes == 1, f"expected one checkpoint mode after resume, got {modes}: {arguments}"
    # unrelated arguments are untouched
    assert arguments["training_steps"] == 16000
    assert arguments["model"] == "from_scratch"


def test_resume_is_idempotent_and_harmless_without_initialization():
    module = _load_app_module()
    arguments = {"model": "from_scratch"}
    first = module._supersede_initialization_for_resume(
        arguments, recovery_path=Path("/artifacts/run/checkpoint.recovery.pt")
    )
    second = module._supersede_initialization_for_resume(
        arguments, recovery_path=Path("/artifacts/run/checkpoint.recovery.pt")
    )
    assert first == [] and second == []
    modes = sum(bool(arguments.get(name)) for name in _gate_exclusive_arguments())
    assert modes == 1


def test_launcher_records_what_it_superseded():
    """The run log must state that initialization was dropped -- a silent drop hides a real change."""
    text = _APP.read_text()
    assert "resume_from_recovery" in text, "resume must emit an auditable phase marker"
    assert "superseded_initialization" in text, (
        "the resume marker must record which initialization arguments were dropped"
    )
