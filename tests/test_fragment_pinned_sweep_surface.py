"""The pinned fragment sweep's argv IS the configuration authority for its rows.

Every published pinned row was produced by a subprocess this app assembled, so a
knob that is plumbed as far as the local entrypoint but never reaches that argv
is INERT while looking wired at every level a reader checks. A contract on disk
does not reach a deployed Modal app either, but that is a deployment question;
this is the part a test can settle.

``shard_argv`` is EXECUTED here, not read. The module imports ``modal``, which is
not installed in the test environment, so the function is compiled out of the
production source and run in an isolated namespace: the bytes under test are the
bytes that ship, and nothing is transcribed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "modal_apps" / "fragment_pinned_sweep_app.py"


def _shard_argv():
    """Compile the production ``shard_argv`` without importing modal."""
    tree = ast.parse(APP_PATH.read_text())
    wanted = {"shard_argv", "ARMS", "SAMPLES"}
    kept = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in wanted)
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id in wanted for t in node.targets
            )
        )
    ]
    assert any(
        isinstance(n, ast.FunctionDef) and n.name == "shard_argv" for n in kept
    ), "the app must keep the argv construction in a plain function"
    namespace: dict = {}
    code = compile(ast.Module(body=kept, type_ignores=[]), str(APP_PATH), "exec")
    exec(code, namespace)  # noqa: S102 -- production bytes, isolated namespace, no import of modal
    return namespace["shard_argv"], namespace["ARMS"]


def _argv(arm: str, **overrides):
    shard_argv, _arms = _shard_argv()
    kwargs = {
        "arm": arm,
        "task": "linker_design",
        "drug": "BARICITINIB",
        "seed": 0,
        "mark_attempts": 96,
        "linker_bridge_atoms": 1,
        "checkpoint": "/ckpt.pt",
        "output": "/out.json",
        "executable": "/python",
        "runner": "/runner.py",
    }
    kwargs.update(overrides)
    return shard_argv(**kwargs)


def test_the_path_program_arm_switches_the_program_on():
    argv = _argv("path_program")
    assert "--path-program" in argv
    assert "--attachment-control" in argv


def test_the_attachment_arm_leaves_the_program_off():
    """The matched comparison isolates ONE difference.

    Both arms of the linker comparison run the attachment controller; only the
    path program differs. An arm that also switched the controller would be
    comparing two mechanisms at once.
    """
    argv = _argv("attachment")
    assert "--attachment-control" in argv
    assert "--path-program" not in argv


def test_the_baseline_arm_switches_neither_on():
    argv = _argv("baseline")
    assert "--attachment-control" not in argv
    assert "--path-program" not in argv


def test_the_seeded_bridge_reaches_the_argv_for_every_arm():
    """A start-state knob must be identical across arms or the arms differ in task."""
    _shard_argv_fn, arms = _shard_argv()
    for arm in arms:
        argv = _argv(arm, linker_bridge_atoms=2)
        assert "--linker-bridge-atoms" in argv
        assert argv[argv.index("--linker-bridge-atoms") + 1] == "2"


def test_the_rejection_budget_reaches_the_argv():
    argv = _argv("path_program", mark_attempts=96)
    assert argv[argv.index("--mark-attempts-per-event") + 1] == "96"


def test_an_unknown_arm_is_refused_rather_than_run_as_a_baseline():
    """A typo must not silently produce an unlabelled arm."""
    with pytest.raises(ValueError):
        _argv("attachement")


def test_no_per_drug_or_per_task_knob_is_expressible_in_a_shard():
    """One frozen configuration per sweep, never one per instance.

    ``--task`` and ``--drug`` SELECT which instance a shard runs; nothing in the
    argv scopes a sampler or controller setting to one.
    """
    argv = _argv("path_program")
    for flag in ("--mark-attempts-per-event", "--linker-bridge-atoms"):
        assert argv.count(flag) == 1
