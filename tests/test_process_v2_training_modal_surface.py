"""Guards on the GPU training surface.

The value in reusing the P50 GPU function is its ENVIRONMENT -- image, A10G,
volume mount, deterministic cuBLAS. The danger is reusing its semantics: it
enforces the frozen fifty-step recipe and publishes a P50 decision. These tests
pin the separation, and the two traps that would each cost a paid run.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "modal_apps/run_process_v2_training_app.py").read_text()


def test_it_reuses_the_p50_environment_but_not_its_semantics() -> None:
    # environment reuse: the expensive, already-proven part
    assert "from modal_apps.run_process_v2_p50_app import" in SOURCE
    assert 'gpu="A10G"' in SOURCE
    assert "DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG" in SOURCE
    assert "volumes={str(ARTIFACT_ROOT): artifact_volume}" in SOURCE
    # semantics NOT reused: no prepared/collated pair, no decision publishing
    for forbidden in ("validate_prepared", "collated_completion", "P50_RECIPE_POLICY",
                      "_load_prerequisites", "PROCESS_V2_P50_DECISION"):
        assert forbidden not in SOURCE, forbidden


def test_the_training_wall_is_not_the_p50_wall() -> None:
    """P50's 20-minute timeout is sized for fifty steps.

    A two-thousand-step run under that wall is killed, losing everything since
    the last checkpoint. The training function needs its own, longer wall.
    """

    from modal_apps import run_process_v2_p50_app as p50
    from modal_apps import run_process_v2_training_app as trainer

    assert trainer.TRAINING_TIMEOUT_SECONDS > p50.GPU_TIMEOUT_SECONDS
    assert trainer.TRAINING_RESERVE_SECONDS > 0


def test_writes_take_the_physical_path_and_are_proven_first() -> None:
    """`/artifacts` is a symlink; the immutable writer refuses symlinked paths.

    A checkpoint write that fails at step 250 discards everything before it, so
    writability is exercised before any compute.
    """

    assert '_require_physical_artifact_path(run_root, field="run_root")' in SOURCE
    preflight = SOURCE.index("TRAINING_WRITABILITY_PREFLIGHT")
    training = SOURCE.index('loaded["run_training"]')
    assert preflight < training, "writability must be proven before training starts"


def test_it_resumes_rather_than_restarting() -> None:
    """A long run must outlive one container, so successive calls continue it."""

    assert 'glob("step-*.pt")' in SOURCE
    assert "resume_from=resume_from" in SOURCE
    assert "checkpoint_every_steps=CHECKPOINT_EVERY" in SOURCE


def test_it_claims_no_authority() -> None:
    assert '"artifact_published": False' in SOURCE
    assert '"checkpoint_selection_authorized": False' in SOURCE


def test_the_production_architecture_is_the_default() -> None:
    """hidden_dim 32 / 2 steps is a unit-test fixture giving a 74k-param toy.

    Production is 256 / 6, which is 6.2M parameters, and every real config uses
    it. The default must not be the fixture.
    """

    tree = ast.parse(SOURCE)
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "train")
    defaults = dict(zip([a.arg for a in node.args.args][-len(node.args.defaults):],
                        [ast.literal_eval(d) for d in node.args.defaults]))
    assert defaults["hidden_dim"] == 256
    assert defaults["message_passing_steps"] == 6
