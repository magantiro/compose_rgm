"""A checkpoint that resumes into a state that never existed is worse than none.

These tests pin the properties that make a multi-hour run attributable: it
resumes exactly (weights, optimizer moments AND RNG), it refuses to continue
against different frozen inputs, and it reports the held-out panel honestly
rather than collapsing two different questions into one number.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from compose_v4.experiments.editing_v2_r_theta_corpus_training import (
    THIN_CELL_ENTRIES,
    RThetaTrainingError,
    RunIdentity,
    assert_training_invariants,
    load_checkpoint,
    selection_criterion,
    summarize_panel,
    write_checkpoint,
)

SHA = "a" * 64


def _identity(**overrides) -> RunIdentity:
    values = dict(
        initialization_seed=20260730,
        initial_model_state_sha256="3" + SHA[1:],
        library_sha256="1" + SHA[1:],
        split_sha256="2" + SHA[1:],
        sampling_law_sha256="4" + SHA[1:],
        manifest_sha256="5" + SHA[1:],
        packed_store_records_sha256="6" + SHA[1:],
        eval_panel_sha256="7" + SHA[1:],
    )
    values.update(overrides)
    return RunIdentity(**values)


def _model() -> torch.nn.Module:
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 1))
    return model


def _trained(model: torch.nn.Module, optimizer: torch.optim.Optimizer, steps: int = 3):
    for _ in range(steps):
        optimizer.zero_grad()
        loss = model(torch.ones(2, 4)).sum()
        loss.backward()
        optimizer.step()


def test_checkpoint_restores_weights_optimizer_and_rng(tmp_path: Path) -> None:
    """Weights alone are not a resume.

    Restoring only weights silently restarts the optimizer moments and the RNG,
    which changes the trajectory while the step count still looks right.
    """

    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    _trained(model, optimizer)
    receipt = write_checkpoint(
        tmp_path / "ck.pt",
        model=model,
        optimizer=optimizer,
        identity=_identity(),
        completed_steps=3,
        selected_step=2,
        selected_criterion=(0.1, -2.0, -2),
        selected_state={k: v.clone() for k, v in model.state_dict().items()},
        trajectory=[{"step": 2, "mean_nll": 2.0}],
        stream_sha256=SHA,
        resume_count=0,
    )
    expected_next = torch.rand(3)

    restored = _model()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    payload = load_checkpoint(
        tmp_path / "ck.pt",
        model=restored,
        optimizer=restored_optimizer,
        identity=_identity(),
        expected_stream_sha256=SHA,
        expected_file_sha256=receipt["file_sha256"],
    )
    assert payload["completed_steps"] == 3
    for name, tensor in model.state_dict().items():
        assert torch.equal(restored.state_dict()[name], tensor)
    # Optimizer moments, not just weights.
    original = optimizer.state_dict()["state"]
    recovered = restored_optimizer.state_dict()["state"]
    assert set(original) == set(recovered) and original
    for key in original:
        assert torch.allclose(original[key]["exp_avg"], recovered[key]["exp_avg"])
    # And the RNG stream continues where it left off.
    assert torch.equal(torch.rand(3), expected_next)


def test_resume_against_different_frozen_inputs_is_refused(tmp_path: Path) -> None:
    """The failure that would silently relabel an experiment."""

    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    write_checkpoint(
        tmp_path / "ck.pt",
        model=model, optimizer=optimizer, identity=_identity(),
        completed_steps=1, selected_step=1, selected_criterion=None,
        selected_state=None, trajectory=[], stream_sha256=SHA, resume_count=0,
    )
    with pytest.raises(RThetaTrainingError, match="different frozen inputs"):
        load_checkpoint(
            tmp_path / "ck.pt",
            model=_model(),
            optimizer=torch.optim.AdamW(_model().parameters(), lr=1e-3),
            identity=_identity(sampling_law_sha256="9" + SHA[1:]),
            expected_stream_sha256=SHA,
        )


def test_resume_against_a_different_draw_stream_is_refused(tmp_path: Path) -> None:
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    write_checkpoint(
        tmp_path / "ck.pt",
        model=model, optimizer=optimizer, identity=_identity(),
        completed_steps=1, selected_step=1, selected_criterion=None,
        selected_state=None, trajectory=[], stream_sha256=SHA, resume_count=0,
    )
    with pytest.raises(RThetaTrainingError, match="different draw stream"):
        load_checkpoint(
            tmp_path / "ck.pt",
            model=_model(),
            optimizer=torch.optim.AdamW(_model().parameters(), lr=1e-3),
            identity=_identity(),
            expected_stream_sha256="b" * 64,
        )


def test_a_tampered_checkpoint_file_is_refused(tmp_path: Path) -> None:
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    receipt = write_checkpoint(
        tmp_path / "ck.pt",
        model=model, optimizer=optimizer, identity=_identity(),
        completed_steps=1, selected_step=1, selected_criterion=None,
        selected_state=None, trajectory=[], stream_sha256=SHA, resume_count=0,
    )
    with pytest.raises(RThetaTrainingError, match="file digest"):
        load_checkpoint(
            tmp_path / "ck.pt",
            model=_model(),
            optimizer=torch.optim.AdamW(_model().parameters(), lr=1e-3),
            identity=_identity(),
            expected_stream_sha256=SHA,
            expected_file_sha256="c" * 64,
        )
    assert receipt["file_bytes"] > 0


def test_checkpoint_write_is_atomic(tmp_path: Path) -> None:
    """No .partial file survives, so a kill cannot leave a torn checkpoint."""

    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    write_checkpoint(
        tmp_path / "ck.pt",
        model=model, optimizer=optimizer, identity=_identity(),
        completed_steps=1, selected_step=1, selected_criterion=None,
        selected_state=None, trajectory=[], stream_sha256=SHA, resume_count=0,
    )
    assert (tmp_path / "ck.pt").exists()
    assert not list(tmp_path.glob("*.partial"))


def _row(family: str, cell: str, nll: float) -> dict:
    return {
        "entry_id": f"{family}-{cell}-{nll}",
        "model_family": family,
        "capability_cell_id": cell,
        "teacher_successor_nll": nll,
        "teacher_successor_probability": math.exp(-nll),
        "teacher_successor_log_probability": -nll,
    }


def test_panel_reports_both_means_because_they_answer_different_questions() -> None:
    """The panel is not distributed like the training draw.

    MEASURED: cycle_insert is 36.33% of the panel against 6.50% of the law.
    A plain panel mean is therefore not an estimate of performance under the
    law, so reporting only one number invites reading it as the other.
    """

    rows = [_row("cycle_insert", "c1", 1.0)] * 8 + [_row("atom_delete", "c2", 5.0)] * 2
    summary = summarize_panel(
        rows, deployment_family_share={"cycle_insert": 0.065, "atom_delete": 0.166}
    )
    native = summary["panel_native"]["mean_nll"]
    weighted = summary["deployment_weighted_mean_nll"]
    assert native == pytest.approx(1.8)          # panel is 80% easy family
    assert weighted == pytest.approx(3.87, abs=0.01)  # law is not
    assert weighted > native


def test_thin_cells_are_flagged_rather_than_silently_gated() -> None:
    """A per-cell mean over two examples fires or not at random."""

    rows = [_row("bond_reroute", "thin", 2.0)] * 2 + [_row("atom_restate", "fat", 2.0)] * 40
    summary = summarize_panel(rows)
    assert summary["thin_cells"] == ["thin"]
    assert summary["by_capability_cell"]["thin"]["entries"] < THIN_CELL_ENTRIES


def test_selection_prefers_the_checkpoint_that_abandoned_nothing() -> None:
    """A mean hides a transition the model has given up on entirely."""

    uniform = summarize_panel([_row("f", "c", 2.0), _row("f", "c", 2.0)])
    lopsided = summarize_panel([_row("f", "c", 0.1), _row("f", "c", 3.5)])
    assert lopsided["panel_native"]["mean_nll"] < uniform["panel_native"]["mean_nll"]
    # ...but selection still prefers the one with no abandoned successor.
    assert selection_criterion(uniform, step=1) > selection_criterion(lopsided, step=1)


def test_empty_panel_is_refused() -> None:
    with pytest.raises(RThetaTrainingError, match="empty panel"):
        summarize_panel([])


def test_moved_hazard_parameters_stop_the_run() -> None:
    model = _model()
    before = {"0.weight": model.state_dict()["0.weight"].clone()}
    assert_training_invariants(model, hazard_before=before, step=1)
    with torch.no_grad():
        dict(model.named_parameters())["0.weight"].add_(1.0)
    with pytest.raises(RThetaTrainingError, match="frozen hazard parameters changed"):
        assert_training_invariants(model, hazard_before=before, step=2)
