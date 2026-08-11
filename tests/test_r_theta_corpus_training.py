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
    capability_baseline,
    capability_floor,
    collapsed_capabilities,
    regressed_capabilities,
    RThetaTrainingError,
    RunIdentity,
    resolve_launch_commit,
    assert_training_invariants,
    load_checkpoint,
    selection_criterion,
    summarize_panel,
    write_checkpoint,
)

SHA = "a" * 64


def _identity(**overrides) -> RunIdentity:
    values = dict(
        git_commit_sha="a1b2c3d4" * 5,
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


def _row(family: str, cell: str, nll: float, *, lane: str = "real",
         support_band: str | None = None, tag: str = "",
         identity: float | None = None) -> dict:
    # P(y|x) = P(F|x).P(y|x,F); the gate reads the identity factor only.
    identity_nll = nll / 2 if identity is None else identity
    return {
        "family_nll": nll - identity_nll,
        "identity_nll": identity_nll,
        "entry_id": f"{family}-{cell}-{nll}-{lane}-{tag}",
        "model_family": family,
        "capability_cell_id": cell,
        "lane": lane,
        "stratum": f"{family}|{lane}",
        "support_band": support_band,
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


def test_collapse_needs_two_consecutive_regressed_evaluations() -> None:
    """A single excursion is not a collapse.

    MEASURED in epoch 1: atom_restate went +0.24 -> +0.61 -> +0.53 -> +0.24
    against its baseline, gating two checkpoints and then returning to exactly
    where it started. One evaluation cannot tell an excursion from a loss.
    """

    shares = {"keep|real": 0.5, "rare|real": 0.5}
    start = summarize_panel(
        [_row("keep", "c1", 4.0, identity=2.0, tag=str(i)) for i in range(4)]
        + [_row("rare", "c2", 4.0, identity=2.0, tag=str(i)) for i in range(4)],
        reference_law_stratum_share=shares)
    baseline = capability_baseline(start)
    assert baseline["source"] == "initialization"

    excursion = summarize_panel(
        [_row("keep", "c1", 4.0, identity=2.0, tag=str(i)) for i in range(4)]
        + [_row("rare", "c2", 5.5, identity=3.5, tag=str(i)) for i in range(4)],
        reference_law_stratum_share=shares)
    # Regressed on this evaluation...
    assert regressed_capabilities(excursion, baseline=baseline) == ("family:rare",)
    # ...but not collapsed, because nothing was regressed before it.
    assert collapsed_capabilities(excursion, baseline=baseline,
                                  previously_regressed=()) == ()
    assert selection_criterion(excursion, step=1, baseline=baseline)[0] == 1
    # A second consecutive regression IS a collapse.
    assert collapsed_capabilities(
        excursion, baseline=baseline,
        previously_regressed=("family:rare",)) == ("family:rare",)
    assert selection_criterion(excursion, step=2, baseline=baseline,
                               previously_regressed=("family:rare",))[0] == 0


def test_family_head_reallocation_does_not_gate() -> None:
    """The joint moved 0.32 nats; the capability did not move at all.

    MEASURED: between steps 3,000 and 4,251 cycle_insert's joint NLL rose 0.274
    entirely on a family-head term of +0.317 while its identity term FELL
    0.043. Gating on the joint called that a capability collapse. It is the
    model reallocating probability across operators, which the reference-law
    metric already judges.
    """

    shares = {"f|real": 1.0}
    start = summarize_panel(
        [_row("f", "c", 4.0, identity=2.0, tag=str(i)) for i in range(4)],
        reference_law_stratum_share=shares)
    baseline = capability_baseline(start)
    # Joint up 1.0 nat, all of it in the family head; identity slightly better.
    reallocated = summarize_panel(
        [_row("f", "c", 5.0, identity=1.95, tag=str(i)) for i in range(4)],
        reference_law_stratum_share=shares)
    assert reallocated["by_family"]["f"]["mean_nll"] - start["by_family"]["f"]["mean_nll"] == 1.0
    assert regressed_capabilities(reallocated, baseline=baseline) == ()
    assert collapsed_capabilities(
        reallocated, baseline=baseline,
        previously_regressed=("family:f", "cell:c")) == ()


def test_a_better_matched_nll_wins_over_a_better_capability_floor() -> None:
    """The gate must not become a second objective.

    A: slightly better minimum capability probability, substantially worse
       reference-law-weighted NLL.
    B: clears every capability gate, much better weighted NLL.

    B must win. An earlier implementation led lexicographically on the
    capability floor itself, so A would have taken it on a 0.02-nat edge in the
    weakest cell while giving up 1.3 nats on the number that matters.
    """

    shares = {"a|real": 0.5, "b|real": 0.5}
    start = summarize_panel(
        [_row("a", "c1", 2.50, tag=str(i)) for i in range(4)]
        + [_row("b", "c2", 2.50, tag=str(i)) for i in range(4)],
        reference_law_stratum_share=shares)
    baseline = capability_baseline(start)

    # A is uniformly mediocre, so its WEAKEST capability is comparatively
    # strong. B is excellent on one and merely holds baseline on the other, so
    # its weakest is slightly weaker -- while its mean is far better.
    checkpoint_a = summarize_panel(
        [_row("a", "c1", 2.40, tag=str(i)) for i in range(4)]
        + [_row("b", "c2", 2.40, tag=str(i)) for i in range(4)],
        reference_law_stratum_share=shares)
    checkpoint_b = summarize_panel(
        [_row("a", "c1", 0.20, tag=str(i)) for i in range(4)]
        + [_row("b", "c2", 2.50, tag=str(i)) for i in range(4)],
        reference_law_stratum_share=shares)

    # A really does hold the better floor: its weakest capability is stronger.
    assert capability_floor(checkpoint_a) > capability_floor(checkpoint_b)
    # B really is substantially better on the number that decides.
    assert (checkpoint_b["reference_law_weighted_mean_nll"]
            < checkpoint_a["reference_law_weighted_mean_nll"] - 1.0)
    # Neither has collapsed, so both compete and B wins on matched NLL.
    assert collapsed_capabilities(checkpoint_a, baseline=baseline) == ()
    assert collapsed_capabilities(checkpoint_b, baseline=baseline) == ()
    assert selection_criterion(checkpoint_b, step=2, baseline=baseline) > \
        selection_criterion(checkpoint_a, step=1, baseline=baseline)


def test_ties_break_toward_the_earlier_step() -> None:
    """Indistinguishable on the reserve: keep the one that trained less."""

    shares = {"f|real": 1.0}
    early = summarize_panel([_row("f", "c", 2.0)], reference_law_stratum_share=shares)
    late = summarize_panel([_row("f", "c", 2.0)], reference_law_stratum_share=shares)
    assert selection_criterion(early, step=100) > selection_criterion(late, step=900)


def test_thin_cells_never_gate() -> None:
    """A mean over a handful of entries would fire or not fire at random."""

    shares = {"f|real": 1.0}
    start = summarize_panel(
        [_row("f", "thin", 2.0, tag=str(i)) for i in range(2)]
        + [_row("f", "fat", 2.0, tag=str(i)) for i in range(THIN_CELL_ENTRIES + 5)],
        reference_law_stratum_share=shares)
    baseline = capability_baseline(start)
    assert "cell:thin" not in baseline

    # The thin cell falls apart; the gate stays silent about it.
    worse = summarize_panel(
        [_row("f", "thin", 9.0, tag=str(i)) for i in range(2)]
        + [_row("f", "fat", 2.0, tag=str(i)) for i in range(THIN_CELL_ENTRIES + 5)],
        reference_law_stratum_share=shares)
    assert "cell:thin" not in collapsed_capabilities(worse, baseline=baseline)


def test_selection_refuses_a_population_the_law_does_not_describe() -> None:
    """Falling back to the panel-native mean would select on the wrong thing.

    The reserve is stratified to the LIBRARY; the model is trained under the
    LAW. Selecting without the weights would silently optimize for a different
    population while every count still looked right.
    """

    summary = summarize_panel([_row("f", "c", 2.0)])
    assert summary["reference_law_weighted_mean_nll"] is None
    with pytest.raises(RThetaTrainingError, match="reference_law_weighted_mean_nll"):
        selection_criterion(summary, step=1)


def test_zero_mass_strata_are_reported_but_never_weighted() -> None:
    """Compiled-but-undrawn chemistry is a diagnostic, not a vote.

    MEASURED: the law gives atom_delete|synthetic exactly zero draw because
    that family's real supply already meets its target. Scoring it would select
    for imitating a teacher policy we deliberately chose not to train.
    """

    rows = ([_row("atom_delete", "c1", 1.0, lane="real", tag=str(i)) for i in range(4)]
            + [_row("atom_delete", "c1", 9.0, lane="synthetic", tag=str(i))
               for i in range(4)])
    summary = summarize_panel(
        rows, reference_law_stratum_share={"atom_delete|real": 0.17,
                                           "atom_delete|synthetic": 0.0})
    # The catastrophic synthetic rows are measured...
    assert summary["zero_mass_stratum"]["atom_delete|synthetic"]["mean_nll"] == 9.0
    # ...and contribute nothing to the number that selects.
    assert summary["reference_law_weighted_mean_nll"] == pytest.approx(1.0)
    assert summary["panel_native"]["mean_nll"] == pytest.approx(5.0)


def test_weights_renormalize_over_strata_the_reserve_can_measure() -> None:
    """Dropping an unmeasurable stratum must reweight, not shrink.

    MEASURED: cycle_insert|real carries 2.2e-05 of the law over a supply of one
    row, which sits on the training side, so the reserve cannot measure it.
    Weighting without renormalizing would drag the mean toward zero.
    """

    summary = summarize_panel(
        [_row("a", "c", 2.0, tag="0"), _row("b", "c", 4.0, tag="1")],
        reference_law_stratum_share={"a|real": 0.25, "b|real": 0.25,
                                     "absent|real": 0.5})
    assert summary["reference_law_weighted_mean_nll"] == pytest.approx(3.0)
    assert summary["reference_law_mass_measured"] == pytest.approx(1.0)


def test_support_bands_report_in_regime_order() -> None:
    """The generalization regimes are a view, not a selection input."""

    rows = ([_row("f", "c", 1.0, support_band="25+", tag=str(i)) for i in range(3)]
            + [_row("f", "c", 3.0, support_band="0", tag=str(i)) for i in range(3)])
    summary = summarize_panel(rows, reference_law_stratum_share={"f|real": 1.0})
    assert list(summary["by_support_band"]) == ["0", "25+"]
    assert summary["by_support_band"]["0"]["mean_nll"] == 3.0
    assert summary["by_support_band"]["25+"]["mean_nll"] == 1.0


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


# --- code-commit binding -------------------------------------------------
#
# A paper-bearing run must bind the code that produced it. Three seeds that
# differ only in seed can only be shown to differ only in seed if the commit
# is recorded, and a commit that does not describe the running bytes is worse
# than none: it makes divergent runs look identically provenanced.


def _repo(tmp_path: Path, dirty: bool = False) -> Path:
    import subprocess
    root = tmp_path / "repo"
    root.mkdir()
    run = lambda *a: subprocess.run(["git", "-C", str(root), *a], check=True,
                                    capture_output=True)
    run("init", "-q")
    run("config", "user.email", "t@t")
    run("config", "user.name", "t")
    (root / "f.txt").write_text("one")
    run("add", "-A")
    run("commit", "-qm", "one")
    if dirty:
        (root / "f.txt").write_text("two")
    return root


def test_fresh_run_records_the_current_commit(tmp_path: Path) -> None:
    import subprocess
    root = _repo(tmp_path)
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert resolve_launch_commit(root) == head
    assert len(head) == 40


def test_dirty_worktree_refuses_launch(tmp_path: Path) -> None:
    root = _repo(tmp_path, dirty=True)
    with pytest.raises(RThetaTrainingError, match="dirty worktree"):
        resolve_launch_commit(root)


def _write(path: Path, identity: RunIdentity):
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return write_checkpoint(
        path, model=model, optimizer=optimizer, identity=identity,
        completed_steps=1, selected_step=1, selected_criterion=None,
        selected_state=None, trajectory=[], stream_sha256=SHA, resume_count=0)


def test_resume_from_the_same_commit_passes(tmp_path: Path) -> None:
    identity = _identity()
    _write(tmp_path / "ck.pt", identity)
    payload = load_checkpoint(
        tmp_path / "ck.pt", model=_model(),
        optimizer=torch.optim.AdamW(_model().parameters(), lr=1e-3),
        identity=identity, expected_stream_sha256=SHA)
    assert payload["identity"]["git_commit_sha"] == identity.git_commit_sha


def test_resume_after_a_code_change_is_refused(tmp_path: Path) -> None:
    _write(tmp_path / "ck.pt", _identity())
    with pytest.raises(RThetaTrainingError, match="refusing to resume across a code change"):
        load_checkpoint(
            tmp_path / "ck.pt", model=_model(),
            optimizer=torch.optim.AdamW(_model().parameters(), lr=1e-3),
            identity=_identity(git_commit_sha="f" * 40),
            expected_stream_sha256=SHA)


def test_checkpoint_carries_the_identity_forward(tmp_path: Path) -> None:
    """A result artifact must be attributable to the run that produced it."""

    identity = _identity()
    _write(tmp_path / "ck.pt", identity)
    payload = load_checkpoint(
        tmp_path / "ck.pt", model=_model(),
        optimizer=torch.optim.AdamW(_model().parameters(), lr=1e-3),
        identity=identity, expected_stream_sha256=SHA)
    assert payload["identity"] == identity.payload()
    assert payload["identity_sha256"] == identity.sha256()
    assert payload["schema_version"] == 2


def test_legacy_development_runs_are_readable_but_not_paper_bearing() -> None:
    """The step-12,500 development checkpoint did its job. It does not get
    retrofitted into the new contract and called paper-bearing."""

    legacy = _identity(git_commit_sha=None)
    assert legacy.paper_bearing is False
    with pytest.raises(RThetaTrainingError, match="LEGACY development run"):
        legacy.require_paper_bearing()
    # ...and a bound run is eligible.
    current = _identity()
    assert current.paper_bearing is True
    current.require_paper_bearing()
    # The two are different experiments by digest.
    assert legacy.sha256() != current.sha256()
