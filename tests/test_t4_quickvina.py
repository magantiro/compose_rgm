"""Command and failure tests mock subprocesses. No docking program is executed."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from compose_v4.experiments import t4_quickvina as docking


@pytest.fixture
def config(tmp_path):
    paths = [tmp_path / name for name in ("obabel", "qvina", "receptor.pdbqt")]
    for path in paths:
        path.write_bytes(b"unit-test placeholder, not an executable or receptor")
        path.chmod(0o700)
    digest = hashlib.sha256(paths[0].read_bytes()).hexdigest()
    return docking.DockingConfig(
        paths[0],
        digest,
        paths[1],
        digest,
        paths[2],
        digest,
        (1.0, 2.0, 3.0),
        (10.0, 11.0, 12.0),
        42,
    )


def test_exact_commands_scores_and_retained_evidence(tmp_path, config, monkeypatch):
    calls = []

    def pretend(argv, **kwargs):
        calls.append((argv, kwargs))
        if argv[0] == str(config.qvina):
            Path(argv[argv.index("--out") + 1]).write_text("REMARK VINA RESULT: -8.2 0 0\n")
        return subprocess.CompletedProcess(argv, 0, stdout=b"test", stderr=b"")

    monkeypatch.setattr(docking.subprocess, "run", pretend)
    evaluator = docking.QuickVinaEvaluator(config, tmp_path / "poses")
    assert not calls
    assert evaluator("CC") == -8.2
    assert [row[1]["timeout"] for row in calls] == [120, 60, 300]
    assert calls[0][0][1:3] == ["-:CC", "--gen3D"]
    assert "--seed" not in calls[0][0]
    argv = calls[-1][0]
    for flag, expected in (
        ("--cpu", "1"),
        ("--num_modes", "10"),
        ("--exhaustiveness", "1"),
        ("--seed", "42"),
    ):
        assert argv[argv.index(flag) + 1] == expected
    assert "--center_z" in argv and "--size_z" in argv
    assert evaluator("CC") == -8.2
    assert len(list((tmp_path / "poses").iterdir())) == 2
    for folder in (tmp_path / "poses").iterdir():
        assert json.loads((folder / "input.json").read_text())["smiles"] == "CC"
        assert (folder / "stage_2.stdout").read_bytes() == b"test"


@pytest.mark.parametrize("failure", ["timeout", "nonzero", "nan", "missing_score"])
def test_docking_failures_are_not_scores(tmp_path, config, monkeypatch, failure):
    def fail(argv, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(argv, 120, output=b"interrupted", stderr=b"timeout")
        if failure == "nonzero":
            return subprocess.CompletedProcess(argv, 1, stdout=b"", stderr=b"failure")
        if argv[0] == str(config.qvina):
            Path(argv[argv.index("--out") + 1]).write_text(
                "REMARK VINA RESULT: nan 0 0\n" if failure == "nan" else "no score\n"
            )
        return subprocess.CompletedProcess(argv, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(docking.subprocess, "run", fail)
    with pytest.raises((RuntimeError, ValueError)):
        docking.QuickVinaEvaluator(config, tmp_path / "poses")("CC")
    assert not list((tmp_path / "poses").glob("*/score.json"))
    assert list((tmp_path / "poses").glob("*/input.json"))


def test_changed_assets_fail_before_subprocess(tmp_path, config, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("changed assets must not run")

    monkeypatch.setattr(docking.subprocess, "run", forbidden)
    evaluator = docking.QuickVinaEvaluator(config, tmp_path / "poses")
    config.receptor.write_text("changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        evaluator("CC")
    assert not (tmp_path / "poses").exists()


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"center": (1, 2)}, "coordinates"),
        ({"center": (1, 2, float("nan"))}, "coordinates"),
        ({"size": (1, -2, 3)}, "positive"),
        ({"cpu": True}, "cpu"),
        ({"seed": -1}, "seed"),
    ],
)
def test_docking_config_validates_box_and_compute(config, changes, message):
    with pytest.raises(ValueError, match=message):
        replace(config, **changes)
