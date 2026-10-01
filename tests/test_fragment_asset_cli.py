"""The public asset check distinguishes missing LFS bytes from evaluator inputs."""

from __future__ import annotations

import hashlib
import json

import pytest

from compose_v4.experiments.fragments.__main__ import main


def _registry(root, checkpoint_bytes: bytes) -> None:
    config = root / "experiments" / "fragments"
    config.mkdir(parents=True)
    payload = {
        "schema": "compose_fragment_assets_v1",
        "assets": {
            "checkpoint": {
                "path": "checkpoint.pt",
                "sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
            },
            "evaluator_metrics": {
                "path": "evaluator/metrics.py",
                "sha256": hashlib.sha256(b"external evaluator").hexdigest(),
            },
        },
        "task_assets": {"demo": ["checkpoint"]},
    }
    (config / "assets.json").write_text(json.dumps(payload))


def test_asset_cli_can_check_generation_inputs_without_evaluator(tmp_path, capsys):
    _registry(tmp_path, b"model bytes")
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "checkpoint.pt").write_bytes(b"model bytes")

    assert (
        main(
            [
                "--root",
                str(tmp_path),
                "assets",
                "--assets",
                str(assets),
                "--task",
                "demo",
                "--no-evaluator",
            ]
        )
        == 0
    )
    assert "Verified 1 local assets" in capsys.readouterr().out


def test_asset_cli_identifies_lfs_pointer(tmp_path, capsys):
    _registry(tmp_path, b"model bytes")
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "checkpoint.pt").write_bytes(
        b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"0" * 64 + b"\nsize 11\n"
    )

    with pytest.raises(SystemExit) as exit_info:
        main(
            [
                "--root",
                str(tmp_path),
                "assets",
                "--assets",
                str(assets),
                "--task",
                "demo",
                "--no-evaluator",
            ]
        )
    assert exit_info.value.code == 1
    assert "Git LFS pointer for checkpoint" in capsys.readouterr().err
