"""Bounded QED test-panel orchestration and fail-closed resume checks."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from compose_v4.experiments.qed_shared_smc import QEDSMCConfig
from tools import run_qed_shared_panel as panel

CONFIG = QEDSMCConfig(horizon=2, particles=3, candidates=1)


def _record(index: int, source: str) -> dict:
    return {
        "schema_version": "compose.qed.shared_result.v1",
        "test_index": index,
        "source_split_sha256": "split",
        "reference_manifest_sha256": "reference-manifest",
        "value_assets_manifest_sha256": "value-assets",
        "value_metadata_sha256": "metadata",
        "code_sha256": {"runner": "code"},
        "result": {
            "source_original": source,
            "value_head_source_split_sha256": "split",
            "reference": {"checkpoint_sha256": "checkpoint", "time": 0.5},
            "configuration": {
                "horizon": 2,
                "particles": 3,
                "candidates": 1,
                "qed_minimum": 0.9,
                "similarity_minimum": 0.4,
            },
            "candidates": [{}],
        },
    }


def _validate(path: Path, index: int, source: str) -> None:
    panel.validate_existing(
        path,
        index,
        source,
        split_sha256="split",
        reference_manifest_sha256="reference-manifest",
        value_assets_manifest_sha256="value-assets",
        value_metadata_sha256="metadata",
        checkpoint_sha256="checkpoint",
        time=0.5,
        code_sha256={"runner": "code"},
        config=CONFIG,
    )


def test_source_command_binds_protocol_and_distinct_output(tmp_path: Path) -> None:
    output = tmp_path / "source_0012.json"
    command = panel.source_command(
        12,
        output,
        value=tmp_path / "value",
        checkpoint=tmp_path / "reference.pt",
        time=0.5,
        config=CONFIG,
    )
    assert command[command.index("--index") + 1] == "12"
    assert command[command.index("--value") + 1] == str(tmp_path / "value")
    assert command[command.index("--horizon") + 1] == "2"
    assert command[command.index("--particles") + 1] == "3"
    assert command[command.index("--candidates") + 1] == "1"
    assert command[command.index("--output") + 1] == str(output)


def test_resume_rejects_changed_value_or_source_before_running(tmp_path: Path) -> None:
    path = tmp_path / "source_0000.json"
    record = _record(0, "CCO")
    path.write_text(json.dumps(record))
    _validate(path, 0, "CCO")
    with pytest.raises(ValueError, match="incompatible source"):
        _validate(path, 0, "CCN")
    record["value_metadata_sha256"] = "another-head"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="incompatible value_metadata_sha256"):
        _validate(path, 0, "CCO")


def test_panel_resumes_only_matching_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "panel"
    value = tmp_path / "value"
    value.mkdir()
    (value / "metadata.json").write_text(json.dumps({"reference": {"time": 0.5}, "budget_max": 2}))
    checkpoint = tmp_path / "reference.pt"
    checkpoint.write_text("checkpoint-bytes")
    monkeypatch.setattr(
        panel,
        "verify_assets",
        lambda *_args: {
            "reference_manifest_sha256": "reference-manifest",
            "value_assets": {"manifest_sha256": "value-assets"},
            "shared_reference_checkpoint": {"sha256": "checkpoint"},
        },
    )
    monkeypatch.setattr(
        panel,
        "load_qed_source_roles",
        lambda *_args: SimpleNamespace(test=("CCO", "CCN"), manifest_sha256="split"),
    )
    monkeypatch.setattr(panel, "_expected_code_hashes", lambda: {"runner": "code"})
    first = output / "source_0000.json"
    first.parent.mkdir()
    record = _record(0, "CCO")
    record["value_metadata_sha256"] = panel.sha256(value / "metadata.json")
    first.write_text(json.dumps(record))
    calls: list[list[str]] = []

    def fake_run(command: list[str], target: Path) -> None:
        calls.append(command)
        target.write_text(json.dumps(_record(1, "CCN")))

    monkeypatch.setattr(panel, "_run", fake_run)
    with pytest.raises(FileExistsError, match="use --resume"):
        panel.run_panel(
            output=output,
            value=value,
            checkpoint=checkpoint,
            time=0.5,
            config=CONFIG,
            workers=1,
            resume=False,
        )
    assert panel.run_panel(
        output=output,
        value=value,
        checkpoint=checkpoint,
        time=0.5,
        config=CONFIG,
        workers=1,
        resume=True,
    ) == (1, 1)
    assert len(calls) == 1
    assert calls[0][calls[0].index("--index") + 1] == "1"


def test_panel_rejects_extra_json_and_invalid_workers_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="workers"):
        panel.run_panel(
            output=tmp_path,
            value=tmp_path,
            checkpoint=tmp_path / "reference.pt",
            time=0.5,
            config=CONFIG,
            workers=0,
            resume=True,
        )
    monkeypatch.setattr(panel, "verify_assets", lambda *_args: {})
    monkeypatch.setattr(
        panel,
        "load_qed_source_roles",
        lambda *_args: SimpleNamespace(test=("CCO",), manifest_sha256="split"),
    )
    (tmp_path / "metadata.json").write_text(
        json.dumps({"reference": {"time": 0.5}, "budget_max": 2})
    )
    (tmp_path / "another.json").write_text("{}")
    with pytest.raises(ValueError, match="unexpected JSON"):
        panel.run_panel(
            output=tmp_path,
            value=tmp_path,
            checkpoint=tmp_path / "reference.pt",
            time=0.5,
            config=CONFIG,
            workers=1,
            resume=True,
        )
