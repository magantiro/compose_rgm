from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.build_editing_t1_p50_decision as decision_cli


def _runtime_contract():
    return SimpleNamespace(
        payload={
            "active8_inventory_manifest_file_sha256": "1" * 64,
            "active8_inventory_sha256": "2" * 64,
            "active8_effective_source_corpus_cache_sha256": "3" * 64,
            "active8_unified_packed_manifest_sha256": "4" * 64,
            "active8_support_contract_sha256": "5" * 64,
        }
    )


def _admission():
    return SimpleNamespace(
        manifest_file_sha256="1" * 64,
        inventory_sha256="2" * 64,
        effective_source_corpus_cache_sha256="3" * 64,
        unified_packed_manifest_sha256="4" * 64,
        support_contract_sha256="5" * 64,
    )


def _paths(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "decision"
    root.mkdir()
    paths = {
        "runtime": root / "runtime.json",
        "manifest": root / "manifest.json",
        "inventory": tmp_path / "active8.json",
        "gate_zero": tmp_path / "gate-zero.json",
        "output": root / "decision.json",
    }
    for name, path in paths.items():
        if name != "output":
            path.write_text("{}\n")
    return paths


def _install_valid_fakes(monkeypatch, captured: dict[str, object]) -> None:
    runtime = _runtime_contract()
    admission = _admission()
    monkeypatch.setattr(
        decision_cli,
        "load_editing_t1_runtime_contract",
        lambda _path: runtime,
    )

    def load_admission(path, **kwargs):
        captured["admission"] = {"path": path, **kwargs}
        return admission

    def validate_gate_zero(payload, **kwargs):
        captured["gate_zero"] = {"payload": payload, **kwargs}
        return payload

    def build_decision(**kwargs):
        captured["decision"] = kwargs
        return {
            "schema": "compose.editing.t1_successor_gate_decision",
            "status": "PASS_BOUNDED_P50_PREREQUISITE",
            "bounded_p50_authorized": True,
            "full_training_authorized": False,
            "decision_sha256": "6" * 64,
        }

    monkeypatch.setattr(decision_cli, "load_active8_trace_admission", load_admission)
    monkeypatch.setattr(
        decision_cli,
        "active8_t1_identity",
        lambda _admission: {
            field: runtime.payload[field] for field in decision_cli.ACTIVE8_T1_IDENTITY_FIELDS
        },
    )
    monkeypatch.setattr(
        decision_cli,
        "validate_gate_zero_structural_evidence",
        validate_gate_zero,
    )
    monkeypatch.setattr(decision_cli, "build_t1_p50_decision", build_decision)


def test_decision_cli_binds_all_physical_parents_and_writes_immutably(
    tmp_path,
    monkeypatch,
) -> None:
    paths = _paths(tmp_path)
    captured: dict[str, object] = {}
    _install_valid_fakes(monkeypatch, captured)

    decision = decision_cli.build_editing_t1_p50_decision_artifact(
        runtime_contract_path=paths["runtime"],
        arm_results_manifest_path=paths["manifest"],
        active8_inventory_path=paths["inventory"],
        active8_inventory_file_sha256="1" * 64,
        gate_zero_evidence_path=paths["gate_zero"],
        output_path=paths["output"],
    )

    assert json.loads(paths["output"].read_bytes()) == decision
    assert captured["admission"] == {
        "path": paths["inventory"].resolve(),
        "expected_manifest_file_sha256": "1" * 64,
        "expected_inventory_sha256": "2" * 64,
        "expected_effective_source_corpus_cache_sha256": "3" * 64,
        "expected_support_contract_sha256": "5" * 64,
    }
    decision_inputs = captured["decision"]
    assert captured["gate_zero"]["source_admission"] is decision_inputs["source_admission"]
    assert decision_inputs["runtime_contract_path"] == paths["runtime"].resolve()
    assert decision_inputs["arm_results_manifest_path"] == paths["manifest"].resolve()
    assert decision_inputs["decision_directory"] == paths["output"].resolve().parent
    assert decision_inputs["source_inventory_file_sha256"] == "1" * 64
    assert (
        decision_inputs["gate_zero_evidence_file_sha256"]
        == hashlib.sha256(paths["gate_zero"].read_bytes()).hexdigest()
    )

    repeated = decision_cli.build_editing_t1_p50_decision_artifact(
        runtime_contract_path=paths["runtime"],
        arm_results_manifest_path=paths["manifest"],
        active8_inventory_path=paths["inventory"],
        active8_inventory_file_sha256="1" * 64,
        gate_zero_evidence_path=paths["gate_zero"],
        output_path=paths["output"],
    )
    assert repeated == decision


def test_decision_cli_fails_closed_on_stale_active8_authority(
    tmp_path,
    monkeypatch,
) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(
        decision_cli,
        "load_editing_t1_runtime_contract",
        lambda _path: _runtime_contract(),
    )
    with pytest.raises(
        decision_cli.EditingT1DecisionCliError,
        match="disagrees with the frozen T1 V4 runtime authority",
    ):
        decision_cli.build_editing_t1_p50_decision_artifact(
            runtime_contract_path=paths["runtime"],
            arm_results_manifest_path=paths["manifest"],
            active8_inventory_path=paths["inventory"],
            active8_inventory_file_sha256="0" * 64,
            gate_zero_evidence_path=paths["gate_zero"],
            output_path=paths["output"],
        )
    assert not paths["output"].exists()


def test_decision_cli_rejects_invalid_gate_zero_without_building(
    tmp_path,
    monkeypatch,
) -> None:
    paths = _paths(tmp_path)
    _install_valid_fakes(monkeypatch, {})

    def reject_gate_zero(*_args, **_kwargs):
        raise decision_cli.EditingP50PrerequisiteError("stale Gate0")

    monkeypatch.setattr(
        decision_cli,
        "validate_gate_zero_structural_evidence",
        reject_gate_zero,
    )
    built = False

    def observe_build(**_kwargs):
        nonlocal built
        built = True

    monkeypatch.setattr(decision_cli, "build_t1_p50_decision", observe_build)
    with pytest.raises(
        decision_cli.EditingT1DecisionCliError,
        match="Gate0 structural evidence is invalid",
    ):
        decision_cli.build_editing_t1_p50_decision_artifact(
            runtime_contract_path=paths["runtime"],
            arm_results_manifest_path=paths["manifest"],
            active8_inventory_path=paths["inventory"],
            active8_inventory_file_sha256="1" * 64,
            gate_zero_evidence_path=paths["gate_zero"],
            output_path=paths["output"],
        )
    assert built is False
    assert not paths["output"].exists()


def test_decision_cli_never_overwrites_a_different_decision(
    tmp_path,
    monkeypatch,
) -> None:
    paths = _paths(tmp_path)
    _install_valid_fakes(monkeypatch, {})
    paths["output"].write_text('{"existing":true}\n')

    with pytest.raises(
        decision_cli.EditingT1DecisionCliError,
        match="immutable T1 decision artifact already differs",
    ):
        decision_cli.build_editing_t1_p50_decision_artifact(
            runtime_contract_path=paths["runtime"],
            arm_results_manifest_path=paths["manifest"],
            active8_inventory_path=paths["inventory"],
            active8_inventory_file_sha256="1" * 64,
            gate_zero_evidence_path=paths["gate_zero"],
            output_path=paths["output"],
        )
    assert paths["output"].read_text() == '{"existing":true}\n'


def test_decision_cli_help_names_every_required_provenance_input() -> None:
    help_text = decision_cli._parser().format_help()
    for option in (
        "--runtime-contract",
        "--arm-results-manifest",
        "--active8-inventory",
        "--active8-inventory-file-sha256",
        "--gate-zero-evidence",
        "--output",
    ):
        assert option in help_text
