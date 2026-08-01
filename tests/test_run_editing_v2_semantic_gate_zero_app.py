"""Focused contract tests for the semantic Gate 0 Modal surface."""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import patch

import pytest

from modal_apps import run_editing_v2_semantic_gate_zero_app as gate_app

SHA = "a" * 64


def _source_revision() -> dict[str, object]:
    hashes = {"source.py": SHA}
    body: dict[str, object] = {
        "schema": gate_app.SOURCE_REVISION_SCHEMA,
        "schema_version": gate_app.SOURCE_REVISION_SCHEMA_VERSION,
        "commit": "1" * 40,
        "tree": "2" * 40,
        "worktree_clean": True,
        "serialized_source_hashes": hashes,
        "serialized_source_hashes_sha256": gate_app._canonical_sha256(hashes),
    }
    return {**body, "source_revision_sha256": gate_app._canonical_sha256(body)}


def _inputs() -> dict[str, dict[str, str]]:
    return {
        name: {
            "artifact_path": f"/artifacts/gate-zero/{name}.json",
            "file_sha256": chr(ord("a") + index) * 64,
        }
        for index, name in enumerate(gate_app._INPUT_NAMES)
    }


def test_request_is_deterministic_exact_and_nonauthorizing() -> None:
    first = gate_app.build_run_request(
        source_revision=_source_revision(),
        input_records=_inputs(),
        contract_file_sha256="f" * 64,
        contract_sha256="9" * 64,
    )
    second = gate_app.build_run_request(
        source_revision=_source_revision(),
        input_records=_inputs(),
        contract_file_sha256="f" * 64,
        contract_sha256="9" * 64,
    )
    assert first == second
    assert first["run_identity_sha256"] == gate_app._canonical_sha256(
        {key: value for key, value in first.items() if key != "run_identity_sha256"}
    )
    assert all(first[field] is False for field in gate_app._NO_AUTHORITY)
    assert list(first["inputs"]) == list(gate_app._INPUT_NAMES)


def test_request_rejects_missing_input_and_escaping_path() -> None:
    inputs = _inputs()
    inputs.pop("decision_completion")
    with pytest.raises(ValueError, match="inventory"):
        gate_app.build_run_request(
            source_revision=_source_revision(),
            input_records=inputs,
            contract_file_sha256="f" * 64,
            contract_sha256="9" * 64,
        )
    inputs = _inputs()
    inputs["decision_completion"]["artifact_path"] = "/tmp/result.json"
    with pytest.raises(ValueError, match="below /artifacts"):
        gate_app.build_run_request(
            source_revision=_source_revision(),
            input_records=inputs,
            contract_file_sha256="f" * 64,
            contract_sha256="9" * 64,
        )


def test_local_source_revision_requires_exact_clean_commit() -> None:
    with (
        patch.object(gate_app, "_git", side_effect=["1" * 40, "2" * 40, ""]),
        patch.object(
            gate_app, "_serialized_source_hashes", return_value={"source.py": SHA}
        ),
    ):
        revision = gate_app.local_source_revision(
            expected_commit="1" * 40, repo_root=Path("/fixture")
        )
    assert revision["commit"] == "1" * 40

    with (
        patch.object(
            gate_app,
            "_git",
            side_effect=["1" * 40, "2" * 40, "?? dirty.py"],
        ),
        pytest.raises(RuntimeError, match="clean committed"),
    ):
        gate_app.local_source_revision(
            expected_commit="1" * 40, repo_root=Path("/fixture")
        )


def test_modal_surface_is_cpu_only_and_never_launches_training() -> None:
    source = Path(gate_app.__file__).read_text()
    tree = ast.parse(source)
    remote = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_gate_zero"
    )
    decorator = ast.get_source_segment(source, remote.decorator_list[0])
    assert decorator is not None
    assert "gpu=" not in decorator
    assert "cpu=8.0" in decorator
    assert "run_semantic_gate_zero_structural_evidence" in source
    assert '"training_launched": False' in source
