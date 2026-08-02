"""Focused contracts for the semantic T1 panel/cache-input Modal surface."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from modal_apps import materialize_editing_v2_semantic_t1_panel_cache_app as t1_app

SHA = "a" * 64


def _source_revision() -> dict[str, object]:
    hashes = {"source.py": SHA}
    body: dict[str, object] = {
        "schema": t1_app.SOURCE_REVISION_SCHEMA,
        "schema_version": t1_app.SOURCE_REVISION_SCHEMA_VERSION,
        "commit": "1" * 40,
        "tree": "2" * 40,
        "worktree_clean": True,
        "serialized_source_hashes": hashes,
        "serialized_source_hashes_sha256": t1_app._sha256(hashes),
    }
    return {**body, "source_revision_sha256": t1_app._sha256(body)}


def _inputs() -> dict[str, dict[str, str]]:
    return {
        name: {
            "artifact_path": f"/artifacts/semantic-t1/{name}.json",
            "file_sha256": format(index + 1, "x") * 64,
        }
        for index, name in enumerate(t1_app._INPUT_NAMES)
    }


def _panel_request() -> dict[str, object]:
    return {
        "schema": "compose.editing_v2.semantic_t1_panel_request",
        "schema_version": 3,
        "status": "BOUNDED_REQUEST_NO_T1_OR_TRAINING_AUTHORITY",
        **t1_app._NO_AUTHORITY,
        "request_id": "fixture",
        "source_revision_sha256": "b" * 64,
        "support_time_hex": "0x1.0000000000000p-1",
        "cell_role_policy_sha256": "e" * 64,
        "minimum_entries_by_family": {"fixture": 1},
        "maximum_entries_by_family": {"fixture": 1},
        "selection_rule": "fixture",
        "selection_seed_sha256": "c" * 64,
        "unique_objective_unit": "fixture",
        "equal_objective_coefficient": 1,
        "hazard_included": False,
        "candidate_size_is_audit_stratum_not_balancing_cell": True,
        "request_sha256": "d" * 64,
    }


def _policy() -> dict[str, object]:
    body: dict[str, object] = {
        "schema": t1_app.PANEL_POLICY_SCHEMA,
        "schema_version": t1_app.PANEL_POLICY_SCHEMA_VERSION,
        "status": t1_app.PANEL_POLICY_STATUS,
        **t1_app._NO_AUTHORITY,
        "request_id": "editing-v2-semantic-t1-capacity-v1",
        "support_time_hex": "0x1.0000000000000p-1",
        "cell_role_policy_sha256": (
            "a26c7084b2cf6f23a7690f5dc5f76f992056b205c0dc189405643a22a90b603c"
        ),
        "minimum_entries_by_family": {
            "atom_delete": 64,
            "atom_insert": 64,
            "atom_restate": 64,
            "bond_reorder": 64,
            "bond_reroute": 64,
            "cycle_attach": 64,
            "cycle_insert": 64,
            "ring_system_restate": 64,
        },
        "maximum_entries_by_family": {
            "atom_delete": 128,
            "atom_insert": 128,
            "atom_restate": 128,
            "bond_reorder": 128,
            "bond_reroute": 128,
            "cycle_attach": 128,
            "cycle_insert": 128,
            "ring_system_restate": 128,
        },
        "panel_kind": t1_app.PANEL_POLICY_PANEL_KIND,
        "objective_unit": t1_app.PANEL_POLICY_OBJECTIVE_UNIT,
        "cache_handoff": t1_app.PANEL_POLICY_CACHE_HANDOFF,
        "repeated_state_panel_included": False,
        "empirical_multiplicity_receipts_included": False,
        "successor_fiber_cache_compiled": False,
        "hazard_included": False,
        "optimizer_policy_included": False,
        "gate_thresholds_included": False,
        "p50_policy_included": False,
    }
    return {**body, "policy_sha256": t1_app._sha256(body)}


def test_run_request_is_exact_deterministic_and_nonauthorizing() -> None:
    arguments = {
        "source_revision": _source_revision(),
        "input_records": _inputs(),
        "panel_request": _panel_request(),
        "panel_policy_file_sha256": "e" * 64,
        "panel_policy_sha256": "f" * 64,
        "panel_implementation_sha256": "9" * 64,
    }
    first = t1_app.build_run_request(**arguments)
    second = t1_app.build_run_request(**arguments)
    assert first == second
    assert first["run_identity_sha256"] == t1_app._sha256(
        {key: value for key, value in first.items() if key != "run_identity_sha256"}
    )
    assert all(first[field] is False for field in t1_app._NO_AUTHORITY)
    assert first["training_launched"] is False
    assert first["successor_cache_compiled"] is False
    assert list(first["inputs"]) == list(t1_app._INPUT_NAMES)
    assert first["panel_policy"] == {
        "source": t1_app.PANEL_POLICY_SOURCE,
        "file_sha256": "e" * 64,
        "policy_sha256": "f" * 64,
    }


def test_run_request_rejects_missing_input_escape_and_authority() -> None:
    inputs = _inputs()
    inputs.pop("gate_zero_completion")
    with pytest.raises(ValueError, match="inventory"):
        t1_app.build_run_request(
            source_revision=_source_revision(),
            input_records=inputs,
            panel_request=_panel_request(),
            panel_policy_file_sha256="e" * 64,
            panel_policy_sha256="f" * 64,
            panel_implementation_sha256="9" * 64,
        )
    inputs = _inputs()
    inputs["decision_plan"]["artifact_path"] = "/tmp/decision.json"
    with pytest.raises(ValueError, match="below /artifacts"):
        t1_app.build_run_request(
            source_revision=_source_revision(),
            input_records=inputs,
            panel_request=_panel_request(),
            panel_policy_file_sha256="e" * 64,
            panel_policy_sha256="f" * 64,
            panel_implementation_sha256="9" * 64,
        )
    request = _panel_request()
    request["t1_authorized"] = True
    with pytest.raises(ValueError, match="forbidden authority"):
        t1_app.build_run_request(
            source_revision=_source_revision(),
            input_records=_inputs(),
            panel_request=request,
            panel_policy_file_sha256="e" * 64,
            panel_policy_sha256="f" * 64,
            panel_implementation_sha256="9" * 64,
        )


def test_committed_panel_policy_is_self_hashed_and_excludes_later_policy(
    tmp_path: Path,
) -> None:
    policy_path = tmp_path / t1_app.PANEL_POLICY_SOURCE
    policy_path.parent.mkdir(parents=True)
    policy_path.write_text(json.dumps(_policy()))
    assert t1_app._load_panel_policy(root=tmp_path) == _policy()

    invalid = _policy()
    invalid["optimizer_policy_included"] = True
    body = {key: value for key, value in invalid.items() if key != "policy_sha256"}
    invalid["policy_sha256"] = t1_app._sha256(body)
    policy_path.write_text(json.dumps(invalid))
    with pytest.raises(RuntimeError, match="identity disagrees"):
        t1_app._load_panel_policy(root=tmp_path)

    invalid = _policy()
    invalid["minimum_entries_by_family"]["atom_insert"] = 63
    body = {key: value for key, value in invalid.items() if key != "policy_sha256"}
    invalid["policy_sha256"] = t1_app._sha256(body)
    policy_path.write_text(json.dumps(invalid))
    with pytest.raises(RuntimeError, match="identity disagrees"):
        t1_app._load_panel_policy(root=tmp_path)


def test_repository_policy_exists_and_freezes_unique_and_repeated_roles() -> None:
    policy = t1_app._load_panel_policy(root=t1_app.ROOT)
    assert policy["panel_kind"] == t1_app.PANEL_POLICY_PANEL_KIND
    assert policy["objective_unit"] == t1_app.PANEL_POLICY_OBJECTIVE_UNIT
    assert policy["cache_handoff"] == t1_app.PANEL_POLICY_CACHE_HANDOFF
    assert policy["repeated_state_panel_included"] is False
    assert policy["empirical_multiplicity_receipts_included"] is False
    assert policy["successor_fiber_cache_compiled"] is False


def test_local_source_revision_requires_the_exact_clean_commit() -> None:
    with (
        patch.object(t1_app, "_git", side_effect=["1" * 40, "2" * 40, ""]),
        patch.object(t1_app, "_serialized_source_hashes", return_value={"source.py": SHA}),
    ):
        revision = t1_app.local_source_revision(
            expected_commit="1" * 40, repo_root=Path("/fixture")
        )
    assert revision["commit"] == "1" * 40

    with (
        patch.object(
            t1_app,
            "_git",
            side_effect=["1" * 40, "2" * 40, "?? dirty.py"],
        ),
        pytest.raises(RuntimeError, match="clean committed"),
    ):
        t1_app.local_source_revision(expected_commit="1" * 40, repo_root=Path("/fixture"))


def test_gate_zero_inputs_must_be_exact_siblings(tmp_path: Path) -> None:
    gate_zero = SimpleNamespace(
        EVIDENCE_FILENAME="STRUCTURAL_EVIDENCE.json",
        DECISION_FILENAME="STRUCTURAL_DECISION.json",
        COMPLETION_FILENAME="COMPLETE.json",
    )
    directory = tmp_path / "gate-zero"
    directory.mkdir()
    assert (
        t1_app._require_gate_zero_siblings(
            evidence=directory / gate_zero.EVIDENCE_FILENAME,
            decision=directory / gate_zero.DECISION_FILENAME,
            completion=directory / gate_zero.COMPLETION_FILENAME,
            gate_zero=gate_zero,
        )
        == directory
    )
    with pytest.raises(RuntimeError, match="exact three sibling"):
        t1_app._require_gate_zero_siblings(
            evidence=directory / gate_zero.EVIDENCE_FILENAME,
            decision=tmp_path / gate_zero.DECISION_FILENAME,
            completion=directory / gate_zero.COMPLETION_FILENAME,
            gate_zero=gate_zero,
        )


def test_completion_uses_canonical_artifact_addresses(tmp_path: Path) -> None:
    artifact_root = tmp_path / "mounted-artifacts"
    output = artifact_root / "editing_v2" / "semantic_t1" / "run"
    output.mkdir(parents=True)
    request_path = output / t1_app.RUN_REQUEST_FILENAME
    panel_path = output / t1_app.PANEL_FILENAME
    request_path.write_bytes(b"request\n")
    panel_path.write_bytes(b"panel\n")
    panel = SimpleNamespace(
        artifact_sha256="1" * 64,
        panel_implementation_sha256="2" * 64,
        gate_zero_binding=SimpleNamespace(
            decision_source_inventory_sha256="3" * 64,
            completion_sha256="4" * 64,
        ),
        identity_body=lambda: {"counts": {"panel_entry_count": 8}},
    )
    run_request = {
        "run_identity_sha256": "5" * 64,
        "source_revision": {"source_revision_sha256": "6" * 64},
        "inputs": {"fixture": "7" * 64},
        "panel_policy": {"policy_sha256": "8" * 64},
    }
    completion = t1_app._completion_body(
        run_request=run_request,
        request_path=request_path,
        panel_path=panel_path,
        panel=panel,
        panel_file_sha256=t1_app._file_sha256(panel_path),
        panel_file_bytes=panel_path.stat().st_size,
        resolved_cache_trace_count=8,
        artifact_root=artifact_root,
    )
    assert completion["request_artifact_path"].startswith("/artifacts/")
    assert completion["panel_artifact_path"].startswith("/artifacts/")
    assert str(tmp_path) not in completion["request_artifact_path"]
    assert completion["cache_trace_inputs_reopened"] is True
    assert completion["resolved_cache_trace_count"] == 8
    assert completion["unique_state_panel_prepared"] is True
    assert completion["repeated_state_panel_prepared"] is False
    assert completion["empirical_multiplicity_receipts_consumed"] is False


def test_modal_surface_is_cpu_only_and_has_no_runtime_policy_arguments() -> None:
    source = Path(t1_app.__file__).read_text()
    tree = ast.parse(source)
    remote = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "materialize_semantic_t1_panel_cache_inputs"
    )
    decorator = ast.get_source_segment(source, remote.decorator_list[0])
    assert decorator is not None
    assert "gpu=" not in decorator
    assert "cpu=8.0" in decorator
    remote_arguments = {argument.arg for argument in remote.args.kwonlyargs}
    assert "support_time_hex" not in remote_arguments
    assert "maximum_entries_by_family" not in remote_arguments
    assert "request_id" not in remote_arguments
    assert "resolve_editing_v2_semantic_active8_decision_source" in source
    assert "prepare_editing_v2_semantic_t1_panel" in source
    assert "build_semantic_t1_repeated_panel_from_occurrence_factory" not in source
    remote_source = ast.get_source_segment(source, remote)
    assert remote_source is not None
    assert remote_source.index("artifact_volume.reload()") < remote_source.index("_driver_impl(")
    assert remote_source.rindex("artifact_volume.commit()") < remote_source.rindex("return result")
    assert "stage_commit=artifact_volume.commit" in source
    assert source.count("stage_commit()") == 3
    assert 'loaded["write_bytes_if_absent"]' in source
    assert "t1_panel.write_semantic_t1_panel" in source
    assert '"training_launched": False' in source
    assert '"successor_cache_compiled": False' in source
