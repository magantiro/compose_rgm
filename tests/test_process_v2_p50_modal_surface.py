"""Focused fail-closed tests for the thin Process-V2 P50 Modal launcher."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import modal_apps.run_process_v2_p50_app as launcher

ROOT = Path(__file__).resolve().parents[1]


def _source() -> str:
    return (ROOT / launcher.LAUNCHER_SOURCE).read_text()


def _function(name: str) -> ast.FunctionDef:
    tree = ast.parse(_source())
    return next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _clean_source_fixture(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "source"
    (root / "modal_apps").mkdir(parents=True)
    (root / "src").mkdir()
    (root / "configs").mkdir()
    (root / launcher.LAUNCHER_SOURCE).write_text("# launcher\n")
    for relative in launcher._RUNNER_IMPLEMENTATION_SOURCES[1:]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n")
    (root / "configs" / "policy.json").write_text("{}\n")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "p50@example.invalid")
    _git(root, "config", "user.name", "P50 Test")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "test: freeze p50 source")
    return root, _git(root, "rev-parse", "HEAD")


def test_launcher_is_the_exact_bounded_fast_path() -> None:
    source = _source()
    assert launcher.MAX_CPU_CONTAINERS == 80
    assert launcher.CPU_PER_LEAF == 1.0
    assert source.count('gpu="A10G"') == 1
    assert "prepare_leaf_remote.starmap(" in source
    assert "prepare_leaf_remote.update_autoscaler(max_containers=max_cpu_containers)" in source
    assert "build_process_v2_p50_selection" in source
    assert "compile_process_v2_p50_entries" in source
    assert "build_process_v2_p50_prepared_inputs" in source
    assert "run_process_v2_p50" in source


def test_checked_in_t1_evidence_manifest_is_canonical_newline_framed_json() -> None:
    path = ROOT / launcher.T1_EVIDENCE_SOURCE
    raw = path.read_bytes()
    value = launcher._read_canonical_object(
        path,
        label="the checked-in Process-V2 P50 T1 evidence source manifest",
    )
    assert raw == launcher._canonical_bytes(value) + b"\n"


def test_launcher_does_not_adapt_the_legacy_p50_pipeline() -> None:
    tree = ast.parse(_source())
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("editing_v2_semantic_p50" in name for name in imported)
    assert "run_editing_v2_semantic_p50_app" not in _source()


def test_gpu_has_no_selected_checkpoint_input_or_load_call() -> None:
    function = _function("run_p50_gpu_remote")
    parameter_names = [argument.arg for argument in function.args.args]
    body = ast.get_source_segment(_source(), function)
    assert body is not None
    assert parameter_names == [
        "prepared_path",
        "active8_run_root",
        "gate_zero_decision_path",
        "t1_result_path",
        "t1_decision_path",
        "output_prefix",
        "revision",
    ]
    assert "torch.load" not in body
    assert 'loaded["build_scratch"](source)' in body
    assert '"t1_selected_checkpoint_loaded": False' in body


def test_driver_orders_selection_fanout_reduction_and_gpu() -> None:
    body = ast.get_source_segment(_source(), _function("driver"))
    assert body is not None
    assert body.index("prepare_selection_remote.remote(") < body.index(
        "prepare_leaf_remote.starmap("
    )
    assert body.index("prepare_leaf_remote.starmap(") < body.index(
        "finalize_prepared_remote.remote("
    )
    assert body.index("finalize_prepared_remote.remote(") < body.index("run_p50_gpu_remote.remote(")
    assert body.count("prepare_leaf_remote.starmap(") == 1
    assert body.count("run_p50_gpu_remote.remote(") == 1


def test_scoped_t1_materialization_binds_the_current_authenticated_source() -> None:
    body = ast.get_source_segment(_source(), _function("materialize_scoped_t1_remote"))
    assert body is not None
    assert "current_process_identity_sha256=source.contracts.process_identity_sha256" in body
    assert "current_active8_completion_sha256=source.index.active8_completion_sha256" in body
    assert 'current_gate_zero_decision_sha256=source.decision["decision_sha256"]' in body


def test_cpu_map_sends_addresses_not_selection_payloads() -> None:
    body = ast.get_source_segment(_source(), _function("driver"))
    assert body is not None
    starmap = body[body.index("prepare_leaf_remote.starmap(") :]
    assert 'selected["selection_path"]' in starmap
    assert "selection," not in starmap
    assert "for wave_index" not in body


def test_decision_is_the_last_scientific_publication() -> None:
    body = ast.get_source_segment(_source(), _function("run_p50_gpu_remote"))
    assert body is not None
    checkpoint = body.index('loaded["write_bytes_if_absent"](checkpoint_path')
    result = body.index("_publish_canonical(result_path")
    decision = body.index("_publish_canonical(decision_path")
    commit = body.index("artifact_volume.commit()", decision)
    assert checkpoint < result < decision < commit
    assert "Decision publication is the atomic completion marker" in body


def test_gpu_sets_deterministic_cublas_before_model_materialization() -> None:
    source = _source()
    assert launcher.DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG == ":4096:8"
    assert '"CUBLAS_WORKSPACE_CONFIG": DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG' in source
    body = ast.get_source_segment(source, _function("run_p50_gpu_remote"))
    assert body is not None
    assert body.index('os.environ.get("CUBLAS_WORKSPACE_CONFIG")') < body.index(
        'scratch.model.to(device="cuda"'
    )


def test_checkpoint_contains_complete_resume_state_and_provenance() -> None:
    model_state = {"weight": torch.tensor([1.0])}
    optimizer_state = {"state": {}, "param_groups": []}
    run = {
        "optimizer_steps_completed": 50,
        "final_model_state": model_state,
        "final_model_state_sha256": "a" * 64,
        "optimizer_state": optimizer_state,
        "rng_state": {"torch_cpu": torch.get_rng_state()},
        "hazard_initial_state_sha256": "b" * 64,
        "hazard_final_state_sha256": "b" * 64,
    }
    loaded = {
        "authority_false_block": lambda: {
            "training_authorized": False,
            "bounded_p50_authorized": False,
            "p500_authorized": False,
            "p2000_authorized": False,
            "long_run_authorized": False,
            "checkpoint_selection_authorized": False,
            "final_test_selection_authorized": False,
        },
        "state_dict_sha256": lambda value: "a" * 64 if value is model_state else "x" * 64,
        "optimizer_state_sha256": lambda value: "c" * 64 if value is optimizer_state else "x" * 64,
    }
    payload = launcher._checkpoint_payload(
        run,
        policy={"contract_sha256": "d" * 64},
        provenance={"runner_source_revision_sha256": "e" * 64},
        loaded=loaded,
    )
    assert launcher._validate_checkpoint_payload(payload, loaded=loaded) == payload
    assert payload["completed_optimizer_steps"] == 50
    assert payload["resume_count"] == 0
    assert payload["model_state"] is model_state
    assert payload["optimizer_state"] is optimizer_state
    assert payload["rng_state"] == run["rng_state"]
    assert payload["configuration"]["contract_sha256"] == "d" * 64
    assert payload["provenance"]["runner_source_revision_sha256"] == "e" * 64


def test_checkpoint_validation_rejects_model_drift() -> None:
    loaded = {
        "authority_false_block": lambda: {
            "training_authorized": False,
            "bounded_p50_authorized": False,
            "p500_authorized": False,
            "p2000_authorized": False,
            "long_run_authorized": False,
            "checkpoint_selection_authorized": False,
            "final_test_selection_authorized": False,
        },
        "state_dict_sha256": lambda _value: "a" * 64,
        "optimizer_state_sha256": lambda _value: "c" * 64,
    }
    run = {
        "optimizer_steps_completed": 50,
        "final_model_state": {"weight": torch.tensor([1.0])},
        "final_model_state_sha256": "a" * 64,
        "optimizer_state": {"state": {}, "param_groups": []},
        "rng_state": {"torch_cpu": torch.get_rng_state()},
        "hazard_initial_state_sha256": "b" * 64,
        "hazard_final_state_sha256": "b" * 64,
    }
    payload = launcher._checkpoint_payload(run, policy={}, provenance={}, loaded=loaded)
    payload["model_state_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="checkpoint disagrees"):
        launcher._validate_checkpoint_payload(payload, loaded=loaded)


def test_leaf_validation_is_restart_safe_and_fail_closed() -> None:
    body = {
        "task_identity_sha256": "a" * 64,
        "selection_sha256": "b" * 64,
        "entry_count": 1,
        "entries": [{"p50_entry_sha256": "c" * 64}],
    }
    leaf = {**body, "leaf_sha256": launcher._sha256(body)}
    assert (
        launcher._validate_leaf(leaf, selection_sha256="b" * 64, task_identity_sha256="a" * 64)
        == leaf
    )
    leaf["entry_count"] = 2
    with pytest.raises(RuntimeError, match="self-hash disagrees"):
        launcher._validate_leaf(leaf, selection_sha256="b" * 64, task_identity_sha256="a" * 64)


def test_prepared_inputs_must_bind_the_fresh_prerequisite_chain() -> None:
    prerequisites = SimpleNamespace(
        binding_sha256="a" * 64,
        t1_initial_model_state_sha256="b" * 64,
    )
    prepared = {
        "prerequisites_binding_sha256": "a" * 64,
        "initial_model_state_sha256": "b" * 64,
    }
    launcher._require_prepared_prerequisite_binding(prepared, prerequisites=prerequisites)
    with pytest.raises(RuntimeError, match="prerequisite binding disagrees"):
        launcher._require_prepared_prerequisite_binding(
            {**prepared, "prerequisites_binding_sha256": "c" * 64},
            prerequisites=prerequisites,
        )
    with pytest.raises(RuntimeError, match="prerequisite binding disagrees"):
        launcher._require_prepared_prerequisite_binding(
            {**prepared, "initial_model_state_sha256": "d" * 64},
            prerequisites=prerequisites,
        )


def test_selection_cache_is_bound_to_the_fresh_prerequisite_chain() -> None:
    prerequisites = SimpleNamespace(
        binding_sha256="a" * 64,
        as_payload=lambda: {"process_identity_sha256": "b" * 64},
    )
    selection = {
        "prerequisites_binding_sha256": "a" * 64,
        "prerequisites": prerequisites.as_payload(),
    }
    launcher._require_selection_prerequisite_binding(selection, prerequisites=prerequisites)
    with pytest.raises(RuntimeError, match="selection prerequisite binding disagrees"):
        launcher._require_selection_prerequisite_binding(
            {**selection, "prerequisites_binding_sha256": "c" * 64},
            prerequisites=prerequisites,
        )


def test_selection_is_reopened_before_the_metadata_scan() -> None:
    body = ast.get_source_segment(_source(), _function("prepare_selection_remote"))
    assert body is not None
    assert "selection_cache_hit = selection_path.is_file()" in body
    assert body.index("if selection_cache_hit:") < body.index('loaded["build_selection"](')
    assert 'label="the cached Process-V2 P50 selection"' in body


def test_cpu_groups_are_complete_disjoint_and_bounded() -> None:
    values = tuple(f"{index:064x}" for index in range(93))
    groups = launcher._groups(values, maximum=40)
    assert [len(group) for group in groups] == [40, 40, 13]
    assert tuple(item for group in groups for item in group) == values
    assert len({item for group in groups for item in group}) == len(values)


@pytest.mark.parametrize("maximum", [0, 81, True])
def test_cpu_groups_refuse_invalid_bounds(maximum: int) -> None:
    with pytest.raises(ValueError, match="must lie"):
        launcher._groups(("a",), maximum=maximum)


def test_local_revision_binds_every_serialized_tracked_byte(tmp_path: Path) -> None:
    root, commit = _clean_source_fixture(tmp_path)
    revision = launcher.local_image_revision(expected_commit=commit, repo_root=root)
    assert revision["commit"] == commit
    assert set(revision["serialized_sources"]) == set(launcher._serialized_source_paths(root))
    assert len(launcher._runner_implementation_sha256(revision)) == 64


def test_local_revision_refuses_dirty_or_wrong_commit(tmp_path: Path) -> None:
    root, commit = _clean_source_fixture(tmp_path)
    (root / "configs" / "policy.json").write_text('{"changed":true}\n')
    with pytest.raises(RuntimeError, match="exact clean commit"):
        launcher.local_image_revision(expected_commit=commit, repo_root=root)
    (root / "configs" / "policy.json").write_text("{}\n")
    with pytest.raises(RuntimeError, match="exact clean commit"):
        launcher.local_image_revision(expected_commit="f" * 40, repo_root=root)


def test_runtime_import_surface_names_only_current_process_v2_modules() -> None:
    body = ast.get_source_segment(_source(), _function("_imports"))
    assert body is not None
    assert "editing_v2_process_v2_p50_prerequisites" in body
    assert "editing_v2_process_v2_p50_result" in body
    assert "editing_v2_process_v2_p50_runtime" in body
    assert "editing_v2_semantic_p50_runner" not in body


def test_local_entrypoint_is_disconnect_safe_by_default() -> None:
    body = ast.get_source_segment(_source(), _function("main"))
    assert body is not None
    assert "wait_for_completion: bool = False" in body
    assert "driver.spawn(*arguments)" in body
    assert '"p500_launched": False' in body
