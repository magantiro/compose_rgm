"""Focused fail-closed tests for the thin Process-V2 P50 Modal launcher."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import modal_apps.run_process_v2_p50_app as launcher
from compose_v4.data.editing_v2_process_v2_schema import authority_false_block

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
    assert launcher.MAX_CPU_CONTAINERS == 100
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
        "collated_completion_path",
        "active8_run_root",
        "gate_zero_decision_path",
        "t1_result_path",
        "t1_decision_path",
        "output_prefix",
        "revision",
    ]
    assert "torch.load" not in body
    assert 'loaded["materialize_batch"](' in body
    assert "materialized=materialized" in body
    assert 'loaded["build_scratch"](source)' in body
    assert '"t1_selected_checkpoint_loaded": False' in body


def test_split_drivers_order_preparation_before_explicit_gpu_training() -> None:
    prepare_body = ast.get_source_segment(_source(), _function("prepare_driver"))
    train_body = ast.get_source_segment(_source(), _function("train_driver"))
    combined_body = ast.get_source_segment(_source(), _function("driver"))
    assert prepare_body is not None
    assert train_body is not None
    assert combined_body is not None
    assert prepare_body.index("prepare_selection_remote.remote(") < prepare_body.index(
        "prepare_leaf_remote.starmap("
    )
    assert prepare_body.index("prepare_leaf_remote.starmap(") < prepare_body.index(
        "finalize_prepared_remote.remote("
    )
    assert "run_p50_gpu_remote" not in prepare_body
    assert train_body.count("run_p50_gpu_remote.remote(") == 1
    assert combined_body.index("prepare_driver.remote(") < combined_body.index(
        "collate_driver.remote("
    ) < combined_body.index("train_driver.remote(")


def test_canary_publishes_one_reusable_worst_case_leaf_without_gpu() -> None:
    body = ast.get_source_segment(_source(), _function("canary"))
    assert body is not None
    assert body.count("prepare_leaf_remote.remote(") == 1
    assert "task_entry_counts" in body
    assert "(-int(counts[identity]), identity)" in body
    assert "run_p50_gpu_remote" not in body
    assert '"p50_training_launched": False' in body


def test_scoped_t1_materialization_binds_the_current_authenticated_source() -> None:
    body = ast.get_source_segment(_source(), _function("materialize_scoped_t1_remote"))
    assert body is not None
    assert "current_process_identity_sha256=source.contracts.process_identity_sha256" in body
    assert "current_active8_completion_sha256=source.index.active8_completion_sha256" in body
    assert 'current_gate_zero_decision_sha256=source.decision["decision_sha256"]' in body


def test_cpu_map_sends_addresses_not_selection_payloads() -> None:
    body = ast.get_source_segment(_source(), _function("prepare_driver"))
    assert body is not None
    starmap = body[body.index("prepare_leaf_remote.starmap(") :]
    assert 'selected["selection_path"]' in starmap
    assert "selection," not in starmap
    assert "for wave_index" not in body


def test_legacy_leaf_migration_is_json_only_and_reuses_current_selection() -> None:
    body = ast.get_source_segment(_source(), _function("migrate_v1_leaves_remote"))
    assert body is not None
    assert 'loaded["convert_v1_leaf"](' in body
    assert "prepare_leaf_remote" not in body
    assert "compile_entries" not in body
    assert '"molecular_reenumeration_count": 0' in body


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


def test_collation_balances_measured_mark_work_without_changing_inventory() -> None:
    leaves = []
    for leaf_index, workloads in enumerate(((100, 1), (90,), (80,), (2, 2, 2))):
        entries = [
            {
                "p50_entry_sha256": f"{leaf_index * 10 + index:064x}",
                "raw_mark_count": workload,
            }
            for index, workload in enumerate(workloads)
        ]
        leaves.append(
            {
                "task_identity_sha256": f"{leaf_index + 100:064x}",
                "leaf_sha256": f"{leaf_index + 200:064x}",
                "entries": entries,
            }
        )
    tasks = launcher._balanced_collated_tasks(leaves, maximum=2)
    observed = [identifier for task in tasks for identifier in task["p50_entry_sha256s"]]
    expected = sorted(
        entry["p50_entry_sha256"] for leaf in leaves for entry in leaf["entries"]
    )
    assert sorted(observed) == expected
    assert len(observed) == len(set(observed))
    assert len(tasks) == 2
    assert max(task["raw_mark_count"] for task in tasks) <= 190


def test_collated_artifacts_use_the_live_process_v2_authority_vocabulary() -> None:
    current = authority_false_block()
    launcher._require_false_authority(
        current,
        label="a current Process-V2 collation artifact",
        authority_false=current,
    )
    stale = {
        "training_authorized": False,
        "bounded_p50_authorized": False,
        "p500_authorized": False,
        "p2000_authorized": False,
        "long_run_authorized": False,
        "checkpoint_selection_authorized": False,
        "final_test_selection_authorized": False,
    }
    with pytest.raises(RuntimeError, match="grants or omits authority"):
        launcher._require_false_authority(
            stale,
            label="a stale Process-V2 collation artifact",
            authority_false=current,
        )


def test_gpu_is_unreachable_until_cpu_collation_completion() -> None:
    body = ast.get_source_segment(_source(), _function("driver"))
    assert body is not None
    assert body.index("collate_driver.remote(") < body.index("train_driver.remote(")
    assert 'collated["completion"]["completion_path"]' in body


@pytest.mark.parametrize("maximum", [0, True])
def test_cpu_groups_refuse_invalid_bounds(maximum: int) -> None:
    with pytest.raises(ValueError, match="must lie"):
        launcher._groups(("a",), maximum=maximum)


def test_cpu_groups_refuse_more_than_the_declared_maximum() -> None:
    """Derived from the launcher's own constant, so raising the cap cannot
    silently leave this test asserting a stale number."""

    with pytest.raises(ValueError, match="must lie"):
        launcher._groups(("a",), maximum=launcher.MAX_CPU_CONTAINERS + 1)


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
    prepare_body = ast.get_source_segment(_source(), _function("prepare"))
    train_body = ast.get_source_segment(_source(), _function("train"))
    assert body is not None
    assert prepare_body is not None
    assert train_body is not None
    assert "wait_for_completion: bool = False" in body
    assert "driver.spawn(*arguments)" in body
    assert "prepare_driver.spawn(*arguments)" in prepare_body
    assert '"p50_training_launched": False' in prepare_body
    assert "train_driver.spawn(*arguments)" in train_body
    assert '"prepared_path": prepared_path' in train_body
    assert '"p500_launched": False' in body


def test_a_single_failed_slice_cannot_discard_the_whole_prep_fan_out() -> None:
    """One slice that times out must cost one slice, not the run.

    The default ``starmap`` raises on the first failure and abandons every slice
    still in flight -- which is what discarded an otherwise healthy fan-out when
    a single 800-entry slice hit the 2,700s worker timeout.  Each worker commits
    its own output on success, and publication is write-if-absent, so surviving
    the failure and re-running only the failures is both safe and cheap.
    """

    source = _source()
    assert "prep_slice_remote.starmap(work, return_exceptions=True)" in source
    assert "isinstance(result, Exception)" in source
    # The failures have to be REPORTED, or a partial run reads as a complete one.
    for field in ('"slices_planned"', '"slices_failed"', '"failures"'):
        assert field in source, field


def test_a_prep_slice_reports_the_setup_the_worker_timeout_also_pays_for() -> None:
    """Sizing must see every term inside the timeout, not just the compile.

    ``slice_size`` was set from a compile-only rate while the worker timeout
    also covered source open and scratch build, so the budget omitted a term
    and the slice died at the wall.
    """

    source = _source()
    assert '"setup_seconds": started - entered' in source
    assert '"worker_seconds": time.monotonic() - entered' in source
    assert '"worker_timeout_seconds": CPU_LEAF_TIMEOUT_SECONDS' in source


def test_the_prep_slice_default_leaves_headroom_under_the_worker_timeout() -> None:
    """800 entries per slice is the size that actually timed out."""

    import ast

    tree = ast.parse(_source())
    node = next(
        item
        for item in ast.walk(tree)
        if isinstance(item, ast.FunctionDef) and item.name == "prep_subset"
    )
    defaults = dict(
        zip(
            [argument.arg for argument in node.args.args][-len(node.args.defaults) :],
            [ast.literal_eval(value) for value in node.args.defaults],
            strict=True,
        )
    )
    default = defaults["slice_size"]
    assert default == 300
    # Even at 2x the 2,046 ms/entry measured rate the compile fits with room
    # left for setup inside the 45-minute wall.
    assert default * 2 * 2.046 < launcher.CPU_LEAF_TIMEOUT_SECONDS * 0.75


def test_a_prep_slice_reports_progress_before_it_does_any_work() -> None:
    """An operator must be able to tell "starting up" from "hung" in minutes.

    The compile only emits once its first sub-batch lands, and setup runs
    BEFORE that -- setup being exactly the unknown these runs measure. Without
    an entry-time signal a stuck worker and a slow one look identical until the
    45-minute wall.
    """

    source = _source()
    tree = ast.parse(source)
    node = next(
        item
        for item in ast.walk(tree)
        if isinstance(item, ast.FunctionDef) and item.name == "prep_slice_remote"
    )
    body = [stmt for stmt in node.body if not isinstance(stmt, ast.Expr)]
    # `entered` must be the first executable statement, or the shard reads and
    # the scratch build fall outside the measurement while staying inside the wall.
    first = body[0]
    assert isinstance(first, ast.Assign)
    assert first.targets[0].id == "entered"

    for phase in (
        "process_v2_prep_slice_entered",
        "process_v2_prep_slice_imports_ready",
        "process_v2_prep_slice_source_open",
        "process_v2_prep_slice_setup_complete",
    ):
        assert phase in source, phase
    assert '"setup_seconds": started - entered' in source
