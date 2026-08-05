"""Static and local gates for the thin Process-V2 T1 Modal launcher."""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import modal_apps.run_process_v2_t1_app as launcher

ROOT = Path(__file__).resolve().parents[1]


def _source() -> str:
    return (ROOT / launcher.LAUNCHER_SOURCE).read_text()


def _function(name: str) -> ast.FunctionDef:
    tree = ast.parse(_source())
    return next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_launcher_freezes_bounded_cpu_fanout_and_one_gpu() -> None:
    source = _source()
    assert launcher.MAX_CPU_CONTAINERS == 40
    assert launcher.CPU_PER_LEAF == 1.0
    assert "max_containers=MAX_CPU_CONTAINERS" in source
    assert source.count('gpu="A10G"') == 1
    assert source.count("run_t1_gpu_remote.remote(") == 2
    assert "fiber_recomputation_count=0" in source
    assert "prepare_leaf_remote.update_autoscaler(max_containers=max_cpu_containers)" in source
    assert "prepare_leaf_remote.starmap(" in source


def test_launcher_binds_the_sparse_seven_point_process_v2_policy() -> None:
    policy = json.loads((ROOT / launcher.CAPACITY_POLICY_SOURCE).read_bytes())
    assert policy["optimization"]["trajectory_evaluation"] == ("step_zero_and_report_points_only")
    assert policy["optimization"]["report_points"] == [1, 10, 50, 100, 250, 500]
    assert policy["optimization"]["maximum_optimizer_steps"] == 500
    assert policy["hazard_included"] is False


def test_runtime_loader_accepts_the_same_sparse_process_v2_policy() -> None:
    from compose_v4.experiments.editing_v2_process_v2_t1_runtime import (
        load_process_v2_t1_capacity_policy,
    )

    policy, file_sha256 = load_process_v2_t1_capacity_policy(
        ROOT / launcher.CAPACITY_POLICY_SOURCE,
        repo_root=ROOT,
    )

    assert policy["optimization"]["trajectory_evaluation"] == (
        "step_zero_and_report_points_only"
    )
    assert policy["policy_sha256"] == policy["contract_sha256"]
    assert len(file_sha256) == 64


def test_cpu_fanout_sends_the_plan_address_not_the_plan_payload() -> None:
    body = ast.get_source_segment(_source(), _function("driver"))
    assert body is not None
    starmap = body[body.index("prepare_leaf_remote.starmap(") :]
    assert 'prepared["plan_path"]' in starmap
    assert "(\n                        plan," not in starmap
    assert "for wave_index" not in body
    assert body.count("prepare_leaf_remote.starmap(") == 1


def test_driver_runs_prepared_publication_before_the_gpu() -> None:
    body = ast.get_source_segment(_source(), _function("driver"))
    assert body is not None
    assert body.index("prepare_plan_remote.remote(") < body.index("prepare_leaf_remote.starmap(")
    assert body.index("prepare_leaf_remote.starmap(") < body.index(
        "finalize_prepared_remote.remote("
    )
    assert body.index("finalize_prepared_remote.remote(") < body.index("run_t1_gpu_remote.remote(")


def test_finalizer_delegates_leaf_validation_and_reduction_to_production() -> None:
    body = ast.get_source_segment(_source(), _function("finalize_prepared_remote"))
    assert body is not None
    assert 'loaded["publish_prepared"](' in body
    assert "_audit_process_v2_t1" not in body
    assert "compile_leaf" not in body
    assert "artifact_volume.commit()" in body
    assert body.index('loaded["publish_prepared"](') < body.index("artifact_volume.commit()")


def test_gpu_binds_process_v2_result_policy_and_runner_hash() -> None:
    body = ast.get_source_segment(_source(), _function("run_t1_gpu_remote"))
    assert body is not None
    assert 'loaded["runner_implementation_sha256"](repo_root=REMOTE_ROOT)' in body
    assert "functools.partial(" in body
    assert "capacity_policy=runtime.capacity_policy" in body
    assert "runner_implementation_sha256=runner_hash" in body
    assert "result_filename=PROCESS_V2_RESULT_FILENAME" in _source()
    assert "STEP_TEN_CHECKPOINT_FILENAME" in body
    assert "completed step 10 without its durable checkpoint" in body


def test_step_ten_checkpoint_is_committed_by_the_gpu_heartbeat() -> None:
    body = ast.get_source_segment(_source(), _function("_gpu_checkpoint_heartbeat"))
    assert body is not None
    assert "checkpoint_path.is_file()" in body
    assert "artifact_volume.commit()" in body
    assert "checkpoint_committed.set()" in body
    assert body.index("checkpoint_path.is_file()") < body.index("artifact_volume.commit()")


def test_gpu_publishes_only_the_recomputed_process_v2_decision() -> None:
    body = ast.get_source_segment(_source(), _function("run_t1_gpu_remote"))
    assert body is not None
    assert 'loaded["build_decision"](' in body
    assert 'loaded["write_bytes_if_absent"](' in body
    assert 'loaded["validate_decision"](' in body
    assert body.index('loaded["build_decision"](') < body.index('loaded["write_bytes_if_absent"](')
    assert body.index('loaded["write_bytes_if_absent"](') < body.index(
        'loaded["validate_decision"]('
    )
    assert '"bounded_p50_authorized": decision["bounded_p50_authorized"]' in body


def test_runner_hash_mismatch_reaches_the_strict_seam_before_training() -> None:
    trained = False

    def strict_runner(
        _model: object,
        _runtime: object,
        *,
        provenance: dict[str, Any],
        expected_runner_implementation_sha256: str,
        **_kwargs: object,
    ) -> dict[str, Any]:
        nonlocal trained
        if provenance["runner_implementation_sha256"] != expected_runner_implementation_sha256:
            raise RuntimeError("runner implementation hash disagrees")
        trained = True
        return {}

    with pytest.raises(RuntimeError, match="runner implementation hash disagrees"):
        launcher._invoke_process_v2_capacity_runner(
            strict_runner,
            model=object(),
            runtime=object(),
            output_directory=Path("/artifacts/run"),
            provenance={"runner_implementation_sha256": "1" * 64},
            result_builder=lambda **_kwargs: {},
            runner_implementation_sha256="2" * 64,
        )
    assert trained is False


def test_shared_runner_exposes_both_process_v2_callback_seams() -> None:
    from compose_v4.experiments.editing_v2_semantic_t1_capacity_runner import (
        run_semantic_t1_capacity,
    )

    parameters = inspect.signature(run_semantic_t1_capacity).parameters
    assert "result_builder" in parameters
    assert "result_filename" in parameters
    assert "expected_runner_implementation_sha256" in parameters


def test_cpu_groups_are_ordered_complete_disjoint_and_bounded() -> None:
    values = tuple(f"{index:064x}" for index in range(93))
    groups = launcher._groups(values, maximum=40)
    assert tuple(item for group in groups for item in group) == values
    assert [len(group) for group in groups] == [40, 40, 13]
    assert len({item for group in groups for item in group}) == len(values)


@pytest.mark.parametrize("maximum", [0, 41, True])
def test_cpu_groups_refuse_invalid_container_bounds(maximum: int) -> None:
    with pytest.raises(ValueError, match="must lie"):
        launcher._groups(("a",), maximum=maximum)


def test_cpu_groups_refuse_repeated_task_identity() -> None:
    with pytest.raises(ValueError, match="repeats"):
        launcher._groups(("a", "a"), maximum=1)


def test_local_revision_refuses_a_dirty_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    answers = {
        ("rev-parse", "HEAD"): "1" * 40,
        ("rev-parse", "HEAD^{tree}"): "2" * 40,
        ("status", "--porcelain=v1", "--untracked-files=all"): " M tracked.py",
    }
    monkeypatch.setattr(
        launcher,
        "_git",
        lambda _root, *arguments: answers[arguments],
    )
    with pytest.raises(RuntimeError, match="exact clean commit"):
        launcher.local_image_revision(expected_commit="1" * 40, repo_root=ROOT)


def test_runtime_source_revision_binds_commit_tree_image_and_inventory() -> None:
    image_body = {
        "schema": launcher.REVISION_SCHEMA,
        "schema_version": launcher.REVISION_SCHEMA_VERSION,
        "commit": "1" * 40,
        "tree": "2" * 40,
        "worktree_clean": True,
        "serialized_sources": {"src/a.py": "3" * 64},
    }
    revision = {**image_body, "image_revision_sha256": launcher._sha256(image_body)}
    observed = launcher._source_revision(revision)
    assert observed == {
        "schema": launcher.SOURCE_REVISION_SCHEMA,
        "schema_version": launcher.SOURCE_REVISION_SCHEMA_VERSION,
        "commit": "1" * 40,
        "tree": "2" * 40,
        "image_revision_sha256": revision["image_revision_sha256"],
        "serialized_source_inventory_sha256": launcher._sha256(revision["serialized_sources"]),
        "source_revision_sha256": launcher._sha256(
            {
                "schema": launcher.SOURCE_REVISION_SCHEMA,
                "schema_version": launcher.SOURCE_REVISION_SCHEMA_VERSION,
                "commit": "1" * 40,
                "tree": "2" * 40,
                "image_revision_sha256": revision["image_revision_sha256"],
                "serialized_source_inventory_sha256": launcher._sha256(
                    revision["serialized_sources"]
                ),
            }
        ),
    }
    from compose_v4.experiments.editing_v2_process_v2_t1_runtime import (
        _validate_source_revision,
    )

    assert _validate_source_revision(observed) == observed


def test_plan_receives_the_authenticated_remote_source_revision() -> None:
    body = ast.get_source_segment(_source(), _function("prepare_plan_remote"))
    assert body is not None
    assert "source_revision = _source_revision(revision)" in body
    assert "source_revision=source_revision" in body


def test_expensive_panel_is_committed_before_plan_construction() -> None:
    body = ast.get_source_segment(_source(), _function("prepare_plan_remote"))
    assert body is not None
    assert 'loaded["load_panel"](' in body
    assert 'loaded["write_panel"](' in body
    cache_commit = body.index("artifact_volume.commit()")
    assert body.index('loaded["write_panel"](') < cache_commit
    assert cache_commit < body.index('loaded["build_plan"](')


def test_panel_cache_address_binds_every_selection_input() -> None:
    source = SimpleNamespace(
        contracts=SimpleNamespace(process_identity_sha256="1" * 64),
        index=SimpleNamespace(
            active8_completion_sha256="2" * 64,
            active8_sentinel_sha256="3" * 64,
        ),
        plan={"plan_sha256": "4" * 64},
        decision={"decision_sha256": "5" * 64},
        policy={"contract_sha256": "6" * 64},
    )
    revision = {"source_revision_sha256": "7" * 64}
    observed = launcher._panel_cache_identity(source, source_revision=revision)
    source.decision["decision_sha256"] = "8" * 64
    changed = launcher._panel_cache_identity(source, source_revision=revision)
    assert len(observed) == 64
    assert observed != changed


def test_remote_revision_refuses_changed_serialized_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher, "_VALIDATED_REVISION_SHA256", None)
    monkeypatch.setattr(launcher, "_serialized_source_paths", lambda _root: ("src/a.py",))
    monkeypatch.setattr(launcher, "_file_sha256", lambda _path: "4" * 64)
    body = {
        "schema": launcher.REVISION_SCHEMA,
        "schema_version": launcher.REVISION_SCHEMA_VERSION,
        "commit": "1" * 40,
        "tree": "2" * 40,
        "worktree_clean": True,
        "serialized_sources": {"src/a.py": "3" * 64},
    }
    revision = {**body, "image_revision_sha256": launcher._sha256(body)}
    with pytest.raises(RuntimeError, match="serialized Process-V2 T1 source differs"):
        launcher._validate_remote_revision(revision)


@pytest.mark.parametrize(
    "value",
    ["relative/path", "/tmp/outside", "/artifacts/../outside"],
)
def test_artifact_paths_cannot_escape_the_mounted_volume(value: str) -> None:
    with pytest.raises(ValueError):
        launcher._require_artifact_path(value, field="test_path")


def test_artifact_path_preserves_identity_but_writes_through_the_physical_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    physical = tmp_path / "physical"
    physical.mkdir()
    logical = tmp_path / "artifacts"
    logical.symlink_to(physical, target_is_directory=True)
    monkeypatch.setattr(launcher, "ARTIFACT_ROOT", logical)

    supplied = str(logical / "editing_v2" / "panel.json")
    identity_path = launcher._require_artifact_path(
        supplied,
        field="panel_path",
    )
    write_path = launcher._require_physical_artifact_path(
        supplied,
        field="panel_path",
    )
    assert identity_path == logical / "editing_v2" / "panel.json"
    assert write_path == physical / "editing_v2" / "panel.json"
    assert logical not in write_path.parents


def test_source_identity_uses_logical_root_and_output_uses_physical_root() -> None:
    source_body = ast.get_source_segment(_source(), _function("_open_source"))
    prepare_body = ast.get_source_segment(_source(), _function("prepare_plan_remote"))
    gpu_body = ast.get_source_segment(_source(), _function("run_t1_gpu_remote"))
    assert source_body is not None and prepare_body is not None and gpu_body is not None
    assert "artifact_root=ARTIFACT_ROOT," in source_body
    assert "artifact_root=ARTIFACT_ROOT.resolve()" not in source_body
    assert "_require_physical_artifact_path(output_prefix" in prepare_body
    assert "_require_physical_artifact_path(output_prefix" in gpu_body


def test_serialized_image_contains_all_process_v2_t1_owners() -> None:
    sources = set(launcher._serialized_source_paths(ROOT))
    assert launcher.LAUNCHER_SOURCE in sources
    for name in (
        "editing_v2_process_v2_t1_panel.py",
        "editing_v2_process_v2_t1_runtime.py",
        "editing_v2_process_v2_t1_result.py",
        "editing_v2_semantic_t1_capacity_runner.py",
    ):
        assert f"src/compose_v4/experiments/{name}" in sources
    assert launcher.CAPACITY_POLICY_SOURCE in sources


def test_remote_surfaces_are_narrow_and_import_launches_nothing() -> None:
    assert tuple(inspect.signature(launcher.prepare_plan_remote.get_raw_f()).parameters) == (
        "active8_run_root",
        "gate_zero_decision_path",
        "output_prefix",
        "revision",
    )
    assert tuple(inspect.signature(launcher.run_t1_gpu_remote.get_raw_f()).parameters) == (
        "prepared_completion_path",
        "output_prefix",
        "revision",
    )
    assert ".spawn(" in ast.get_source_segment(_source(), _function("main"))


def test_launcher_never_starts_p50_and_launch_envelope_grants_no_authority() -> None:
    source = _source()
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("p50" in name.lower() for name in imported)
    assert "run_p50" not in source
    # Launch and preparation envelopes remain false.  Only the validated T1
    # decision may report a true bounded-P50 gate, and no P50 call exists.
    assert '"bounded_p50_authorized": False' in source
    assert '"p50_launched": False' in source
