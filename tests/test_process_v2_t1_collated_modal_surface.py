"""The optimized T1 launcher keeps chemistry collation off the GPU."""

from __future__ import annotations

import inspect

from modal_apps import run_process_v2_t1_app as launcher


def _body(function) -> str:
    return inspect.getsource(function.get_raw_f())


def test_collated_driver_finishes_cpu_cache_before_allocating_gpu() -> None:
    cache_body = _body(launcher.collated_cache_only_driver)
    assert cache_body.index("collate_t1_leaf_remote.starmap(") < cache_body.index(
        "finalize_collated_remote.remote("
    )
    assert "run_t1_collated_gpu_remote" not in cache_body
    assert "prepare_leaf_remote" not in cache_body
    assert '"training_launched": False' in cache_body
    body = _body(launcher.collated_reuse_driver)
    assert body.index("collated_cache_only_driver.remote(") < body.index(
        "run_t1_collated_gpu_remote.remote("
    )
    assert '"fiber_recomputation_count": 0' in body
    assert '"gpu_side_collation_count": 0' in body


def test_cpu_collation_uses_one_replenishing_bounded_starmap() -> None:
    body = _body(launcher.collated_cache_only_driver)
    assert body.count("collate_t1_leaf_remote.starmap(") == 1
    assert "update_autoscaler(max_containers=max_cpu_containers)" in body
    assert "for identifier in missing" in body


def test_cached_gpu_uses_authenticated_materialized_panel() -> None:
    body = _body(launcher.run_t1_collated_gpu_remote)
    assert body.index('loaded["load_materialized_collated_panel"](') < body.index(
        'scratch.model.to(device="cuda"'
    )
    assert "materialized_panel=materialized_panel" in body
    assert "expected_runner_source_revision_sha256=" in body
    assert "_materialize_panel" not in body
    assert '"gpu_side_collation_count": 0' in body


def test_cached_entrypoint_defaults_to_32_tasks_for_the_512_entry_panel() -> None:
    assert launcher.COLLATED_ENTRIES_PER_TASK == 16
    assert launcher.MAX_CPU_CONTAINERS == 40
    assert 512 // launcher.COLLATED_ENTRIES_PER_TASK == 32


def test_cached_gpu_binds_cache_completion_into_run_identity() -> None:
    body = _body(launcher.run_t1_collated_gpu_remote)
    assert '"collated_completion_sha256": collated_completion["completion_sha256"]' in body
    assert '"prepared_completion_sha256": provenance["prepared_completion_sha256"]' in body
    assert '"runner_source_revision_sha256"' in body


def test_failure_scope_reuses_the_cache_and_fans_out_independent_arms() -> None:
    body = _body(launcher.run_t1_failure_scope_gpu_remote)
    assert 'loaded["load_materialized_collated_panel"](' in body
    assert 'loaded["failing_families"](' in body
    assert 'if family not in failing_families:' in body
    assert 'scratch.model.to(device="cuda"' in body
    assert 'loaded["run_failure_scope"](' in body
    assert '"fiber_recomputation_count": 0' in body
    assert '"gpu_side_collation_count": 0' in body
    assert "starmap(" not in body
    driver = _body(launcher.failure_scope_driver)
    assert 'loaded["failing_families"](' in driver
    assert "update_autoscaler(" in driver
    assert "max_containers=len(families)" in driver
    assert "run_t1_failure_scope_gpu_remote.starmap(" in driver


def test_failure_scope_publishes_each_family_restart_safely() -> None:
    body = _body(launcher.run_t1_failure_scope_gpu_remote)
    assert "if result_path.is_file():" in body
    assert body.index('loaded["write_bytes_if_absent"](') < body.index(
        "artifact_volume.commit()"
    )
    assert '"bounded_p50_authorized": False' in body
    assert '"p50_launched": False' in body


def test_next_failure_scope_binds_predecessors_and_uses_one_gpu() -> None:
    body = _body(launcher.run_t1_failure_scope_gpu_remote)
    assert 'loaded["validate_failure_scope"](' in body
    assert 'loaded["next_failure_scope"](' in body
    assert 'loaded["run_next_failure_scope"](' in body
    assert '"prior_scope_result_file_sha256"' in body
    assert '"prior_scope_result_sha256"' in body
    driver = _body(launcher.next_failure_scope_driver)
    assert "set(prior_scope_result_paths) != set(families)" in driver
    assert "update_autoscaler(max_containers=1)" in driver
    assert 'execution_mode="one_gpu_sequential_arms"' in driver
    assert "run_t1_failure_scope_gpu_remote.starmap(" in driver
    assert '"fiber_recomputation_count": 0' in driver
    assert launcher.FAILURE_SCOPE_GPU_TIMEOUT_SECONDS == 20 * 60


def test_score_revision_repair_runs_only_three_cached_families_on_one_gpu() -> None:
    body = _body(launcher.run_t1_score_revision_repair_remote)
    assert launcher._SCORE_REVISION_REPAIR_FAMILIES == (
        "bond_reroute",
        "cycle_attach",
        "ring_system_restate",
    )
    assert body.count("score_revision_rebind=True") == 3
    assert 'loaded["load_materialized_collated_panel"](' in body
    assert 'loaded["run_failure_scope"](' in body
    assert 'loaded["run_next_failure_scope"](' in body
    assert "copy.deepcopy(scratch.model)" in body
    assert "compile_state_successor_map" not in body
    assert "starmap(" not in body
    assert '"fiber_recomputation_count": 0' in body
    assert '"gpu_side_collation_count": 0' in body
    assert '"p50_launched": False' in body
