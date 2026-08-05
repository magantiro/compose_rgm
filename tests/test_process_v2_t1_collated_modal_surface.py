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
