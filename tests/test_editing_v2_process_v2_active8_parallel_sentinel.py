"""Parallel release-sentinel behavior over the genuine Process-V2 pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import test_editing_v2_process_v2_active8_pipeline as pipeline_fixture
from compose_v4.data.editing_v2_process_v2_active8_reduce import (
    COMPLETION_FILENAME,
    ProcessV2Active8Incomplete,
    ProcessV2Active8ReduceError,
    finalize_process_v2_active8_reduction,
    load_process_v2_active8_reduction_preparation,
    prepare_process_v2_active8_reduction,
    run_process_v2_active8_sentinel_partition,
    task_output_path,
    validate_process_v2_active8_reduction_preparation,
)
from compose_v4.data.editing_v2_process_v2_active8_sentinel import (
    SENTINEL_PARTITION_MISMATCH,
    ProcessV2Active8SentinelError,
    completed_release_sentinel_partition_ids,
    load_release_sentinel_partition_result,
    prepare_release_sentinel_plan,
    reduce_release_sentinel_partitions,
    require_sentinel_passed,
    select_sentinel_pairs,
    sentinel_partition_identities,
    validate_release_sentinel_partition_result,
    write_release_sentinel_partition_result,
)
from compose_v4.data.editing_v2_process_v2_pipeline_schema import (
    PIPELINE_STATUS_NO_AUTHORITY,
    SENTINEL_GLOBAL_EXAMPLES,
    SENTINEL_ORACLE_EXAMPLES,
    SENTINEL_PARTITION_MATCHED,
    SENTINEL_PARTITION_RESULT_SCHEMA,
    SENTINEL_PARTITION_RESULT_SCHEMA_VERSION,
)
from compose_v4.data.editing_v2_process_v2_schema import (
    authority_false_block,
    canonical_bytes,
    canonical_sha256,
    self_hashed,
)

ROOT = Path(__file__).resolve().parents[1]


def _run_parallel(stage, *, max_pairs: int, publish: bool):
    prepared = prepare_process_v2_active8_reduction(
        stage.plan,
        runtime=stage.runtime,
        artifact_root=stage.artifact_root,
        repo_root=ROOT,
        max_sentinel_pairs_per_partition=max_pairs,
        publish=publish,
    )
    results = [
        run_process_v2_active8_sentinel_partition(
            stage.plan,
            prepared,
            identity,
            runtime=stage.runtime,
            artifact_root=stage.artifact_root,
            repo_root=ROOT,
            publish=publish,
        )
        for identity in sentinel_partition_identities(prepared["sentinel_plan"])
    ]
    return prepared, results


def _run_root(stage) -> Path:
    return stage.artifact_root / str(stage.plan["run_artifact_root"]).removeprefix(
        "/artifacts/"
    )


def _matched_partition_results(sentinel_plan):
    results = []
    for partition in sentinel_plan["partitions"]:
        selected_keys = [
            [pair["source_state_sha256"], pair["action_sha256"]]
            for pair in partition["pairs"]
        ]
        body = {
            "schema": SENTINEL_PARTITION_RESULT_SCHEMA,
            "schema_version": SENTINEL_PARTITION_RESULT_SCHEMA_VERSION,
            "status": PIPELINE_STATUS_NO_AUTHORITY,
            **authority_false_block(),
            "partition_outcome": SENTINEL_PARTITION_MATCHED,
            "sentinel_plan_sha256": sentinel_plan["sentinel_plan_sha256"],
            "partition_identity_sha256": partition[
                "partition_identity_sha256"
            ],
            "partition_index": partition["partition_index"],
            "binding_sha256": sentinel_plan["binding_sha256"],
            "plan_sha256": sentinel_plan["plan_sha256"],
            "run_identity_sha256": sentinel_plan["run_identity_sha256"],
            "task_inventory_sha256": sentinel_plan["task_inventory_sha256"],
            "result_inventory_sha256": sentinel_plan["result_inventory_sha256"],
            "selected_pairs": partition["selected_pairs"],
            "selected_pairs_sha256": partition["selected_pairs_sha256"],
            "evaluated_pairs": partition["selected_pairs"],
            "evaluated_pairs_sha256": canonical_sha256(selected_keys),
            "oracle_examples": partition["oracle_examples"],
            "evidence_mismatches": [],
            "cell_mismatches": [],
            "oracle_mismatches": [],
        }
        results.append(self_hashed(body, field="partition_result_sha256"))
    return results


def test_parallel_partitions_preserve_the_serial_final_bytes_and_restart(
    tmp_path: Path,
) -> None:
    stage = pipeline_fixture._Stage(
        tmp_path / "parallel", prefix="/artifacts/active8_parallel_sentinel"
    )
    stage.run()
    prepared, results = _run_parallel(stage, max_pairs=1, publish=True)
    identities = sentinel_partition_identities(prepared["sentinel_plan"])
    assert len(identities) == prepared["sentinel_plan"]["selected_pairs"] > 1

    reopened = load_process_v2_active8_reduction_preparation(
        str(stage.plan["run_artifact_root"]),
        active8_plan=stage.plan,
        artifact_root=stage.artifact_root,
    )
    assert reopened == prepared
    assert completed_release_sentinel_partition_ids(
        active8_plan=stage.plan,
        sentinel_plan=prepared["sentinel_plan"],
        artifact_root=stage.artifact_root,
    ) == set(identities)
    reopened_results = [
        load_release_sentinel_partition_result(
            identity,
            active8_plan=stage.plan,
            sentinel_plan=prepared["sentinel_plan"],
            artifact_root=stage.artifact_root,
        )
        for identity in identities
    ]
    assert reopened_results == results

    forward = finalize_process_v2_active8_reduction(
        stage.plan,
        prepared,
        results,
        artifact_root=stage.artifact_root,
        repo_root=ROOT,
        publish=False,
    )
    reversed_order = finalize_process_v2_active8_reduction(
        stage.plan,
        prepared,
        list(reversed(results)),
        artifact_root=stage.artifact_root,
        repo_root=ROOT,
        publish=False,
    )
    serial = stage.reduce(publish=False)
    assert forward == reversed_order == serial

    published = finalize_process_v2_active8_reduction(
        stage.plan,
        prepared,
        list(reversed(results)),
        artifact_root=stage.artifact_root,
        repo_root=ROOT,
        publish=True,
    )
    assert published == forward
    assert (_run_root(stage) / COMPLETION_FILENAME).read_bytes() == (
        canonical_bytes(published) + b"\n"
    )

    # Exact retries reuse immutable bytes instead of creating another result.
    _, retried = _run_parallel(stage, max_pairs=1, publish=True)
    assert retried == results

    foreign_plan = dict(stage.plan)
    foreign_plan.update(
        {
            "binding_sha256": "a" * 64,
            "plan_sha256": "b" * 64,
            "run_identity_sha256": "c" * 64,
            "task_inventory_sha256": "d" * 64,
            "run_artifact_root": "/artifacts/a_different_active8_run",
        }
    )
    with pytest.raises(ProcessV2Active8SentinelError, match="another Active8 run"):
        write_release_sentinel_partition_result(
            results[0],
            active8_plan=foreign_plan,
            sentinel_plan=prepared["sentinel_plan"],
            artifact_root=stage.artifact_root,
        )


def test_partition_validation_and_reduction_refuse_inventory_mutations(
    tmp_path: Path,
) -> None:
    stage = pipeline_fixture._Stage(
        tmp_path / "inventory", prefix="/artifacts/active8_parallel_inventory"
    )
    stage.run()
    prepared, results = _run_parallel(stage, max_pairs=1, publish=True)
    sentinel_plan = prepared["sentinel_plan"]

    with pytest.raises(ProcessV2Active8SentinelError, match="incomplete"):
        reduce_release_sentinel_partitions(stage.plan, sentinel_plan, results[:-1])
    with pytest.raises(ProcessV2Active8SentinelError, match="twice"):
        reduce_release_sentinel_partitions(
            stage.plan, sentinel_plan, [*results, results[0]]
        )

    body = {
        key: value
        for key, value in results[0].items()
        if key != "partition_result_sha256"
    }
    body["partition_index"] = int(body["partition_index"]) + 1
    mutated = {**body, "partition_result_sha256": canonical_sha256(body)}
    with pytest.raises(ProcessV2Active8SentinelError, match="partition_index"):
        validate_release_sentinel_partition_result(
            mutated, sentinel_plan=sentinel_plan
        )
    unsealed_plan = dict(sentinel_plan)
    unsealed_plan["max_pairs_per_partition"] += 1
    with pytest.raises(ProcessV2Active8SentinelError, match="self-hash"):
        validate_release_sentinel_partition_result(
            results[0], sentinel_plan=unsealed_plan
        )

    sentinel_body = {
        key: value
        for key, value in sentinel_plan.items()
        if key != "sentinel_plan_sha256"
    }
    sentinel_body.update(
        {
            "unique_accepted_pairs": 0,
            "selection_mode": "exhaustive_all_unique_accepted_pairs",
            "selected_pairs": 0,
            "selected_pairs_sha256": canonical_sha256([]),
            "per_cell_examples": 0,
            "global_examples": 0,
            "oracle_examples": 0,
            "partitions": [],
            "partition_inventory_sha256": canonical_sha256([]),
        }
    )
    empty_sentinel = {
        **sentinel_body,
        "sentinel_plan_sha256": canonical_sha256(sentinel_body),
    }
    preparation_body = {
        key: value for key, value in prepared.items() if key != "preparation_sha256"
    }
    preparation_body["sentinel_plan"] = empty_sentinel
    disabled = {
        **preparation_body,
        "preparation_sha256": canonical_sha256(preparation_body),
    }
    with pytest.raises(ProcessV2Active8ReduceError, match="sentinel census"):
        validate_process_v2_active8_reduction_preparation(
            disabled, active8_plan=stage.plan
        )


def test_a_parallel_sentinel_mismatch_blocks_completion(tmp_path: Path) -> None:
    stage = pipeline_fixture._Stage(
        tmp_path / "mismatch", prefix="/artifacts/active8_parallel_mismatch"
    )
    stage.run()
    task = next(
        task
        for task in stage.plan["tasks"]
        if any(
            row["admission_status"] == "accepted"
            for row in pipeline_fixture.read_task_rows(
                task_output_path(
                    stage.plan, task, artifact_root=stage.artifact_root
                )
            )
        )
    )
    pipeline_fixture._replace_target_state_sha256(
        task_output_path(stage.plan, task, artifact_root=stage.artifact_root)
    )
    prepared, results = _run_parallel(stage, max_pairs=1, publish=True)
    assert any(
        result["partition_outcome"] == SENTINEL_PARTITION_MISMATCH
        for result in results
    )
    with pytest.raises(ProcessV2Active8SentinelError, match="release sentinel"):
        finalize_process_v2_active8_reduction(
            stage.plan,
            prepared,
            results,
            artifact_root=stage.artifact_root,
            repo_root=ROOT,
        )
    assert not (_run_root(stage) / COMPLETION_FILENAME).exists()


def test_in_memory_partition_results_cannot_publish_completion(tmp_path: Path) -> None:
    stage = pipeline_fixture._Stage(
        tmp_path / "unpublished", prefix="/artifacts/active8_parallel_unpublished"
    )
    stage.run()
    prepared = prepare_process_v2_active8_reduction(
        stage.plan,
        runtime=stage.runtime,
        artifact_root=stage.artifact_root,
        repo_root=ROOT,
        max_sentinel_pairs_per_partition=1,
        publish=True,
    )
    results = [
        run_process_v2_active8_sentinel_partition(
            stage.plan,
            prepared,
            identity,
            runtime=stage.runtime,
            artifact_root=stage.artifact_root,
            repo_root=ROOT,
            publish=False,
        )
        for identity in sentinel_partition_identities(prepared["sentinel_plan"])
    ]
    with pytest.raises(ProcessV2Active8SentinelError, match="absent"):
        finalize_process_v2_active8_reduction(
            stage.plan,
            prepared,
            results,
            artifact_root=stage.artifact_root,
            repo_root=ROOT,
            publish=True,
        )
    assert not (_run_root(stage) / COMPLETION_FILENAME).exists()


def test_changed_map_inventory_after_preparation_blocks_completion(
    tmp_path: Path,
) -> None:
    stage = pipeline_fixture._Stage(
        tmp_path / "stale", prefix="/artifacts/active8_parallel_stale"
    )
    stage.run()
    prepared, results = _run_parallel(stage, max_pairs=1, publish=True)
    task = stage.plan["tasks"][0]
    output = task_output_path(stage.plan, task, artifact_root=stage.artifact_root)
    summary_path = output / pipeline_fixture.SUMMARY_FILENAME
    summary = json.loads(summary_path.read_bytes())
    body = {key: value for key, value in summary.items() if key != "summary_sha256"}
    body["rows_stream_sha256"] = (
        "0" * 64 if body["rows_stream_sha256"] != "0" * 64 else "f" * 64
    )
    changed = {**body, "summary_sha256": canonical_sha256(body)}
    summary_path.write_bytes(canonical_bytes(changed) + b"\n")
    # This field is not one of the map validator's derived comparisons, so the
    # exact content inventory, rather than a convenient receipt-only inventory,
    # must catch the change at finalization.
    pipeline_fixture.validate_process_v2_active8_task_result(output)
    with pytest.raises(
        ProcessV2Active8Incomplete,
        match="task_result_content_inventory changed",
    ):
        finalize_process_v2_active8_reduction(
            stage.plan,
            prepared,
            results,
            artifact_root=stage.artifact_root,
            repo_root=ROOT,
        )
    assert not (_run_root(stage) / COMPLETION_FILENAME).exists()


def test_ranked_selection_and_final_bytes_ignore_input_order_and_partition_geometry():
    task_identity = "e" * 64
    required_cells = [f"cell-{index:02d}" for index in range(17)]
    plan = {
        "binding": {"required_cell_ids": required_cells},
        "binding_sha256": "1" * 64,
        "plan_sha256": "2" * 64,
        "run_identity_sha256": "3" * 64,
        "task_inventory_sha256": "4" * 64,
        "tasks": [{"task_identity_sha256": task_identity}],
    }
    transitions = []
    for index in range(6_785):
        source_sha256 = canonical_sha256(["source", index])
        action_sha256 = canonical_sha256(["action", index])
        transitions.append(
            {
                "task_identity_sha256": task_identity,
                "v1_task_identity_sha256": canonical_sha256(["legacy", index]),
                "entry_index": index,
                "step_index": 0,
                "candidate_evidence": {
                    "source_state_sha256": source_sha256,
                    "action_sha256": action_sha256,
                },
                "capability_cell_id": required_cells[index % len(required_cells)],
                "family_context": "synthetic-ranked-probe",
                "audit_axes": {},
                "model_family": "atom_delete",
            }
        )

    selection = select_sentinel_pairs(
        transitions,
        required_cell_ids=required_cells,
    )
    assert selection["unique_accepted_pairs"] == 6_785
    assert selection["global_examples"] == SENTINEL_GLOBAL_EXAMPLES
    assert len(selection["selected"]) == SENTINEL_GLOBAL_EXAMPLES

    inventory_sha256 = canonical_sha256([[task_identity, "f" * 64]])
    plan_73 = prepare_release_sentinel_plan(
        plan,
        transitions,
        result_inventory_sha256=inventory_sha256,
        max_pairs_per_partition=73,
    )
    reversed_plan_73 = prepare_release_sentinel_plan(
        plan,
        list(reversed(transitions)),
        result_inventory_sha256=inventory_sha256,
        max_pairs_per_partition=73,
    )
    plan_512 = prepare_release_sentinel_plan(
        plan,
        transitions,
        result_inventory_sha256=inventory_sha256,
        max_pairs_per_partition=512,
    )
    assert plan_73 == reversed_plan_73
    assert plan_73["oracle_examples"] == SENTINEL_ORACLE_EXAMPLES
    assert len(plan_73["partitions"]) > len(plan_512["partitions"])

    results_73 = _matched_partition_results(plan_73)
    final_73 = reduce_release_sentinel_partitions(
        plan,
        plan_73,
        list(reversed(results_73)),
    )
    final_512 = reduce_release_sentinel_partitions(
        plan,
        plan_512,
        _matched_partition_results(plan_512),
    )
    assert final_73 == final_512

    weakened_body = {
        key: value for key, value in final_73.items() if key != "sentinel_sha256"
    }
    weakened_body["global_examples"] -= 1
    weakened = self_hashed(weakened_body, field="sentinel_sha256")
    with pytest.raises(ProcessV2Active8SentinelError, match="ranked sentinel"):
        require_sentinel_passed(weakened)
