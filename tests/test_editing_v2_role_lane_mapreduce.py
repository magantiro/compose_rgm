"""Restart, integrity, and reference-equivalence tests for role/lane map/reduce."""

from __future__ import annotations

import gzip
import json
from pathlib import Path, PurePosixPath
from unittest.mock import patch

import pytest

from compose_v4.data import editing_v2_role_lane_mapreduce as mapreduce_module
from compose_v4.data.editing_v2_active8_source_adapter import (
    resolve_editing_v2_active8_sources,
)
from compose_v4.data.editing_v2_packed_candidate_materializer import (
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
)
from compose_v4.data.editing_v2_role_lane_mapreduce import (
    PLAN_FILENAME,
    TASK_RECEIPT_FILENAME,
    EditingV2RoleLaneMapReduceError,
    EditingV2RoleLaneMapReduceIncomplete,
    map_role_lane_source_shard,
    plan_role_lane_mapreduce,
    reduce_role_lane_cell,
    reduce_role_lane_mapreduce,
    validate_role_lane_mapreduce_plan,
    validate_role_lane_task_receipt,
)
from compose_v4.data.editing_v2_role_lane_packed_materializer import (
    LANE_REGISTRY_FILENAME,
    RESOLVED_MEMBERSHIP_FILENAME,
)
from compose_v4.data.packed_trace_store import manifest_path_for
from tests.test_editing_v2_active8_source_adapter import (
    CORPUS_CONTRACT,
    LANES,
    _self_hash,
    _write_candidate_materialization,
    _write_json,
    _write_split_assignment,
)
from tests.test_editing_v2_role_lane_packed_materializer import (
    _candidate_for_source,
    _inputs,
    _materialize,
    _write_original_source,
)


def _plan(inputs: dict, *, final_prefix: str) -> tuple[dict, Path]:
    def validate_bridge(*args, **kwargs):
        assert kwargs.get("_validated_candidate_materialization") is not None
        return {"source_stream": inputs["source_stream"]}

    with patch(
        "compose_v4.data.editing_v2_role_lane_mapreduce."
        "validate_candidate_provenance_bridge",
        side_effect=validate_bridge,
    ):
        plan = plan_role_lane_mapreduce(
            candidate_materialization_dir=inputs["candidate_root"],
            candidate_provenance_bridge_dir=inputs["bridge_root"],
            candidate_provenance_registry_path=inputs["registry_path"],
            split_assignment_path=inputs["split_path"],
            editing_corpus_contract_path=CORPUS_CONTRACT,
            artifact_root=inputs["artifact_root"],
            code_revision="a" * 40,
            final_output_artifact_prefix=final_prefix,
            execution_artifact_prefix="/artifacts/editing_v2/role_lane_mapreduce_fixture",
            expected_source_shards=inputs.get(
                "expected_source_shards", len(inputs["rows"])
            ),
        )
    plan_path = (
        inputs["artifact_root"]
        / PurePosixPath(plan["run_artifact_root"]).relative_to("/artifacts")
        / PLAN_FILENAME
    )
    return plan, plan_path


def test_verified_cell_copy_is_not_a_hard_link_and_fails_closed_on_hash(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    source.write_bytes(b"verified reusable cell payload")

    mapreduce_module._copy_verified_bytes(
        source,
        destination,
        expected_sha256=file_sha256(source),
    )

    assert destination.read_bytes() == source.read_bytes()
    assert destination.stat().st_ino != source.stat().st_ino
    rejected = tmp_path / "rejected.bin"
    with pytest.raises(
        EditingV2RoleLaneMapReduceError,
        match="copied payload SHA-256 disagrees",
    ):
        mapreduce_module._copy_verified_bytes(
            source,
            rejected,
            expected_sha256="0" * 64,
        )
    assert not rejected.exists()


def _multi_entry_inputs(root: Path) -> dict:
    """Build four shards with multiple entries and cross-shard same-cell rows."""

    root.mkdir(parents=True, exist_ok=True)
    artifact_root = root / "artifacts"
    artifact_root.mkdir()
    lanes = [lane for lane in LANES for _ in range(4)] + [LANES[0], LANES[0]]
    base_entries: dict[int, dict] = {}
    for index in range(len(lanes)):
        _, entry, _, _, _ = _write_original_source(artifact_root, index=index)
        base_entries[index] = entry
    groups = (
        (0, 1, 2, 3, 4, 5),
        (20, 6, 7, 8, 9, 10),
        (21, 11, 12, 13, 14, 15),
        (16, 17, 18, 19),
    )
    rows_by_index: dict[int, dict] = {}
    for shard_index, indices in enumerate(groups):
        relative = f"combined/source_{shard_index:02d}.jsonl.gz"
        path = artifact_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        entries = [base_entries[index] for index in indices]
        with (
            path.open("wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        ):
            for entry in entries:
                compressed.write(canonical_json_bytes(entry) + b"\n")
        manifest = {
            "schema": "compose.data.packed_trace",
            "schema_version": 1,
            "sampler_contract": {"fixture": True},
            "entries": len(entries),
            "states": 2 * len(entries),
            "provenance": {"fixture": True},
        }
        manifest_path = manifest_path_for(path)
        _write_json(manifest_path, manifest)
        overlay_path = Path(f"{path}.provenance.json")
        _write_json(
            overlay_path,
            {
                "schema": "compose.data.provenance_overlay",
                "schema_version": 1,
                "fixture": True,
            },
        )
        shard_sha256 = file_sha256(path)
        manifest_sha256 = file_sha256(manifest_path)
        overlay_sha256 = file_sha256(overlay_path)
        for entry_index, index in enumerate(indices):
            entry = base_entries[index]
            row = _candidate_for_source(
                index=index,
                lane=lanes[index],
                relative_path=relative,
                entry=entry,
                shard_sha256=shard_sha256,
                manifest_sha256=manifest_sha256,
                overlay_sha256=overlay_sha256,
            )
            row["packed_address"]["entry_index"] = entry_index
            address_body = {
                "source_asset_sha256": row["source_asset"]["source_asset_sha256"],
                "packed_shard_file_sha256": shard_sha256,
                "historical_provenance_overlay_file_sha256": overlay_sha256,
                "entry_index": entry_index,
                "trace_id": entry["trace"]["trace_id"],
                "layer": entry["trace"]["layer"],
                "cache_partition": entry["trace"]["partition"],
                "path_length": entry["trace"]["path_length"],
                "state_count": len(entry["states"]),
                "encoded_state_stream_sha256": row["exact_states"][
                    "encoded_state_stream_sha256"
                ],
            }
            row["packed_address"]["address_sha256"] = canonical_sha256(address_body)
            rows_by_index[index] = _self_hash(row, "row_sha256")
    rows = [rows_by_index[index] for index in range(len(lanes))]
    candidate_root, candidate_manifest = _write_candidate_materialization(root, rows)
    split_path, assignment, role_by_candidate = _write_split_assignment(
        root,
        rows,
        candidate_root=candidate_root,
        candidate_manifest=candidate_manifest,
    )
    assignment["total_mass_units_by_lane"] = {
        lane: sum(row["data_lane"] == lane for row in rows) for lane in LANES
    }
    assignment = _self_hash(assignment, "assignment_sha256")
    _write_json(split_path, assignment)
    return {
        "artifact_root": artifact_root,
        "rows": rows,
        "candidate_root": candidate_root,
        "candidate_manifest": candidate_manifest,
        "split_path": split_path,
        "assignment": assignment,
        "source_stream": assignment["source_stream"],
        "bridge_root": root / "candidate_provenance_bridge",
        "registry_path": root / "candidate_provenance_registry.json",
        "role_by_candidate": role_by_candidate,
        "expected_source_shards": len(groups),
    }


def _map_all(inputs: dict, plan: dict, plan_path: Path) -> list[dict]:
    return [
        map_role_lane_source_shard(
            plan_path=plan_path,
            task_id=task_id,
            artifact_root=inputs["artifact_root"],
        )
        for task_id in plan["task_order"]
    ]


def _reduce_all_cells(inputs: dict, plan: dict, plan_path: Path) -> list[dict]:
    lane_order = plan["reference_materialization"]["run_identity_body"]["lane_order"]
    role_order = plan["reference_materialization"]["run_identity_body"]["role_order"]
    return [
        reduce_role_lane_cell(
            plan_path=plan_path,
            lane=lane,
            role=role,
            artifact_root=inputs["artifact_root"],
        )
        for lane in lane_order
        for role in role_order
    ]


def _output_root(inputs: dict, manifest: dict) -> Path:
    return inputs["artifact_root"] / PurePosixPath(
        manifest["run_artifact_root"]
    ).relative_to("/artifacts")


def _receipt_cells(root: Path) -> dict[tuple[str, str], dict]:
    receipt = json.loads((root / RESOLVED_MEMBERSHIP_FILENAME).read_text())
    return {
        (shard["data_lane"], shard["partition_role"]): shard
        for shard in receipt["shards"]
    }


def _decompressed_rows(inputs: dict, shard: dict) -> list[dict]:
    path = inputs["artifact_root"] / PurePosixPath(shard["artifact_path"]).relative_to(
        "/artifacts"
    )
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle]


def test_mapreduce_matches_reference_rows_and_ordered_memberships(
    tmp_path: Path,
) -> None:
    inputs = _multi_entry_inputs(tmp_path)
    reference_manifest, reference_root = _materialize(inputs)
    plan, plan_path = _plan(
        inputs,
        final_prefix="/artifacts/editing_v2/role_lane_mapreduce_final",
    )

    assert len(plan["tasks"]) == inputs["expected_source_shards"]
    assert plan["selected_candidates"] == len(inputs["rows"])
    assert max(task["selected_candidates"] for task in plan["tasks"]) > 1
    _map_all(inputs, plan, plan_path)
    _reduce_all_cells(inputs, plan, plan_path)
    reduced = reduce_role_lane_mapreduce(
        plan_path=plan_path,
        artifact_root=inputs["artifact_root"],
    )
    reduced_root = _output_root(inputs, reduced)

    assert reduced["totals"] == reference_manifest["totals"]
    assert reduced["training_authorized"] is False
    assert reduced["active8_admission_status"] == "NOT_RUN"
    resolved = resolve_editing_v2_active8_sources(
        candidate_materialization_dir=inputs["candidate_root"],
        split_assignment_path=inputs["split_path"],
        editing_corpus_contract_path=CORPUS_CONTRACT,
        lane_registry_path=reduced_root / LANE_REGISTRY_FILENAME,
        artifact_root=inputs["artifact_root"],
        membership_receipt_path=reduced_root / RESOLVED_MEMBERSHIP_FILENAME,
    )
    assert len(resolved.shards) == 20
    reference_cells = _receipt_cells(reference_root)
    reduced_cells = _receipt_cells(reduced_root)
    assert set(reduced_cells) == set(reference_cells)
    for cell, expected in reference_cells.items():
        observed = reduced_cells[cell]
        assert _decompressed_rows(inputs, observed) == _decompressed_rows(
            inputs, expected
        )
        assert [
            (
                row["candidate_id"],
                row["original_packed_address"],
                row["original_packed_row_sha256"],
                row["output_packed_row_sha256"],
            )
            for row in observed["entry_memberships"]
        ] == [
            (
                row["candidate_id"],
                row["original_packed_address"],
                row["original_packed_row_sha256"],
                row["output_packed_row_sha256"],
            )
            for row in expected["entry_memberships"]
        ]
    cross_shard_cell = reduced_cells[(LANES[0], "train")]
    assert len(cross_shard_cell["entry_memberships"]) == 2


def test_task_and_reduce_reuse_exact_valid_artifacts(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    plan, plan_path = _plan(
        inputs,
        final_prefix="/artifacts/editing_v2/role_lane_mapreduce_reuse_final",
    )
    first_task = plan["tasks"][0]
    first_receipt = map_role_lane_source_shard(
        plan_path=plan_path,
        task_id=first_task["task_identity_sha256"],
        artifact_root=inputs["artifact_root"],
    )
    task_root = (
        inputs["artifact_root"]
        / PurePosixPath(plan["run_artifact_root"]).relative_to("/artifacts")
        / "task_results"
        / first_task["task_identity_sha256"]
    )
    receipt_file_sha256 = file_sha256(task_root / TASK_RECEIPT_FILENAME)
    reopened = map_role_lane_source_shard(
        plan_path=plan_path,
        task_id=first_task["task_identity_sha256"],
        artifact_root=inputs["artifact_root"],
    )
    assert reopened == first_receipt
    assert file_sha256(task_root / TASK_RECEIPT_FILENAME) == receipt_file_sha256

    for task_id in plan["task_order"][1:]:
        map_role_lane_source_shard(
            plan_path=plan_path,
            task_id=task_id,
            artifact_root=inputs["artifact_root"],
        )
    first_cells = _reduce_all_cells(inputs, plan, plan_path)
    assert _reduce_all_cells(inputs, plan, plan_path) == first_cells
    first_result = reduce_role_lane_mapreduce(
        plan_path=plan_path,
        artifact_root=inputs["artifact_root"],
    )
    output_manifest = _output_root(inputs, first_result) / first_result.get(
        "schema", "unused"
    )
    second_result = reduce_role_lane_mapreduce(
        plan_path=plan_path,
        artifact_root=inputs["artifact_root"],
    )
    assert second_result == first_result
    assert not output_manifest.exists()


def test_missing_extra_and_tampered_task_receipts_fail_closed(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path / "missing")
    plan, plan_path = _plan(
        inputs,
        final_prefix="/artifacts/editing_v2/role_lane_mapreduce_missing_final",
    )
    map_role_lane_source_shard(
        plan_path=plan_path,
        task_id=plan["task_order"][0],
        artifact_root=inputs["artifact_root"],
    )
    with pytest.raises(EditingV2RoleLaneMapReduceIncomplete, match="incomplete"):
        reduce_role_lane_mapreduce(
            plan_path=plan_path,
            artifact_root=inputs["artifact_root"],
        )

    inputs = _inputs(tmp_path / "tamper")
    plan, plan_path = _plan(
        inputs,
        final_prefix="/artifacts/editing_v2/role_lane_mapreduce_tamper_final",
    )
    receipts = _map_all(inputs, plan, plan_path)
    task = plan["tasks"][0]
    fragment = receipts[0]["fragments"][0]
    fragment_path = inputs["artifact_root"] / PurePosixPath(
        fragment["membership_artifact_path"]
    ).relative_to("/artifacts")
    fragment_path.write_bytes(fragment_path.read_bytes() + b"{}\n")
    with pytest.raises(EditingV2RoleLaneMapReduceError, match="tampered"):
        validate_role_lane_task_receipt(
            plan=plan,
            task=task,
            artifact_root=inputs["artifact_root"],
        )

    result_parent = (
        inputs["artifact_root"]
        / PurePosixPath(plan["run_artifact_root"]).relative_to("/artifacts")
        / "task_results"
    )
    (result_parent / ("f" * 64)).mkdir()
    with pytest.raises(EditingV2RoleLaneMapReduceError, match="unplanned"):
        reduce_role_lane_mapreduce(
            plan_path=plan_path,
            artifact_root=inputs["artifact_root"],
        )


def test_plan_reuse_and_task_input_tamper_are_detected(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    plan, plan_path = _plan(
        inputs,
        final_prefix="/artifacts/editing_v2/role_lane_mapreduce_plan_reuse_final",
    )
    reopened, reopened_path = _plan(
        inputs,
        final_prefix="/artifacts/editing_v2/role_lane_mapreduce_plan_reuse_final",
    )
    assert reopened == plan
    assert reopened_path == plan_path

    task_input = inputs["artifact_root"] / PurePosixPath(
        plan["tasks"][0]["selection_artifact_path"]
    ).relative_to("/artifacts")
    task_input.write_bytes(task_input.read_bytes() + b"{}\n")
    with pytest.raises(EditingV2RoleLaneMapReduceError, match="tampered"):
        validate_role_lane_mapreduce_plan(
            plan_path,
            artifact_root=inputs["artifact_root"],
            validate_task_inputs=True,
        )


def test_planning_uses_header_only_manifest_then_unchanged_reference_indexer(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    bridge_calls = 0

    def validate_bridge(*args, **kwargs):
        nonlocal bridge_calls
        bridge_calls += 1
        prevalidated = kwargs.get("_validated_candidate_materialization")
        assert prevalidated is not None
        assert (
            prevalidated["manifest_sha256"]
            == inputs["candidate_manifest"]["manifest_sha256"]
        )
        assert "validated_header_consumer" not in kwargs
        return {"source_stream": inputs["source_stream"]}

    with (
        patch(
            "compose_v4.data.editing_v2_role_lane_mapreduce."
            "validate_candidate_provenance_bridge",
            side_effect=validate_bridge,
        ),
        patch.object(
            mapreduce_module.reference,
            "_index_selected_candidates",
            wraps=mapreduce_module.reference._index_selected_candidates,
        ) as reference_indexer,
    ):
        plan = plan_role_lane_mapreduce(
            candidate_materialization_dir=inputs["candidate_root"],
            candidate_provenance_bridge_dir=inputs["bridge_root"],
            candidate_provenance_registry_path=inputs["registry_path"],
            split_assignment_path=inputs["split_path"],
            editing_corpus_contract_path=CORPUS_CONTRACT,
            artifact_root=inputs["artifact_root"],
            code_revision="a" * 40,
            final_output_artifact_prefix=(
                "/artifacts/editing_v2/role_lane_two_pass_final"
            ),
            execution_artifact_prefix=(
                "/artifacts/editing_v2/role_lane_two_pass_execution"
            ),
            expected_source_shards=len(inputs["rows"]),
        )

    assert bridge_calls == 1
    assert reference_indexer.call_count == 1
    assert plan["selected_candidates"] == len(inputs["rows"])
