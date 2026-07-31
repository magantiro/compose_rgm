"""Physical Editing-V2 role/lane materialization integration tests."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath

import pytest

from compose_v4.data.editing_v2_active8_source_adapter import (
    RESOLVED_PACKED_MEMBERSHIP_SCHEMA_VERSION,
    EditingV2Active8SourceAdapterError,
    resolve_editing_v2_active8_sources,
)
from compose_v4.data.editing_v2_packed_candidate_materializer import (
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
)
from compose_v4.data.editing_v2_role_lane_packed_materializer import (
    LANE_REGISTRY_FILENAME,
    RESOLVED_MEMBERSHIP_FILENAME,
    ROLE_LANE_MATERIALIZATION_FILENAME,
    EditingV2RoleLaneMaterializationError,
    materialize_editing_v2_role_lane_packed,
)
from compose_v4.data.packed_trace_store import manifest_path_for
from tests.test_editing_v2_active8_source_adapter import (
    CORPUS_CONTRACT,
    LANES,
    PARTITION_ROLES,
    _candidate_row,
    _self_hash,
    _write_candidate_materialization,
    _write_json,
    _write_split_assignment,
)


def _state_stream_sha256(states: list[dict]) -> str:
    digest = hashlib.sha256()
    for state in states:
        digest.update(canonical_json_bytes(state))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_original_source(
    artifact_root: Path,
    *,
    index: int,
) -> tuple[Path, dict, str, str, str]:
    relative_path = f"original/shard_{index:03d}.jsonl.gz"
    path = artifact_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    source_state = {
        "n_slots": 3,
        "atom_types": [1, 0, 0],
        "formal_charges": [0, 0, 0],
        "implicit_h_counts": [0, 0, 0],
        "bonds": [],
    }
    target_state = {
        **source_state,
        "atom_types": [2, 0, 0],
    }
    trace = {
        "schema": "compose.rewrite.trace",
        "schema_version": 2,
        "source_state": source_state,
        "source_key": f"source-key-{index}",
        "target_key": f"target-key-{index}",
        "n_slots": 3,
        "steps": [
            {
                "action": {
                    "schema": "compose.rewrite.action",
                    "schema_version": 2,
                    "executor_rule": "atom_restate",
                    "model_family": "atom_restate",
                    "payload_type": "FixturePayload",
                    "payload": {"slot": 0},
                },
                "successor_key": f"target-key-{index}",
                "atom_count_delta": 0,
                "cycle_rank_delta": 0,
            }
        ],
        "path_length": 1,
        "family_histogram": {"atom_restate": 1},
        "operator_histogram": {"atom_restate": 1},
        "atom_count_delta": 0,
        "cycle_rank_delta": 0,
        "trace_id": f"trace-{index:03d}",
        "partition": "legacy_train",
        "layer": "mmp_analogue",
        "seed": index,
        "metadata": {
            "fixture": True,
            "partition_isolation": {
                "schema": "compose.data.partition_isolation",
                "schema_version": 1,
                "molecule_ids": [f"molecule-{index}"],
                "scaffold_ids": [f"scaffold-{index}"],
                "source_group_id": f"group-{index}",
            },
        },
    }
    entry = {"trace": trace, "states": [source_state, target_state]}
    with path.open("wb") as raw:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            mtime=0,
        ) as compressed:
            compressed.write(canonical_json_bytes(entry) + b"\n")
    manifest = {
        "schema": "compose.data.packed_trace",
        "schema_version": 1,
        "sampler_contract": {"fixture": True},
        "entries": 1,
        "states": 2,
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
    return (
        path,
        entry,
        file_sha256(path),
        file_sha256(manifest_path),
        file_sha256(overlay_path),
    )


def _candidate_for_source(
    *,
    index: int,
    lane: str,
    relative_path: str,
    entry: dict,
    shard_sha256: str,
    manifest_sha256: str,
    overlay_sha256: str,
) -> dict:
    row = _candidate_row(index, lane)
    trace = entry["trace"]
    states = entry["states"]
    state_stream_sha256 = _state_stream_sha256(states)
    address_body = {
        "source_asset_sha256": row["source_asset"]["source_asset_sha256"],
        "packed_shard_file_sha256": shard_sha256,
        "historical_provenance_overlay_file_sha256": overlay_sha256,
        "entry_index": 0,
        "trace_id": trace["trace_id"],
        "layer": trace["layer"],
        "cache_partition": trace["partition"],
        "path_length": trace["path_length"],
        "state_count": len(states),
        "encoded_state_stream_sha256": state_stream_sha256,
    }
    row["packed_address"] = {
        "relative_path": relative_path,
        "packed_shard_file_sha256": shard_sha256,
        "packed_shard_manifest_file_sha256": manifest_sha256,
        "historical_provenance_overlay_file_sha256": overlay_sha256,
        "entry_index": 0,
        "trace_id": trace["trace_id"],
        "layer": trace["layer"],
        "cache_partition": trace["partition"],
        "path_length": trace["path_length"],
        "address_sha256": canonical_sha256(address_body),
    }
    row["trace_envelope_sha256"] = canonical_sha256(trace)
    row["exact_states"]["state_count"] = len(states)
    row["exact_states"]["encoded_state_stream_sha256"] = state_stream_sha256
    row["candidate_payload_sha256"] = canonical_sha256(
        {
            "trace_envelope_sha256": row["trace_envelope_sha256"],
            "encoded_state_stream_sha256": state_stream_sha256,
        }
    )
    return _self_hash(row, "row_sha256")


def _inputs(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    artifact_root = root / "artifacts"
    artifact_root.mkdir()
    rows = []
    for index, lane in enumerate(lane for lane in LANES for _ in PARTITION_ROLES):
        (
            source_path,
            entry,
            shard_sha256,
            manifest_sha256,
            overlay_sha256,
        ) = _write_original_source(artifact_root, index=index)
        rows.append(
            _candidate_for_source(
                index=index,
                lane=lane,
                relative_path=str(source_path.relative_to(artifact_root)),
                entry=entry,
                shard_sha256=shard_sha256,
                manifest_sha256=manifest_sha256,
                overlay_sha256=overlay_sha256,
            )
        )
    candidate_root, candidate_manifest = _write_candidate_materialization(
        root,
        rows,
    )
    split_path, assignment, role_by_candidate = _write_split_assignment(
        root,
        rows,
    )
    return {
        "artifact_root": artifact_root,
        "rows": rows,
        "candidate_root": candidate_root,
        "candidate_manifest": candidate_manifest,
        "split_path": split_path,
        "assignment": assignment,
        "role_by_candidate": role_by_candidate,
    }


def _materialize(inputs: dict) -> tuple[dict, Path]:
    result = materialize_editing_v2_role_lane_packed(
        candidate_materialization_dir=inputs["candidate_root"],
        split_assignment_path=inputs["split_path"],
        editing_corpus_contract_path=CORPUS_CONTRACT,
        artifact_root=inputs["artifact_root"],
        code_revision="a" * 40,
    )
    output_root = inputs["artifact_root"] / PurePosixPath(result["run_artifact_root"]).relative_to(
        "/artifacts"
    )
    return result, output_root


def test_materializes_twenty_deterministic_cells_and_adapter_accepts(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    result, output_root = _materialize(inputs)

    assert result["totals"] == {
        "routed_candidates": 20,
        "output_shards": 20,
        "output_records": 20,
        "output_states": 40,
        "output_transitions": 20,
    }
    assert result["training_authorized"] is False
    assert result["active8_admission_status"] == "NOT_RUN"
    assert (output_root / ROLE_LANE_MATERIALIZATION_FILENAME).is_file()
    receipt = json.loads((output_root / RESOLVED_MEMBERSHIP_FILENAME).read_text())
    assert receipt["schema_version"] == RESOLVED_PACKED_MEMBERSHIP_SCHEMA_VERSION == 2
    resolved = resolve_editing_v2_active8_sources(
        candidate_materialization_dir=inputs["candidate_root"],
        split_assignment_path=inputs["split_path"],
        editing_corpus_contract_path=CORPUS_CONTRACT,
        lane_registry_path=output_root / LANE_REGISTRY_FILENAME,
        artifact_root=inputs["artifact_root"],
        membership_receipt_path=(output_root / RESOLVED_MEMBERSHIP_FILENAME),
    )
    assert len(resolved.shards) == 20
    assert {(shard.manifest_layer, shard.partition) for shard in resolved.shards} == {
        (lane, role) for lane in LANES for role in PARTITION_ROLES
    }

    for shard in resolved.shards:
        with gzip.open(shard.path, "rt") as handle:
            output_row = json.loads(next(handle))
        membership_shard = next(
            row
            for row in receipt["shards"]
            if row["data_lane"] == shard.manifest_layer and row["partition_role"] == shard.partition
        )
        member = membership_shard["entry_memberships"][0]
        original_index = int(member["candidate_id"].rsplit("-", 1)[1])
        original_path = inputs["artifact_root"] / f"original/shard_{original_index:03d}.jsonl.gz"
        with gzip.open(original_path, "rt") as handle:
            original_row = json.loads(next(handle))
        restored = copy.deepcopy(output_row)
        restored["trace"]["layer"] = original_row["trace"]["layer"]
        restored["trace"]["partition"] = original_row["trace"]["partition"]
        assert restored == original_row


def test_materialization_is_byte_deterministic_across_local_roots(
    tmp_path: Path,
) -> None:
    first_inputs = _inputs(tmp_path / "first")
    second_inputs = _inputs(tmp_path / "second")
    first, first_root = _materialize(first_inputs)
    second, second_root = _materialize(second_inputs)

    assert first["run_identity_sha256"] == second["run_identity_sha256"]
    assert first["manifest_sha256"] == second["manifest_sha256"]
    for relative in (
        ROLE_LANE_MATERIALIZATION_FILENAME,
        LANE_REGISTRY_FILENAME,
        RESOLVED_MEMBERSHIP_FILENAME,
    ):
        assert file_sha256(first_root / relative) == file_sha256(second_root / relative)


def test_non_envelope_mutation_fails_before_atomic_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path)
    from compose_v4.data import editing_v2_role_lane_packed_materializer as module

    original_rewrite = module._rewrite_envelope_only

    def corrupting_rewrite(entry, *, lane, role):
        rewritten = original_rewrite(entry, lane=lane, role=role)
        rewritten["trace"]["source_key"] = "mutated-source-key"
        return rewritten

    monkeypatch.setattr(module, "_rewrite_envelope_only", corrupting_rewrite)
    with pytest.raises(
        EditingV2RoleLaneMaterializationError,
        match="source_key",
    ):
        _materialize(inputs)
    output_parent = inputs["artifact_root"] / "editing_v2" / "role_lane_packed"
    assert not output_parent.exists() or not any(
        path.is_dir() and not path.name.startswith(".") for path in output_parent.iterdir()
    )


def test_source_hash_and_unselected_candidate_fail_closed(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path / "hash")
    source = inputs["artifact_root"] / "original/shard_000.jsonl.gz"
    source.write_bytes(source.read_bytes() + b"tamper")
    with pytest.raises(
        EditingV2RoleLaneMaterializationError,
        match="shard is absent or its SHA-256 disagrees",
    ):
        _materialize(inputs)

    inputs = _inputs(tmp_path / "unselected")
    assignment = json.loads(inputs["split_path"].read_text())
    assignment["candidate_resolutions"].pop()
    assignment["candidate_resolution_stream_sha256"] = canonical_sha256(
        assignment["candidate_resolutions"]
    )
    assignment = _self_hash(assignment, "assignment_sha256")
    _write_json(inputs["split_path"], assignment)
    with pytest.raises(
        EditingV2RoleLaneMaterializationError,
        match="unselected",
    ):
        _materialize(inputs)


def test_adapter_rejects_legacy_membership_v1(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    _, output_root = _materialize(inputs)
    receipt_path = output_root / RESOLVED_MEMBERSHIP_FILENAME
    receipt = json.loads(receipt_path.read_text())
    receipt["schema_version"] = 1
    receipt = _self_hash(receipt, "receipt_sha256")
    _write_json(receipt_path, receipt)

    with pytest.raises(
        EditingV2Active8SourceAdapterError,
        match="authority boundary",
    ):
        resolve_editing_v2_active8_sources(
            candidate_materialization_dir=inputs["candidate_root"],
            split_assignment_path=inputs["split_path"],
            editing_corpus_contract_path=CORPUS_CONTRACT,
            lane_registry_path=output_root / LANE_REGISTRY_FILENAME,
            artifact_root=inputs["artifact_root"],
            membership_receipt_path=receipt_path,
        )
