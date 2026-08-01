"""The Editing-V2 source adapter must prove exact pre-Active8 membership."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from compose_v4.data.editing_v2_active8_source_adapter import (
    ACTIVE8_ADMISSION_STATUS,
    CANDIDATE_AUTHORITY,
    REQUIRED_BLOCKERS,
    RESOLVED_PACKED_MEMBERSHIP_SCHEMA,
    RESOLVED_PACKED_MEMBERSHIP_SCHEMA_VERSION,
    RESOLVED_PACKED_MEMBERSHIP_STATUS,
    EditingV2Active8SourceAdapterError,
    resolve_editing_v2_active8_sources,
)
from compose_v4.data.editing_v2_candidate_provenance_bridge import (
    SOURCE_STREAM_SCHEMA,
    SOURCE_STREAM_SCHEMA_VERSION,
)
from compose_v4.data.editing_v2_lane_registry import (
    LANE_COMPLETION_FILENAME,
    LANE_COMPLETION_SCHEMA,
    LANE_COMPLETION_SCHEMA_VERSION,
    LANE_COMPLETION_STATUS,
    LANE_REGISTRY_SCHEMA,
    LANE_REGISTRY_SCHEMA_VERSION,
    LANE_REGISTRY_STATUS,
    canonical_sha256,
    editing_corpus_contract_identity,
    editing_v2_lane_definitions,
    file_sha256,
)
from compose_v4.data.editing_v2_packed_candidate_materializer import (
    CANDIDATE_HEADER_SCHEMA,
    CANDIDATE_HEADER_SCHEMA_VERSION,
    CANDIDATE_HEADER_STATUS,
    CANDIDATE_ROWS_FILENAME,
    MATERIALIZATION_FILENAME,
    MATERIALIZATION_SCHEMA,
    MATERIALIZATION_SCHEMA_VERSION,
    MATERIALIZATION_STATUS,
    canonical_json_bytes,
)
from compose_v4.data.editing_v2_split_assignment import (
    SPLIT_ASSIGNMENT_SCHEMA,
    SPLIT_ASSIGNMENT_STATUS,
    SPLIT_ASSIGNMENT_VERSION,
)
from compose_v4.data.editing_v2_split_census import PARTITION_ROLES
from compose_v4.data.packed_trace_store import (
    PACKED_STORE_SCHEMA,
    PACKED_STORE_SCHEMA_VERSION,
    manifest_path_for,
)

ROOT = Path(__file__).resolve().parents[1]
CORPUS_CONTRACT = ROOT / "configs" / "editing_corpus_v2_contract.json"
LANES = tuple(
    lane.lane_id for lane in editing_v2_lane_definitions(json.loads(CORPUS_CONTRACT.read_text()))
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _self_hash(value: dict[str, Any], field: str) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != field}
    return {**body, field: canonical_sha256(body)}


def _candidate_row(index: int, lane: str) -> dict[str, Any]:
    candidate_id = f"candidate-{index:03d}"
    original_shard_sha256 = hashlib.sha256(f"original-shard-{index}".encode()).hexdigest()
    original_manifest_sha256 = hashlib.sha256(f"original-manifest-{index}".encode()).hexdigest()
    original_overlay_sha256 = hashlib.sha256(f"original-overlay-{index}".encode()).hexdigest()
    source_asset_sha256 = hashlib.sha256(f"source-asset-{index}".encode()).hexdigest()
    address_body = {
        "source_asset_sha256": source_asset_sha256,
        "packed_shard_file_sha256": original_shard_sha256,
        "historical_provenance_overlay_file_sha256": (original_overlay_sha256),
        "entry_index": index,
        "trace_id": f"trace-{index:03d}",
        "layer": "legacy",
        "cache_partition": "legacy_train",
        "path_length": 1,
        "state_count": 2,
        "encoded_state_stream_sha256": hashlib.sha256(f"states-{index}".encode()).hexdigest(),
    }
    address_sha256 = canonical_sha256(address_body)
    row_body = {
        "schema": CANDIDATE_HEADER_SCHEMA,
        "schema_version": CANDIDATE_HEADER_SCHEMA_VERSION,
        "status": CANDIDATE_HEADER_STATUS,
        "training_authorized": False,
        "attempt_index": index,
        "candidate_id": candidate_id,
        "candidate_payload_sha256": hashlib.sha256(f"payload-{index}".encode()).hexdigest(),
        "disposition": "routed_candidate",
        "rejection": None,
        "source_kind": "fixture",
        "source_asset": {
            "source_asset_id": f"source-{index:03d}",
            "source_asset_path": f"/artifacts/original/source-{index:03d}",
            "source_asset_sha256": source_asset_sha256,
        },
        "packed_address": {
            "relative_path": f"original/shard-{index:03d}.jsonl.gz",
            "packed_shard_file_sha256": original_shard_sha256,
            "packed_shard_manifest_file_sha256": (original_manifest_sha256),
            "historical_provenance_overlay_file_sha256": (original_overlay_sha256),
            "entry_index": index,
            "trace_id": f"trace-{index:03d}",
            "layer": "legacy",
            "cache_partition": "legacy_train",
            "path_length": 1,
            "address_sha256": address_sha256,
        },
        "endpoint_identity": {
            "source_key": f"source-key-{index}",
            "target_key": f"target-key-{index}",
        },
        "groups": {
            "molecule_ids": [f"molecule-{index}"],
            "scaffold_ids": [f"scaffold-{index}"],
            "source_group_ids": [f"group-{index}"],
        },
        "partition_group_derivation": "explicit_partition_isolation_v1",
        "compiler_path_class": "fixture",
        "data_lane": lane,
        "lane_resolution": {"fixture": True},
        "routing_policy_sha256": "1" * 64,
        "evidence_profile_id": "fixture",
        "evidence_components": {"fixture": "fixture"},
        "evidence_sha256": "2" * 64,
        "operator_summary": {
            "operator_families": ["atom_restate"],
            "family_histogram": {"atom_restate": 1},
            "executor_histogram": {"atom_restate": 1},
            "action_stream_sha256": "3" * 64,
            "atom_count_delta": 0,
            "graph_cycle_rank_delta": 0,
            "step_delta_stream_sha256": "4" * 64,
        },
        "exact_states": {
            "encoding": "fixture",
            "n_slots": 40,
            "state_count": 2,
            "encoded_state_stream_sha256": address_body["encoded_state_stream_sha256"],
            "progress_address_stream_sha256": "5" * 64,
        },
        "trace_envelope_sha256": "6" * 64,
    }
    return {**row_body, "row_sha256": canonical_sha256(row_body)}


def _write_candidate_materialization(
    root: Path,
    rows: list[dict[str, Any]],
) -> tuple[Path, dict[str, Any]]:
    output = root / "candidates"
    output.mkdir()
    rows_path = output / CANDIDATE_ROWS_FILENAME
    row_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    rows_path.write_bytes(row_bytes)
    semantic = hashlib.sha256()
    addresses = hashlib.sha256()
    for row in rows:
        semantic.update(row["row_sha256"].encode())
        semantic.update(b"\n")
        addresses.update(row["packed_address"]["address_sha256"].encode())
        addresses.update(b"\n")
    lane_histogram = {lane: sum(row["data_lane"] == lane for row in rows) for lane in LANES}
    manifest_body = {
        "schema": MATERIALIZATION_SCHEMA,
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "status": MATERIALIZATION_STATUS,
        "training_authorized": False,
        "rows": {
            "relative_path": CANDIDATE_ROWS_FILENAME,
            "file_sha256": file_sha256(rows_path),
            "semantic_sha256": semantic.hexdigest(),
            "address_stream_sha256": addresses.hexdigest(),
            "totals": {
                "rows": len(rows),
                "routed_rows": len(rows),
                "rejected_rows": 0,
                "states": 2 * len(rows),
                "transitions": len(rows),
                "atom_count_delta_sum": 0,
                "graph_cycle_rank_delta_sum": 0,
                "family_histogram": {"atom_restate": len(rows)},
                "routed_family_histogram": {"atom_restate": len(rows)},
                "rejected_family_histogram": {},
                "lane_histogram": lane_histogram,
                "rejection_reason_histogram": {},
                "partition_group_derivation_histogram": {
                    "explicit_partition_isolation_v1": len(rows)
                },
            },
        },
    }
    manifest = {
        **manifest_body,
        "manifest_sha256": canonical_sha256(manifest_body),
    }
    _write_json(output / MATERIALIZATION_FILENAME, manifest)
    return output, manifest


def _write_split_assignment(
    root: Path,
    rows: list[dict[str, Any]],
    *,
    candidate_root: Path,
    candidate_manifest: dict[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, str]]:
    role_by_candidate = {
        row["candidate_id"]: PARTITION_ROLES[row["attempt_index"] % len(PARTITION_ROLES)]
        for row in rows
    }
    resolutions = [
        {
            "candidate_id": candidate_id,
            "component_id": f"component-{candidate_id}",
            "status": "assigned",
            "assigned_role": role,
            "source_endpoint_role": role,
            "target_endpoint_role": role,
        }
        for candidate_id, role in sorted(role_by_candidate.items())
    ]
    source_stream_body = {
        "schema": SOURCE_STREAM_SCHEMA,
        "schema_version": SOURCE_STREAM_SCHEMA_VERSION,
        "nonempty_jsonl_rows": len(rows),
        "candidate_materialization": {
            "manifest_file_sha256": file_sha256(candidate_root / MATERIALIZATION_FILENAME),
            "manifest_sha256": candidate_manifest["manifest_sha256"],
            "rows_file_sha256": candidate_manifest["rows"]["file_sha256"],
            "rows_semantic_sha256": candidate_manifest["rows"]["semantic_sha256"],
            "address_stream_sha256": candidate_manifest["rows"]["address_stream_sha256"],
        },
        "provenance_registry": {
            "file_sha256": "a" * 64,
            "registry_sha256": "b" * 64,
        },
        "candidate_audit_ledger": {
            "file_sha256": "c" * 64,
            "semantic_sha256": "d" * 64,
            "rows_sha256": "e" * 64,
        },
        "split_candidates": {
            "file_sha256": "f" * 64,
            "semantic_sha256": "0" * 64,
        },
    }
    source_stream = {
        **source_stream_body,
        "source_stream_sha256": canonical_sha256(source_stream_body),
    }
    body = {
        "schema": SPLIT_ASSIGNMENT_SCHEMA,
        "schema_version": SPLIT_ASSIGNMENT_VERSION,
        "status": SPLIT_ASSIGNMENT_STATUS,
        "training_authorized": False,
        "split_assignment_selected": True,
        "split_assignment_authorized": False,
        "blockers": ["physical_lane_shards.not_built"],
        "policy": {"fixture": True},
        "policy_sha256": "7" * 64,
        "source_stream": source_stream,
        "source_census_sha256": "8" * 64,
        "source_component_inventory_sha256": "9" * 64,
        "partition_roles": list(PARTITION_ROLES),
        "observed_lanes": list(LANES),
        "total_mass_units": len(rows),
        "total_mass_units_by_lane": {lane: len(PARTITION_ROLES) for lane in LANES},
        "role_summaries": {},
        "lane_absolute_ratio_deviations": {},
        "gate_results": {
            "all_roles_nonempty": True,
            "all_observed_lanes_present_in_every_role": True,
            "overall_mass_ratio_within_limit": True,
            "per_lane_mass_ratio_within_limit": True,
        },
        "component_assignments": [],
        "candidate_resolutions": resolutions,
        "candidate_resolution_stream_sha256": canonical_sha256(resolutions),
        "sealed_final_test_policy": "fixture sealed",
    }
    assignment = {
        **body,
        "assignment_sha256": canonical_sha256(body),
    }
    path = root / "split_assignment.json"
    _write_json(path, assignment)
    return path, assignment, role_by_candidate


def _write_packed_output(
    path: Path,
    *,
    lane: str,
    role: str,
    candidate_id: str,
    with_overlay: bool,
) -> tuple[dict[str, Any], str, str | None, dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "trace": {
            "trace_id": candidate_id,
            "layer": lane,
            "partition": role,
            "path_length": 1,
            "source_key": f"{candidate_id}-source",
            "target_key": f"{candidate_id}-target",
        },
        "states": [{"fixture": 0}, {"fixture": 1}],
    }
    with (
        path.open("wb") as raw,
        gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            mtime=0,
        ) as compressed,
    ):
        compressed.write(canonical_json_bytes(entry) + b"\n")
    manifest = {
        "schema": PACKED_STORE_SCHEMA,
        "schema_version": PACKED_STORE_SCHEMA_VERSION,
        "sampler_contract": {"fixture": True},
        "entries": 1,
        "states": 2,
        "provenance": {"fixture": True},
    }
    manifest_path = manifest_path_for(path)
    _write_json(manifest_path, manifest)
    overlay_sha256 = None
    if with_overlay:
        overlay_path = Path(f"{path}.provenance.json")
        _write_json(
            overlay_path,
            {
                "schema": "compose.data.provenance_overlay",
                "schema_version": 1,
                "fixture": True,
            },
        )
        overlay_sha256 = file_sha256(overlay_path)
    return manifest, file_sha256(manifest_path), overlay_sha256, entry


def _write_lane_registry_and_outputs(
    root: Path,
    rows: list[dict[str, Any]],
    role_by_candidate: dict[str, str],
) -> tuple[Path, Path, dict[tuple[str, str], dict[str, Any]]]:
    artifact_root = root / "artifacts"
    artifact_root.mkdir()
    contract = json.loads(CORPUS_CONTRACT.read_text())
    contract_identity = editing_corpus_contract_identity(
        contract,
        contract_file_sha256=file_sha256(CORPUS_CONTRACT),
    )
    by_cell = {(row["data_lane"], role_by_candidate[row["candidate_id"]]): row for row in rows}
    registry_lanes = []
    physical: dict[tuple[str, str], dict[str, Any]] = {}
    for lane_index, lane_definition in enumerate(editing_v2_lane_definitions(contract)):
        lane = lane_definition.lane_id
        provisional_shards = []
        for role_index, role in enumerate(PARTITION_ROLES):
            row = by_cell[(lane, role)]
            relative_path = f"{role}/shard_0000.jsonl.gz"
            provisional_shards.append(
                {
                    "partition": role,
                    "relative_path": relative_path,
                    "candidate_id": row["candidate_id"],
                    "with_overlay": (lane_index + role_index) % 2 == 0,
                }
            )
        # The content-addressed root depends on final bytes, so write below a
        # temporary root, compute final inventory, then move into its identity.
        temporary_root = artifact_root / "editing_v2" / "staging" / lane
        completed_shards = []
        for shard in provisional_shards:
            local_path = temporary_root / shard["relative_path"]
            _, manifest_sha, overlay_sha, output_entry = _write_packed_output(
                local_path,
                lane=lane,
                role=shard["partition"],
                candidate_id=shard["candidate_id"],
                with_overlay=shard["with_overlay"],
            )
            completed_shards.append(
                {
                    "partition": shard["partition"],
                    "relative_path": shard["relative_path"],
                    "sha256": file_sha256(local_path),
                    "bytes": local_path.stat().st_size,
                    "records": 1,
                    "states": 2,
                    "transitions": 1,
                    "_manifest_sha256": manifest_sha,
                    "_overlay_sha256": overlay_sha,
                    "_candidate_id": shard["candidate_id"],
                    "_output_entry": output_entry,
                }
            )
        inventory_rows = [
            {key: value for key, value in shard.items() if not key.startswith("_")}
            for shard in completed_shards
        ]
        inventory_sha256 = canonical_sha256(inventory_rows)
        declared_root = f"/artifacts/editing_v2/lanes/{lane}/{inventory_sha256}"
        final_root = artifact_root / PurePosixPath(declared_root).relative_to("/artifacts")
        final_root.parent.mkdir(parents=True, exist_ok=True)
        temporary_root.rename(final_root)
        by_partition = {
            role: {
                "records": 1,
                "states": 2,
                "transitions": 1,
                "shards": 1,
            }
            for role in PARTITION_ROLES
        }
        completion_body = {
            "schema": LANE_COMPLETION_SCHEMA,
            "schema_version": LANE_COMPLETION_SCHEMA_VERSION,
            "status": LANE_COMPLETION_STATUS,
            "training_authorized": False,
            "editing_corpus_contract": contract_identity,
            "lane_id": lane,
            "admissible_evidence_profiles": list(lane_definition.admissible_evidence_profiles),
            "reserved_evidence_profiles": list(lane_definition.reserved_evidence_profiles),
            "declared_root": declared_root,
            "partition_roles": list(PARTITION_ROLES),
            "counts": {
                "records": 4,
                "states": 8,
                "transitions": 4,
                "shards": 4,
                "by_partition": by_partition,
            },
            "shards": inventory_rows,
            "shard_inventory_sha256": inventory_sha256,
        }
        completion = {
            **completion_body,
            "completion_sha256": canonical_sha256(completion_body),
        }
        completion_path = final_root / LANE_COMPLETION_FILENAME
        _write_json(completion_path, completion)
        registry_lanes.append(
            {
                "lane_id": lane,
                "admissible_evidence_profiles": list(lane_definition.admissible_evidence_profiles),
                "reserved_evidence_profiles": list(lane_definition.reserved_evidence_profiles),
                "declared_root": declared_root,
                "completion_manifest_path": (f"{declared_root}/{LANE_COMPLETION_FILENAME}"),
                "completion_manifest_sha256": file_sha256(completion_path),
            }
        )
        for shard in completed_shards:
            artifact_path = f"{declared_root}/{shard['relative_path']}"
            physical[(lane, shard["partition"])] = {
                **shard,
                "artifact_path": artifact_path,
                "local_path": (
                    artifact_root / PurePosixPath(artifact_path).relative_to("/artifacts")
                ),
            }
    registry_body = {
        "schema": LANE_REGISTRY_SCHEMA,
        "schema_version": LANE_REGISTRY_SCHEMA_VERSION,
        "status": LANE_REGISTRY_STATUS,
        "training_authorized": False,
        "editing_corpus_contract": contract_identity,
        "partition_roles": list(PARTITION_ROLES),
        "lanes": registry_lanes,
    }
    registry = {
        **registry_body,
        "registry_sha256": canonical_sha256(registry_body),
    }
    registry_path = root / "lane_registry.json"
    _write_json(registry_path, registry)
    return registry_path, artifact_root, physical


def _receipt_inputs(
    *,
    candidate_root: Path,
    candidate_manifest: dict[str, Any],
    split_path: Path,
    assignment: dict[str, Any],
    registry_path: Path,
) -> dict[str, Any]:
    registry = json.loads(registry_path.read_text())
    completion_file_hashes = {}
    completion_hashes = {}
    inventory_hashes = {}
    for lane in registry["lanes"]:
        completion_path = (
            registry_path.parent
            / "artifacts"
            / PurePosixPath(lane["completion_manifest_path"]).relative_to("/artifacts")
        )
        completion = json.loads(completion_path.read_text())
        completion_file_hashes[lane["lane_id"]] = file_sha256(completion_path)
        completion_hashes[lane["lane_id"]] = completion["completion_sha256"]
        inventory_hashes[lane["lane_id"]] = completion["shard_inventory_sha256"]
    rows = candidate_manifest["rows"]
    return {
        "candidate_materialization": {
            "manifest_file_sha256": file_sha256(candidate_root / MATERIALIZATION_FILENAME),
            "manifest_sha256": candidate_manifest["manifest_sha256"],
            "rows_file_sha256": rows["file_sha256"],
            "rows_semantic_sha256": rows["semantic_sha256"],
            "address_stream_sha256": rows["address_stream_sha256"],
        },
        "split_assignment": {
            "file_sha256": file_sha256(split_path),
            "assignment_sha256": assignment["assignment_sha256"],
            "candidate_resolution_stream_sha256": assignment["candidate_resolution_stream_sha256"],
            "source_stream_sha256": assignment["source_stream"]["source_stream_sha256"],
        },
        "lane_registry": {
            "registry_file_sha256": file_sha256(registry_path),
            "registry_sha256": registry["registry_sha256"],
            "editing_corpus_contract_file_sha256": file_sha256(CORPUS_CONTRACT),
            "editing_corpus_contract_id": registry["editing_corpus_contract"]["contract_id"],
            "lane_completion_manifest_file_sha256": completion_file_hashes,
            "lane_completion_sha256": completion_hashes,
            "lane_shard_inventory_sha256": inventory_hashes,
        },
    }


def _write_membership_receipt(
    root: Path,
    *,
    rows: list[dict[str, Any]],
    candidate_root: Path,
    candidate_manifest: dict[str, Any],
    split_path: Path,
    assignment: dict[str, Any],
    registry_path: Path,
    physical: dict[tuple[str, str], dict[str, Any]],
) -> tuple[Path, dict[str, Any]]:
    role_by_candidate = {
        resolution["candidate_id"]: resolution["assigned_role"]
        for resolution in assignment["candidate_resolutions"]
    }
    row_by_cell = {(row["data_lane"], role_by_candidate[row["candidate_id"]]): row for row in rows}
    shards = []
    for lane in LANES:
        for role in PARTITION_ROLES:
            row = row_by_cell[(lane, role)]
            source = physical[(lane, role)]
            artifact_path = source["artifact_path"]
            overlay_sha = source["_overlay_sha256"]
            output_address_body = {
                "packed_shard_file_sha256": source["sha256"],
                "packed_shard_manifest_file_sha256": source["_manifest_sha256"],
                "packed_provenance_overlay_file_sha256": overlay_sha,
                "packed_shard_name": source["local_path"].name,
                "entry_index": 0,
                "trace_id": source["_output_entry"]["trace"]["trace_id"],
                "layer": lane,
                "partition": role,
                "source_key": source["_output_entry"]["trace"]["source_key"],
                "target_key": source["_output_entry"]["trace"]["target_key"],
                "path_length": 1,
            }
            shards.append(
                {
                    "data_lane": lane,
                    "partition_role": role,
                    "relative_path": source["relative_path"],
                    "artifact_path": artifact_path,
                    "file_sha256": source["sha256"],
                    "packed_manifest_artifact_path": str(
                        PurePosixPath(artifact_path).with_suffix(".manifest.json")
                    ),
                    "packed_manifest_file_sha256": source["_manifest_sha256"],
                    "packed_provenance_overlay_artifact_path": (
                        None if overlay_sha is None else f"{artifact_path}.provenance.json"
                    ),
                    "packed_provenance_overlay_file_sha256": overlay_sha,
                    "entry_memberships": [
                        {
                            "output_entry_index": 0,
                            "candidate_id": row["candidate_id"],
                            "original_packed_address": {
                                "source_asset": row["source_asset"],
                                "packed_address": row["packed_address"],
                            },
                            "original_packed_row_sha256": hashlib.sha256(
                                f"original-row-{row['candidate_id']}".encode()
                            ).hexdigest(),
                            "output_packed_row_sha256": canonical_sha256(source["_output_entry"]),
                            "output_packed_address": {
                                **output_address_body,
                                "address_sha256": canonical_sha256(output_address_body),
                            },
                        }
                    ],
                }
            )
    body = {
        "schema": RESOLVED_PACKED_MEMBERSHIP_SCHEMA,
        "schema_version": RESOLVED_PACKED_MEMBERSHIP_SCHEMA_VERSION,
        "status": RESOLVED_PACKED_MEMBERSHIP_STATUS,
        "training_authorized": False,
        "candidate_materialization_authority": CANDIDATE_AUTHORITY,
        "active8_admission_status": ACTIVE8_ADMISSION_STATUS,
        "membership_complete": True,
        "blockers": list(REQUIRED_BLOCKERS),
        "inputs": _receipt_inputs(
            candidate_root=candidate_root,
            candidate_manifest=candidate_manifest,
            split_path=split_path,
            assignment=assignment,
            registry_path=registry_path,
        ),
        "shards": shards,
    }
    receipt = {**body, "receipt_sha256": canonical_sha256(body)}
    path = root / "resolved_membership.json"
    _write_json(path, receipt)
    return path, receipt


def _fixture(tmp_path: Path) -> dict[str, Any]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows = [
        _candidate_row(index, lane)
        for index, lane in enumerate(lane for lane in LANES for _ in PARTITION_ROLES)
    ]
    candidate_root, candidate_manifest = _write_candidate_materialization(
        tmp_path,
        rows,
    )
    split_path, assignment, roles = _write_split_assignment(
        tmp_path,
        rows,
        candidate_root=candidate_root,
        candidate_manifest=candidate_manifest,
    )
    registry_path, artifact_root, physical = _write_lane_registry_and_outputs(tmp_path, rows, roles)
    receipt_path, receipt = _write_membership_receipt(
        tmp_path,
        rows=rows,
        candidate_root=candidate_root,
        candidate_manifest=candidate_manifest,
        split_path=split_path,
        assignment=assignment,
        registry_path=registry_path,
        physical=physical,
    )
    return {
        "rows": rows,
        "candidate_root": candidate_root,
        "candidate_manifest": candidate_manifest,
        "split_path": split_path,
        "assignment": assignment,
        "registry_path": registry_path,
        "artifact_root": artifact_root,
        "physical": physical,
        "receipt_path": receipt_path,
        "receipt": receipt,
    }


def _resolve(fixture: dict[str, Any]):
    return resolve_editing_v2_active8_sources(
        candidate_materialization_dir=fixture["candidate_root"],
        split_assignment_path=fixture["split_path"],
        editing_corpus_contract_path=CORPUS_CONTRACT,
        lane_registry_path=fixture["registry_path"],
        artifact_root=fixture["artifact_root"],
        membership_receipt_path=fixture["receipt_path"],
    )


def _rewrite_receipt(
    fixture: dict[str, Any],
    mutate,
) -> None:
    receipt = copy.deepcopy(fixture["receipt"])
    mutate(receipt)
    receipt = _self_hash(receipt, "receipt_sha256")
    _write_json(fixture["receipt_path"], receipt)


def test_resolves_exact_five_lane_four_role_sources_without_authorizing_active8(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    resolved = _resolve(fixture)

    assert len(resolved.bindings) == len(LANES) * len(PARTITION_ROLES)
    assert {(shard.manifest_layer, shard.partition) for shard in resolved.shards} == {
        (lane, role) for lane in LANES for role in PARTITION_ROLES
    }
    assert all(shard.manifest_layer == shard.envelope_layer for shard in resolved.shards)
    assert resolved.source_manifest["active8_admission_status"] == ACTIVE8_ADMISSION_STATUS
    assert resolved.source_manifest["training_authorized"] is False
    assert resolved.candidate_provenance_source_stream == fixture["assignment"]["source_stream"]
    assert (
        resolved.candidate_provenance_source_stream["source_stream_sha256"]
        == fixture["assignment"]["source_stream"]["source_stream_sha256"]
    )
    assert any(
        binding.packed_provenance_overlay_file_sha256 is None for binding in resolved.bindings
    )
    assert any(
        binding.packed_provenance_overlay_file_sha256 is not None for binding in resolved.bindings
    )
    assert sum(len(binding.candidate_ids) for binding in resolved.bindings) == 20


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda receipt: receipt["shards"][0]["entry_memberships"].clear(),
            "no entry memberships",
        ),
        (
            lambda receipt: receipt["shards"][0]["entry_memberships"].append(
                copy.deepcopy(receipt["shards"][0]["entry_memberships"][0])
            ),
            "contiguous|more than once|row count disagrees",
        ),
        (
            lambda receipt: receipt["shards"][0]["entry_memberships"][0].update(
                {"candidate_id": receipt["shards"][1]["entry_memberships"][0]["candidate_id"]}
            ),
            "wrong lane/role|more than once",
        ),
        (
            lambda receipt: receipt["shards"][0]["entry_memberships"][0]["original_packed_address"][
                "packed_address"
            ].update({"address_sha256": "0" * 64}),
            "original packed address disagrees",
        ),
    ],
)
def test_missing_duplicate_unselected_or_mutated_addresses_fail_closed(
    tmp_path: Path,
    mutation,
    match: str,
) -> None:
    fixture = _fixture(tmp_path)
    _rewrite_receipt(fixture, mutation)

    with pytest.raises(EditingV2Active8SourceAdapterError, match=match):
        _resolve(fixture)


def test_full_source_manifest_and_optional_overlay_hashes_are_enforced(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    def mutate(receipt):
        receipt["shards"][0]["packed_manifest_file_sha256"] = "0" * 64

    _rewrite_receipt(fixture, mutate)
    with pytest.raises(
        EditingV2Active8SourceAdapterError,
        match="packed manifest SHA-256 disagrees",
    ):
        _resolve(fixture)

    fixture = _fixture(tmp_path / "second")

    def undeclare_overlay(receipt):
        shard = next(
            shard
            for shard in receipt["shards"]
            if shard["packed_provenance_overlay_file_sha256"] is not None
        )
        shard["packed_provenance_overlay_artifact_path"] = None
        shard["packed_provenance_overlay_file_sha256"] = None

    _rewrite_receipt(fixture, undeclare_overlay)
    with pytest.raises(
        EditingV2Active8SourceAdapterError,
        match="overlay exists but is unbound",
    ):
        _resolve(fixture)


def test_candidate_and_split_input_identity_drift_fail_closed(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    def mutate(receipt):
        receipt["inputs"]["candidate_materialization"]["address_stream_sha256"] = "0" * 64

    _rewrite_receipt(fixture, mutate)
    with pytest.raises(
        EditingV2Active8SourceAdapterError,
        match="input identities disagree",
    ):
        _resolve(fixture)

    fixture = _fixture(tmp_path / "second")
    assignment = json.loads(fixture["split_path"].read_text())
    assignment["candidate_resolutions"].pop()
    assignment["candidate_resolution_stream_sha256"] = canonical_sha256(
        assignment["candidate_resolutions"]
    )
    assignment = _self_hash(assignment, "assignment_sha256")
    _write_json(fixture["split_path"], assignment)
    with pytest.raises(
        EditingV2Active8SourceAdapterError,
        match="candidate provenance source-stream identity disagrees",
    ):
        _resolve(fixture)


def test_lane_or_role_drift_and_premature_authority_fail_closed(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    def wrong_role(receipt):
        receipt["shards"][0]["partition_role"] = "validation"

    _rewrite_receipt(fixture, wrong_role)
    with pytest.raises(
        EditingV2Active8SourceAdapterError,
        match="disagrees with lane registry",
    ):
        _resolve(fixture)

    fixture = _fixture(tmp_path / "second")

    def authorize(receipt):
        receipt["active8_admission_status"] = "PASSED"
        receipt["training_authorized"] = True

    _rewrite_receipt(fixture, authorize)
    with pytest.raises(
        EditingV2Active8SourceAdapterError,
        match="authority boundary",
    ):
        _resolve(fixture)
