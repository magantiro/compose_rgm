from __future__ import annotations

import copy
import hashlib
import json
import platform
from pathlib import Path

import pytest

from compose_v4.data.editing_corpus_contract import load_editing_corpus_contract
from compose_v4.data.editing_v2_candidate_provenance_bridge import (
    AUDIT_LEDGER_FILENAME,
    BRIDGE_MANIFEST_FILENAME,
    SOURCE_STREAM_SCHEMA,
    SPLIT_ROWS_FILENAME,
    EditingV2CandidateProvenanceBridgeError,
    build_candidate_provenance_registry,
    materialize_candidate_provenance_bridge,
    validate_candidate_provenance_bridge,
    write_candidate_provenance_registry,
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
    canonical_sha256,
    file_sha256,
)
from compose_v4.data.editing_v2_split_census import (
    CANDIDATE_ROW_SCHEMA_VERSION,
    IDENTITY_DEFINITION_CONTRACT_SCHEMA,
    IDENTITY_DEFINITION_CONTRACT_SCHEMA_VERSION,
    RELATIONSHIP_NAMESPACE_SCHEMA,
    RELATIONSHIP_NAMESPACE_SCHEMA_VERSION,
    build_split_component_census,
    default_split_census_policy,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs" / "editing_corpus_v2_contract.json"
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64


def _profile_components(profile_id: str) -> dict[str, str]:
    contract = load_editing_corpus_contract(CONTRACT_PATH)
    return dict(
        next(
            profile
            for profile in contract["evidence_component_contract"]["profiles"]
            if profile["id"] == profile_id
        )["components"]
    )


def _header(index: int, *, routed: bool) -> dict:
    address_sha = canonical_sha256({"fixture-address": index})
    candidate_id = f"candidate-{index}"
    source_asset = {
        "source_asset_id": "fixture-source",
        "source_asset_path": "packed/fixture-source.jsonl.gz",
        "source_asset_sha256": SHA_A,
    }
    packed_address = {
        "relative_path": "packed/fixture-source.jsonl.gz",
        "packed_shard_file_sha256": SHA_A,
        "packed_shard_manifest_file_sha256": SHA_B,
        "historical_provenance_overlay_file_sha256": SHA_C,
        "entry_index": index,
        "trace_id": f"trace-{index}",
        "layer": "mmp_analogue",
        "cache_partition": "legacy-train",
        "path_length": 1,
        "address_sha256": address_sha,
    }
    lane = "operator_aware_real_endpoint" if routed else None
    resolution = (
        {
            "data_lane": lane,
            "evidence_profile_id": "inferred_relation_compiled_path",
            "route_reason": "direct_non_topology_active8",
            "routing_policy_sha256": SHA_B,
            "resolver_identity_sha256": SHA_C,
            "receipt_sha256": canonical_sha256({"route": candidate_id}),
        }
        if routed
        else None
    )
    rejection = (
        None
        if routed
        else {
            "code": "operator.not_in_active8_support",
            "attempts": [
                {
                    "step_index": 0,
                    "family": "ring_system_grow",
                    "reason": "disabled_family",
                }
            ],
        }
    )
    body = {
        "schema": CANDIDATE_HEADER_SCHEMA,
        "schema_version": CANDIDATE_HEADER_SCHEMA_VERSION,
        "status": CANDIDATE_HEADER_STATUS,
        "training_authorized": False,
        "attempt_index": index,
        "candidate_id": candidate_id,
        "candidate_payload_sha256": canonical_sha256({"payload": index}),
        "disposition": "routed_candidate" if routed else "rejected_candidate",
        "rejection": rejection,
        "source_kind": "inferred_real_endpoint_pair",
        "source_asset": source_asset,
        "packed_address": packed_address,
        "endpoint_identity": {
            "source_key": f"source-molecule-{index}",
            "target_key": f"target-molecule-{index}",
        },
        "groups": {
            "molecule_ids": [f"source-molecule-{index}", f"target-molecule-{index}"],
            "scaffold_ids": ["fixture-scaffold"],
            "source_group_ids": ["fixture-source-group"],
        },
        "compiler_path_class": "direct_atom_restate",
        "data_lane": lane,
        "lane_resolution": resolution,
        "routing_policy_sha256": SHA_B,
        "evidence_profile_id": "inferred_relation_compiled_path",
        "evidence_components": _profile_components("inferred_relation_compiled_path"),
        "evidence_sha256": canonical_sha256({"evidence": index}),
        "operator_summary": {
            "operator_families": ["atom_restate" if routed else "ring_system_grow"],
            "family_histogram": {"atom_restate" if routed else "ring_system_grow": 1},
            "executor_histogram": {"atom_restate" if routed else "ring_system_grow": 1},
            "action_stream_sha256": canonical_sha256({"actions": index}),
            "atom_count_delta": 0,
            "graph_cycle_rank_delta": 0,
            "step_delta_stream_sha256": canonical_sha256({"deltas": index}),
        },
        "exact_states": {
            "state_count": 2,
            "n_slots": 40,
            "encoded_state_stream_sha256": canonical_sha256({"states": index}),
            "encoded_state_byte_count": 10,
        },
        "trace_envelope_sha256": canonical_sha256({"trace": index}),
    }
    return {**body, "row_sha256": canonical_sha256(body)}


def _candidate_materialization(tmp_path: Path) -> Path:
    candidate_root = tmp_path / "candidates"
    candidate_root.mkdir()
    rows = [_header(0, routed=True), _header(1, routed=False)]
    rows_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    (candidate_root / CANDIDATE_ROWS_FILENAME).write_bytes(rows_bytes)
    semantic = hashlib.sha256()
    addresses = hashlib.sha256()
    for row in rows:
        semantic.update(row["row_sha256"].encode("ascii") + b"\n")
        addresses.update(
            row["packed_address"]["address_sha256"].encode("ascii") + b"\n"
        )
    totals = {
        "rows": 2,
        "routed_rows": 1,
        "rejected_rows": 1,
        "states": 4,
        "transitions": 2,
        "atom_count_delta_sum": 0,
        "graph_cycle_rank_delta_sum": 0,
        "family_histogram": {"atom_restate": 1, "ring_system_grow": 1},
        "routed_family_histogram": {"atom_restate": 1},
        "rejected_family_histogram": {"ring_system_grow": 1},
        "lane_histogram": {"operator_aware_real_endpoint": 1},
        "rejection_reason_histogram": {"disabled_family": 1},
    }
    body = {
        "schema": MATERIALIZATION_SCHEMA,
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "status": MATERIALIZATION_STATUS,
        "training_authorized": False,
        "implementation": {"file_sha256": SHA_C},
        "inputs": {
            "source_manifest": {
                "file_sha256": SHA_D,
                "manifest_sha256": SHA_E,
            },
            "routing_policy": {
                "file_sha256": SHA_A,
                "semantic_sha256": SHA_B,
            },
        },
        "sources": [
            {
                "source_asset_id": "fixture-source",
                "source_asset_path": "packed/fixture-source.jsonl.gz",
                "source_asset_sha256": SHA_A,
            }
        ],
        "rows": {
            "relative_path": CANDIDATE_ROWS_FILENAME,
            "file_sha256": hashlib.sha256(rows_bytes).hexdigest(),
            "semantic_sha256": semantic.hexdigest(),
            "address_stream_sha256": addresses.hexdigest(),
            "totals": totals,
        },
    }
    manifest = {**body, "manifest_sha256": canonical_sha256(body)}
    (candidate_root / MATERIALIZATION_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return candidate_root


def _identity_definitions() -> dict:
    return {
        "schema": IDENTITY_DEFINITION_CONTRACT_SCHEMA,
        "schema_version": IDENTITY_DEFINITION_CONTRACT_SCHEMA_VERSION,
        "exact_molecule": {
            "definition_id": "fixture-exact-molecule-v1",
            "implementation_sha256": SHA_A,
            "identity_namespace": "fixture-exact-molecule",
        },
        "partition_scaffold": {
            "definition_id": "fixture-partition-scaffold-v1",
            "implementation_sha256": SHA_B,
            "identity_namespace": "fixture-partition-scaffold",
        },
        "source_group": {
            "definition_id": "fixture-source-group-v1",
            "implementation_sha256": SHA_C,
            "identity_namespace": "fixture-source-group",
        },
    }


def _evidence_identity() -> dict:
    return {
        "definition_id": "fixture-header-evidence-reference-v1",
        "implementation_sha256": SHA_D,
        "source_record_id_fields": [
            "packed_address.address_sha256",
            "packed_address.trace_id",
            "source_asset.source_asset_id",
        ],
        "component_reference_fields": {
            "source_endpoint": ["endpoint_identity.source_key"],
            "target_endpoint": ["endpoint_identity.target_key"],
            "pair_relationship": [
                "endpoint_identity.source_key",
                "endpoint_identity.target_key",
            ],
            "path": ["packed_address.trace_id", "trace_envelope_sha256"],
            "intermediates": ["exact_states.encoded_state_stream_sha256"],
            "action_sequence": ["operator_summary.action_stream_sha256"],
        },
    }


def _source_assets() -> list[dict]:
    return [
        {
            "source_asset_id": "fixture-source",
            "source_asset_path": "packed/fixture-source.jsonl.gz",
            "source_asset_sha256": SHA_A,
            "source_version": "fixture-v1",
            "access_basis": "test fixture",
            "source_group_namespace": {
                "schema": RELATIONSHIP_NAMESPACE_SCHEMA,
                "schema_version": RELATIONSHIP_NAMESPACE_SCHEMA_VERSION,
                "relationship_type": "source_group",
                "namespace_id": "fixture-source-groups",
                "source_asset_id": "fixture-source",
                "source_asset_sha256": SHA_A,
                "definition_id": "fixture-source-group-v1",
                "implementation_sha256": SHA_C,
                "cross_lane_sharing_authorized": True,
            },
        }
    ]


def _registry(candidate_root: Path) -> dict:
    return build_candidate_provenance_registry(
        candidate_root=candidate_root,
        editing_corpus_contract_path=CONTRACT_PATH,
        compiler_identity={
            "implementation_sha256": SHA_A,
            "config_sha256": SHA_B,
            "operator_contract_sha256": SHA_C,
            "canonicalizer_sha256": SHA_D,
        },
        policy_bindings={
            "mapping_policy_sha256": SHA_D,
            "metric_policy_sha256": SHA_E,
        },
        evidence_identity=_evidence_identity(),
        identity_definitions=_identity_definitions(),
        source_assets=_source_assets(),
    )


def _materialize(tmp_path: Path) -> tuple[Path, Path, Path, dict]:
    candidate_root = _candidate_materialization(tmp_path)
    registry_path = tmp_path / "registry.json"
    write_candidate_provenance_registry(_registry(candidate_root), registry_path)
    output = tmp_path / "bridge"
    manifest = materialize_candidate_provenance_bridge(
        candidate_root=candidate_root,
        provenance_registry_path=registry_path,
        editing_corpus_contract_path=CONTRACT_PATH,
        output_dir=output,
    )
    return candidate_root, registry_path, output, manifest


def test_bridge_preserves_rejections_and_only_routes_unit_mass_split_rows(
    tmp_path: Path,
) -> None:
    candidate_root, registry_path, output, manifest = _materialize(tmp_path)

    assert manifest["counts"] == {
        "attempted": 2,
        "routed": 1,
        "rejected": 1,
        "split_rows": 1,
    }
    ledger = json.loads((output / AUDIT_LEDGER_FILENAME).read_bytes())
    assert [row["disposition"] for row in ledger["rows"]] == [
        "compiled_candidate",
        "rejected_candidate",
    ]
    assert ledger["rows"][1]["rejection"]["code"] == ("operator.not_in_active8_support")
    assert ledger["compiler_identity"] == {
        "implementation_sha256": SHA_A,
        "config_sha256": SHA_B,
        "operator_contract_sha256": SHA_C,
        "canonicalizer_sha256": SHA_D,
    }
    assert ledger["rows"][0]["policy_bindings"] == {
        "mapping_policy_sha256": SHA_D,
        "metric_policy_sha256": SHA_E,
        "unresolved_mapping_policies": [],
        "unresolved_metric_policies": [],
    }
    assert ledger["rows"][0]["evidence"]["source_endpoint"]["reference_ids"] == [
        "source-molecule-0"
    ]
    assert ledger["rows"][0]["evidence"]["action_sequence"]["reference_ids"] == [
        _header(0, routed=True)["operator_summary"]["action_stream_sha256"]
    ]

    split_rows = [
        json.loads(line)
        for line in (output / SPLIT_ROWS_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(split_rows) == 1
    row = split_rows[0]
    assert row["schema_version"] == CANDIDATE_ROW_SCHEMA_VERSION == 4
    assert row["candidate_id"] == "candidate-0"
    assert row["mass_units"] == 1
    assert row["source_group_ids"] == ["fixture-source-group"]
    assert row["source_group_namespace"]["cross_lane_sharing_authorized"] is True
    assert row["document_group_id"] is None
    assert row["document_provenance"] is None
    assert row["series_group_id"] is None
    assert row["series_provenance"] is None
    assert row["candidate_ledger_row_sha256"] == ledger["rows"][0]["row_sha256"]
    header = _header(0, routed=True)
    assert header["candidate_payload_sha256"] != header["row_sha256"]
    assert row["candidate_envelope_sha256"] == header["candidate_payload_sha256"]
    assert row["metadata"]["candidate_header_row_sha256"] == header["row_sha256"]

    validated = validate_candidate_provenance_bridge(
        output,
        candidate_root=candidate_root,
        provenance_registry_path=registry_path,
        editing_corpus_contract_path=CONTRACT_PATH,
    )
    assert validated["manifest_sha256"] == manifest["manifest_sha256"]


def test_source_stream_binds_exact_materialization_ledger_and_split_hashes(
    tmp_path: Path,
) -> None:
    candidate_root, registry_path, output, manifest = _materialize(tmp_path)
    stream = manifest["source_stream"]
    materialization = json.loads(
        (candidate_root / MATERIALIZATION_FILENAME).read_bytes()
    )

    assert stream["schema"] == SOURCE_STREAM_SCHEMA
    assert stream["nonempty_jsonl_rows"] == 1
    assert stream["candidate_materialization"] == {
        "manifest_file_sha256": file_sha256(candidate_root / MATERIALIZATION_FILENAME),
        "manifest_sha256": materialization["manifest_sha256"],
        "rows_file_sha256": materialization["rows"]["file_sha256"],
        "rows_semantic_sha256": materialization["rows"]["semantic_sha256"],
        "address_stream_sha256": materialization["rows"]["address_stream_sha256"],
    }
    ledger = json.loads((output / AUDIT_LEDGER_FILENAME).read_bytes())
    assert stream["candidate_audit_ledger"] == {
        "file_sha256": file_sha256(output / AUDIT_LEDGER_FILENAME),
        "semantic_sha256": ledger["ledger_sha256"],
        "rows_sha256": ledger["rows_sha256"],
    }
    assert stream["split_candidates"]["file_sha256"] == file_sha256(
        output / SPLIT_ROWS_FILENAME
    )
    assert stream["provenance_registry"]["file_sha256"] == file_sha256(registry_path)
    assert (output / BRIDGE_MANIFEST_FILENAME).is_file()


def test_bridge_is_byte_deterministic(tmp_path: Path) -> None:
    candidate_root = _candidate_materialization(tmp_path)
    registry_path = tmp_path / "registry.json"
    write_candidate_provenance_registry(_registry(candidate_root), registry_path)
    manifests = []
    outputs = []
    for name in ("first", "second"):
        output = tmp_path / name
        manifests.append(
            materialize_candidate_provenance_bridge(
                candidate_root=candidate_root,
                provenance_registry_path=registry_path,
                editing_corpus_contract_path=CONTRACT_PATH,
                output_dir=output,
            )
        )
        outputs.append(output)
    assert manifests[0] == manifests[1]
    for name in (AUDIT_LEDGER_FILENAME, SPLIT_ROWS_FILENAME, BRIDGE_MANIFEST_FILENAME):
        assert (outputs[0] / name).read_bytes() == (outputs[1] / name).read_bytes()


def test_registry_missing_source_provenance_fails_loudly(tmp_path: Path) -> None:
    candidate_root = _candidate_materialization(tmp_path)
    source_assets = _source_assets()
    del source_assets[0]["access_basis"]

    with pytest.raises(
        EditingV2CandidateProvenanceBridgeError,
        match=r"source_assets\[0\].*missing=.*access_basis",
    ):
        build_candidate_provenance_registry(
            candidate_root=candidate_root,
            editing_corpus_contract_path=CONTRACT_PATH,
            compiler_identity={
                "implementation_sha256": SHA_A,
                "config_sha256": SHA_B,
                "operator_contract_sha256": SHA_C,
                "canonicalizer_sha256": SHA_D,
            },
            policy_bindings={
                "mapping_policy_sha256": SHA_D,
                "metric_policy_sha256": SHA_E,
            },
            evidence_identity=_evidence_identity(),
            identity_definitions=_identity_definitions(),
            source_assets=source_assets,
        )


def test_missing_evidence_selector_fails_without_partial_publication(
    tmp_path: Path,
) -> None:
    candidate_root = _candidate_materialization(tmp_path)
    registry = _registry(candidate_root)
    body = copy.deepcopy(registry)
    del body["registry_sha256"]
    body["evidence_identity"]["component_reference_fields"]["path"] = [
        "missing.path_receipt"
    ]
    evidence_body = dict(body["evidence_identity"])
    del evidence_body["evidence_identity_sha256"]
    body["evidence_identity"]["evidence_identity_sha256"] = canonical_sha256(
        evidence_body
    )
    registry = {**body, "registry_sha256": canonical_sha256(body)}
    registry_path = tmp_path / "registry.json"
    write_candidate_provenance_registry(registry, registry_path)
    output = tmp_path / "bridge"

    with pytest.raises(
        EditingV2CandidateProvenanceBridgeError,
        match="missing.path_receipt.*absent",
    ):
        materialize_candidate_provenance_bridge(
            candidate_root=candidate_root,
            provenance_registry_path=registry_path,
            editing_corpus_contract_path=CONTRACT_PATH,
            output_dir=output,
        )
    assert not output.exists()


def test_emitted_rows_are_direct_v4_census_inputs(tmp_path: Path) -> None:
    _, _, output, manifest = _materialize(tmp_path)
    rows = [
        json.loads(line)
        for line in (output / SPLIT_ROWS_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    contract = load_editing_corpus_contract(CONTRACT_PATH)
    policy = default_split_census_policy(
        contract,
        identity_definitions=_identity_definitions(),
    )
    census = build_split_component_census(
        rows,
        policy=policy,
        editing_corpus_contract=contract,
        implementation_provenance={
            "schema": "compose.editing_v2_split_census_provenance",
            "schema_version": 1,
            "code_revision": {"commit_sha": "1" * 40, "dirty": False},
            "python_runtime": {
                "implementation": platform.python_implementation(),
                "version": platform.python_version(),
            },
            "source_files": [
                {
                    "path": "src/compose_v4/data/editing_v2_split_census.py",
                    "sha256": SHA_A,
                    "bytes": 1,
                }
            ],
            "policy_file": {"path": "fixture-policy.json", "sha256": SHA_B, "bytes": 1},
            "editing_corpus_contract_file": {
                "path": "configs/editing_corpus_v2_contract.json",
                "sha256": SHA_C,
                "bytes": 1,
            },
        },
        source_stream=manifest["source_stream"],
    )

    assert census["input_summary"]["source_rows"] == 1
    assert census["input_summary"]["valid_vertices"] == 1
    assert census["invalid_rows"] == []
    assert census["vertex_inventory"][0]["source_group_ids"] == ["fixture-source-group"]
    assert census["source_stream"] == manifest["source_stream"]
