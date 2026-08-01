from __future__ import annotations

import builtins
import copy
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from compose_v4.data.editing_v2_candidate_router import (
    EXECUTOR_GENERATED_WALK_FROM_REAL_ENDPOINT,
    INFERRED_REAL_ENDPOINT_PAIR,
)
from compose_v4.data.editing_v2_packed_candidate_materializer import (
    CANDIDATE_ROWS_FILENAME,
    HISTORICAL_OVERLAY_ROLE,
    MATERIALIZATION_FILENAME,
    EditingV2PackedCandidateMaterializationError,
    _partition_groups,
    build_overlay_source_binding_registry,
    build_packed_candidate_source_manifest,
    build_packed_candidate_source_manifest_from_overlay_completion,
    canonical_sha256,
    canonical_json_bytes,
    file_sha256,
    load_overlay_source_binding_registry,
    materialize_packed_candidate_headers,
    validate_historical_provenance_overlay,
    validate_packed_candidate_materialization,
    write_packed_candidate_source_manifest,
)
from modal_apps.apply_provenance_overlays_app import (
    FROZEN_MMP_V2_ROOT,
    build_overlay_completion,
)


ROOT = Path(__file__).resolve().parents[1]
ROUTING_POLICY = ROOT / "configs" / "editing_v2_candidate_routing_policy_v1.json"
CORPUS_CONTRACT = ROOT / "configs" / "editing_corpus_v2_contract.json"
SOURCE_BINDINGS = ROOT / "configs" / "editing_v2_overlay_source_bindings_v1.json"


def _source_manifest_identities(sources: list[dict]) -> tuple[dict, dict]:
    inventory = sorted(
        (
            {"source_asset_id": source["source_asset_id"], **shard}
            for source in sources
            for shard in source["shards"]
        ),
        key=lambda item: (item["source_asset_id"], item["relative_path"]),
    )
    return (
        {
            "artifact_path": "/artifacts/editing_v2/upstream_overlays/fixture.json",
            "file_sha256": "3" * 64,
            "semantic_sha256": "4" * 64,
            "candidate_source_shards": len(inventory),
            "candidate_source_inventory_sha256": canonical_sha256(inventory),
        },
        {
            "artifact_path": "configs/fixture_source_bindings.json",
            "file_sha256": "5" * 64,
            "semantic_sha256": "6" * 64,
        },
    )


def test_project_source_binding_registry_is_frozen_and_valid() -> None:
    registry = load_overlay_source_binding_registry(SOURCE_BINDINGS)

    assert registry["registry_sha256"] == (
        "008870e90bbe76ecddf4d26e2ea3f2b0a3a038bf463b17f273b27b8d9e983780"
    )
    assert {layer for binding in registry["bindings"] for layer in binding["allowed_layers"]} == {
        "corruption",
        "cycle_ops",
        "mmp_analogue",
    }


def _build_fixture_source_manifest(sources: list[dict]) -> dict:
    completion_identity, registry_identity = _source_manifest_identities(sources)
    return build_packed_candidate_source_manifest(
        sources,
        upstream_overlay_completion=completion_identity,
        source_binding_registry=registry_identity,
    )


def _state(label: int) -> dict:
    # Structurally exact packed-state wire data. It intentionally is not
    # converted to a chemistry object anywhere in these tests.
    return {
        "n_slots": 3,
        "atom_types": [label, 0, 0],
        "formal_charges": [0, 0, 0],
        "implicit_h_counts": [0, 0, 0],
        "bonds": [],
    }


def _action(executor_rule: str, family: str) -> dict:
    payload = {"neighbors": []} if executor_rule == "atom_insert" else {}
    return {
        "schema": "compose.rewrite.action",
        "schema_version": 2,
        "executor_rule": executor_rule,
        "model_family": family,
        "payload_type": "FixturePayload",
        "payload": payload,
    }


def _entry(
    *,
    trace_id: str,
    layer: str,
    executor_rule: str,
    family: str,
    source_group: str,
    compiler_path_class: str | None,
    atom_delta: int = 0,
    cycle_delta: int = 0,
    legacy_partition_groups: bool = False,
) -> dict:
    source = _state(1)
    target = _state(2)
    metadata = {
        "partition_isolation": {
            "schema": "compose.data.partition_isolation",
            "schema_version": 1,
            "molecule_ids": [f"{source_group}-source", f"{source_group}-target"],
            "scaffold_ids": [f"{source_group}-scaffold"],
            "source_group_id": source_group,
        }
    }
    if compiler_path_class is not None:
        metadata["compiler_path_class"] = compiler_path_class
    if legacy_partition_groups:
        metadata.pop("partition_isolation")
        metadata["origin_smiles"] = f"{trace_id}-source-key"
    trace = {
        "schema": "compose.rewrite.trace",
        "schema_version": 2,
        "source_state": source,
        "source_key": f"{trace_id}-source-key",
        "target_key": f"{trace_id}-target-key",
        "source_scaffold": f"{source_group}-scaffold",
        "n_slots": 3,
        "steps": [
            {
                "action": _action(executor_rule, family),
                "successor_key": f"{trace_id}-successor-key",
                "atom_count_delta": atom_delta,
                "cycle_rank_delta": cycle_delta,
            }
        ],
        "path_length": 1,
        "family_histogram": {family: 1},
        "operator_histogram": {executor_rule: 1},
        "atom_count_delta": atom_delta,
        "cycle_rank_delta": cycle_delta,
        "trace_id": trace_id,
        "partition": "legacy_train",
        "layer": layer,
        "metadata": metadata,
    }
    return {"trace": trace, "states": [source, target]}


def _write_packed_shard(
    path: Path,
    entries: list[dict],
) -> tuple[str, str, str, str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_handle,
            mtime=0,
        ) as compressed:
            for entry in entries:
                compressed.write(canonical_json_bytes(entry) + b"\n")
    manifest_path = path.with_suffix(".manifest.json")
    manifest = {
        "schema": "compose.data.packed_trace",
        "schema_version": 1,
        "sampler_contract": {},
        "entries": len(entries),
        "states": sum(len(entry["states"]) for entry in entries),
        "provenance": {"fixture": True},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    shard_sha256 = file_sha256(path)
    manifest_sha256 = file_sha256(manifest_path)
    overlay_path = Path(f"{path}.provenance.json")
    overlay = {
        "schema": "compose.data.provenance_overlay",
        "schema_version": 1,
        "shard": path.name,
        "packed_shard_content_sha256": shard_sha256,
        "original_manifest_sha256": manifest_sha256,
        "fields": {
            "codec_implementation_hash": "baaa75367f25a8c6",
            "trace_schema_version": 2,
            "packed_store_schema_version": 1,
            # These are deliberately the historical values that now fail the
            # live semantic overlay loader after the charge-policy change.
            "tensorization_implementation_hash": "7b0c88b166828f44",
        },
        "packer_commit": "a7546e2",
        "upgrade_implementation_hash": "2ba5bc5702f33b8c",
        "certification": {
            "replay_verified": True,
            "sentinel_entries_replayed": len(entries),
        },
    }
    overlay_path.write_text(json.dumps(overlay, indent=2, sort_keys=True) + "\n")
    return (
        str(path.relative_to(path.parents[1])),
        shard_sha256,
        manifest_sha256,
        str(overlay_path.relative_to(path.parents[1])),
        file_sha256(overlay_path),
    )


def _source_manifest(tmp_path: Path, *, malformed: bool = False) -> Path:
    artifact_root = tmp_path / "artifacts"
    synthetic_path = artifact_root / "synthetic" / "cycle.jsonl.gz"
    synthetic_entries = [
        _entry(
            trace_id="cycle-open",
            layer="cycle_ops",
            executor_rule="bond_delete",
            family="cycle_attach",
            source_group="synthetic-a",
            compiler_path_class=None,
            cycle_delta=-1,
            legacy_partition_groups=True,
        ),
        _entry(
            trace_id="disabled-ring-delete",
            layer="corruption",
            executor_rule="ring_system_delete",
            family="ring_system_delete",
            source_group="synthetic-b",
            compiler_path_class=None,
            atom_delta=-1,
            legacy_partition_groups=True,
        ),
    ]
    if malformed:
        synthetic_entries[-1]["trace"]["path_length"] = 2
    (
        synthetic_relative,
        synthetic_sha,
        synthetic_manifest_sha,
        synthetic_overlay_relative,
        synthetic_overlay_sha,
    ) = _write_packed_shard(synthetic_path, synthetic_entries)

    mmp_path = artifact_root / "mmp" / "direct.jsonl.gz"
    (
        mmp_relative,
        mmp_sha,
        mmp_manifest_sha,
        mmp_overlay_relative,
        mmp_overlay_sha,
    ) = _write_packed_shard(
        mmp_path,
        [
            _entry(
                trace_id="direct-reroute",
                layer="mmp_analogue",
                executor_rule="bond_reroute",
                family="bond_reroute",
                source_group="mmp-a",
                compiler_path_class="direct_bond_reroute",
            )
        ],
    )
    # Source kinds are alphabetically ordered by the source-manifest contract.
    sources = [
        {
            "source_asset_id": "synthetic-corruption-cycle-fixture",
            "source_asset_path": "/artifacts/source/synthetic.json",
            "source_asset_sha256": "1" * 64,
            "source_kind": EXECUTOR_GENERATED_WALK_FROM_REAL_ENDPOINT,
            "allowed_layers": ["corruption", "cycle_ops"],
            "shards": [
                {
                    "relative_path": synthetic_relative,
                    "file_sha256": synthetic_sha,
                    "manifest_relative_path": str(
                        Path(synthetic_relative).with_suffix(".manifest.json")
                    ),
                    "manifest_file_sha256": synthetic_manifest_sha,
                    "historical_provenance_overlay_relative_path": (synthetic_overlay_relative),
                    "historical_provenance_overlay_file_sha256": (synthetic_overlay_sha),
                    "historical_provenance_overlay_role": (HISTORICAL_OVERLAY_ROLE),
                }
            ],
        },
        {
            "source_asset_id": "frozen-mmp-fixture",
            "source_asset_path": "/artifacts/source/mmp_pool.jsonl",
            "source_asset_sha256": "2" * 64,
            "source_kind": INFERRED_REAL_ENDPOINT_PAIR,
            "allowed_layers": ["mmp_analogue"],
            "shards": [
                {
                    "relative_path": mmp_relative,
                    "file_sha256": mmp_sha,
                    "manifest_relative_path": str(Path(mmp_relative).with_suffix(".manifest.json")),
                    "manifest_file_sha256": mmp_manifest_sha,
                    "historical_provenance_overlay_relative_path": (mmp_overlay_relative),
                    "historical_provenance_overlay_file_sha256": (mmp_overlay_sha),
                    "historical_provenance_overlay_role": (HISTORICAL_OVERLAY_ROLE),
                }
            ],
        },
    ]
    manifest = _build_fixture_source_manifest(sources)
    manifest_path = tmp_path / "sources.json"
    write_packed_candidate_source_manifest(manifest_path, manifest)
    return manifest_path


def _materialize(tmp_path: Path, output_name: str, *, malformed: bool = False):
    source_manifest = _source_manifest(tmp_path, malformed=malformed)
    output = tmp_path / output_name
    built = materialize_packed_candidate_headers(
        source_manifest_path=source_manifest,
        artifact_root=tmp_path / "artifacts",
        routing_policy_path=ROUTING_POLICY,
        editing_corpus_contract_path=CORPUS_CONTRACT,
        output_dir=output,
        code_revision="a" * 40,
    )
    return output, built


def _rows(output: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (output / CANDIDATE_ROWS_FILENAME).read_text().splitlines()
        if line
    ]


def test_streams_packed_headers_routes_active8_and_preserves_disabled_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "rdkit" or name.startswith("rdkit."):
            raise AssertionError("candidate materialization must not import RDKit")
        if name.startswith("compose_v4.rewrite") or name.startswith("compose_v4.chem"):
            raise AssertionError(
                "candidate materialization must not import replay or chemistry code"
            )
        if name in {
            "compose_v4.data.packed_trace_store",
            "compose_v4.data.provenance_overlay",
        }:
            raise AssertionError(
                "candidate materialization must not invoke live packed-cache or overlay semantics"
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    output, built = _materialize(tmp_path, "materialized")
    rows = _rows(output)

    assert len(rows) == 3
    assert built["training_authorized"] is False
    assert all(row["training_authorized"] is False for row in rows)
    cycle, disabled, reroute = rows
    assert cycle["data_lane"] == "reversible_synthetic_walk"
    assert cycle["evidence_profile_id"] == "executor_generated_walk"
    assert cycle["operator_summary"]["family_histogram"] == {"cycle_attach": 1}
    assert cycle["operator_summary"]["executor_histogram"] == {"bond_delete": 1}
    assert cycle["operator_summary"]["graph_cycle_rank_delta"] == -1
    assert cycle["partition_group_derivation"] == ("legacy_synthetic_origin_scaffold_v1")
    assert cycle["groups"] == {
        "molecule_ids": ["cycle-open-source-key", "cycle-open-target-key"],
        "scaffold_ids": ["synthetic-a-scaffold"],
        "source_group_ids": ["cycle-open-source-key"],
    }
    assert cycle["exact_states"]["state_count"] == 2
    assert len(cycle["exact_states"]["encoded_state_stream_sha256"]) == 64

    assert disabled["disposition"] == "rejected_candidate"
    assert disabled["data_lane"] is None
    assert disabled["evidence_profile_id"] == "executor_generated_walk"
    assert disabled["evidence_components"]["path"] == "executor_generated_walk"
    assert disabled["rejection"]["attempts"] == [
        {
            "executor_rule": "ring_system_delete",
            "family": "ring_system_delete",
            "reason": "disabled_family",
            "step_index": 0,
        }
    ]
    assert disabled["packed_address"]["entry_index"] == 1

    assert reroute["data_lane"] == "linker_positional_topology_analogue"
    assert reroute["compiler_path_class"] == "direct_bond_reroute"
    assert reroute["evidence_profile_id"] == "inferred_relation_compiled_path"
    assert reroute["partition_group_derivation"] == "explicit_partition_isolation_v1"
    assert reroute["groups"] == {
        "molecule_ids": ["mmp-a-source", "mmp-a-target"],
        "scaffold_ids": ["mmp-a-scaffold"],
        "source_group_ids": ["mmp-a"],
    }

    totals = built["rows"]["totals"]
    assert totals["rows"] == 3
    assert totals["routed_rows"] == 2
    assert totals["rejected_rows"] == 1
    assert totals["states"] == 6
    assert totals["transitions"] == 3
    assert totals["rejection_reason_histogram"] == {"disabled_family": 1}
    assert totals["partition_group_derivation_histogram"] == {
        "explicit_partition_isolation_v1": 1,
        "legacy_synthetic_origin_scaffold_v1": 2,
    }
    assert built["implementation"]["uses_rdkit_or_executor_replay"] is False
    assert built["implementation"]["uses_live_packed_overlay_semantic_validation"] is False
    assert (
        built["provenance_boundary"][
            "historical_sidecar_fields_used_as_current_training_provenance"
        ]
        is False
    )
    legacy_derivation = built["provenance_boundary"]["partition_group_derivation_contract"][
        "legacy_synthetic_origin_scaffold_v1"
    ]
    assert legacy_derivation["allowed_layers"] == ["corruption", "cycle_ops"]
    assert legacy_derivation["origin_must_identify_trace_endpoint"] is True
    overlay_receipt = built["sources"][0]["shards"][0]["historical_provenance_overlay"]
    assert overlay_receipt["role"] == HISTORICAL_OVERLAY_ROLE
    assert overlay_receipt["training_provenance_authorized"] is False
    assert overlay_receipt["live_overlay_semantic_validation_applied"] is False
    assert overlay_receipt["recorded_upgrade_implementation_hash"] == "2ba5bc5702f33b8c"
    assert overlay_receipt["recorded_tensorization_implementation_hash"] == "7b0c88b166828f44"
    assert (
        cycle["packed_address"]["historical_provenance_overlay_file_sha256"]
        == overlay_receipt["file_sha256"]
    )
    assert validate_packed_candidate_materialization(output) == built


def test_materializer_has_no_live_packed_loader_or_overlay_import() -> None:
    source = (
        ROOT / "src" / "compose_v4" / "data" / "editing_v2_packed_candidate_materializer.py"
    ).read_text()
    assert "read_addressed_packed_shard(" not in source
    assert "from compose_v4.data.packed_trace_store" not in source
    assert "from compose_v4.data.provenance_overlay" not in source


def test_legacy_partition_group_derivation_is_limited_to_synthetic_layers() -> None:
    entry = _entry(
        trace_id="legacy-cycle",
        layer="cycle_ops",
        executor_rule="bond_delete",
        family="cycle_attach",
        source_group="legacy-group",
        compiler_path_class=None,
        legacy_partition_groups=True,
    )
    groups, derivation = _partition_groups(entry["trace"], address="fixture")
    assert derivation == "legacy_synthetic_origin_scaffold_v1"
    assert groups["source_group_ids"] == ["legacy-cycle-source-key"]

    entry["trace"]["layer"] = "mmp_analogue"
    with pytest.raises(
        EditingV2PackedCandidateMaterializationError,
        match="partition_isolation is required",
    ):
        _partition_groups(entry["trace"], address="fixture")


def test_legacy_partition_group_derivation_rejects_unbound_origin() -> None:
    entry = _entry(
        trace_id="legacy-cycle",
        layer="cycle_ops",
        executor_rule="bond_delete",
        family="cycle_attach",
        source_group="legacy-group",
        compiler_path_class=None,
        legacy_partition_groups=True,
    )
    entry["trace"]["metadata"]["origin_smiles"] = "unrelated-molecule"
    with pytest.raises(
        EditingV2PackedCandidateMaterializationError,
        match="must identify one trace endpoint",
    ):
        _partition_groups(entry["trace"], address="fixture")


def test_materialization_is_byte_deterministic(tmp_path: Path) -> None:
    source_manifest = _source_manifest(tmp_path)
    outputs = [tmp_path / "first", tmp_path / "second"]
    built = []
    for output in outputs:
        built.append(
            materialize_packed_candidate_headers(
                source_manifest_path=source_manifest,
                artifact_root=tmp_path / "artifacts",
                routing_policy_path=ROUTING_POLICY,
                editing_corpus_contract_path=CORPUS_CONTRACT,
                output_dir=output,
                code_revision="b" * 40,
            )
        )
    assert built[0] == built[1]
    assert (outputs[0] / CANDIDATE_ROWS_FILENAME).read_bytes() == (
        outputs[1] / CANDIDATE_ROWS_FILENAME
    ).read_bytes()
    assert (outputs[0] / MATERIALIZATION_FILENAME).read_bytes() == (
        outputs[1] / MATERIALIZATION_FILENAME
    ).read_bytes()


def test_malformed_late_row_leaves_no_partial_publication(tmp_path: Path) -> None:
    source_manifest = _source_manifest(tmp_path, malformed=True)
    output = tmp_path / "must-not-exist"
    with pytest.raises(
        EditingV2PackedCandidateMaterializationError,
        match=r"path_length \+ 1",
    ):
        materialize_packed_candidate_headers(
            source_manifest_path=source_manifest,
            artifact_root=tmp_path / "artifacts",
            routing_policy_path=ROUTING_POLICY,
            editing_corpus_contract_path=CORPUS_CONTRACT,
            output_dir=output,
            code_revision="c" * 40,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".must-not-exist.*.staging"))


def test_reopen_detects_row_and_manifest_tampering(tmp_path: Path) -> None:
    output, built = _materialize(tmp_path, "materialized")
    rows_path = output / CANDIDATE_ROWS_FILENAME
    rows_path.write_bytes(rows_path.read_bytes() + b" \n")
    with pytest.raises(
        EditingV2PackedCandidateMaterializationError,
        match="physical SHA-256",
    ):
        validate_packed_candidate_materialization(output)

    output_2, built_2 = _materialize(tmp_path / "second-run", "materialized")
    manifest_path = output_2 / MATERIALIZATION_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["rows"]["totals"]["rows"] += 1
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    with pytest.raises(
        EditingV2PackedCandidateMaterializationError,
        match="self-hash",
    ):
        validate_packed_candidate_materialization(
            output_2,
            expected_manifest_sha256=built_2["manifest_sha256"],
        )

    assert built["manifest_sha256"] == built_2["manifest_sha256"]


def test_source_manifest_self_hash_rejects_semantic_drift(tmp_path: Path) -> None:
    source_manifest_path = _source_manifest(tmp_path)
    payload = json.loads(source_manifest_path.read_text())
    drifted = copy.deepcopy(payload)
    drifted["sources"][0]["source_asset_sha256"] = "f" * 64
    source_manifest_path.write_text(json.dumps(drifted))
    with pytest.raises(
        EditingV2PackedCandidateMaterializationError,
        match="self-hash",
    ):
        materialize_packed_candidate_headers(
            source_manifest_path=source_manifest_path,
            artifact_root=tmp_path / "artifacts",
            routing_policy_path=ROUTING_POLICY,
            editing_corpus_contract_path=CORPUS_CONTRACT,
            output_dir=tmp_path / "output",
            code_revision="d" * 40,
        )


def test_fixture_source_bytes_are_physically_bound(tmp_path: Path) -> None:
    source_manifest_path = _source_manifest(tmp_path)
    source_manifest = json.loads(source_manifest_path.read_text())
    relative = source_manifest["sources"][0]["shards"][0]["relative_path"]
    shard = tmp_path / "artifacts" / relative
    original_sha256 = hashlib.sha256(shard.read_bytes()).hexdigest()
    assert original_sha256 == source_manifest["sources"][0]["shards"][0]["file_sha256"]
    shard.write_bytes(shard.read_bytes() + b"tamper")
    with pytest.raises(
        EditingV2PackedCandidateMaterializationError,
        match="packed shard SHA-256 mismatch",
    ):
        materialize_packed_candidate_headers(
            source_manifest_path=source_manifest_path,
            artifact_root=tmp_path / "artifacts",
            routing_policy_path=ROUTING_POLICY,
            editing_corpus_contract_path=CORPUS_CONTRACT,
            output_dir=tmp_path / "output",
            code_revision="e" * 40,
        )


def test_historical_overlay_bytes_are_physically_bound(tmp_path: Path) -> None:
    source_manifest_path = _source_manifest(tmp_path)
    source_manifest = json.loads(source_manifest_path.read_text())
    shard_spec = source_manifest["sources"][0]["shards"][0]
    overlay = tmp_path / "artifacts" / shard_spec["historical_provenance_overlay_relative_path"]
    overlay.write_bytes(overlay.read_bytes() + b" ")
    with pytest.raises(
        EditingV2PackedCandidateMaterializationError,
        match="historical packed provenance overlay SHA-256 mismatch",
    ):
        materialize_packed_candidate_headers(
            source_manifest_path=source_manifest_path,
            artifact_root=tmp_path / "artifacts",
            routing_policy_path=ROUTING_POLICY,
            editing_corpus_contract_path=CORPUS_CONTRACT,
            output_dir=tmp_path / "output",
            code_revision="f" * 40,
        )


def test_historical_overlay_validator_accepts_old_implementation_identity(
    tmp_path: Path,
) -> None:
    source_manifest_path = _source_manifest(tmp_path)
    source_manifest = json.loads(source_manifest_path.read_text())
    shard_spec = source_manifest["sources"][0]["shards"][0]
    overlay_path = (
        tmp_path / "artifacts" / shard_spec["historical_provenance_overlay_relative_path"]
    )

    receipt = validate_historical_provenance_overlay(
        overlay_path,
        expected_file_sha256=shard_spec["historical_provenance_overlay_file_sha256"],
        expected_shard_name=Path(shard_spec["relative_path"]).name,
        expected_packed_shard_sha256=shard_spec["file_sha256"],
        expected_original_manifest_sha256=shard_spec["manifest_file_sha256"],
    )

    assert receipt["recorded_upgrade_implementation_hash"] == "2ba5bc5702f33b8c"
    assert receipt["live_overlay_semantic_validation_applied"] is False
    assert receipt["training_provenance_authorized"] is False


@pytest.mark.parametrize(
    ("binding_field", "message"),
    [
        (
            "packed_shard_content_sha256",
            "does not bind the observed shard",
        ),
        (
            "original_manifest_sha256",
            "does not bind the observed original manifest",
        ),
    ],
)
def test_historical_overlay_must_bind_shard_and_original_manifest(
    tmp_path: Path,
    binding_field: str,
    message: str,
) -> None:
    source_manifest_path = _source_manifest(tmp_path)
    source_manifest = json.loads(source_manifest_path.read_text())
    shard_spec = source_manifest["sources"][0]["shards"][0]
    overlay_path = (
        tmp_path / "artifacts" / shard_spec["historical_provenance_overlay_relative_path"]
    )
    overlay = json.loads(overlay_path.read_text())
    overlay[binding_field] = "f" * 64
    overlay_path.write_text(json.dumps(overlay, indent=2, sort_keys=True) + "\n")
    shard_spec["historical_provenance_overlay_file_sha256"] = file_sha256(overlay_path)
    refreshed_completion_identity, _ = _source_manifest_identities(source_manifest["sources"])
    source_manifest["upstream_overlay_completion"]["candidate_source_inventory_sha256"] = (
        refreshed_completion_identity["candidate_source_inventory_sha256"]
    )
    rebuilt = build_packed_candidate_source_manifest(
        source_manifest["sources"],
        upstream_overlay_completion=source_manifest["upstream_overlay_completion"],
        source_binding_registry=source_manifest["source_binding_registry"],
    )
    write_packed_candidate_source_manifest(source_manifest_path, rebuilt)
    with pytest.raises(
        EditingV2PackedCandidateMaterializationError,
        match=message,
    ):
        materialize_packed_candidate_headers(
            source_manifest_path=source_manifest_path,
            artifact_root=tmp_path / "artifacts",
            routing_policy_path=ROUTING_POLICY,
            editing_corpus_contract_path=CORPUS_CONTRACT,
            output_dir=tmp_path / "output",
            code_revision="1" * 40,
        )


def test_oversized_packed_row_fails_without_partial_publication(
    tmp_path: Path,
) -> None:
    source_manifest_path = _source_manifest(tmp_path)
    output = tmp_path / "oversized-row-output"
    with pytest.raises(
        EditingV2PackedCandidateMaterializationError,
        match="packed entry exceeds",
    ):
        materialize_packed_candidate_headers(
            source_manifest_path=source_manifest_path,
            artifact_root=tmp_path / "artifacts",
            routing_policy_path=ROUTING_POLICY,
            editing_corpus_contract_path=CORPUS_CONTRACT,
            output_dir=output,
            code_revision="2" * 40,
            max_entry_bytes=64,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".oversized-row-output.*.staging"))


def test_source_manifest_rejects_duplicate_physical_shard_paths(
    tmp_path: Path,
) -> None:
    source_manifest_path = _source_manifest(tmp_path)
    source_manifest = json.loads(source_manifest_path.read_text())
    source_manifest["sources"][1]["shards"][0] = copy.deepcopy(
        source_manifest["sources"][0]["shards"][0]
    )
    with pytest.raises(
        EditingV2PackedCandidateMaterializationError,
        match="physical shard relative paths must be unique",
    ):
        build_packed_candidate_source_manifest(
            source_manifest["sources"],
            upstream_overlay_completion=source_manifest["upstream_overlay_completion"],
            source_binding_registry=source_manifest["source_binding_registry"],
        )


def _overlay_consumer_fixture(tmp_path: Path) -> tuple[Path, Path]:
    packed_root = "/artifacts/edit-packed-v1"
    receipts = []
    for index, (layer, partition) in enumerate(
        (layer, partition)
        for layer in ("corruption", "cycle_ops", "mmp_analogue")
        for partition in ("train", "validation", "test")
    ):
        root = (
            Path(FROZEN_MMP_V2_ROOT) / partition
            if layer == "mmp_analogue"
            else Path(packed_root) / layer / partition
        )
        shard = root / f"shard_{index:04d}.jsonl.gz"
        receipts.append(
            {
                "layer": layer,
                "partition": partition,
                "shard_path": str(shard),
                "shard_file_sha256": hashlib.sha256(f"shard-{index}".encode()).hexdigest(),
                "manifest_path": str(shard.with_suffix(".manifest.json")),
                "manifest_file_sha256": hashlib.sha256(f"manifest-{index}".encode()).hexdigest(),
                "entries": index + 1,
                "states": index + 2,
                "overlay_path": f"{shard}.provenance.json",
                "overlay_file_sha256": hashlib.sha256(f"overlay-{index}".encode()).hexdigest(),
                "overlay_fields_sha256": hashlib.sha256(f"fields-{index}".encode()).hexdigest(),
                "overlay_semantic_sha256": hashlib.sha256(f"semantic-{index}".encode()).hexdigest(),
                "overlay_validation_role": (
                    "current_live_overlay_at_completion"
                    if index % 2
                    else "historical_immutable_byte_receipt"
                ),
            }
        )

    def inventory_sha256(layers: set[str]) -> str:
        rows = sorted(
            (
                {
                    "layer": receipt["layer"],
                    "partition": receipt["partition"],
                    "shard": Path(receipt["shard_path"]).name,
                    "packed_sha256": receipt["shard_file_sha256"],
                    "manifest_sha256": receipt["manifest_file_sha256"],
                    "entries": receipt["entries"],
                    "states": receipt["states"],
                }
                for receipt in receipts
                if receipt["layer"] in layers
            ),
            key=lambda row: (row["layer"], row["partition"], row["shard"]),
        )
        return canonical_sha256(rows)

    completion = build_overlay_completion(
        commit="1" * 40,
        launcher_source_sha256="2" * 64,
        packed_root=packed_root,
        mmp_root=FROZEN_MMP_V2_ROOT,
        upstream_completions={
            "audit_pack": {
                "path": f"{packed_root}/PACK_COMPLETE.json",
                "file_sha256": "3" * 64,
                "completion_flag": "PACK_COMPLETE",
                "inventory_origin": "reconciled_legacy_completion_totals",
                "expected_shards": 6,
                "shard_inventory_sha256": inventory_sha256({"corruption", "cycle_ops"}),
            },
            "mmp_pack": {
                "path": f"{FROZEN_MMP_V2_ROOT}/MMP_PACK_COMPLETE.json",
                "file_sha256": "4" * 64,
                "completion_flag": "MMP_PACK_COMPLETE",
                "inventory_origin": "completion_shard_artifacts",
                "expected_shards": 3,
                "shard_inventory_sha256": inventory_sha256({"mmp_analogue"}),
            },
        },
        new_overlay_fields={"trace_schema_version": 2},
        shard_receipts=receipts,
    )
    completion_path = tmp_path / "OVERLAY_COMPLETION.json"
    completion_path.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")

    registry = build_overlay_source_binding_registry(
        [
            {
                "source_asset_id": "synthetic-cache-source",
                "source_asset_path": "/artifacts/source/synthetic.json",
                "source_asset_sha256": "5" * 64,
                "source_kind": EXECUTOR_GENERATED_WALK_FROM_REAL_ENDPOINT,
                "allowed_layers": ["corruption", "cycle_ops"],
            },
            {
                "source_asset_id": "mmp-cache-source",
                "source_asset_path": "/artifacts/source/mmp.jsonl",
                "source_asset_sha256": "6" * 64,
                "source_kind": INFERRED_REAL_ENDPOINT_PAIR,
                "allowed_layers": ["mmp_analogue"],
            },
        ]
    )
    registry_path = tmp_path / "SOURCE_BINDINGS.json"
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    return completion_path, registry_path


def _consume_overlay_fixture(completion_path: Path, registry_path: Path) -> dict:
    return build_packed_candidate_source_manifest_from_overlay_completion(
        overlay_completion_path=completion_path,
        expected_overlay_completion_file_sha256=file_sha256(completion_path),
        source_binding_registry_path=registry_path,
        expected_source_binding_registry_file_sha256=file_sha256(registry_path),
    )


def test_overlay_completion_builds_exact_schema_v2_source_manifest(
    tmp_path: Path,
) -> None:
    completion_path, registry_path = _overlay_consumer_fixture(tmp_path)

    manifest = _consume_overlay_fixture(completion_path, registry_path)

    shards = [shard for source in manifest["sources"] for shard in source["shards"]]
    assert manifest["schema_version"] == 2
    assert manifest["upstream_overlay_completion"]["file_sha256"] == file_sha256(completion_path)
    assert manifest["upstream_overlay_completion"]["candidate_source_shards"] == 9
    assert len(shards) == 9
    assert {shard["historical_provenance_overlay_role"] for shard in shards} == {
        HISTORICAL_OVERLAY_ROLE
    }


@pytest.mark.parametrize("mutation", ["omission", "duplicate", "escape"])
def test_overlay_completion_rejects_nonexact_or_escaping_inventory(
    tmp_path: Path,
    mutation: str,
) -> None:
    completion_path, registry_path = _overlay_consumer_fixture(tmp_path)
    completion = json.loads(completion_path.read_text())
    if mutation == "omission":
        completion["shards"].pop()
    elif mutation == "duplicate":
        completion["shards"][-1] = copy.deepcopy(completion["shards"][0])
    else:
        completion["shards"][0]["shard_path"] = (
            "/artifacts/edit-packed-v1/corruption/train/../escape.jsonl.gz"
        )
    body = {key: value for key, value in completion.items() if key != "completion_sha256"}
    completion["completion_sha256"] = canonical_sha256(body)
    completion_path.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")

    with pytest.raises(EditingV2PackedCandidateMaterializationError):
        _consume_overlay_fixture(completion_path, registry_path)


def test_overlay_completion_rejects_physical_or_semantic_identity_drift(
    tmp_path: Path,
) -> None:
    completion_path, registry_path = _overlay_consumer_fixture(tmp_path)
    expected_file_sha256 = file_sha256(completion_path)
    completion = json.loads(completion_path.read_text())
    completion["commit"] = "9" * 40
    completion_path.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")

    with pytest.raises(
        EditingV2PackedCandidateMaterializationError,
        match="physical SHA-256 disagrees",
    ):
        build_packed_candidate_source_manifest_from_overlay_completion(
            overlay_completion_path=completion_path,
            expected_overlay_completion_file_sha256=expected_file_sha256,
            source_binding_registry_path=registry_path,
            expected_source_binding_registry_file_sha256=file_sha256(registry_path),
        )

    with pytest.raises(
        EditingV2PackedCandidateMaterializationError,
        match="self-hash disagrees",
    ):
        _consume_overlay_fixture(completion_path, registry_path)
