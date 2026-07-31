"""The editing-V2 lane resolver must fail closed before any training consumer."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

import pytest

from compose_v4.data.editing_corpus_contract import load_editing_corpus_contract
from compose_v4.data.editing_v2_lane_registry import (
    EditingV2LaneRegistryError,
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
    editing_v2_partition_roles,
    file_sha256,
    resolve_editing_v2_lane_registry,
)

REPO = Path(__file__).resolve().parent.parent
AUTHORITATIVE_CONTRACT = REPO / "configs" / "editing_corpus_v2_contract.json"
PARTITION_ROLES = (
    "train",
    "validation",
    "controller_validation",
    "final_test",
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _with_semantic_hash(
    payload: dict[str, Any],
    *,
    field: str,
) -> dict[str, Any]:
    body = {key: value for key, value in payload.items() if key != field}
    return {**body, field: canonical_sha256(body)}


@dataclass
class RegistryFixture:
    contract_path: Path
    registry_path: Path
    artifact_root: Path
    contract: dict[str, Any]
    registry: dict[str, Any]

    def write_registry(self) -> None:
        self.registry = _with_semantic_hash(
            self.registry,
            field="registry_sha256",
        )
        _write_json(self.registry_path, self.registry)

    def completion_path(self, lane_index: int) -> Path:
        artifact_path = self.registry["lanes"][lane_index]["completion_manifest_path"]
        relative = Path(artifact_path).relative_to("/artifacts")
        return self.artifact_root / relative

    def rewrite_completion(
        self,
        lane_index: int,
        mutation: Callable[[dict[str, Any]], None],
    ) -> None:
        path = self.completion_path(lane_index)
        completion = json.loads(path.read_text())
        mutation(completion)
        completion = _with_semantic_hash(
            completion,
            field="completion_sha256",
        )
        _write_json(path, completion)
        self.registry["lanes"][lane_index]["completion_manifest_sha256"] = file_sha256(path)
        self.write_registry()


@pytest.fixture()
def complete_registry(tmp_path: Path) -> RegistryFixture:
    contract = json.loads(AUTHORITATIVE_CONTRACT.read_text())
    contract["split_contract"]["partition_roles"] = list(PARTITION_ROLES)
    contract_path = tmp_path / "editing_corpus_v2_contract.json"
    _write_json(contract_path, contract)
    validated_contract = load_editing_corpus_contract(contract_path)
    contract_identity = editing_corpus_contract_identity(
        validated_contract,
        contract_file_sha256=file_sha256(contract_path),
    )

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    registry_lanes: list[dict[str, Any]] = []
    for lane_index, lane in enumerate(validated_contract["data_lanes"]):
        lane_id = lane["id"]
        shard_bytes = (
            json.dumps(
                {
                    "data_lane": lane_id,
                    "evidence_profile_id": lane["admissible_evidence_profiles"][0],
                    "record_id": f"{lane_id}-fixture",
                },
                sort_keys=True,
            ).encode()
            + b"\n"
        )
        relative_path = "train/shard_0000.jsonl"
        shard = {
            "partition": "train",
            "relative_path": relative_path,
            "sha256": hashlib.sha256(shard_bytes).hexdigest(),
            "bytes": len(shard_bytes),
            "records": 1,
            "states": lane_index + 2,
            "transitions": lane_index + 1,
        }
        shard_inventory_sha256 = canonical_sha256([shard])
        declared_root = f"/artifacts/editing_v2/lanes/{lane_id}/{shard_inventory_sha256}"
        local_root = artifact_root / Path(declared_root).relative_to("/artifacts")
        shard_path = local_root / relative_path
        shard_path.parent.mkdir(parents=True)
        shard_path.write_bytes(shard_bytes)

        by_partition = {
            partition: {
                "records": 0,
                "states": 0,
                "transitions": 0,
                "shards": 0,
            }
            for partition in PARTITION_ROLES
        }
        by_partition["train"] = {
            "records": shard["records"],
            "states": shard["states"],
            "transitions": shard["transitions"],
            "shards": 1,
        }
        completion = {
            "schema": LANE_COMPLETION_SCHEMA,
            "schema_version": LANE_COMPLETION_SCHEMA_VERSION,
            "status": LANE_COMPLETION_STATUS,
            "training_authorized": False,
            "editing_corpus_contract": contract_identity,
            "lane_id": lane_id,
            "admissible_evidence_profiles": lane["admissible_evidence_profiles"],
            "reserved_evidence_profiles": lane["reserved_evidence_profiles"],
            "declared_root": declared_root,
            "partition_roles": list(PARTITION_ROLES),
            "counts": {
                "records": shard["records"],
                "states": shard["states"],
                "transitions": shard["transitions"],
                "shards": 1,
                "by_partition": by_partition,
            },
            "shards": [shard],
            "shard_inventory_sha256": shard_inventory_sha256,
        }
        completion = _with_semantic_hash(
            completion,
            field="completion_sha256",
        )
        completion_path = local_root / LANE_COMPLETION_FILENAME
        _write_json(completion_path, completion)
        registry_lanes.append(
            {
                "lane_id": lane_id,
                "admissible_evidence_profiles": lane["admissible_evidence_profiles"],
                "reserved_evidence_profiles": lane["reserved_evidence_profiles"],
                "declared_root": declared_root,
                "completion_manifest_path": (f"{declared_root}/{LANE_COMPLETION_FILENAME}"),
                "completion_manifest_sha256": file_sha256(completion_path),
            }
        )

    registry = {
        "schema": LANE_REGISTRY_SCHEMA,
        "schema_version": LANE_REGISTRY_SCHEMA_VERSION,
        "status": LANE_REGISTRY_STATUS,
        "training_authorized": False,
        "editing_corpus_contract": contract_identity,
        "partition_roles": list(PARTITION_ROLES),
        "lanes": registry_lanes,
    }
    registry = _with_semantic_hash(registry, field="registry_sha256")
    registry_path = tmp_path / "EDITING_V2_LANE_REGISTRY.json"
    _write_json(registry_path, registry)
    return RegistryFixture(
        contract_path=contract_path,
        registry_path=registry_path,
        artifact_root=artifact_root,
        contract=validated_contract,
        registry=registry,
    )


def _resolve(fixture: RegistryFixture):
    return resolve_editing_v2_lane_registry(
        editing_corpus_contract_path=fixture.contract_path,
        registry_path=fixture.registry_path,
        artifact_root=fixture.artifact_root,
    )


def test_resolves_exact_five_lane_registry(
    complete_registry: RegistryFixture,
) -> None:
    resolved = _resolve(complete_registry)

    assert [lane.lane_id for lane in resolved.lanes] == [
        lane["id"] for lane in complete_registry.contract["data_lanes"]
    ]
    assert [lane.admissible_evidence_profiles for lane in resolved.lanes] == [
        tuple(lane["admissible_evidence_profiles"])
        for lane in complete_registry.contract["data_lanes"]
    ]
    assert [lane.reserved_evidence_profiles for lane in resolved.lanes] == [
        tuple(lane["reserved_evidence_profiles"])
        for lane in complete_registry.contract["data_lanes"]
    ]
    assert resolved.partition_roles == PARTITION_ROLES
    assert all(len(lane.shards) == 1 for lane in resolved.lanes)
    assert all(lane.shards[0].local_path.is_file() for lane in resolved.lanes)
    assert resolved.editing_corpus_contract_file_sha256 == file_sha256(
        complete_registry.contract_path
    )


def test_contract_without_partition_roles_fails_closed() -> None:
    contract = json.loads(AUTHORITATIVE_CONTRACT.read_text())
    contract["split_contract"].pop("partition_roles", None)

    with pytest.raises(
        EditingV2LaneRegistryError,
        match=r"split_contract\.partition_roles",
    ):
        editing_v2_partition_roles(contract)


def test_schema_v2_requires_exactly_five_contract_lanes() -> None:
    contract = json.loads(AUTHORITATIVE_CONTRACT.read_text())
    contract["data_lanes"].pop()

    with pytest.raises(EditingV2LaneRegistryError, match="exactly 5"):
        editing_v2_lane_definitions(contract)


@pytest.mark.parametrize("case", ["missing", "unknown", "duplicate"])
def test_rejects_lane_set_disagreement(
    complete_registry: RegistryFixture,
    case: str,
) -> None:
    if case == "missing":
        complete_registry.registry["lanes"].pop()
    elif case == "unknown":
        complete_registry.registry["lanes"][-1]["lane_id"] = "unknown_lane"
    else:
        complete_registry.registry["lanes"][1]["lane_id"] = complete_registry.registry["lanes"][0][
            "lane_id"
        ]
    complete_registry.write_registry()

    with pytest.raises(EditingV2LaneRegistryError):
        _resolve(complete_registry)


def test_rejects_registry_evidence_profile_mismatch(
    complete_registry: RegistryFixture,
) -> None:
    complete_registry.registry["lanes"][0]["admissible_evidence_profiles"] = [
        "executor_generated_walk"
    ]
    complete_registry.write_registry()

    with pytest.raises(
        EditingV2LaneRegistryError,
        match="evidence-profile set disagrees",
    ):
        _resolve(complete_registry)


def test_rejects_nonlist_registry_evidence_profiles(
    complete_registry: RegistryFixture,
) -> None:
    complete_registry.registry["lanes"][-1]["reserved_evidence_profiles"] = ""
    complete_registry.write_registry()

    with pytest.raises(EditingV2LaneRegistryError, match="must be a list"):
        _resolve(complete_registry)


def test_rejects_nonlist_completion_evidence_profiles(
    complete_registry: RegistryFixture,
) -> None:
    def mutate(completion: dict[str, Any]) -> None:
        completion["reserved_evidence_profiles"] = ""

    complete_registry.rewrite_completion(4, mutate)

    with pytest.raises(EditingV2LaneRegistryError, match="must be a list"):
        _resolve(complete_registry)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "compose.editing_v2_lane_registry.stale"),
        ("schema_version", 1),
        ("status", "PARTIAL"),
        ("training_authorized", True),
    ],
)
def test_rejects_registry_schema_status_or_authority(
    complete_registry: RegistryFixture,
    field: str,
    value: object,
) -> None:
    complete_registry.registry[field] = value
    complete_registry.write_registry()

    with pytest.raises(EditingV2LaneRegistryError, match="schema, status, or authority"):
        _resolve(complete_registry)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "compose.editing_v2_lane_completion.stale"),
        ("schema_version", 1),
        ("status", "PARTIAL"),
        ("training_authorized", True),
    ],
)
def test_rejects_completion_schema_status_or_authority(
    complete_registry: RegistryFixture,
    field: str,
    value: object,
) -> None:
    def mutate(completion: dict[str, Any]) -> None:
        completion[field] = value

    complete_registry.rewrite_completion(0, mutate)

    with pytest.raises(EditingV2LaneRegistryError, match="schema, status, or authority"):
        _resolve(complete_registry)


def test_rejects_unexpected_partition(
    complete_registry: RegistryFixture,
) -> None:
    def mutate(completion: dict[str, Any]) -> None:
        completion["shards"][0]["partition"] = "unexpected_partition"

    complete_registry.rewrite_completion(0, mutate)

    with pytest.raises(
        EditingV2LaneRegistryError,
        match="not allowed",
    ):
        _resolve(complete_registry)


@pytest.mark.parametrize(
    ("target", "replacement"),
    [
        (
            "declared_root",
            "/artifacts/editing_v2/lanes/observed_local_analogue/../escape",
        ),
        ("completion_manifest_path", "/artifacts/../escape/LANE_COMPLETE.json"),
    ],
)
def test_rejects_registry_path_traversal(
    complete_registry: RegistryFixture,
    target: str,
    replacement: str,
) -> None:
    complete_registry.registry["lanes"][0][target] = replacement
    complete_registry.write_registry()

    with pytest.raises(EditingV2LaneRegistryError, match="normalized"):
        _resolve(complete_registry)


def test_rejects_shard_path_traversal(
    complete_registry: RegistryFixture,
) -> None:
    def mutate(completion: dict[str, Any]) -> None:
        completion["shards"][0]["relative_path"] = "train/../escape.jsonl"

    complete_registry.rewrite_completion(0, mutate)

    with pytest.raises(EditingV2LaneRegistryError, match="without traversal"):
        _resolve(complete_registry)


def test_rejects_absent_completion_manifest(
    complete_registry: RegistryFixture,
) -> None:
    complete_registry.completion_path(0).unlink()

    with pytest.raises(EditingV2LaneRegistryError, match="completion manifest is absent"):
        _resolve(complete_registry)


def test_rejects_completion_file_hash_mismatch(
    complete_registry: RegistryFixture,
) -> None:
    completion_path = complete_registry.completion_path(0)
    completion_path.write_bytes(completion_path.read_bytes() + b"\n")

    with pytest.raises(EditingV2LaneRegistryError, match="file SHA-256 disagrees"):
        _resolve(complete_registry)


def test_rejects_absent_shard(complete_registry: RegistryFixture) -> None:
    first_lane = complete_registry.registry["lanes"][0]
    shard_path = (
        complete_registry.artifact_root
        / Path(first_lane["declared_root"]).relative_to("/artifacts")
        / "train"
        / "shard_0000.jsonl"
    )
    shard_path.unlink()

    with pytest.raises(EditingV2LaneRegistryError, match="shards\\[0\\] is absent"):
        _resolve(complete_registry)


def test_rejects_partial_counts(
    complete_registry: RegistryFixture,
) -> None:
    def mutate(completion: dict[str, Any]) -> None:
        completion["counts"]["records"] += 1

    complete_registry.rewrite_completion(0, mutate)

    with pytest.raises(EditingV2LaneRegistryError, match="partition sum"):
        _resolve(complete_registry)


def test_rejects_physical_shard_hash_mismatch(
    complete_registry: RegistryFixture,
) -> None:
    first_lane = complete_registry.registry["lanes"][0]
    shard_path = (
        complete_registry.artifact_root
        / Path(first_lane["declared_root"]).relative_to("/artifacts")
        / "train"
        / "shard_0000.jsonl"
    )
    original = shard_path.read_bytes()
    shard_path.write_bytes(bytes([original[0] ^ 1]) + original[1:])

    with pytest.raises(EditingV2LaneRegistryError, match="sha256 disagrees"):
        _resolve(complete_registry)


def test_rejects_cross_lane_contract_mismatch(
    complete_registry: RegistryFixture,
) -> None:
    def mutate(completion: dict[str, Any]) -> None:
        completion["editing_corpus_contract"]["contract_id"] = "other-contract"

    complete_registry.rewrite_completion(1, mutate)

    with pytest.raises(
        EditingV2LaneRegistryError,
        match="authoritative editing-corpus contract",
    ):
        _resolve(complete_registry)


def test_rejects_non_content_addressed_lane_root(
    complete_registry: RegistryFixture,
) -> None:
    original_root = complete_registry.registry["lanes"][0]["declared_root"]
    replacement_root = str(PurePosixPath(original_root).parent / ("0" * 64))
    original_local = complete_registry.artifact_root / Path(original_root).relative_to("/artifacts")
    replacement_local = complete_registry.artifact_root / Path(replacement_root).relative_to(
        "/artifacts"
    )
    original_local.rename(replacement_local)
    complete_registry.registry["lanes"][0]["declared_root"] = replacement_root
    complete_registry.registry["lanes"][0]["completion_manifest_path"] = (
        f"{replacement_root}/{LANE_COMPLETION_FILENAME}"
    )

    def mutate(completion: dict[str, Any]) -> None:
        completion["declared_root"] = replacement_root

    complete_registry.rewrite_completion(0, mutate)

    with pytest.raises(EditingV2LaneRegistryError, match="not content-addressed"):
        _resolve(complete_registry)
