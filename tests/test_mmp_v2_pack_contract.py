"""The V2 MMP packer must consume a frozen pool identity and cannot adopt V1 output."""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "modal_apps"))

import pack_mmp_pool_app as packer  # noqa: E402
from pack_mmp_pool_app import (  # noqa: E402
    MODAL_VOLUME_V1_MAX_CONCURRENT_WRITERS,
    PACK_OUTPUT_SCHEMA,
    PACK_REQUEST_SCHEMA,
    POOL_CONTRACT_SCHEMA,
    PackRequestError,
    build_pack_request,
    packing_concurrency_plan,
    preflight_frozen_input,
    require_unchanged_completed_shards,
    validate_pack_task_plan,
    validate_existing_output_claim,
    validate_pack_request,
    verify_local_launch_commit,
)
from compose_v4.data.packed_trace_store import write_packed_shard  # noqa: E402

POOL_PATH = "/artifacts/mining_v2/edit_pool_full.jsonl"
CONTRACT_PATH = "/artifacts/mining_v2/MMP_POOL_COMPLETE.json"
COMMIT = "a7546e2"
GATE = "0123456789abcdef"
SUPPORT = "frozen-ringcore-edit-compiler-v2"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fixture(
    tmp_path: Path,
    *,
    records: int = 2,
    complete: bool = True,
    support: str = SUPPORT,
    row_support: str | None = None,
) -> tuple[dict, Path, Path]:
    mining = tmp_path / "mining_v2"
    mining.mkdir()
    pool = mining / "edit_pool_full.jsonl"
    pool_rows = [
        {"row": 1, "metadata": {"support_contract": row_support or support}},
        {"row": 2, "metadata": {"support_contract": row_support or support}},
    ]
    pool_bytes = (json.dumps(pool_rows[0], sort_keys=True) + "\n\n").encode() + (
        json.dumps(pool_rows[1], sort_keys=True) + "\n"
    ).encode()
    pool.write_bytes(pool_bytes)
    contract = mining / "MMP_POOL_COMPLETE.json"
    contract_payload = {
        "schema": POOL_CONTRACT_SCHEMA,
        "MMP_POOL_COMPLETE": complete,
        "pool_path": POOL_PATH,
        "pool_sha256": _sha256(pool_bytes),
        "pool_records": records,
        "analogue_support_contract": support,
        "upstream_provenance": {"compiler_commit": "deadbee"},
    }
    contract.write_text(json.dumps(contract_payload, indent=2, sort_keys=True) + "\n")
    request = build_pack_request(
        pool_path=POOL_PATH,
        expected_pool_sha256=_sha256(pool_bytes),
        expected_pool_records=records,
        pool_contract_path=CONTRACT_PATH,
        expected_pool_contract_sha256=_sha256(contract.read_bytes()),
        expected_analogue_support_contract=support,
        launch_commit=COMMIT,
        gate_sha256=GATE,
    )
    return request, pool, contract


def test_valid_frozen_request_preflights_and_uses_only_v2_namespace(tmp_path):
    request, _, _ = _fixture(tmp_path)

    observed = preflight_frozen_input(request, artifact_root=tmp_path)

    assert request["schema"] == PACK_REQUEST_SCHEMA
    assert request["output_subdir"].startswith("mmp_packed_v2_")
    assert request["output_subdir"] != "mmp_packed_v1"
    assert len(request["input_identity"]) == 64
    assert len(request["output_identity"]) == 64
    assert observed == {
        "input_identity": request["input_identity"],
        "pool_sha256": request["expected_pool_sha256"],
        "pool_records": 2,
        "pool_contract_sha256": request["expected_pool_contract_sha256"],
        "row_support_contract_verified": True,
    }


def test_v2_request_builder_has_no_path_count_or_output_defaults():
    parameters = inspect.signature(build_pack_request).parameters
    assert "subdir" not in parameters
    for name in (
        "pool_path",
        "expected_pool_sha256",
        "expected_pool_records",
        "pool_contract_path",
        "expected_pool_contract_sha256",
        "expected_analogue_support_contract",
        "launch_commit",
        "gate_sha256",
    ):
        assert parameters[name].default is inspect.Parameter.empty


def test_input_and_output_identities_move_with_their_scientific_inputs(tmp_path):
    base, _, contract = _fixture(tmp_path)

    changed_count = build_pack_request(
        pool_path=POOL_PATH,
        expected_pool_sha256=base["expected_pool_sha256"],
        expected_pool_records=3,
        pool_contract_path=CONTRACT_PATH,
        expected_pool_contract_sha256=base["expected_pool_contract_sha256"],
        expected_analogue_support_contract=SUPPORT,
        launch_commit=COMMIT,
        gate_sha256=GATE,
    )
    changed_support = build_pack_request(
        pool_path=POOL_PATH,
        expected_pool_sha256=base["expected_pool_sha256"],
        expected_pool_records=2,
        pool_contract_path=CONTRACT_PATH,
        expected_pool_contract_sha256=base["expected_pool_contract_sha256"],
        expected_analogue_support_contract=SUPPORT + "-changed",
        launch_commit=COMMIT,
        gate_sha256=GATE,
    )
    changed_authorization = build_pack_request(
        pool_path=POOL_PATH,
        expected_pool_sha256=base["expected_pool_sha256"],
        expected_pool_records=2,
        pool_contract_path=CONTRACT_PATH,
        expected_pool_contract_sha256=_sha256(contract.read_bytes()),
        expected_analogue_support_contract=SUPPORT,
        launch_commit="a7546e3",
        gate_sha256=GATE,
    )

    assert changed_count["input_identity"] != base["input_identity"]
    assert changed_support["input_identity"] != base["input_identity"]
    assert changed_authorization["input_identity"] == base["input_identity"]
    assert changed_authorization["output_identity"] != base["output_identity"]
    assert changed_authorization["output_subdir"] != base["output_subdir"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("pool_path", "edit_pool_full.jsonl", "below /artifacts"),
        ("pool_path", "/artifacts/../v1/pool.jsonl", "traversal"),
        ("expected_pool_sha256", "abc", "full 64-character"),
        ("expected_pool_records", 0, "positive integer"),
        ("expected_pool_records", 2.0, "positive integer"),
        ("pool_contract_path", "/tmp/contract.json", "below /artifacts"),
        ("expected_pool_contract_sha256", "f" * 63, "full 64-character"),
        ("expected_analogue_support_contract", "", "explicit and non-empty"),
        ("launch_commit", "", "Git SHA"),
        ("gate_sha256", "abc", "16-64"),
    ],
)
def test_incomplete_or_mutable_request_is_rejected(field, value, message):
    kwargs = {
        "pool_path": POOL_PATH,
        "expected_pool_sha256": "a" * 64,
        "expected_pool_records": 2,
        "pool_contract_path": CONTRACT_PATH,
        "expected_pool_contract_sha256": "b" * 64,
        "expected_analogue_support_contract": SUPPORT,
        "launch_commit": COMMIT,
        "gate_sha256": GATE,
    }
    kwargs[field] = value
    with pytest.raises(PackRequestError, match=message):
        build_pack_request(**kwargs)


def test_serialized_identity_fields_cannot_be_changed_to_v1(tmp_path):
    request, _, _ = _fixture(tmp_path)
    tampered = dict(request)
    tampered["output_subdir"] = "mmp_packed_v1"

    with pytest.raises(PackRequestError, match="computed identities"):
        validate_pack_request(tampered)


def test_existing_namespace_may_resume_only_the_identical_request(tmp_path):
    request, _, _ = _fixture(tmp_path)
    validate_existing_output_claim(dict(request), request)

    foreign = dict(request)
    foreign["schema"] = "foreign"
    with pytest.raises(PackRequestError, match="different pack request"):
        validate_existing_output_claim(foreign, request)


def test_completed_namespace_reuse_requires_exact_packed_and_manifest_bytes():
    frozen = [
        {
            "partition": "validation",
            "shard": "shard_0000.jsonl.gz",
            "packed_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "entries": 2,
            "states": 5,
        }
    ]
    require_unchanged_completed_shards(
        {"shard_artifacts": frozen},
        list(frozen),
    )

    changed = [dict(frozen[0], packed_sha256="c" * 64)]
    with pytest.raises(PackRequestError, match="changed after freeze"):
        require_unchanged_completed_shards(
            {"shard_artifacts": frozen},
            changed,
        )
    with pytest.raises(PackRequestError, match="lacks packed-shard"):
        require_unchanged_completed_shards({}, list(frozen))


def test_local_launch_commit_must_match_clean_serialized_source(monkeypatch):
    outputs = iter(
        [
            SimpleNamespace(stdout="a7546e2" + "0" * 33 + "\n"),
            SimpleNamespace(stdout=""),
        ]
    )
    monkeypatch.setattr(packer.subprocess, "run", lambda *args, **kwargs: next(outputs))
    verify_local_launch_commit(COMMIT)


def test_local_launch_refuses_wrong_commit_or_dirty_serialized_source(monkeypatch):
    wrong = iter([SimpleNamespace(stdout="b" * 40 + "\n")])
    monkeypatch.setattr(packer.subprocess, "run", lambda *args, **kwargs: next(wrong))
    with pytest.raises(PackRequestError, match="does not identify local HEAD"):
        verify_local_launch_commit(COMMIT)

    dirty = iter(
        [
            SimpleNamespace(stdout="a7546e2" + "0" * 33 + "\n"),
            SimpleNamespace(stdout=" M modal_apps/pack_mmp_pool_app.py\n M docs/note.md\n"),
        ]
    )
    monkeypatch.setattr(packer.subprocess, "run", lambda *args, **kwargs: next(dirty))
    with pytest.raises(PackRequestError, match="dirty serialized-code tree"):
        verify_local_launch_commit(COMMIT)


def test_pool_byte_drift_fails_preflight(tmp_path):
    request, pool, _ = _fixture(tmp_path)
    extra = {"row": 3, "metadata": {"support_contract": SUPPORT}}
    pool.write_bytes(pool.read_bytes() + (json.dumps(extra) + "\n").encode())

    with pytest.raises(PackRequestError, match="pool SHA-256 mismatch"):
        preflight_frozen_input(request, artifact_root=tmp_path)


def test_pool_count_drift_fails_even_when_bytes_and_contract_sha_are_frozen(tmp_path):
    request, _, _ = _fixture(tmp_path, records=3)

    with pytest.raises(PackRequestError, match="has 2 records; expected 3"):
        preflight_frozen_input(request, artifact_root=tmp_path)


def test_incomplete_upstream_contract_cannot_be_reblessed(tmp_path):
    request, _, _ = _fixture(tmp_path, complete=False)

    with pytest.raises(PackRequestError, match="disagrees with the launch request"):
        preflight_frozen_input(request, artifact_root=tmp_path)


def test_completion_flag_must_be_json_boolean_not_integer(tmp_path):
    request, _, contract = _fixture(tmp_path)
    payload = json.loads(contract.read_text())
    payload["MMP_POOL_COMPLETE"] = 1
    contract.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    request = build_pack_request(
        pool_path=POOL_PATH,
        expected_pool_sha256=request["expected_pool_sha256"],
        expected_pool_records=2,
        pool_contract_path=CONTRACT_PATH,
        expected_pool_contract_sha256=_sha256(contract.read_bytes()),
        expected_analogue_support_contract=SUPPORT,
        launch_commit=COMMIT,
        gate_sha256=GATE,
    )

    with pytest.raises(PackRequestError, match="disagrees with the launch request"):
        preflight_frozen_input(request, artifact_root=tmp_path)


def test_every_row_must_carry_the_frozen_compiler_support_contract(tmp_path):
    request, _, _ = _fixture(tmp_path, row_support="stale-v1-contract")

    with pytest.raises(PackRequestError, match="row contract failed"):
        preflight_frozen_input(request, artifact_root=tmp_path)


def test_completion_schema_is_distinct_from_request_schema():
    assert PACK_OUTPUT_SCHEMA != PACK_REQUEST_SCHEMA


def _pack_task(index: int, *, partition: str = "train") -> dict:
    return {
        "partition": partition,
        "shard": f"shard_{index:04d}.jsonl",
        "records": 3,
        "row_ids_sha256": f"{index:016x}",
        "raw_content_sha256": f"{index:064x}",
    }


def test_volume_v1_writer_budget_queues_logical_tasks_without_changing_plan():
    tasks = [_pack_task(index) for index in range(18)]

    checked = validate_pack_task_plan(tasks, expected_shards=18)
    concurrency = packing_concurrency_plan(len(checked))

    assert checked == tasks
    assert MODAL_VOLUME_V1_MAX_CONCURRENT_WRITERS == 5
    assert concurrency == {
        "logical_tasks": 18,
        "configured_writer_container_limit": 5,
        "max_active_writer_containers": 5,
        "queued_tasks_after_first_wave": 13,
        "scheduled_waves": 4,
        "estimated_map_critical_path_seconds": 960,
        "peak_total_containers_including_driver": 6,
        "requires_exclusive_volume_writer_budget": True,
    }


def test_pack_task_plan_rejects_worker_destination_aliases():
    duplicate = [_pack_task(0), _pack_task(0)]

    with pytest.raises(PackRequestError, match="aliases worker destination"):
        validate_pack_task_plan(duplicate, expected_shards=2)


def test_v2_packed_gzip_bytes_are_stable_across_paths(tmp_path):
    first = tmp_path / "first.jsonl.gz"
    second = tmp_path / "second.jsonl.gz"
    entries = [{"trace": {"trace_id": "t0"}, "states": []}]

    manifest_a = write_packed_shard(
        first,
        entries,
        provenance={"source": "frozen"},
        deterministic_gzip=True,
    )
    manifest_b = write_packed_shard(
        second,
        entries,
        provenance={"source": "frozen"},
        deterministic_gzip=True,
    )

    assert first.read_bytes() == second.read_bytes()
    assert manifest_a == manifest_b


def test_packer_source_uses_five_writer_containers_and_shared_immutable_publication():
    source = (REPO / "modal_apps" / "pack_mmp_pool_app.py").read_text()
    packer_surface = source[source.index("@app.function", source.index("def partition_pool")) :]

    assert "max_containers=64" not in packer_surface
    assert (
        "max_containers=MODAL_VOLUME_V1_MAX_CONCURRENT_WRITERS"
        in packer_surface
    )
    assert source.count("max_containers=1") == 3
    assert "write_bytes_if_absent_modal_volume_v1" in source
    assert ".replace(manifest_path)" not in source
    assert ".replace(dest)" not in source
