"""The cache-fed rebind: same evidence as the raw oracle, none of the rescan.

The production Process-V2 rebind now reads one verified chunk of a committed
chunk cache per proof task and never opens a packed shard.  That is only safe if
the two reads are the *same* read, so this module proves it on a twenty-cell
payload rather than asserting it:

* every decoded semantic row, every proof row, every rejection row, every count
  and every census is identical between the two geometries;
* the per-task semantic digests (``proof_stream_sha256`` and
  ``v1_record_inventory_sha256``) are identical task for task, because the
  oracle is planned at the cache's own chunk size, so the two partitions of each
  shard coincide exactly;
* only the *addresses* differ -- task identity, output path, source geometry and
  the published source binding -- which is what a different source geometry is
  allowed to change and all it is allowed to change.

It also pins the three properties that make the geometry usable at scale:
concurrency 1, 20 and 40 with randomized completion order reduce byte-
identically; an uncommitted cache generation cannot be planned against; and a
restart reuses only exact valid receipts.
"""

from __future__ import annotations

import ast
import gzip
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import pytest

from compose_v4.data.editing_process_v2_rebind import (
    COMPLETION_FILENAME,
    MANIFEST_FILENAME,
    PLAN_FILENAME,
    PRODUCTION_SOURCE_GEOMETRY,
    PROOF_FILENAME,
    RECEIPT_FILENAME,
    SOURCE_GEOMETRY_CACHE_CHUNK,
    SOURCE_GEOMETRY_V1_ENTRY_RANGE,
    TASK_DIRNAME,
    ProcessV2RebindError,
    bind_process_v2_chunk_cache_generation,
    completed_process_v2_rebind_task_ids,
    execute_process_v2_rebind_task,
    plan_process_v2_rebind,
    reduce_process_v2_rebind,
    require_production_source_geometry,
    validate_process_v2_rebind_plan,
    write_process_v2_rebind_plan,
)
from compose_v4.data.editing_v2_process_v2_chunk_cache import (
    COMPLETION_FILENAME as CACHE_COMPLETION_FILENAME,
)
from compose_v4.data.editing_v2_process_v2_chunk_cache import (
    execute_process_v2_chunk_cache_task,
    plan_process_v2_chunk_cache,
    reduce_process_v2_chunk_cache,
    run_bounded_map,
    write_process_v2_chunk_cache_plan,
)
from compose_v4.data.editing_v2_process_v2_schema import canonical_bytes, canonical_sha256
from compose_v4.data.semantic_packed_trace_store import (
    SHARD_FILENAME as SEMANTIC_SHARD_FILENAME,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra_path in (str(_REPO_ROOT / "tests"), str(_REPO_ROOT / "scripts")):
    if _extra_path not in sys.path:
        sys.path.insert(0, _extra_path)

import plan_process_v2_rebind as plan_driver  # noqa: E402
import test_editing_process_v2_rebind as v1_fixture  # noqa: E402
import test_editing_v2_process_v2_completion_binder as binder_fixture  # noqa: E402

_register_extra_fixture_traces = binder_fixture._register_extra_fixture_traces
_effective_mask_authority = v1_fixture._effective_mask_authority

ROOT = _REPO_ROOT
RECORDS_PER_CHUNK = 2
# Three real traces per cell, so every cell publishes two chunks and the second
# starts above zero. `open_benzene` is admitted by the V1 migration and excluded
# by the Process-V2 fiber, so both admission outcomes are exercised in both
# geometries.
_CELL = ("cyclize_hexane", "open_benzene", "trim_methylcyclohexane")
_TASKS = tuple((lane, role, _CELL) for lane, role, _names in binder_fixture._TASKS)

# What a different source geometry is allowed to move, and nothing else.
_ADDRESS_ONLY_COMPLETION_FIELDS = {
    "source_geometry",
    "run_identity_sha256",
    "plan_sha256",
    "plan_file_sha256",
    "result_inventory",
    "result_inventory_sha256",
    "task_inventory_sha256",
    "rejected_trace_inventory_sha256",
    "cache_binding",
    "completion_sha256",
}
_ADDRESS_ONLY_RECEIPT_FIELDS = {
    "source_geometry",
    "task_source_binding",
    "run_identity_sha256",
    "task_identity_sha256",
    "receipt_sha256",
    # The manifest carries the geometry and the source binding too, so its own
    # self-hash and physical hash move with them. Every measured field inside it
    # is compared directly below.
    "manifest_sha256",
    "manifest_physical_sha256",
}
_ADDRESS_ONLY_MANIFEST_FIELDS = {
    "source_geometry",
    "task_source_binding",
    "manifest_sha256",
}


# ---- Fixtures -----------------------------------------------------------------


def _cached_payload(tmp_path: Path):
    """One V1 payload, one committed chunk cache over it."""

    payload, _completion, expectation = binder_fixture.build_migration_run(
        tmp_path / "artifacts", tasks=_TASKS
    )
    binding = binder_fixture.bind(payload, expectation)
    cache_plan = plan_process_v2_chunk_cache(
        binding,
        output_artifact_prefix="/artifacts/chunk_fed_cache",
        records_per_chunk=RECORDS_PER_CHUNK,
    )
    write_process_v2_chunk_cache_plan(cache_plan, artifact_root=payload.artifact_root)
    for task in cache_plan["tasks"]:
        execute_process_v2_chunk_cache_task(
            cache_plan, task["task_identity_sha256"], artifact_root=payload.artifact_root
        )
    cache_completion = reduce_process_v2_chunk_cache(
        cache_plan, artifact_root=payload.artifact_root
    )
    return payload, binding, cache_plan, cache_completion


def _cache_fed_plan(payload, cache_plan, *, prefix: str = "/artifacts/chunk_fed_rebind"):
    cache_binding = bind_process_v2_chunk_cache_generation(
        str(cache_plan["run_artifact_root"]),
        artifact_root=payload.artifact_root,
        repo_root=ROOT,
    )
    process_identity, builder_identity = v1_fixture._pinned_identities()
    binding = v1_fixture.bind_v1_semantic_payload(
        payload_root_artifact_path=v1_fixture.PAYLOAD_ARTIFACT_PATH,
        artifact_root=payload.artifact_root,
        pinned_process_identity=process_identity,
        pinned_builder_identity=builder_identity,
    )
    return plan_process_v2_rebind(
        binding,
        source_revision=v1_fixture._source_revision(),
        repo_root=ROOT,
        pinned_process_identity=process_identity,
        pinned_builder_identity=builder_identity,
        cache_binding=cache_binding,
        output_artifact_prefix=prefix,
        entries_per_task=RECORDS_PER_CHUNK,
    )


def _oracle_plan(payload, *, prefix: str = "/artifacts/chunk_fed_oracle"):
    """The bounded raw oracle at the cache's own chunk size, so ranges coincide."""

    process_identity, builder_identity = v1_fixture._pinned_identities()
    binding = v1_fixture.bind_v1_semantic_payload(
        payload_root_artifact_path=v1_fixture.PAYLOAD_ARTIFACT_PATH,
        artifact_root=payload.artifact_root,
        pinned_process_identity=process_identity,
        pinned_builder_identity=builder_identity,
    )
    return plan_process_v2_rebind(
        binding,
        source_revision=v1_fixture._source_revision(),
        repo_root=ROOT,
        pinned_process_identity=process_identity,
        pinned_builder_identity=builder_identity,
        output_artifact_prefix=prefix,
        entries_per_task=RECORDS_PER_CHUNK,
    )


def _execute_and_reduce(payload, plan, *, order: list[int] | None = None):
    write_process_v2_rebind_plan(plan, artifact_root=payload.artifact_root, repo_root=ROOT)
    indices = range(len(plan["tasks"])) if order is None else order
    for index in indices:
        execute_process_v2_rebind_task(
            plan,
            plan["tasks"][index]["task_identity_sha256"],
            artifact_root=payload.artifact_root,
            repo_root=ROOT,
        )
    return reduce_process_v2_rebind(plan, artifact_root=payload.artifact_root, repo_root=ROOT)


def _task_output(payload, task) -> Path:
    return payload.artifact_root / str(task["output_artifact_path"]).removeprefix("/artifacts/")


def _task_documents(payload, plan) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Every published task keyed by its ``(data_lane, split, entry_start)``."""

    documents: dict[tuple[str, str, int], dict[str, Any]] = {}
    for task in plan["tasks"]:
        output = _task_output(payload, task)
        receipt = json.loads((output / RECEIPT_FILENAME).read_bytes())
        manifest = json.loads((output / MANIFEST_FILENAME).read_bytes())
        with gzip.open(output / PROOF_FILENAME, "rb") as handle:
            proofs = [json.loads(line) for line in handle if line.strip()]
        key = (str(task["data_lane"]), str(task["split"]), int(task["entry_start"]))
        assert key not in documents
        documents[key] = {"receipt": receipt, "manifest": manifest, "proofs": proofs}
    return documents


def _semantic_result_digest(documents: dict[tuple[str, str, int], dict[str, Any]]) -> str:
    """An address-free digest of everything the run decided.

    Deliberately built from the published proof and rejection rows rather than
    from any run-level hash: every run-level hash carries the task addresses,
    which the two geometries are allowed to differ on.
    """

    rows: list[Any] = []
    for key in sorted(documents):
        entry = documents[key]
        rows.append([list(key), entry["proofs"], entry["manifest"]["rejected_traces"]])
    return canonical_sha256(rows)


# ---- Acceptance test 13: the two geometries decide the same thing -------------


def test_the_cache_path_and_the_raw_oracle_publish_the_same_evidence(
    tmp_path: Path,
) -> None:
    payload, _binding, cache_plan, _cache_completion = _cached_payload(tmp_path)
    cached_plan = _cache_fed_plan(payload, cache_plan)
    oracle_plan = _oracle_plan(payload)

    assert cached_plan["source_geometry"] == SOURCE_GEOMETRY_CACHE_CHUNK
    assert oracle_plan["source_geometry"] == SOURCE_GEOMETRY_V1_ENTRY_RANGE
    assert cached_plan["run_identity_sha256"] != oracle_plan["run_identity_sha256"]
    assert cached_plan["expected_task_count"] == oracle_plan["expected_task_count"]

    cached_completion = _execute_and_reduce(payload, cached_plan)
    oracle_completion = _execute_and_reduce(payload, oracle_plan)

    cached_documents = _task_documents(payload, cached_plan)
    oracle_documents = _task_documents(payload, oracle_plan)
    assert set(cached_documents) == set(oracle_documents)
    assert len(cached_documents) == 20 * 2

    for key in sorted(cached_documents):
        cached, oracle = cached_documents[key], oracle_documents[key]
        # Decoded rows, proof decisions, whole-trace admission and rejection
        # reason codes: identical, byte for byte.
        assert cached["proofs"] == oracle["proofs"]
        assert cached["manifest"]["rejected_traces"] == oracle["manifest"]["rejected_traces"]
        # Per-task semantic digests: identical, because the partitions coincide.
        assert (
            cached["receipt"]["proof_stream_sha256"] == oracle["receipt"]["proof_stream_sha256"]
        )
        assert (
            cached["manifest"]["v1_record_inventory_sha256"]
            == oracle["manifest"]["v1_record_inventory_sha256"]
        )
        assert cached["receipt"]["counts"] == oracle["receipt"]["counts"]
        # And only the address moves, in the manifest and in the receipt.
        assert {
            field
            for field in set(cached["manifest"]) | set(oracle["manifest"])
            if cached["manifest"].get(field) != oracle["manifest"].get(field)
        } <= _ADDRESS_ONLY_MANIFEST_FIELDS
        assert {
            field
            for field in set(cached["receipt"]) | set(oracle["receipt"])
            if cached["receipt"].get(field) != oracle["receipt"].get(field)
        } <= _ADDRESS_ONLY_RECEIPT_FIELDS

    # Every count and census, run-wide.
    for field in (
        "counts",
        "family_histogram",
        "teacher_census",
        "process_v2_atom_delete_census",
        "rejected_traces_by_code",
        "unsupported_teacher_steps_by_code",
    ):
        assert cached_completion[field] == oracle_completion[field], field
    assert cached_completion["counts"]["rejected_entries"] > 0
    assert cached_completion["counts"]["admitted_entries"] > 0

    # The address-free semantic digest of everything decided.
    assert _semantic_result_digest(cached_documents) == _semantic_result_digest(
        oracle_documents
    )

    moved = {
        field
        for field in set(cached_completion) | set(oracle_completion)
        if cached_completion.get(field) != oracle_completion.get(field)
    }
    assert moved <= _ADDRESS_ONLY_COMPLETION_FIELDS
    assert "source_geometry" in moved


def test_no_task_of_the_production_geometry_opens_a_packed_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No raw-packed-shard fallback exists on the production proof path.

    Planning does read the shards once, to bind the immutable payload by hash;
    that is the planning boundary and is stated rather than hidden. What is
    measured here is the proof-task and reduction path, which is where a
    per-range rescan would live.
    """

    payload, _binding, cache_plan, _cache_completion = _cached_payload(tmp_path)
    plan = _cache_fed_plan(payload, cache_plan)
    write_process_v2_rebind_plan(plan, artifact_root=payload.artifact_root, repo_root=ROOT)

    opened: list[str] = []
    real_path_open = Path.open
    real_gzip_open = gzip.open

    def path_open(path_self, *args: Any, **kwargs: Any):
        opened.append(Path(path_self).name)
        return real_path_open(path_self, *args, **kwargs)

    def gzip_open(filename: Any, *args: Any, **kwargs: Any):
        opened.append(Path(filename).name)
        return real_gzip_open(filename, *args, **kwargs)

    monkeypatch.setattr(Path, "open", path_open)
    monkeypatch.setattr(gzip, "open", gzip_open)
    for task in plan["tasks"]:
        execute_process_v2_rebind_task(
            plan,
            task["task_identity_sha256"],
            artifact_root=payload.artifact_root,
            repo_root=ROOT,
        )
    reduce_process_v2_rebind(plan, artifact_root=payload.artifact_root, repo_root=ROOT)

    assert SEMANTIC_SHARD_FILENAME not in opened
    # The positive control: the oracle geometry does open it, so the instrument
    # is demonstrably able to see the read it is asserting the absence of.
    oracle = _oracle_plan(payload)
    write_process_v2_rebind_plan(oracle, artifact_root=payload.artifact_root, repo_root=ROOT)
    opened.clear()
    execute_process_v2_rebind_task(
        oracle,
        oracle["tasks"][0]["task_identity_sha256"],
        artifact_root=payload.artifact_root,
        repo_root=ROOT,
    )
    assert SEMANTIC_SHARD_FILENAME in opened


def test_the_raw_range_reader_is_called_from_the_oracle_and_nowhere_else() -> None:
    """A call-graph boundary, which no behavioural test can observe.

    A module that reads the raw shard from a second place still computes the
    right answer until the day someone routes production through it. The scan
    is over the shipped source, so it holds at every commit, including one
    written by someone who has not read this file.
    """

    module = Path(_REPO_ROOT / "src/compose_v4/data/editing_process_v2_rebind.py")
    source = module.read_text(encoding="utf-8")
    tree = ast.parse(source)
    callers = sorted(
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(call.func, ast.Name)
            and call.func.id == "read_semantic_packed_artifact_range_rows"
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
        )
    )
    assert callers == ["read_v1_records_through_the_bounded_raw_oracle"]

    # And the oracle is not reachable from the production geometry: it is
    # selected only in the ``else`` of the branch that tests for it.
    prove = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_prove_entry_range"
    )
    geometry_branch = next(
        node
        for node in ast.walk(prove)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "source_geometry"
        and any(
            isinstance(comparator, ast.Name)
            and comparator.id == "SOURCE_GEOMETRY_CACHE_CHUNK"
            for comparator in node.test.comparators
        )
    )

    def _names(nodes) -> set[str]:
        return {
            call.func.id
            for statement in nodes
            for call in ast.walk(statement)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        }

    assert "read_v1_records_through_the_bounded_raw_oracle" not in _names(geometry_branch.body)
    assert "_read_cache_chunk_rows" in _names(geometry_branch.body)
    assert "read_v1_records_through_the_bounded_raw_oracle" in _names(geometry_branch.orelse)


def test_an_oracle_artifact_is_refused_as_production_evidence(tmp_path: Path) -> None:
    payload, _binding, cache_plan, _cache_completion = _cached_payload(tmp_path)
    cached_plan = _cache_fed_plan(payload, cache_plan)
    oracle_plan = _oracle_plan(payload)
    cached_completion = _execute_and_reduce(payload, cached_plan)
    oracle_completion = _execute_and_reduce(payload, oracle_plan)

    assert require_production_source_geometry(cached_plan, label="plan") == (
        PRODUCTION_SOURCE_GEOMETRY
    )
    assert require_production_source_geometry(cached_completion, label="completion")
    for document in (oracle_plan, oracle_completion):
        with pytest.raises(ProcessV2RebindError, match="bounded raw-oracle geometry"):
            require_production_source_geometry(document, label="artifact")
    # Every published task result of the oracle run says so too.
    for task in oracle_plan["tasks"]:
        receipt = json.loads((_task_output(payload, task) / RECEIPT_FILENAME).read_bytes())
        with pytest.raises(ProcessV2RebindError, match="bounded raw-oracle geometry"):
            require_production_source_geometry(receipt, label="receipt")


# ---- Acceptance test 14: schedule cannot change the artifact ------------------


# ---- Plan-level guards, exercised through resealed plans ---------------------


def _reseal_plan(plan: dict[str, Any], tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Rebuild a plan around mutated tasks, re-addressing each one.

    The task identity and output path are re-derived from the plan's own
    declared run identity, so the forged plan is internally consistent at every
    level it declares and only the guard under test can refuse it.
    """

    run_identity = str(plan["run_identity_sha256"])
    run_root = str(plan["run_artifact_root"])
    address_fields = {"entry_start", "entry_stop"} | set(
        key for key in tasks[0] if key.startswith(("cache_", "chunk_"))
    )
    rebuilt: list[dict[str, Any]] = []
    for task in tasks:
        body = {
            key: value
            for key, value in task.items()
            if key not in {"task_identity_sha256", "output_artifact_path"}
        }
        identity = canonical_sha256({"run_identity_sha256": run_identity, "task": body})
        rebuilt.append(
            {
                **body,
                "task_identity_sha256": identity,
                "output_artifact_path": f"{run_root}/{TASK_DIRNAME}/{identity}",
            }
        )
    assert address_fields
    body = {
        **{key: value for key, value in plan.items() if key != "plan_sha256"},
        "tasks": rebuilt,
        "expected_task_count": len(rebuilt),
        "task_inventory_sha256": canonical_sha256(rebuilt),
    }
    return {**body, "plan_sha256": canonical_sha256(body)}


def test_a_task_naming_another_cache_generation_is_refused(tmp_path: Path) -> None:
    payload, _binding, cache_plan, _cache_completion = _cached_payload(tmp_path)
    plan = _cache_fed_plan(payload, cache_plan)
    tasks = [dict(task) for task in plan["tasks"]]
    tasks[0]["cache_physical_identity_sha256"] = "0" * 64
    with pytest.raises(ProcessV2RebindError, match="chunk of another cache generation"):
        validate_process_v2_rebind_plan(
            _reseal_plan(plan, tasks), repo_root=ROOT
        )


def test_a_plan_that_reads_one_cache_chunk_twice_is_refused(tmp_path: Path) -> None:
    payload, _binding, cache_plan, _cache_completion = _cached_payload(tmp_path)
    plan = _cache_fed_plan(payload, cache_plan)
    tasks = [dict(task) for task in plan["tasks"]]
    donor, victim = tasks[0], next(
        task
        for task in tasks[1:]
        if int(task["chunk_row_count"]) == int(tasks[0]["chunk_row_count"])
        and task["cache_source_task_identity_sha256"]
        != tasks[0]["cache_source_task_identity_sha256"]
    )
    # The victim keeps its own entry range, so the shard partitions still cover
    # exactly; only the cache address repeats.
    for field in (
        "cache_source_artifact_path",
        "cache_source_task_identity_sha256",
        "cache_source_manifest_sha256",
        "chunk_index",
        "chunk_filename",
        "chunk_file_sha256",
        "chunk_uncompressed_sha256",
    ):
        victim[field] = donor[field]
    with pytest.raises(ProcessV2RebindError, match="reads one cache chunk twice"):
        validate_process_v2_rebind_plan(
            _reseal_plan(plan, tasks), repo_root=ROOT
        )


def _plan_against_a_resealed_binding(payload, cache_plan, mutate):
    """Plan the cache against a resealed V1 binding, so only the join can refuse.

    ``validate_v1_semantic_payload_binding`` re-derives the binding's own census
    and self-hash but reads no payload, so a resealed binding is accepted and
    the cache-to-payload join is the next thing that can say no. That is exactly
    the shape of a cache built against one payload and planned against another.
    """

    cache_binding = bind_process_v2_chunk_cache_generation(
        str(cache_plan["run_artifact_root"]),
        artifact_root=payload.artifact_root,
        repo_root=ROOT,
    )
    process_identity, builder_identity = v1_fixture._pinned_identities()
    binding = v1_fixture.bind_v1_semantic_payload(
        payload_root_artifact_path=v1_fixture.PAYLOAD_ARTIFACT_PATH,
        artifact_root=payload.artifact_root,
        pinned_process_identity=process_identity,
        pinned_builder_identity=builder_identity,
    )
    v1_tasks = [dict(task) for task in binding["v1_tasks"]]
    mutate(v1_tasks)
    body = {
        **{key: value for key, value in binding.items() if key != "binding_sha256"},
        "v1_tasks": v1_tasks,
        "v1_tasks_sha256": canonical_sha256(v1_tasks),
        "v1_entry_count": sum(int(task["v1_entries"]) for task in v1_tasks),
    }
    return plan_process_v2_rebind(
        {**body, "binding_sha256": canonical_sha256(body)},
        source_revision=v1_fixture._source_revision(),
        repo_root=ROOT,
        pinned_process_identity=process_identity,
        pinned_builder_identity=builder_identity,
        cache_binding=cache_binding,
        output_artifact_prefix="/artifacts/chunk_fed_rebind",
        entries_per_task=RECORDS_PER_CHUNK,
    )


def test_a_cache_that_does_not_cover_every_v1_row_is_refused(tmp_path: Path) -> None:
    """The join is a bijection over rows, not merely over task identities."""

    payload, _binding, cache_plan, _cache_completion = _cached_payload(tmp_path)

    def claim_one_more_record(v1_tasks: list[dict[str, Any]]) -> None:
        v1_tasks[0]["v1_entries"] = int(v1_tasks[0]["v1_entries"]) + 1

    with pytest.raises(ProcessV2RebindError, match="which declares"):
        _plan_against_a_resealed_binding(payload, cache_plan, claim_one_more_record)


def test_a_task_size_other_than_the_cache_chunk_size_is_refused(tmp_path: Path) -> None:
    """The chunk boundary is the task boundary, so the size is not a free dial."""

    payload, _binding, cache_plan, _cache_completion = _cached_payload(tmp_path)
    cache_binding = bind_process_v2_chunk_cache_generation(
        str(cache_plan["run_artifact_root"]),
        artifact_root=payload.artifact_root,
        repo_root=ROOT,
    )
    process_identity, builder_identity = v1_fixture._pinned_identities()
    binding = v1_fixture.bind_v1_semantic_payload(
        payload_root_artifact_path=v1_fixture.PAYLOAD_ARTIFACT_PATH,
        artifact_root=payload.artifact_root,
        pinned_process_identity=process_identity,
        pinned_builder_identity=builder_identity,
    )
    with pytest.raises(ProcessV2RebindError, match="not the cache's chunk size"):
        plan_process_v2_rebind(
            binding,
            source_revision=v1_fixture._source_revision(),
            repo_root=ROOT,
            pinned_process_identity=process_identity,
            pinned_builder_identity=builder_identity,
            cache_binding=cache_binding,
            output_artifact_prefix="/artifacts/chunk_fed_rebind",
            entries_per_task=RECORDS_PER_CHUNK + 1,
        )


def test_a_published_result_that_relabels_its_geometry_is_refused(tmp_path: Path) -> None:
    """The published manifest and receipt must agree on where the rows came from.

    Both documents are resealed, and the receipt is re-pointed at the mutated
    manifest, so the self-hash and physical-hash bindings all agree and the
    geometry comparison is the only thing left that can refuse. Resealing only
    the manifest tripped the manifest-binds-its-receipt check instead, which is
    a different guard -- the mutation survived that version of this test.
    """

    payload, _binding, cache_plan, _cache_completion = _cached_payload(tmp_path)
    plan = _cache_fed_plan(payload, cache_plan)
    _execute_and_reduce(payload, plan)
    output = _task_output(payload, plan["tasks"][2])
    manifest_path, receipt_path = output / MANIFEST_FILENAME, output / RECEIPT_FILENAME
    original_manifest = manifest_path.read_bytes()
    original_receipt = receipt_path.read_bytes()

    def _republish(mutate) -> None:
        manifest = json.loads(original_manifest)
        mutate(manifest)
        body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        manifest = {**body, "manifest_sha256": canonical_sha256(body)}
        manifest_bytes = canonical_bytes(manifest) + b"\n"
        manifest_path.write_bytes(manifest_bytes)

        receipt = json.loads(original_receipt)
        receipt["manifest_sha256"] = manifest["manifest_sha256"]
        receipt["manifest_physical_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
        receipt_body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        receipt_path.write_bytes(
            canonical_bytes({**receipt_body, "receipt_sha256": canonical_sha256(receipt_body)})
            + b"\n"
        )

    def _relabel_geometry(manifest: dict[str, Any]) -> None:
        manifest["source_geometry"] = SOURCE_GEOMETRY_V1_ENTRY_RANGE

    def _relabel_source_binding(manifest: dict[str, Any]) -> None:
        manifest["task_source_binding"] = {
            **manifest["task_source_binding"],
            "chunk_index": 999,
        }

    for mutate in (_relabel_geometry, _relabel_source_binding):
        _republish(mutate)
        with pytest.raises(ProcessV2RebindError, match="does not bind its receipt"):
            completed_process_v2_rebind_task_ids(
                plan, artifact_root=payload.artifact_root, repo_root=ROOT
            )

    manifest_path.write_bytes(original_manifest)
    receipt_path.write_bytes(original_receipt)
    assert completed_process_v2_rebind_task_ids(
        plan, artifact_root=payload.artifact_root, repo_root=ROOT
    )


def test_a_cache_of_another_shard_for_the_same_task_is_refused(tmp_path: Path) -> None:
    """Same V1 task identity, different packed shard: still not this payload."""

    payload, _binding, cache_plan, _cache_completion = _cached_payload(tmp_path)

    def relabel_the_shard(v1_tasks: list[dict[str, Any]]) -> None:
        v1_tasks[0]["v1_semantic_shard_sha256"] = "0" * 64

    with pytest.raises(ProcessV2RebindError, match="semantic_shard_sha256"):
        _plan_against_a_resealed_binding(payload, cache_plan, relabel_the_shard)


@pytest.mark.parametrize("max_map_containers", [1, 20, 40])
def test_concurrency_and_randomized_completion_order_reduce_byte_identically(
    tmp_path: Path, max_map_containers: int
) -> None:
    reference_payload, _b, reference_cache, _c = _cached_payload(tmp_path / "reference")
    reference_plan = _cache_fed_plan(reference_payload, reference_cache)
    reference = _execute_and_reduce(reference_payload, reference_plan)

    payload, _binding, cache_plan, _cache_completion = _cached_payload(
        tmp_path / f"w{max_map_containers}"
    )
    plan = _cache_fed_plan(payload, cache_plan)
    assert plan["run_identity_sha256"] == reference_plan["run_identity_sha256"]
    write_process_v2_rebind_plan(plan, artifact_root=payload.artifact_root, repo_root=ROOT)

    task_ids = [str(task["task_identity_sha256"]) for task in plan["tasks"]]
    shuffled = list(task_ids)
    random.Random(max_map_containers).shuffle(shuffled)
    submitted: list[tuple[str, ...]] = []

    def submit(wave: tuple[str, ...]):
        submitted.append(wave)
        # Inside a wave, results arrive in whatever order the fleet finishes.
        order = list(wave)
        random.Random(len(submitted)).shuffle(order)
        return [
            execute_process_v2_rebind_task(
                plan, task_id, artifact_root=payload.artifact_root, repo_root=ROOT
            )
            for task_id in order
        ]

    run_bounded_map(shuffled, max_map_containers=max_map_containers, submit=submit)
    assert max(len(wave) for wave in submitted) <= max_map_containers
    assert len(submitted) == -(-len(task_ids) // max_map_containers)

    completion = reduce_process_v2_rebind(
        plan, artifact_root=payload.artifact_root, repo_root=ROOT
    )
    assert canonical_bytes(completion) == canonical_bytes(reference)


def test_the_reduction_is_ordered_by_lane_split_and_entry_start(tmp_path: Path) -> None:
    """The reduction sorts; it does not inherit whatever order the plan had.

    The two geometries make that separable.  The cache plan is already emitted
    in ``(data_lane, split, entry_start)`` order, because that is the order the
    cache generation enumerates its chunks in -- so it proves the reduction is
    *consistent* with the key but not that it applies it.  The oracle plan is
    ordered by the V1 task's content hash, which is unrelated to the lane/role
    grid, so its reduction genuinely has to reorder.
    """

    payload, _binding, cache_plan, _cache_completion = _cached_payload(tmp_path)

    plan = _cache_fed_plan(payload, cache_plan)
    completion = _execute_and_reduce(
        payload, plan, order=list(reversed(range(len(plan["tasks"]))))
    )
    observed = [
        (str(row["data_lane"]), str(row["split"]), int(row["entry_start"]))
        for row in completion["result_inventory"]
    ]
    assert observed == sorted(observed)

    oracle = _oracle_plan(payload)
    planned = [
        (str(task["data_lane"]), str(task["split"]), int(task["entry_start"]))
        for task in oracle["tasks"]
    ]
    assert planned != sorted(planned), (
        "the oracle plan must not already be in reduction order, or this proves nothing"
    )
    oracle_completion = _execute_and_reduce(
        payload, oracle, order=list(reversed(range(len(oracle["tasks"]))))
    )
    oracle_observed = [
        (str(row["data_lane"]), str(row["split"]), int(row["entry_start"]))
        for row in oracle_completion["result_inventory"]
    ]
    assert oracle_observed == sorted(oracle_observed) == observed


# ---- Acceptance test 15: interruption and restart -----------------------------


def test_an_uncommitted_cache_generation_cannot_be_planned_against(
    tmp_path: Path,
) -> None:
    payload, _binding, cache_plan, _cache_completion = _cached_payload(tmp_path)
    run_root = payload.artifact_root / str(cache_plan["run_artifact_root"]).removeprefix(
        "/artifacts/"
    )
    (run_root / CACHE_COMPLETION_FILENAME).unlink()
    with pytest.raises(ProcessV2RebindError, match="cannot be bound"):
        _cache_fed_plan(payload, cache_plan)


def test_a_crashed_task_publishes_nothing_and_a_restart_reuses_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, _binding, cache_plan, _cache_completion = _cached_payload(tmp_path)
    plan = _cache_fed_plan(payload, cache_plan)
    write_process_v2_rebind_plan(plan, artifact_root=payload.artifact_root, repo_root=ROOT)
    victim = plan["tasks"][3]

    class _Dies(Exception):
        pass

    real_gzip_open = gzip.open

    def die_on_the_chunk(filename: Any, *args: Any, **kwargs: Any):
        if Path(filename).name == victim["chunk_filename"]:
            raise _Dies("container terminated")
        return real_gzip_open(filename, *args, **kwargs)

    monkeypatch.setattr(gzip, "open", die_on_the_chunk)
    with pytest.raises(_Dies):
        execute_process_v2_rebind_task(
            plan,
            victim["task_identity_sha256"],
            artifact_root=payload.artifact_root,
            repo_root=ROOT,
        )
    monkeypatch.undo()

    task_root = payload.artifact_root / str(plan["run_artifact_root"]).removeprefix(
        "/artifacts/"
    ) / TASK_DIRNAME
    assert not (task_root / victim["task_identity_sha256"]).exists()
    assert (
        completed_process_v2_rebind_task_ids(
            plan, artifact_root=payload.artifact_root, repo_root=ROOT
        )
        == set()
    )

    first = [
        execute_process_v2_rebind_task(
            plan,
            task["task_identity_sha256"],
            artifact_root=payload.artifact_root,
            repo_root=ROOT,
        )
        for task in plan["tasks"]
    ]
    assert all(result["reused"] is False for result in first)
    second = [
        execute_process_v2_rebind_task(
            plan,
            task["task_identity_sha256"],
            artifact_root=payload.artifact_root,
            repo_root=ROOT,
        )
        for task in plan["tasks"]
    ]
    assert all(result["reused"] is True for result in second)
    assert [result["receipt_sha256"] for result in first] == [
        result["receipt_sha256"] for result in second
    ]


def test_a_restart_reuses_only_exact_valid_receipts(tmp_path: Path) -> None:
    payload, _binding, cache_plan, _cache_completion = _cached_payload(tmp_path)
    plan = _cache_fed_plan(payload, cache_plan)
    _execute_and_reduce(payload, plan)
    output = _task_output(payload, plan["tasks"][5])

    # A receipt whose bytes moved after publication is not a reusable result.
    receipt_path = output / RECEIPT_FILENAME
    original = receipt_path.read_bytes()
    receipt = json.loads(original)
    receipt["counts"] = {**receipt["counts"], "admitted_entries": 999}
    receipt_path.write_bytes(canonical_bytes(receipt) + b"\n")
    with pytest.raises(ProcessV2RebindError):
        completed_process_v2_rebind_task_ids(
            plan, artifact_root=payload.artifact_root, repo_root=ROOT
        )
    receipt_path.write_bytes(original)
    assert len(
        completed_process_v2_rebind_task_ids(
            plan, artifact_root=payload.artifact_root, repo_root=ROOT
        )
    ) == len(plan["tasks"])

    # A published proof ledger whose bytes moved is not one either.
    proof_path = output / PROOF_FILENAME
    proof_bytes = proof_path.read_bytes()
    proof_path.write_bytes(proof_bytes + b"\x00")
    with pytest.raises(ProcessV2RebindError):
        completed_process_v2_rebind_task_ids(
            plan, artifact_root=payload.artifact_root, repo_root=ROOT
        )
    proof_path.write_bytes(proof_bytes)


# ---- The plan driver's production entry point ---------------------------------


def test_the_production_plan_driver_derives_the_payload_root(tmp_path: Path) -> None:
    """Defect #10: the payload root is derived and bound, never supplied."""

    payload, _completion_doc, expectation = binder_fixture.build_migration_run(
        tmp_path / "artifacts", tasks=_TASKS
    )
    binding = binder_fixture.bind(payload, expectation)
    cache_plan = plan_process_v2_chunk_cache(
        binding,
        output_artifact_prefix="/artifacts/chunk_fed_cache",
        records_per_chunk=RECORDS_PER_CHUNK,
    )
    write_process_v2_chunk_cache_plan(cache_plan, artifact_root=payload.artifact_root)
    for task in cache_plan["tasks"]:
        execute_process_v2_chunk_cache_task(
            cache_plan, task["task_identity_sha256"], artifact_root=payload.artifact_root
        )
    reduce_process_v2_chunk_cache(cache_plan, artifact_root=payload.artifact_root)

    plan = plan_driver.build_cache_fed_plan(
        artifact_root=payload.artifact_root,
        cache_run_artifact_root=str(cache_plan["run_artifact_root"]),
        expectation=expectation,
        output_artifact_prefix="/artifacts/chunk_fed_rebind",
        source_revision=v1_fixture._source_revision(),
        repo_root=ROOT,
    )
    assert plan["source_geometry"] == PRODUCTION_SOURCE_GEOMETRY
    assert plan["entries_per_task"] == RECORDS_PER_CHUNK
    assert plan["v1_payload_binding"]["payload_root_artifact_path"] == (
        v1_fixture.PAYLOAD_ARTIFACT_PATH
    )
    envelope = plan_driver.plan_envelope(plan, written=None)
    assert envelope["source_geometry"] == PRODUCTION_SOURCE_GEOMETRY
    assert envelope["cache_completion_sha256"] == plan["cache_binding"]["cache_completion_sha256"]
    assert [value for key, value in envelope.items() if key.endswith("_authorized")] == [
        False
    ] * len([key for key in envelope if key.endswith("_authorized")])

    completion = _execute_and_reduce(payload, plan)
    published = json.loads(
        (
            payload.artifact_root
            / str(plan["run_artifact_root"]).removeprefix("/artifacts/")
            / COMPLETION_FILENAME
        ).read_bytes()
    )
    assert published == completion
    assert (
        payload.artifact_root
        / str(plan["run_artifact_root"]).removeprefix("/artifacts/")
        / PLAN_FILENAME
    ).is_file()


def test_a_cache_of_another_payload_cannot_be_planned_against(tmp_path: Path) -> None:
    """A cache is bound to the payload it cached, not to whichever is present."""

    payload, _completion, expectation = binder_fixture.build_migration_run(
        tmp_path / "artifacts", tasks=_TASKS
    )
    # A second, genuinely different payload: two traces per cell rather than
    # three, so its cache holds a different census and a different generation.
    other_tasks = tuple((lane, role, _CELL[:2]) for lane, role, _names in binder_fixture._TASKS)
    other_payload, _other_completion, other_expectation = binder_fixture.build_migration_run(
        tmp_path / "other", tasks=other_tasks
    )
    other_binding = binder_fixture.bind(other_payload, other_expectation)
    other_cache_plan = plan_process_v2_chunk_cache(
        other_binding,
        output_artifact_prefix="/artifacts/chunk_fed_cache",
        records_per_chunk=RECORDS_PER_CHUNK,
    )
    write_process_v2_chunk_cache_plan(
        other_cache_plan, artifact_root=other_payload.artifact_root
    )
    for task in other_cache_plan["tasks"]:
        execute_process_v2_chunk_cache_task(
            other_cache_plan,
            task["task_identity_sha256"],
            artifact_root=other_payload.artifact_root,
        )
    reduce_process_v2_chunk_cache(other_cache_plan, artifact_root=other_payload.artifact_root)

    # Copy the foreign cache generation under the first artifact root, so the
    # path resolves and only the identities and the census disagree. The two
    # generations are content-addressed differently, so they coexist.
    foreign_root = str(other_cache_plan["run_artifact_root"]).removeprefix("/artifacts/")
    for path in sorted((other_payload.artifact_root / foreign_root).rglob("*")):
        if path.is_file():
            target = payload.artifact_root / path.relative_to(other_payload.artifact_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())

    with pytest.raises(plan_driver.PlanDriverError, match="not a cache of this payload"):
        plan_driver.build_cache_fed_plan(
            artifact_root=payload.artifact_root,
            cache_run_artifact_root=str(other_cache_plan["run_artifact_root"]),
            expectation=expectation,
            output_artifact_prefix="/artifacts/chunk_fed_rebind",
            source_revision=v1_fixture._source_revision(),
            repo_root=ROOT,
        )
