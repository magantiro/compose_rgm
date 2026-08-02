"""Proof, restart, refusal, and provenance tests for the Process-V2 rebind.

Every fixture payload is produced by the production writers over real small
molecules: a legacy packed shard is written by ``write_packed_shard`` and then
migrated by ``materialize_frozen_packed_shard``, which itself publishes through
``write_semantic_packed_artifact``.  No V1 artifact is hand-written.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.data.editing_corpus_contract import (
    REQUIRED_DATA_LANES,
    REQUIRED_PARTITION_ROLES,
)
from compose_v4.data.editing_process_v2_rebind import (
    COMPLETION_FILENAME,
    MANIFEST_FILENAME,
    PLAN_FILENAME,
    PROOF_FILENAME,
    RECEIPT_FILENAME,
    TASK_DIRNAME,
    ProcessV2RebindError,
    ProcessV2RebindIncomplete,
    ProcessV2RebindMismatch,
    ProcessV2RebindReasonCode,
    bind_v1_semantic_payload,
    build_process_v2_rebind_source_revision,
    completed_process_v2_rebind_task_ids,
    execute_process_v2_rebind_task,
    independent_process_v2_atom_delete_slots,
    plan_process_v2_rebind,
    reduce_process_v2_rebind,
    validate_process_v2_rebind_task_ranges,
    validate_v1_semantic_payload_binding,
    write_process_v2_rebind_plan,
)
from compose_v4.data.packed_trace_store import (
    build_packed_entry,
    manifest_path_for,
    write_packed_shard,
)
from compose_v4.data.semantic_packed_trace_store import (
    COMPLETION_FILENAME as SEMANTIC_COMPLETION_FILENAME,
)
from compose_v4.data.semantic_packed_trace_store import (
    MANIFEST_FILENAME as SEMANTIC_MANIFEST_FILENAME,
)
from compose_v4.data.semantic_packed_trace_store import (
    SHARD_FILENAME as SEMANTIC_SHARD_FILENAME,
)
from compose_v4.data.semantic_packed_trace_store import (
    SemanticPackedStoreError,
    load_semantic_packed_manifest,
    semantic_packed_builder_identity,
)
from compose_v4.data.semantic_trace_migration_mapreduce import (
    COMPLETION_FILENAME as V1_RUN_COMPLETION_FILENAME,
)
from compose_v4.data.semantic_trace_migration_mapreduce import (
    PLAN_FILENAME as V1_RUN_PLAN_FILENAME,
)
from compose_v4.data.semantic_trace_migration_materializer import (
    RECEIPT_FILENAME as V1_RECEIPT_FILENAME,
)
from compose_v4.data.semantic_trace_migration_materializer import (
    SEMANTIC_ARTIFACT_DIRNAME,
    SemanticTraceMigrationTask,
    materialize_frozen_packed_shard,
)
from compose_v4.rewrite import action_codec_v4
from compose_v4.rewrite import editing_v2_process_identity as process_identity_module
from compose_v4.rewrite.editing_v2_process_identity import (
    PROCESS_IDENTITY_SCHEMA,
    EditingV2ProcessIdentityError,
    editing_v2_process_identity,
)
from compose_v4.rewrite.kernel import de_novo_rewrite_system
from compose_v4.rewrite.operators import AtomDelete, BondInsert
from compose_v4.rewrite.process_v2_atom_delete import (
    process_v2_connected_nonleaf_atom_delete_mask,
)
from compose_v4.rewrite.progress import TraceProgressCTMC
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace
from compose_v4.rewrite.trace_shard import encode_trace_record

ROOT = Path(__file__).resolve().parents[1]
SLOTS = 16
PAYLOAD_ARTIFACT_PATH = "/artifacts/v1_payload"
PROCESS_V2_SEMANTICS = "semantic_editing_v2_v2"
PROCESS_V2_CONTRACT_RELATIVE_PATH = "configs/editing_v2_semantic_process_v2.json"

# One real molecule and one real legacy edit path per fixture trace.  The last
# one deletes a ring atom of cyclohexane, so the payload deliberately contains a
# teacher that only the Process-V2 fiber can represent.
_FIXTURE_TRACES = {
    "cyclize_hexane": ("CCCCCC", (("bond_insert", BondInsert(0, 5, 1)),)),
    "cyclize_fluoropentane": ("CCCCCF", (("bond_insert", BondInsert(0, 4, 1)),)),
    "trim_methylcyclohexane": (
        "CCCCCCC",
        (("bond_insert", BondInsert(0, 5, 1)), ("atom_delete", AtomDelete(6))),
    ),
    "open_cyclohexane": (
        "CCCCCC",
        (("bond_insert", BondInsert(0, 5, 1)), ("atom_delete", AtomDelete(0))),
    ),
}
_V1_TASKS = (
    (REQUIRED_DATA_LANES[0], REQUIRED_PARTITION_ROLES[0], ("cyclize_hexane", "cyclize_fluoropentane")),
    (
        REQUIRED_DATA_LANES[1],
        REQUIRED_PARTITION_ROLES[1],
        ("trim_methylcyclohexane", "open_cyclohexane"),
    ),
)


# ---- Frozen-identity entry points supplied by the Process-V2 contract module --


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _self_hash(value: Mapping[str, object], field: str) -> str:
    return _canonical_sha256({key: item for key, item in value.items() if key != field})


def _local_validate_frozen_process_identity(payload: object) -> dict[str, object]:
    """Local equivalent of the frozen-identity validator, per its frozen contract."""

    if not isinstance(payload, Mapping):
        raise EditingV2ProcessIdentityError("frozen process identity must be an object")
    frozen = dict(payload)
    if set(frozen) != set(editing_v2_process_identity()):
        raise EditingV2ProcessIdentityError("frozen process identity fields disagree")
    if frozen.get("schema") != PROCESS_IDENTITY_SCHEMA:
        raise EditingV2ProcessIdentityError("frozen process identity schema disagrees")
    if frozen.get("process_identity_sha256") != _self_hash(frozen, "process_identity_sha256"):
        raise EditingV2ProcessIdentityError("frozen process identity self-hash disagrees")
    return frozen


def _local_editing_process_v2_identity() -> dict[str, object]:
    """Local stand-in for the live Process-V2 identity, distinct from V1."""

    body = {
        **{
            key: value
            for key, value in editing_v2_process_identity().items()
            if key != "process_identity_sha256"
        },
        "process_semantics": PROCESS_V2_SEMANTICS,
        "contract_relative_path": PROCESS_V2_CONTRACT_RELATIVE_PATH,
    }
    return {**body, "process_identity_sha256": _canonical_sha256(body)}


@pytest.fixture(autouse=True)
def _frozen_identity_entry_points(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install the frozen-identity entry points only while they are still absent.

    ``validate_frozen_process_identity`` and ``editing_process_v2_identity`` ship
    with the Process-V2 contract module.  Once they exist this fixture installs
    nothing and every test below exercises the real implementations.
    """

    for name, local in (
        ("validate_frozen_process_identity", _local_validate_frozen_process_identity),
        ("editing_process_v2_identity", _local_editing_process_v2_identity),
    ):
        if not hasattr(process_identity_module, name):
            monkeypatch.setattr(process_identity_module, name, local, raising=False)


# ---- Real V1 payload fixture --------------------------------------------------


@dataclass(frozen=True)
class _V1Payload:
    artifact_root: Path
    payload_root: Path
    v1_task_identities: tuple[str, ...]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _legacy_trace(name: str) -> tuple[RewriteTrace, tuple]:
    smiles, steps = _FIXTURE_TRACES[name]
    runtime = de_novo_rewrite_system()
    state = pad_molecular_graph(smiles_to_molecular_graph(smiles), SLOTS)
    states = [state]
    rewrite_steps = []
    for rule, action in steps:
        state = runtime.apply(state, rule, action)
        states.append(state)
        rewrite_steps.append(RewriteStep(rule, action))
    trace = RewriteTrace(states[0], states[-1], tuple(rewrite_steps), {"fixture": name})
    return trace, tuple(states)


def _v1_task_identity(index: int) -> str:
    return hashlib.sha256(f"process-v2-rebind-fixture-task-{index}".encode()).hexdigest()


def _build_v1_payload(artifact_root: Path) -> _V1Payload:
    payload_root = artifact_root / "v1_payload"
    identities: list[str] = []
    for index, (lane, split, names) in enumerate(_V1_TASKS):
        shard = artifact_root / "v1_source" / lane / split / "shard_0000.jsonl.gz"
        shard.parent.mkdir(parents=True, exist_ok=True)
        entries = []
        for offset, name in enumerate(names):
            trace, _ = _legacy_trace(name)
            envelope = encode_trace_record(
                trace,
                n_slots=SLOTS,
                seed=1000 + 10 * index + offset,
                trace_id=f"legacy-{index}-{offset}",
                partition=split,
                layer=lane,
                extra=dict(trace.metadata),
            )
            entries.append(build_packed_entry(envelope, TraceProgressCTMC(trace)))
        write_packed_shard(
            shard,
            entries,
            provenance={"fixture_task": index},
            deterministic_gzip=True,
        )
        identity = _v1_task_identity(index)
        materialize_frozen_packed_shard(
            SemanticTraceMigrationTask(
                source_path=shard,
                source_shard_sha256=_sha256(shard),
                source_manifest_sha256=_sha256(manifest_path_for(shard)),
                source_overlay_sha256=None,
                source_unified_manifest_sha256=hashlib.sha256(b"fixture-unified").hexdigest(),
                source_entry_count=len(names),
                implementation_revision="a" * 40,
                data_lane=lane,
                split=split,
                output_dir=payload_root / TASK_DIRNAME / identity,
            )
        )
        identities.append(identity)
    return _V1Payload(artifact_root, payload_root, tuple(identities))


def _pinned_identities() -> tuple[dict, dict]:
    return (
        json.loads(json.dumps(editing_v2_process_identity())),
        json.loads(json.dumps(semantic_packed_builder_identity())),
    )


def _source_revision() -> dict:
    return build_process_v2_rebind_source_revision(
        commit="a" * 40,
        tree="b" * 40,
        repo_root=ROOT,
        worktree_clean=True,
    )


def _plan_for(payload: _V1Payload, *, entries_per_task: int = 1) -> dict:
    process_identity, builder_identity = _pinned_identities()
    binding = bind_v1_semantic_payload(
        payload_root_artifact_path=PAYLOAD_ARTIFACT_PATH,
        artifact_root=payload.artifact_root,
        pinned_process_identity=process_identity,
        pinned_builder_identity=builder_identity,
    )
    return plan_process_v2_rebind(
        binding,
        source_revision=_source_revision(),
        repo_root=ROOT,
        pinned_process_identity=process_identity,
        pinned_builder_identity=builder_identity,
        output_artifact_prefix="/artifacts/rebind_fixture",
        entries_per_task=entries_per_task,
    )


def _run_root(payload: _V1Payload, plan: Mapping[str, object]) -> Path:
    return payload.artifact_root / PurePosixPath(str(plan["run_artifact_root"])).relative_to(
        "/artifacts"
    )


def _execute_all(payload: _V1Payload, plan: Mapping[str, object]) -> None:
    write_process_v2_rebind_plan(plan, artifact_root=payload.artifact_root, repo_root=ROOT)
    for task in plan["tasks"]:
        execute_process_v2_rebind_task(
            plan,
            task["task_identity_sha256"],
            artifact_root=payload.artifact_root,
            repo_root=ROOT,
        )


def _semantic_dir(payload: _V1Payload, index: int) -> Path:
    return (
        payload.payload_root
        / TASK_DIRNAME
        / payload.v1_task_identities[index]
        / SEMANTIC_ARTIFACT_DIRNAME
    )


def _read_v1_records(semantic_dir: Path) -> list[dict]:
    with gzip.open(semantic_dir / SEMANTIC_SHARD_FILENAME, "rb") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _canonical_line(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _reseal_v1_task(task_dir: Path, mutate) -> None:
    """Apply ``mutate`` to every V1 record and re-seal every content address.

    The physical hashes, the manifest, the completion receipt and the migration
    receipt are all recomputed, so the tampered chemistry survives every
    transport gate and has to be caught by the rebind proof itself.
    """

    semantic_dir = task_dir / SEMANTIC_ARTIFACT_DIRNAME
    lines: list[bytes] = []
    for record in _read_v1_records(semantic_dir):
        mutate(record)
        record["record_sha256"] = _self_hash(record, "record_sha256")
        lines.append(_canonical_line(record))
    shard_path = semantic_dir / SEMANTIC_SHARD_FILENAME
    with shard_path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as compressed:
            for line in lines:
                compressed.write(line)
    stream = hashlib.sha256()
    for line in lines:
        stream.update(line)

    manifest_path = semantic_dir / SEMANTIC_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["shard_sha256"] = _sha256(shard_path)
    manifest["record_stream_sha256"] = stream.hexdigest()
    manifest["manifest_sha256"] = _self_hash(manifest, "manifest_sha256")
    manifest_bytes = _canonical_line(manifest)
    manifest_path.write_bytes(manifest_bytes)

    completion_path = semantic_dir / SEMANTIC_COMPLETION_FILENAME
    completion = json.loads(completion_path.read_text())
    completion["shard_sha256"] = manifest["shard_sha256"]
    completion["manifest_physical_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    completion["completion_sha256"] = _self_hash(completion, "completion_sha256")
    completion_path.write_bytes(_canonical_line(completion))

    receipt_path = task_dir / V1_RECEIPT_FILENAME
    receipt = json.loads(receipt_path.read_text())
    receipt["semantic_shard_sha256"] = manifest["shard_sha256"]
    receipt["semantic_manifest_sha256"] = completion["manifest_physical_sha256"]
    receipt["receipt_sha256"] = _self_hash(receipt, "receipt_sha256")
    receipt_path.write_bytes(_canonical_line(receipt))


def _refuses_with(payload: _V1Payload, code: ProcessV2RebindReasonCode) -> None:
    """Require the refusing range to publish nothing at all, then stay incomplete."""

    plan = _plan_for(payload)
    write_process_v2_rebind_plan(plan, artifact_root=payload.artifact_root, repo_root=ROOT)
    task_root = _run_root(payload, plan) / TASK_DIRNAME
    refused: str | None = None
    findings: tuple = ()
    message = ""
    for task in plan["tasks"]:
        try:
            execute_process_v2_rebind_task(
                plan,
                task["task_identity_sha256"],
                artifact_root=payload.artifact_root,
                repo_root=ROOT,
            )
        except ProcessV2RebindMismatch as error:
            refused = task["task_identity_sha256"]
            findings = error.findings
            message = str(error)
            break
    assert refused is not None, "the corrupted payload must refuse to publish"
    assert code in {finding.code for finding in findings}
    assert code.value in message
    assert not (task_root / refused).exists(), "a refused range must leave no output"
    published = {path.name for path in task_root.iterdir()} if task_root.exists() else set()
    assert not any(name.endswith(".staging") for name in published)
    assert published <= {task["task_identity_sha256"] for task in plan["tasks"]} - {refused}
    with pytest.raises(ProcessV2RebindIncomplete):
        reduce_process_v2_rebind(plan, artifact_root=payload.artifact_root, repo_root=ROOT)


# ---- Independent oracle -------------------------------------------------------


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("C1CCCCC1", (0, 1, 2, 3, 4, 5)),
        ("C1CCOCC1", (0, 1, 2, 3, 4, 5)),
        ("c1ccccc1", ()),
        ("Cc1ccccc1", ()),
        ("CC1CCCCC1", (2, 3, 4, 5, 6)),
        ("CCO", ()),
        ("C", ()),
    ],
)
def test_independent_oracle_reproduces_the_production_process_v2_fiber(
    smiles: str,
    expected: tuple[int, ...],
) -> None:
    state = pad_molecular_graph(smiles_to_molecular_graph(smiles), SLOTS)
    authority = tuple(
        int(slot)
        for slot, admitted in enumerate(process_v2_connected_nonleaf_atom_delete_mask(state))
        if admitted
    )
    assert independent_process_v2_atom_delete_slots(state) == expected
    assert authority == expected


# ---- Happy path ---------------------------------------------------------------


def test_rebind_proves_the_whole_v1_payload_and_publishes_a_v2_completion(
    tmp_path: Path,
) -> None:
    payload = _build_v1_payload(tmp_path / "artifacts")
    plan = _plan_for(payload)
    assert plan["training_authorized"] is False
    assert plan["expected_entry_count"] == 4
    assert plan["expected_task_count"] == 4
    plan_path = write_process_v2_rebind_plan(
        plan,
        artifact_root=payload.artifact_root,
        repo_root=ROOT,
    )
    for task in plan["tasks"]:
        result = execute_process_v2_rebind_task(
            plan,
            task["task_identity_sha256"],
            artifact_root=payload.artifact_root,
            repo_root=ROOT,
        )
        assert result["reused"] is False
        assert result["counts"]["entries"] == 1

    completion = reduce_process_v2_rebind(
        plan,
        artifact_root=payload.artifact_root,
        repo_root=ROOT,
    )
    run_root = _run_root(payload, plan)
    assert plan_path == run_root / PLAN_FILENAME
    assert json.loads((run_root / COMPLETION_FILENAME).read_text()) == completion
    assert completion["plan_file_sha256"] == _sha256(plan_path)
    assert completion["task_count"] == 4
    assert completion["counts"]["entries"] == 4
    assert completion["counts"]["transitions"] == 6
    assert completion["mismatches_by_code"] == {}
    assert completion["unsupported_teachers_by_code"] == {}
    assert completion["completion_sha256"] == _self_hash(completion, "completion_sha256")
    assert (
        completion["process_v2_identity_sha256"]
        != completion["pinned_process_identity_sha256"]
    ), "the V2 completion must not be bound to the superseded V1 identity"
    assert (
        reduce_process_v2_rebind(plan, artifact_root=payload.artifact_root, repo_root=ROOT)
        == completion
    )


def test_rebind_grants_no_training_authority_at_any_level(tmp_path: Path) -> None:
    payload = _build_v1_payload(tmp_path / "artifacts")
    plan = _plan_for(payload)
    _execute_all(payload, plan)
    completion = reduce_process_v2_rebind(
        plan,
        artifact_root=payload.artifact_root,
        repo_root=ROOT,
    )
    assert completion["training_authorized"] is False
    assert completion["gate_zero_run"] is False
    assert "NO_TRAINING_AUTHORITY" in completion["status"]
    run_root = _run_root(payload, plan)
    for task in plan["tasks"]:
        output = run_root / TASK_DIRNAME / task["task_identity_sha256"]
        for filename in (RECEIPT_FILENAME, MANIFEST_FILENAME):
            document = json.loads((output / filename).read_text())
            assert document["training_authorized"] is False
            assert "NO_TRAINING_AUTHORITY" in document["status"]


def test_published_rebind_bytes_are_stable_across_an_independent_rerun(
    tmp_path: Path,
) -> None:
    roots = []
    for name in ("first", "second"):
        payload = _build_v1_payload(tmp_path / name / "artifacts")
        plan = _plan_for(payload)
        _execute_all(payload, plan)
        reduce_process_v2_rebind(plan, artifact_root=payload.artifact_root, repo_root=ROOT)
        roots.append(_run_root(payload, plan))
    first, second = roots
    assert first.relative_to(first.parents[1]) == second.relative_to(second.parents[1])
    produced = sorted(path.relative_to(first) for path in first.rglob("*") if path.is_file())
    assert produced
    assert produced == sorted(
        path.relative_to(second) for path in second.rglob("*") if path.is_file()
    )
    for relative in produced:
        assert (first / relative).read_bytes() == (second / relative).read_bytes(), relative


# ---- Restart safety -----------------------------------------------------------


def test_completed_task_is_reused_exactly_and_staging_never_blocks_a_retry(
    tmp_path: Path,
) -> None:
    payload = _build_v1_payload(tmp_path / "artifacts")
    plan = _plan_for(payload)
    write_process_v2_rebind_plan(plan, artifact_root=payload.artifact_root, repo_root=ROOT)
    task_root = _run_root(payload, plan) / TASK_DIRNAME
    for task in plan["tasks"][:-1]:
        assert (
            execute_process_v2_rebind_task(
                plan,
                task["task_identity_sha256"],
                artifact_root=payload.artifact_root,
                repo_root=ROOT,
            )["reused"]
            is False
        )
    first = plan["tasks"][0]
    output = task_root / first["task_identity_sha256"]
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    reused = execute_process_v2_rebind_task(
        plan,
        first["task_identity_sha256"],
        artifact_root=payload.artifact_root,
        repo_root=ROOT,
    )
    assert reused["reused"] is True
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before

    interrupted = plan["tasks"][-1]["task_identity_sha256"]
    staging = task_root / f".{interrupted}.interrupted.staging"
    staging.mkdir()
    (staging / PROOF_FILENAME).write_bytes(b"partial")
    assert (
        len(
            completed_process_v2_rebind_task_ids(
                plan,
                artifact_root=payload.artifact_root,
                repo_root=ROOT,
            )
        )
        == 3
    )
    with pytest.raises(ProcessV2RebindIncomplete, match="missing 1"):
        reduce_process_v2_rebind(plan, artifact_root=payload.artifact_root, repo_root=ROOT)
    assert (
        execute_process_v2_rebind_task(
            plan,
            interrupted,
            artifact_root=payload.artifact_root,
            repo_root=ROOT,
        )["reused"]
        is False
    )
    assert reduce_process_v2_rebind(
        plan,
        artifact_root=payload.artifact_root,
        repo_root=ROOT,
    )["task_count"] == 4


# ---- Range completeness and namespace hygiene ---------------------------------


def test_range_contract_rejects_gaps_duplicates_and_overlaps() -> None:
    assert validate_process_v2_rebind_task_ranges(((2, 5), (0, 2)), entries=5) == ((0, 2), (2, 5))
    with pytest.raises(ProcessV2RebindError, match="gap"):
        validate_process_v2_rebind_task_ranges(((0, 2), (3, 5)), entries=5)
    with pytest.raises(ProcessV2RebindError, match="overlap"):
        validate_process_v2_rebind_task_ranges(((0, 2), (0, 2)), entries=5)
    with pytest.raises(ProcessV2RebindError, match="overlap"):
        validate_process_v2_rebind_task_ranges(((0, 3), (2, 5)), entries=5)
    with pytest.raises(ProcessV2RebindError, match="does not cover"):
        validate_process_v2_rebind_task_ranges(((0, 4),), entries=5)


def test_reducer_rejects_a_missing_range_and_an_unexpected_task_object(
    tmp_path: Path,
) -> None:
    payload = _build_v1_payload(tmp_path / "artifacts")
    plan = _plan_for(payload)
    _execute_all(payload, plan)
    task_root = _run_root(payload, plan) / TASK_DIRNAME

    unexpected = task_root / ("f" * 64)
    unexpected.mkdir()
    with pytest.raises(ProcessV2RebindError, match="unexpected objects"):
        reduce_process_v2_rebind(plan, artifact_root=payload.artifact_root, repo_root=ROOT)
    unexpected.rmdir()

    removed = task_root / plan["tasks"][1]["task_identity_sha256"]
    for path in removed.iterdir():
        path.unlink()
    removed.rmdir()
    with pytest.raises(ProcessV2RebindIncomplete, match="missing 1"):
        reduce_process_v2_rebind(plan, artifact_root=payload.artifact_root, repo_root=ROOT)


def test_v1_run_root_documents_are_bound_by_bytes(tmp_path: Path) -> None:
    """The V1 plan and completion assert the superseded live identity, so they are
    bound by physical hash only; changing one byte still moves the rebind run
    identity, and a stray directory in the payload root is refused."""

    payload = _build_v1_payload(tmp_path / "artifacts")
    (payload.payload_root / V1_RUN_PLAN_FILENAME).write_bytes(b"opaque-v1-plan-bytes\n")
    (payload.payload_root / V1_RUN_COMPLETION_FILENAME).write_bytes(b"opaque-v1-completion\n")
    first = _plan_for(payload)
    assert set(first["v1_payload_binding"]["payload_root_files"]) == {
        V1_RUN_PLAN_FILENAME,
        V1_RUN_COMPLETION_FILENAME,
    }
    (payload.payload_root / V1_RUN_COMPLETION_FILENAME).write_bytes(b"other-v1-completion\n")
    assert _plan_for(payload)["run_identity_sha256"] != first["run_identity_sha256"]

    (payload.payload_root / "stray").mkdir()
    with pytest.raises(ProcessV2RebindError, match="unexpected object"):
        _plan_for(payload)


def test_payload_binding_rejects_a_repeated_v1_task(tmp_path: Path) -> None:
    payload = _build_v1_payload(tmp_path / "artifacts")
    process_identity, builder_identity = _pinned_identities()
    binding = bind_v1_semantic_payload(
        payload_root_artifact_path=PAYLOAD_ARTIFACT_PATH,
        artifact_root=payload.artifact_root,
        pinned_process_identity=process_identity,
        pinned_builder_identity=builder_identity,
    )
    duplicated = dict(binding)
    duplicated["v1_tasks"] = [binding["v1_tasks"][0], binding["v1_tasks"][0]]
    with pytest.raises(ProcessV2RebindError, match="repeats a task identity"):
        validate_v1_semantic_payload_binding(
            duplicated,
            pinned_process_identity=process_identity,
            pinned_builder_identity=builder_identity,
        )


# ---- Pinned identity ----------------------------------------------------------


def test_a_pinned_identity_that_does_not_match_the_payload_is_rejected(
    tmp_path: Path,
) -> None:
    payload = _build_v1_payload(tmp_path / "artifacts")
    process_identity, builder_identity = _pinned_identities()
    # Mutate a field the identity schema does NOT definitionally bind, so the
    # object stays structurally valid and is rejected for the reason under test
    # (it is not the payload's identity) rather than for being malformed.
    # Rewriting a bound field such as ``contract_relative_path`` is a different
    # failure, covered by ``test_a_relabelled_v1_identity_cannot_masquerade``.
    other = dict(process_identity)
    other["contract_sha256"] = "0" * 64
    other["process_identity_sha256"] = _self_hash(other, "process_identity_sha256")
    with pytest.raises(ProcessV2RebindError, match="pinned process identity"):
        bind_v1_semantic_payload(
            payload_root_artifact_path=PAYLOAD_ARTIFACT_PATH,
            artifact_root=payload.artifact_root,
            pinned_process_identity=other,
            pinned_builder_identity=builder_identity,
        )
    wrong_builder = dict(builder_identity)
    wrong_builder["schema_version"] = 999
    wrong_builder["identity_sha256"] = _self_hash(wrong_builder, "identity_sha256")
    with pytest.raises(ProcessV2RebindError, match="pinned builder identity"):
        bind_v1_semantic_payload(
            payload_root_artifact_path=PAYLOAD_ARTIFACT_PATH,
            artifact_root=payload.artifact_root,
            pinned_process_identity=process_identity,
            pinned_builder_identity=wrong_builder,
        )


def test_a_relabelled_v1_identity_cannot_masquerade(tmp_path: Path) -> None:
    """A V1 identity relabelled to V2 and re-self-hashed is still refused.

    Re-hashing makes the object internally consistent, so self-consistency
    alone cannot catch it.  The identity schema binds its own schema version,
    process semantics, and contract path, which is what closes the gap.
    """

    payload = _build_v1_payload(tmp_path / "artifacts")
    process_identity, builder_identity = _pinned_identities()
    relabelled = dict(process_identity)
    relabelled["schema"] = "compose.editing.semantic_process_v2_identity"
    relabelled["process_semantics"] = "semantic_editing_v2_v2"
    relabelled["contract_relative_path"] = PROCESS_V2_CONTRACT_RELATIVE_PATH
    relabelled["process_identity_sha256"] = _self_hash(
        relabelled,
        "process_identity_sha256",
    )
    with pytest.raises(ProcessV2RebindError):
        bind_v1_semantic_payload(
            payload_root_artifact_path=PAYLOAD_ARTIFACT_PATH,
            artifact_root=payload.artifact_root,
            pinned_process_identity=relabelled,
            pinned_builder_identity=builder_identity,
        )


def test_a_self_inconsistent_pinned_identity_is_rejected(tmp_path: Path) -> None:
    payload = _build_v1_payload(tmp_path / "artifacts")
    process_identity, builder_identity = _pinned_identities()
    tampered_process = dict(process_identity)
    tampered_process["contract_relative_path"] = PROCESS_V2_CONTRACT_RELATIVE_PATH
    with pytest.raises(ProcessV2RebindError, match="not internally self-consistent"):
        bind_v1_semantic_payload(
            payload_root_artifact_path=PAYLOAD_ARTIFACT_PATH,
            artifact_root=payload.artifact_root,
            pinned_process_identity=tampered_process,
            pinned_builder_identity=builder_identity,
        )
    tampered_builder = dict(builder_identity)
    tampered_builder["schema_version"] = 999
    with pytest.raises(ProcessV2RebindError, match="self-hash does not match"):
        bind_v1_semantic_payload(
            payload_root_artifact_path=PAYLOAD_ARTIFACT_PATH,
            artifact_root=payload.artifact_root,
            pinned_process_identity=process_identity,
            pinned_builder_identity=tampered_builder,
        )


def test_pinned_identity_reads_a_payload_whose_live_identity_has_been_superseded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pinned path is the only readable path once the live identity moves."""

    payload = _build_v1_payload(tmp_path / "artifacts")
    process_identity, builder_identity = _pinned_identities()
    semantic_dir = _semantic_dir(payload, 0)
    receipt = json.loads(
        (payload.payload_root / TASK_DIRNAME / payload.v1_task_identities[0] / V1_RECEIPT_FILENAME)
        .read_text()
    )
    import compose_v4.data.semantic_packed_trace_store as semantic_store

    def superseded(_expected: str) -> dict:
        raise EditingV2ProcessIdentityError("live identity has been superseded")

    monkeypatch.setattr(semantic_store, "require_editing_v2_process_identity", superseded)
    with pytest.raises(SemanticPackedStoreError, match="process identity is stale"):
        load_semantic_packed_manifest(
            semantic_dir,
            expected_shard_sha256=receipt["semantic_shard_sha256"],
            expected_manifest_sha256=receipt["semantic_manifest_sha256"],
        )
    manifest = load_semantic_packed_manifest(
        semantic_dir,
        expected_shard_sha256=receipt["semantic_shard_sha256"],
        expected_manifest_sha256=receipt["semantic_manifest_sha256"],
        expected_process_identity=process_identity,
        expected_builder_identity=builder_identity,
    )
    assert manifest["process_identity_sha256"] == process_identity["process_identity_sha256"]


# ---- Refusal ------------------------------------------------------------------


def test_a_one_slot_successor_corruption_refuses_to_publish(tmp_path: Path) -> None:
    payload = _build_v1_payload(tmp_path / "artifacts")
    fluorine, chlorine = 5, 8

    def swap_one_slot(record: dict) -> None:
        successor = record["states"][1]["atom_types"]
        if successor[5] == fluorine:
            successor[5] = chlorine

    _reseal_v1_task(
        payload.payload_root / TASK_DIRNAME / payload.v1_task_identities[0],
        swap_one_slot,
    )
    _refuses_with(payload, ProcessV2RebindReasonCode.SUCCESSOR_ARRAY_DISAGREES)


def test_a_corrupted_canonical_key_refuses_to_publish(tmp_path: Path) -> None:
    payload = _build_v1_payload(tmp_path / "artifacts")

    def retarget_key(record: dict) -> None:
        if record["path_length"] != 1:
            return
        record["canonical_state_keys"][1] = "CCCCCC|fixture-forged-key"
        record["steps"][0]["successor_key"] = "CCCCCC|fixture-forged-key"

    _reseal_v1_task(
        payload.payload_root / TASK_DIRNAME / payload.v1_task_identities[0],
        retarget_key,
    )
    _refuses_with(payload, ProcessV2RebindReasonCode.CANONICAL_KEY_DISAGREES)


def test_an_unexecutable_persisted_action_refuses_to_publish(tmp_path: Path) -> None:
    payload = _build_v1_payload(tmp_path / "artifacts")

    def target_a_null_slot(record: dict) -> None:
        action = record["steps"][0]["action"]
        if action["executor_rule"] == "cycle_close":
            action["payload"]["b"] = SLOTS - 1

    _reseal_v1_task(
        payload.payload_root / TASK_DIRNAME / payload.v1_task_identities[0],
        target_a_null_slot,
    )
    _refuses_with(payload, ProcessV2RebindReasonCode.EXECUTOR_REPLAY_FAILED)


def test_a_non_canonical_action_reencoding_refuses_to_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The V4 codec is round-trip canonical by construction, so this guard can
    only be exercised by injecting a non-canonical re-encoding.  It exists to
    catch a future lossy codec change, which would otherwise let a persisted
    action drift away from the action the model is scored against."""

    payload = _build_v1_payload(tmp_path / "artifacts")
    original = action_codec_v4.encode_action

    def perturbed(executor_rule: str, action: object) -> dict:
        record = original(executor_rule, action)
        payload_fields = dict(record["payload"])
        field = sorted(payload_fields)[0]
        if isinstance(payload_fields[field], int):
            payload_fields[field] += 1
        return {**record, "payload": payload_fields}

    monkeypatch.setattr(action_codec_v4, "encode_action", perturbed)
    _refuses_with(payload, ProcessV2RebindReasonCode.ACTION_CODEC_ROUNDTRIP_DISAGREES)


def test_a_production_mask_that_leaves_the_independent_oracle_refuses_to_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The oracle comparison is the only evidence that the two derivations agree,
    so it must be able to fail.  Perturbing the production mask the rebind reads
    proves the comparison is load-bearing rather than trivially satisfied."""

    payload = _build_v1_payload(tmp_path / "artifacts")
    import compose_v4.data.editing_process_v2_rebind as rebind_module

    original = rebind_module.process_v2_connected_nonleaf_atom_delete_mask

    def widened(state):
        mask = original(state).copy()
        mask[0] = True
        return mask

    monkeypatch.setattr(
        rebind_module,
        "process_v2_connected_nonleaf_atom_delete_mask",
        widened,
    )
    _refuses_with(payload, ProcessV2RebindReasonCode.PROCESS_V2_MASK_DISAGREES)


# ---- Provenance ---------------------------------------------------------------


def test_source_lane_split_and_lineage_are_carried_through_unchanged(
    tmp_path: Path,
) -> None:
    payload = _build_v1_payload(tmp_path / "artifacts")
    plan = _plan_for(payload)
    _execute_all(payload, plan)
    run_root = _run_root(payload, plan)
    v1_records = {
        record["trace_id"]: record
        for index in range(len(payload.v1_task_identities))
        for record in _read_v1_records(_semantic_dir(payload, index))
    }
    v1_receipts = {
        identity: json.loads(
            (payload.payload_root / TASK_DIRNAME / identity / V1_RECEIPT_FILENAME).read_text()
        )
        for identity in payload.v1_task_identities
    }
    proved = 0
    connected_nonleaf_teachers = 0
    evaluated_states = 0
    candidate_slots = 0
    for task in plan["tasks"]:
        output = run_root / TASK_DIRNAME / task["task_identity_sha256"]
        receipt = json.loads((output / RECEIPT_FILENAME).read_text())
        manifest = json.loads((output / MANIFEST_FILENAME).read_text())
        v1_receipt = v1_receipts[receipt["v1_task_identity_sha256"]]
        assert receipt["v1_source_binding"] == v1_receipt["source_binding"]
        assert receipt["data_lane"] == v1_receipt["data_lane"]
        assert receipt["split"] == v1_receipt["split"]
        connected_nonleaf_teachers += manifest["teacher_census"][
            "connected_nonleaf_atom_delete_teachers"
        ]
        evaluated_states += manifest["process_v2_atom_delete_census"]["states_evaluated"]
        candidate_slots += manifest["process_v2_atom_delete_census"]["candidate_slots"]
        with gzip.open(output / PROOF_FILENAME, "rb") as handle:
            for line in handle:
                proof = json.loads(line)
                original = v1_records[proof["trace_id"]]
                assert proof["source_address"] == original["source_address"]
                assert proof["lineage"] == original["lineage"]
                assert proof["data_lane"] == original["data_lane"]
                assert proof["split"] == original["split"]
                assert proof["canonical_state_keys"] == original["canonical_state_keys"]
                assert proof["family_histogram"] == original["family_histogram"]
                assert proof["v1_record_sha256"] == original["record_sha256"]
                proved += 1
    assert proved == 4
    # The payload deliberately contains one ring-atom deletion, which the V1
    # dense mask could not represent and Process V2 admits.
    assert connected_nonleaf_teachers == 1
    # Ten progress states over four traces, and the cyclic ones expose a
    # nonempty Process-V2 fiber, so the mask comparison is not vacuous.
    assert evaluated_states == 10
    assert candidate_slots > 0
