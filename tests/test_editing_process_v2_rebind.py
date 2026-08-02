"""Integrity, support-exclusion, census and provenance tests for the rebind.

Every fixture payload is produced by the production writers over real small
molecules: a legacy packed shard is written by ``write_packed_shard`` and then
migrated by ``materialize_frozen_packed_shard``, which itself publishes through
``write_semantic_packed_artifact``.  No V1 artifact is hand-written.

The central invariant under test is that the two outcomes are never conflated.
An integrity mismatch aborts the task and publishes nothing.  An unsupported
teacher rejects exactly one trace, records its measured reason code and exact
identity, and lets the run complete.

Pending integration of the unified admission authority
------------------------------------------------------

The complete effective mask is
``compose_v4.rewrite.process_v2_atom_delete.process_v2_atom_delete_mask``.  When
that symbol is absent from the worktree, ``_effective_mask_authority`` installs
this module's own independent oracle in its place and ``AUTHORITY_SUBSTITUTED``
is True.  Under substitution the mask *agreement* assertion is vacuous, and the
one test that exists to prove agreement is skipped rather than reported as
passing; every other assertion below stays live, including the assertion that a
*disagreeing* authority refuses, which installs a deliberately wrong mask and
therefore never becomes vacuous.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import SCAR_IDX, smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.data.editing_corpus_contract import (
    REQUIRED_DATA_LANES,
    REQUIRED_PARTITION_ROLES,
)
from compose_v4.data.editing_process_v2_rebind import (
    COMPLETION_FILENAME,
    CONNECTED_NONLEAF_CANDIDATE_SOURCE,
    INHERITED_CANDIDATE_SOURCE,
    MANIFEST_FILENAME,
    PLAN_FILENAME,
    PROOF_FILENAME,
    RECEIPT_FILENAME,
    REFUSAL_SCHEMA,
    TASK_DIRNAME,
    ProcessV2RebindError,
    ProcessV2RebindExclusionCode,
    ProcessV2RebindIncomplete,
    ProcessV2RebindIntegrityCode,
    ProcessV2RebindMismatch,
    authority_process_v2_atom_delete_slots,
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
from compose_v4.data.editing_process_v2_rebind import (
    _teacher_support_exclusion as teacher_support_exclusion,
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
from compose_v4.rewrite import process_v2_atom_delete as process_v2_atom_delete_module
from compose_v4.rewrite.editing_v2_process_identity import (
    EditingV2ProcessIdentityError,
    editing_v2_process_identity,
)
from compose_v4.rewrite.kernel import de_novo_rewrite_system
from compose_v4.rewrite.operators import AtomDelete, BondInsert
from compose_v4.rewrite.progress import TraceProgressCTMC
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace
from compose_v4.rewrite.trace_shard import encode_trace_record

ROOT = Path(__file__).resolve().parents[1]
SLOTS = 16
PAYLOAD_ARTIFACT_PATH = "/artifacts/v1_payload"
PROCESS_V2_CONTRACT_RELATIVE_PATH = "configs/editing_v2_semantic_process_v2.json"
AUTHORITY_NAME = "process_v2_atom_delete_mask"
AUTHORITY_SUBSTITUTED = not hasattr(process_v2_atom_delete_module, AUTHORITY_NAME)

# One real molecule and one real legacy edit path per fixture trace.
# ``open_cyclohexane`` deletes a ring atom, which only the Process-V2 fiber can
# represent.  ``trim_methylcyclohexane`` deletes an inherited leaf, which the
# complete mask must still admit.
#
# ``open_benzene`` is the measured support exclusion, and which gate produces it
# is not arbitrary.  The V1 semantic runtime already enforces connectivity and
# the authoritative charge policy as hard conditions, so no V1 *payload* can
# carry a disconnecting or charge-violating teacher: those gaps exist in the V1
# dense model mask, not in the migrated data.  What the V1 payload can carry,
# and does, is an aromatic connected-nonleaf deletion, which Process V2 excludes
# by frozen decision because its successor is representation-sensitive.
# Deleting one Kekule slot of benzene yields ``C=CC=CC``: valid, connected,
# charge-preserving, recorded by the V1 migration, and outside the V2 fiber.
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
    "open_benzene": ("c1ccccc1", (("atom_delete", AtomDelete(0)),)),
}
_V1_TASKS = (
    (
        REQUIRED_DATA_LANES[0],
        REQUIRED_PARTITION_ROLES[0],
        ("cyclize_hexane", "cyclize_fluoropentane"),
    ),
    (
        REQUIRED_DATA_LANES[1],
        REQUIRED_PARTITION_ROLES[1],
        ("trim_methylcyclohexane", "open_cyclohexane"),
    ),
)
_V1_TASKS_WITH_EXCLUSION = (
    _V1_TASKS[0],
    (
        REQUIRED_DATA_LANES[1],
        REQUIRED_PARTITION_ROLES[1],
        ("trim_methylcyclohexane", "open_benzene", "open_cyclohexane"),
    ),
)


# ---- The complete effective mask authority ------------------------------------


def _oracle_effective_mask(state) -> np.ndarray:
    mask = np.zeros(state.n_atoms, dtype=np.bool_)
    for slot in independent_process_v2_atom_delete_slots(state):
        mask[slot] = True
    return mask


@pytest.fixture(autouse=True)
def _effective_mask_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for the unified admission authority only while it is absent.

    Once ``process_v2_atom_delete_mask`` exists this fixture installs nothing
    and every test below drives the real production authority.
    """

    if AUTHORITY_SUBSTITUTED:
        monkeypatch.setattr(
            process_v2_atom_delete_module,
            AUTHORITY_NAME,
            _oracle_effective_mask,
            raising=False,
        )


# ---- Real V1 payload fixture --------------------------------------------------


@dataclass(frozen=True)
class _V1Payload:
    artifact_root: Path
    payload_root: Path
    v1_task_identities: tuple[str, ...]


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _self_hash(value: Mapping[str, object], field: str) -> str:
    return _canonical_sha256({key: item for key, item in value.items() if key != field})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _legacy_trace(name: str) -> RewriteTrace:
    smiles, steps = _FIXTURE_TRACES[name]
    runtime = de_novo_rewrite_system()
    state = pad_molecular_graph(smiles_to_molecular_graph(smiles), SLOTS)
    states = [state]
    rewrite_steps = []
    for rule, action in steps:
        state = runtime.apply(state, rule, action)
        states.append(state)
        rewrite_steps.append(RewriteStep(rule, action))
    return RewriteTrace(states[0], states[-1], tuple(rewrite_steps), {"fixture": name})


def _v1_task_identity(index: int) -> str:
    return hashlib.sha256(f"process-v2-rebind-fixture-task-{index}".encode()).hexdigest()


def _build_v1_payload(artifact_root: Path, tasks=_V1_TASKS) -> _V1Payload:
    payload_root = artifact_root / "v1_payload"
    identities: list[str] = []
    for index, (lane, split, names) in enumerate(tasks):
        shard = artifact_root / "v1_source" / lane / split / "shard_0000.jsonl.gz"
        shard.parent.mkdir(parents=True, exist_ok=True)
        entries = []
        for offset, name in enumerate(names):
            trace = _legacy_trace(name)
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


def _reseal_v1_shard(task_dir: Path, lines: list[bytes]) -> None:
    """Rewrite one V1 shard and re-seal every content address around it.

    The physical hashes, the manifest, the completion receipt and the migration
    receipt are all recomputed, so tampered bytes survive every transport gate
    and have to be caught by the rebind proof itself.
    """

    semantic_dir = task_dir / SEMANTIC_ARTIFACT_DIRNAME
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


def _reseal_v1_task(task_dir: Path, mutate) -> None:
    """Apply ``mutate`` to every V1 record, then re-seal every content address."""

    lines: list[bytes] = []
    for record in _read_v1_records(task_dir / SEMANTIC_ARTIFACT_DIRNAME):
        mutate(record)
        record["record_sha256"] = _self_hash(record, "record_sha256")
        lines.append(_canonical_line(record))
    _reseal_v1_shard(task_dir, lines)


def _refuses_with(payload: _V1Payload, code: ProcessV2RebindIntegrityCode) -> None:
    """Require the refusing range to publish nothing at all, then stay incomplete."""

    plan = _plan_for(payload)
    write_process_v2_rebind_plan(plan, artifact_root=payload.artifact_root, repo_root=ROOT)
    task_root = _run_root(payload, plan) / TASK_DIRNAME
    refused: str | None = None
    error: ProcessV2RebindMismatch | None = None
    for task in plan["tasks"]:
        try:
            execute_process_v2_rebind_task(
                plan,
                task["task_identity_sha256"],
                artifact_root=payload.artifact_root,
                repo_root=ROOT,
            )
        except ProcessV2RebindMismatch as mismatch:
            refused = task["task_identity_sha256"]
            error = mismatch
            break
    assert refused is not None and error is not None, "the corrupted payload must refuse"
    assert code in {finding.code for finding in error.findings}
    assert code.value in str(error)
    # The histogram is a measurement of this refusal, not a constant.
    assert error.mismatches_by_code[code.value] >= 1
    assert sum(error.mismatches_by_code.values()) == len(error.findings)
    report = error.as_report()
    assert report["schema"] == REFUSAL_SCHEMA
    assert report["published"] is False
    assert report["training_authorized"] is False
    assert report["mismatches_by_code"] == error.mismatches_by_code
    assert report["refusal_sha256"] == _self_hash(report, "refusal_sha256")
    assert not (task_root / refused).exists(), "a refused range must leave no output"
    published = {path.name for path in task_root.iterdir()} if task_root.exists() else set()
    assert not any(name.endswith(".staging") for name in published)
    assert published <= {task["task_identity_sha256"] for task in plan["tasks"]} - {refused}
    with pytest.raises(ProcessV2RebindIncomplete):
        reduce_process_v2_rebind(plan, artifact_root=payload.artifact_root, repo_root=ROOT)


# ---- The independent oracle derives the COMPLETE effective fiber --------------


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        # Connected-nonleaf: every ring atom of a saturated carbocycle.
        ("C1CCCCC1", (0, 1, 2, 3, 4, 5)),
        ("C1CCOCC1", (0, 1, 2, 3, 4, 5)),
        # Inherited leaves are candidates too: a complete mask, not an extension.
        ("CCCCCC", (0, 5)),
        ("CCO", (0, 2)),
        # A single atom deletes to the null state.
        ("C", (0,)),
        # Aromatic ring atoms stay excluded; the methyl leaf does not.
        ("c1ccccc1", ()),
        ("Cc1ccccc1", (0,)),
        # Articulation points stay excluded; ring atoms and the methyl do not.
        ("CC1CCCCC1", (0, 2, 3, 4, 5, 6)),
        # The authoritative charge policy applies to inherited leaves too: only
        # the carbonyl oxygen survives it on this zwitterion.
        ("C[N+](C)(C)CC(=O)[O-]", (6,)),
    ],
)
def test_the_independent_oracle_derives_the_complete_effective_fiber(
    smiles: str,
    expected: tuple[int, ...],
) -> None:
    state = pad_molecular_graph(smiles_to_molecular_graph(smiles), SLOTS)
    assert independent_process_v2_atom_delete_slots(state) == expected


def test_the_oracle_excludes_a_scar_incident_ring_atom(tmp_path: Path) -> None:
    """A SCAR-adjacent ring deletion is newly introduced by V2 and is excluded.

    A SCAR-adjacent *leaf* stays reachable, because it was already reachable
    under V1 and the recorded decision withdraws no inherited capability.
    """

    base = pad_molecular_graph(smiles_to_molecular_graph("C1CCCCC1C"), SLOTS)
    types = base.atom_types.copy()
    types[6] = SCAR_IDX
    scarred = replace(base, atom_types=types)
    admitted = independent_process_v2_atom_delete_slots(scarred)
    assert 5 not in admitted, "the ring atom bonded to the SCAR must be excluded"
    assert admitted == (0, 1, 2, 3, 4)
    assert SCAR_IDX not in {int(scarred.atom_types[slot]) for slot in admitted}


@pytest.mark.skipif(
    AUTHORITY_SUBSTITUTED,
    reason=(
        "pending integration: compose_v4.rewrite.process_v2_atom_delete."
        "process_v2_atom_delete_mask is absent, so the production authority is "
        "substituted by this module's oracle and an agreement assertion would be vacuous"
    ),
)
@pytest.mark.parametrize(
    "smiles",
    [
        "C1CCCCC1",
        "CCCCCC",
        "Cc1ccccc1",
        "CC1CCCCC1",
        "C[N+](C)(C)CC(=O)[O-]",
        "O=C1NC(O)C2CCCCC12",
    ],
)
def test_the_production_authority_and_the_independent_oracle_agree(smiles: str) -> None:
    state = pad_molecular_graph(smiles_to_molecular_graph(smiles), SLOTS)
    assert authority_process_v2_atom_delete_slots(state) == (
        independent_process_v2_atom_delete_slots(state)
    )


def test_an_absent_mask_authority_names_itself_and_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(process_v2_atom_delete_module, AUTHORITY_NAME, raising=False)
    state = pad_molecular_graph(smiles_to_molecular_graph("C1CCCCC1"), SLOTS)
    with pytest.raises(ProcessV2RebindError, match="process_v2_atom_delete_mask"):
        authority_process_v2_atom_delete_slots(state)


def test_a_mask_authority_of_the_wrong_shape_or_dtype_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = pad_molecular_graph(smiles_to_molecular_graph("C1CCCCC1"), SLOTS)
    monkeypatch.setattr(
        process_v2_atom_delete_module,
        AUTHORITY_NAME,
        lambda _state: np.zeros(state.n_atoms, dtype=np.int8),
        raising=False,
    )
    with pytest.raises(ProcessV2RebindError, match="boolean mask"):
        authority_process_v2_atom_delete_slots(state)
    monkeypatch.setattr(
        process_v2_atom_delete_module,
        AUTHORITY_NAME,
        lambda _state: np.zeros(3, dtype=np.bool_),
        raising=False,
    )
    with pytest.raises(ProcessV2RebindError, match="boolean mask"):
        authority_process_v2_atom_delete_slots(state)


# ---- Frozen teacher support ---------------------------------------------------


def test_the_frozen_support_predicate_reads_its_registries() -> None:
    """A disabled rule and an unknown family are support exclusions, not defects.

    Reachability boundary, reported rather than dressed up: the packed reader
    decodes every action before this predicate ever sees it, and the codec
    refuses an executor rule outside the frozen surface, so on a *readable* V1
    payload these two codes cannot fire and only
    ``atom_delete_outside_process_v2_mask`` can.  They are defence in depth for
    a future reader that decodes more permissively.
    """

    supported = {"executor_rule": "atom_delete", "model_family": "atom_delete"}
    assert teacher_support_exclusion(supported) is None
    disabled = teacher_support_exclusion(
        {"executor_rule": "ring_system_delete", "model_family": "ring_system_delete"}
    )
    assert disabled is not None
    assert disabled[0] is ProcessV2RebindExclusionCode.TEACHER_RULE_OUTSIDE_FROZEN_SUPPORT
    unknown_family = teacher_support_exclusion(
        {"executor_rule": "atom_delete", "model_family": "ring_system_grow"}
    )
    assert unknown_family is not None
    assert unknown_family[0] is ProcessV2RebindExclusionCode.TEACHER_FAMILY_OUTSIDE_FROZEN_SUPPORT


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
        assert result["counts"]["source_entries"] == 1
        assert result["counts"]["admitted_entries"] == 1
        assert result["counts"]["rejected_entries"] == 0

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
    assert completion["counts"] == {
        "source_entries": 4,
        "admitted_entries": 4,
        "rejected_entries": 0,
        "admitted_states": 10,
        "admitted_transitions": 6,
    }
    assert completion["rejected_traces_by_code"] == {}
    assert completion["unsupported_teacher_steps_by_code"] == {}
    # Measured, not constant: the payload contains one ring-atom deletion and
    # one inherited-leaf deletion, and the mask was evaluated at every state.
    assert completion["teacher_census"] == {
        "atom_delete_teachers": 2,
        f"atom_delete_teachers_{CONNECTED_NONLEAF_CANDIDATE_SOURCE}": 1,
        f"atom_delete_teachers_{INHERITED_CANDIDATE_SOURCE}": 1,
        "teacher_transitions": 6,
    }
    assert completion["process_v2_atom_delete_census"]["states_evaluated"] == 10
    assert completion["process_v2_atom_delete_census"]["candidate_slots"] > 0
    assert completion["completion_sha256"] == _self_hash(completion, "completion_sha256")
    assert (
        completion["process_v2_identity_sha256"] != completion["pinned_process_identity_sha256"]
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


# ---- Support exclusion, which must NOT abort the run --------------------------


def test_an_unsupported_teacher_rejects_exactly_that_trace_and_the_run_completes(
    tmp_path: Path,
) -> None:
    """The measured correction case: an aromatic connected-nonleaf deletion.

    Round 1 aborted the entire run on this, because it treated an expected
    support exclusion as an integrity mismatch.  The run must instead complete
    and account for every trace.
    """

    payload = _build_v1_payload(tmp_path / "artifacts", tasks=_V1_TASKS_WITH_EXCLUSION)
    plan = _plan_for(payload)
    assert plan["expected_entry_count"] == 5
    _execute_all(payload, plan)
    completion = reduce_process_v2_rebind(
        plan,
        artifact_root=payload.artifact_root,
        repo_root=ROOT,
    )
    code = ProcessV2RebindExclusionCode.ATOM_DELETE_OUTSIDE_PROCESS_V2_MASK.value
    # The census is non-empty and measured; an assertion that {} == {} is
    # exactly what this round removes.
    assert completion["rejected_traces_by_code"] == {code: 1}
    assert completion["unsupported_teacher_steps_by_code"] == {code: 1}
    assert completion["counts"]["source_entries"] == 5
    assert completion["counts"]["admitted_entries"] == 4
    assert completion["counts"]["rejected_entries"] == 1
    assert (
        completion["counts"]["admitted_entries"] + completion["counts"]["rejected_entries"]
        == completion["counts"]["source_entries"]
    )
    assert completion["rejected_trace_inventory_sha256"] != _canonical_sha256([])

    run_root = _run_root(payload, plan)
    rejected_rows = []
    admitted_trace_ids = set()
    for task in plan["tasks"]:
        output = run_root / TASK_DIRNAME / task["task_identity_sha256"]
        manifest = json.loads((output / MANIFEST_FILENAME).read_text())
        rejected_rows.extend(manifest["rejected_traces"])
        assert (
            manifest["admitted_entries"] + manifest["rejected_entries"]
            == manifest["source_entries"]
        )
        with gzip.open(output / PROOF_FILENAME, "rb") as handle:
            for line in handle:
                admitted_trace_ids.add(json.loads(line)["trace_id"])
    # Exactly one trace identity, addressed precisely, and never admitted.
    assert len(rejected_rows) == 1
    row = rejected_rows[0]
    assert row["exclusion_code"] == code
    assert row["step_index"] == 0
    assert row["unsupported_teacher_steps"] == 1
    assert row["trace_id"] not in admitted_trace_ids
    assert row["rejection_sha256"] == _self_hash(row, "rejection_sha256")
    v1_records = {
        record["trace_id"]: record for record in _read_v1_records(_semantic_dir(payload, 1))
    }
    original = v1_records[row["trace_id"]]
    assert row["v1_record_sha256"] == original["record_sha256"]
    assert original["family_histogram"] == {"atom_delete": 1}
    assert len(admitted_trace_ids) == 4


def test_a_rejected_trace_is_never_partially_admitted(tmp_path: Path) -> None:
    """A rejected trace contributes zero rows and zero counted transitions."""

    payload = _build_v1_payload(tmp_path / "artifacts", tasks=_V1_TASKS_WITH_EXCLUSION)
    plan = _plan_for(payload)
    _execute_all(payload, plan)
    run_root = _run_root(payload, plan)
    rejected_entry_indices = set()
    admitted_entry_indices = set()
    for task in plan["tasks"]:
        output = run_root / TASK_DIRNAME / task["task_identity_sha256"]
        manifest = json.loads((output / MANIFEST_FILENAME).read_text())
        for row in manifest["rejected_traces"]:
            rejected_entry_indices.add((task["v1_task_identity_sha256"], row["entry_index"]))
        with gzip.open(output / PROOF_FILENAME, "rb") as handle:
            for line in handle:
                proof = json.loads(line)
                admitted_entry_indices.add(
                    (task["v1_task_identity_sha256"], proof["entry_index"])
                )
        if manifest["rejected_entries"]:
            # The rejecting range publishes no proof row at all for that entry.
            assert manifest["admitted_entries"] == 0
            assert manifest["admitted_states"] == 0
            assert manifest["admitted_transitions"] == 0
            # Its states were still evaluated against the mask oracle, because
            # integrity is proven before support is decided.
            assert manifest["process_v2_atom_delete_census"]["states_evaluated"] == 2
    assert rejected_entry_indices and not (rejected_entry_indices & admitted_entry_indices)


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
    task_dir = payload.payload_root / TASK_DIRNAME / payload.v1_task_identities[0]
    receipt = json.loads((task_dir / V1_RECEIPT_FILENAME).read_text())
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


# ---- Integrity refusal: abort, publish nothing --------------------------------


def test_an_unreadable_v1_row_refuses_to_publish(tmp_path: Path) -> None:
    """A row that survives every transport gate but cannot be decoded is named.

    The reason code exists to address exactly this row; before the correction
    an unreadable row died as an untyped read failure and the code was
    unreachable.
    """

    payload = _build_v1_payload(tmp_path / "artifacts")
    task_dir = payload.payload_root / TASK_DIRNAME / payload.v1_task_identities[0]
    undecodable = _canonical_line({"not": "a semantic trace record"})
    lines = [
        undecodable if index == 0 else _canonical_line(record)
        for index, record in enumerate(_read_v1_records(task_dir / SEMANTIC_ARTIFACT_DIRNAME))
    ]
    _reseal_v1_shard(task_dir, lines)
    _refuses_with(payload, ProcessV2RebindIntegrityCode.V1_RECORD_UNREADABLE)


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
    _refuses_with(payload, ProcessV2RebindIntegrityCode.SUCCESSOR_ARRAY_DISAGREES)


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
    _refuses_with(payload, ProcessV2RebindIntegrityCode.CANONICAL_KEY_DISAGREES)


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
    _refuses_with(payload, ProcessV2RebindIntegrityCode.EXECUTOR_REPLAY_FAILED)


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
    _refuses_with(payload, ProcessV2RebindIntegrityCode.ACTION_CODEC_ROUNDTRIP_DISAGREES)


def test_a_production_mask_that_leaves_the_independent_oracle_refuses_to_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The oracle comparison is the only evidence that the two derivations agree,
    so it must be able to fail.  Perturbing the production authority proves the
    comparison is load-bearing rather than trivially satisfied, and it stays
    meaningful whether or not the authority is substituted."""

    payload = _build_v1_payload(tmp_path / "artifacts")
    original = getattr(process_v2_atom_delete_module, AUTHORITY_NAME)

    def widened(state):
        mask = np.asarray(original(state)).copy()
        mask[0] = True
        return mask

    monkeypatch.setattr(process_v2_atom_delete_module, AUTHORITY_NAME, widened, raising=False)
    _refuses_with(payload, ProcessV2RebindIntegrityCode.PROCESS_V2_MASK_DISAGREES)


def test_an_integrity_mismatch_is_never_reported_as_a_support_exclusion(
    tmp_path: Path,
) -> None:
    """Integrity is decided first: a corrupt payload is never merely rejected."""

    payload = _build_v1_payload(tmp_path / "artifacts", tasks=_V1_TASKS_WITH_EXCLUSION)

    def retarget_key(record: dict) -> None:
        record["canonical_state_keys"][-1] = "CCCCCC|fixture-forged-key"
        record["steps"][-1]["successor_key"] = "CCCCCC|fixture-forged-key"

    _reseal_v1_task(
        payload.payload_root / TASK_DIRNAME / payload.v1_task_identities[1],
        retarget_key,
    )
    plan = _plan_for(payload)
    write_process_v2_rebind_plan(plan, artifact_root=payload.artifact_root, repo_root=ROOT)
    refusals = 0
    for task in plan["tasks"]:
        try:
            execute_process_v2_rebind_task(
                plan,
                task["task_identity_sha256"],
                artifact_root=payload.artifact_root,
                repo_root=ROOT,
            )
        except ProcessV2RebindMismatch as error:
            refusals += 1
            assert set(error.mismatches_by_code) <= {
                code.value for code in ProcessV2RebindIntegrityCode
            }
            assert not set(error.mismatches_by_code) & {
                code.value for code in ProcessV2RebindExclusionCode
            }
    assert refusals == 3, "every range of the corrupted V1 task must refuse"
    with pytest.raises(ProcessV2RebindIncomplete):
        reduce_process_v2_rebind(plan, artifact_root=payload.artifact_root, repo_root=ROOT)


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
    inherited_teachers = 0
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
            f"atom_delete_teachers_{CONNECTED_NONLEAF_CANDIDATE_SOURCE}"
        ]
        inherited_teachers += manifest["teacher_census"][
            f"atom_delete_teachers_{INHERITED_CANDIDATE_SOURCE}"
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
    # One ring-atom deletion the V1 dense mask could not represent, and one
    # inherited leaf deletion the complete mask must still admit.
    assert connected_nonleaf_teachers == 1
    assert inherited_teachers == 1
    # Ten progress states over four traces, and the cyclic ones expose a
    # nonempty Process-V2 fiber, so the mask comparison is not vacuous.
    assert evaluated_states == 10
    assert candidate_slots > 0
