"""Fail-closed tests for the Process-V2 admitted editing source adapter.

The fixture payload is produced by the production writers over real molecules
and then proven by the production rebind, so the overlay under test is a real
published artifact rather than a hand-written one.  The adapter must join the
immutable V1 chemistry to that overlay without relabelling either, and must
refuse a missing, incomplete, stale or disagreeing overlay rather than yield a
smaller corpus.

Pending integration of the unified admission authority, this module installs the
rebind's own independent oracle as the effective-mask authority when
``process_v2_atom_delete_mask`` is absent; see the rebind test module for the
claim boundary that creates.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.data import editing_process_v2_admitted_source as admitted_source_module
from compose_v4.data.editing_corpus_contract import (
    REQUIRED_DATA_LANES,
    REQUIRED_PARTITION_ROLES,
)
from compose_v4.data.editing_process_v2_admitted_source import (
    ADMITTED_SOURCE_SCHEMA,
    ADMITTED_SOURCE_SCHEMA_VERSION,
    ADMITTED_SOURCE_STATUS,
    PHYSICAL_EXECUTION_IDENTITY_SCHEMA,
    PHYSICAL_EXECUTION_IDENTITY_SCHEMA_VERSION,
    ProcessV2AdmittedSource,
    ProcessV2AdmittedSourceError,
    ProcessV2AdmittedSourceIdentityError,
    ProcessV2AdmittedSourceIncomplete,
    resolve_process_v2_admitted_source,
    validate_process_v2_admitted_source_identity,
)
from compose_v4.data.editing_v2_process_v2_schema import (
    AUTHORITY_FIELDS,
    REJECTION_LEDGER_SCHEMA,
    REJECTION_LEDGER_SCHEMA_VERSION,
)
from compose_v4.data.editing_process_v2_rebind import (
    COMPLETION_FILENAME,
    TASK_DIRNAME,
    bind_v1_semantic_payload,
    build_process_v2_rebind_source_revision,
    execute_process_v2_rebind_task,
    independent_process_v2_atom_delete_slots,
    plan_process_v2_rebind,
    reduce_process_v2_rebind,
    write_process_v2_rebind_plan,
)
from compose_v4.data.packed_trace_store import (
    build_packed_entry,
    manifest_path_for,
    write_packed_shard,
)
from compose_v4.data.semantic_packed_trace_store import semantic_packed_builder_identity
from compose_v4.data.semantic_trace_migration_materializer import (
    SemanticTraceMigrationTask,
    materialize_frozen_packed_shard,
)
from compose_v4.rewrite import process_v2_atom_delete as process_v2_atom_delete_module
from compose_v4.rewrite.editing_v2_process_identity import (
    PROCESS_SEMANTICS,
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
AUTHORITY_NAME = "process_v2_atom_delete_mask"
AUTHORITY_SUBSTITUTED = not hasattr(process_v2_atom_delete_module, AUTHORITY_NAME)

# ``open_benzene`` is the aromatic connected-nonleaf deletion the frozen
# Process-V2 fiber excludes; every other trace is admitted.
_FIXTURE_TRACES = {
    "cyclize_hexane": ("CCCCCC", (("bond_insert", BondInsert(0, 5, 1)),)),
    "open_cyclohexane": (
        "CCCCCC",
        (("bond_insert", BondInsert(0, 5, 1)), ("atom_delete", AtomDelete(0))),
    ),
    "open_benzene": ("c1ccccc1", (("atom_delete", AtomDelete(0)),)),
}
_ADMITTED_ONLY_TASKS = (
    (REQUIRED_DATA_LANES[0], REQUIRED_PARTITION_ROLES[0], ("cyclize_hexane",)),
    (REQUIRED_DATA_LANES[1], REQUIRED_PARTITION_ROLES[1], ("open_cyclohexane",)),
)
_TASKS_WITH_EXCLUSION = (
    (REQUIRED_DATA_LANES[0], REQUIRED_PARTITION_ROLES[0], ("cyclize_hexane",)),
    (
        REQUIRED_DATA_LANES[1],
        REQUIRED_PARTITION_ROLES[1],
        ("open_cyclohexane", "open_benzene"),
    ),
)


def _oracle_effective_mask(state) -> np.ndarray:
    mask = np.zeros(state.n_atoms, dtype=np.bool_)
    for slot in independent_process_v2_atom_delete_slots(state):
        mask[slot] = True
    return mask


@pytest.fixture(autouse=True)
def _effective_mask_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    if AUTHORITY_SUBSTITUTED:
        monkeypatch.setattr(
            process_v2_atom_delete_module,
            AUTHORITY_NAME,
            _oracle_effective_mask,
            raising=False,
        )


# ---- Real published overlay fixture -------------------------------------------


@dataclass(frozen=True)
class _Resolved:
    artifact_root: Path
    payload_root: Path
    plan: dict
    run_root: Path


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _self_hash(value: Mapping[str, object], field: str) -> str:
    return _canonical_sha256({key: item for key, item in value.items() if key != field})


def _reseal(payload: Mapping[str, object]) -> dict:
    """Sort and re-seal a mutated descriptor so it fails on the mutation.

    Without the reseal every mutation would be caught by the self-hash, which
    tests one guard seven ways and none of the others.
    """

    body = {
        key: item for key, item in sorted(payload.items()) if key != "admitted_source_sha256"
    }
    sealed = {**body, "admitted_source_sha256": _canonical_sha256(body)}
    return dict(sorted(sealed.items()))


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


def _build_and_prove(artifact_root: Path, tasks) -> _Resolved:
    payload_root = artifact_root / "v1_payload"
    for index, (lane, split, names) in enumerate(tasks):
        shard = artifact_root / "v1_source" / lane / split / "shard_0000.jsonl.gz"
        shard.parent.mkdir(parents=True, exist_ok=True)
        entries = []
        for offset, name in enumerate(names):
            trace = _legacy_trace(name)
            envelope = encode_trace_record(
                trace,
                n_slots=SLOTS,
                seed=2000 + 10 * index + offset,
                trace_id=f"legacy-{index}-{offset}",
                partition=split,
                layer=lane,
                extra=dict(trace.metadata),
            )
            entries.append(build_packed_entry(envelope, TraceProgressCTMC(trace)))
        write_packed_shard(shard, entries, provenance={"fixture": index}, deterministic_gzip=True)
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
                output_dir=payload_root
                / TASK_DIRNAME
                / hashlib.sha256(f"admitted-source-fixture-{index}".encode()).hexdigest(),
            )
        )
    process_identity = json.loads(json.dumps(editing_v2_process_identity()))
    builder_identity = json.loads(json.dumps(semantic_packed_builder_identity()))
    binding = bind_v1_semantic_payload(
        payload_root_artifact_path=PAYLOAD_ARTIFACT_PATH,
        artifact_root=artifact_root,
        pinned_process_identity=process_identity,
        pinned_builder_identity=builder_identity,
    )
    plan = plan_process_v2_rebind(
        binding,
        source_revision=build_process_v2_rebind_source_revision(
            commit="a" * 40,
            tree="b" * 40,
            repo_root=ROOT,
            worktree_clean=True,
        ),
        repo_root=ROOT,
        pinned_process_identity=process_identity,
        pinned_builder_identity=builder_identity,
        output_artifact_prefix="/artifacts/admitted_source_fixture",
        entries_per_task=1,
    )
    write_process_v2_rebind_plan(plan, artifact_root=artifact_root, repo_root=ROOT)
    for task in plan["tasks"]:
        execute_process_v2_rebind_task(
            plan,
            task["task_identity_sha256"],
            artifact_root=artifact_root,
            repo_root=ROOT,
        )
    reduce_process_v2_rebind(plan, artifact_root=artifact_root, repo_root=ROOT)
    run_root = artifact_root / PurePosixPath(str(plan["run_artifact_root"])).relative_to(
        "/artifacts"
    )
    return _Resolved(artifact_root, payload_root, plan, run_root)


def _resolve(fixture: _Resolved):
    return resolve_process_v2_admitted_source(
        fixture.plan,
        artifact_root=fixture.artifact_root,
        repo_root=ROOT,
    )


# ---- Happy path ---------------------------------------------------------------


def test_the_admitted_source_resolves_a_complete_overlay_and_keeps_v1_identities(
    tmp_path: Path,
) -> None:
    fixture = _build_and_prove(tmp_path / "artifacts", _ADMITTED_ONLY_TASKS)
    source = _resolve(fixture)
    assert source.counts == {
        "source_entries": 2,
        "admitted_entries": 2,
        "rejected_entries": 0,
        "admitted_states": 5,
        "admitted_transitions": 3,
    }
    records = list(source.iter_records())
    assert len(records) == 2
    live_v2_sha256 = str(fixture.plan["process_v2_identity"]["process_identity_sha256"])
    pinned_v1_sha256 = str(fixture.plan["pinned_process_identity"]["process_identity_sha256"])
    assert live_v2_sha256 != pinned_v1_sha256
    for record in records:
        # The V1 row keeps its V1 identity and is never relabelled as V2.
        assert record.v1_identity.process_semantics == PROCESS_SEMANTICS
        assert record.v1_identity.process_identity_sha256 == pinned_v1_sha256
        assert record.v1_record["process_identity_sha256"] == pinned_v1_sha256
        assert record.v1_identity.record_sha256 == record.v1_record["record_sha256"]
        # The V2 decision is a separate object carrying the live V2 identity.
        assert record.v2_admission.admitted is True
        assert record.v2_admission.process_v2_identity_sha256 == live_v2_sha256
        assert record.v2_admission.run_identity_sha256 == fixture.plan["run_identity_sha256"]
        assert len(record.v2_admission.process_v2_atom_delete_candidates) == len(
            record.addressed.path.states
        )
        assert record.addressed.address.trace_id == record.v1_identity.trace_id
        overlay = record.v2_admission.as_payload()
        assert overlay["schema"].endswith("admission_overlay")
        assert record.v1_identity.as_payload()["schema"].endswith("v1_payload_identity")


def test_the_admitted_source_yields_only_admitted_records(tmp_path: Path) -> None:
    fixture = _build_and_prove(tmp_path / "artifacts", _TASKS_WITH_EXCLUSION)
    source = _resolve(fixture)
    assert source.counts["source_entries"] == 3
    assert source.counts["admitted_entries"] == 2
    assert source.counts["rejected_entries"] == 1
    assert source.rejected_traces_by_code == {"atom_delete_outside_process_v2_mask": 1}
    records = list(source.iter_records())
    assert len(records) == 2
    assert "c1ccccc1" not in {
        key for record in records for key in record.v1_record["canonical_state_keys"]
    }


def test_the_admitted_source_grants_no_authority(tmp_path: Path) -> None:
    fixture = _build_and_prove(tmp_path / "artifacts", _ADMITTED_ONLY_TASKS)
    source = _resolve(fixture)
    identity = source.identity()
    assert identity["schema"] == ADMITTED_SOURCE_SCHEMA
    assert identity["schema_version"] == ADMITTED_SOURCE_SCHEMA_VERSION == 3
    assert identity["status"] == ADMITTED_SOURCE_STATUS
    assert "NO_TRAINING_AUTHORITY" in identity["status"]
    for name in AUTHORITY_FIELDS:
        assert getattr(source, name) is False, name
        assert identity[name] is False, name
    assert identity["admitted_source_sha256"] == _self_hash(identity, "admitted_source_sha256")
    assert identity["counts"] == dict(source.counts)
    assert identity["adapter_implementation_sha256"] == _sha256(
        ROOT / "src/compose_v4/data/editing_process_v2_admitted_source.py"
    )
    # The retired spelling is gone from the object as well as from the artifact.
    # A property that still answered it would keep the second vocabulary alive
    # exactly where a consumer is most likely to read it.
    assert not hasattr(source, "p50_authorized")
    assert "p50_authorized" not in identity


def test_the_authority_surface_is_exactly_the_frozen_vocabulary() -> None:
    """The properties cannot drift away from ``AUTHORITY_FIELDS``.

    They are written out one by one for readability, so without this the seven
    could fall out of step with the vocabulary they are supposed to be.
    """

    observed = {
        name
        for name in dir(ProcessV2AdmittedSource)
        if name.endswith("_authorized")
        and isinstance(getattr(ProcessV2AdmittedSource, name), property)
    }
    assert observed == set(AUTHORITY_FIELDS)
    # And none of them is settable: the defect being closed is a constructor
    # argument whose value the descriptor ignored.
    for name in AUTHORITY_FIELDS:
        assert getattr(ProcessV2AdmittedSource, name).fset is None, name
    with pytest.raises(TypeError):
        ProcessV2AdmittedSource(  # type: ignore[call-arg]
            plan={},
            artifact_root=Path("/artifacts"),
            repo_root=ROOT,
            completion={},
            counts={},
            rejected_traces_by_code={},
            adapter_implementation_sha256="0" * 64,
            training_authorized=True,
        )


def test_the_identity_is_sorted_at_every_depth_and_survives_canonical_reserialization(
    tmp_path: Path,
) -> None:
    """Byte stability is what lets a consumer embed this descriptor verbatim."""

    fixture = _build_and_prove(tmp_path / "artifacts", _TASKS_WITH_EXCLUSION)
    identity = _resolve(fixture).identity()

    def _mappings(node: object):
        if isinstance(node, Mapping):
            yield node
            for value in node.values():
                yield from _mappings(value)
        elif isinstance(node, list):
            for value in node:
                yield from _mappings(value)

    seen = list(_mappings(identity))
    assert len(seen) >= 4, "the nested bodies must be reached"
    for node in seen:
        assert list(node) == sorted(node), node

    # `sort_keys=True` is what every canonical serializer in the chain uses, so
    # a descriptor that is not already sorted comes back reordered.
    reserialized = json.loads(json.dumps(identity, sort_keys=True))
    assert json.dumps(reserialized) == json.dumps(identity)


# ---- The owning identity validator --------------------------------------------


def test_a_real_resolved_identity_validates(tmp_path: Path) -> None:
    fixture = _build_and_prove(tmp_path / "artifacts", _TASKS_WITH_EXCLUSION)
    identity = _resolve(fixture).identity()
    assert validate_process_v2_admitted_source_identity(identity, repo_root=ROOT) == identity


def test_schema_version_two_is_refused_as_an_incompatibility(tmp_path: Path) -> None:
    """Not read as a subset of version 3, and told why it cannot be converted."""

    fixture = _build_and_prove(tmp_path / "artifacts", _ADMITTED_ONLY_TASKS)
    identity = _resolve(fixture).identity()
    downgraded = {**identity, "schema_version": 2}
    downgraded["admitted_source_sha256"] = _self_hash(downgraded, "admitted_source_sha256")
    with pytest.raises(
        ProcessV2AdmittedSourceIdentityError, match="incompatible with 3"
    ) as raised:
        validate_process_v2_admitted_source_identity(downgraded, repo_root=ROOT)
    assert "converter would have to invent" in str(raised.value)


def test_a_coerced_or_uppercase_field_is_refused(tmp_path: Path) -> None:
    fixture = _build_and_prove(tmp_path / "artifacts", _ADMITTED_ONLY_TASKS)
    identity = _resolve(fixture).identity()
    for key, value, message in (
        ("completion_sha256", identity["completion_sha256"].upper(), "lowercase hex"),
        ("completion_sha256", "0" * 63, "lowercase hex"),
        ("schema_version", True, "not an exact int"),
        ("status", 3, "not str"),
    ):
        broken = _reseal({**identity, key: value})
        with pytest.raises(ProcessV2AdmittedSourceIdentityError, match=message):
            validate_process_v2_admitted_source_identity(broken, repo_root=ROOT)


def test_a_missing_extra_or_renamed_field_is_refused(tmp_path: Path) -> None:
    fixture = _build_and_prove(tmp_path / "artifacts", _ADMITTED_ONLY_TASKS)
    identity = _resolve(fixture).identity()

    missing = _reseal({k: v for k, v in identity.items() if k != "plan_sha256"})
    with pytest.raises(ProcessV2AdmittedSourceIdentityError, match="missing"):
        validate_process_v2_admitted_source_identity(missing, repo_root=ROOT)

    extra = _reseal({**identity, "measured_later": "0" * 64})
    with pytest.raises(ProcessV2AdmittedSourceIdentityError, match="unexpected"):
        validate_process_v2_admitted_source_identity(extra, repo_root=ROOT)

    renamed = {k: v for k, v in identity.items() if k != "plan_sha256"}
    renamed["rebind_plan_sha256"] = identity["plan_sha256"]
    with pytest.raises(ProcessV2AdmittedSourceIdentityError, match="missing.*unexpected"):
        validate_process_v2_admitted_source_identity(_reseal(renamed), repo_root=ROOT)


def test_an_unsorted_mapping_is_refused_even_though_it_self_hashes(
    tmp_path: Path,
) -> None:
    """The canonical hash sorts, so nothing else would catch a reordering."""

    fixture = _build_and_prove(tmp_path / "artifacts", _ADMITTED_ONLY_TASKS)
    identity = _resolve(fixture).identity()
    shuffled = dict(reversed(list(identity.items())))
    assert shuffled == identity, "reordering alone must not change equality"
    assert shuffled["admitted_source_sha256"] == _self_hash(
        shuffled, "admitted_source_sha256"
    ), "the self-hash cannot see the reordering, which is why the check exists"
    with pytest.raises(ProcessV2AdmittedSourceIdentityError, match="sorted"):
        validate_process_v2_admitted_source_identity(shuffled, repo_root=ROOT)


def test_a_granted_authority_field_is_refused_at_any_depth(tmp_path: Path) -> None:
    """Including inside a mapping nested in a list, which no field-set check sees."""

    fixture = _build_and_prove(tmp_path / "artifacts", _ADMITTED_ONLY_TASKS)
    identity = _resolve(fixture).identity()

    for name in AUTHORITY_FIELDS:
        with pytest.raises(ProcessV2AdmittedSourceIdentityError, match="authority"):
            validate_process_v2_admitted_source_identity(
                _reseal({**identity, name: True}), repo_root=ROOT
            )

    nested_mapping = _reseal(
        {
            **identity,
            "physical_execution_identity": {
                **identity["physical_execution_identity"],
                "gate_zero_authorized": True,
            },
        }
    )
    with pytest.raises(ProcessV2AdmittedSourceIdentityError, match="authority"):
        validate_process_v2_admitted_source_identity(nested_mapping, repo_root=ROOT)

    nested_in_list = _reseal(
        {
            **identity,
            "physical_execution_identity": {
                **identity["physical_execution_identity"],
                "range_task_inventory": [{"t1_authorized": True}],
            },
        }
    )
    with pytest.raises(ProcessV2AdmittedSourceIdentityError, match="authority") as raised:
        validate_process_v2_admitted_source_identity(nested_in_list, repo_root=ROOT)
    assert "[0].t1_authorized" in str(raised.value)


def test_the_retired_bounded_p50_spelling_is_refused(tmp_path: Path) -> None:
    fixture = _build_and_prove(tmp_path / "artifacts", _ADMITTED_ONLY_TASKS)
    identity = _resolve(fixture).identity()
    retired = {k: v for k, v in identity.items() if k != "bounded_p50_authorized"}
    retired["p50_authorized"] = False
    with pytest.raises(ProcessV2AdmittedSourceIdentityError, match="retired authority"):
        validate_process_v2_admitted_source_identity(_reseal(retired), repo_root=ROOT)


def test_a_moved_adapter_or_process_identity_is_refused(tmp_path: Path) -> None:
    fixture = _build_and_prove(tmp_path / "artifacts", _ADMITTED_ONLY_TASKS)
    identity = _resolve(fixture).identity()
    moved_adapter = _reseal({**identity, "adapter_implementation_sha256": "0" * 64})
    with pytest.raises(
        ProcessV2AdmittedSourceIdentityError, match="different adapter"
    ):
        validate_process_v2_admitted_source_identity(moved_adapter, repo_root=ROOT)
    moved_process = _reseal({**identity, "process_v2_identity_sha256": "1" * 64})
    with pytest.raises(
        ProcessV2AdmittedSourceIdentityError, match="live Process-V2 identity"
    ):
        validate_process_v2_admitted_source_identity(moved_process, repo_root=ROOT)


def test_the_census_and_its_evidence_must_both_reconcile(tmp_path: Path) -> None:
    fixture = _build_and_prove(tmp_path / "artifacts", _TASKS_WITH_EXCLUSION)
    identity = _resolve(fixture).identity()

    unbalanced = _reseal(
        {**identity, "counts": {**identity["counts"], "source_entries": 99}}
    )
    with pytest.raises(ProcessV2AdmittedSourceIdentityError, match="does not reconcile"):
        validate_process_v2_admitted_source_identity(unbalanced, repo_root=ROOT)

    # `source == admitted + rejected` still holds here; only the state/transition
    # identity breaks, so this is the check that adds information.
    inflated = _reseal(
        {
            **identity,
            "counts": {
                **identity["counts"],
                "admitted_states": int(identity["counts"]["admitted_states"]) + 1,
            },
        }
    )
    with pytest.raises(
        ProcessV2AdmittedSourceIdentityError, match="evidence does not reconcile"
    ):
        validate_process_v2_admitted_source_identity(inflated, repo_root=ROOT)


def test_a_rejection_census_that_does_not_sum_is_refused(tmp_path: Path) -> None:
    fixture = _build_and_prove(tmp_path / "artifacts", _TASKS_WITH_EXCLUSION)
    identity = _resolve(fixture).identity()
    assert identity["rejected_traces_by_code"], "the fixture must reject something"
    code = next(iter(identity["rejected_traces_by_code"]))
    broken = _reseal(
        {**identity, "rejected_traces_by_code": {code: 7}}
    )
    with pytest.raises(ProcessV2AdmittedSourceIdentityError, match="sums to 7"):
        validate_process_v2_admitted_source_identity(broken, repo_root=ROOT)

    unknown = _reseal({**identity, "rejected_traces_by_code": {"invented_reason": 1}})
    with pytest.raises(ProcessV2AdmittedSourceIdentityError, match="does not define"):
        validate_process_v2_admitted_source_identity(unknown, repo_root=ROOT)


def test_the_nested_bodies_validate_through_their_own_schemas(tmp_path: Path) -> None:
    fixture = _build_and_prove(tmp_path / "artifacts", _TASKS_WITH_EXCLUSION)
    identity = _resolve(fixture).identity()

    physical = identity["physical_execution_identity"]
    assert physical["schema"] == PHYSICAL_EXECUTION_IDENTITY_SCHEMA
    assert physical["schema_version"] == PHYSICAL_EXECUTION_IDENTITY_SCHEMA_VERSION == 1
    ledger = identity["rejection_ledger"]
    assert ledger["schema"] == REJECTION_LEDGER_SCHEMA
    assert ledger["schema_version"] == REJECTION_LEDGER_SCHEMA_VERSION

    for block, key, value, message in (
        ("physical_execution_identity", "schema_version", 2, "schema_version"),
        ("physical_execution_identity", "range_task_count", "3", "not an exact int"),
        ("rejection_ledger", "sort_order", "entry_index_only", "sort_order"),
        ("rejection_ledger", "schema", "compose.other", "schema"),
    ):
        nested = dict(identity[block])
        nested[key] = value
        self_hash_field = (
            "physical_execution_sha256"
            if block == "physical_execution_identity"
            else "rejection_ledger_sha256"
        )
        nested[self_hash_field] = _self_hash(nested, self_hash_field)
        broken = _reseal({**identity, block: dict(sorted(nested.items()))})
        with pytest.raises(ProcessV2AdmittedSourceIdentityError, match=message):
            validate_process_v2_admitted_source_identity(broken, repo_root=ROOT)


def test_a_replay_address_for_another_run_is_refused(tmp_path: Path) -> None:
    """Every other field checks out; only the nested run identity moved."""

    fixture = _build_and_prove(tmp_path / "artifacts", _ADMITTED_ONLY_TASKS)
    identity = _resolve(fixture).identity()
    physical = {**identity["physical_execution_identity"], "run_identity_sha256": "9" * 64}
    physical["physical_execution_sha256"] = _self_hash(physical, "physical_execution_sha256")
    broken = _reseal({**identity, "physical_execution_identity": dict(sorted(physical.items()))})
    with pytest.raises(
        ProcessV2AdmittedSourceIdentityError, match="must address this run"
    ):
        validate_process_v2_admitted_source_identity(broken, repo_root=ROOT)


def test_a_disagreeing_self_hash_is_refused(tmp_path: Path) -> None:
    fixture = _build_and_prove(tmp_path / "artifacts", _ADMITTED_ONLY_TASKS)
    identity = _resolve(fixture).identity()
    with pytest.raises(ProcessV2AdmittedSourceIdentityError, match="self-hash"):
        validate_process_v2_admitted_source_identity(
            {**identity, "admitted_source_sha256": "2" * 64}, repo_root=ROOT
        )


def test_a_non_object_identity_is_refused() -> None:
    with pytest.raises(ProcessV2AdmittedSourceIdentityError, match="must be an object"):
        validate_process_v2_admitted_source_identity([], repo_root=ROOT)


# ---- Fail-closed refusals -----------------------------------------------------


def test_a_missing_completion_refuses(tmp_path: Path) -> None:
    fixture = _build_and_prove(tmp_path / "artifacts", _ADMITTED_ONLY_TASKS)
    (fixture.run_root / COMPLETION_FILENAME).unlink()
    with pytest.raises(ProcessV2AdmittedSourceIncomplete, match="completion is absent"):
        _resolve(fixture)


def test_a_missing_task_result_refuses(tmp_path: Path) -> None:
    fixture = _build_and_prove(tmp_path / "artifacts", _ADMITTED_ONLY_TASKS)
    removed = fixture.run_root / TASK_DIRNAME / fixture.plan["tasks"][0]["task_identity_sha256"]
    for path in removed.iterdir():
        path.unlink()
    removed.rmdir()
    with pytest.raises(ProcessV2AdmittedSourceIncomplete, match="missing 1 planned task"):
        _resolve(fixture)


def test_a_tampered_published_plan_refuses(tmp_path: Path) -> None:
    from compose_v4.data.editing_process_v2_rebind import PLAN_FILENAME

    fixture = _build_and_prove(tmp_path / "artifacts", _ADMITTED_ONLY_TASKS)
    (fixture.run_root / PLAN_FILENAME).write_bytes(b'{"plan": "not the published bytes"}\n')
    with pytest.raises(ProcessV2AdmittedSourceIncomplete, match="exact published"):
        _resolve(fixture)


def test_a_stale_live_process_v2_identity_refuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The overlay was written under one V2 identity; a moved one cannot resolve."""

    import compose_v4.rewrite.editing_v2_process_identity as identity_module

    fixture = _build_and_prove(tmp_path / "artifacts", _ADMITTED_ONLY_TASKS)
    original = identity_module.editing_process_v2_identity()
    moved = {
        **{key: value for key, value in original.items() if key != "process_identity_sha256"},
        "contract_sha256": "0" * 64,
    }
    moved["process_identity_sha256"] = _self_hash(moved, "process_identity_sha256")
    monkeypatch.setattr(identity_module, "editing_process_v2_identity", lambda: moved)
    with pytest.raises(ProcessV2AdmittedSourceError, match="stale or invalid"):
        _resolve(fixture)


def test_a_completion_that_does_not_bind_the_live_identity_refuses(tmp_path: Path) -> None:
    fixture = _build_and_prove(tmp_path / "artifacts", _ADMITTED_ONLY_TASKS)
    completion_path = fixture.run_root / COMPLETION_FILENAME
    completion = json.loads(completion_path.read_text())
    completion["process_v2_identity_sha256"] = "0" * 64
    completion["completion_sha256"] = _self_hash(completion, "completion_sha256")
    completion_path.write_bytes(
        json.dumps(
            completion,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        + b"\n"
    )
    with pytest.raises(ProcessV2AdmittedSourceError, match="does not bind this plan"):
        _resolve(fixture)


def test_partial_overlay_coverage_refuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An overlay that decides only some rows is refused, never partially used."""

    fixture = _build_and_prove(tmp_path / "artifacts", _ADMITTED_ONLY_TASKS)
    original = admitted_source_module._overlay_index

    def dropped(plan, *, artifact_root):
        index = original(plan, artifact_root=artifact_root)
        first = sorted(index)[0]
        index[first] = {}
        return index

    monkeypatch.setattr(admitted_source_module, "_overlay_index", dropped)
    with pytest.raises(ProcessV2AdmittedSourceIncomplete, match="decides 0 of 1 records"):
        _resolve(fixture)


def test_an_overlay_that_disagrees_with_its_payload_refuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both sides are content addressed, so this guard is exercised directly.

    A published proof and its V1 row cannot disagree without one of them
    failing its own hash, which is why the disagreement is injected here rather
    than written to disk.  The guard still has to exist: it is what makes the
    join safe if a future writer stops content addressing one side.
    """

    fixture = _build_and_prove(tmp_path / "artifacts", _ADMITTED_ONLY_TASKS)
    source = _resolve(fixture)
    original = admitted_source_module._read_proof_rows

    def retargeted(path: Path):
        rows = original(path)
        return [{**row, "trace_id": "fixture-forged-trace-id"} for row in rows]

    monkeypatch.setattr(admitted_source_module, "_read_proof_rows", retargeted)
    with pytest.raises(ProcessV2AdmittedSourceError, match="disagrees with its V1 payload"):
        list(source.iter_records())


def test_an_unreadable_v1_payload_refuses(tmp_path: Path) -> None:
    from compose_v4.data.semantic_packed_trace_store import SHARD_FILENAME
    from compose_v4.data.semantic_trace_migration_materializer import (
        SEMANTIC_ARTIFACT_DIRNAME,
    )

    fixture = _build_and_prove(tmp_path / "artifacts", _ADMITTED_ONLY_TASKS)
    source = _resolve(fixture)
    v1_task = fixture.plan["v1_payload_binding"]["v1_tasks"][0]
    shard = (
        fixture.payload_root
        / TASK_DIRNAME
        / str(v1_task["v1_task_identity_sha256"])
        / SEMANTIC_ARTIFACT_DIRNAME
        / SHARD_FILENAME
    )
    shard.write_bytes(b"not a gzip shard")
    with pytest.raises(ProcessV2AdmittedSourceError, match="unreadable under its pinned"):
        list(source.iter_records())


# ---- The retired spelling stays inside the frozen boundary ---------------------


#: The only places the retired bounded-P50 spelling may appear as a real key.
#: Everything else is a producer, and a producer emitting it puts a second name
#: for one concept back into circulation.
_RETIRED_P50_ALLOWED: dict[str, str] = {
    "src/compose_v4/rewrite/editing_v2_process_identity.py": (
        "the frozen process-contract builder; its bytes are hashed into the "
        "scientific Process-V2 identity, so renaming the field here moves that "
        "identity and is an owner decision rather than a migration"
    ),
    "configs/editing_v2_semantic_process_v2.json": (
        "the serialization of that same frozen contract, hashed alongside it"
    ),
    "src/compose_v4/data/editing_v2_process_v2_schema.py": (
        "RETIRED_AUTHORITY_FIELDS names the spelling in order to REFUSE it; a "
        "refusal table is the opposite of a producer"
    ),
}

_RETIRED_P50 = "p50_authorized"


def _emitted_keys(path: Path) -> set[str]:
    """Field names a file actually emits, ignoring prose.

    A byte scan cannot tell a dict key from a sentence describing one, and this
    guard has to survive the modules whose docstrings explain why the spelling was
    retired. So keys are read structurally: a mapping key, a keyword argument, an
    attribute, or a bound name.
    """

    if path.suffix == ".json":
        found: set[str] = set()

        def walk(node: object) -> None:
            if isinstance(node, dict):
                found.update(node)
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(json.loads(path.read_text()))
        return found
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return set()
    keys: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys.update(
                key.value
                for key in node.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            )
        elif isinstance(node, ast.keyword) and node.arg:
            keys.add(node.arg)
        elif isinstance(node, ast.Attribute):
            keys.add(node.attr)
        elif isinstance(node, ast.Name):
            keys.add(node.id)
    return keys


def test_no_artifact_producer_emits_the_retired_p50_spelling() -> None:
    """One spelling outside the frozen boundary, enforced rather than intended."""

    offenders: dict[str, str] = {}
    for area in ("src", "scripts", "modal_apps", "configs", "recipes"):
        base = ROOT / area
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix not in {".py", ".json"}:
                continue
            relative = str(path.relative_to(ROOT))
            if relative in _RETIRED_P50_ALLOWED:
                continue
            if _RETIRED_P50 in _emitted_keys(path):
                offenders[relative] = "emits the retired bounded-P50 spelling"
    assert offenders == {}, offenders

    # The allowlist is not aspirational: each entry must still contain it, so a
    # stale exemption is a failure rather than dead weight.
    for relative, reason in sorted(_RETIRED_P50_ALLOWED.items()):
        assert _RETIRED_P50 in _emitted_keys(ROOT / relative), (relative, reason)


def test_the_producer_guard_actually_detects_a_new_producer(tmp_path: Path) -> None:
    """Mutation control: without this the scan could match nothing and pass."""

    offender = tmp_path / "producer.py"
    offender.write_text('PAYLOAD = {"p50_authorized": False}\n')
    assert _RETIRED_P50 in _emitted_keys(offender)

    prose = tmp_path / "prose.py"
    prose.write_text('"""Explains why p50_authorized was retired."""\n')
    assert _RETIRED_P50 not in _emitted_keys(prose), (
        "a sentence about the field is not an emission of it"
    )

    config = tmp_path / "artifact.json"
    config.write_text('{"decisions": {"p50_authorized": false}}\n')
    assert _RETIRED_P50 in _emitted_keys(config), "nested JSON keys must be reached"
