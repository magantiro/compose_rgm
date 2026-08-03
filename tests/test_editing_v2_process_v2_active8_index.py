"""The Process-V2 Active8 decision index, and what it must refuse.

Acceptance test 10 lives here: the index satisfies the frozen structural
protocol, has no V1 class anywhere in its MRO, and refuses an injected V1 plan,
V1 process identity, V1 policy and V1 model/process contract.

The chain is the real one built by
``test_editing_v2_process_v2_active8_decisions``; rebuilding it here would create
a second builder whose drift from the first is invisible.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import pytest

from compose_v4.data.editing_v2_process_v2_active8_decision_index import (
    ACCEPTED_TRANSITION_FIELDS,
    RESOLVED_TRACE_FIELDS,
    ProcessV2Active8DecisionIndex,
    ProcessV2Active8DecisionIndexError,
    resolve_process_v2_active8_decision_index,
)
from compose_v4.data.editing_v2_process_v2_active8_interfaces import (
    ACTIVE8_CENSUS_FIELDS,
    ACTIVE8_EXCLUDED,
    TRACE_KEY_FIELDS,
    UPSTREAM_REJECTED,
    ProcessV2Active8Index,
)
from compose_v4.data.editing_v2_process_v2_active8_mapreduce import (
    COMPLETION_FILENAME,
    PLAN_FILENAME,
    ProcessV2Active8Incomplete,
)
from compose_v4.data.editing_v2_process_v2_schema import (
    StructuralDecisionIndex,
    canonical_bytes,
    canonical_sha256,
    require_census_reconciles,
)
from compose_v4.data.editing_v2_semantic_active8_decision_source import (
    EditingV2SemanticActive8DecisionIndex,
)
from compose_v4.data.editing_v2_semantic_active8_source_adapter import (
    EditingV2SemanticActive8SourceInventory,
)
from compose_v4.experiments.editing_gate_zero_semantic_contract import (
    build_gate_zero_semantic_contract,
)
from compose_v4.rewrite.editing_v2_process_identity import editing_v2_process_identity

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "tests"))

import test_editing_v2_process_v2_active8_decisions as chain_fixture  # noqa: E402

ROOT = _REPO_ROOT


@pytest.fixture(name="resolved", scope="module")
def _resolved(tmp_path_factory: pytest.TempPathFactory):
    """One completed Active8 run plus the index resolved from its bytes."""

    monkeypatch = pytest.MonkeyPatch()
    for name, definition in {
        **chain_fixture.binder_fixture._EXTRA_TRACES,
        **chain_fixture._EXTRA_TRACES,
    }.items():
        monkeypatch.setitem(chain_fixture.v1_fixture._FIXTURE_TRACES, name, definition)
    try:
        chain = chain_fixture.build_chain(tmp_path_factory.mktemp("active8_index"))
    finally:
        monkeypatch.undo()
    plan, completion = chain_fixture.run_active8(chain, prefix="/artifacts/active8_index")
    run_root = chain_fixture.mounted(chain, plan["run_artifact_root"])
    index = resolve_process_v2_active8_decision_index(
        plan_path=run_root / PLAN_FILENAME,
        completion_path=run_root / COMPLETION_FILENAME,
        artifact_root=chain.artifact_root,
        repo_root=ROOT,
    )
    return chain, plan, completion, run_root, index


# ---- Acceptance test 10 -------------------------------------------------------


def test_the_index_satisfies_both_protocols_with_no_v1_class_in_its_mro(
    resolved,
) -> None:
    """Acceptance 10. Without this a V1 base could be acquired unnoticed.

    Inheriting a V1 decision index would drag in live-V1 schema and identity
    revalidation, and the payload underneath this run is sealed under a
    superseded V1 identity, so a V1 base would refuse it by construction.  The
    MRO is asserted exactly, not merely searched for known V1 names, because the
    next V1 class to exist would not be in a name list.
    """

    _chain, _plan, _completion, _run_root, index = resolved

    assert isinstance(index, StructuralDecisionIndex)
    assert isinstance(index, ProcessV2Active8Index)

    assert type(index).__mro__ == (ProcessV2Active8DecisionIndex, object)
    assert [cls.__name__ for cls in type(index).__mro__] == [
        "ProcessV2Active8DecisionIndex",
        "object",
    ]
    for v1_class in (
        EditingV2SemanticActive8DecisionIndex,
        EditingV2SemanticActive8SourceInventory,
    ):
        assert v1_class not in type(index).__mro__
    # Structural satisfaction, not declared: neither protocol is a base either.
    assert StructuralDecisionIndex not in type(index).__mro__
    assert ProcessV2Active8Index not in type(index).__mro__

    # And the census the smaller protocol declares actually reconciles, which
    # `isinstance` alone never proves: the runtime check is by member name.
    require_census_reconciles(dict(index.counts()), label="the resolved index")


def test_the_index_refuses_an_injected_v1_plan(resolved) -> None:
    """Acceptance 10, V1 plan. Without this a V1 run could be read as a V2 one."""

    _chain, _plan, _completion, run_root, _index = resolved
    v1_plan = {
        "schema": "compose.data.editing_v2_semantic_active8_decision_plan",
        "schema_version": 1,
        "status": "FROZEN_DECISION_TASKS_NO_DOWNSTREAM_AUTHORITY",
        "tasks": [],
    }
    injected = run_root / "V1_PLAN.json"
    injected.write_bytes(canonical_bytes(v1_plan) + b"\n")
    with pytest.raises(ProcessV2Active8DecisionIndexError, match="failed revalidation") as refusal:
        resolve_process_v2_active8_decision_index(
            plan_path=injected,
            completion_path=run_root / COMPLETION_FILENAME,
            artifact_root=_chain.artifact_root,
            repo_root=ROOT,
        )
    # The cause is pinned so all four V1-injection refusals cannot pass for one
    # shared reason: this one is refused on its field set.
    assert "plan fields disagree" in str(refusal.value.__cause__)


def test_the_index_refuses_an_injected_v1_process_identity(resolved) -> None:
    """Acceptance 10, V1 identity. Without this the superseded process could bind."""

    chain, plan, _completion, run_root, _index = resolved
    v1_identity = editing_v2_process_identity()
    assert (
        v1_identity["process_identity_sha256"]
        != plan["process_v2_identity"]["process_identity_sha256"]
    )
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    body["process_v2_identity"] = dict(v1_identity)
    injected = run_root / "V1_IDENTITY_PLAN.json"
    injected.write_bytes(
        canonical_bytes({**body, "plan_sha256": canonical_sha256(body)}) + b"\n"
    )
    with pytest.raises(ProcessV2Active8DecisionIndexError, match="failed revalidation") as refusal:
        resolve_process_v2_active8_decision_index(
            plan_path=injected,
            completion_path=run_root / COMPLETION_FILENAME,
            artifact_root=chain.artifact_root,
            repo_root=ROOT,
        )
    assert "another Process-V2 process identity" in str(refusal.value.__cause__)


def test_the_index_refuses_an_injected_v1_policy(resolved) -> None:
    """Acceptance 10, V1 policy. Without this two admission laws share one hash.

    The V1 policy is not a synthetic stand-in: it is the real V1 admission
    policy body, which differs from the Process-V2 one exactly where the two
    processes differ -- the process identity it binds and the atom-delete
    semantics it requires.
    """

    from compose_v4.data.editing_v2_semantic_active8_admission import (
        build_semantic_active8_admission_policy,
    )

    chain, plan, _completion, run_root, _index = resolved
    v1_policy = build_semantic_active8_admission_policy().as_payload()
    assert v1_policy["policy_sha256"] != plan["policy"]["policy_sha256"]
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    body["policy"] = v1_policy
    injected = run_root / "V1_POLICY_PLAN.json"
    injected.write_bytes(
        canonical_bytes({**body, "plan_sha256": canonical_sha256(body)}) + b"\n"
    )
    with pytest.raises(ProcessV2Active8DecisionIndexError, match="failed revalidation") as refusal:
        resolve_process_v2_active8_decision_index(
            plan_path=injected,
            completion_path=run_root / COMPLETION_FILENAME,
            artifact_root=chain.artifact_root,
            repo_root=ROOT,
        )
    assert "does not carry the live Active8 policy" in str(refusal.value.__cause__)


def test_the_index_refuses_an_injected_v1_model_process_contract(resolved) -> None:
    """Acceptance 10, V1 contract. Without this a V1 model could decide V2 rows.

    The runtime identity is resealed around the real V1 model/process contract
    and the real V1 semantic model identity, so it is internally consistent and
    differs from the V2 one only in what it describes.  That is the case a
    self-hash check cannot catch.
    """

    chain, plan, _completion, run_root, _index = resolved
    v1_contract = build_gate_zero_semantic_contract()
    runtime = dict(plan["model_runtime_identity"])
    runtime_body = {
        key: value for key, value in runtime.items() if key != "identity_sha256"
    }
    runtime_body["semantic_model_identity"] = dict(v1_contract["model_identity"])
    runtime_body["semantic_model_process_contract_sha256"] = str(
        v1_contract["contract_sha256"]
    )
    runtime_body["process_identity_sha256"] = str(v1_contract["process_identity_sha256"])
    resealed = {**runtime_body, "identity_sha256": canonical_sha256(runtime_body)}
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    body["model_runtime_identity"] = resealed
    injected = run_root / "V1_CONTRACT_PLAN.json"
    injected.write_bytes(
        canonical_bytes({**body, "plan_sha256": canonical_sha256(body)}) + b"\n"
    )
    with pytest.raises(ProcessV2Active8DecisionIndexError, match="failed revalidation") as refusal:
        resolve_process_v2_active8_decision_index(
            plan_path=injected,
            completion_path=run_root / COMPLETION_FILENAME,
            artifact_root=chain.artifact_root,
            repo_root=ROOT,
        )
    assert "another editing process" in str(refusal.value.__cause__)
    # And the runtime validator refuses it directly, so the refusal above is not
    # an accident of some other field moving.
    from compose_v4.data.editing_v2_process_v2_active8_runtime import (
        ProcessV2Active8RuntimeError,
        validate_process_v2_active8_model_runtime_identity,
    )

    with pytest.raises(ProcessV2Active8RuntimeError, match="another editing process"):
        validate_process_v2_active8_model_runtime_identity(resealed)


# ---- What the index reopens ---------------------------------------------------


def test_the_index_reopens_the_artifacts_rather_than_trusting_an_object(
    resolved,
) -> None:
    """Without this an in-memory object could stand in for the published run.

    Removing one published task's bytes must make a *newly resolved* index
    refuse.  An index that had captured a Python object at construction time
    would keep answering from it.
    """

    chain, plan, _completion, run_root, index = resolved
    assert index.counts()["source_entries"] > 0

    task = plan["tasks"][0]
    output = chain_fixture.mounted(chain, task["output_artifact_path"])
    # Moved OUT of the task namespace rather than renamed inside it: an
    # unexpected sibling is a different refusal, and this test is about the
    # absent result, not about the extra object.
    quarantine = chain.artifact_root / "quarantine"
    quarantine.mkdir(exist_ok=True)
    moved = quarantine / str(task["task_identity_sha256"])
    output.rename(moved)
    try:
        with pytest.raises(ProcessV2Active8Incomplete):
            resolve_process_v2_active8_decision_index(
                plan_path=run_root / PLAN_FILENAME,
                completion_path=run_root / COMPLETION_FILENAME,
                artifact_root=chain.artifact_root,
                repo_root=ROOT,
            )
    finally:
        moved.rename(output)

    # Restored, and a fresh resolve succeeds again with the same identity.
    again = resolve_process_v2_active8_decision_index(
        plan_path=run_root / PLAN_FILENAME,
        completion_path=run_root / COMPLETION_FILENAME,
        artifact_root=chain.artifact_root,
        repo_root=ROOT,
    )
    assert again.index_identity_sha256 == index.index_identity_sha256


def test_the_index_identity_is_deterministic_and_grants_nothing(resolved) -> None:
    """Without this the provenance a consumer records could drift per resolve."""

    chain, _plan, _completion, run_root, index = resolved
    second = resolve_process_v2_active8_decision_index(
        plan_path=run_root / PLAN_FILENAME,
        completion_path=run_root / COMPLETION_FILENAME,
        artifact_root=chain.artifact_root,
        repo_root=ROOT,
    )
    assert canonical_bytes(dict(index.identity())) == canonical_bytes(
        dict(second.identity())
    )
    identity = dict(index.identity())
    assert identity["index_identity_sha256"] == index.index_identity_sha256
    body = {k: v for k, v in identity.items() if k != "index_identity_sha256"}
    assert canonical_sha256(body) == index.index_identity_sha256
    for field in (
        "training_authorized",
        "gate_zero_authorized",
        "t1_authorized",
        "bounded_p50_authorized",
        "long_training_authorized",
        "checkpoint_selection_authorized",
        "final_test_selection_authorized",
    ):
        assert identity[field] is False, field
        assert getattr(index, field) is False, field


def test_the_index_keeps_the_two_process_identities_distinct(resolved) -> None:
    """Without this the superseded payload identity could be read as current."""

    _chain, plan, _completion, _run_root, index = resolved
    assert index.process_identity_sha256 == (
        plan["process_v2_identity"]["process_identity_sha256"]
    )
    assert index.pinned_payload_process_identity_sha256 == (
        plan["pinned_process_identity"]["process_identity_sha256"]
    )
    assert index.process_identity_sha256 != index.pinned_payload_process_identity_sha256


def test_the_reason_coded_census_never_merges_the_two_categories(resolved) -> None:
    """Without this "we did not look" and "we looked and said no" become one number."""

    _chain, _plan, completion, _run_root, index = resolved
    census = dict(index.rejected_traces_by_code())
    upstream = {k: v for k, v in census.items() if k.startswith(f"{UPSTREAM_REJECTED}:")}
    excluded = {k: v for k, v in census.items() if k.startswith(f"{ACTIVE8_EXCLUDED}:")}
    assert upstream and excluded
    assert set(census) == set(upstream) | set(excluded)
    counts = index.counts()
    assert sum(upstream.values()) == counts["upstream_rejected_entries"]
    assert sum(excluded.values()) == counts["active8_excluded_entries"]
    assert sum(census.values()) == counts["rejected_entries"]
    # And the four seam census fields are all present and reconcile.
    assert set(ACTIVE8_CENSUS_FIELDS) <= set(counts)
    assert counts["source_entries"] == (
        counts["upstream_rejected_entries"]
        + counts["active8_accepted_entries"]
        + counts["active8_excluded_entries"]
    )
    assert dict(index.active8_counts()) == completion["active8_counts"]


# ---- Streaming and transitions -----------------------------------------------


def test_every_resolved_row_exposes_the_whole_trace_key_and_its_category(
    resolved,
) -> None:
    """Without this a consumer would rebuild the key from somewhere else."""

    _chain, _plan, _completion, _run_root, index = resolved
    rows = list(index.iter_resolved_traces())
    assert len(rows) == index.counts()["source_entries"]
    for row in rows:
        assert set(row) == set(RESOLVED_TRACE_FIELDS)
        assert set(TRACE_KEY_FIELDS) <= set(row)
        assert row["rejection_category"] in (None, UPSTREAM_REJECTED, ACTIVE8_EXCLUDED)
    # Deterministic order: a second stream is the same sequence.
    assert [dict(row) for row in index.iter_resolved_traces()] == [
        dict(row) for row in rows
    ]


def test_an_accepted_trace_yields_exact_states_and_action_identities(resolved) -> None:
    """Without this the transitions could be right in shape and wrong in content.

    The action identity is recomputed from the cached ActionV4 record and the
    states are compared by their persistent-slot hashes, which is what makes a
    slot-addressed state that state and not a canonicalized redrawing of it.
    """

    _chain, _plan, _completion, _run_root, index = resolved
    accepted = [row for row in index.iter_resolved_traces() if row["rejection_category"] is None]
    assert accepted
    seen = 0
    for row in accepted[:4]:
        key = tuple(row[field] for field in TRACE_KEY_FIELDS)
        transitions = list(index.accepted_transitions_for(key))
        assert len(transitions) == row["accepted_transition_count"] == row["path_length"]
        for step_index, transition in enumerate(transitions):
            assert set(transition) == set(ACCEPTED_TRANSITION_FIELDS)
            assert transition["step_index"] == step_index
            assert transition["source_progress_index"] == step_index
            assert transition["successor_progress_index"] == step_index + 1
            assert transition["successor_is_terminal"] is (
                step_index + 1 == row["path_length"]
            )
            assert tuple(transition[field] for field in TRACE_KEY_FIELDS) == key
            # The ActionV4 record decodes back to the exact executor action.
            from compose_v4.rewrite import action_codec_v4

            rule, _action = action_codec_v4.decode_action(transition["action_record"])
            assert rule == transition["executor_rule"]
            assert canonical_sha256(transition["action_record"]) == (
                transition["action_sha256"]
            )
            # The exact slot-addressed states, not a reconstruction.
            from compose_v4.chem.persistent_state_identity import (
                persistent_slot_state_sha256,
            )

            assert persistent_slot_state_sha256(transition["source_state"]) == (
                transition["source_state_sha256"]
            )
            assert persistent_slot_state_sha256(transition["successor_state"]) == (
                transition["successor_state_sha256"]
            )
            assert transition["successor_alias_count"] >= 1
            assert transition["raw_mark_count"] >= transition["matching_mark_count"] >= 1
            seen += 1
    assert seen >= 4


def test_an_excluded_or_upstream_rejected_trace_has_no_transitions(resolved) -> None:
    """Absence of transitions is the answer, not an error."""

    _chain, _plan, _completion, _run_root, index = resolved
    for category in (UPSTREAM_REJECTED, ACTIVE8_EXCLUDED):
        rows = [row for row in index.iter_resolved_traces() if row["rejection_category"] == category]
        assert rows, category
        key = tuple(rows[0][field] for field in TRACE_KEY_FIELDS)
        assert list(index.accepted_transitions_for(key)) == []
        assert rows[0]["accepted_transition_count"] == 0


def test_the_bulk_stream_decodes_each_cached_chunk_exactly_once(
    resolved, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without this a full pass would reinstate the triangular rescan.

    The point lookup costs one chunk decode, which is correct for a point
    lookup and quadratic in a loop over the corpus.  The bulk stream must decode
    each chunk exactly once, and the counter below wraps -- never replaces --
    the production reader, so what is measured is the real call graph.
    """

    import compose_v4.data.editing_v2_process_v2_active8_decision_index as module

    real = module.read_process_v2_chunk_target
    calls: list[str] = []

    def counting(source_output, **kwargs):
        calls.append(str(kwargs["target"].chunk_filename))
        return real(source_output, **kwargs)

    monkeypatch.setattr(module, "read_process_v2_chunk_target", counting)

    _chain, plan, _completion, _run_root, index = resolved
    transitions = list(index.iter_accepted_transitions())
    accepted = [row for row in index.iter_resolved_traces() if row["rejection_category"] is None]
    assert len(transitions) == sum(row["path_length"] for row in accepted)
    # One decode per task that holds at least one accepted trace, and no more.
    tasks_with_accepted = {row["task_identity_sha256"] for row in accepted}
    assert len(calls) == len(tasks_with_accepted)
    assert len(calls) == len(set(calls)) or len(set(calls)) <= len(plan["tasks"])

    # And the bulk stream agrees with the point lookup, transition for
    # transition, so the fast path is not a second implementation.
    def comparable(transition: Mapping[str, Any]) -> dict[str, Any]:
        # The exact states are compared by their persistent-slot hashes, which
        # is what identifies a slot-addressed state; the arrays themselves have
        # no total equality.
        return {
            key: value
            for key, value in transition.items()
            if key not in {"source_state", "successor_state"}
        }

    by_key: dict[tuple, list[dict[str, Any]]] = {}
    for transition in transitions:
        key = tuple(transition[field] for field in TRACE_KEY_FIELDS)
        by_key.setdefault(key, []).append(comparable(transition))
    sample = sorted(by_key)[0]
    monkeypatch.setattr(module, "read_process_v2_chunk_target", real)
    assert [
        comparable(t) for t in index.accepted_transitions_for(sample)
    ] == by_key[sample]


def test_a_bare_trace_id_is_not_an_address(resolved) -> None:
    """The seam's correction, enforced on the production index.

    A trace id is unique within a V1 task and nothing guarantees it across
    tasks.  The index therefore refuses anything that is not the whole
    ``(v1_task_identity_sha256, entry_index, trace_id)`` tuple, so a bare-id
    lookup cannot be written by accident and cannot silently merge two traces.
    """

    _chain, _plan, _completion, _run_root, index = resolved
    row = next(iter(index.iter_resolved_traces()))
    for bad in (
        row["trace_id"],
        (row["trace_id"],),
        (row["v1_task_identity_sha256"], row["entry_index"]),
        (row["entry_index"], row["v1_task_identity_sha256"], row["trace_id"]),
    ):
        with pytest.raises(ProcessV2Active8DecisionIndexError):
            list(index.accepted_transitions_for(bad))


def test_one_trace_id_in_two_v1_tasks_resolves_to_two_different_traces(
    resolved,
) -> None:
    """The collision case, on the production index rather than a stand-in.

    The fixture's ids are content-addressed and therefore globally distinct, so
    the collision is CONSTRUCTED here: two real rows from two different V1 tasks
    are looked up under keys that differ only in the task identity.  If the
    index keyed by the bare id, one of the two lookups would return the other's
    transitions and nothing would look wrong.
    """

    _chain, _plan, _completion, _run_root, index = resolved
    accepted = [row for row in index.iter_resolved_traces() if row["rejection_category"] is None]
    by_task: dict[str, dict[str, Any]] = {}
    for row in accepted:
        by_task.setdefault(row["v1_task_identity_sha256"], row)
    assert len(by_task) >= 2
    first, second = (by_task[key] for key in sorted(by_task)[:2])
    assert first["v1_task_identity_sha256"] != second["v1_task_identity_sha256"]

    left = list(
        index.accepted_transitions_for(
            tuple(first[field] for field in TRACE_KEY_FIELDS)
        )
    )
    right = list(
        index.accepted_transitions_for(
            tuple(second[field] for field in TRACE_KEY_FIELDS)
        )
    )
    assert left and right
    assert {t["v1_task_identity_sha256"] for t in left} == {
        first["v1_task_identity_sha256"]
    }
    assert {t["v1_task_identity_sha256"] for t in right} == {
        second["v1_task_identity_sha256"]
    }
    # Swapping only the task identity addresses a trace that does not exist,
    # which is what a bare-id lookup would silently answer.
    with pytest.raises(ProcessV2Active8DecisionIndexError, match="absent"):
        list(
            index.accepted_transitions_for(
                (
                    second["v1_task_identity_sha256"],
                    first["entry_index"],
                    first["trace_id"],
                )
            )
        )


def test_validate_accepted_transition_refuses_a_tampered_transition(resolved) -> None:
    """Gate 0 revalidates rather than trusting what it was handed."""

    _chain, _plan, _completion, _run_root, index = resolved
    row = next(row for row in index.iter_resolved_traces() if row["rejection_category"] is None)
    key = tuple(row[field] for field in TRACE_KEY_FIELDS)
    transition = dict(next(iter(index.accepted_transitions_for(key))))
    index.validate_accepted_transition(transition)

    for field, value in (
        ("successor_alias_count", transition["successor_alias_count"] + 1),
        ("action_sha256", "0" * 64),
        ("source_state_sha256", "1" * 64),
        ("index_identity_sha256", "2" * 64),
        ("step_index", 99),
    ):
        with pytest.raises(ProcessV2Active8DecisionIndexError):
            index.validate_accepted_transition({**transition, field: value})
    with pytest.raises(ProcessV2Active8DecisionIndexError, match="declared fields"):
        index.validate_accepted_transition(
            {k: v for k, v in transition.items() if k != "step_index"}
        )
    # A transition whose exact successor state is swapped for its own source
    # state is refused, which is the check that makes the state field load
    # bearing rather than decorative.
    with pytest.raises(ProcessV2Active8DecisionIndexError, match="persistent-slot"):
        index.validate_accepted_transition(
            {**transition, "successor_state": transition["source_state"]}
        )


def test_a_plan_path_outside_the_run_root_is_refused(resolved) -> None:
    """Without this an index could be resolved against a plan nobody published."""

    chain, plan, _completion, run_root, _index = resolved
    elsewhere = chain.artifact_root / "elsewhere.json"
    elsewhere.write_bytes(canonical_bytes(plan) + b"\n")
    with pytest.raises(ProcessV2Active8DecisionIndexError, match="not the one this run"):
        resolve_process_v2_active8_decision_index(
            plan_path=elsewhere,
            completion_path=run_root / COMPLETION_FILENAME,
            artifact_root=chain.artifact_root,
            repo_root=ROOT,
        )


def test_a_resealed_completion_that_moves_a_count_is_refused(resolved) -> None:
    """Without this a correctly resealed census could replace the measured one."""

    chain, _plan, completion, run_root, _index = resolved
    body = {k: v for k, v in completion.items() if k != "completion_sha256"}
    body["active8_counts"] = {
        **body["active8_counts"],
        "active8_accepted_entries": int(body["active8_counts"]["active8_accepted_entries"])
        + 1,
    }
    target = run_root / COMPLETION_FILENAME
    original = target.read_bytes()
    target.write_bytes(
        canonical_bytes({**body, "completion_sha256": canonical_sha256(body)}) + b"\n"
    )
    try:
        with pytest.raises(
            ProcessV2Active8DecisionIndexError, match="differs from the published completion"
        ):
            resolve_process_v2_active8_decision_index(
                plan_path=run_root / PLAN_FILENAME,
                completion_path=target,
                artifact_root=chain.artifact_root,
                repo_root=ROOT,
            )
    finally:
        target.write_bytes(original)


def test_the_index_is_json_serializable_as_provenance(resolved) -> None:
    """A consumer records the identity; it has to be writable."""

    _chain, _plan, _completion, _run_root, index = resolved
    encoded = json.loads(canonical_bytes(dict(index.identity())).decode())
    assert encoded["schema"].endswith("active8_decision_index")
    assert encoded["status"].endswith("NO_DOWNSTREAM_AUTHORITY")
