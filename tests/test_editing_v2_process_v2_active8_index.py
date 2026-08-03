"""The Process-V2 Active8 decision index, and what it must refuse.

Acceptance test 10 lives here: the index satisfies the frozen structural
protocol, has no V1 class anywhere in its MRO, and refuses an injected V1 plan,
V1 process identity, V1 policy and V1 model/process contract.

The chain is the real one built by
``test_editing_v2_process_v2_active8_decisions``; rebuilding it here would create
a second builder whose drift from the first is invisible.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import pytest

from compose_v4.data.editing_v2_process_v2_active8_decision_index import (
    ACCEPTED_TRANSITION_FIELDS,
    DECISION_ELIGIBLE_PARTITION_ROLES,
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


def _accepted(index, *, decision_eligible: bool = True) -> list[Mapping[str, Any]]:
    """Accepted resolved rows, by default only the ones this index will decode.

    The index decodes molecular states for ``DECISION_ELIGIBLE_PARTITION_ROLES``
    only, so a test that wants transitions has to ask for a trace in one of them;
    asking for a held-out trace is itself a refusal, and is asserted separately.
    """

    return [
        row
        for row in index.iter_resolved_traces()
        if row["rejection_category"] is None
        and (
            (row["partition_role"] in DECISION_ELIGIBLE_PARTITION_ROLES)
            is decision_eligible
        )
    ]


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
    accepted = _accepted(index)
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

    opened_roles: list[str] = []

    def counting(source_output, **kwargs):
        calls.append(str(kwargs["target"].chunk_filename))
        opened_roles.append(str(kwargs["target"].split))
        return real(source_output, **kwargs)

    monkeypatch.setattr(module, "read_process_v2_chunk_target", counting)

    _chain, plan, _completion, _run_root, index = resolved
    transitions = list(index.iter_accepted_transitions())
    accepted = _accepted(index)
    held_out = _accepted(index, decision_eligible=False)
    assert accepted and held_out, "both sides of the role filter must be populated"
    assert len(transitions) == sum(row["path_length"] for row in accepted)
    # One decode per DECISION-ELIGIBLE task that holds an accepted trace, and no
    # more.  A held-out task's chunk is never opened at all, which is what makes
    # the seal a property of what was never read.
    tasks_with_accepted = {row["task_identity_sha256"] for row in accepted}
    assert len(calls) == len(tasks_with_accepted)
    assert len(calls) == len(set(calls)) or len(set(calls)) <= len(plan["tasks"])
    # No held-out chunk was opened AT ALL, which is what makes the seal a
    # property of what was never read rather than of what was never counted.
    assert set(opened_roles) <= set(DECISION_ELIGIBLE_PARTITION_ROLES)
    assert {row["partition_role"] for row in accepted} <= set(
        DECISION_ELIGIBLE_PARTITION_ROLES
    )
    assert {row["partition_role"] for row in held_out} - set(
        DECISION_ELIGIBLE_PARTITION_ROLES
    )

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
    accepted = _accepted(index)
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
    row = _accepted(index)[0]
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


# ---- Resealed mutations: the evidence has to be refused on its content --------
#
# A self-hash proves an artifact was not edited after it was sealed.  It proves
# nothing about whether the artifact was TRUE when it was sealed, and content
# hashes are not signatures, so an unresealed mutation being refused is not
# evidence of anything: it only shows the hash moved.  Every mutation below is
# RESEALED -- the decision row's self-hash, the receipt's two decision-byte
# hashes and its own self-hash, and the completion's inventory hash and self-hash
# are all recomputed, so the artifact is internally consistent by every hash it
# carries -- and each must STILL be refused, on what it says rather than on what
# it hashes to.


def _reseal_run(
    chain,
    run_root: Path,
    plan: Mapping[str, Any],
    completion: Mapping[str, Any],
    *,
    task_identity: str,
    rows: list[dict[str, Any]] | None = None,
    count_delta: Mapping[str, int] | None = None,
    inventory_edit=None,
) -> None:
    """Republish one task and the completion with every self-hash recomputed.

    Deliberately a FULL reseal.  Rows get a fresh ``decision_sha256``, the
    receipt gets fresh ``decision_file_sha256``/``decision_stream_sha256`` over
    the rebuilt deterministic gzip and a fresh ``receipt_sha256``, and the
    completion gets a fresh ``result_inventory_sha256`` and ``completion_sha256``
    -- so nothing below is caught by a hash that failed to move.
    """

    from compose_v4.data.editing_v2_process_v2_active8_mapreduce import (
        DECISION_FILENAME,
        RECEIPT_FILENAME,
        _deterministic_gzip,
    )

    task = next(
        item for item in plan["tasks"] if item["task_identity_sha256"] == task_identity
    )
    output = chain_fixture.mounted(chain, task["output_artifact_path"])
    receipt = json.loads((output / RECEIPT_FILENAME).read_bytes())
    if rows is not None:
        sealed = []
        for row in rows:
            body = {k: v for k, v in row.items() if k != "decision_sha256"}
            sealed.append({**body, "decision_sha256": canonical_sha256(body)})
        raw = b"".join(canonical_bytes(row) + b"\n" for row in sealed)
        payload = _deterministic_gzip(raw)
        (output / DECISION_FILENAME).write_bytes(payload)
        receipt = {
            **receipt,
            "decision_file_sha256": hashlib.sha256(payload).hexdigest(),
            "decision_stream_sha256": hashlib.sha256(raw).hexdigest(),
        }
    if count_delta:
        receipt = {
            **receipt,
            "counts": {
                field: int(value) + int(count_delta.get(field, 0))
                for field, value in receipt["counts"].items()
            },
        }
    body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    receipt = {**body, "receipt_sha256": canonical_sha256(body)}
    (output / RECEIPT_FILENAME).write_bytes(canonical_bytes(receipt) + b"\n")

    inventory = []
    for item in completion["result_inventory"]:
        if item["task_identity_sha256"] != task_identity:
            inventory.append(dict(item))
            continue
        updated = {
            **item,
            "receipt_sha256": receipt["receipt_sha256"],
            "decision_file_sha256": receipt["decision_file_sha256"],
            "counts": dict(receipt["counts"]),
        }
        inventory.append(inventory_edit(updated) if inventory_edit else updated)
    counts = {
        field: int(value) + int((count_delta or {}).get(field, 0))
        for field, value in completion["active8_counts"].items()
    }
    body = {
        **{k: v for k, v in completion.items() if k != "completion_sha256"},
        "result_inventory": inventory,
        "result_inventory_sha256": canonical_sha256(inventory),
        "active8_counts": counts,
    }
    sealed_completion = {**body, "completion_sha256": canonical_sha256(body)}
    (run_root / COMPLETION_FILENAME).write_bytes(
        canonical_bytes(sealed_completion) + b"\n"
    )


def _snapshot(chain, run_root: Path, plan: Mapping[str, Any]) -> dict[Path, bytes]:
    from compose_v4.data.editing_v2_process_v2_active8_mapreduce import (
        DECISION_FILENAME,
        RECEIPT_FILENAME,
    )

    files = {run_root / COMPLETION_FILENAME: (run_root / COMPLETION_FILENAME).read_bytes()}
    for task in plan["tasks"]:
        output = chain_fixture.mounted(chain, task["output_artifact_path"])
        for name in (DECISION_FILENAME, RECEIPT_FILENAME):
            files[output / name] = (output / name).read_bytes()
    return files


def _restore(files: Mapping[Path, bytes]) -> None:
    for path, payload in files.items():
        path.write_bytes(payload)


def _decision_rows(chain, task: Mapping[str, Any]) -> list[dict[str, Any]]:
    from compose_v4.data.editing_v2_process_v2_active8_mapreduce import DECISION_FILENAME

    output = chain_fixture.mounted(chain, task["output_artifact_path"])
    raw = gzip.decompress((output / DECISION_FILENAME).read_bytes())
    return [json.loads(line) for line in raw.splitlines()]


def _accepted_target(chain, plan: Mapping[str, Any]):
    """A real accepted, decision-eligible row and the task that published it."""

    for task in plan["tasks"]:
        if str(task["split"]) not in DECISION_ELIGIBLE_PARTITION_ROLES:
            continue
        rows = _decision_rows(chain, task)
        for index, row in enumerate(rows):
            if row["category"] is None and row["actions"]:
                return task, rows, index
    raise AssertionError("the fixture published no accepted decision-eligible trace")


def _resolve(chain, run_root: Path):
    return resolve_process_v2_active8_decision_index(
        plan_path=run_root / PLAN_FILENAME,
        completion_path=run_root / COMPLETION_FILENAME,
        artifact_root=chain.artifact_root,
        repo_root=ROOT,
    )


def _drain(index) -> None:
    """Force every accepted transition, which is where per-step checks live."""

    for _row, transitions in index.iter_accepted_trace_transitions():
        assert transitions


#: A dynamic reason whose declared arithmetic an ACCEPTED row's own counts
#: already satisfy (it needs a positive exact-mark and alias count), so a
#: mutation using it is refused for the contradiction with acceptance rather
#: than for a count that gives it away.
_CONSISTENT_REASON = "teacher_successor_is_virtual_self_transition"


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (
            {"supported": False, "exclusion_reason": _CONSISTENT_REASON},
            "cannot carry unsupported evidence",
        ),
        (
            {"exclusion_reason": _CONSISTENT_REASON},
            "supported candidate evidence is incomplete",
        ),
    ],
)
def test_a_resealed_accepted_row_whose_evidence_refuses_its_action(
    resolved, mutation: Mapping[str, Any], expected: str
) -> None:
    """Acceptance and support are the same claim; a row cannot make both.

    Nothing reconstructs ``ProcessV2CandidateEvidence`` when a row is read, so
    its ``__post_init__`` guard never runs over a published artifact and both of
    these were previously inert.  The first case is a self-consistent piece of
    UNSUPPORTED evidence sitting on an ACCEPTED trace; the second is supported
    evidence that also names a reason to exclude, which is the same contradiction
    written the other way round.
    """

    chain, plan, completion, run_root, _index = resolved
    task, rows, index_of = _accepted_target(chain, plan)
    saved = _snapshot(chain, run_root, plan)
    try:
        mutated = [dict(row) for row in rows]
        actions = [dict(action) for action in mutated[index_of]["actions"]]
        evidence = {**actions[0]["candidate_evidence"], **mutation}
        actions[0] = {**actions[0], "candidate_evidence": evidence}
        mutated[index_of] = {**mutated[index_of], "actions": actions}
        _reseal_run(
            chain,
            run_root,
            plan,
            completion,
            task_identity=str(task["task_identity_sha256"]),
            rows=mutated,
        )
        with pytest.raises(Exception, match=expected):
            _drain(_resolve(chain, run_root))
    finally:
        _restore(saved)


def test_a_resealed_canonical_successor_key_is_refused(resolved) -> None:
    """The key is a function of the successor state, so it is recomputed.

    The successor state is in hand -- it came out of the cache chunk -- so the
    production canonicalizer answers this without enumerating anything, which is
    why the read path can still refuse it with no model.
    """

    chain, plan, completion, run_root, _index = resolved
    task, rows, index_of = _accepted_target(chain, plan)
    saved = _snapshot(chain, run_root, plan)
    try:
        mutated = [dict(row) for row in rows]
        actions = [dict(action) for action in mutated[index_of]["actions"]]
        evidence = dict(actions[0]["candidate_evidence"])
        evidence["canonical_successor_key"] = "TAMPERED-NOT-A-MOLECULE"
        actions[0] = {**actions[0], "candidate_evidence": evidence}
        mutated[index_of] = {**mutated[index_of], "actions": actions}
        _reseal_run(
            chain,
            run_root,
            plan,
            completion,
            task_identity=str(task["task_identity_sha256"]),
            rows=mutated,
        )
        with pytest.raises(
            ProcessV2Active8DecisionIndexError, match="does not canonicalize"
        ):
            _drain(_resolve(chain, run_root))
    finally:
        _restore(saved)


@pytest.mark.parametrize(
    "count_field",
    [
        "raw_mark_count",
        "canonical_successor_count",
        "matching_mark_count",
        "exact_successor_mark_count",
        "successor_alias_count",
    ],
)
def test_a_resealed_candidate_count_is_refused(resolved, count_field: str) -> None:
    """Every one of the five, moved in the evidence and correctly resealed.

    The refusal is the row's own aggregate: ``candidate_totals`` is recomputed
    from the per-action evidence it summarises rather than read, so evidence and
    total can no longer disagree quietly.  An aggregate that agrees with itself
    proves nothing.
    """

    chain, plan, completion, run_root, _index = resolved
    task, rows, index_of = _accepted_target(chain, plan)
    saved = _snapshot(chain, run_root, plan)
    try:
        mutated = [dict(row) for row in rows]
        actions = [dict(action) for action in mutated[index_of]["actions"]]
        evidence = dict(actions[0]["candidate_evidence"])
        evidence[count_field] = int(evidence[count_field]) + 1000
        actions[0] = {**actions[0], "candidate_evidence": evidence}
        mutated[index_of] = {**mutated[index_of], "actions": actions}
        _reseal_run(
            chain,
            run_root,
            plan,
            completion,
            task_identity=str(task["task_identity_sha256"]),
            rows=mutated,
        )
        with pytest.raises(Exception, match="derived census disagrees|arithmetic"):
            _drain(_resolve(chain, run_root))
    finally:
        _restore(saved)


def test_a_resealed_row_aggregate_is_refused(resolved) -> None:
    """The other direction: the stored total moved and the evidence did not."""

    chain, plan, completion, run_root, _index = resolved
    task, rows, index_of = _accepted_target(chain, plan)
    saved = _snapshot(chain, run_root, plan)
    try:
        mutated = [dict(row) for row in rows]
        totals = dict(mutated[index_of]["candidate_totals"])
        totals["raw_candidate_marks"] = int(totals["raw_candidate_marks"]) + 1000
        mutated[index_of] = {**mutated[index_of], "candidate_totals": totals}
        _reseal_run(
            chain,
            run_root,
            plan,
            completion,
            task_identity=str(task["task_identity_sha256"]),
            rows=mutated,
            count_delta={"raw_candidate_marks": 1000},
        )
        with pytest.raises(Exception, match="derived census disagrees"):
            _drain(_resolve(chain, run_root))
    finally:
        _restore(saved)


def test_a_resealed_stale_receipt_identity_in_the_inventory_is_refused(
    resolved,
) -> None:
    """The completion names a receipt; following the pointer is the check.

    A stale identity is one that was VALID for some execution of this task, so it
    is not distinguishable from a fresh one by shape -- only by comparing it with
    the receipt the task actually holds now.
    """

    chain, plan, completion, run_root, _index = resolved
    task, _rows, _index_of = _accepted_target(chain, plan)
    saved = _snapshot(chain, run_root, plan)
    try:
        other = next(
            item
            for item in completion["result_inventory"]
            if item["task_identity_sha256"] != task["task_identity_sha256"]
        )
        _reseal_run(
            chain,
            run_root,
            plan,
            completion,
            task_identity=str(task["task_identity_sha256"]),
            inventory_edit=lambda row: {
                **row,
                "receipt_sha256": other["receipt_sha256"],
            },
        )
        with pytest.raises(
            ProcessV2Active8DecisionIndexError, match="does not match the receipts"
        ):
            _resolve(chain, run_root)
    finally:
        _restore(saved)


def test_a_resealed_stale_decision_file_identity_in_the_inventory_is_refused(
    resolved,
) -> None:
    """Same for the decision bytes the completion binds each task to."""

    chain, plan, completion, run_root, _index = resolved
    task, _rows, _index_of = _accepted_target(chain, plan)
    saved = _snapshot(chain, run_root, plan)
    try:
        other = next(
            item
            for item in completion["result_inventory"]
            if item["task_identity_sha256"] != task["task_identity_sha256"]
        )
        _reseal_run(
            chain,
            run_root,
            plan,
            completion,
            task_identity=str(task["task_identity_sha256"]),
            inventory_edit=lambda row: {
                **row,
                "decision_file_sha256": other["decision_file_sha256"],
            },
        )
        with pytest.raises(
            ProcessV2Active8DecisionIndexError, match="does not match the receipts"
        ):
            _resolve(chain, run_root)
    finally:
        _restore(saved)


def test_a_fully_propagated_candidate_count_is_the_stated_residual(resolved) -> None:
    """The one vector the read path CANNOT refuse, pinned rather than hidden.

    A per-action count moved inside its arithmetic bounds and propagated through
    the row total, the receipt and the completion is internally consistent
    everywhere, and distinguishing it from the truth requires re-enumerating the
    fiber -- which the read path is forbidden to do, precisely so Gate 0 can fan
    out with no model.  This test asserts the residual EXISTS so that a future
    change which closes it fails here and gets noticed, and so that the boundary
    is a recorded property rather than an assumption.
    """

    chain, plan, completion, run_root, _index = resolved
    task, rows, index_of = _accepted_target(chain, plan)
    saved = _snapshot(chain, run_root, plan)
    try:
        mutated = [dict(row) for row in rows]
        actions = [dict(action) for action in mutated[index_of]["actions"]]
        evidence = dict(actions[0]["candidate_evidence"])
        moved = int(evidence["raw_mark_count"]) + 1000
        evidence["raw_mark_count"] = moved
        actions[0] = {**actions[0], "candidate_evidence": evidence}
        totals = dict(mutated[index_of]["candidate_totals"])
        totals["raw_candidate_marks"] = int(totals["raw_candidate_marks"]) + 1000
        mutated[index_of] = {
            **mutated[index_of],
            "actions": actions,
            "candidate_totals": totals,
        }
        _reseal_run(
            chain,
            run_root,
            plan,
            completion,
            task_identity=str(task["task_identity_sha256"]),
            rows=mutated,
            count_delta={"raw_candidate_marks": 1000},
        )
        index = _resolve(chain, run_root)
        _drain(index)
        assert index.counts()["raw_candidate_marks"] == (
            int(completion["active8_counts"]["raw_candidate_marks"]) + 1000
        )
        observed = {
            transition["raw_mark_count"]
            for transition in index.iter_accepted_transitions()
        }
        assert moved in observed, (
            "the residual is that a consistently propagated count survives; if this "
            "fails the read path has gained a way to refuse it and the note above "
            "must be rewritten"
        )
    finally:
        _restore(saved)
