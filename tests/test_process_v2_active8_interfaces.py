"""The seam must be self-consistent, and its trace address must be collision-safe.

The first version of this module froze the join key as `(v1_task_identity_sha256,
entry_index)` and then, forty lines later, addressed transitions by a bare
`trace_id`. Both statements cannot be true: if the id alone identified a trace,
the join key would not need the task identity.

That is the defect these tests exist for. A trace id is unique WITHIN a V1 task
and nothing guarantees it across tasks, so a bare-id lookup merges two different
traces into one answer -- and the merged answer does not look wrong, it looks
like a trace that happened to have more transitions than it does.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import pytest

from compose_v4.data.editing_v2_process_v2_active8_interfaces import (
    ACTIVE8_CENSUS_FIELDS,
    ACTIVE8_EXCLUDED,
    JOIN_KEY_FIELDS,
    REJECTION_CATEGORIES,
    TRACE_KEY_FIELDS,
    UPSTREAM_REJECTED,
    ProcessV2Active8Index,
    ProcessV2TraceKey,
)
from compose_v4.data.editing_v2_process_v2_schema import StructuralDecisionIndex

_TASK_A = "a" * 64
_TASK_B = "b" * 64
#: The same id in two different V1 tasks. Legal, and the whole point.
_SHARED_ID = "trace-0001"


class _KeyedIndex:
    """A minimal index that keys by the full trace key, inheriting nothing.

    Structural satisfaction is the requirement -- the production index must have
    no V1 class in its MRO -- so this deliberately subclasses neither protocol.
    """

    def __init__(self, transitions: Mapping[ProcessV2TraceKey, tuple[str, ...]]) -> None:
        self._transitions = dict(transitions)

    process_identity_sha256 = "c" * 64
    completion_sha256 = "d" * 64

    def counts(self) -> Mapping[str, int]:
        return dict.fromkeys(ACTIVE8_CENSUS_FIELDS, 0)

    def rejected_traces_by_code(self) -> Mapping[str, int]:
        return {}

    def identity(self) -> Mapping[str, Any]:
        return {}

    def iter_resolved_traces(self) -> Iterator[Mapping[str, Any]]:
        for task, entry, trace_id in sorted(self._transitions):
            yield {
                "v1_task_identity_sha256": task,
                "entry_index": entry,
                "trace_id": trace_id,
            }

    def accepted_transitions_for(
        self, trace_key: ProcessV2TraceKey
    ) -> Iterator[Mapping[str, Any]]:
        for name in self._transitions.get(trace_key, ()):
            yield {"action": name}

    def validate_accepted_transition(self, transition: Mapping[str, Any]) -> None:
        if "action" not in transition:
            raise ValueError("not a transition this index published")


@pytest.fixture(name="index")
def _index() -> _KeyedIndex:
    return _KeyedIndex(
        {
            (_TASK_A, 7, _SHARED_ID): ("insert",),
            (_TASK_B, 7, _SHARED_ID): ("delete", "restate"),
        }
    )


def test_one_trace_id_in_two_v1_tasks_cannot_collide(index: _KeyedIndex) -> None:
    """The correction Codex required, stated as the failure it prevents."""

    from_a = list(index.accepted_transitions_for((_TASK_A, 7, _SHARED_ID)))
    from_b = list(index.accepted_transitions_for((_TASK_B, 7, _SHARED_ID)))

    assert [t["action"] for t in from_a] == ["insert"]
    assert [t["action"] for t in from_b] == ["delete", "restate"]
    # The give-away a bare-id lookup would produce: not an error, just a trace
    # that appears to have more transitions than it has.
    assert from_a != from_b
    assert len(from_a) + len(from_b) == 3


def test_the_same_entry_index_in_two_tasks_is_two_traces(index: _KeyedIndex) -> None:
    """`entry_index` is global within a task, not across the corpus."""

    rows = list(index.iter_resolved_traces())
    assert len(rows) == 2
    assert {row["entry_index"] for row in rows} == {7}
    assert {row["v1_task_identity_sha256"] for row in rows} == {_TASK_A, _TASK_B}


def test_every_resolved_row_exposes_the_whole_trace_key(index: _KeyedIndex) -> None:
    """A consumer must never have to reconstruct the key from somewhere else."""

    for row in index.iter_resolved_traces():
        assert set(TRACE_KEY_FIELDS) <= set(row)
        key = tuple(row[field] for field in TRACE_KEY_FIELDS)
        assert list(index.accepted_transitions_for(key))


def test_the_trace_key_extends_the_join_key_rather_than_replacing_it() -> None:
    """The two keys must agree, which is exactly what the first version broke."""

    assert TRACE_KEY_FIELDS[: len(JOIN_KEY_FIELDS)] == JOIN_KEY_FIELDS
    assert TRACE_KEY_FIELDS[len(JOIN_KEY_FIELDS) :] == ("trace_id",)


def test_the_index_satisfies_both_protocols_without_inheriting_either(
    index: _KeyedIndex,
) -> None:
    assert isinstance(index, StructuralDecisionIndex)
    assert isinstance(index, ProcessV2Active8Index)
    assert [cls.__name__ for cls in type(index).__mro__] == ["_KeyedIndex", "object"]


def test_an_index_missing_the_keyed_lookup_does_not_satisfy_the_protocol() -> None:
    """Negative control: the protocol has to be able to refuse something."""

    class Partial:
        process_identity_sha256 = "e" * 64
        completion_sha256 = "f" * 64

        def counts(self) -> Mapping[str, int]:
            return {}

        def rejected_traces_by_code(self) -> Mapping[str, int]:
            return {}

        def identity(self) -> Mapping[str, Any]:
            return {}

        def iter_resolved_traces(self) -> Iterator[Mapping[str, Any]]:
            return iter(())

    partial = Partial()
    assert isinstance(partial, StructuralDecisionIndex)
    assert not isinstance(partial, ProcessV2Active8Index)


def test_the_two_rejection_categories_stay_distinct() -> None:
    assert UPSTREAM_REJECTED != ACTIVE8_EXCLUDED
    assert set(REJECTION_CATEGORIES) == {UPSTREAM_REJECTED, ACTIVE8_EXCLUDED}
