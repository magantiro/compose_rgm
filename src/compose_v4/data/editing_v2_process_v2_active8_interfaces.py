"""The seam between the Process-V2 source, its Active8 decisions, and Gate 0.

WHY THIS EXISTS, AND WHY IT IS THIN
-----------------------------------
Three concerns are built against each other here: a cache-fed source iterator, an
Active8 decision engine that consumes it, and a Gate-0 runner that consumes the
resulting index.  Written naively that is a serial chain, and each stage can only
be built once the one before it exists.

This module is the contract that breaks the chain, and it is deliberately ONLY a
contract: a protocol, the reconciliation identity, and the reason-code
categories.  It holds no machinery, owns no artifact, and computes nothing.  A
second generic contract framework is exactly what this wave was told not to
build, so anything with behaviour belongs in the owning module, not here.

WHAT IS FROZEN
--------------
* :class:`ProcessV2Active8Index` -- what Gate 0 may assume about a decision
  index.  It extends the structural minimum with the three streaming operations
  Gate 0 actually needs, so Gate 0 never reaches for a concrete class.
* :data:`ACTIVE8_CENSUS_IDENTITY` -- the reconciliation every stage must satisfy.
* The two rejection CATEGORIES, which are distinct and must never be merged:
  a trace the rebind already refused was never a candidate, and a trace Active8
  excluded was evaluated and failed.  Collapsing them makes an unevaluated trace
  indistinguishable from a rejected one, which is the difference between "we did
  not look" and "we looked and said no".

THE JOIN KEY IS PART OF THE CONTRACT
------------------------------------
A cache row is joined to a rebind decision by ``(v1_task_identity_sha256,
entry_index)`` -- the V1 task identity plus the GLOBAL entry index -- and never
by chunk position.  Chunk position is an artifact of the chunk size, which is a
partitioning choice; two caches of the same corpus at different chunk sizes hold
the same rows at different positions.  A positional join would silently pair a
decision with the wrong trace whenever the chunk size moved.

Addressing ONE trace needs one field more: :data:`TRACE_KEY_FIELDS` adds
``trace_id``, because a trace id is unique within a V1 task and nothing
guarantees it across tasks.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, Protocol, runtime_checkable

from compose_v4.data.editing_v2_process_v2_schema import StructuralDecisionIndex

# ---- The census identity ----

#: Every Active8 stage reconciles exactly this way.  `source_entries` is the
#: whole corpus the cache published; the other three partition it.
ACTIVE8_CENSUS_IDENTITY = (
    "source_entries == upstream_rejected_entries + active8_accepted_entries "
    "+ active8_excluded_entries"
)

ACTIVE8_CENSUS_FIELDS: tuple[str, ...] = (
    "source_entries",
    "upstream_rejected_entries",
    "active8_accepted_entries",
    "active8_excluded_entries",
)

# ---- Rejection categories, deliberately distinct ----

#: The rebind already refused this trace; Active8 never evaluated it.
UPSTREAM_REJECTED = "upstream_rebind_rejected"

#: Active8 evaluated this trace and excluded it.
ACTIVE8_EXCLUDED = "active8_candidate_excluded"

REJECTION_CATEGORIES: tuple[str, ...] = (UPSTREAM_REJECTED, ACTIVE8_EXCLUDED)

# ---- The join key ----

#: The two fields that pair a cache row with its rebind decision.  Ordered, so a
#: consumer building a key tuple cannot transpose them.
JOIN_KEY_FIELDS: tuple[str, ...] = ("v1_task_identity_sha256", "entry_index")

#: The authoritative address of ONE resolved trace: the join key plus the trace
#: id.  A bare ``trace_id`` is NOT an address -- it is unique within a V1 task
#: and nothing guarantees it across tasks, so two tasks may legitimately carry
#: the same id for different traces.  Keying transitions by the bare id would
#: silently merge them, and the merge would look like a trace that simply had
#: more transitions than it does.
TRACE_KEY_FIELDS: tuple[str, ...] = (
    "v1_task_identity_sha256",
    "entry_index",
    "trace_id",
)

#: ``(v1_task_identity_sha256, entry_index, trace_id)``, in that order.
ProcessV2TraceKey = tuple[str, int, str]

#: The minimum every row from :meth:`ProcessV2Active8Index.iter_resolved_traces`
#: carries.  A LOWER BOUND: an index may publish richer provenance, and a
#: consumer must not require that it does not.
#:
#: Freezing the row, and not only the key, is a correction.  The first version of
#: this seam froze the key and the rejection categories and left the row itself
#: unstated, so the producer named these two fields ``split`` and ``category``
#: while the consumer required ``partition_role`` and ``rejection_category``.
#: Both suites passed; the chain did not join.  A seam that names a concept but
#: not the field carrying it has not named it.
#:
#: The spellings are the policy layer's, deliberately.  ``split`` is the data
#: path's name for the same value, correct in the chunk cache and the rebind
#: where a shard genuinely is a split; this row is a policy artifact that a gate
#: reads to decide which traces are eligible, and ``partition_role`` is what that
#: vocabulary is called (`REQUIRED_PARTITION_ROLES`).  ``rejection_category``
#: matches :data:`REJECTION_CATEGORIES` rather than shortening to ``category``,
#: which says nothing on its own.
RESOLVED_TRACE_ROW_FIELDS: tuple[str, ...] = (
    *TRACE_KEY_FIELDS,
    "partition_role",
    "rejection_category",
    "decision_sha256",
    "accepted_transition_count",
)


# ---- What Gate 0 may assume ----


@runtime_checkable
class ProcessV2Active8Index(StructuralDecisionIndex, Protocol):
    """The decision index Gate 0 consumes.

    Extends the structural minimum rather than replacing it, so an index that
    satisfies this also satisfies every consumer written against the smaller
    protocol.  Expressed as a Protocol because the Process-V2 index must have no
    V1 class in its MRO: structural satisfaction is the whole point, and
    inheriting a V1 base would drag in the live-V1 revalidation that refuses the
    historical payload by construction.
    """

    def iter_resolved_traces(self) -> Iterator[Mapping[str, Any]]:
        """Every resolved trace, in a deterministic order.

        Every row exposes all of :data:`TRACE_KEY_FIELDS`, so a consumer can
        address a trace without reconstructing its key from somewhere else.

        Includes upstream-rejected traces, which carry their category and are
        present for the census but must never be candidate-evaluated.
        """

    def accepted_transitions_for(
        self, trace_key: ProcessV2TraceKey
    ) -> Iterator[Mapping[str, Any]]:
        """The accepted transitions of one trace, in trace order.

        Addressed by the full :data:`ProcessV2TraceKey`, never by a bare
        ``trace_id``: the id is unique within a V1 task and nothing guarantees
        it across tasks, so a bare-id lookup can merge two different traces into
        one answer that looks merely longer than it should be.

        Empty for a trace that was excluded or upstream-rejected -- absence of
        transitions is not an error, it is the answer.
        """

    def validate_accepted_transition(self, transition: Mapping[str, Any]) -> None:
        """Raise if this transition is not exactly what the index published.

        Gate 0 re-validates rather than trusting what it was handed, because an
        index reopened from disk and an index held in memory must agree.
        """


__all__ = [
    "ACTIVE8_CENSUS_FIELDS",
    "ACTIVE8_CENSUS_IDENTITY",
    "ACTIVE8_EXCLUDED",
    "JOIN_KEY_FIELDS",
    "ProcessV2Active8Index",
    "ProcessV2TraceKey",
    "REJECTION_CATEGORIES",
    "RESOLVED_TRACE_ROW_FIELDS",
    "TRACE_KEY_FIELDS",
    "UPSTREAM_REJECTED",
]
