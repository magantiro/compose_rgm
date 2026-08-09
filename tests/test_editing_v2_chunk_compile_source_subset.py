"""A source subset must change WHICH rows compile, never HOW they compile.

The subset exists to avoid materializing 28.6x the wanted chemistry when a
selection is scattered (the V2 recipe's 4,921 ring_system_restate rows sit
~195-per-task across all 27 synthetic train tasks, inside 140,743 records).

That saving is only safe if a subset-compiled row is byte-identical to the same
row compiled whole. If it were not, the corpus would silently contain two
flavours of the same transition depending on how it happened to be compiled --
and the receipts would look fine.

These tests drive the filtering and identity logic directly against a fake
row stream, so they run without the artifact tree or a scorer.
"""

from __future__ import annotations

import pytest

from compose_v4.data.editing_v2_process_v2_schema import canonical_sha256
from compose_v4.experiments.editing_v2_process_v2_chunk_compile import (
    ProcessV2ChunkCompileError,
)


def _row(source_key: str, successor: str = "CCO") -> dict:
    return {
        "candidate_evidence": {
            "source_canonical_key": source_key,
            "canonical_successor_key": successor,
            "exclusion_reason": None,
        }
    }


def _apply_subset(rows, subset):
    """Mirror of the filter in compile_process_v2_chunk_shard."""

    if subset is None:
        return list(rows), None
    wanted = frozenset(str(k) for k in subset)
    if not wanted:
        raise ProcessV2ChunkCompileError("source subset must name at least one source")
    digest = canonical_sha256(sorted(wanted))
    kept = [
        r for r in rows if str(r["candidate_evidence"]["source_canonical_key"]) in wanted
    ]
    if not kept:
        raise ProcessV2ChunkCompileError("source subset selects no transitions")
    return kept, digest


ROWS = [_row(f"S{i}") for i in range(10)]


def test_subset_preserves_order_and_membership() -> None:
    """Filtering must not reorder: offsets address the selected stream."""

    kept, _ = _apply_subset(ROWS, {"S7", "S1", "S4"})
    assert [r["candidate_evidence"]["source_canonical_key"] for r in kept] == [
        "S1",
        "S4",
        "S7",
    ]


def test_subset_rows_are_the_same_objects_as_the_whole_stream() -> None:
    """The subset selects rows; it must not transform them.

    Identity of the row objects is the strongest available statement that a
    subset-compiled entry sees exactly the input a whole-task compile would.
    """

    whole, _ = _apply_subset(ROWS, None)
    kept, _ = _apply_subset(ROWS, {"S3"})
    assert kept[0] is whole[3]


def test_digest_is_order_independent_but_content_sensitive() -> None:
    _, a = _apply_subset(ROWS, ["S1", "S2"])
    _, b = _apply_subset(ROWS, ["S2", "S1"])
    _, c = _apply_subset(ROWS, ["S1", "S3"])
    assert a == b, "the same set in a different order is the same subset"
    assert a != c, "a different set must be a different subset"


def test_whole_task_compile_has_no_subset_digest() -> None:
    """Absent means 'compiled whole' and must stay distinguishable from a subset."""

    _, digest = _apply_subset(ROWS, None)
    assert digest is None


def test_empty_subset_is_refused() -> None:
    with pytest.raises(ProcessV2ChunkCompileError, match="at least one source"):
        _apply_subset(ROWS, [])


def test_subset_naming_nothing_present_is_refused_not_silently_empty() -> None:
    """A typo'd subset must fail loudly rather than publish an empty slice."""

    with pytest.raises(ProcessV2ChunkCompileError, match="selects no transitions"):
        _apply_subset(ROWS, {"NOT_A_SOURCE"})


def test_offsets_address_the_selected_stream() -> None:
    """offset/limit apply AFTER filtering, so no slice lands on unwanted rows.

    Filtering after slicing would make most containers compile ranges holding
    nothing the recipe wants -- which is the entire cost problem this solves.
    """

    kept, _ = _apply_subset(ROWS, {"S2", "S5", "S8"})
    window = kept[1:3]
    assert [r["candidate_evidence"]["source_canonical_key"] for r in window] == [
        "S5",
        "S8",
    ]
