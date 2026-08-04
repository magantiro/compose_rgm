"""Focused tests for Process-V2 T1 unique-state candidate selection."""

from __future__ import annotations

from typing import Any

import pytest

from compose_v4.data.editing_v2_process_v2_schema import canonical_sha256
from compose_v4.experiments.editing_v2_process_v2_t1_panel import (
    ProcessV2T1PanelError,
    candidates_from_sentinel_plan,
    census_candidate_targets,
    select_process_v2_t1_candidates,
)


def _sha(character: str) -> str:
    return character * 64


def _occurrence(*, source: str, target: str, cell: str) -> dict[str, Any]:
    context = cell.rsplit(":", 1)[-1]
    return {
        "task_identity_sha256": _sha("1"),
        "v1_task_identity_sha256": _sha("2"),
        "entry_index": int(source[0], 16),
        "step_index": 0,
        "candidate_evidence": {
            "supported": True,
            "exclusion_reason": None,
            "action_sha256": _sha("3"),
            "source_state_sha256": source,
            "target_state_sha256": _sha("4"),
            "source_canonical_key": f"source-{source}",
            "canonical_successor_key": target,
            "raw_mark_count": 3,
            "matching_mark_count": 1,
            "exact_successor_mark_count": 1,
        },
        "capability_cell_id": cell,
        "family_context": context,
        "audit_axes": {"context": context},
    }


def _pair(*, source: str, target: str, family: str, cell: str, rank: str) -> dict[str, Any]:
    body = {
        "source_state_sha256": source,
        "action_sha256": _sha("3"),
        "rank_sha256": rank,
        "source_task_identity_sha256": _sha("1"),
        "source_entry_index": int(source[0], 16),
        "source_step_index": 0,
        "source_model_family": family,
        "oracle_required": False,
        "occurrences": [
            _occurrence(
                source=source,
                target=target,
                cell=cell,
            )
        ],
    }
    return {**body, "pair_sha256": canonical_sha256(body)}


def _plan(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    return {"partitions": [{"pairs": pairs}]}


def _transition(*, source: str, target: str, assignment: str) -> dict[str, Any]:
    return {
        "assignment_sha256": assignment,
        "candidate_evidence": {
            "source_state_sha256": source,
            "canonical_successor_key": target,
        },
    }


def test_sentinel_candidates_are_deterministic_and_deduplicate_occurrences() -> None:
    cell = "cells:atom_insert:one_neighbor_birth"
    pair = _pair(
        source=_sha("a"),
        target="CC",
        family="atom_insert",
        cell=cell,
        rank=_sha("5"),
    )
    duplicate = dict(pair)
    duplicate["occurrences"] = [dict(pair["occurrences"][0])]

    first = candidates_from_sentinel_plan(
        _plan([pair, duplicate]),
        required_cell_ids=[cell],
        active_families=["atom_insert"],
    )
    second = candidates_from_sentinel_plan(
        _plan([duplicate, pair]),
        required_cell_ids=[cell],
        active_families=["atom_insert"],
    )

    assert first == second
    assert len(first) == 1
    assert first[0].source_state_sha256 == _sha("a")


def test_target_census_detects_a_repeated_target_source() -> None:
    source = _sha("a")
    census = census_candidate_targets(
        (
            _transition(source=source, target="CC", assignment=_sha("1")),
            _transition(source=source, target="CO", assignment=_sha("2")),
            _transition(source=_sha("b"), target="CN", assignment=_sha("3")),
        ),
        candidate_source_state_sha256s=[source],
    )

    assert census.accepted_transition_count == 3
    assert census.sampled_source_transition_count == 2
    assert census.target_sets[source] == ("CC", "CO")


def test_selection_covers_cells_and_family_bounds_without_repeating_sources() -> None:
    cells = (
        "cells:atom_delete:leaf_death",
        "cells:atom_delete:connected_nonleaf_death",
    )
    pairs = [
        _pair(
            source=_sha(character),
            target=f"target-{character}",
            family="atom_delete",
            cell=cells[index % 2],
            rank=_sha(str(index + 1)),
        )
        for index, character in enumerate(("a", "b", "c", "d"))
    ]
    candidates = candidates_from_sentinel_plan(
        _plan(pairs),
        required_cell_ids=cells,
        active_families=["atom_delete"],
    )
    census = census_candidate_targets(
        (
            _transition(
                source=candidate.source_state_sha256,
                target=candidate.canonical_successor_key,
                assignment=_sha(str(index + 1)),
            )
            for index, candidate in enumerate(candidates)
        ),
        candidate_source_state_sha256s=(
            candidate.source_state_sha256 for candidate in candidates
        ),
    )

    selection = select_process_v2_t1_candidates(
        candidates,
        target_census=census,
        required_cell_ids=cells,
        minimum_entries_by_family={"atom_delete": 3},
        maximum_entries_by_family={"atom_delete": 3},
    )

    assert dict(selection.counts_by_family) == {"atom_delete": 3}
    assert set(dict(selection.counts_by_cell)) == set(cells)
    assert len({item.source_state_sha256 for item in selection.selected}) == 3


def test_repeated_target_sources_are_ineligible() -> None:
    cell = "cells:atom_insert:one_neighbor_birth"
    candidate = candidates_from_sentinel_plan(
        _plan(
            [
                _pair(
                    source=_sha("a"),
                    target="CC",
                    family="atom_insert",
                    cell=cell,
                    rank=_sha("1"),
                )
            ]
        ),
        required_cell_ids=[cell],
        active_families=["atom_insert"],
    )
    census = census_candidate_targets(
        (
            _transition(source=_sha("a"), target="CC", assignment=_sha("1")),
            _transition(source=_sha("a"), target="CO", assignment=_sha("2")),
        ),
        candidate_source_state_sha256s=[_sha("a")],
    )

    with pytest.raises(ProcessV2T1PanelError, match="no single-target source"):
        select_process_v2_t1_candidates(
            candidate,
            target_census=census,
            required_cell_ids=[cell],
            minimum_entries_by_family={"atom_insert": 1},
            maximum_entries_by_family={"atom_insert": 1},
        )


def test_required_cell_matching_reroutes_shared_sources() -> None:
    first = "cells:atom_restate:element_identity_change"
    second = "cells:atom_restate:valence_state_change"
    shared = _sha("a")
    alternate = _sha("b")
    pairs = [
        _pair(
            source=shared,
            target="CO",
            family="atom_restate",
            cell=first,
            rank=_sha("1"),
        ),
        _pair(
            source=alternate,
            target="CN",
            family="atom_restate",
            cell=first,
            rank=_sha("2"),
        ),
        _pair(
            source=shared,
            target="CO",
            family="atom_restate",
            cell=second,
            rank=_sha("1"),
        ),
    ]
    candidates = candidates_from_sentinel_plan(
        _plan(pairs),
        required_cell_ids=[first, second],
        active_families=["atom_restate"],
    )
    census = census_candidate_targets(
        (
            _transition(source=shared, target="CO", assignment=_sha("6")),
            _transition(source=alternate, target="CN", assignment=_sha("7")),
        ),
        candidate_source_state_sha256s=[shared, alternate],
    )

    selection = select_process_v2_t1_candidates(
        candidates,
        target_census=census,
        required_cell_ids=[first, second],
        minimum_entries_by_family={"atom_restate": 2},
        maximum_entries_by_family={"atom_restate": 2},
    )

    assert set(dict(selection.counts_by_cell)) == {first, second}
    assert {item.source_state_sha256 for item in selection.selected} == {
        shared,
        alternate,
    }


def test_family_fill_uses_each_cells_own_rank_cursor() -> None:
    first = "cells:atom_restate:element_identity_change"
    second = "cells:atom_restate:valence_state_change"
    pairs = [
        _pair(
            source=_sha(character),
            target=f"target-{character}",
            family="atom_restate",
            cell=cell,
            rank=_sha(rank),
        )
        for character, cell, rank in (
            ("a", first, "1"),
            ("b", first, "2"),
            ("c", first, "3"),
            ("d", second, "1"),
            ("e", second, "2"),
        )
    ]
    candidates = candidates_from_sentinel_plan(
        _plan(pairs),
        required_cell_ids=[first, second],
        active_families=["atom_restate"],
    )
    census = census_candidate_targets(
        (
            _transition(
                source=candidate.source_state_sha256,
                target=candidate.canonical_successor_key,
                assignment=_sha(str(index + 1)),
            )
            for index, candidate in enumerate(candidates)
        ),
        candidate_source_state_sha256s=(
            candidate.source_state_sha256 for candidate in candidates
        ),
    )

    selection = select_process_v2_t1_candidates(
        candidates,
        target_census=census,
        required_cell_ids=[first, second],
        minimum_entries_by_family={"atom_restate": 5},
        maximum_entries_by_family={"atom_restate": 5},
    )

    assert dict(selection.counts_by_family) == {"atom_restate": 5}


def test_required_cell_coverage_cannot_exceed_the_family_maximum() -> None:
    cells = ("cells:atom_restate:first", "cells:atom_restate:second")
    pairs = [
        _pair(
            source=_sha(character),
            target=f"target-{character}",
            family="atom_restate",
            cell=cell,
            rank=_sha("1"),
        )
        for character, cell in zip(("a", "b"), cells, strict=True)
    ]
    candidates = candidates_from_sentinel_plan(
        _plan(pairs),
        required_cell_ids=cells,
        active_families=["atom_restate"],
    )
    census = census_candidate_targets(
        (
            _transition(
                source=candidate.source_state_sha256,
                target=candidate.canonical_successor_key,
                assignment=_sha(str(index + 1)),
            )
            for index, candidate in enumerate(candidates)
        ),
        candidate_source_state_sha256s=(
            candidate.source_state_sha256 for candidate in candidates
        ),
    )

    with pytest.raises(ProcessV2T1PanelError, match="above the frozen maximum"):
        select_process_v2_t1_candidates(
            candidates,
            target_census=census,
            required_cell_ids=cells,
            minimum_entries_by_family={"atom_restate": 1},
            maximum_entries_by_family={"atom_restate": 1},
        )
