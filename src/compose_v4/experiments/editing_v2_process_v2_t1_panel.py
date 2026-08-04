"""Deterministic Process-V2 T1 candidate selection.

The release sentinel already freezes a bounded, per-cell sample of accepted
train teachers.  T1 reuses that sample instead of inventing another corpus
scan or ranking law.  This module performs the two cheap operations that must
precede molecular successor compilation:

* verify that each sampled exact source has only one canonical teacher target
  in the complete accepted-train stream; and
* select one unique source per objective while covering every required cell
  and the frozen per-family cardinality bounds.

It does not enumerate marks, compile successor fibers, construct a model, or
grant downstream authority.  Those operations remain later T1 stages.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from compose_v4.data.editing_v2_process_v2_schema import canonical_sha256


class ProcessV2T1PanelError(RuntimeError):
    """The sentinel sample cannot form the frozen unique-state T1 panel."""


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProcessV2T1PanelError(f"{field} must be nonempty text")
    return value


def _integer(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProcessV2T1PanelError(f"{field} must be a nonnegative integer")
    return value


def _sha256(value: object, *, field: str) -> str:
    text = _text(value, field=field)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ProcessV2T1PanelError(f"{field} must be a lowercase SHA-256")
    return text


@dataclass(frozen=True, slots=True)
class ProcessV2T1Candidate:
    """One sentinel-ranked molecular objective before successor compilation."""

    source_state_sha256: str
    canonical_successor_key: str
    action_sha256: str
    rank_sha256: str
    model_family: str
    capability_cell_id: str
    family_context: str
    task_identity_sha256: str
    v1_task_identity_sha256: str
    entry_index: int
    step_index: int
    sentinel_pair_sha256: str
    source_canonical_key: str
    target_state_sha256: str
    raw_mark_count: int
    candidate_sha256: str = ""

    def __post_init__(self) -> None:
        for field in (
            "source_state_sha256",
            "action_sha256",
            "rank_sha256",
            "task_identity_sha256",
            "v1_task_identity_sha256",
            "sentinel_pair_sha256",
            "target_state_sha256",
        ):
            _sha256(getattr(self, field), field=field)
        for field in (
            "canonical_successor_key",
            "model_family",
            "capability_cell_id",
            "family_context",
            "source_canonical_key",
        ):
            _text(getattr(self, field), field=field)
        _integer(self.entry_index, field="entry_index")
        _integer(self.step_index, field="step_index")
        if _integer(self.raw_mark_count, field="raw_mark_count") < 1:
            raise ProcessV2T1PanelError("raw_mark_count must be positive")
        if self.source_canonical_key == self.canonical_successor_key:
            raise ProcessV2T1PanelError("a T1 candidate cannot be a self transition")
        expected = canonical_sha256(self.identity_body())
        if self.candidate_sha256:
            if self.candidate_sha256 != expected:
                raise ProcessV2T1PanelError("T1 candidate self-hash disagrees")
        else:
            object.__setattr__(self, "candidate_sha256", expected)

    def identity_body(self) -> dict[str, object]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
            if field != "candidate_sha256"
        }


@dataclass(frozen=True, slots=True)
class ProcessV2T1TargetCensus:
    """Complete accepted-stream target census restricted to sampled sources."""

    accepted_transition_count: int
    sampled_source_transition_count: int
    accepted_transition_stream_sha256: str
    targets_by_source: tuple[tuple[str, tuple[str, ...]], ...]
    census_sha256: str = ""

    def __post_init__(self) -> None:
        _integer(self.accepted_transition_count, field="accepted_transition_count")
        _integer(
            self.sampled_source_transition_count,
            field="sampled_source_transition_count",
        )
        if self.sampled_source_transition_count > self.accepted_transition_count:
            raise ProcessV2T1PanelError("sampled-source census exceeds the accepted stream")
        _sha256(
            self.accepted_transition_stream_sha256,
            field="accepted_transition_stream_sha256",
        )
        if tuple(sorted(self.targets_by_source)) != self.targets_by_source:
            raise ProcessV2T1PanelError("target census sources are not canonical")
        if len(dict(self.targets_by_source)) != len(self.targets_by_source):
            raise ProcessV2T1PanelError("target census repeats a source")
        if any(
            not targets or tuple(sorted(set(targets))) != targets
            for _, targets in self.targets_by_source
        ):
            raise ProcessV2T1PanelError("target census targets are empty or noncanonical")
        expected = canonical_sha256(self.identity_body())
        if self.census_sha256:
            if self.census_sha256 != expected:
                raise ProcessV2T1PanelError("target census self-hash disagrees")
        else:
            object.__setattr__(self, "census_sha256", expected)

    def identity_body(self) -> dict[str, object]:
        return {
            "accepted_transition_count": self.accepted_transition_count,
            "sampled_source_transition_count": self.sampled_source_transition_count,
            "accepted_transition_stream_sha256": self.accepted_transition_stream_sha256,
            "targets_by_source": [
                [source, list(targets)] for source, targets in self.targets_by_source
            ],
        }

    @property
    def target_sets(self) -> dict[str, tuple[str, ...]]:
        return dict(self.targets_by_source)


@dataclass(frozen=True, slots=True)
class ProcessV2T1CandidateSelection:
    """Nonauthorizing bounded selection ready for exact-state reopening."""

    selected: tuple[ProcessV2T1Candidate, ...]
    counts_by_family: tuple[tuple[str, int], ...]
    counts_by_cell: tuple[tuple[str, int], ...]
    target_census_sha256: str
    selection_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.selected:
            raise ProcessV2T1PanelError("T1 candidate selection is empty")
        if tuple(
            sorted(
                self.selected,
                key=lambda item: (
                    item.model_family,
                    item.capability_cell_id,
                    item.rank_sha256,
                    item.source_state_sha256,
                    item.action_sha256,
                ),
            )
        ) != self.selected:
            raise ProcessV2T1PanelError("T1 candidate selection is not canonical")
        sources = [item.source_state_sha256 for item in self.selected]
        if len(set(sources)) != len(sources):
            raise ProcessV2T1PanelError("T1 candidate selection repeats an exact source")
        expected_family = tuple(sorted(Counter(item.model_family for item in self.selected).items()))
        expected_cell = tuple(
            sorted(Counter(item.capability_cell_id for item in self.selected).items())
        )
        if self.counts_by_family != expected_family or self.counts_by_cell != expected_cell:
            raise ProcessV2T1PanelError("T1 selection counts disagree")
        _sha256(self.target_census_sha256, field="target_census_sha256")
        expected = canonical_sha256(self.identity_body())
        if self.selection_sha256:
            if self.selection_sha256 != expected:
                raise ProcessV2T1PanelError("T1 selection self-hash disagrees")
        else:
            object.__setattr__(self, "selection_sha256", expected)

    def identity_body(self) -> dict[str, object]:
        return {
            "selected_candidate_sha256s": [item.candidate_sha256 for item in self.selected],
            "counts_by_family": dict(self.counts_by_family),
            "counts_by_cell": dict(self.counts_by_cell),
            "target_census_sha256": self.target_census_sha256,
        }


def candidates_from_sentinel_plan(
    sentinel_plan: Mapping[str, Any],
    *,
    required_cell_ids: Sequence[str],
    active_families: Sequence[str],
) -> tuple[ProcessV2T1Candidate, ...]:
    """Project the already-validated sentinel plan into unique objectives."""

    required = frozenset(required_cell_ids)
    families = frozenset(active_families)
    if not required or not families:
        raise ProcessV2T1PanelError("T1 requires nonempty cells and families")
    candidates: dict[tuple[str, str, str], ProcessV2T1Candidate] = {}
    partitions = sentinel_plan.get("partitions")
    if not isinstance(partitions, list):
        raise ProcessV2T1PanelError("sentinel plan has no partition inventory")
    for partition in partitions:
        if not isinstance(partition, Mapping) or not isinstance(partition.get("pairs"), list):
            raise ProcessV2T1PanelError("sentinel partition is malformed")
        for pair in partition["pairs"]:
            if not isinstance(pair, Mapping) or not isinstance(pair.get("occurrences"), list):
                raise ProcessV2T1PanelError("sentinel pair is malformed")
            family = _text(pair.get("source_model_family"), field="source_model_family")
            if family not in families:
                continue
            for occurrence in pair["occurrences"]:
                if not isinstance(occurrence, Mapping):
                    raise ProcessV2T1PanelError("sentinel occurrence is malformed")
                cell = occurrence.get("capability_cell_id")
                if cell not in required:
                    continue
                evidence = occurrence.get("candidate_evidence")
                if not isinstance(evidence, Mapping):
                    raise ProcessV2T1PanelError("sentinel occurrence lacks candidate evidence")
                candidate = ProcessV2T1Candidate(
                    source_state_sha256=_text(
                        evidence.get("source_state_sha256"), field="source_state_sha256"
                    ),
                    canonical_successor_key=_text(
                        evidence.get("canonical_successor_key"),
                        field="canonical_successor_key",
                    ),
                    action_sha256=_text(evidence.get("action_sha256"), field="action_sha256"),
                    rank_sha256=_text(pair.get("rank_sha256"), field="rank_sha256"),
                    model_family=family,
                    capability_cell_id=_text(cell, field="capability_cell_id"),
                    family_context=_text(
                        occurrence.get("family_context"), field="family_context"
                    ),
                    task_identity_sha256=_text(
                        occurrence.get("task_identity_sha256"),
                        field="task_identity_sha256",
                    ),
                    v1_task_identity_sha256=_text(
                        occurrence.get("v1_task_identity_sha256"),
                        field="v1_task_identity_sha256",
                    ),
                    entry_index=_integer(occurrence.get("entry_index"), field="entry_index"),
                    step_index=_integer(occurrence.get("step_index"), field="step_index"),
                    sentinel_pair_sha256=_sha256(
                        pair.get("pair_sha256"), field="pair_sha256"
                    ),
                    source_canonical_key=_text(
                        evidence.get("source_canonical_key"), field="source_canonical_key"
                    ),
                    target_state_sha256=_text(
                        evidence.get("target_state_sha256"), field="target_state_sha256"
                    ),
                    raw_mark_count=_integer(
                        evidence.get("raw_mark_count"), field="raw_mark_count"
                    ),
                )
                key = (
                    candidate.source_state_sha256,
                    candidate.canonical_successor_key,
                    candidate.capability_cell_id,
                )
                prior = candidates.get(key)
                if prior is None or (
                    candidate.rank_sha256,
                    candidate.action_sha256,
                    candidate.sentinel_pair_sha256,
                    candidate.v1_task_identity_sha256,
                    candidate.entry_index,
                    candidate.step_index,
                ) < (
                    prior.rank_sha256,
                    prior.action_sha256,
                    prior.sentinel_pair_sha256,
                    prior.v1_task_identity_sha256,
                    prior.entry_index,
                    prior.step_index,
                ):
                    candidates[key] = candidate
    return tuple(
        sorted(
            candidates.values(),
            key=lambda item: (
                item.model_family,
                item.capability_cell_id,
                item.rank_sha256,
                item.source_state_sha256,
                item.action_sha256,
            ),
        )
    )


def census_candidate_targets(
    transitions: Iterable[Mapping[str, Any]],
    *,
    candidate_source_state_sha256s: Iterable[str],
) -> ProcessV2T1TargetCensus:
    """Scan the complete accepted stream while retaining only sampled sources."""

    wanted = frozenset(candidate_source_state_sha256s)
    if not wanted:
        raise ProcessV2T1PanelError("target census has no sampled sources")
    targets: dict[str, set[str]] = defaultdict(set)
    stream = hashlib.sha256()
    total = 0
    sampled = 0
    for transition in transitions:
        assignment = _text(
            transition.get("assignment_sha256"), field="assignment_sha256"
        )
        stream.update(assignment.encode("ascii"))
        stream.update(b"\n")
        total += 1
        evidence = transition.get("candidate_evidence")
        if not isinstance(evidence, Mapping):
            raise ProcessV2T1PanelError("accepted transition lacks candidate evidence")
        source = _text(evidence.get("source_state_sha256"), field="source_state_sha256")
        if source not in wanted:
            continue
        targets[source].add(
            _text(
                evidence.get("canonical_successor_key"),
                field="canonical_successor_key",
            )
        )
        sampled += 1
    missing = sorted(wanted - set(targets))
    if missing:
        raise ProcessV2T1PanelError(
            f"{len(missing)} sentinel-sampled sources are absent from the accepted stream"
        )
    return ProcessV2T1TargetCensus(
        accepted_transition_count=total,
        sampled_source_transition_count=sampled,
        accepted_transition_stream_sha256=stream.hexdigest(),
        targets_by_source=tuple(
            (source, tuple(sorted(values))) for source, values in sorted(targets.items())
        ),
    )


def _required_cell_matching(
    candidates_by_cell: Mapping[str, tuple[ProcessV2T1Candidate, ...]],
) -> dict[str, ProcessV2T1Candidate]:
    """Deterministic augmenting-path matching of required cells to unique sources."""

    assigned_source: dict[str, str] = {}
    assigned_cell: dict[str, ProcessV2T1Candidate] = {}

    def augment(cell: str, seen: set[str]) -> bool:
        for candidate in candidates_by_cell[cell]:
            source = candidate.source_state_sha256
            if source in seen:
                continue
            seen.add(source)
            previous = assigned_source.get(source)
            if previous is None or augment(previous, seen):
                assigned_source[source] = cell
                assigned_cell[cell] = candidate
                return True
        return False

    for cell in sorted(candidates_by_cell, key=lambda item: (len(candidates_by_cell[item]), item)):
        if not augment(cell, set()):
            raise ProcessV2T1PanelError(
                f"required cell {cell!r} cannot receive a unique exact source"
            )
    return assigned_cell


def select_process_v2_t1_candidates(
    candidates: Sequence[ProcessV2T1Candidate],
    *,
    target_census: ProcessV2T1TargetCensus,
    required_cell_ids: Sequence[str],
    minimum_entries_by_family: Mapping[str, int],
    maximum_entries_by_family: Mapping[str, int],
) -> ProcessV2T1CandidateSelection:
    """Select the frozen bounded panel without repeating an exact source."""

    required = tuple(sorted(set(required_cell_ids)))
    minimums = dict(minimum_entries_by_family)
    maximums = dict(maximum_entries_by_family)
    if not required or set(minimums) != set(maximums):
        raise ProcessV2T1PanelError("T1 family bounds or required cells are incomplete")
    if any(
        not isinstance(minimums[family], int)
        or not isinstance(maximums[family], int)
        or minimums[family] <= 0
        or minimums[family] > maximums[family]
        for family in minimums
    ):
        raise ProcessV2T1PanelError("T1 family bounds are invalid")

    target_sets = target_census.target_sets
    eligible = [
        candidate
        for candidate in candidates
        if candidate.capability_cell_id in required
        and candidate.model_family in minimums
        and target_sets.get(candidate.source_state_sha256)
        == (candidate.canonical_successor_key,)
    ]
    by_cell: dict[str, tuple[ProcessV2T1Candidate, ...]] = {}
    for cell in required:
        rows = tuple(
            sorted(
                (candidate for candidate in eligible if candidate.capability_cell_id == cell),
                key=lambda item: (
                    item.rank_sha256,
                    item.source_state_sha256,
                    item.action_sha256,
                    item.sentinel_pair_sha256,
                ),
            )
        )
        if not rows:
            raise ProcessV2T1PanelError(f"required cell {cell!r} has no single-target source")
        by_cell[cell] = rows

    mandatory = _required_cell_matching(by_cell)
    selected_by_source = {
        candidate.source_state_sha256: candidate for candidate in mandatory.values()
    }
    cells_by_family: dict[str, tuple[str, ...]] = {
        family: tuple(
            cell
            for cell in required
            if any(candidate.model_family == family for candidate in by_cell[cell])
        )
        for family in sorted(minimums)
    }
    for family, cells in cells_by_family.items():
        if not cells:
            raise ProcessV2T1PanelError(f"required family {family!r} has no required cell")
        family_count = sum(
            candidate.model_family == family for candidate in selected_by_source.values()
        )
        if family_count > maximums[family]:
            raise ProcessV2T1PanelError(
                f"required-cell coverage needs {family_count} {family!r} sources, above "
                f"the frozen maximum {maximums[family]}"
            )
        positions = {cell: 0 for cell in cells}
        while family_count < maximums[family]:
            added = False
            for cell in cells:
                rows = by_cell[cell]
                while positions[cell] < len(rows):
                    candidate = rows[positions[cell]]
                    positions[cell] += 1
                    if candidate.source_state_sha256 not in selected_by_source:
                        selected_by_source[candidate.source_state_sha256] = candidate
                        family_count += 1
                        added = True
                        break
                if family_count == maximums[family]:
                    break
            if not added:
                break
        if family_count < minimums[family]:
            raise ProcessV2T1PanelError(
                f"family {family!r} has {family_count} unique sources, below "
                f"the frozen minimum {minimums[family]}"
            )

    selected = tuple(
        sorted(
            selected_by_source.values(),
            key=lambda item: (
                item.model_family,
                item.capability_cell_id,
                item.rank_sha256,
                item.source_state_sha256,
                item.action_sha256,
            ),
        )
    )
    selected_cells = {item.capability_cell_id for item in selected}
    if selected_cells != set(required):
        raise ProcessV2T1PanelError("selected panel does not cover every required cell")
    return ProcessV2T1CandidateSelection(
        selected=selected,
        counts_by_family=tuple(sorted(Counter(item.model_family for item in selected).items())),
        counts_by_cell=tuple(
            sorted(Counter(item.capability_cell_id for item in selected).items())
        ),
        target_census_sha256=target_census.census_sha256,
    )


__all__ = [
    "ProcessV2T1Candidate",
    "ProcessV2T1CandidateSelection",
    "ProcessV2T1PanelError",
    "ProcessV2T1TargetCensus",
    "candidates_from_sentinel_plan",
    "census_candidate_targets",
    "select_process_v2_t1_candidates",
]
