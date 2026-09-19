"""FiberControl pool policy with an additive protonation-aware proposal expert.

The historical three-expert policy remains immutable in
``t4_integrated_route_fiber``.  This module changes only the declared expert
vocabulary and corresponding provenance features for the prospective
Editing-V3 rescue.  Candidate generation, endpoint gates, acquisition, reward
learning and recursive archive semantics are otherwise identical.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from compose_v4.control.fiber_control import (
    ProgramValue,
    SearchState,
    acquisition,
    program_features,
)

SCHEMA_VERSION = "t4_protonation_integrated_route_fiber_policy_v1"
PROTONATION_EXPERT = "protonation_aware_retained_subgraph"
EXPERTS = (
    "shallow",
    "anchored_replacement",
    "route_complete_region",
    PROTONATION_EXPERT,
)


def _experts(record: dict) -> tuple[str, ...]:
    values = record.get("proposal_experts") or [record.get("proposal_lane")]
    result = tuple(sorted({str(value) for value in values if value}))
    unknown = set(result).difference(EXPERTS)
    if unknown:
        raise ValueError(f"unknown proposal experts: {sorted(unknown)}")
    return result


def merge_expert_pools(pools: dict[str, Iterable[dict]]) -> list[dict]:
    """Canonical endpoint union with deterministic complete provenance."""

    merged: dict[str, dict] = {}
    origins: dict[str, list[dict]] = {}
    expert_sets: dict[str, set[str]] = {}
    family_sets: dict[str, set[str]] = {}
    for expert in EXPERTS:
        for position, source in enumerate(pools.get(expert, ())):
            row = dict(source)
            if row.get("proposal_lane") not in (None, expert):
                raise ValueError(
                    f"pool {expert!r} contains lane {row.get('proposal_lane')!r}"
                )
            row["proposal_lane"] = expert
            row["proposal_experts"] = sorted(set(_experts(row)) | {expert})
            smiles = str(row.get("smiles") or "")
            if not smiles:
                raise ValueError(f"proposal from {expert!r} has no canonical smiles")
            origin = {
                "expert": expert,
                "position": position,
                "parent": row.get("parent"),
                "parent_score": row.get("parent_score"),
                "program_families": sorted(set(row.get("program_families") or [])),
                "route_program_id": row.get("route_program_id"),
            }
            origins.setdefault(smiles, []).append(origin)
            expert_sets.setdefault(smiles, set()).update(row["proposal_experts"])
            family_sets.setdefault(smiles, set()).update(row.get("families") or [])
            incumbent = merged.get(smiles)
            key = (
                float(row.get("parent_score", float("inf"))),
                EXPERTS.index(expert),
                position,
            )
            incumbent_key = None
            if incumbent is not None:
                incumbent_key = (
                    float(incumbent.get("parent_score", float("inf"))),
                    EXPERTS.index(str(incumbent["proposal_lane"])),
                    int(incumbent["_pool_position"]),
                )
            if incumbent is None or key < incumbent_key:
                merged[smiles] = {**row, "_pool_position": position}
    result = []
    for smiles in sorted(merged):
        row = dict(merged[smiles])
        row.pop("_pool_position", None)
        row["proposal_experts"] = sorted(expert_sets[smiles])
        row["families"] = sorted(family_sets[smiles])
        row["proposal_origins"] = sorted(
            origins[smiles],
            key=lambda value: (
                value["expert"],
                str(value.get("parent") or ""),
                int(value["position"]),
            ),
        )
        result.append(row)
    return result


def integrated_features(record: dict, state: SearchState) -> np.ndarray:
    """Existing program features plus four proposal-expert indicators."""

    base = program_features(record, state)
    experts = set(_experts(record))
    return np.concatenate(
        [
            base[:-1],
            np.asarray([float(name in experts) for name in EXPERTS], dtype=float),
            np.asarray([float(len(experts) > 1), 1.0], dtype=float),
        ]
    )


def attach_features(records: Iterable[dict], state: SearchState, fiber) -> list[dict]:
    """Attach current-state features and endpoint fingerprints."""

    from rdkit import Chem

    prepared = []
    for source in records:
        row = dict(source)
        molecule = Chem.MolFromSmiles(row["smiles"])
        if molecule is None:
            continue
        row["features"] = integrated_features(row, state)
        row["fingerprint"] = set(fiber.generator.GetFingerprint(molecule).GetOnBits())
        prepared.append(row)
    return prepared


def select_batch(
    candidates: list[dict],
    value: ProgramValue,
    state: SearchState,
    rng,
    *,
    round_index: int,
    batch: int,
    exploration: int,
    expert_floor_rounds: int,
    expert_floor_counts: Mapping[str, int] | None = None,
    route_scale_floor_rounds: int = 0,
    route_scale_floor_counts: Mapping[str, int] | None = None,
) -> list[dict]:
    """Use the frozen selector with a floor for every available expert."""

    if (
        batch < 1
        or exploration < 0
        or exploration > batch
        or route_scale_floor_rounds < 0
    ):
        raise ValueError("invalid batch or exploration quota")
    route_bands = ("small", "medium", "large")
    expert_counts = {expert: int(expert_floor_counts is None) for expert in EXPERTS}
    if expert_floor_counts is not None:
        unknown = set(expert_floor_counts).difference(EXPERTS)
        if unknown:
            raise ValueError(f"unknown expert floor names: {sorted(unknown)}")
        for expert, count in expert_floor_counts.items():
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(
                    f"invalid expert floor count for {expert!r}: {count!r}"
                )
            expert_counts[expert] = count
    scale_counts = {band: int(route_scale_floor_counts is None) for band in route_bands}
    if route_scale_floor_counts is not None:
        unknown = set(route_scale_floor_counts).difference(route_bands)
        if unknown:
            raise ValueError(f"unknown route scale floor bands: {sorted(unknown)}")
        for band, count in route_scale_floor_counts.items():
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(
                    f"invalid route scale floor count for {band!r}: {count!r}"
                )
            scale_counts[band] = count
    available = [row for row in candidates if row["smiles"] not in state.archive]
    selected: list[dict] = []
    used: set[str] = set()
    if round_index <= route_scale_floor_rounds:
        for band in route_bands:
            choices = sorted(
                (
                    row
                    for row in available
                    if "route_complete_region" in _experts(row)
                    and row.get("realized_primitive_band") == band
                    and row["smiles"] not in used
                ),
                key=lambda row: (
                    int(row.get("route_proposal_rank", 1 << 30)),
                    row["smiles"],
                ),
            )
            for chosen in choices[: scale_counts[band]]:
                if len(selected) >= batch:
                    break
                selected.append({**chosen, "selection_kind": "route_scale_floor"})
                used.add(chosen["smiles"])
    if round_index <= expert_floor_rounds:
        for expert in EXPERTS:
            already_selected = sum(expert in _experts(row) for row in selected)
            needed = max(0, expert_counts[expert] - already_selected)
            choices = [
                row
                for row in available
                if expert in _experts(row) and row["smiles"] not in used
            ]
            if not choices or len(selected) >= batch or needed == 0:
                continue
            for index in rng.permutation(len(choices))[:needed]:
                if len(selected) >= batch:
                    break
                chosen = choices[int(index)]
                selected.append({**chosen, "selection_kind": "expert_floor"})
                used.add(chosen["smiles"])

    remaining = [row for row in available if row["smiles"] not in used]
    room = min(batch - len(selected), len(remaining))
    if room:
        random_quota = min(exploration, room)
        indices = acquisition(
            remaining,
            value,
            state,
            rng,
            batch=room,
            exploration=random_quota,
        )
        model_count = room - random_quota if value.weights is not None else 0
        for position, index in enumerate(indices):
            kind = "model" if position < model_count else "exploration"
            selected.append({**remaining[index], "selection_kind": kind})
    return selected


def expert_census(records: Iterable[dict]) -> dict[str, int]:
    counts = {expert: 0 for expert in EXPERTS}
    counts["multi_expert"] = 0
    counts["unique_endpoints"] = 0
    for record in records:
        experts = _experts(record)
        for expert in experts:
            counts[expert] += 1
        counts["multi_expert"] += int(len(experts) > 1)
        counts["unique_endpoints"] += 1
    return counts


__all__ = [
    "EXPERTS",
    "PROTONATION_EXPERT",
    "SCHEMA_VERSION",
    "attach_features",
    "expert_census",
    "integrated_features",
    "merge_expert_pools",
    "select_batch",
]
