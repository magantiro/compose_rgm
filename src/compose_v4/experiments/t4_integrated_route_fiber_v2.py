"""Four-expert FiberControl policy including retained-core feasibility repair."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from compose_v4.control.fiber_control import (
    ProgramValue,
    SearchState,
    acquisition,
    program_features,
)

EXPERTS = (
    "shallow",
    "anchored_replacement",
    "route_complete_region",
    "retained_core_prune",
)


def _experts(record: dict) -> tuple[str, ...]:
    values = record.get("proposal_experts") or [record.get("proposal_lane")]
    result = tuple(sorted({str(value) for value in values if value}))
    unknown = set(result).difference(EXPERTS)
    if unknown:
        raise ValueError(f"unknown proposal experts: {sorted(unknown)}")
    return result


def merge_expert_pools(pools: dict[str, Iterable[dict]]) -> list[dict]:
    merged: dict[str, dict] = {}
    origins: dict[str, list[dict]] = {}
    expert_sets: dict[str, set[str]] = {}
    family_sets: dict[str, set[str]] = {}
    for expert in EXPERTS:
        for position, source in enumerate(pools.get(expert, ())):
            row = dict(source)
            if row.get("proposal_lane") not in (None, expert):
                raise ValueError(f"pool {expert!r} contains lane {row.get('proposal_lane')!r}")
            row["proposal_lane"] = expert
            row["proposal_experts"] = sorted(set(_experts(row)) | {expert})
            smiles = str(row.get("smiles") or "")
            if not smiles:
                raise ValueError(f"proposal from {expert!r} has no canonical smiles")
            origins.setdefault(smiles, []).append(
                {
                    "expert": expert,
                    "position": position,
                    "parent": row.get("parent"),
                    "parent_score": row.get("parent_score"),
                    "program_families": sorted(set(row.get("program_families") or [])),
                    "route_program_id": row.get("route_program_id"),
                }
            )
            expert_sets.setdefault(smiles, set()).update(row["proposal_experts"])
            family_sets.setdefault(smiles, set()).update(row.get("families") or [])
            incumbent = merged.get(smiles)
            key = (float(row.get("parent_score", float("inf"))), EXPERTS.index(expert), position)
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
) -> list[dict]:
    if batch < 1 or exploration < 0 or exploration > batch:
        raise ValueError("invalid batch or exploration quota")
    available = [row for row in candidates if row["smiles"] not in state.archive]
    selected: list[dict] = []
    used: set[str] = set()
    if round_index <= expert_floor_rounds:
        for expert in EXPERTS:
            choices = [
                row
                for row in available
                if expert in _experts(row) and row["smiles"] not in used
            ]
            if not choices or len(selected) >= batch:
                continue
            chosen = choices[int(rng.integers(len(choices)))]
            selected.append({**chosen, "selection_kind": "expert_floor"})
            used.add(chosen["smiles"])
    remaining = [row for row in available if row["smiles"] not in used]
    room = min(batch - len(selected), len(remaining))
    if room:
        random_quota = min(exploration, room)
        indices = acquisition(
            remaining, value, state, rng, batch=room, exploration=random_quota
        )
        model_count = room - random_quota if value.weights is not None else 0
        for position, index in enumerate(indices):
            kind = "model" if position < model_count else "exploration"
            selected.append({**remaining[index], "selection_kind": kind})
    return selected


def expert_census(records: Iterable[dict]) -> dict[str, int]:
    counts = {expert: 0 for expert in EXPERTS}
    counts.update({"multi_expert": 0, "unique_endpoints": 0})
    for record in records:
        experts = _experts(record)
        for expert in experts:
            counts[expert] += 1
        counts["multi_expert"] += int(len(experts) > 1)
        counts["unique_endpoints"] += 1
    return counts


__all__ = [
    "EXPERTS",
    "attach_features",
    "expert_census",
    "integrated_features",
    "merge_expert_pools",
    "select_batch",
]
