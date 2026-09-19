"""Shared FiberControl over shallow, anchored, and route-distilled proposals.

This module contains only deterministic pool reduction, feature construction, and
selection policy.  Proposal generation and docking stay in the remote orchestration
layer.  The learned controller sees proposal provenance and counted outcomes from the
current cell, but no target name, comparator score, teacher endpoint, or route identity.
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

SCHEMA_VERSION = "t4_integrated_route_fiber_policy_v1"
EXPERTS = ("shallow", "anchored_replacement", "route_complete_region")
PROTONATION_AWARE_EXPERT = "protonation_aware_retained_subgraph"
SUPPORTED_EXPERT_VOCABULARIES = (
    EXPERTS,
    (*EXPERTS, PROTONATION_AWARE_EXPERT),
)


def validate_expert_vocabulary(experts: Iterable[str]) -> tuple[str, ...]:
    """Validate a versioned expert vocabulary without changing legacy defaults."""

    result = tuple(str(value) for value in experts)
    if result not in SUPPORTED_EXPERT_VOCABULARIES:
        raise ValueError(f"unsupported proposal expert vocabulary: {result!r}")
    return result


def _experts(record: dict, *, experts: tuple[str, ...] = EXPERTS) -> tuple[str, ...]:
    experts = validate_expert_vocabulary(experts)
    values = record.get("proposal_experts") or [record.get("proposal_lane")]
    result = tuple(sorted({str(value) for value in values if value}))
    unknown = set(result).difference(experts)
    if unknown:
        raise ValueError(f"unknown proposal experts: {sorted(unknown)}")
    return result


def merge_expert_pools(
    pools: dict[str, Iterable[dict]], *, experts: tuple[str, ...] = EXPERTS
) -> list[dict]:
    """Canonical endpoint union with complete, deterministic origin provenance.

    An endpoint reachable from several parents retains the best-scored parent as its
    learning contrast.  All proposal origins remain attached so a collision does not
    erase evidence that several experts support the same endpoint.
    """

    experts = validate_expert_vocabulary(experts)
    unknown_pools = sorted(set(pools).difference(experts))
    if unknown_pools:
        raise ValueError(f"unknown proposal pools: {unknown_pools}")
    merged: dict[str, dict] = {}
    origins: dict[str, list[dict]] = {}
    expert_sets: dict[str, set[str]] = {}
    family_sets: dict[str, set[str]] = {}
    for expert in experts:
        for position, source in enumerate(pools.get(expert, ())):
            row = dict(source)
            if row.get("proposal_lane") not in (None, expert):
                raise ValueError(
                    f"pool {expert!r} contains lane {row.get('proposal_lane')!r}"
                )
            row["proposal_lane"] = expert
            row["proposal_experts"] = sorted(
                set(_experts(row, experts=experts)) | {expert}
            )
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
            for field in (
                "proposal_program_sha256",
                "program_kind",
                "structural_lane",
            ):
                if row.get(field) is not None:
                    origin[field] = row[field]
            origins.setdefault(smiles, []).append(origin)
            expert_sets.setdefault(smiles, set()).update(row["proposal_experts"])
            family_sets.setdefault(smiles, set()).update(row.get("families") or [])
            incumbent = merged.get(smiles)
            key = (
                float(row.get("parent_score", float("inf"))),
                experts.index(expert),
                position,
            )
            incumbent_key = None
            if incumbent is not None:
                incumbent_key = (
                    float(incumbent.get("parent_score", float("inf"))),
                    experts.index(str(incumbent["proposal_lane"])),
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


def integrated_features(
    record: dict,
    state: SearchState,
    *,
    experts: tuple[str, ...] = EXPERTS,
) -> np.ndarray:
    """Existing whole-program features plus proposal-expert provenance.

    The intercept remains last because :class:`ProgramValue` leaves only its final
    coefficient unpenalized.
    """

    vocabulary = validate_expert_vocabulary(experts)
    base = program_features(record, state)
    record_experts = set(_experts(record, experts=vocabulary))
    return np.concatenate(
        [
            base[:-1],
            np.asarray(
                [float(name in record_experts) for name in vocabulary], dtype=float
            ),
            np.asarray([float(len(record_experts) > 1), 1.0], dtype=float),
        ]
    )


def attach_features(
    records: Iterable[dict],
    state: SearchState,
    fiber,
    *,
    experts: tuple[str, ...] = EXPERTS,
) -> list[dict]:
    """Attach current-state features and endpoint fingerprints."""

    from rdkit import Chem

    prepared = []
    for source in records:
        row = dict(source)
        molecule = Chem.MolFromSmiles(row["smiles"])
        if molecule is None:
            continue
        row["features"] = integrated_features(row, state, experts=experts)
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
    route_scale_floor_rounds: int = 0,
    route_scale_floor_counts: Mapping[str, int] | None = None,
    scale_floor_scope: str = "route_prior",
    experts: tuple[str, ...] = EXPERTS,
) -> list[dict]:
    """Select one batch, seeding available route scales and experts early.

    By default, the route-scale floor preserves the original policy of selecting
    one proposal from each of the small, medium, and large primitive bands.  A
    contract may request more than one proposal per band without changing the
    ordering rule: proposals are selected by route-prior rank, then canonical
    SMILES.
    """

    experts = validate_expert_vocabulary(experts)
    if (
        batch < 1
        or exploration < 0
        or exploration > batch
        or route_scale_floor_rounds < 0
    ):
        raise ValueError("invalid batch or exploration quota")
    if scale_floor_scope not in {"route_prior", "all_generic"}:
        raise ValueError(f"unknown scale floor scope: {scale_floor_scope!r}")
    route_bands = ("small", "medium", "large")
    scale_counts = {band: 1 for band in route_bands}
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
            eligible = [
                row
                for row in available
                if (
                    row.get("proposal_scale_band")
                    if scale_floor_scope == "all_generic"
                    else row.get("realized_primitive_band")
                )
                == band
                and row["smiles"] not in used
                and (
                    scale_floor_scope == "all_generic"
                    or "route_complete_region" in _experts(row, experts=experts)
                )
            ]
            if scale_floor_scope == "route_prior":
                choices = sorted(
                    eligible,
                    key=lambda row: (
                        int(row.get("route_proposal_rank", 1 << 30)),
                        row["smiles"],
                    ),
                )
                selection_kind = "route_scale_floor"
            else:
                choices = [eligible[index] for index in rng.permutation(len(eligible))]
                selection_kind = "generic_scale_floor"
            selected_in_band = 0
            for chosen in choices:
                if len(selected) >= batch:
                    break
                if chosen["smiles"] in used:
                    continue
                selected.append({**chosen, "selection_kind": selection_kind})
                used.add(chosen["smiles"])
                selected_in_band += 1
                if selected_in_band >= scale_counts[band]:
                    break
    if round_index <= expert_floor_rounds:
        for expert in experts:
            if any(expert in _experts(row, experts=experts) for row in selected):
                continue
            choices = [
                row
                for row in available
                if expert in _experts(row, experts=experts)
                and row["smiles"] not in used
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


def expert_census(
    records: Iterable[dict], *, experts: tuple[str, ...] = EXPERTS
) -> dict[str, int]:
    experts = validate_expert_vocabulary(experts)
    counts = {expert: 0 for expert in experts}
    counts["multi_expert"] = 0
    counts["unique_endpoints"] = 0
    for record in records:
        record_experts = _experts(record, experts=experts)
        for expert in record_experts:
            counts[expert] += 1
        counts["multi_expert"] += int(len(record_experts) > 1)
        counts["unique_endpoints"] += 1
    return counts


__all__ = [
    "EXPERTS",
    "PROTONATION_AWARE_EXPERT",
    "SCHEMA_VERSION",
    "SUPPORTED_EXPERT_VOCABULARIES",
    "attach_features",
    "expert_census",
    "integrated_features",
    "merge_expert_pools",
    "select_batch",
    "validate_expert_vocabulary",
]
