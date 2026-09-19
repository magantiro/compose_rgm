"""Empirical corpus-marginal baseline for the tracelet action fiber."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import comb

import numpy as np
import torch

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.experiments.cnof_conditional import PathRecord
from compose_v4.model.rate_model import FiberRatePrediction
from compose_v4.rewrite.fiber import group_transition_indices_by_successor
from compose_v4.rewrite.operators import (
    AtomDelete,
    AtomInsert,
    AtomRestate,
    BondDelete,
    BondInsert,
    BondReorder,
    BondReroute,
)
from compose_v4.rewrite.tracelet_fiber import (
    TRACELET_RULE_FAMILIES,
    TRACELET_TRANSPORT_RULE_FAMILIES,
    TraceletFiber,
    enumerate_tracelet_cnof_fiber,
)
from compose_v4.rewrite.tracelets import (
    CycleAttach,
    CycleInsert,
    RingEarInsert,
    RingSystemDelete,
    RingSystemGrow,
    RingSystemRestate,
)


ActionSignature = tuple[int, ...]


@dataclass(frozen=True)
class TraceletCorpusMarginalTable:
    times: np.ndarray
    mean_path_length: float
    family_probabilities: tuple[dict[str, float], ...]
    signature_probabilities: tuple[dict[str, dict[ActionSignature, float]], ...]
    rule_families: tuple[str, ...] = TRACELET_RULE_FAMILIES


class TraceletCorpusMarginalRateModel:
    """Training-only empirical prior with no graph-state neural dependence.

    This is an explicit baseline, not a supposedly uninformative prior. Global
    family/signature frequencies come only from training traces. Positive
    smoothing preserves every currently enabled operator family.
    """

    def __init__(
        self,
        table: TraceletCorpusMarginalTable,
        *,
        family_smoothing: float = 1e-4,
        action_smoothing: float = 1e-6,
    ) -> None:
        if family_smoothing <= 0.0 or action_smoothing <= 0.0:
            raise ValueError("prior smoothing constants must be positive")
        self.table = table
        self.family_smoothing = float(family_smoothing)
        self.action_smoothing = float(action_smoothing)

    def predict_tracelet_fiber(
        self,
        state: MolecularGraph,
        time: float,
        *,
        fiber: TraceletFiber | None = None,
    ) -> FiberRatePrediction:
        rule_families = self.table.rule_families
        legal_fiber = fiber or enumerate_tracelet_cnof_fiber(
            state,
            allow_bond_reroute="bond_reroute" in rule_families,
        )
        transitions = legal_fiber.transitions
        if not transitions:
            return FiberRatePrediction((), torch.empty(0), {})

        index = int(np.argmin(np.abs(self.table.times - float(time))))
        family_prior = self.table.family_probabilities[index]
        signature_prior = self.table.signature_probabilities[index]
        enabled = [
            family
            for family in rule_families
            if legal_fiber.by_family.get(family)
        ]
        rates = torch.zeros(len(transitions), dtype=torch.float32)
        total_hazard = self.table.mean_path_length * max(1.0 - float(time), 0.0)
        if not enabled or total_hazard <= 0.0:
            return FiberRatePrediction(
                transitions,
                rates,
                group_transition_indices_by_successor(transitions),
            )

        family_weights = {
            family: family_prior.get(family, 0.0) + self.family_smoothing
            for family in enabled
        }
        family_normalizer = sum(family_weights.values())
        offset = 0
        for family in rule_families:
            family_transitions = legal_fiber.by_family.get(family, ())
            count = len(family_transitions)
            if count == 0:
                continue
            weights = np.asarray(
                [
                    signature_prior.get(family, {}).get(
                        _action_signature(family, item.action, state),
                        0.0,
                    )
                    + self.action_smoothing
                    for item in family_transitions
                ],
                dtype=np.float64,
            )
            weights /= weights.sum()
            family_mass = family_weights[family] / family_normalizer
            rates[offset : offset + count] = torch.as_tensor(
                total_hazard * family_mass * weights,
                dtype=rates.dtype,
            )
            offset += count
        if offset != len(transitions):
            raise RuntimeError("tracelet prior transition order mismatch")
        return FiberRatePrediction(
            transitions,
            rates,
            group_transition_indices_by_successor(transitions),
        )


def fit_tracelet_corpus_marginal_rate_model(
    records: tuple[PathRecord, ...],
    *,
    time_bins: int = 64,
    family_smoothing: float = 1e-4,
    action_smoothing: float = 1e-6,
) -> TraceletCorpusMarginalRateModel:
    if not records:
        raise ValueError("records must be non-empty")
    if time_bins <= 1:
        raise ValueError("time_bins must exceed one")

    # The progress probability and teacher rate depend only on scheduler power,
    # path length, and progress. Aggregate chemical signatures once, then apply
    # the 64 time-bin weights to those sufficient statistics. The former loop
    # recomputed every signature for every time bin.
    observed_families = {
        step.rule_name
        for record in records
        for step in record.path.trace.steps
    }
    base_rule_families = (
        TRACELET_TRANSPORT_RULE_FAMILIES
        if "bond_reroute" in observed_families
        else TRACELET_RULE_FAMILIES
    )
    rule_families = (
        *base_rule_families,
        *tuple(sorted(observed_families - set(base_rule_families))),
    )
    family_counts: dict[tuple[float, int], list[Counter[str]]] = {}
    signature_counts: dict[
        tuple[float, int],
        list[dict[str, Counter[ActionSignature]]],
    ] = {}
    for record in records:
        path = record.path
        length = path.path_length
        key = (float(path.scheduler.power), length)
        if key not in family_counts:
            family_counts[key] = [Counter() for _ in range(length)]
            signature_counts[key] = [
                {family: Counter() for family in rule_families}
                for _ in range(length)
            ]
        for progress, (step, state) in enumerate(
            zip(path.trace.steps, path.iter_states())
        ):
            family_counts[key][progress][step.rule_name] += 1
            signature_counts[key][progress][step.rule_name][
                _action_signature(
                    step.rule_name,
                    step.action,
                    state,
                )
            ] += 1

    times = (np.arange(time_bins, dtype=np.float64) + 0.5) / time_bins
    family_tables = []
    signature_tables = []
    for time in times:
        family_weights: Counter[str] = Counter()
        signature_weights: dict[str, Counter[ActionSignature]] = {
            family: Counter() for family in rule_families
        }
        for (scheduler_power, length), progress_families in family_counts.items():
            alpha = 1.0 - (1.0 - float(time)) ** scheduler_power
            for progress, progress_family_counts in enumerate(progress_families):
                probability = (
                    comb(length, progress)
                    * alpha**progress
                    * (1.0 - alpha) ** (length - progress)
                )
                weight = float(probability) * (length - progress)
                if weight == 0.0:
                    continue
                for family, count in progress_family_counts.items():
                    family_weights[family] += weight * count
                    for signature, signature_count in signature_counts[
                        (scheduler_power, length)
                    ][progress][family].items():
                        signature_weights[family][signature] += (
                            weight * signature_count
                        )

        total_family_weight = sum(family_weights.values())
        if total_family_weight <= 0.0:
            family_tables.append({})
        else:
            family_tables.append(
                {
                    family: family_weights[family] / total_family_weight
                    for family in rule_families
                    if family_weights[family] > 0.0
                }
            )
        signature_tables.append(
            {
                family: {
                    signature: weight / sum(signature_weights[family].values())
                    for signature, weight in signature_weights[family].items()
                }
                for family in rule_families
                if signature_weights[family]
            }
        )

    table = TraceletCorpusMarginalTable(
        times=times,
        mean_path_length=float(np.mean([record.path.path_length for record in records])),
        family_probabilities=tuple(family_tables),
        signature_probabilities=tuple(signature_tables),
        rule_families=rule_families,
    )
    return TraceletCorpusMarginalRateModel(
        table,
        family_smoothing=family_smoothing,
        action_smoothing=action_smoothing,
    )


def _action_signature(rule_name: str, action, state: MolecularGraph) -> ActionSignature:
    if isinstance(action, AtomInsert):
        order = int(action.neighbors[0][1]) if action.neighbors else 0
        return int(action.atom_type), order
    if isinstance(action, AtomDelete):
        return (int(state.atom_types[action.v]),)
    if isinstance(action, AtomRestate):
        return int(action.atom_type), int(action.implicit_h_count)
    if isinstance(action, BondInsert):
        return int(action.order), min(_graph_distance(state, action.a, action.b) + 1, 33)
    if isinstance(action, BondDelete):
        return (int(state.bonds[action.a, action.b]),)
    if isinstance(action, BondReorder):
        return int(state.bonds[action.a, action.b]), int(action.new_order)
    if isinstance(action, BondReroute):
        return (
            int(state.bonds[action.a, action.b]),
            int(action.new_order),
            min(_graph_distance(state, action.u, action.v), 33),
            int(state.atom_types[action.u]),
            int(state.atom_types[action.v]),
        )
    if isinstance(action, CycleInsert):
        return (len(action.atoms),)
    if isinstance(action, CycleAttach):
        return (
            len(action.atoms),
            int(action.attachment_order),
            int(state.atom_types[action.anchor]),
        )
    if isinstance(action, RingEarInsert):
        return (
            int(action.a == action.b),
            len(action.atoms),
            min(_graph_distance(state, action.a, action.b), 33),
        )
    if isinstance(action, RingSystemGrow):
        return (
            len(action.system_atoms),
            len(action.interface_atoms),
            len(action.atom_insertions),
            len(action.bond_insertions),
            int(bool(action.aromatic_edges)),
        )
    if isinstance(action, RingSystemDelete):
        return (
            len(action.system_atoms),
            len(action.retained_system_atoms),
            len(action.atom_deletions),
            len(action.bond_deletions),
            int(bool(action.aromatic_edges)),
        )
    if isinstance(action, RingSystemRestate):
        increases = sum(
            int(change.new_order > int(state.bonds[change.a, change.b]))
            for change in action.changes
        )
        decreases = len(action.changes) - increases
        return len(action.changes), increases, decreases
    raise TypeError(f"unsupported tracelet prior action: {type(action).__name__}")


def _graph_distance(state: MolecularGraph, source: int, target: int) -> int:
    if int(source) == int(target):
        return 0
    seen = {int(source)}
    frontier = [int(source)]
    distance = 0
    while frontier:
        distance += 1
        next_frontier = []
        for vertex in frontier:
            for neighbor in np.flatnonzero(state.bonds[vertex] != 0):
                neighbor = int(neighbor)
                if neighbor == int(target):
                    return distance
                if neighbor not in seen:
                    seen.add(neighbor)
                    next_frontier.append(neighbor)
        frontier = next_frontier
    return 33
