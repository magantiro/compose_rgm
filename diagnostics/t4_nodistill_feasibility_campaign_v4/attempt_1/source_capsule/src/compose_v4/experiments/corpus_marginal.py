"""Matched corpus-marginal stochastic rewrite-policy baseline."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.experiments.cnof_conditional import PathRecord
from compose_v4.model.rate_model import FiberRatePrediction, RULE_NAMES
from compose_v4.rewrite.factorized_fiber import (
    FactorizedFiber,
    enumerate_factorized_cnof_fiber,
)
from compose_v4.rewrite.fiber import group_transition_indices_by_successor
from compose_v4.rewrite.operators import (
    AtomDelete,
    AtomInsert,
    AtomRestate,
    BondDelete,
    BondInsert,
    BondReorder,
)


ActionSignature = tuple[int, ...]


@dataclass(frozen=True)
class CorpusMarginalRateTable:
    times: np.ndarray
    mean_path_length: float
    family_probabilities: tuple[dict[str, float], ...]
    signature_probabilities: tuple[dict[str, dict[ActionSignature, float]], ...]


class CorpusMarginalRateModel:
    """Time-dependent rewrite prior with no neural graph-state dependence.

    It uses the same exact action fiber, CTMC clock, compiler traces, and
    ancestral sampler as the learned generator. Corpus statistics determine
    only total hazard, operator-family mass, and action-attribute priors.
    """

    def __init__(
        self,
        table: CorpusMarginalRateTable,
        *,
        smoothing: float = 1e-6,
    ) -> None:
        if smoothing <= 0.0:
            raise ValueError("smoothing must be positive")
        self.table = table
        self.smoothing = float(smoothing)

    def predict_factorized_fiber(
        self,
        state: MolecularGraph,
        time: float,
        *,
        fiber: FactorizedFiber | None = None,
    ) -> FiberRatePrediction:
        legal_fiber = fiber or enumerate_factorized_cnof_fiber(state)
        transitions = legal_fiber.transitions
        if not transitions:
            return FiberRatePrediction((), torch.empty(0), {})

        index = int(np.argmin(np.abs(self.table.times - float(time))))
        family_prior = self.table.family_probabilities[index]
        signature_prior = self.table.signature_probabilities[index]
        enabled = [
            family
            for family in RULE_NAMES
            if legal_fiber.by_family.get(family) and family_prior.get(family, 0.0) > 0.0
        ]
        rates = torch.zeros(len(transitions), dtype=torch.float32)
        total_hazard = self.table.mean_path_length * max(1.0 - float(time), 0.0)
        if not enabled or total_hazard <= 0.0:
            return FiberRatePrediction(
                transitions,
                rates,
                group_transition_indices_by_successor(transitions),
            )

        family_normalizer = sum(family_prior[family] for family in enabled)
        offset = 0
        for family in RULE_NAMES:
            family_transitions = legal_fiber.by_family.get(family, ())
            count = len(family_transitions)
            if count == 0:
                continue
            if family in enabled:
                weights = np.asarray(
                    [
                        signature_prior.get(family, {}).get(
                            _action_signature(family, item.action, state),
                            0.0,
                        )
                        + self.smoothing
                        for item in family_transitions
                    ],
                    dtype=np.float64,
                )
                weights /= weights.sum()
                family_mass = family_prior[family] / family_normalizer
                rates[offset : offset + count] = torch.as_tensor(
                    total_hazard * family_mass * weights,
                    dtype=rates.dtype,
                )
            offset += count
        return FiberRatePrediction(
            transitions,
            rates,
            group_transition_indices_by_successor(transitions),
        )


def fit_corpus_marginal_rate_model(
    records: tuple[PathRecord, ...],
    *,
    time_bins: int = 64,
    smoothing: float = 1e-6,
) -> CorpusMarginalRateModel:
    if not records:
        raise ValueError("records must be non-empty")
    if time_bins <= 1:
        raise ValueError("time_bins must exceed one")

    times = (np.arange(time_bins, dtype=np.float64) + 0.5) / time_bins
    family_tables = []
    signature_tables = []
    for time in times:
        family_weights: Counter[str] = Counter()
        signature_weights: dict[str, Counter[ActionSignature]] = {
            family: Counter() for family in RULE_NAMES
        }
        for record in records:
            path = record.path
            marginal = path.marginal(float(time))
            for progress, probability in enumerate(marginal[:-1]):
                teacher_rate = path.operational_jump_rate(progress)
                weight = float(probability) * teacher_rate
                if weight == 0.0:
                    continue
                step = path.trace.steps[progress]
                family_weights[step.rule_name] += weight
                signature = _action_signature(
                    step.rule_name,
                    step.action,
                    path.states[progress],
                )
                signature_weights[step.rule_name][signature] += weight

        total_family_weight = sum(family_weights.values())
        family_tables.append(
            {
                family: family_weights[family] / total_family_weight
                for family in RULE_NAMES
                if family_weights[family] > 0.0
            }
        )
        signature_tables.append(
            {
                family: {
                    signature: weight / sum(signature_weights[family].values())
                    for signature, weight in signature_weights[family].items()
                }
                for family in RULE_NAMES
                if signature_weights[family]
            }
        )

    table = CorpusMarginalRateTable(
        times=times,
        mean_path_length=float(np.mean([record.path.path_length for record in records])),
        family_probabilities=tuple(family_tables),
        signature_probabilities=tuple(signature_tables),
    )
    return CorpusMarginalRateModel(table, smoothing=smoothing)


def _action_signature(
    rule_name: str,
    action,
    state: MolecularGraph,
) -> ActionSignature:
    if isinstance(action, AtomInsert):
        order = int(action.neighbors[0][1]) if action.neighbors else 0
        return int(action.atom_type), order
    if isinstance(action, AtomDelete):
        return (int(state.atom_types[action.v]),)
    if isinstance(action, AtomRestate):
        return int(action.atom_type), int(action.implicit_h_count)
    if isinstance(action, BondInsert):
        cycle_size = _graph_distance(state, int(action.a), int(action.b)) + 1
        return int(action.order), cycle_size
    if isinstance(action, BondDelete):
        return (int(state.bonds[action.a, action.b]),)
    if isinstance(action, BondReorder):
        return int(state.bonds[action.a, action.b]), int(action.new_order)
    raise TypeError(f"unsupported rewrite action: {type(action).__name__}")


def _graph_distance(state: MolecularGraph, source: int, target: int) -> int:
    if source == target:
        return 0
    seen = {source}
    frontier = [source]
    distance = 0
    while frontier:
        distance += 1
        next_frontier = []
        for vertex in frontier:
            for neighbor in np.flatnonzero(state.bonds[vertex] != 0):
                neighbor = int(neighbor)
                if neighbor == target:
                    return distance
                if neighbor not in seen:
                    seen.add(neighbor)
                    next_frontier.append(neighbor)
        frontier = next_frontier
    return state.n_atoms
