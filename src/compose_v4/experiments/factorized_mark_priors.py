"""Corpus base measures for residual factorized mark logits."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from compose_v4.experiments.cnof_conditional import PathRecord
from compose_v4.model.factorized_tracelet_rate_model import (
    FactorizedMarkEmpiricalPriors,
)
from compose_v4.rewrite.factorized_fiber import CNOF_ATOM_TYPES
from compose_v4.rewrite.operators import AtomInsert, AtomRestate, BondReorder
from compose_v4.rewrite.typed_ring_catalog import ring_system_electronic_alias
from compose_v4.rewrite.tracelets import RingSystemGrow


_CNOF_TO_INDEX = {
    int(atom_type): index for index, atom_type in enumerate(CNOF_ATOM_TYPES)
}
_ORDER_TO_INDEX = {1: 0, 2: 1, 3: 2}


def _ring_electronic_categories(action: RingSystemGrow) -> tuple[int, ...]:
    """Read semantic C/N/O/F-by-role labels from a saved ring action.

    Structured ``RingSystemGrow`` actions already persist the scaffold bond
    orders, target reorders, inserted bonds, aromatic edges, and final atom
    states.  The semantic aromatic role is therefore exactly whether the
    saved Kekule lowering assigns an incident double bond to that aromatic
    member.  Reading it directly avoids replaying and sanitizing every state
    in every compiled training path merely to fit fixed corpus counts.
    """

    alias = ring_system_electronic_alias(action)
    pattern = alias.pattern
    aromatic_edges = {
        (min(int(left), int(right)), max(int(left), int(right)))
        for left, right in pattern.target_aromatic_edges
    }
    target_orders = {
        (min(int(left), int(right)), max(int(left), int(right))): int(order)
        for left, right, order in pattern.target_bonds
    }
    categories = []
    for position, (atom_type, charge, _hydrogens) in enumerate(alias.target_atoms):
        if int(charge) != 0:
            raise ValueError("v1 semantic ring prior supports neutral labels")
        try:
            atom_index = _CNOF_TO_INDEX[int(atom_type)]
        except KeyError as error:
            raise ValueError("semantic ring teacher is outside C/N/O/F") from error
        demand = sum(
            int(target_orders[edge]) == 2
            for edge in aromatic_edges
            if int(position) in edge
        )
        if demand not in (0, 1):
            raise ValueError("saved aromatic ring lowering is not a matching")
        categories.append(2 * atom_index + int(demand))
    return tuple(categories)


def _normalized_log_probabilities(
    counts: np.ndarray,
    *,
    smoothing: float,
) -> np.ndarray:
    adjusted = np.asarray(counts, dtype=np.float64) + float(smoothing)
    return np.log(adjusted) - np.log(adjusted.sum())


def fit_factorized_mark_empirical_priors(
    records: Sequence[PathRecord],
    *,
    smoothing: float = 1.0,
) -> FactorizedMarkEmpiricalPriors:
    """Fit fixed global mark-category bases from compiled teacher paths.

    The scan reads existing paths only.  Ring labels use the same semantic
    C/N/O/F-by-electronic-role mapping as production teacher scoring, so the
    prior does not introduce a second aromaticity interpretation.
    """

    if not records:
        raise ValueError("empirical mark priors require non-empty path records")
    if not np.isfinite(smoothing) or smoothing <= 0.0:
        raise ValueError("empirical mark-prior smoothing must be positive")

    root_atom_counts = np.zeros(len(CNOF_ATOM_TYPES), dtype=np.int64)
    connected_atom_order_counts = np.zeros(
        (3, len(CNOF_ATOM_TYPES)),
        dtype=np.int64,
    )
    atom_restate_counts = np.zeros(len(CNOF_ATOM_TYPES), dtype=np.int64)
    bond_reorder_counts = np.zeros(3, dtype=np.int64)
    ring_electronic_counts = np.zeros(2 * len(CNOF_ATOM_TYPES), dtype=np.int64)

    for record in records:
        path = record.path
        for step in path.trace.steps:
            action = step.action
            if isinstance(action, AtomInsert):
                try:
                    atom_index = _CNOF_TO_INDEX[int(action.atom_type)]
                except KeyError as error:
                    raise ValueError("atom-insert teacher is outside C/N/O/F") from error
                if not action.neighbors:
                    root_atom_counts[atom_index] += 1
                else:
                    if len(action.neighbors) != 1:
                        raise ValueError(
                            "factorized empirical prior found a multi-neighbor atom insert"
                        )
                    try:
                        order_index = _ORDER_TO_INDEX[int(action.neighbors[0][1])]
                    except KeyError as error:
                        raise ValueError(
                            "atom-insert teacher has an unsupported bond order"
                        ) from error
                    connected_atom_order_counts[order_index, atom_index] += 1
            elif isinstance(action, AtomRestate):
                try:
                    atom_index = _CNOF_TO_INDEX[int(action.atom_type)]
                except KeyError as error:
                    raise ValueError("atom-restate teacher is outside C/N/O/F") from error
                atom_restate_counts[atom_index] += 1
            elif isinstance(action, BondReorder):
                try:
                    order_index = _ORDER_TO_INDEX[int(action.new_order)]
                except KeyError as error:
                    raise ValueError(
                        "bond-reorder teacher has an unsupported target order"
                    ) from error
                bond_reorder_counts[order_index] += 1
            elif isinstance(action, RingSystemGrow):
                for category in _ring_electronic_categories(action):
                    ring_electronic_counts[int(category)] += 1

    root_log = _normalized_log_probabilities(
        root_atom_counts,
        smoothing=smoothing,
    )
    connected_log = _normalized_log_probabilities(
        connected_atom_order_counts,
        smoothing=smoothing,
    )
    restate_log = _normalized_log_probabilities(
        atom_restate_counts,
        smoothing=smoothing,
    )
    reorder_log = _normalized_log_probabilities(
        bond_reorder_counts,
        smoothing=smoothing,
    )
    ring_log = _normalized_log_probabilities(
        ring_electronic_counts,
        smoothing=smoothing,
    )
    return FactorizedMarkEmpiricalPriors(
        root_atom_log_probabilities=tuple(float(value) for value in root_log),
        connected_atom_order_log_probabilities=tuple(
            tuple(float(value) for value in row) for row in connected_log
        ),
        atom_restate_log_probabilities=tuple(
            float(value) for value in restate_log
        ),
        bond_reorder_log_probabilities=tuple(
            float(value) for value in reorder_log
        ),
        ring_electronic_log_probabilities=tuple(
            float(value) for value in ring_log
        ),
        root_atom_observations=int(root_atom_counts.sum()),
        connected_atom_observations=int(connected_atom_order_counts.sum()),
        atom_restate_observations=int(atom_restate_counts.sum()),
        bond_reorder_observations=int(bond_reorder_counts.sum()),
        ring_electronic_observations=int(ring_electronic_counts.sum()),
    )


__all__ = ["fit_factorized_mark_empirical_priors"]
