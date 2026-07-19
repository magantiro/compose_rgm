from __future__ import annotations

import numpy as np
import torch

from compose_v4.chem.molecular_graph import MolecularGraph, smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.parallel_tracelet_sampling import (
    _trajectory_seeds,
    sample_tracelet_ancestral_many,
)
from compose_v4.experiments.tracelet_conditional import (
    canonical_tracelet_representative,
)
from compose_v4.model.tracelet_rate_model import TraceletRateModel
from compose_v4.rewrite.kernel import canonical_state_key


def _rollout_signature(rollout) -> tuple:
    return (
        canonical_state_key(rollout.final_state),
        rollout.event_times,
        rollout.event_rules,
        rollout.exhausted_event_budget,
        rollout.diagnostics,
    )


def test_trajectory_seeds_are_deterministic_and_distinct() -> None:
    first = _trajectory_seeds(17, 8)
    second = _trajectory_seeds(17, 8)
    assert first == second
    assert len(set(first)) == len(first)


def test_canonical_representative_is_independent_of_slot_permutation() -> None:
    state = pad_molecular_graph(smiles_to_molecular_graph("c1ccncc1"), 9)
    permutation = np.asarray([7, 2, 5, 0, 8, 1, 6, 3, 4])
    permuted = MolecularGraph(
        atom_types=state.atom_types[permutation],
        formal_charges=state.formal_charges[permutation],
        implicit_h_counts=state.implicit_h_counts[permutation],
        bonds=state.bonds[np.ix_(permutation, permutation)],
    )
    first = canonical_tracelet_representative(state)
    second = canonical_tracelet_representative(permuted)

    assert np.array_equal(first.atom_types, second.atom_types)
    assert np.array_equal(first.formal_charges, second.formal_charges)
    assert np.array_equal(first.implicit_h_counts, second.implicit_h_counts)
    assert np.array_equal(first.bonds, second.bonds)


def test_parallel_rollouts_match_serial_rollouts_by_sample_index() -> None:
    torch.manual_seed(18)
    model = TraceletRateModel(hidden_dim=8, message_passing_steps=1).eval()
    arguments = {
        "seed": 19,
        "samples": 3,
        "n_slots": 5,
        "operational_horizon": 0.2,
        "time_step": 0.1,
        "max_events": 2,
    }
    serial = sample_tracelet_ancestral_many(model, workers=1, **arguments)
    parallel = sample_tracelet_ancestral_many(model, workers=2, **arguments)

    assert tuple(map(_rollout_signature, parallel)) == tuple(
        map(_rollout_signature, serial)
    )
