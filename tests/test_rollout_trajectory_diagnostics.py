from __future__ import annotations

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import (
    ELEMENT_TO_IDX,
    SCAR_IDX,
    MolecularGraph,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import (
    empty_molecular_graph,
    is_valid_state,
    pad_molecular_graph,
)
from compose_v4.experiments.cnof_conditional import (
    CompactTrajectoryDiagnostics,
    compact_state_observation,
    corpus_rollout_metrics,
)
from compose_v4.experiments.tracelet_conditional import (
    TraceletRollout,
    sample_tracelet_ancestral,
)
from compose_v4.model.factorized_tracelet_rate_model import SampledRewriteMark
from compose_v4.rewrite.operators import AtomDelete, AtomInsert


class _FixedEthanePrior:
    def sample(self, rng, *, n_slots: int):
        del rng
        return pad_molecular_graph(smiles_to_molecular_graph("CC"), n_slots)


class _InsertThenDeleteDirectModel:
    def __init__(self) -> None:
        self.events = 0

    def sample_rewrite_mark(self, state, time, rng):
        del time, rng
        if self.events == 0:
            self.events += 1
            return SampledRewriteMark(
                total_hazard=1e9,
                rule_name="atom_insert",
                action=AtomInsert(
                    slot=2,
                    atom_type=ELEMENT_TO_IDX["C"],
                    formal_charge=0,
                    implicit_h_count=3,
                    neighbors=((1, 1),),
                ),
            )
        if self.events == 1:
            self.events += 1
            return SampledRewriteMark(
                total_hazard=1e9,
                rule_name="atom_delete",
                action=AtomDelete(2),
            )
        return SampledRewriteMark(0.0, "atom_delete", AtomDelete(0))


def _rollout(
    keys: tuple[str, ...],
    atom_counts: tuple[int, ...],
    rules: tuple[str, ...],
) -> TraceletRollout:
    return TraceletRollout(
        final_state=smiles_to_molecular_graph(keys[-1]),
        event_times=tuple(float(index + 1) for index in range(len(rules))),
        event_rules=rules,
        exhausted_event_budget=False,
        diagnostics=CompactTrajectoryDiagnostics(
            canonical_state_keys=keys,
            atom_counts=atom_counts,
            state_valid=(True,) * len(keys),
            state_connected_or_null=(True,) * len(keys),
        ),
    )


def test_trajectory_metrics_distinguish_self_backtrack_and_delete_regrow() -> None:
    rollouts = (
        _rollout(("CC", "CCC"), (2, 3), ("atom_insert",)),
        _rollout(
            ("CCC", "C", "CC"),
            (3, 1, 2),
            ("atom_delete", "atom_insert"),
        ),
        _rollout(
            ("CC", "CCC", "CC"),
            (2, 3, 2),
            ("bond_reroute", "bond_reroute"),
        ),
        _rollout(("CC", "CC"), (2, 2), ("bond_reroute",)),
    )

    metrics = corpus_rollout_metrics(
        rollouts,
        train_smiles=("CC", "CCC"),
        reference_smiles=("CC", "CCC"),
    )
    trajectory = metrics["trajectory_diagnostics"]

    assert trajectory["available"] is True
    assert trajectory["available_fraction"] == 1.0
    assert trajectory["all_step_valid_trajectory_fraction"] == 1.0
    assert trajectory["all_step_connected_or_null_trajectory_fraction"] == 1.0
    assert trajectory["minimum_atom_count"] == 1
    assert trajectory["mean_trajectory_min_atoms"] == pytest.approx(1.75)
    assert trajectory["hit_one_atom_trajectory_fraction"] == 0.25
    assert trajectory["collapsed_from_larger_source_to_one_atom_fraction"] == 0.25
    assert trajectory["delete_to_one_then_regrow_fraction"] == 0.25
    assert trajectory["canonical_self_event_count"] == 1
    assert trajectory["canonical_self_event_fraction"] == pytest.approx(1 / 6)
    assert trajectory["trajectories_with_canonical_self_event_fraction"] == 0.25
    assert trajectory["immediate_backtrack_count"] == 1
    assert trajectory["immediate_backtrack_per_opportunity_fraction"] == 0.5
    assert trajectory["trajectories_with_immediate_backtrack_fraction"] == 0.25
    assert trajectory["immediate_backtrack_rule_pair_counts"] == {
        "bond_reroute -> bond_reroute": 1
    }
    assert metrics["event_rule_fractions"] == {
        "atom_delete": pytest.approx(1 / 6),
        "atom_insert": pytest.approx(2 / 6),
        "bond_reroute": pytest.approx(3 / 6),
    }


def test_trajectory_metrics_report_validity_and_connectivity_failures() -> None:
    base = smiles_to_molecular_graph("CC")
    invalid_disconnected = MolecularGraph(
        atom_types=base.atom_types.copy(),
        formal_charges=base.formal_charges.copy(),
        implicit_h_counts=np.asarray([5, 4], dtype=np.int32),
        bonds=np.zeros_like(base.bonds),
    )
    invalid_key, _, valid, connected = compact_state_observation(
        invalid_disconnected
    )
    assert valid is False
    assert connected is False
    diagnostics = CompactTrajectoryDiagnostics(
        canonical_state_keys=("CC", invalid_key, "CCC"),
        atom_counts=(2, 2, 3),
        state_valid=(True, valid, True),
        state_connected_or_null=(True, connected, True),
    )
    rollout = TraceletRollout(
        final_state=smiles_to_molecular_graph("CCC"),
        event_times=(1.0, 2.0),
        event_rules=("bond_reroute", "bond_reroute"),
        exhausted_event_budget=False,
        diagnostics=diagnostics,
    )

    trajectory = corpus_rollout_metrics(
        (rollout,),
        train_smiles=("CC",),
        reference_smiles=("CC",),
    )["trajectory_diagnostics"]

    assert trajectory["all_step_valid_trajectory_fraction"] == 0.0
    assert trajectory["all_step_connected_or_null_trajectory_fraction"] == 0.0
    assert trajectory["invalid_state_count"] == 1
    assert trajectory["disconnected_state_count"] == 1
    assert trajectory["invalid_committed_state_count"] == 1
    assert trajectory["disconnected_committed_state_count"] == 1
    assert trajectory["canonical_self_event_count"] == 0
    assert trajectory["immediate_backtrack_count"] == 0


def test_compact_observation_uses_authoritative_validity_predicate() -> None:
    invalid = smiles_to_molecular_graph("CC")
    invalid.implicit_h_counts[0] = 5
    scar = smiles_to_molecular_graph("CCC")
    scar.atom_types[1] = SCAR_IDX
    scar.formal_charges[1] = 0
    scar.implicit_h_counts[1] = 0
    states = (
        empty_molecular_graph(4),
        smiles_to_molecular_graph("c1ccncc1"),
        scar,
        invalid,
    )

    for state in states:
        _, _, observed_valid, _ = compact_state_observation(state)
        assert observed_valid is is_valid_state(state)


def test_direct_mark_sampler_records_source_and_every_committed_state() -> None:
    rollout = sample_tracelet_ancestral(
        _InsertThenDeleteDirectModel(),
        rng=np.random.default_rng(91),
        n_slots=3,
        operational_horizon=0.2,
        time_step=0.2,
        max_events=2,
        source_prior=_FixedEthanePrior(),
    )

    assert rollout.event_rules == ("atom_insert", "atom_delete")
    assert rollout.diagnostics is not None
    assert rollout.diagnostics.canonical_state_keys == ("CC", "CCC", "CC")
    assert rollout.diagnostics.atom_counts == (2, 3, 2)
    assert all(rollout.diagnostics.state_valid)
    assert all(rollout.diagnostics.state_connected_or_null)
    trajectory = corpus_rollout_metrics(
        (rollout,),
        train_smiles=("CC",),
        reference_smiles=("CC",),
    )["trajectory_diagnostics"]
    assert trajectory["immediate_backtrack_count"] == 1
    assert trajectory["canonical_self_event_count"] == 0
