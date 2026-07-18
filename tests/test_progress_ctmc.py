from __future__ import annotations

import numpy as np
import pickle
import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import is_valid_state, pad_molecular_graph
from compose_v4.rewrite.compiler import compile_null_to_target
from compose_v4.rewrite.progress import PowerSurvivalScheduler, TraceProgressCTMC
from compose_v4.rewrite.tracelet_compiler import compile_null_to_target_tracelets


def _path(smiles: str = "c1ccccc1") -> TraceProgressCTMC:
    target = pad_molecular_graph(smiles_to_molecular_graph(smiles), 24)
    assert target is not None
    return TraceProgressCTMC(compile_null_to_target(target))


def test_progress_marginal_has_exact_endpoints_and_valid_states() -> None:
    path = _path()
    start = np.zeros(path.path_length + 1)
    start[0] = 1.0
    end = np.zeros(path.path_length + 1)
    end[-1] = 1.0
    assert np.array_equal(path.marginal(0.0), start)
    assert np.array_equal(path.marginal(1.0), end)
    assert all(is_valid_state(state) for state in path.states)


def test_closed_form_marginal_satisfies_forward_equation() -> None:
    path = _path("C1CC2CCC1C2")
    t = 0.37
    dt = 1e-6
    numerical = (path.marginal(t + dt) - path.marginal(t - dt)) / (2.0 * dt)
    probabilities = path.marginal(t)
    expected = np.zeros_like(probabilities)
    for k in range(path.path_length + 1):
        outgoing = path.jump_rate(k, t)
        expected[k] -= probabilities[k] * outgoing
        if k < path.path_length:
            expected[k + 1] += probabilities[k] * outgoing
    assert np.allclose(numerical, expected, atol=2e-7, rtol=2e-6)


def test_empirical_progress_matches_binomial_marginal() -> None:
    path = _path("CC(=O)NCCO")
    rng = np.random.default_rng(20260714)
    t = 0.43
    draws = np.asarray([path.sample_progress(t, rng) for _ in range(50_000)])
    empirical = np.bincount(draws, minlength=path.path_length + 1) / len(draws)
    assert np.max(np.abs(empirical - path.marginal(t))) < 0.012


def test_sample_exposes_next_successor_but_not_trace_in_state() -> None:
    path = _path("N#CCO")
    rng = np.random.default_rng(7)
    sample = path.sample(0.4, rng)
    assert sample.next_step is not None
    assert sample.next_successor is path.states[sample.progress + 1]
    assert sample.teacher_rate == path.jump_rate(sample.progress, sample.time)
    assert not hasattr(sample.state, "trace")


def test_scheduler_changes_timing_not_endpoint() -> None:
    target = pad_molecular_graph(
        smiles_to_molecular_graph("C1CCC2(CC1)CCCC2"),
        24,
    )
    assert target is not None
    trace = compile_null_to_target(target)
    slow = TraceProgressCTMC(trace, scheduler=PowerSurvivalScheduler(0.5))
    fast = TraceProgressCTMC(trace, scheduler=PowerSurvivalScheduler(2.0))
    assert slow.marginal(0.5)[-1] != fast.marginal(0.5)[-1]
    assert slow.marginal(1.0)[-1] == fast.marginal(1.0)[-1] == 1.0


def test_operational_clock_removes_singular_teacher_rate() -> None:
    path = _path("C1COC1")
    for t in (0.0, 0.2, 0.8, 0.999):
        operational = path.scheduler.operational_time(t)
        assert path.scheduler.time_from_operational(operational) == pytest.approx(t)
    progress = 1
    physical = path.jump_rate(progress, 0.8)
    operational = path.operational_jump_rate(progress)
    assert physical > operational
    assert operational == path.path_length - progress


def test_checkpointed_path_reconstructs_every_state_exactly() -> None:
    target = pad_molecular_graph(
        smiles_to_molecular_graph("CC(=O)NCCc1ccccc1"),
        40,
    )
    trace = compile_null_to_target_tracelets(target, typed_ring_payloads=True)
    eager = TraceProgressCTMC(trace)
    checkpointed = TraceProgressCTMC(trace, checkpoint_interval=3)

    for progress in range(eager.path_length + 1):
        left = eager.state_at(progress)
        right = checkpointed.state_at(progress)
        assert np.array_equal(left.atom_types, right.atom_types)
        assert np.array_equal(left.formal_charges, right.formal_charges)
        assert np.array_equal(left.implicit_h_counts, right.implicit_h_counts)
        assert np.array_equal(left.bonds, right.bonds)
    assert len(tuple(checkpointed.iter_states())) == eager.path_length + 1
    assert checkpointed.retained_checkpoint_count < eager.path_length


def test_checkpointed_path_pickle_is_materially_smaller() -> None:
    target = pad_molecular_graph(
        smiles_to_molecular_graph("CC(=O)NCCc1ccccc1"),
        40,
    )
    trace = compile_null_to_target_tracelets(target, typed_ring_payloads=True)
    eager = TraceProgressCTMC(trace)
    checkpointed = TraceProgressCTMC(trace, checkpoint_interval=4)

    assert len(pickle.dumps(checkpointed)) < 0.4 * len(pickle.dumps(eager))
