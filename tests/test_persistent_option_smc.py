import math

import numpy as np
import pytest

from compose_v4.control.docking_value import identity
from compose_v4.control.option_particles import advance
from compose_v4.control.persistent_option_smc import (
    OptionTransition,
    PersistentOptionPopulation,
    advance_population,
)


def transitions(population, potentials, *, reference=None, proposal=None, alive=None):
    n = len(population.particles)
    reference = [1] * n if reference is None else reference
    proposal = reference if proposal is None else proposal
    alive = [True] * n if alive is None else alive
    return [
        OptionTransition(
            particle.particle_id,
            {"state": i + 10} if alive[i] else None,
            math.log(reference[i]),
            math.log(proposal[i]),
            potentials[i] if alive[i] else None,
            alive[i],
            {"option": "generic", "slot": i},
        )
        for i, particle in enumerate(population.particles)
    ]


def test_q_equal_reference_matches_existing_option_particle_update():
    population = PersistentOptionPopulation.start(
        [{"state": i} for i in range(3)], seed=0, controller_snapshot="controller-a"
    )
    following, receipt = advance_population(
        population, transitions(population, [0.2, 2.0, 0.7]), resample=False
    )
    legacy = advance(
        [-math.log(3)] * 3,
        [0] * 3,
        [0.2, 2.0, 0.7],
        [True] * 3,
        np.random.default_rng(0),
        resample=False,
    )
    assert receipt["weights"] == pytest.approx(legacy["weights"])
    assert [particle.log_weight for particle in following.particles] == pytest.approx(
        legacy["log_weights"]
    )


def test_proposal_correction_and_exact_option_twist():
    population = PersistentOptionPopulation.start(
        [{"state": 0}, {"state": 0}], seed=4, controller_snapshot="controller-a"
    )
    reference = [0.9, 0.1]
    h_next = [0.2, 0.8]
    h_previous = 0.9 * 0.2 + 0.1 * 0.8
    q = np.asarray(reference) * np.asarray(h_next) / h_previous
    corrected = transitions(
        population,
        [math.log(value) for value in h_next],
        reference=reference,
        proposal=q,
    )
    rooted = PersistentOptionPopulation(
        tuple(
            type(p)(
                p.particle_id,
                p.exact_state,
                p.history,
                p.log_weight,
                math.log(h_previous),
                p.alive,
                p.rng_identity,
                p.controller_snapshot,
            )
            for p in population.particles
        ),
        population.option_boundary,
        population.rng_state,
        population.controller_snapshot,
    )
    _, receipt = advance_population(rooted, corrected, resample=False)
    assert receipt["increments"] == pytest.approx([0, 0])
    assert receipt["weights"] == pytest.approx([0.5, 0.5])


def test_enumerated_two_boundary_smdp_recovers_exact_target_path_law():
    population = PersistentOptionPopulation.start(
        [{"root": 0}] * 4, seed=8, controller_snapshot="controller-a"
    )
    h_a, h_b = 0.68, 0.5
    h_root = 0.6 * h_a + 0.4 * h_b
    rooted = PersistentOptionPopulation(
        tuple(
            type(p)(
                p.particle_id,
                p.exact_state,
                p.history,
                p.log_weight,
                math.log(h_root),
                p.alive,
                p.rng_identity,
                p.controller_snapshot,
            )
            for p in population.particles
        ),
        0,
        population.rng_state,
        population.controller_snapshot,
    )
    first = [
        OptionTransition(
            particle.particle_id,
            {"branch": branch},
            math.log(0.6 if branch == "a" else 0.4),
            math.log(0.5),
            math.log(h_a if branch == "a" else h_b),
            True,
            {"option": branch},
        )
        for particle, branch in zip(rooted.particles, ("a", "a", "b", "b"), strict=True)
    ]
    middle, _ = advance_population(rooted, first, resample=False)
    terminal_values = (0.2, 0.8, 0.1, 0.9)
    second_reference = (0.2, 0.8, 0.5, 0.5)
    second = [
        OptionTransition(
            particle.particle_id,
            {"terminal": index},
            math.log(second_reference[index]),
            math.log(0.5),
            math.log(terminal_values[index]),
            True,
            {"terminal": index},
        )
        for index, particle in enumerate(middle.particles)
    ]
    _, receipt = advance_population(middle, second, resample=False)
    exact = np.asarray((0.024, 0.384, 0.02, 0.18)) / h_root
    assert receipt["weights"] == pytest.approx(exact)


def test_restart_round_trip_reproduces_resampling_and_histories():
    population = PersistentOptionPopulation.start(
        [{"state": i} for i in range(4)], seed=13, controller_snapshot="controller-a"
    )
    rows = transitions(population, [10, 0, 0, 0])
    first, receipt = advance_population(population, rows)
    restored = PersistentOptionPopulation.from_dict(population.to_dict())
    second, second_receipt = advance_population(restored, rows)
    assert receipt == second_receipt
    assert first.to_dict() == second.to_dict()
    assert receipt["resampled"]
    assert all(p.history[-1]["option"] == "generic" for p in first.particles)
    assert all(p.exact_state_id for p in first.particles)


def test_version_one_population_receipt_remains_readable():
    population = PersistentOptionPopulation.start(
        [{"state": 0}], seed=7, controller_snapshot="controller-a"
    )
    current = population.to_dict()
    current.pop("population_id")
    current.pop("control_context")
    current["schema_version"] = "persistent_option_population_v1"
    legacy = {**current, "population_id": identity(current)}
    restored = PersistentOptionPopulation.from_dict(legacy)
    assert restored.control_context is None
    assert restored.particles == population.particles


def test_extinction_and_zero_proposal_fail_explicitly():
    population = PersistentOptionPopulation.start(
        [{"state": i} for i in range(2)], seed=0, controller_snapshot="controller-a"
    )
    following, receipt = advance_population(
        population, transitions(population, [None, None], alive=[False, False])
    )
    assert receipt["status"] == "extinct"
    assert not any(p.alive for p in following.particles)
    partial, partial_receipt = advance_population(
        population, transitions(population, [0, None], alive=[True, False]), resample=False
    )
    assert partial_receipt["weights"] == [1, 0]
    assert [particle.alive for particle in partial.particles] == [True, False]
    with pytest.raises(ValueError, match="log probabilities"):
        OptionTransition("p", {"state": 1}, 0, -math.inf, 0, True, {})
    with pytest.raises(ValueError, match="next state"):
        OptionTransition("p", {"state": 1}, 0, 0, None, False, {})
