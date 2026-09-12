import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_option_continuation import fixture_law

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.improvement_value import (
    MonotoneImprovementModel,
    improvement_parameter_id,
)
from compose_v4.control.molecular_task_search import (
    MolecularHierarchy,
    MolecularSearchState,
)
from compose_v4.control.option_continuation import OptionContinuationKernel
from compose_v4.control.option_controller import OptionBoundaryController
from compose_v4.control.option_controller_runtime import (
    HowControl,
    OptionControllerRuntime,
)
from compose_v4.control.option_features import (
    REGION_CONTEXT_NAMES,
    option_feature_names,
)
from compose_v4.control.option_policy import (
    AdvantageWeightedOptionActor,
    option_actor_parameter_id,
)
from compose_v4.control.persistent_option_smc import PersistentOptionPopulation
from compose_v4.rewrite.kernel import editing_v2_rewrite_system


def runtime():
    kernel = OptionContinuationKernel(
        fixture_law,
        editing_v2_rewrite_system(),
        max_executor_applications=500,
    )
    hierarchy = MolecularHierarchy(
        kernel,
        generic_horizon=1,
        ring_options=(),
        lazy_applicability=True,
    )
    value_dim = 4  # one molecular feature plus utility/incumbent/clock
    actor = AdvantageWeightedOptionActor(
        value_dim + len(REGION_CONTEXT_NAMES), len(option_feature_names()), hidden=4
    )
    value = MonotoneImprovementModel(value_dim, 1, 1, hidden=4)
    for model in (actor, value):
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
    controller = OptionBoundaryController(
        actor,
        value,
        actor_snapshot=option_actor_parameter_id(actor),
        value_snapshot=improvement_parameter_id(value),
        horizons=(1,),
        thresholds=(0.1,),
        beta=2,
    )
    return OptionControllerRuntime(
        hierarchy,
        controller,
        total_options=1,
        reference_snapshot="reference-fixture",
        feature_snapshot="feature-fixture",
        utility_snapshot="utility-fixture",
        how=HowControl("reference", "how-reference-fixture", 20, 20),
        utility=lambda node: node.graph.n_real_atoms / 40,
        how_terminal_weight=lambda _: 1.0,
        how_fallback_weight=lambda _: 1.0,
        molecular_feature_fn=lambda node: [node.graph.n_real_atoms / 40],
        allowed_options=frozenset({"generic"}),
    )


def roots():
    # Eight free persistent slots make the legacy eleven-step program appear in
    # the hierarchy row. The new runtime must still leave it disabled by default.
    graph = pad_molecular_graph(smiles_to_molecular_graph("CCCC"), 12)
    return tuple(
        MolecularSearchState.start(graph, budget=1, root_id=f"root-{index}")
        for index in range(2)
    )


def test_runtime_executes_where_what_how_and_locks_complete_candidates():
    engine = runtime()
    population = engine.start_population(roots(), incumbent=0.1, seed=3)
    following, receipt = engine.advance(population, incumbent=0.1, resample=False)
    assert receipt["boundary"] == 1
    assert receipt["remaining_options"] == 0
    assert receipt["particle_update"]["status"] == "live"
    assert all(row["status"] == "complete" for row in receipt["transitions"])
    assert all(
        "build_ring_system" not in row["applicable_options"]
        for row in receipt["transitions"]
    )
    assert all(
        row["option_reference_probability"] > 0 for row in receipt["transitions"]
    )
    assert all(row["option_proposal_probability"] > 0 for row in receipt["transitions"])
    assert all(particle.history for particle in following.particles if particle.alive)
    candidates = engine.locked_candidates(following)
    assert candidates
    assert len({row["canonical_smiles"] for row in candidates}) == len(candidates)
    assert all(row["locked"] and row["complete"] for row in candidates)
    draws = np.tile(np.arange(len(candidates), dtype=float), (4, 1))
    acquisition = engine.acquire(following, draws, batch_size=1)
    assert len(acquisition["selected_candidate_ids"]) == 1


def test_runtime_resume_reproduces_exact_population():
    first_engine, second_engine = runtime(), runtime()
    initial = first_engine.start_population(roots(), incumbent=0.1, seed=17)
    restored = PersistentOptionPopulation.from_dict(
        json.loads(json.dumps(initial.to_dict()))
    )
    assert restored.control_context == initial.control_context
    first, _ = first_engine.advance(initial, incumbent=0.1, resample=False)
    second, _ = second_engine.advance(restored, incumbent=0.1, resample=False)
    assert first.to_dict() == second.to_dict()


def test_runtime_rejects_wrong_snapshot_and_missing_horizons():
    engine = runtime()
    population = engine.start_population(roots(), incumbent=0.1, seed=1)
    damaged = PersistentOptionPopulation(
        tuple(
            replace(particle, controller_snapshot="another-runtime")
            for particle in population.particles
        ),
        population.option_boundary,
        population.rng_state,
        "another-runtime",
    )
    with pytest.raises(ValueError, match="another runtime"):
        engine.advance(damaged, incumbent=0.1)
    with pytest.raises(ValueError, match="control context"):
        engine.advance(population, incumbent=0.2)


def test_runtime_recovers_pathwise_best_utility_from_persistent_history():
    engine = runtime()
    particle = engine.start_population(roots()[:1], incumbent=0.1, seed=9).particles[0]
    particle = replace(
        particle,
        history=(
            {"after_utility": 0.3},
            {"after_utility": 0.2},
        ),
    )
    assert engine._path_incumbent(particle, 0.1) == pytest.approx(0.3)
