"""Focused support, feedback, isolation and resume tests. No oracle/network."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control import dynamic_program_synthesis_v1 as v1
from compose_v4.control.dynamic_program_synthesis import DynamicProgramOptimizer
from compose_v4.control.objective_program_search import (
    ObjectiveProgramSearch,
    ObjectiveSearchConfig,
    v0_search_config,
)
from compose_v4.control.progressive_bootstrap import ProgressiveBootstrap
from compose_v4.control.progressive_structured_sampler import progressive_module


def graph():
    return pad_molecular_graph(smiles_to_molecular_graph("CCCC"), 48)


def config():
    return replace(
        v0_search_config(seed=31), attempts_per_batch=8, candidates_per_batch=2, wall_seconds=300
    )


def eligible(_):
    return {"oracle_eligible": True}


def boot():
    return ProgressiveBootstrap(
        graph(), config(), source_group="fixture", oracle_protocol="fixture:no-oracle"
    )


def fixture_search(*, rich=0, parent="score_blind"):
    bootstrap = boot()
    candidates = bootstrap.next_batch(eligible)["candidates"]
    assert len(candidates) == 2
    search = ObjectiveProgramSearch(
        config(),
        ObjectiveSearchConfig(
            parent_allocation=parent,
            structured_attempts=rich,
            structured_candidates=2,
            structured_wall_seconds=300,
            query_batch_size=2,
        ),
        source_group="fixture",
        oracle_protocol="fixture:no-oracle",
    )
    for i, candidate in enumerate(candidates):
        search.add_measured_program(candidate, receipt_id=f"synthetic-fixture:{i}", score=-1.0 - i)
    return search


def without_timing(batch):
    return {k: v for k, v in batch.items() if k != "proposal_seconds"}


def test_empty_bootstrap_advances_and_resume_is_exact():
    a = boot()
    first = a.next_batch(lambda _: {"oracle_eligible": False})
    assert not first["candidates"] and a.cursor == 8
    b = ProgressiveBootstrap.restore(deepcopy(a.snapshot()))
    second = a.next_batch(lambda _: {"oracle_eligible": False})
    resumed = b.next_batch(lambda _: {"oracle_eligible": False})
    assert without_timing(second) == without_timing(resumed)
    assert second["attempts"][0]["attempt"] == 8
    assert first["batch_id"] != second["batch_id"]


def test_bootstrap_rejects_bad_snapshot_and_nonboolean_eligibility():
    snapshot = boot().snapshot()
    snapshot["cursor"] += 1
    with pytest.raises(ValueError, match="corrupt"):
        ProgressiveBootstrap.restore(snapshot)
    with pytest.raises(ValueError, match="boolean"):
        boot().next_batch(lambda _: {"oracle_eligible": 1})


def test_progressive_ring_panels_consume_new_rng_draws(monkeypatch):
    draws = []

    def capture(_source, rng, **_kwargs):
        draws.append(int(rng.integers(2**30)))
        return []

    monkeypatch.setattr(v1, "enumerate_substituted_rings", capture)
    rng = np.random.default_rng(7)
    for _ in range(6):
        with pytest.raises(ValueError):
            progressive_module(graph(), rng, "construct_substituted_ring")
    assert len(set(draws)) == 6


def test_v0_proposal_law_is_preserved_before_pool_competition():
    search = fixture_search()
    reference = DynamicProgramOptimizer.restore(search.base.snapshot())
    expected = reference.propose_batch(eligible)
    combined = search.propose_batch(eligible)
    assert combined["proposal_pool"]["candidates"] == expected["candidates"]
    assert search.base.rng.bit_generator.state == reference.rng.bit_generator.state
    assert search.base.config.composition_probability == 0.25
    assert search.base.config.max_composed_programs == 3
    assert search.base.config.current_state_edit_probability == 0


def test_objective_allocation_uses_measured_scores():
    blind = fixture_search()
    ranked = fixture_search(parent="score_rank")
    keys, probabilities = blind.base.selection()
    assert np.allclose(probabilities, [0.5, 0.5])
    keys, probabilities = ranked.base.selection()
    best_key = min(keys, key=lambda k: ranked.base.entries[k]["static_score"])
    assert probabilities[keys.index(best_key)] > 0.5


def test_structured_work_does_not_advance_shallow_rng():
    search = fixture_search(rich=2)
    before = deepcopy(search.base.rng.bit_generator.state)
    search._structured_pool(eligible, {e["endpoint"] for e in search.base.entries.values()})
    assert before == search.base.rng.bit_generator.state


def test_portfolio_resume_and_receipt_boundary():
    search = fixture_search()
    batch = search.propose_batch(eligible)
    restored = ObjectiveProgramSearch.restore(search.snapshot())
    assert restored.snapshot() == search.snapshot()
    outcomes = [
        {
            "candidate_id": c["candidate_id"],
            "receipt_id": f"synthetic:{i}",
            "score": -1.5,
            "failure": None,
            "oracle_protocol": search.base.oracle_protocol,
        }
        for i, c in enumerate(batch["candidates"])
    ]
    bad = deepcopy(outcomes)
    bad[0]["oracle_protocol"] = "wrong"
    stats = deepcopy(search.channel_stats)
    with pytest.raises(ValueError, match="protocol"):
        search.observe_batch(batch["batch_id"], bad)
    assert stats == search.channel_stats
    for engine in (search, restored):
        engine.observe_batch(batch["batch_id"], outcomes)
    assert search.snapshot() == restored.snapshot()
    assert (
        search.channel_stats["shallow"]["reward_sum"] == stats["shallow"]["reward_sum"]
    )  # Better than a weak parent is not incumbent gain.
    assert without_timing(search.propose_batch(eligible)) == without_timing(
        restored.propose_batch(eligible)
    )


def test_corrupt_portfolio_snapshot_is_rejected():
    search = fixture_search()
    snapshot = search.snapshot()
    snapshot["policy"]["parent_allocation"] = "niche_score"
    with pytest.raises(ValueError, match="corrupt"):
        ObjectiveProgramSearch.restore(snapshot)


def test_config_bounds_and_unknown_direction():
    with pytest.raises(ValueError):
        ObjectiveSearchConfig(query_batch_size=0)
    with pytest.raises(ValueError):
        ObjectiveSearchConfig(structured_wall_seconds=float("nan"))
    with pytest.raises(ValueError, match="minimize"):
        ObjectiveProgramSearch(
            replace(config(), score_direction="maximize"),
            ObjectiveSearchConfig(),
            source_group="fixture",
            oracle_protocol="fixture:no-oracle",
        )
